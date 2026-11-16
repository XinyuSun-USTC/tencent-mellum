# SPDX-License-Identifier: MIT
# Mellum model adapted from this repository's Qwen3-MoE interfaces.
# Official semantics: HF Transformers v5.10.1 Mellum (Apache-2.0).
from typing import Optional, Union, Any
import json
import logging
import os
from pathlib import Path
import torch
from torch import nn
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from atom.config import Config
from atom.model_config.mellum_utils import aiter_rope_parameters, mellum_load_target
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import QKVParallelLinear, RowParallelLinear, ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from atom.models.utils import (IntermediateTensors, PPMissingLayer,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from atom.utils import envs

ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION
ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION = envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION

class MellumSparseMoeBlock(nn.Module):
    """Step0 correctness backend; preserve the gate/expert module interface."""
    def __init__(self, config, quant_config=None, prefix=''):
        super().__init__()
        if quant_config is not None and quant_config.quant_dtype != torch.bfloat16:
            raise ValueError('Mellum torch experts support unquantized Step0 only')
        from atom.model_config.mellum_experts import MellumTorchExperts
        self.experts = MellumTorchExperts(config)
        self.hf_generation_experts = os.getenv('MELLUM_HF_GENERATION_EXPERTS','0') == '1'
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts,
                                     bias=False, quant_config=None, prefix=f'{prefix}.gate')
        router_backend=os.getenv('MELLUM_ROUTER_BACKEND','aiter')
        if router_backend=='torch':
            from atom.model_config.mellum_projection import install_reference_router
            install_reference_router(self.gate)
        elif router_backend!='aiter':raise ValueError(f'Unknown Mellum router backend: {router_backend}')

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        batched = False
        if self.hf_generation_experts:
            from atom.utils.forward_context import get_forward_context
            batched = not get_forward_context().context.is_prefill
        return self.experts(hidden_states, logits, batched=batched)

class MellumRMSNorm(nn.Module):
    """Match official Mellum's FP32 normalization and BF16 rounding order.

    The optional residual API follows ATOM, but rounds the residual addition
    in the input dtype before normalization, as the HF decoder does.
    """
    def __init__(self, dim: int, eps: float = 1e-6, fused_allreduce: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        # Mellum Step0 is TP1: no all-reduce or quantization is needed here.

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        combined = x if residual is None else x + residual
        dtype = combined.dtype
        value = combined.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        output = self.weight * value.to(dtype)
        return output if residual is None else (output, combined)


def make_mellum_norm(dim,eps=1e-6,fused_allreduce=False):
    from atom.model_config.mellum_fastpaths import make_norm
    return make_norm(MellumRMSNorm,dim,eps,fused_allreduce)


class MellumAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        kv_cache_dtype: str = "fp16",
        layer_num: int = 0,
        atom_config: Config = None,
        prefix: str = "",
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        qkv_backend = os.getenv('MELLUM_QKV_BACKEND', 'packed')
        if qkv_backend == 'separate':
            if tp_size != 1:
                raise ValueError('Separate Mellum QKV diagnostic requires TP1')
            from atom.model_config.mellum_projection import install_separate_qkv
            install_separate_qkv(self.qkv_proj, self.q_size, self.kv_size)
        elif qkv_backend != 'packed':
            raise ValueError(f'Unknown Mellum QKV backend: {qkv_backend}')

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
            cos, sin = self.rotary_emb.cos_cache, self.rotary_emb.sin_cache
            joint_cache = torch.cat((cos, sin), dim=-1)
            self.rotary_emb.register_buffer(
                "cos_sin_cache",
                joint_cache.view(joint_cache.size(0), self.head_dim),
                persistent=False,
            )

        self.q_norm = make_mellum_norm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = make_mellum_norm(self.head_dim, eps=rms_norm_eps)
        self.fused_qk_rope = os.getenv('MELLUM_FUSED_QK_ROPE','0')=='1'
        if self.fused_qk_rope and (ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION or os.getenv('MELLUM_DIRECT_KV_WRITE')!='1'):
            raise ValueError('Experimental QK fusion requires direct KV and disables upstream fusion')

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            per_layer_sliding_window=sliding_window,
            rotary_emb=self.rotary_emb,
            prefix=f"{prefix}.attn",
            q_norm=self.q_norm if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION else None,
            k_norm=self.k_norm if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION else None,
        )

        if self.fused_qk_rope:
            self.attn.impl._mellum_fused_norms=(self.q_norm,self.k_norm)

        self.kv_cache_dtype = kv_cache_dtype
        self.layer_num = layer_num
        prefill_backend = os.getenv('MELLUM_PREFILL_BACKEND', 'aiter')
        if prefill_backend in ('sdpa_diagnostic', 'sdpa_rope_diagnostic', 'sdpa_cpu_rope_diagnostic'):
            from atom.model_config.mellum_attention import install_sdpa_prefill
            install_sdpa_prefill(self.attn.impl)
            self.attn.impl._mellum_stepwise_rope = prefill_backend in ('sdpa_rope_diagnostic','sdpa_cpu_rope_diagnostic')
            if prefill_backend == 'sdpa_cpu_rope_diagnostic':
                from atom.model_config.mellum_attention import cpu_inverse_frequency
                self.attn.impl._mellum_cpu_rope = cpu_inverse_frequency(self.rotary_emb)
        elif prefill_backend != 'aiter':
            raise ValueError(f'Unknown Mellum prefill backend: {prefill_backend}')
        swa_impl = os.getenv('MELLUM_SWA_PREFILL_IMPL', 'ck')
        if swa_impl not in ('ck', 'triton'):
            raise ValueError('MELLUM_SWA_PREFILL_IMPL must be ck or triton')
        swa_opt = os.getenv('MELLUM_SWA_PREFILL_OPT', '0')
        if swa_opt not in ('0', '1'):
            raise ValueError('MELLUM_SWA_PREFILL_OPT must be 0 or 1')
        if swa_opt == '1':
            if prefill_backend != 'sdpa_cpu_rope_diagnostic':
                raise ValueError('SWA candidate requires reference CPU RoPE prefill')
            self.attn.impl._mellum_swa_prefill_opt = True
            self.attn.impl._mellum_swa_prefill_impl = swa_impl
        if os.getenv('MELLUM_REFERENCE_KV_ROPE','0')=='1':
            if prefill_backend!='sdpa_cpu_rope_diagnostic':
                raise ValueError('Reference KV rotation requires the CPU RoPE diagnostic backend')
            from atom.model_config.mellum_attention import install_reference_kv_rope
            install_reference_kv_rope(self.attn.impl)
        if os.getenv('MELLUM_SDPA_DECODE','0')=='1':
            if os.getenv('MELLUM_REFERENCE_KV_ROPE','0')!='1':
                raise ValueError('Diagnostic SDPA decode requires reference KV rotation')
            from atom.model_config.mellum_attention import install_sdpa_decode
            install_sdpa_decode(self.attn.impl)
        execution=os.getenv('MELLUM_ATTENTION_EXECUTION','reference')
        if execution not in ('reference','native','native_prefill','native_decode'):
            raise ValueError('Unknown Mellum attention execution mode')
        if execution!='reference':
            if os.getenv('MELLUM_REFERENCE_KV_ROPE','0')!='1' or os.getenv('MELLUM_SDPA_DECODE','0')!='0':
                raise ValueError('Native attention requires reference KV RoPE and disables SDPA decode')
            if os.getenv('MELLUM_REQUEST_BATCHING','0')!='0':
                raise ValueError('Native attention cannot install the reference ragged bridge')
            from atom.model_config.mellum_native import install_native_attention
            install_native_attention(self.attn.impl,mode={'native':'both','native_prefill':'prefill_only','native_decode':'decode_only'}[execution])
        batching=os.getenv('MELLUM_REQUEST_BATCHING','0')
        if batching not in ('0','1'):
            raise ValueError('MELLUM_REQUEST_BATCHING must be 0 or 1')
        if batching=='1':
            if prefill_backend!='sdpa_cpu_rope_diagnostic' or os.getenv('MELLUM_SDPA_DECODE','0')!='1':
                raise ValueError('Ragged bridge requires reference CPU RoPE and SDPA decode')
            if any(os.getenv(name) for name in ('MELLUM_FORCED_DECODE_PLAN','MELLUM_FORCED_DECODE_DIR','MELLUM_SWA_QKV_CAPTURE_DIR','MELLUM_MOE_REPLAY_INPUT_DIR','MELLUM_PREFILL_LOGITS_DIR','MELLUM_FREE_LOGITS_DIR')):
                raise ValueError('Ragged bridge cannot be combined with forced/capture diagnostics')
            from atom.model_config.mellum_batch import install_ragged_attention
            install_ragged_attention(self.attn.impl)
        capture_root = os.getenv('MELLUM_SWA_QKV_CAPTURE_DIR')
        if capture_root:
            if prefill_backend != 'sdpa_cpu_rope_diagnostic' or swa_opt != '0':
                raise ValueError('QKV capture requires the Step0 reference prefill')
            from atom.model_config.mellum_swa_capture import install_swa_capture
            install_swa_capture(self.attn.impl, capture_root)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            attn_output = self.attn(
                query=q, key=k, value=v, positions=positions, q_scale=None, qkv=qkv
            )
        else:
            # Add qk-norm (per-head)
            if not self.fused_qk_rope:
                q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                    -1, self.num_heads * self.head_dim
                )
                k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
                    -1, self.num_kv_heads * self.head_dim
                )

            if getattr(self.attn.impl, '_mellum_stepwise_rope', False):
                self.attn.impl._mellum_raw_qk = (
                    q.view(-1,self.num_heads,self.head_dim).clone(),
                    k.view(-1,self.num_kv_heads,self.head_dim).clone(), positions)

            attn_output = self.attn(
                query=q, key=k, value=v, positions=positions, **model_kwargs
            )
        output = self.o_proj(attn_output)
        return output


class MellumDecoderLayer(nn.Module):
    def __init__(self, atom_config=None, layer_num: int = 0, prefix: str = "") -> None:
        super().__init__()

        self.atom_config = atom_config
        config = self.atom_config.hf_config
        self.hidden_size = config.hidden_size
        layer_type = config.layer_types[layer_num]
        rope_params = aiter_rope_parameters(config.rope_parameters[layer_type])
        rope_theta = rope_params["rope_theta"]
        rope_scaling = rope_params
        kv_cache_dtype = atom_config.kv_cache_dtype
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        self.layer_idx = layer_num

        self.self_attn = MellumAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            sliding_window=config.sliding_window if layer_type == "sliding_attention" else None,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            atom_config=atom_config,
            prefix=f"{prefix}.self_attn",
        )

        backend = os.getenv('MELLUM_MOE_BACKEND', 'torch')
        if backend not in ('torch', 'aiter', 'fused_torch_route', 'fused_isolated'):
            raise ValueError(f'Unknown Mellum MoE backend: {backend}')
        if backend in ('fused_torch_route','fused_isolated'):
            from atom.model_config.mellum_native import MellumFusedMoeBlock,MellumIsolatedMoeBlock
            moe_class = MellumIsolatedMoeBlock if backend=='fused_isolated' else MellumFusedMoeBlock
        else:
            moe_class = MellumSparseMoeBlock if backend == 'torch' else Qwen3MoeSparseMoeBlock
        self.mlp = moe_class(
            config, quant_config=self.atom_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = make_mellum_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION and self.layer_idx > 0,
        )
        self.post_attention_layernorm = make_mellum_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **model_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **model_kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MellumModel(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()

        self.config = atom_config.hf_config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix, layer_num=None: MellumDecoderLayer(
                atom_config=atom_config,
                layer_num=layer_num,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        if get_pp_group().is_last_rank:
            self.norm = make_mellum_norm(
                self.config.hidden_size,
                eps=self.config.rms_norm_eps,
                fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )
        if os.getenv("MELLUM_STAGE_DUMP_DIR"):
            from atom.model_config.mellum_diagnostics import install_stage_hooks
            def capture_path():
                from atom.utils.forward_context import get_forward_context
                context = get_forward_context().context
                if not context.is_prefill or context.is_dummy_run:
                    return None
                index = getattr(self, "_layer_dump_index", 0)
                return Path(os.environ["MELLUM_STAGE_DUMP_DIR"]) / f"sample_{index:06d}"
            stage_layer = int(os.getenv("MELLUM_STAGE_LAYER", "0"))
            if not 0 <= stage_layer < len(self.layers):
                raise ValueError("Invalid Mellum stage observation layer")
            self._stage_hooks = install_stage_hooks(self.layers[stage_layer], capture_path, "atom")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        layer_dump_dir = os.getenv("MELLUM_LAYER_DUMP_DIR")
        diagnostic = False
        if layer_dump_dir:
            from atom.utils.forward_context import get_forward_context
            context = get_forward_context().context
            diagnostic = context.is_prefill and not context.is_dummy_run
            if diagnostic:
                if context.batch_size != 1:
                    raise RuntimeError("Mellum layer diagnostic requires batch_size=1")
                index = getattr(self, "_layer_dump_index", 0)
                dump_path = Path(layer_dump_dir) / f"sample_{index:06d}"
                dump_path.mkdir(parents=True, exist_ok=True)
                torch.save(hidden_states.detach().cpu(), dump_path / "embedding.pt")

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                positions, hidden_states, residual, **model_kwargs
            )
            if diagnostic:
                materialized = hidden_states if residual is None else hidden_states + residual
                torch.save(materialized.detach().cpu(), dump_path / f"layer_{layer.layer_idx:02d}.pt")

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if diagnostic:
            torch.save(hidden_states.detach().cpu(), dump_path / "final_norm.pt")
            self._layer_dump_index = index + 1

        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )


class MellumForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        self.atom_config = atom_config
        self.config = self.atom_config.hf_config

        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Mellum Step0 supports TP=1 only")
        self.model = MellumModel(
            atom_config=self.atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size,
                bias=False,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self._install_load_audit()
        from atom.model_config.mellum_fastpaths import write_audit
        write_audit(self)
        if os.getenv("MELLUM_BATCH_CAMPAIGN_CONFIG"):
            from atom.model_config.mellum_campaign import install
            install(self)

    def _install_load_audit(self):
        from atom.model_loader.loader import default_weight_loader
        self._loaded_mellum_parts = set()
        for name, param in self.named_parameters():
            original = getattr(param, "weight_loader", default_weight_loader)
            def audited(parameter, weight, *args, _name=name, _original=original, **kwargs):
                result = _original(parameter, weight, *args, **kwargs)
                if ".mlp.experts." in _name:
                    shard = kwargs.get("shard_id", args[1] if len(args) > 1 else None)
                    expert = kwargs.get("expert_id", args[2] if len(args) > 2 else None)
                else:
                    shard = kwargs.get("shard_id", args[0] if args else None)
                    expert = None
                self._loaded_mellum_parts.add((_name, shard, expert))
                return result
            param.weight_loader = audited

    def validate_loaded_weights(self):
        index_path = Path(self.atom_config.model) / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        expected = {mellum_load_target(name) for name in index["weight_map"]}
        if len(expected) != len(index["weight_map"]):
            raise RuntimeError("Mellum source-to-load mapping is not one-to-one")
        missing = expected - self._loaded_mellum_parts
        unexpected = self._loaded_mellum_parts - expected
        report = {"source_tensors": len(index["weight_map"]),
                  "successful_parts": len(self._loaded_mellum_parts),
                  "missing": sorted(map(repr, missing)), "unexpected": sorted(map(repr, unexpected))}
        path = os.getenv("MELLUM_LOAD_AUDIT_PATH")
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + "\n")
        if missing or unexpected:
            raise RuntimeError(f"Mellum loading audit failed: {len(missing)} missing, {len(unexpected)} unexpected")
        logging.getLogger("atom").info("Mellum loading audit passed: %s source tensors", len(expected))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any] | None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        plan=os.getenv('MELLUM_FORCED_DECODE_PLAN')
        if plan:
            from atom.model_config.mellum_decode_probe import ForcedDecodeProbe
            from atom.utils.forward_context import get_forward_context
            if not hasattr(self,'_decode_probe'):
                self._decode_probe=ForcedDecodeProbe(plan,os.environ['MELLUM_FORCED_DECODE_DIR'])
                if os.getenv('MELLUM_DECODE_LAYER_CAPTURE','0')=='1':
                    self._decode_probe_handles=self._decode_probe.install_layer_hooks(self.model)
            self._decode_probe.observe(input_ids,positions,get_forward_context().context)
        elif os.getenv('MELLUM_FREE_LOGITS_DIR'):
            from atom.model_config.mellum_decode_probe import FreeGenerationProbe
            from atom.utils.forward_context import get_forward_context
            if not hasattr(self,'_free_probe'):
                self._free_probe=FreeGenerationProbe(os.environ['MELLUM_FREE_LOGITS_DIR'])
            self._free_probe.observe(input_ids,positions,get_forward_context().context)
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        dump_dir = os.getenv("MELLUM_PREFILL_LOGITS_DIR")
        if dump_dir:
            from atom.utils.forward_context import get_forward_context
            context = get_forward_context().context
            if context.is_prefill and not context.is_dummy_run:
                if context.batch_size != 1:
                    raise RuntimeError("Mellum prefill diagnostic requires batch_size=1")
                index = getattr(self, "_prefill_dump_index", 0)
                target = Path(dump_dir)
                target.mkdir(parents=True, exist_ok=True)
                torch.save(logits[-1].detach().cpu(), target / f"prefill_{index:06d}.pt")
                self._prefill_dump_index = index + 1
        if hasattr(self,'_decode_probe'):
            from atom.utils.forward_context import get_forward_context
            return self._decode_probe.force(logits,get_forward_context().context)
        if hasattr(self,'_free_probe'):
            from atom.utils.forward_context import get_forward_context
            self._free_probe.save(logits,get_forward_context().context)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
