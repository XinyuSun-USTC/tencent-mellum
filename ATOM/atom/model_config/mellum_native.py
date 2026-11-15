# SPDX-License-Identifier: MIT
"""Opt-in native ATOM execution. No fallback to reference math on kernel failure."""
import json
import os
import types
from pathlib import Path
import torch


def hf_route(hidden_states, gating_output, topk, renormalize):
    """Keep the verified softmax/top-k normalization; expert math is native."""
    weights, ids = gating_output.float().softmax(-1).topk(topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    # Frozen AITER sorting requires FP32 storage; preserve BF16-rounded values.
    return weights.to(gating_output.dtype).float(), ids.to(torch.int32)


def record(kind, layer, ctx, rows, backend, **extra):
    path = os.getenv('MELLUM_NATIVE_AUDIT_PATH')
    if not path or layer != 0:
        return
    meta = ctx.attn_metadata
    if meta is None or getattr(ctx.context,'is_dummy_run',False):
        return
    phase = 'prefill' if ctx.context.is_prefill else 'decode'
    requests = meta.cu_seqlens_q.numel()-1 if phase == 'prefill' else meta.context_lens.numel()
    with Path(path).open('a') as stream:
        stream.write(json.dumps(dict(kind=kind,phase=phase,rows=rows,requests=requests,
                                    backend=backend,**extra))+'\n')


class MellumFusedMoeBlock(torch.nn.Module):
    """Frozen FusedMoE on all scheduled tokens, with Mellum's reference gate."""
    def __init__(self, config, quant_config=None, prefix=''):
        super().__init__()
        from atom.model_ops.moe import FusedMoE
        from atom.model_ops.linear import ReplicatedLinear
        from atom.model_config.mellum_projection import install_reference_router
        self.gate = ReplicatedLinear(config.hidden_size,config.num_experts,bias=False,
                                     quant_config=None,prefix=f'{prefix}.gate')
        install_reference_router(self.gate)
        self.experts = FusedMoE(num_experts=config.num_experts,top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,intermediate_size=config.moe_intermediate_size,
            reduce_results=False,renormalize=config.norm_topk_prob,quant_config=quant_config,
            use_grouped_topk=False,has_bias=False,custom_routing_function=hf_route,
            prefix=f'{prefix}.experts',config=config)
        self.layer_num = int(prefix.split('.layers.')[-1].split('.')[0]) if '.layers.' in prefix else -1

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        result = self.experts(hidden_states=hidden_states,router_logits=logits)
        if os.getenv('MELLUM_NATIVE_AUDIT_PATH') and self.layer_num == 0:
            from atom.utils.forward_context import get_forward_context
            record('moe',self.layer_num,get_forward_context(),len(hidden_states),'FusedMoE',
                   quant_method=type(self.experts.quant_method).__name__,routing='torch_softmax_topk',
                   shuffled_w13=bool(getattr(self.experts.w13_weight,'is_shuffled',False)),
                   shuffled_w2=bool(getattr(self.experts.w2_weight,'is_shuffled',False)))
        return result


def native_prefill(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx):
    """Uncached ragged GQA prefill, window counts history excluding current token."""
    import aiter
    from atom.model_config.mellum_batch import audit
    meta=ctx.attn_metadata
    if q.shape[0]!=k.shape[0] or k.shape!=v.shape or meta.cu_seqlens_q.numel()!=meta.cu_seqlens_k.numel():
        raise ValueError('Native Mellum prefill requires full uncached ragged Q/K/V')
    if not torch.equal(meta.cu_seqlens_q,meta.cu_seqlens_k):
        raise ValueError('Native Mellum first version does not enable prefix/chunked prefill')
    if getattr(self,'layer_num',-1)==0:
        bounds=meta.cu_seqlens_q.tolist()
        audit(self,'prefill',[b-a for a,b in zip(bounds,bounds[1:])])
    window=(self.sliding_window-1,0,0) if self.sliding_window>0 else (-1,-1,0)
    output=aiter.flash_attn_varlen_func(q=q.contiguous(),k=k.contiguous(),v=v.contiguous(),
        cu_seqlens_q=meta.cu_seqlens_q,cu_seqlens_k=meta.cu_seqlens_k,
        max_seqlen_q=meta.max_seqlen_q,max_seqlen_k=meta.max_seqlen_k,
        dropout_p=0.0,softmax_scale=self.scale,causal=True,window_size=window,
        return_lse=False,return_attn_probs=False)
    record('attention',getattr(self,'layer_num',-1),ctx,len(q),'aiter_varlen',window=window)
    return output


def install_native_attention(impl):
    """Retain reference RoPE/K write, replace both ragged attention compute paths."""
    original = impl.dispatch_backend
    impl._mellum_stepwise_rope = False  # native kernels consume corrected q/k from rope_cache
    impl.prefill_attention = types.MethodType(native_prefill,impl)
    def dispatch(self,ctx):
        if ctx.context.is_prefill:
            return self.prefill_attention
        backend = original(ctx)
        if 'paged_attention' not in backend.__name__:
            raise ValueError('Native Mellum decode did not select a paged backend')
        def decode(q,k,v,k_cache,v_cache,k_scale,v_scale,fwd_ctx):
            from atom.model_config.mellum_batch import audit
            if getattr(self,'layer_num',-1)==0:
                audit(self,'decode',fwd_ctx.attn_metadata.context_lens.tolist())
            result=backend(q,k,v,k_cache,v_cache,k_scale,v_scale,fwd_ctx)
            record('attention',getattr(self,'layer_num',-1),fwd_ctx,len(q),backend.__name__)
            return result
        return decode
    impl.dispatch_backend = types.MethodType(dispatch,impl)
