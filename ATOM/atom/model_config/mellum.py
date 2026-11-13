# SPDX-License-Identifier: MIT
"""Mellum configuration for the frozen Transformers 5.2 runtime.

Field defaults follow the official HF Mellum configuration; this class only
bridges configuration loading, and does not register an executable model.
"""
from copy import deepcopy

from transformers import PretrainedConfig


class MellumConfig(PretrainedConfig):
    model_type = "mellum"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        defaults = dict(
            vocab_size=98304, hidden_size=2304, intermediate_size=7168,
            num_hidden_layers=28, num_attention_heads=32, num_key_value_heads=4,
            head_dim=128, hidden_act="silu", max_position_embeddings=131072,
            initializer_range=0.02, rms_norm_eps=1e-6, use_cache=True,
            tie_word_embeddings=False, attention_bias=False, attention_dropout=0.0,
            sliding_window=1024, moe_intermediate_size=896, num_experts=64,
            num_experts_per_tok=8, norm_topk_prob=True, output_router_logits=False,
            router_aux_loss_coef=0.001,
        )
        for name, default in defaults.items():
            setattr(self, name, kwargs.pop(name, default))
        self.layer_types = deepcopy(kwargs.pop("layer_types", None))
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.mlp_layer_types = deepcopy(kwargs.pop("mlp_layer_types", None))
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers
        self.rope_parameters = deepcopy(kwargs.pop("rope_parameters", None))
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            }
        self._validate_mellum(kwargs)
        # The 5.2 base initializes dtype and common fields. Structure/rope must
        # already exist before its rotary configuration handling runs.
        kwargs.setdefault("architectures", ["MellumForCausalLM"])
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # HF 5.2 mutates the supplied dict to inject runtime-only fields.
        return super().from_dict(deepcopy(config_dict), **kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # Mellum has only the nested format. Do not flatten per-layer values.
        return kwargs

    def _validate_mellum(self, extra):
        for name in ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim",
                     "max_position_embeddings", "num_experts", "num_experts_per_tok",
                     "moe_intermediate_size"]:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok exceeds num_experts")
        if len(self.layer_types) != self.num_hidden_layers or not set(self.layer_types) <= {"full_attention", "sliding_attention"}:
            raise ValueError("layer_types must specify full/sliding_attention for every layer")
        if len(self.mlp_layer_types) != self.num_hidden_layers or set(self.mlp_layer_types) != {"sparse"}:
            raise ValueError("Step0 currently supports only sparse MLP layers")
        if "sliding_attention" in self.layer_types and (type(self.sliding_window) is not int or self.sliding_window <= 0):
            raise ValueError("sliding_window must be positive for sliding layers")
        if extra.get("num_mtp_layers", 0) or extra.get("num_nextn_predict_layers", 0):
            raise ValueError("MTP is not supported by the Step0 adapter")
        for layer_type in set(self.layer_types):
            params = self.rope_parameters.get(layer_type, {})
            if params.get("rope_type") not in {"default", "yarn"} or params.get("rope_theta", 0) <= 0:
                raise ValueError(f"Invalid RoPE parameters for {layer_type}")
            if params["rope_type"] == "yarn" and (params.get("factor", 0) <= 0 or params.get("original_max_position_embeddings", 0) <= 0):
                raise ValueError("YaRN requires positive factor and original_max_position_embeddings")
