# SPDX-License-Identifier: MIT
"""Pure configuration and checkpoint-name helpers for Mellum Step0."""
from copy import deepcopy
import math
import re


def aiter_rope_parameters(parameters):
    """Preserve HF cos/sin amplitude while translating AITER's multiplier."""
    out = deepcopy(parameters)
    if out['rope_type'] == 'yarn':
        factor = float(out['factor'])
        default_scale = 1.0 if factor <= 1 else 1.0 + 0.1 * math.log(factor)
        desired = float(out.pop('attention_factor', default_scale))
        out['attn_factor'] = desired / default_scale
    return out


def mellum_load_target(name):
    """Return (target parameter, shard, expert) for a main BF16 tensor."""
    match = re.fullmatch(r'(model\.layers\.\d+\.mlp\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight', name)
    if match:
        prefix, expert, projection = match.groups()
        shard = {'gate_proj':'w1', 'up_proj':'w3', 'down_proj':'w2'}[projection]
        param = 'w2_weight' if projection == 'down_proj' else 'w13_weight'
        return prefix + 'experts.' + param, shard, int(expert)
    match = re.fullmatch(r'(model\.layers\.\d+\.self_attn\.)(q_proj|k_proj|v_proj)\.weight', name)
    if match:
        prefix, projection = match.groups()
        return prefix + 'qkv_proj.weight', projection[0], None
    return name, None, None
