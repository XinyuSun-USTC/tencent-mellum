# SPDX-License-Identifier: MIT
"""Explicit basic fast paths; reference defaults and parameter names are retained."""
import json,os
from pathlib import Path
from collections import Counter


def make_norm(reference,dim,eps=1e-6,fused_allreduce=False):
    backend=os.getenv('MELLUM_NORM_BACKEND','reference')
    if backend=='reference':return reference(dim,eps=eps,fused_allreduce=fused_allreduce)
    if backend!='aiter':raise ValueError('Unknown Mellum norm backend')
    from atom.model_ops.layernorm import RMSNorm
    return RMSNorm(dim,eps=eps,fused_allreduce=fused_allreduce)


def write_audit(model):
    target=os.getenv('MELLUM_FASTPATH_AUDIT_PATH')
    if not target:return
    norms=[m for n,m in model.named_modules() if n=='model.norm' or n.endswith(('.input_layernorm','.post_attention_layernorm','.q_norm','.k_norm'))]
    qkv=[m for n,m in model.named_modules() if n.endswith('.qkv_proj')]
    moe=[m for n,m in model.named_modules() if n.endswith('.mlp')]
    attn=[m for n,m in model.named_modules() if n.endswith('.self_attn')]
    p=Path(target);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'norm_backend':os.getenv('MELLUM_NORM_BACKEND','reference'),
        'moe_classes':dict(Counter(type(m).__name__ for m in moe)),
        'prefill_functions':dict(Counter(m.attn.impl.prefill_attention.__name__ for m in attn)),
        'norm_classes':dict(Counter(type(m).__name__ for m in norms)),'norm_count':len(norms),
        'qkv_backend':os.getenv('MELLUM_QKV_BACKEND','packed'),'qkv_count':len(qkv),
        'separate_qkv_count':sum(hasattr(m,'_mellum_projection_sizes') for m in qkv),
        'resident_inverse_frequency':os.getenv('MELLUM_RESIDENT_ROPE','0')=='1'},indent=2)+'\n')
