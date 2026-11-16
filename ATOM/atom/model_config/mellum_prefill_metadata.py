# SPDX-License-Identifier: MIT
"""Default-off validation reuse confined to one ForwardContext; no output caching."""
import os
import torch
import json
from pathlib import Path
_AUDITED=set()

def receipt(ctx,reused):
    path=os.getenv("MELLUM_METADATA_AUDIT_PATH")
    if not path or getattr(ctx.context,"is_dummy_run",False) or (path,reused) in _AUDITED:return
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"reused":reused,"scope":"one_forward_context","cached_suffix":bool(getattr(ctx.attn_metadata,"has_cached",False))})+"\n")
    _AUDITED.add((path,reused))



def geometry(ctx,total_q,total_k):
    meta=ctx.attn_metadata;cached=bool(getattr(meta,'has_cached',False))
    reuse=os.getenv('MELLUM_PREFILL_METADATA_REUSE','0')=='1'
    tensors=[meta.cu_seqlens_q,meta.cu_seqlens_k]
    if cached:tensors.append(ctx.context.positions)
    def version(t):
        try:return t._version
        except RuntimeError:return None  # inference tensors; context is fresh per forward
    key=(cached,total_q,total_k,int(meta.max_seqlen_q),int(meta.max_seqlen_k),
         tuple((id(t),version(t)) for t in tensors))
    saved=getattr(ctx,'_mellum_prefill_geometry',None)
    if reuse and saved is not None and saved[0]==key:
        receipt(ctx,True);return saved[1]
    if cached:
        if os.getenv('MELLUM_PREFIX_COMPAT','0')!='1':raise ValueError('Native prefix/chunked prefill needs explicit compatibility')
        from atom.model_config.mellum_prefix import native_prefix_geometry
        new,keys=native_prefix_geometry(meta,ctx.context.positions,total_q,total_k)
    else:
        if total_q!=total_k or meta.cu_seqlens_q.numel()!=meta.cu_seqlens_k.numel():
            raise ValueError('Native Mellum prefill requires full uncached ragged Q/K/V')
        if not torch.equal(meta.cu_seqlens_q,meta.cu_seqlens_k):
            raise ValueError('Native prefix/chunked prefill needs explicit compatibility')
        bounds=meta.cu_seqlens_q.tolist();new=[b-a for a,b in zip(bounds,bounds[1:])];keys=list(new)
    value=(new,keys)
    if reuse:
        ctx._mellum_prefill_geometry=(key,value);receipt(ctx,False)
    return value
