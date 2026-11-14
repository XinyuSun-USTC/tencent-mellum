# SPDX-License-Identifier: MIT
"""Opt-in ragged reference attention bridge, using existing paged KV ownership."""
import json
import os
import types
from pathlib import Path
import torch
from atom.model_config.mellum_attention import sdpa_single_prefill,sdpa_cached_decode


def proxy(impl):
    result=types.SimpleNamespace(sliding_window=impl.sliding_window,scale=impl.scale,
        _mellum_stepwise_rope=getattr(impl,'_mellum_stepwise_rope',False),
        _mellum_swa_prefill_opt=getattr(impl,'_mellum_swa_prefill_opt',False),
        _mellum_swa_prefill_impl=getattr(impl,'_mellum_swa_prefill_impl','ck'),
        _cache_format=getattr(impl,'_cache_format',None),layer_num=getattr(impl,'layer_num',-1))
    if hasattr(impl,'_mellum_cpu_rope'):result._mellum_cpu_rope=impl._mellum_cpu_rope
    if hasattr(impl,'rotary_emb'):result.rotary_emb=impl.rotary_emb
    return result


def audit(impl,phase,lengths):
    path=os.getenv('MELLUM_BATCH_AUDIT_PATH')
    if not path or getattr(impl,'layer_num',-1)!=0:return
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('a') as stream:
        stream.write(json.dumps({'phase':phase,'batch_size':len(lengths),'lengths':lengths})+'\n')


def ragged_prefill(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx):
    meta=ctx.attn_metadata
    bounds=meta.cu_seqlens_q.tolist();kbounds=meta.cu_seqlens_k.tolist()
    if len(bounds)<2 or bounds[0]!=0 or bounds[-1]!=len(q) or bounds!=kbounds:
        raise ValueError('Ragged prefill requires full uncached request boundaries')
    if any(b<=a for a,b in zip(bounds,bounds[1:])) or len(k)!=len(q) or len(v)!=len(q):
        raise ValueError('Invalid ragged Q/K/V lengths')
    lengths=[b-a for a,b in zip(bounds,bounds[1:])];audit(self,'prefill',lengths)
    if len(lengths)==1:return sdpa_single_prefill(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx)
    raw=getattr(self,'_mellum_raw_qk',None)
    if getattr(self,'_mellum_stepwise_rope',False) and raw is None:raise ValueError('Missing reference raw QK')
    positions=ctx.context.positions.flatten().tolist()
    if len(positions)!=len(q) or any(positions[a:b]!=list(range(b-a)) for a,b in zip(bounds,bounds[1:])):
        raise ValueError('Cached/chunked prefill is not supported by ragged bridge')
    outputs=[]
    for a,b in zip(bounds,bounds[1:]):
        local=proxy(self)
        if raw is not None:local._mellum_raw_qk=(raw[0][a:b],raw[1][a:b],raw[2].flatten()[a:b])
        sub=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(
            cu_seqlens_q=meta.cu_seqlens_q.new_tensor([0,b-a]),
            cu_seqlens_k=meta.cu_seqlens_k.new_tensor([0,b-a])))
        outputs.append(sdpa_single_prefill(local,q[a:b],k[a:b],v[a:b],k_cache,v_cache,k_scale,v_scale,sub))
    return torch.cat(outputs,dim=0)


def ragged_decode(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx):
    meta=ctx.attn_metadata;count=meta.context_lens.numel()
    if count<1 or len(q)!=count or len(k)!=count or len(v)!=count or meta.block_tables.shape[0]!=count:
        raise ValueError('Ragged decode requires one token per real request, no graph padding/MTP')
    positions=ctx.context.positions.flatten()
    if positions.numel()!=count:raise ValueError('Decode positions differ from request count')
    lengths=meta.context_lens.tolist()
    if any(n<=0 for n in lengths):raise ValueError('Invalid decode context lengths')
    audit(self,'decode',lengths)
    if count==1:return sdpa_cached_decode(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx)
    outputs=[]
    for i in range(count):
        sub=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(context_lens=meta.context_lens[i:i+1],
            block_tables=meta.block_tables[i:i+1]),context=types.SimpleNamespace(positions=positions[i:i+1]))
        outputs.append(sdpa_cached_decode(proxy(self),q[i:i+1],k[i:i+1],v[i:i+1],k_cache,v_cache,k_scale,v_scale,sub))
    return torch.cat(outputs,dim=0)


def install_ragged_attention(impl):
    original=impl.dispatch_backend
    impl.prefill_attention=types.MethodType(ragged_prefill,impl)
    def dispatch(self,ctx):
        if not ctx.context.is_prefill:return types.MethodType(ragged_decode,self)
        return original(ctx)
    impl.dispatch_backend=types.MethodType(dispatch,impl)
