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


def audit(impl,phase,lengths,cached=None):
    path=os.getenv('MELLUM_BATCH_AUDIT_PATH')
    if not path or getattr(impl,'layer_num',-1)!=0:return
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('a') as stream:
        stream.write(json.dumps({'phase':phase,'batch_size':len(lengths),'lengths':lengths,'cached_tokens':cached})+'\n')


def ragged_prefill(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx):
    meta=ctx.attn_metadata
    if os.getenv('MELLUM_REFERENCE_CHUNKED','0')=='1' or os.getenv('MELLUM_REFERENCE_PREFIX','0')=='1':
        from atom.model_config.mellum_chunked import cached_prefill
        lengths=meta.cu_seqlens_q.diff().tolist();audit(self,'prefill',lengths,(meta.cu_seqlens_k.diff()-meta.cu_seqlens_q.diff()).tolist())
        return cached_prefill(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx)
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
    once=os.getenv('MELLUM_DECODE_META_ONCE','0')=='1'
    if once:
        from atom.model_config.mellum_grouped_decode import host_geometry
        lengths,host_positions=host_geometry(ctx)
    else:lengths=meta.context_lens.tolist();host_positions=None
    if any(n<=0 for n in lengths):raise ValueError('Invalid decode context lengths')
    audit(self,'decode',lengths)
    if count==1 and not once:return sdpa_cached_decode(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx)
    group=int(os.getenv('MELLUM_SWA_DECODE_GROUP','1'))
    if group not in (1,2,4):raise ValueError('SWA decode groups support1/2/4')
    eligible=(group>1 and self.sliding_window>0 and getattr(self,'_cache_format',None)=='NHD')
    outputs=[];i=0;used=0
    while i<count:
        stop=min(count,i+group)
        if eligible and stop-i>1 and all(n>=self.sliding_window for n in lengths[i:stop]):
            from atom.model_config.mellum_grouped_decode import grouped_swa
            outputs.append(grouped_swa(q[i:stop],k_cache,v_cache,meta.block_tables[i:stop],meta.context_lens[i:stop],self.sliding_window,self.scale))
            used+=1;i=stop;continue
        sub=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(context_lens=meta.context_lens[i:i+1],
            block_tables=meta.block_tables[i:i+1]),context=types.SimpleNamespace(positions=positions[i:i+1]))
        if once:sub._mellum_decode_length=lengths[i];sub._mellum_decode_position=host_positions[i]
        outputs.append(sdpa_cached_decode(proxy(self),q[i:i+1],k[i:i+1],v[i:i+1],k_cache,v_cache,k_scale,v_scale,sub));i+=1
    if used:
        key=(group,count)
        seen=getattr(self,'_mellum_grouped_decode_seen',set())
        if key not in seen:
            from atom.model_config.mellum_native import record
            record('swa_decode_groups',self.layer_num,ctx,count,'torch_sdpa_grouped_swa',group_size=group,group_calls=used,window=self.sliding_window)
            seen.add(key);self._mellum_grouped_decode_seen=seen
    return torch.cat(outputs,dim=0)


def install_ragged_attention(impl):
    original=impl.dispatch_backend
    impl.prefill_attention=types.MethodType(ragged_prefill,impl)
    def dispatch(self,ctx):
        if not ctx.context.is_prefill:return types.MethodType(ragged_decode,self)
        return original(ctx)
    impl.dispatch_backend=types.MethodType(dispatch,impl)
