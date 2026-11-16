# SPDX-License-Identifier: MIT
"""Reference SDPA for absolute-position prefill suffixes in existing paged KV."""
import torch
from atom.model_config.mellum_attention import gather_reference_kv,sdpa_single_prefill


def suffix_geometry(query_bounds,key_bounds,positions):
    if len(query_bounds)!=len(key_bounds) or len(query_bounds)<2 or query_bounds[0]!=0 or key_bounds[0]!=0:
        raise ValueError('Invalid chunk boundaries')
    if query_bounds[-1]!=len(positions):raise ValueError('Chunk positions differ from queries')
    rows=[]
    for i,(a,b,c,d) in enumerate(zip(query_bounds,query_bounds[1:],key_bounds,key_bounds[1:])):
        n=b-a;end=d-c;cached=end-n
        if n<=0 or cached<0 or positions[a:b]!=list(range(cached,end)):
            raise ValueError('Chunk requires absolute contiguous suffix positions')
        rows.append((i,a,b,cached,end))
    return rows


def suffix_mask(cached,end,window,device):
    start=max(0,cached-window+1) if window>0 else 0
    queries=torch.arange(cached,end,device=device)[:,None]
    keys=torch.arange(start,end,device=device)[None,:]
    mask=keys<=queries
    if window>0:mask=mask & (keys>queries-window)
    return start,mask[None,None]


def cached_prefill(impl,q,k,v,kc,vc,ks,vs,ctx):
    meta=ctx.attn_metadata
    rows=suffix_geometry(meta.cu_seqlens_q.tolist(),meta.cu_seqlens_k.tolist(),ctx.context.positions.flatten().tolist())
    if len(q)!=meta.cu_seqlens_q[-1] or len(k)!=len(q) or len(v)!=len(q):
        raise ValueError('Chunk bridge expects current Q/K/V only')
    outputs=[]
    from atom.model_config.mellum_batch import proxy
    from types import SimpleNamespace
    for i,a,b,cached,end in rows:
        if cached==0:
            local=proxy(impl)
            raw=getattr(impl,'_mellum_raw_qk',None)
            if raw is not None:local._mellum_raw_qk=tuple(x[a:b] for x in raw)
            sub=SimpleNamespace(attn_metadata=SimpleNamespace(cu_seqlens_q=meta.cu_seqlens_q.new_tensor([0,b-a]),cu_seqlens_k=meta.cu_seqlens_k.new_tensor([0,b-a])))
            outputs.append(sdpa_single_prefill(local,q[a:b],k[a:b],v[a:b],kc,vc,ks,vs,sub))
            continue
        start,mask=suffix_mask(cached,end,impl.sliding_window,q.device)
        keys,values=gather_reference_kv(kc,vc,meta.block_tables[i],start,end,impl._cache_format)
        query=q[a:b].transpose(0,1)[None]
        key=keys.transpose(0,1)[None];value=values.transpose(0,1)[None]
        groups=q.shape[1]//keys.shape[1]
        key=key.repeat_interleave(groups,dim=1);value=value.repeat_interleave(groups,dim=1)
        out=torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=mask,
            is_causal=False,dropout_p=0.0,scale=impl.scale,enable_gqa=False)
        outputs.append(out[0].transpose(0,1).contiguous())
    return torch.cat(outputs,dim=0)
