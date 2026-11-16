# SPDX-License-Identifier: MIT
"""Cached suffix geometry from c34a96a/e4cfb91, integrated with direct KV writes."""
import os,json
from pathlib import Path
import torch


def gather_direct_prefix(impl,q,k,v,kc,vc,ks,vs,ctx):
    """Read logical tokens only; never permute/contiguous the entire NHD pool."""
    meta=ctx.attn_metadata
    if impl._cache_format=='SHUFFLE':
        return impl._gather_prefix_and_concat_kv(q,k,v,kc,vc,ks,vs,meta)
    from atom.model_config.mellum_attention import gather_reference_kv
    if os.getenv('MELLUM_PREFILL_METADATA_REUSE','0')=='1':
        from atom.model_config.mellum_prefill_metadata import geometry
        _,lengths=geometry(ctx,len(q),int(meta.total_kv))
    else:lengths=meta.context_lens.tolist()
    pieces=[gather_reference_kv(kc,vc,meta.block_tables[i],0,n,'NHD')
            for i,n in enumerate(lengths)]
    return q,torch.cat([a for a,b in pieces]),torch.cat([b for a,b in pieces]),kc,vc,ks,vs


def native_prefix_geometry(meta,positions,total_q,total_k):
    qb=meta.cu_seqlens_q.tolist();kb=meta.cu_seqlens_k.tolist()
    if len(qb)!=len(kb) or len(qb)<2 or qb[0]!=0 or kb[0]!=0 or qb[-1]!=total_q or kb[-1]!=total_k:
        raise ValueError('Invalid native prefix ragged boundaries')
    pos=positions.flatten().tolist();new=[];keys=[]
    if len(pos)!=total_q:raise ValueError('Native prefix positions differ from query count')
    for a,b,c,d in zip(qb,qb[1:],kb,kb[1:]):
        nq=b-a;nk=d-c
        if nq<=0 or nk<nq or pos[a:b]!=list(range(nk-nq,nk)):
            raise ValueError('Native prefix requires absolute suffix query positions')
        new.append(nq);keys.append(nk)
    if max(new)!=int(meta.max_seqlen_q) or max(keys)!=int(meta.max_seqlen_k):
        raise ValueError('Native prefix maximum lengths disagree with boundaries')
    return new,keys


def audit_native_prefix(impl,new,keys):
    path=os.getenv('MELLUM_PREFIX_AUDIT_PATH')
    if path and getattr(impl,'layer_num',-1)==0:
        with Path(path).open('a') as stream:
            stream.write(json.dumps({'new_tokens':new,'cached_tokens':[k-q for q,k in zip(new,keys)],
                'key_lengths':keys,'backend':'aiter_varlen_cached'})+'\n')
