# SPDX-License-Identifier: MIT
"""Opt-in Mellum cached-prefill compatibility using the engine's KV ownership."""
import copy
import json
import os
import types
from pathlib import Path
import torch


def install_prefix_rope(impl,original,inv,factor):
    from atom.model_config.mellum_attention import reference_position_rope,overwrite_reference_keys
    def corrected(self,q,k,v,qkv,position,ctx):
        if getattr(self,'use_flash_layout',False):raise ValueError('Prefix path requires existing BF16 non-flash cache')
        qr,kr=reference_position_rope(q.clone(),k.clone(),position,inv,factor)
        # Original owns V/cache format and slot writes. Defer its gather until
        # reference K has been written, without mutating shared layer metadata.
        cached=bool(ctx.attn_metadata.has_cached)
        local=ctx
        if cached:
            local=copy.copy(ctx);local.attn_metadata=copy.copy(ctx.attn_metadata)
            local.attn_metadata.has_cached=False
        result=original(q,k,v,qkv,position,local)
        overwrite_reference_keys(result[3],kr,ctx.attn_metadata.slot_mapping)
        if cached:
            if getattr(self,'_cache_format','SHUFFLE')=='NHD':
                # Original NHD gather permutes the entire allocated pool to a
                # contiguous layout. Read only this batch's logical tokens.
                from atom.model_config.mellum_attention import gather_reference_kv
                lengths=ctx.attn_metadata.context_lens.tolist();parts=[]
                for i,n in enumerate(lengths):
                    parts.append(gather_reference_kv(result[3],result[4],ctx.attn_metadata.block_tables[i],0,n,'NHD'))
                return qr,torch.cat([x[0] for x in parts]),torch.cat([x[1] for x in parts]),*result[3:]
            return self._gather_prefix_and_concat_kv(qr,kr,result[2],*result[3:],ctx.attn_metadata)
        return qr,kr,result[2],*result[3:]
    impl.rope_cache=types.MethodType(corrected,impl)


def suffix_mask(q_len,k_len,window,device):
    if not 0<q_len<=k_len:raise ValueError('Cached queries must be a nonempty suffix of keys')
    positions=torch.arange(k_len-q_len,k_len,device=device)[:,None]
    keys=torch.arange(k_len,device=device)[None,:]
    allowed=keys<=positions
    if window>0:allowed=allowed & (keys>positions-window)
    return allowed[None,None]


def cached_prefill(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx):
    meta=ctx.attn_metadata;qb=meta.cu_seqlens_q.tolist();kb=meta.cu_seqlens_k.tolist()
    if len(qb)!=len(kb) or len(qb)<2 or qb[0]!=0 or kb[0]!=0 or qb[-1]!=len(q) or kb[-1]!=len(k) or len(v)!=len(k):
        raise ValueError('Invalid cached ragged boundaries')
    pos=ctx.context.positions.flatten().tolist();outputs=[];lengths=[];cached=[]
    for a,b,c,d in zip(qb,qb[1:],kb,kb[1:]):
        nq=b-a;nk=d-c
        if nq<=0 or nk<nq or pos[a:b]!=list(range(nk-nq,nk)):
            raise ValueError('Cached prefix absolute query positions disagree with full key length')
        query=q[a:b].transpose(0,1)[None];key=k[c:d].transpose(0,1)[None];value=v[c:d].transpose(0,1)[None]
        mask=suffix_mask(nq,nk,self.sliding_window,q.device)
        # Explicit rectangular mask follows HF GQA repeat semantics; optimize
        # this separately after prefix cache behavior has been measured.
        groups=q.shape[1]//k.shape[1]
        key=key.repeat_interleave(groups,dim=1);value=value.repeat_interleave(groups,dim=1)
        out=torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=mask,
            is_causal=False,dropout_p=0.0,scale=self.scale,enable_gqa=False)
        outputs.append(out[0].transpose(0,1).contiguous());lengths.append(nq);cached.append(nk-nq)
    from atom.model_config.mellum_batch import audit
    audit(self,'prefill',lengths)
    path=os.getenv('MELLUM_PREFIX_AUDIT_PATH')
    if path and getattr(self,'layer_num',-1)==0:
        with Path(path).open('a') as stream:
            stream.write(json.dumps({'new_tokens':lengths,'cached_tokens':cached,'key_lengths':[b-a for a,b in zip(kb,kb[1:])]})+'\n')
    return torch.cat(outputs,dim=0)


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
