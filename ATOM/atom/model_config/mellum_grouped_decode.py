# SPDX-License-Identifier: MIT
"""Default-off grouped reference SWA decode; full attention remains requestwise."""
import torch


def host_geometry(ctx):
    cached=getattr(ctx,'_mellum_decode_host',None)
    if cached is None:
        lengths=ctx.attn_metadata.context_lens.tolist()
        positions=ctx.context.positions.flatten().tolist()
        if len(lengths)!=len(positions) or any(n<=0 or p+1!=n for n,p in zip(lengths,positions)):
            raise ValueError('Decode lengths/absolute positions disagree')
        cached=(lengths,positions);ctx._mellum_decode_host=cached
    return cached


def grouped_swa(q,k_cache,v_cache,block_tables,lengths,window,scale):
    if window<=0 or q.ndim!=3 or k_cache.ndim!=5 or v_cache.ndim!=4 or q.dtype!=k_cache.dtype or q.dtype!=v_cache.dtype:
        raise ValueError('Grouped reference SWA requires current BF16 paged NHD cache')
    b,hq,dim=q.shape;blocks,hkv,chunks,block_size,x=k_cache.shape
    if hq%hkv or dim!=chunks*x or v_cache.shape!=(blocks,hkv,dim,block_size) or len(lengths)!=b:
        raise ValueError('Grouped SWA cache/head geometry differs')
    logical=lengths[:,None]-window+torch.arange(window,device=q.device,dtype=torch.long)[None,:]
    physical=block_tables.gather(1,(logical//block_size).long()).long();offset=logical%block_size
    keys=k_cache[physical,:,:,offset,:].reshape(b,window,hkv,dim).transpose(1,2)
    values=v_cache[physical,:,:,offset].transpose(1,2)
    keys=keys.repeat_interleave(hq//hkv,dim=1);values=values.repeat_interleave(hq//hkv,dim=1)
    # Match reference SWA decode: explicit all-true mask, expanded KV heads.
    mask=torch.ones((1,1,1,window),device=q.device,dtype=torch.bool)
    output=torch.nn.functional.scaled_dot_product_attention(q[:,:,None,:],keys,values,
        attn_mask=mask,is_causal=False,dropout_p=0.0,scale=scale,enable_gqa=False)
    return output[:,:,0,:].contiguous()
