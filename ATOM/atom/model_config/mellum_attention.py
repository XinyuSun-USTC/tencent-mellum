# SPDX-License-Identifier: MIT
"""Restricted same-layout SDPA prefill intervention for Mellum diagnosis."""
import types
import os
from pathlib import Path
import torch


def stepwise_rope(q, k, positions, rotary):
    cos = rotary.cos_cache.index_select(0, positions.flatten()).reshape(q.shape[0], 1, -1)
    sin = rotary.sin_cache.index_select(0, positions.flatten()).reshape(q.shape[0], 1, -1)
    cos = torch.cat((cos, cos), dim=-1).to(q.dtype)
    sin = torch.cat((sin, sin), dim=-1).to(q.dtype)
    def apply(x):
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), dim=-1) * sin
    return apply(q), apply(k)


def cpu_inverse_frequency(rotary):
    # Existing AITER default/YaRN formulas were CPU-checked against HF.
    # Explicit CPU overrides the engine's default GPU device.
    with torch.device('cpu'):
        argument = getattr(rotary, 'scaling_factor', rotary.base)
        inv = rotary._compute_inv_freq(argument).float()
    return inv, float(getattr(rotary, 'mscale', 1.0))


def reference_position_rope(q, k, positions, inverse_frequency, attention_factor):
    inv = inverse_frequency.to(device=q.device, dtype=torch.float32)[None,:,None]
    pos = positions.flatten()[None,None,:].float()
    with torch.autocast(device_type=q.device.type, enabled=False):
        freqs = (inv.float() @ pos.float()).transpose(1,2)
        angles = torch.cat((freqs,freqs),dim=-1)
        cos = (angles.cos()*attention_factor).to(q.dtype).reshape(q.shape[0],1,q.shape[-1])
        sin = (angles.sin()*attention_factor).to(q.dtype).reshape(q.shape[0],1,q.shape[-1])
    def apply(x):
        left,right=x.chunk(2,dim=-1)
        return x*cos+torch.cat((-right,left),dim=-1)*sin
    return apply(q),apply(k)


def sdpa_single_prefill(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx):
    meta = fwd_ctx.attn_metadata
    # Do not silently use a cache-free prefill formula for cached/chunked batches.
    if meta.cu_seqlens_q.numel() != 2 or meta.cu_seqlens_k.numel() != 2:
        raise ValueError('Diagnostic SDPA requires one request')
    n = q.shape[0]
    if k.shape[0] != n or v.shape[0] != n:
        raise ValueError('Diagnostic SDPA requires full uncached prefill')
    if meta.cu_seqlens_q.tolist() != [0,n] or meta.cu_seqlens_k.tolist() != [0,n]:
        raise ValueError('Diagnostic SDPA does not support cached/chunked prefill')
    if getattr(self, '_mellum_stepwise_rope', False):
        raw_q, raw_k, positions = self._mellum_raw_qk
        if hasattr(self, '_mellum_cpu_rope'):
            inv,factor=self._mellum_cpu_rope
            q,k=reference_position_rope(raw_q,raw_k,positions,inv,factor)
        else:
            q, k = stepwise_rope(raw_q, raw_k, positions, self.rotary_emb)
    # Opt-in Step1 candidate: retain reference RoPE and all cache/decode work.
    # AITER's local bounds are inclusive: left=W-1 gives exactly W keys.
    if (getattr(self, '_mellum_swa_prefill_opt', False)
            and self.sliding_window > 0 and n >= self.sliding_window):
        if getattr(self, '_mellum_swa_prefill_impl', 'ck') == 'triton':
            from atom.model_config.mellum_swa_triton import triton_swa_prefill
            return triton_swa_prefill(q,k,v,meta.cu_seqlens_q,self.sliding_window,self.scale)
        return aiter_swa_prefill(q, k, v, meta.cu_seqlens_q,
                                 self.sliding_window, self.scale)
    query=q.transpose(0,1)[None];key=k.transpose(0,1)[None];value=v.transpose(0,1)[None]
    mask=None
    if self.sliding_window > 0 and n >= self.sliding_window:
        # HF emits an explicit mask at the window threshold. Its SDPA wrapper
        # then expands KV heads rather than enabling GQA with a mask.
        mask=torch.ones((n,n),device=q.device,dtype=torch.bool).tril().triu(1-self.sliding_window)[None,None]
        groups=q.shape[1]//k.shape[1]
        key=key.repeat_interleave(groups,dim=1)
        value=value.repeat_interleave(groups,dim=1)
    result = torch.nn.functional.scaled_dot_product_attention(
        query,key,value,attn_mask=mask,
        dropout_p=0.0,is_causal=mask is None,scale=self.scale,enable_gqa=mask is None)
    return result[0].transpose(0,1).contiguous()


def aiter_swa_prefill(q, k, v, cu_seqlens, window, scale):
    """Full uncached local prefill, native GQA, no dense mask/KV expansion.

    Caller validates single-request metadata. Failures propagate; no silent
    fallback that could make a benchmark measure the reference path instead.
    """
    if window <= 0 or q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError('Invalid local prefill window or Q/K/V layout')
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError('Local prefill requires equal lengths and valid GQA heads')
    import aiter
    return aiter.flash_attn_varlen_func(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=q.shape[0], max_seqlen_k=k.shape[0],
        dropout_p=0.0, softmax_scale=scale, causal=True,
        window_size=(window-1, 0, 0), return_lse=False,
        return_attn_probs=False)


def install_sdpa_prefill(impl):
    # Bind only this Mellum instance. Existing rope/cache/decode remain in use.
    impl.prefill_attention = types.MethodType(sdpa_single_prefill, impl)


def overwrite_reference_keys(cache, keys, slot_mapping):
    """Write BF16 keys to the existing TP1 NHD/ASM five-dimensional K layout."""
    if cache.ndim!=5 or cache.dtype!=torch.bfloat16 or keys.dtype!=cache.dtype:
        raise ValueError('Reference KV diagnostic requires BF16 five-dimensional K cache')
    blocks,heads,dim_chunks,block_size,x=cache.shape
    slots=slot_mapping.flatten().to(torch.long)
    if keys.shape!=(slots.numel(),heads,dim_chunks*x):
        raise ValueError('Reference keys/slots/cache shape mismatch')
    valid=slots>=0;slots=slots[valid]
    cache[slots//block_size,:,:,slots%block_size,:]=keys[valid].reshape(-1,heads,dim_chunks,x)


def install_reference_kv_rope(impl):
    """One-variable diagnostic: correct Q/K rotation and K slots; retain V/decode kernels."""
    original=impl.rope_cache
    inv,factor=cpu_inverse_frequency(impl.rotary_emb)
    if os.getenv('MELLUM_DIRECT_KV_WRITE','0')=='1':
        from atom.model_config.mellum_direct_kv import install_direct_kv_rope
        install_direct_kv_rope(impl,inv,factor)
        return
    def corrected(self,q,k,v,qkv,position,fwd_ctx):
        if getattr(self,'use_flash_layout',False):
            raise ValueError('Reference KV diagnostic does not support flash-layout cache')
        qr,kr=reference_position_rope(q.clone(),k.clone(),position,inv,factor)
        result=original(q,k,v,qkv,position,fwd_ctx)
        overwrite_reference_keys(result[3],kr,fwd_ctx.attn_metadata.slot_mapping)
        return qr,kr,result[2],result[3],result[4],result[5],result[6]
    impl.rope_cache=types.MethodType(corrected,impl)


def gather_reference_kv(k_cache,v_cache,block_table,start,end,cache_format):
    """Read one request's existing paged BF16 cache in logical token order."""
    if k_cache.ndim!=5 or k_cache.dtype!=torch.bfloat16 or v_cache.dtype!=k_cache.dtype:
        raise ValueError('SDPA decode requires BF16 five-dimensional K cache')
    blocks,heads,chunks,block_size,x=k_cache.shape
    logical=torch.arange(start,end,device=k_cache.device,dtype=torch.long)
    physical=block_table[logical//block_size].long();offset=logical%block_size
    keys=k_cache[physical,:,:,offset,:].reshape(-1,heads,chunks*x)
    if cache_format=='SHUFFLE':
        shuffled=v_cache.view(blocks,heads,block_size//x,chunks*x,x)
        values=shuffled[physical,:,offset//x,:,offset%x]
    elif cache_format=='NHD' and v_cache.ndim==4:
        values=v_cache[physical,:,:,offset]
    else:raise ValueError('Unsupported V cache layout for diagnostic decode')
    return keys,values


def sdpa_cached_decode(self,q,k,v,k_cache,v_cache,k_scale,v_scale,fwd_ctx):
    meta=fwd_ctx.attn_metadata
    if q.shape[0]!=1 or meta.context_lens.numel()!=1:
        raise ValueError('Diagnostic cached SDPA requires single-request single-token decode')
    n=getattr(fwd_ctx,'_mellum_decode_length',None)
    if n is None:n=int(meta.context_lens[0].item())
    position=getattr(fwd_ctx,'_mellum_decode_position',None)
    if position is None:position=int(fwd_ctx.context.positions.flatten()[-1].item())
    if n!=position+1:
        raise ValueError('Decode context length and absolute position disagree')
    start=max(0,n-self.sliding_window) if self.sliding_window>0 else 0
    keys,values=gather_reference_kv(k_cache,v_cache,meta.block_tables[0],start,n,self._cache_format)
    query=q.transpose(0,1)[None];key=keys.transpose(0,1)[None];value=values.transpose(0,1)[None]
    mask=None
    if self.sliding_window>0 and n>=self.sliding_window:
        mask=torch.ones((1,1,1,n-start),device=q.device,dtype=torch.bool)
        groups=q.shape[1]//keys.shape[1]
        key=key.repeat_interleave(groups,dim=1);value=value.repeat_interleave(groups,dim=1)
    output=torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=mask,
        is_causal=False,dropout_p=0.0,scale=self.scale,enable_gqa=mask is None)
    output=output[0].transpose(0,1).contiguous()
    root=os.getenv('MELLUM_FORCED_DECODE_DIR')
    all_layers=os.getenv('MELLUM_DECODE_LAYER_CAPTURE','0')=='1'
    if root and (getattr(self,'layer_num',-1)==0 or all_layers) and not getattr(self,'_mellum_decode_capture_saved',False):
        native=self._mellum_original_decode_dispatch(fwd_ctx)
        native_output=native(q.clone(),k,v,k_cache,v_cache,k_scale,v_scale,fwd_ctx)
        target=Path(root)/f'sample_{self._mellum_decode_sample:06d}'
        target.mkdir(parents=True,exist_ok=True)
        torch.save({'query':q.detach().cpu(),'keys':keys.detach().cpu(),'values':values.detach().cpu(),
            'sdpa_output':output.detach().cpu(),'native_output':native_output.detach().cpu(),
            'logical_start':start,'logical_end':n,'cache_format':self._cache_format,
            'block_table':meta.block_tables[0].detach().cpu()},target/(f'layer_{self.layer_num:02d}_attention.pt' if all_layers else 'first_decode_attention.pt'))
        self._mellum_decode_capture_saved=True
    return output


def install_sdpa_decode(impl):
    original=impl.dispatch_backend
    impl._mellum_original_decode_dispatch=original
    def dispatch(self,fwd_ctx):
        if not fwd_ctx.context.is_prefill:return types.MethodType(sdpa_cached_decode,self)
        self._mellum_decode_sample=getattr(self,'_mellum_decode_sample',-1)+1
        self._mellum_decode_capture_saved=False
        return original(fwd_ctx)
    impl.dispatch_backend=types.MethodType(dispatch,impl)
