# SPDX-License-Identifier: MIT
"""Write reference-rotated BF16 K/V once with the existing AITER cache kernel."""
import types
import os
import torch


def install_direct_kv_rope(impl,inv,factor):
    from atom.model_config.mellum_attention import reference_position_rope
    resident=os.getenv('MELLUM_RESIDENT_ROPE','0')=='1'
    def direct(self,q,k,v,qkv,position,ctx):
        if (getattr(self,'use_flash_layout',False) or self.kv_cache_dtype!='bf16'
                or self.q_norm is not None or self.k_norm is not None):
            raise ValueError('Direct reference KV requires BF16 non-flash cache and separate QK norm')
        meta=ctx.attn_metadata
        cached_suffix=bool(getattr(meta,'has_cached',False)) and ctx.context.is_prefill
        if cached_suffix and os.getenv('MELLUM_PREFIX_COMPAT','0')!='1':
            raise ValueError('Direct KV cached/chunked prefill requires explicit prefix compatibility')
        data=ctx.kv_cache_data[f'layer_{self.layer_num}']
        kc,vc,ks,vs=data.k_cache,data.v_cache,data.k_scale,data.v_scale
        if kc.ndim!=5 or kc.dtype!=torch.bfloat16 or vc.dtype!=kc.dtype or k.dtype!=kc.dtype or v.dtype!=kc.dtype:
            raise ValueError('Direct reference KV requires existing five-dimensional BF16 K layout')
        if (q.ndim!=3 or k.ndim!=3 or v.shape!=k.shape or len(q)!=len(k)
                or q.shape[-1]!=k.shape[-1] or meta.slot_mapping.numel()!=len(k)
                or kc.shape[1]!=k.shape[1] or kc.shape[2]*kc.shape[4]!=k.shape[2]
                or vc.numel()!=kc.numel()):
            raise ValueError('Direct reference KV token/head/cache shape mismatch')
        frequency=inv
        if resident:
            cached=getattr(self,'_mellum_gpu_inverse_frequency',None)
            if cached is None or cached.device!=q.device:
                cached=inv.to(device=q.device,dtype=torch.float32)
                self._mellum_gpu_inverse_frequency=cached
            frequency=cached
        fused=os.getenv('MELLUM_FUSED_ROPE','0')=='1' or hasattr(self,'_mellum_fused_norms')
        if fused:
            from atom.model_config.mellum_fused_rope import fused_qk_rope
            qr,kr=fused_qk_rope(q,k,position,frequency,factor,getattr(self,'_mellum_fused_norms',None))
        else:
            qr,kr=reference_position_rope(q,k,position,frequency,factor)
        self.use_triton_attn=(self.sliding_window!=-1 or self.head_dim!=128 or self.num_heads==self.num_kv_heads)
        asm_layout=not (self.use_triton_attn and vc.dim()!=5)
        self._cache_format='SHUFFLE' if asm_layout else 'NHD'
        import aiter
        aiter.reshape_and_cache(kr,v,kc,vc,meta.slot_mapping,kv_cache_dtype='auto',
            k_scale=None,v_scale=None,asm_layout=asm_layout)
        # Audit once per real phase rather than doing per-token file I/O.
        if self.layer_num==0 and not getattr(ctx.context,'is_dummy_run',False) and not getattr(ctx,'in_hipgraph',False):
            phase='prefill' if ctx.context.is_prefill else 'decode'
            seen=getattr(self,'_mellum_direct_kv_audited',set())
            if phase not in seen:
                from atom.model_config.mellum_native import record
                record('kv_write',self.layer_num,ctx,len(q),'aiter_reference_kv_once',cache_format=self._cache_format,resident_inverse=resident,inverse_device=str(frequency.device),rope_backend='triton_qk_norm_rope' if hasattr(self,'_mellum_fused_norms') else 'triton_rope' if fused else 'torch')
                seen.add(phase);self._mellum_direct_kv_audited=seen
        if cached_suffix:
            from atom.model_config.mellum_prefix import gather_direct_prefix
            return gather_direct_prefix(self,qr,kr,v,kc,vc,ks,vs,ctx)
        return qr,kr,v,kc,vc,ks,vs
    impl.rope_cache=types.MethodType(direct,impl)
