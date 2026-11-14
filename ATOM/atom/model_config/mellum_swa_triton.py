# SPDX-License-Identifier: MIT
"""Screening adapter for the frozen AITER Triton local-prefill implementation.

Not connected to the production Mellum backend until GPU screening is reviewed.
"""
import importlib


def triton_swa_prefill(q,k,v,cu_seqlens,window,scale):
    if window<2:
        raise ValueError('Screening adapter requires window>=2 (zero disables Triton local mask)')
    if q.ndim!=3 or k.ndim!=3 or v.shape!=k.shape or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1] or q.shape[1]%k.shape[1]:
        raise ValueError('Equal-length THD QKV with valid GQA heads required')
    mha=importlib.import_module('aiter.ops.triton.attention.mha')
    if mha._MHA_IMPL!='default':
        raise ValueError('Screening requires default Triton MHA; dao_ai does not support local attention')
    # This frozen backend rejects right != -1. causal=True supplies the upper
    # bound; its kernel uses k >= q - SLIDING_WINDOW inclusively for the lower.
    return mha.flash_attn_varlen_func(q=q.contiguous(),k=k.contiguous(),v=v.contiguous(),
        cu_seqlens_q=cu_seqlens,cu_seqlens_k=cu_seqlens,
        max_seqlen_q=q.shape[0],max_seqlen_k=k.shape[0],
        dropout_p=0.,softmax_scale=scale,causal=True,window_size=(window-1,-1),
        return_lse=False,return_attn_probs=False)
