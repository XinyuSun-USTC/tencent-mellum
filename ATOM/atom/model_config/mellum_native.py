# SPDX-License-Identifier: MIT
"""Opt-in native ATOM execution. No fallback to reference math on kernel failure."""
import json
import os
import types
from pathlib import Path
import torch


def hf_route(hidden_states, gating_output, topk, renormalize):
    """Keep the verified softmax/top-k normalization; expert math is native."""
    weights, ids = gating_output.float().softmax(-1).topk(topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    # Frozen AITER sorting requires FP32 storage; preserve BF16-rounded values.
    return weights.to(gating_output.dtype).float(), ids.to(torch.int32)


def record(kind, layer, ctx, rows, backend, **extra):
    path = os.getenv('MELLUM_NATIVE_AUDIT_PATH')
    if not path or layer != 0 or getattr(ctx,'in_hipgraph',False):
        return
    meta = ctx.attn_metadata
    if meta is None or getattr(ctx.context,'is_dummy_run',False):
        return
    phase = 'prefill' if ctx.context.is_prefill else 'decode'
    requests = meta.cu_seqlens_q.numel()-1 if phase == 'prefill' else meta.context_lens.numel()
    with Path(path).open('a') as stream:
        stream.write(json.dumps(dict(kind=kind,phase=phase,rows=rows,requests=requests,
                                    backend=backend,**extra))+'\n')


class MellumFusedMoeBlock(torch.nn.Module):
    """Frozen FusedMoE on all scheduled tokens, with Mellum's reference gate."""
    def __init__(self, config, quant_config=None, prefix=''):
        super().__init__()
        from atom.model_ops.moe import FusedMoE
        from atom.model_ops.linear import ReplicatedLinear
        from atom.model_config.mellum_projection import install_reference_router
        self.gate = ReplicatedLinear(config.hidden_size,config.num_experts,bias=False,
                                     quant_config=None,prefix=f'{prefix}.gate')
        install_reference_router(self.gate)
        self.experts = FusedMoE(num_experts=config.num_experts,top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,intermediate_size=config.moe_intermediate_size,
            reduce_results=False,renormalize=config.norm_topk_prob,quant_config=quant_config,
            use_grouped_topk=False,has_bias=False,custom_routing_function=None if os.getenv('MELLUM_NATIVE_ROUTING','0')=='1' else hf_route,
            prefix=f'{prefix}.experts',config=config)
        self.layer_num = int(prefix.split('.layers.')[-1].split('.')[0]) if '.layers.' in prefix else -1

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        result = self.experts(hidden_states=hidden_states,router_logits=logits)
        if os.getenv('MELLUM_NATIVE_AUDIT_PATH') and self.layer_num == 0:
            from atom.utils.forward_context import get_forward_context
            record('moe',self.layer_num,get_forward_context(),len(hidden_states),'FusedMoE',
                   quant_method=type(self.experts.quant_method).__name__,routing='aiter_native' if os.getenv('MELLUM_NATIVE_ROUTING','0')=='1' else 'torch_softmax_topk',
                   shuffled_w13=bool(getattr(self.experts.w13_weight,'is_shuffled',False)),
                   shuffled_w2=bool(getattr(self.experts.w2_weight,'is_shuffled',False)))
        return result


def isolated_fused_experts(hidden, weights, ids, w13, w2):
    """One pseudo-token per route; no cross-expert accumulation in native output.

    The native kernel may still use atomic instructions, but only one logical
    expert contributes to each pseudo-token. This is not a claim about its
    internal split-K implementation or numerical equivalence to reference GEMM.
    """
    from aiter.fused_moe import fused_moe
    from aiter import ActivationType,QuantType
    if hidden.ndim!=2 or weights.shape!=ids.shape or len(ids)!=len(hidden) or ids.ndim!=2:
        raise ValueError('Invalid isolated expert route geometry')
    if hidden.dtype!=torch.bfloat16 or w13.dtype!=hidden.dtype or w2.dtype!=hidden.dtype:
        raise ValueError('Isolated experts require unquantized BF16')
    if ids.dtype!=torch.int32 or weights.dtype!=torch.float32:
        raise ValueError('Isolated experts require AITER int32 IDs and FP32 route storage')
    n,k=ids.shape
    expanded=hidden.repeat_interleave(k,dim=0)
    assigned=ids.reshape(-1,1).contiguous()
    unit=torch.ones((n*k,1),dtype=torch.float32,device=hidden.device)
    contributions=fused_moe(expanded,w13,w2,unit,assigned,
                            activation=ActivationType.Silu,quant_type=QuantType.No)
    if contributions.shape!=(n*k,hidden.shape[1]):raise ValueError('Unexpected isolated native output geometry')
    # Match reference's BF16 weighting, then a fixed top-k reduction tree.
    weighted=contributions.view(n,k,-1)*weights.to(hidden.dtype).unsqueeze(-1)
    return weighted.sum(1)


class MellumIsolatedMoeBlock(MellumFusedMoeBlock):
    """Retain canonical loading/shuffle; bypass only native expert combination."""
    def forward(self, hidden_states):
        logits=self.gate(hidden_states)
        weights,ids=hf_route(hidden_states,logits,self.experts.top_k,self.experts.renormalize)
        result=isolated_fused_experts(hidden_states,weights,ids,
                                     self.experts.w13_weight,self.experts.w2_weight)
        if os.getenv('MELLUM_NATIVE_AUDIT_PATH') and self.layer_num==0:
            from atom.utils.forward_context import get_forward_context
            record('moe',0,get_forward_context(),len(hidden_states),'isolated_fused_experts',
                   expanded_rows=ids.numel(),logical_topk=ids.shape[1],native_topk=1,
                   quant_method=type(self.experts.quant_method).__name__,
                   shuffled_w13=bool(getattr(self.experts.w13_weight,'is_shuffled',False)),
                   shuffled_w2=bool(getattr(self.experts.w2_weight,'is_shuffled',False)))
        return result


def native_prefill(self,q,k,v,k_cache,v_cache,k_scale,v_scale,ctx):
    """Uncached ragged GQA prefill, window counts history excluding current token."""
    import aiter
    from atom.model_config.mellum_batch import audit
    meta=ctx.attn_metadata
    if q.shape[0]!=k.shape[0] or k.shape!=v.shape or meta.cu_seqlens_q.numel()!=meta.cu_seqlens_k.numel():
        raise ValueError('Native Mellum prefill requires full uncached ragged Q/K/V')
    if not torch.equal(meta.cu_seqlens_q,meta.cu_seqlens_k):
        raise ValueError('Native Mellum first version does not enable prefix/chunked prefill')
    if getattr(self,'layer_num',-1)==0:
        bounds=meta.cu_seqlens_q.tolist()
        audit(self,'prefill',[b-a for a,b in zip(bounds,bounds[1:])])
    window=(self.sliding_window-1,0,0) if self.sliding_window>0 else (-1,-1,0)
    output=aiter.flash_attn_varlen_func(q=q.contiguous(),k=k.contiguous(),v=v.contiguous(),
        cu_seqlens_q=meta.cu_seqlens_q,cu_seqlens_k=meta.cu_seqlens_k,
        max_seqlen_q=meta.max_seqlen_q,max_seqlen_k=meta.max_seqlen_k,
        dropout_p=0.0,softmax_scale=self.scale,causal=True,window_size=window,
        return_lse=False,return_attn_probs=False)
    record('attention',getattr(self,'layer_num',-1),ctx,len(q),'aiter_varlen',window=window)
    return output


def install_native_attention(impl, mode='both'):
    """Explicit phase ablations; reference phase is deliberate, never fallback."""
    if mode not in ('both','prefill_only','decode_only'):
        raise ValueError('Unknown native attention phase mode')
    original = impl.dispatch_backend
    use_native_prefill = mode != 'decode_only'
    use_native_decode = mode != 'prefill_only'
    from atom.model_config.mellum_batch import ragged_prefill,ragged_decode,audit
    impl._mellum_stepwise_rope = not use_native_prefill
    impl.prefill_attention = types.MethodType(native_prefill if use_native_prefill else ragged_prefill,impl)
    def dispatch(self,ctx):
        if ctx.context.is_prefill:
            if use_native_prefill:return self.prefill_attention
            def prefill(*args):
                result=self.prefill_attention(*args)
                record('attention',getattr(self,'layer_num',-1),ctx,len(args[0]),'reference_ragged_prefill',mode=mode)
                return result
            return prefill
        backend = original(ctx) if use_native_decode else types.MethodType(ragged_decode,self)
        if use_native_decode and 'paged_attention' not in backend.__name__:
            raise ValueError('Native Mellum decode did not select a paged backend')
        def decode(q,k,v,k_cache,v_cache,k_scale,v_scale,fwd_ctx):
            if use_native_decode and getattr(self,'layer_num',-1)==0 and not getattr(fwd_ctx,'in_hipgraph',False):
                audit(self,'decode',fwd_ctx.attn_metadata.context_lens.tolist())
            result=backend(q,k,v,k_cache,v_cache,k_scale,v_scale,fwd_ctx)
            name=backend.__name__ if use_native_decode else 'reference_ragged_decode'
            record('attention',getattr(self,'layer_num',-1),fwd_ctx,len(q),name,mode=mode)
            return result
        return decode
    impl.dispatch_backend = types.MethodType(dispatch,impl)
