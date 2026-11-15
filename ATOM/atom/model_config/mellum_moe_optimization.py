# SPDX-License-Identifier: MIT
"""Default-off MoE candidates. CPD remains an explicit requestwise fallback."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

VARIANTS = ('cpd', 'shared_routing', 'indexed_decode', 'prefill_shared_elementwise')


def shared_prefill_elementwise(module, hidden, weights, ids, bounds):
    """Keep each request's sort and two grouped-mm shapes; share elementwise work.

    No expert weight packing, cross-request GEMM or FP32 activation fusion.
    Extra temporary storage grows with scheduled tokens, not the context limit.
    """
    from atom.model_config.mellum_experts import grouped_linear
    offsets_parts, permutations, gates, ups = [], [], [], []
    for a, b in zip(bounds, bounds[1:]):
        expert_ids, perm = ids[a:b].reshape(-1).sort()
        selected = hidden[a:b][perm // module.top_k]
        counts = torch.histc(expert_ids.int() if selected.is_cuda else expert_ids.float(),
                            bins=module.num_experts, min=0, max=module.num_experts-1)
        offsets = counts.cumsum(0, dtype=torch.int32)
        gate, up = grouped_linear(selected, module.w13_weight, offsets).chunk(2, dim=-1)
        offsets_parts.append(offsets)
        permutations.append(perm + a * module.top_k)
        gates.append(gate)
        ups.append(up)
    # Preserve the reference's separate BF16 silu and multiplication rounding.
    activated = F.silu(torch.cat(gates)) * torch.cat(ups)
    del gates, ups
    projections = []
    for a, b, offsets in zip(bounds, bounds[1:], offsets_parts):
        projections.append(grouped_linear(activated[a*module.top_k:b*module.top_k],
                                          module.w2_weight, offsets))
    perm = torch.cat(permutations)
    weighted = torch.cat(projections) * weights.reshape(-1)[perm, None]
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return weighted[inverse].view(len(hidden), module.top_k, -1).sum(1)


def optimized_experts(module, hidden, logits, bounds, batched, variant):
    if variant not in VARIANTS: raise ValueError('Unknown MoE variant')
    if bounds[0] != 0 or bounds[-1] != len(hidden) or any(b<=a for a,b in zip(bounds,bounds[1:])):
        raise ValueError('Invalid MoE request boundaries')
    if variant == 'cpd':
        return torch.cat([module(hidden[a:b],logits[a:b],batched=batched)
                          for a,b in zip(bounds,bounds[1:])],dim=0)
    if module.custom_routing_function is not None:
        raise ValueError('Shared routing candidate does not support custom routing')
    weights, ids = module.route(hidden, logits)
    if variant == 'prefill_shared_elementwise' and not batched:
        return shared_prefill_elementwise(module, hidden, weights, ids, bounds)
    if variant == 'indexed_decode' and batched:
        if any(b-a != 1 for a,b in zip(bounds,bounds[1:])):
            raise ValueError('Indexed decode requires one token per real request')
        from atom.model_config.mellum_indexed_gemv import indexed_linear
        gate,up=indexed_linear(hidden,ids,module.w13_weight,module.top_k).chunk(2,-1)
        # Preserve two BF16 rounding points of the reference activation.
        activated=F.silu(gate)*up
        projected=indexed_linear(activated,ids,module.w2_weight,1)
        weighted=projected*weights.reshape(-1,1)
        return weighted.view(len(hidden),module.top_k,-1).sum(1).to(hidden.dtype)
    return torch.cat([module.forward_routed(hidden[a:b],weights[a:b],ids[a:b],batched=batched)
                      for a,b in zip(bounds,bounds[1:])],dim=0)


def install_experts(campaign, module, layer):
    original=module.forward
    # Reference CPD calls must bypass this wrapper to avoid recursive dispatch.
    class Original:
        def __getattr__(self,name):return getattr(module,name)
        def __call__(self,*args,**kwargs):return original(*args,**kwargs)
    reference=Original()
    def forward(hidden,logits,batched=False):
        if not (campaign.active and campaign.bounds and campaign.phase in campaign.config.get('phases',[])):
            return original(hidden,logits,batched=batched)
        variant=campaign.moe_variant
        output=optimized_experts(reference,hidden,logits,campaign.bounds,batched,variant)
        audit=campaign.config.get('moe_audit')
        if audit and layer==0:
            with Path(audit).open('a') as stream:
                stream.write(json.dumps({'phase':campaign.phase,'variant':variant,'rows':len(hidden),
                    'requests':len(campaign.bounds)-1,
                    'backend':'indexed_triton' if variant=='indexed_decode' and batched else
                              'requestwise_gemm_shared_elementwise' if variant=='prefill_shared_elementwise' and not batched else
                              'requestwise_gemm_shared_route' if variant!='cpd' else 'original_cpd'})+'\n')
        return output
    module.forward=forward
