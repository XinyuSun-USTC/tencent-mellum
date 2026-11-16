# SPDX-License-Identifier: MIT
"""Default-off MoE candidates. CPD remains an explicit requestwise fallback."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

VARIANTS = ('cpd', 'shared_routing', 'indexed_decode')


def decode_expert_bounds(bounds, group_size, batched):
    if type(group_size) is not int or group_size not in (1,2,4):
        raise ValueError('Decode expert grouping supports1/2/4 requests')
    if not batched or group_size==1:return bounds
    if not bounds or bounds[0]!=0 or any(b-a!=1 for a,b in zip(bounds,bounds[1:])):
        raise ValueError('Grouped expert decode needs one token per real request')
    return [bounds[0],*bounds[group_size:-1:group_size],bounds[-1]]


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
    group_size=campaign.config.get('decode_expert_group_size',1)
    decode_expert_bounds([0,1],group_size,False)
    # Reference CPD calls must bypass this wrapper to avoid recursive dispatch.
    class Original:
        def __getattr__(self,name):return getattr(module,name)
        def __call__(self,*args,**kwargs):return original(*args,**kwargs)
    reference=Original()
    def forward(hidden,logits,batched=False):
        if not (campaign.active and campaign.bounds and campaign.phase in campaign.config.get('phases',[])):
            return original(hidden,logits,batched=batched)
        variant=campaign.moe_variant
        if group_size>1 and variant!='shared_routing':
            raise ValueError('Grouped decode requires the reference shared routing variant')
        bounds=decode_expert_bounds(campaign.bounds,group_size,batched)
        output=optimized_experts(reference,hidden,logits,bounds,batched,variant)
        audit=campaign.config.get('moe_audit')
        if audit and layer==0:
            with Path(audit).open('a') as stream:
                stream.write(json.dumps({'phase':campaign.phase,'variant':variant,'rows':len(hidden),
                    'requests':len(campaign.bounds)-1,
                    'expert_groups':len(bounds)-1,
                    'decode_group_size':group_size if batched else 1,
                    'backend':'indexed_triton' if variant=='indexed_decode' and batched else
                              'requestwise_gemm_shared_route' if variant!='cpd' else 'original_cpd'})+'\n')
        return output
    module.forward=forward
