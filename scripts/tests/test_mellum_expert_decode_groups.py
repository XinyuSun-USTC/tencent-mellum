import sys,json,importlib.util
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from mellum_execution_profiles import profile_settings
spec=importlib.util.spec_from_file_location('expert_ops',ROOT/'ATOM/atom/model_config/mellum_moe_optimization.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_decode_groups_preserve_real_request_order_and_remainder():
    assert m.decode_expert_bounds(list(range(8)),2,True)==[0,2,4,6,7]
    assert m.decode_expert_bounds(list(range(8)),4,True)==[0,4,7]
    assert m.decode_expert_bounds([0,8,19],4,False)==[0,8,19]
    with pytest.raises(ValueError):m.decode_expert_bounds([0,2,3],2,True)
    with pytest.raises(ValueError):m.decode_expert_bounds([0,1],True,True)


def test_only_expert_decode_group_configuration_changes():
    base=profile_settings('r')
    for size in [2,4]:
        candidate=profile_settings('r_expert_decode'+str(size));key='MELLUM_BATCH_CAMPAIGN_CONFIG'
        assert {k:v for k,v in candidate.items() if k!=key}=={k:v for k,v in base.items() if k!=key}
        assert json.loads(candidate[key])=={**json.loads(base[key]),'decode_expert_group_size':size}


def test_grouped_partition_does_not_mix_tokens_or_routes_on_cpu():
    class Expert:
        custom_routing_function=None
        def route(self,h,l):return torch.ones(len(h),1),torch.zeros(len(h),1,dtype=torch.long)
        def forward_routed(self,h,w,i,batched=False):return h*w+1
    hidden=torch.arange(21,dtype=torch.float32).reshape(7,3);logits=torch.ones(7,1);expert=Expert()
    for size in [1,2,4]:
        bounds=m.decode_expert_bounds(list(range(8)),size,True)
        out=m.optimized_experts(expert,hidden,logits,bounds,True,'shared_routing')
        assert torch.equal(out,hidden+1)


def test_installed_wrapper_groups_decode_but_keeps_prefill_request_shapes():
    from types import SimpleNamespace as NS
    class Expert:
        custom_routing_function=None
        def __init__(self):self.calls=[]
        def route(self,h,l):return torch.ones(len(h),1),torch.zeros(len(h),1,dtype=torch.long)
        def forward_routed(self,h,w,i,batched=False):self.calls.append((len(h),batched));return h+1
        def forward(self,h,l,batched=False):
            w,i=self.route(h,l);return self.forward_routed(h,w,i,batched)
    campaign=NS(config={'decode_expert_group_size':2,'phases':['prefill','decode']},active=True,phase='prefill',bounds=[0,2,5],moe_variant='shared_routing')
    expert=Expert();m.install_experts(campaign,expert,0)
    expert.forward(torch.zeros(5,3),torch.ones(5,1),batched=False)
    assert expert.calls==[(2,False),(3,False)]
    expert.calls=[];campaign.phase='decode';campaign.bounds=list(range(5))
    out=expert.forward(torch.arange(12).reshape(4,3).float(),torch.ones(4,1),batched=True)
    assert expert.calls==[(2,True),(2,True)] and torch.equal(out,torch.arange(12).reshape(4,3)+1)
