"""Requestwise semantics, shared routing and selected-weight layout contracts."""
from types import SimpleNamespace as NS
import torch
import pytest
from atom.model_config.mellum_experts import MellumTorchExperts
from atom.model_config.mellum_moe_optimization import optimized_experts,install_experts
from atom.model_config.mellum_indexed_gemv import indexed_linear


def experts(dtype=torch.bfloat16,normalize=True):
    torch.manual_seed(717)
    cfg=NS(num_experts=5,moe_intermediate_size=7,hidden_size=11,num_experts_per_tok=3,norm_topk_prob=normalize)
    module=MellumTorchExperts(cfg).to(dtype)
    with torch.no_grad():
        module.w13_weight.normal_(std=.1);module.w2_weight.normal_(std=.1)
    return module


@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('normalize',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_shared_route_preserves_requestwise_reference(batched,normalize,dtype):
    module=experts(dtype,normalize);bounds=list(range(9)) if batched else [0,2,7,8]
    x=torch.randn(8,11).to(dtype);logits=torch.randn(8,5).to(dtype)
    expected=torch.cat([module(x[a:b],logits[a:b],batched=batched) for a,b in zip(bounds,bounds[1:])])
    for mode in ('cpd','shared_routing'):
        torch.testing.assert_close(optimized_experts(module,x,logits,bounds,batched,mode),expected,rtol=0,atol=0)


def test_shared_routing_computed_once_experts_keep_request_shapes():
    module=experts();x=torch.randn(6,11).bfloat16();logits=torch.randn(6,5).bfloat16()
    route=module.route;forward=module.forward_routed;routes=[];sizes=[]
    def routed(x,l):routes.append(len(x));return route(x,l)
    def compute(x,*args,**kwargs):sizes.append(len(x));return forward(x,*args,**kwargs)
    module.route=routed;module.forward_routed=compute
    optimized_experts(module,x,logits,[0,2,3,6],False,'shared_routing')
    assert routes==[6] and sizes==[2,1,3]


def test_installed_variant_switch_and_inactive_path_no_recursion():
    module=experts();x=torch.randn(4,11).bfloat16();logits=torch.randn(4,5).bfloat16()
    baseline=torch.cat([module(x[i:i+1],logits[i:i+1],batched=True) for i in range(4)])
    native=module(x,logits,batched=True)
    c=NS(active=True,bounds=[0,1,2,3,4],phase='decode',config={'phases':['prefill','decode']},moe_variant='cpd')
    install_experts(c,module,0)
    torch.testing.assert_close(module(x,logits,batched=True),baseline,rtol=0,atol=0)
    c.moe_variant='shared_routing'
    torch.testing.assert_close(module(x,logits,batched=True),baseline,rtol=0,atol=0)
    c.active=False
    torch.testing.assert_close(module(x,logits,batched=True),native,rtol=0,atol=0)


def test_candidate_refuses_custom_route_bad_bounds_and_non_decode_layout():
    module=experts();x=torch.randn(2,11).bfloat16();logits=torch.randn(2,5).bfloat16()
    with pytest.raises(ValueError,match='boundaries'):optimized_experts(module,x,logits,[0,2,1],False,'shared_routing')
    with pytest.raises(ValueError,match='one token'):optimized_experts(module,x,logits,[0,2],True,'indexed_decode')
    module.custom_routing_function=lambda **kwargs:None
    with pytest.raises(ValueError,match='custom routing'):optimized_experts(module,x,logits,[0,2],False,'shared_routing')
    with pytest.raises(ValueError,match='Unknown'):optimized_experts(module,x,logits,[0,2],False,'other')


def test_indexed_api_never_silently_falls_back_to_cpu():
    with pytest.raises(ValueError,match='GPU BF16'):
        indexed_linear(torch.ones(1,11).bfloat16(),torch.tensor([[0]]),torch.ones(2,7,11).bfloat16(),1)
