import pytest
import torch
from atom.model_config.mellum_projection import install_separate_qkv
from atom.model_config.mellum_projection import install_reference_router

def test_reference_router_keeps_parameters_loading_and_hooks():
    module=torch.nn.Linear(8,4,bias=False).bfloat16()
    weight=module.weight;sentinel=object();weight.weight_loader=sentinel
    seen=[];module.register_forward_hook(lambda m,a,o:seen.append(o.clone()))
    install_reference_router(module)
    x=torch.randn(1,8).bfloat16()
    expected=torch.nn.functional.linear(x,weight)
    torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
    assert module.weight is weight and weight.weight_loader is sentinel
    assert torch.equal(seen[0],expected) and list(module.state_dict())==['weight']
    with pytest.raises(ValueError):module(x.float())

def test_separate_qkv_keeps_parameters_loaders_and_observers():
    torch.manual_seed(13)
    module=torch.nn.Linear(8,16,bias=True).bfloat16()
    weight=module.weight;bias=module.bias
    sentinel=object();weight.weight_loader=sentinel
    outputs=[];module.register_forward_hook(lambda m,a,o:outputs.append(o.detach().clone()))
    install_separate_qkv(module,8,4)
    x=torch.randn(7,8).bfloat16()
    expected=torch.cat([torch.nn.functional.linear(x,w,b) for w,b in zip(weight.split([8,4,4]),bias.split([8,4,4]))],-1)
    torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
    assert module.weight is weight and module.bias is bias and weight.weight_loader is sentinel
    assert list(module.state_dict())==['weight','bias']
    assert len(outputs)==1 and torch.equal(outputs[0],expected)
    with pytest.raises(ValueError,match='BF16'):module(x.float())
    with pytest.raises(ValueError):module(x,x_scale=torch.ones(1))

def test_separate_qkv_rejects_noncanonical_layout():
    with pytest.raises(ValueError):install_separate_qkv(torch.nn.Linear(8,16),8,4)
    with pytest.raises(ValueError):install_separate_qkv(torch.nn.Linear(8,15).bfloat16(),8,4)
