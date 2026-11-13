import types
import pytest
import torch
from atom.model_config.mellum_experts import MellumTorchExperts

def config():
    return types.SimpleNamespace(num_experts=4,moe_intermediate_size=3,
        hidden_size=8,num_experts_per_tok=2,norm_topk_prob=True)

def test_expert_loading_shards_and_rejections():
    module=MellumTorchExperts(config())
    gate=torch.arange(24).reshape(3,8).float()
    module.weight_loader(module.w13_weight,gate,'name','w1',2)
    module.weight_loader(module.w13_weight,gate+10,'name','w3',2)
    assert torch.equal(module.w13_weight[2,:3],gate)
    assert torch.equal(module.w13_weight[2,3:],gate+10)
    with pytest.raises(ValueError):module.weight_loader(module.w13_weight,gate,'name','w2',2)
    with pytest.raises(ValueError):module.weight_loader(module.w13_weight,gate,'name','w1',4)
    with pytest.raises(ValueError):module.weight_loader(module.w13_weight,gate[:1],'name','w1',0)

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_expert_math_matches_explicit_token_reference(dtype):
    torch.manual_seed(37)
    module=MellumTorchExperts(config()).to(dtype)
    with torch.no_grad():
        module.w13_weight.copy_(torch.randn_like(module.w13_weight)/4)
        module.w2_weight.copy_(torch.randn_like(module.w2_weight)/4)
    x=torch.randn(5,8).to(dtype);logits=torch.randn(5,4).to(dtype)
    weights,ids=logits.float().softmax(-1).topk(2,-1)
    weights=(weights/weights.sum(-1,keepdim=True)).to(dtype)
    expected=[]
    for t in range(5):
        contributions=[]
        for k in range(2):
            e=ids[t,k]
            gate,up=torch.nn.functional.linear(x[t:t+1],module.w13_weight[e]).chunk(2,-1)
            z=torch.nn.functional.linear(torch.nn.functional.silu(gate)*up,module.w2_weight[e])
            contributions.append(z*weights[t,k])
        expected.append(torch.stack(contributions).sum(0))
    expected=torch.cat(expected)
    torch.testing.assert_close(module(x,logits),expected,rtol=1e-5 if dtype==torch.float32 else 0,
                               atol=1e-7 if dtype==torch.float32 else 0)
