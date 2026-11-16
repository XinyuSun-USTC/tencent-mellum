import sys
from types import SimpleNamespace as NS
import torch
from atom.model_config.mellum_phase_moe import MellumPhaseMoeBlock
from atom.model_config.mellum_native import MellumFusedMoeBlock


def test_dual_copy_loader_keeps_canonical_layout_and_registered_keys(monkeypatch):
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w13_weight=torch.nn.Parameter(torch.zeros(2,6,4,dtype=torch.bfloat16));self.w2_weight=torch.nn.Parameter(torch.zeros(2,4,3,dtype=torch.bfloat16))
            def loader(p,w,weight_name='',shard_id='',expert_id=0):
                if shard_id=='w2':p.data[expert_id].copy_(w)
                else:p.data[expert_id,(0 if shard_id=='w1' else 3):(3 if shard_id=='w1' else 6)].copy_(w)
            for p in self.parameters():p.weight_loader=loader
        def forward(self,hidden_states,router_logits):return hidden_states*3
    def init(self,*args,**kwargs):
        torch.nn.Module.__init__(self);self.experts=Experts();self.gate=lambda x:torch.zeros(len(x),2,dtype=x.dtype);self.layer_num=1
    monkeypatch.setattr(MellumFusedMoeBlock,'__init__',init)
    monkeypatch.setenv('MELLUM_NATIVE_EXPERT_PHASE','prefill');monkeypatch.setenv('MELLUM_DECODE_GRAPH','0')
    m=MellumPhaseMoeBlock(NS(num_experts=2,moe_intermediate_size=3,num_experts_per_tok=1,norm_topk_prob=True,hidden_size=4))
    assert list(dict(m.named_parameters()))==['experts.w13_weight','experts.w2_weight']
    for expert in range(2):
        for shard in ['w1','w3','w2']:
            p=m.experts.w2_weight if shard=='w2' else m.experts.w13_weight
            w=torch.full((4,3) if shard=='w2' else (3,4),expert+1,dtype=torch.bfloat16)
            p.weight_loader(p,w,'checkpoint',shard,expert)
    assert len(m._reference_loaded)==6
    assert torch.equal(m.experts.w13_weight,m._reference_experts.w13_weight)
    assert torch.equal(m.experts.w2_weight,m._reference_experts.w2_weight)
    ctx=NS(context=NS(is_prefill=True))
    monkeypatch.setitem(sys.modules,'atom.utils.forward_context',NS(get_forward_context=lambda:ctx))
    monkeypatch.setattr(m._reference_experts,'forward',lambda hidden,logits,batched=False:hidden*2)
    x=torch.ones(1,4,dtype=torch.bfloat16)
    assert torch.equal(m(x),x*3)
    ctx.context.is_prefill=False
    assert torch.equal(m(x),x*2)
