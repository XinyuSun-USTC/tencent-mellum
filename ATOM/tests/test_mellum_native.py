"""Native Mellum contracts without importing GPU operators."""
import sys
from types import SimpleNamespace as NS
import torch
import pytest
from atom.model_config.mellum_native import hf_route,native_prefill,install_native_attention,MellumFusedMoeBlock,record


@pytest.mark.parametrize('normalize',[False,True])
def test_native_route_preserves_BF16_rounded_values_in_AITER_FP32_storage(normalize):
    x=torch.randn(5,7).bfloat16();logits=torch.randn(5,9).bfloat16()
    expected,ids=logits.float().softmax(-1).topk(3,dim=-1)
    if normalize:expected/=expected.sum(-1,keepdim=True)
    weights,actual=hf_route(x,logits,3,normalize)
    assert weights.dtype==torch.float32 and actual.dtype==torch.int32
    torch.testing.assert_close(weights,expected.bfloat16().float(),rtol=0,atol=0)
    assert torch.equal(actual.long(),ids)


@pytest.mark.parametrize('window',[-1,4])
def test_native_varlen_has_no_cross_request_attention_and_window_includes_current(monkeypatch,window):
    calls=[]
    def flash(**kwargs):
        calls.append(kwargs);q,k,v=kwargs['q'],kwargs['k'],kwargs['v'];cu=kwargs['cu_seqlens_q'].tolist();outputs=[]
        for a,b in zip(cu,cu[1:]):
            n=b-a;mask=torch.ones(n,n,dtype=torch.bool).tril()
            if kwargs['window_size'][0]>=0:mask=mask.triu(-kwargs['window_size'][0])
            outputs.append(torch.nn.functional.scaled_dot_product_attention(q[a:b].transpose(0,1)[None],
                k[a:b].transpose(0,1)[None],v[a:b].transpose(0,1)[None],attn_mask=mask,
                scale=kwargs['softmax_scale'],enable_gqa=True)[0].transpose(0,1))
        return torch.cat(outputs)
    monkeypatch.setitem(sys.modules,'aiter',NS(flash_attn_varlen_func=flash))
    impl=NS(layer_num=1,sliding_window=window,scale=.5)
    q=torch.zeros(11,4,4);k=torch.zeros(11,2,4)
    v=torch.arange(11).float()[:,None,None].expand(11,2,4).contiguous()
    cu=torch.tensor([0,5,11],dtype=torch.int32)
    ctx=NS(attn_metadata=NS(cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=6,max_seqlen_k=6),context=NS(is_prefill=True))
    output=native_prefill(impl,q,k,v,None,None,None,None,ctx)
    # Last token means last W values, including itself, or complete request values.
    expected=v[(7 if window>0 else 5):11].mean(0)
    torch.testing.assert_close(output[-1],expected.repeat_interleave(2,0))
    assert calls[0]['window_size']==((window-1,0,0) if window>0 else (-1,-1,0))
    changed=v.clone();changed[5:]=10000
    again=native_prefill(impl,q,k,changed,None,None,None,None,ctx)
    torch.testing.assert_close(again[:5],output[:5])
    ctx.attn_metadata.cu_seqlens_k=torch.tensor([0,6,11])
    with pytest.raises(ValueError,match='prefix/chunked'):native_prefill(impl,q,k,v,None,None,None,None,ctx)


def test_native_decode_uses_original_paged_backend_once_for_whole_batch(monkeypatch):
    monkeypatch.delenv('MELLUM_NATIVE_AUDIT_PATH',raising=False)
    calls=[]
    def paged_attention_triton(q,*args):calls.append(len(q));return q+1
    impl=NS(layer_num=1,dispatch_backend=lambda ctx:paged_attention_triton,_mellum_stepwise_rope=True)
    install_native_attention(impl)
    ctx=NS(context=NS(is_prefill=False),attn_metadata=NS(context_lens=torch.tensor([2,7,5])))
    q=torch.zeros(3,4,8);value=impl.dispatch_backend(ctx)(q,None,None,None,None,None,None,ctx)
    assert calls==[3] and torch.equal(value,q+1) and not impl._mellum_stepwise_rope
    impl=NS(dispatch_backend=lambda ctx:lambda *args:None)
    install_native_attention(impl)
    with pytest.raises(ValueError,match='paged'):impl.dispatch_backend(ctx)


def test_fused_block_executes_all_rows_once_and_preserves_reference_gate(monkeypatch):
    calls=[];created=[]
    class Fused(torch.nn.Module):
        def __init__(self,**kwargs):super().__init__();created.append(kwargs)
        def forward(self,hidden_states,router_logits):
            calls.append((len(hidden_states),router_logits.clone()));return hidden_states+1
    def linear(a,b,bias,**kwargs):return torch.nn.Linear(a,b,bias=bias).bfloat16()
    monkeypatch.setitem(sys.modules,'atom.model_ops.moe',NS(FusedMoE=Fused))
    monkeypatch.setitem(sys.modules,'atom.model_ops.linear',NS(ReplicatedLinear=linear))
    monkeypatch.delenv('MELLUM_NATIVE_AUDIT_PATH',raising=False)
    cfg=NS(hidden_size=7,num_experts=5,num_experts_per_tok=2,moe_intermediate_size=3,norm_topk_prob=True)
    block=MellumFusedMoeBlock(cfg,prefix='model.layers.0.mlp');x=torch.randn(13,7).bfloat16()
    assert torch.equal(block(x),x+1) and [r[0] for r in calls]==[13]
    torch.testing.assert_close(calls[0][1],torch.nn.functional.linear(x,block.gate.weight),rtol=0,atol=0)
    assert created[0]['custom_routing_function'] is hf_route and created[0]['renormalize']
    assert created[0]['intermediate_size']==3 and not created[0]['has_bias']


def test_native_audit_ignores_dummy_forward(monkeypatch,tmp_path):
    p=tmp_path/'audit.jsonl';monkeypatch.setenv('MELLUM_NATIVE_AUDIT_PATH',str(p))
    record('moe',0,NS(attn_metadata=None,context=NS(is_dummy_run=True)),1,'FusedMoE')
    assert not p.exists()


@pytest.mark.parametrize('mode',['prefill_only','decode_only'])
def test_split_installer_keeps_exactly_one_phase_reference(monkeypatch,mode):
    import types
    import atom.model_config.mellum_native as native
    import atom.model_config.mellum_batch as batch
    calls=[]
    def native_pre(self,*args):calls.append('native_pre');return args[0]+10
    def ref_pre(self,*args):calls.append('ref_pre');return args[0]+20
    def ref_dec(self,*args):calls.append('ref_dec');return args[0]+30
    def paged_attention_triton(*args):calls.append('native_dec');return args[0]+40
    monkeypatch.setattr(native,'native_prefill',native_pre)
    monkeypatch.setattr(batch,'ragged_prefill',ref_pre);monkeypatch.setattr(batch,'ragged_decode',ref_dec)
    monkeypatch.setattr(native,'record',lambda *a,**kw:None)
    impl=NS(layer_num=1,dispatch_backend=lambda ctx:paged_attention_triton)
    native.install_native_attention(impl,mode)
    q=torch.zeros(2,3);ctx=NS(context=NS(is_prefill=True),attn_metadata=None)
    impl.dispatch_backend(ctx)(q,None,None,None,None,None,None,ctx)
    ctx.context.is_prefill=False;impl.dispatch_backend(ctx)(q,None,None,None,None,None,None,ctx)
    assert calls==(['native_pre','ref_dec'] if mode=='prefill_only' else ['ref_pre','native_dec'])
    assert impl._mellum_stepwise_rope==(mode=='decode_only')
