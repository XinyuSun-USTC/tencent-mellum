import types
import pytest
import torch
from atom.model_config.mellum_attention import install_sdpa_prefill
from atom.model_config.mellum_attention import stepwise_rope
from atom.model_config.mellum_attention import cpu_inverse_frequency,reference_position_rope
from atom.model_config.mellum_attention import overwrite_reference_keys
from atom.model_config.mellum_attention import gather_reference_kv,sdpa_cached_decode

@pytest.mark.parametrize('layout',['NHD','SHUFFLE'])
def test_paged_decode_gather_uses_physical_blocks_and_window(layout):
    torch.manual_seed(33)
    keys=torch.randn(18,2,16).bfloat16();values=torch.randn_like(keys)
    kc=torch.zeros(3,2,2,16,8,dtype=torch.bfloat16)
    table=torch.tensor([2,0]);logical=torch.arange(18);slots=table[logical//16]*16+logical%16
    overwrite_reference_keys(kc,keys,slots)
    vc=torch.zeros(3,2,16,16,dtype=torch.bfloat16)
    if layout=='NHD':vc[slots//16,:,:,slots%16]=values
    else:vc.view(3,2,2,16,8)[slots//16,:,(slots%16)//8,:,slots%8]=values
    got_k,got_v=gather_reference_kv(kc,vc,table,2,18,layout)
    torch.testing.assert_close(got_k,keys[2:],rtol=0,atol=0)
    torch.testing.assert_close(got_v,values[2:],rtol=0,atol=0)
    impl=types.SimpleNamespace(sliding_window=16,scale=0.25,_cache_format=layout)
    meta=types.SimpleNamespace(context_lens=torch.tensor([18]),block_tables=table[None])
    ctx=types.SimpleNamespace(attn_metadata=meta,context=types.SimpleNamespace(positions=torch.tensor([17])))
    q=torch.randn(1,4,16).bfloat16()
    got=sdpa_cached_decode(impl,q,keys[-1:],values[-1:],kc,vc,None,None,ctx)
    expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],
        keys[2:].repeat_interleave(2,1).transpose(0,1)[None],values[2:].repeat_interleave(2,1).transpose(0,1)[None],
        attn_mask=torch.ones(1,1,1,16,dtype=torch.bool),is_causal=False,scale=0.25)[0].transpose(0,1)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)

def test_reference_key_write_preserves_other_slots_across_block_boundary():
    cache=torch.full((3,2,2,16,8),-1,dtype=torch.bfloat16)
    keys=torch.arange(4*2*16).reshape(4,2,16).bfloat16()
    slots=torch.tensor([15,16,17,-1])
    overwrite_reference_keys(cache,keys,slots)
    decoded=cache.permute(0,3,1,2,4).reshape(48,2,16)
    for i,slot in enumerate([15,16,17]):assert torch.equal(decoded[slot],keys[i])
    untouched=torch.ones(48,dtype=torch.bool);untouched[torch.tensor([15,16,17])]=False
    assert torch.all(decoded[untouched]==-1)
    with pytest.raises(ValueError,match='shape'):overwrite_reference_keys(cache,keys[:1],slots)

def test_reference_rope_forces_cpu_initialization_and_disables_autocast():
    class Rotary:
        base=10000.0
        def _compute_inv_freq(self,base):
            return 1/(base**(torch.arange(0,8,2,dtype=torch.float32)/8))
    with torch.device('meta'):
        inv,factor=cpu_inverse_frequency(Rotary())
    assert inv.device.type=='cpu' and inv.dtype==torch.float32 and factor==1
    torch.manual_seed(12)
    q=torch.randn(7,4,8).bfloat16();k=torch.randn(7,2,8).bfloat16();pos=torch.arange(7)+43
    expected=reference_position_rope(q,k,pos,inv,factor)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        actual=reference_position_rope(q,k,pos,inv,factor)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)

def test_prefill_window_threshold_uses_explicit_mask_and_expanded_kv(monkeypatch):
    impl=types.SimpleNamespace(sliding_window=4,scale=0.5)
    install_sdpa_prefill(impl)
    q=torch.randn(4,4,4);k=torch.randn(4,2,4);v=torch.randn_like(k)
    meta=types.SimpleNamespace(cu_seqlens_q=torch.tensor([0,4]),cu_seqlens_k=torch.tensor([0,4]))
    original=torch.nn.functional.scaled_dot_product_attention
    calls=[]
    def observe(q,k,v,**kwargs):
        calls.append((q.shape,k.shape,kwargs))
        return original(q,k,v,**kwargs)
    monkeypatch.setattr(torch.nn.functional,'scaled_dot_product_attention',observe)
    impl.prefill_attention(q,k,v,None,None,None,None,types.SimpleNamespace(attn_metadata=meta))
    _,key_shape,options=calls[0]
    assert key_shape==(1,4,4,4)
    assert not options['enable_gqa'] and not options['is_causal']
    assert torch.equal(options['attn_mask'][0,0],torch.ones(4,4,dtype=torch.bool).tril())

def test_stepwise_rope_preserves_bf16_operation_boundaries():
    torch.manual_seed(6)
    q=torch.randn(7,4,8).bfloat16();k=torch.randn(7,2,8).bfloat16()
    cos=torch.randn(9,1,1,4).bfloat16();sin=torch.randn_like(cos)
    positions=torch.arange(1,8)
    rotary=types.SimpleNamespace(cos_cache=cos,sin_cache=sin)
    got_q,got_k=stepwise_rope(q,k,positions,rotary)
    c=cos[positions].reshape(7,1,4).repeat(1,1,2)
    s=sin[positions].reshape(7,1,4).repeat(1,1,2)
    for x,got in [(q,got_q),(k,got_k)]:
        expected=x*c+torch.cat((-x[...,4:],x[...,:4]),dim=-1)*s
        torch.testing.assert_close(got,expected,rtol=0,atol=0)

def test_single_prefill_matches_expanded_head_reference_and_rejects_cache():
    impl=types.SimpleNamespace(sliding_window=8,scale=0.5)
    install_sdpa_prefill(impl)
    torch.manual_seed(1)
    q=torch.randn(5,4,4);k=torch.randn(5,2,4);v=torch.randn_like(k)
    meta=types.SimpleNamespace(cu_seqlens_q=torch.tensor([0,5]),cu_seqlens_k=torch.tensor([0,5]))
    ctx=types.SimpleNamespace(attn_metadata=meta)
    got=impl.prefill_attention(q,k,v,None,None,None,None,ctx)
    expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],
        k.repeat_interleave(2,1).transpose(0,1)[None],v.repeat_interleave(2,1).transpose(0,1)[None],
        is_causal=True,scale=0.5)[0].transpose(0,1)
    torch.testing.assert_close(got,expected)
    meta.cu_seqlens_k=torch.tensor([0,6])
    with pytest.raises(ValueError,match='cached/chunked'):impl.prefill_attention(q,k,v,None,None,None,None,ctx)
    meta.cu_seqlens_k=torch.tensor([0,5]);impl.sliding_window=4
    got=impl.prefill_attention(q,k,v,None,None,None,None,ctx)
    mask=torch.tensor([[j<=i and j>i-4 for j in range(5)] for i in range(5)])
    expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],
        k.repeat_interleave(2,1).transpose(0,1)[None],v.repeat_interleave(2,1).transpose(0,1)[None],
        attn_mask=mask[None,None],is_causal=False,scale=0.5)[0].transpose(0,1)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)


def test_long_prefill_sliding_mask_excludes_old_and_future_values():
    impl=types.SimpleNamespace(sliding_window=4,scale=0.5)
    install_sdpa_prefill(impl)
    n=11
    q=torch.zeros(n,2,4);k=torch.zeros(n,1,4)
    v=torch.arange(n,dtype=torch.float32)[:,None,None].expand(n,1,4).contiguous()
    meta=types.SimpleNamespace(cu_seqlens_q=torch.tensor([0,n]),cu_seqlens_k=torch.tensor([0,n]))
    got=impl.prefill_attention(q,k,v,None,None,None,None,types.SimpleNamespace(attn_metadata=meta))
    # Equal scores must average exactly the allowed causal window per query.
    expected=torch.tensor([sum(range(max(0,i-3),i+1))/min(i+1,4) for i in range(n)])
    torch.testing.assert_close(got[:,0,0],expected)

@pytest.mark.parametrize('n,window,opt,expected_calls',[(7,8,True,0),(8,8,True,1),(9,8,True,1),(9,-1,True,0),(9,8,False,0)])
def test_swa_opt_dispatch_preserves_reference_paths(n,window,opt,expected_calls,monkeypatch):
    import atom.model_config.mellum_attention as m
    calls=[]
    def local(q,k,v,cu,win,scale):
        calls.append((q.shape,k.shape,win));return torch.zeros_like(q)
    monkeypatch.setattr(m,'aiter_swa_prefill',local)
    impl=types.SimpleNamespace(sliding_window=window,scale=0.5,_mellum_swa_prefill_opt=opt)
    install_sdpa_prefill(impl)
    q=torch.randn(n,4,4);k=torch.randn(n,2,4);v=torch.randn_like(k)
    ctx=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(cu_seqlens_q=torch.tensor([0,n]),cu_seqlens_k=torch.tensor([0,n])))
    got=impl.prefill_attention(q,k,v,None,None,None,None,ctx)
    assert len(calls)==expected_calls and got.shape==q.shape
    if calls:assert calls[0]==(q.shape,k.shape,window)


def test_aiter_local_window_includes_exactly_window_keys_without_head_expansion(monkeypatch):
    import sys
    from atom.model_config.mellum_attention import aiter_swa_prefill
    calls=[]
    def flash(**kw):
        calls.append(kw)
        # Independent scalar mean oracle for zero Q/K; inclusive AITER bounds.
        left=kw['window_size'][0];v=kw['v'].repeat_interleave(2,dim=1)
        return torch.stack([v[max(0,i-left):i+1].mean(0) for i in range(v.shape[0])])
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(flash_attn_varlen_func=flash))
    n=9;q=torch.zeros(n,4,2);k=torch.zeros(n,2,2);v=torch.arange(n).float()[:,None,None].expand(n,2,2)
    got=aiter_swa_prefill(q,k,v,torch.tensor([0,n],dtype=torch.int32),4,0.5)
    assert calls[0]['k'].shape[1]==2 and calls[0]['window_size']==(3,0,0)
    assert calls[0]['causal'] is True
    torch.testing.assert_close(got[:,0,0],torch.tensor([0,.5,1,1.5,2.5,3.5,4.5,5.5,6.5]))


def test_swa_candidate_does_not_hide_kernel_failure(monkeypatch):
    import sys
    from atom.model_config.mellum_attention import aiter_swa_prefill
    def fail(**kwargs):raise RuntimeError('local kernel unavailable')
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(flash_attn_varlen_func=fail))
    with pytest.raises(RuntimeError,match='unavailable'):
        aiter_swa_prefill(torch.zeros(8,4,4),torch.zeros(8,2,4),torch.zeros(8,2,4),torch.tensor([0,8]),4,.5)
