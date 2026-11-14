import types
import pytest
import torch
from atom.model_config.mellum_batch import ragged_prefill,ragged_decode
from atom.model_config.mellum_attention import sdpa_single_prefill,overwrite_reference_keys,sdpa_cached_decode

@pytest.mark.parametrize('window',[-1,4])
def test_ragged_prefill_isolated_requests_and_reordered_inputs(window):
    torch.manual_seed(54)
    impl=types.SimpleNamespace(sliding_window=window,scale=.5)
    q=torch.randn(12,4,4);k=torch.randn(12,2,4);v=torch.randn_like(k)
    bounds=[0,5,12];cu=torch.tensor(bounds)
    ctx=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(cu_seqlens_q=cu,cu_seqlens_k=cu),
        context=types.SimpleNamespace(positions=torch.tensor(list(range(5))+list(range(7)))))
    got=ragged_prefill(impl,q,k,v,None,None,None,None,ctx)
    expected=[]
    for a,b in zip(bounds,bounds[1:]):
        sub=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(cu_seqlens_q=torch.tensor([0,b-a]),cu_seqlens_k=torch.tensor([0,b-a])))
        expected.append(sdpa_single_prefill(impl,q[a:b],k[a:b],v[a:b],None,None,None,None,sub))
    torch.testing.assert_close(got,torch.cat(expected),rtol=0,atol=0)
    changed=v.clone();changed[5:]=1000
    second=ragged_prefill(impl,q,k,changed,None,None,None,None,ctx)
    torch.testing.assert_close(second[:5],got[:5],rtol=0,atol=0)
    ctx.context.positions[-1]=100
    with pytest.raises(ValueError,match='chunked'):ragged_prefill(impl,q,k,v,None,None,None,None,ctx)
    ctx.attn_metadata.cu_seqlens_k=torch.tensor([0,5,13])
    with pytest.raises(ValueError,match='uncached'):ragged_prefill(impl,q,k,v,None,None,None,None,ctx)

@pytest.mark.parametrize('window',[-1,8])
@pytest.mark.parametrize('layout',['NHD','SHUFFLE'])
def test_ragged_decode_physical_pages_window_and_row_reorder(window,layout):
    torch.manual_seed(55)
    kc=torch.zeros(6,2,2,16,8,dtype=torch.bfloat16);vc=torch.zeros(6,2,16,16,dtype=torch.bfloat16)
    tables=torch.tensor([[4,1],[3,0]]);lengths=torch.tensor([18,9]);last_k=[];last_v=[]
    for i,n in enumerate(lengths.tolist()):
        keys=torch.randn(n,2,16).bfloat16();values=torch.randn_like(keys);logical=torch.arange(n)
        slots=tables[i,logical//16]*16+logical%16;overwrite_reference_keys(kc,keys,slots)
        if layout=='NHD':vc[slots//16,:,:,slots%16]=values
        else:vc.view(6,2,2,16,8)[slots//16,:,(slots%16)//8,:,slots%8]=values
        last_k.append(keys[-1:]);last_v.append(values[-1:])
    q=torch.randn(2,4,16).bfloat16();k=torch.cat(last_k);v=torch.cat(last_v)
    impl=types.SimpleNamespace(sliding_window=window,scale=.25,_cache_format=layout)
    ctx=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(context_lens=lengths,block_tables=tables),
        context=types.SimpleNamespace(positions=lengths-1))
    got=ragged_decode(impl,q,k,v,kc,vc,None,None,ctx);expected=[]
    for i in range(2):
        sub=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(context_lens=lengths[i:i+1],block_tables=tables[i:i+1]),
            context=types.SimpleNamespace(positions=lengths[i:i+1]-1))
        expected.append(sdpa_cached_decode(impl,q[i:i+1],k[i:i+1],v[i:i+1],kc,vc,None,None,sub))
    torch.testing.assert_close(got,torch.cat(expected),rtol=0,atol=0)
    ctx.attn_metadata.context_lens=lengths.flip(0);ctx.attn_metadata.block_tables=tables.flip(0);ctx.context.positions=lengths.flip(0)-1
    reverse=ragged_decode(impl,q.flip(0),k.flip(0),v.flip(0),kc,vc,None,None,ctx)
    torch.testing.assert_close(reverse,got.flip(0),rtol=0,atol=0)


def test_reference_rope_raw_slices_use_request_positions_without_mutating_impl():
    torch.manual_seed(56)
    q=torch.randn(9,4,4).bfloat16();k=torch.randn(9,2,4).bfloat16();v=torch.randn_like(k)
    positions=torch.tensor([0,1,2,3,0,1,2,3,4]);raw=(q.clone(),k.clone(),positions)
    impl=types.SimpleNamespace(sliding_window=4,scale=.5,_mellum_stepwise_rope=True,
        _mellum_cpu_rope=(torch.tensor([1.,.01]),1.),_mellum_raw_qk=raw)
    cu=torch.tensor([0,4,9]);ctx=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(cu_seqlens_q=cu,cu_seqlens_k=cu),context=types.SimpleNamespace(positions=positions))
    actual=ragged_prefill(impl,q,k,v,None,None,None,None,ctx)
    expected=[]
    for a,b in [(0,4),(4,9)]:
        local=types.SimpleNamespace(sliding_window=4,scale=.5,_mellum_stepwise_rope=True,
            _mellum_cpu_rope=impl._mellum_cpu_rope,_mellum_raw_qk=(raw[0][a:b],raw[1][a:b],positions[a:b]))
        sub=types.SimpleNamespace(attn_metadata=types.SimpleNamespace(cu_seqlens_q=torch.tensor([0,b-a]),cu_seqlens_k=torch.tensor([0,b-a])))
        expected.append(sdpa_single_prefill(local,q[a:b],k[a:b],v[a:b],None,None,None,None,sub))
    torch.testing.assert_close(actual,torch.cat(expected),rtol=0,atol=0)
    assert impl._mellum_raw_qk is raw
