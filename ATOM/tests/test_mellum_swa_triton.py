import sys,types
import pytest
import torch
from atom.model_config.mellum_swa_triton import triton_swa_prefill

MODULE='aiter.ops.triton.attention.mha'


def test_adapter_maps_inclusive_window_and_gqa_without_expansion(monkeypatch):
    calls=[]
    def flash(**kw):
        calls.append(kw);q,k,v=kw['q'],kw['k'],kw['v']
        assert kw['window_size'][1]==-1 and kw['causal']
        left=kw['window_size'][0];values=v.repeat_interleave(q.shape[1]//k.shape[1],1)
        return torch.stack([values[max(0,i-left):i+1].mean(0) for i in range(len(q))])
    monkeypatch.setitem(sys.modules,MODULE,types.SimpleNamespace(_MHA_IMPL='default',flash_attn_varlen_func=flash))
    q=torch.zeros(9,4,2);k=torch.zeros(9,2,2);v=torch.zeros_like(k);v[0,0]=1;v[0,1]=2
    got=triton_swa_prefill(q,k,v,torch.tensor([0,9],dtype=torch.int32),4,.5)
    assert calls[0]['window_size']==(3,-1) and calls[0]['k'].shape[1]==2
    torch.testing.assert_close(got[0,:,0],torch.tensor([1.,1.,2.,2.]))
    torch.testing.assert_close(got[3,:,0],torch.tensor([.25,.25,.5,.5]))
    assert torch.equal(got[4],torch.zeros_like(got[4]))


def test_adapter_refuses_unsupported_global_implementation_and_singleton_window(monkeypatch):
    monkeypatch.setitem(sys.modules,MODULE,types.SimpleNamespace(_MHA_IMPL='dao_ai'))
    q=torch.zeros(8,4,4);k=torch.zeros(8,2,4);cu=torch.tensor([0,8])
    with pytest.raises(ValueError,match='dao_ai'):triton_swa_prefill(q,k,k,cu,4,.5)
    with pytest.raises(ValueError,match='window>=2'):triton_swa_prefill(q,k,k,cu,1,.5)
    with pytest.raises(ValueError,match='THD'):triton_swa_prefill(q,k[:7],k[:7],cu,4,.5)


def test_adapter_does_not_hide_compile_failure(monkeypatch):
    def fail(**kwargs):raise RuntimeError('Compilation failed')
    monkeypatch.setitem(sys.modules,MODULE,types.SimpleNamespace(_MHA_IMPL='default',flash_attn_varlen_func=fail))
    q=torch.zeros(8,4,4);k=torch.zeros(8,2,4)
    with pytest.raises(RuntimeError,match='Compilation'):triton_swa_prefill(q,k,k,torch.tensor([0,8]),4,.5)
