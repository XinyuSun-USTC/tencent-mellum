import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnose_mellum_swa_layer import oracle,metrics

@pytest.mark.parametrize('window',[1,3,7])
def test_blocked_fp32_oracle_matches_independent_per_head_float64(window):
    torch.manual_seed(18)
    q=torch.randn(7,4,8);k=torch.randn(7,2,8);v=torch.randn_like(k)
    actual=oracle(q,k,v,window,.25,block=2)
    expected=torch.empty_like(q,dtype=torch.float64)
    for i in range(7):
        for h in range(4):
            first=max(0,i-window+1)
            keys=k[first:i+1,h//2].double();values=v[first:i+1,h//2].double()
            probs=((keys@q[i,h].double())*.25).softmax(0)
            expected[i,h]=probs@values
    torch.testing.assert_close(actual.double(),expected,atol=3e-7,rtol=2e-6)
    assert metrics(actual,actual)['exact']
