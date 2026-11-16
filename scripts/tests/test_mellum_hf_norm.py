import sys
from pathlib import Path
import torch,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atom.model_config.mellum_hf_norm import HFCompatibleRMSNorm
from atom.model_config.mellum_fastpaths import make_norm
from mellum_execution_profiles import profile_settings
import run_mellum_greedy_recovery as recovery


def test_norm_preserves_checkpoint_keys_shape_and_cpu_guard():
    norm=HFCompatibleRMSNorm(2304).bfloat16()
    assert list(norm.state_dict())==['weight'] and norm.weight.shape==(2304,)
    with pytest.raises(ValueError,match='device'):norm(torch.ones(2,2304,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='residual'):norm(torch.ones(2,2304,dtype=torch.bfloat16),torch.ones(1,2304,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='TP1'):HFCompatibleRMSNorm(128,fused_allreduce=True)


def test_factory_and_profiles_are_opt_in(monkeypatch):
    monkeypatch.setenv('MELLUM_NORM_BACKEND','triton_hf')
    assert isinstance(make_norm(None,128),HFCompatibleRMSNorm)
    for control,fast in [('agree_norms','agree_fast_norms'),('agree_norms_route','agree_fast_norms_route')]:
        a,b=profile_settings(control),profile_settings(fast)
        assert b=={**a,'MELLUM_NORM_BACKEND':'triton_hf'}
        assert b['MELLUM_DECODE_GRAPH']==b['MELLUM_HF_QK_ROUNDING']=='1'


def test_no_new_gpu_experiment_without_positive_evidence(monkeypatch):
    monkeypatch.setattr(sys,'argv',['recovery','--track','hf_norm'])
    with pytest.raises(SystemExit):recovery.main()
