import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('router_replay',Path(__file__).parents[1]/'check_mellum_router.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_hf_route_preserves_bf16_weights_and_normalization():
    logits=torch.arange(64,dtype=torch.bfloat16).reshape(1,64)/8
    weights,ids=m.hf_route(logits)
    assert weights.dtype==torch.bfloat16
    assert ids.tolist()==[list(range(63,55,-1))]
    assert abs(weights.float().sum().item()-1)<0.005

def test_real_replay_boundary_ties_exist():
    import json
    root=Path('/workspace/results/mellum_validation/20261111T120512707283Z')
    atom=Path(json.loads((root/'summary.json').read_text())['atom_run'])/'stages'
    for name,expected in [('sample_000000',9),('sample_000001',6)]:
        logits=torch.load(atom/name/'router_logits.pt',weights_only=True).float()
        top=logits.softmax(-1).topk(9).values
        assert int((top[:,7]==top[:,8]).sum())==expected
