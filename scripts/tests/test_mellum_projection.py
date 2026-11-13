import copy
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

SCRIPTS=Path(__file__).parents[1]
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('projection_study',SCRIPTS/'diagnose_mellum_projection_math.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_saved_inputs_preflight_is_complete_and_cpu_only():
    torch.set_num_threads(4)
    before=torch.cuda.is_initialized()
    manifest,weights,inputs=m.prepare(m.DEFAULT_RUN)
    assert len(manifest['tensor_files_sha256'])==37
    assert len(weights)==5 and len(inputs)==4
    assert [len(t['input_norm']) for _,_,t in inputs]==[152,166,142,151]
    assert all(v.device.type=='cpu' for v in weights.values())
    assert torch.cuda.is_initialized()==before


def test_bad_tensor_and_output_are_rejected():
    with pytest.raises(ValueError,match='expected BF16'):m.check_tensor(torch.ones(2,3),(2,3),'x')
    with pytest.raises(ValueError,match='expected BF16'):m.check_tensor(torch.ones(2,3).bfloat16(),(3,2),'x')
    with pytest.raises(ValueError,match='nonfinite'):m.check_tensor(torch.full((2,3),float('nan')).bfloat16(),(2,3),'x')
    with pytest.raises(ValueError,match='shape/dtype'):m.compare(torch.zeros(3).bfloat16(),torch.zeros(2).bfloat16())


def test_decision_gate_does_not_blame_kernel_without_reference_and_stability():
    row={'comparisons':{'torch_separate':{'q':{'exact':True}},
                       'o_proj':{'torch':{'exact':True}},'router':{'torch':{'exact':True}}},
         'repeat_stability':{'torch':True,'aiter':True},
         'packing_effect':{'q':{'exact':False}},'kernel_effect':{'q':{'exact':False}}}
    assert m.summarize([row])['interpretation_gate']=='inspect_projection_differences'
    changed=copy.deepcopy(row);changed['comparisons']['torch_separate']['q']['exact']=False
    assert m.summarize([changed])['interpretation_gate']=='resolve_reference_or_repeat_instability_first'
    changed=copy.deepcopy(row);changed['repeat_stability']['aiter']=False
    assert m.summarize([changed])['interpretation_gate']=='resolve_reference_or_repeat_instability_first'


def test_repeat_check_detects_reused_mutating_output_buffer():
    shared=torch.zeros(2,dtype=torch.bfloat16)
    def compute():
        shared.add_(1)
        return shared
    first,stable=m.repeat_output(compute,3)
    assert not stable
    assert torch.equal(first,torch.ones_like(shared))
    assert torch.equal(shared,torch.full_like(shared,3))
