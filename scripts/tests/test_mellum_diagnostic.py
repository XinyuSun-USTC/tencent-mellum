import importlib.util
from pathlib import Path
import torch

scripts=Path(__file__).parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,scripts/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_same_prefix_from_real_failure():
    m=load('diagnose_mellum_mismatch')
    data=m.prepare(Path('/workspace/results/mellum_validation/20261111T113407765642Z'))
    assert [r['name'] for r in data['records']]==['python_add_prefix_0','python_add_prefix_37','arithmetic_prefix_0','arithmetic_prefix_6']
    assert [len(r['input_ids'][0]) for r in data['records']]==[24,61,18,24]
    assert all(r['generation_settings']['max_new_tokens']==1 for r in data['records'])
    assert all(len(r['forced_prefix_ids']) in [0,6,37] for r in data['records'])

def test_layer_comparison(tmp_path):
    m=load('compare_mellum_layers')
    left=tmp_path/'hf';right=tmp_path/'atom'
    for root in [left,right]:(root/'sample_000000').mkdir(parents=True)
    x=torch.ones(3,4)
    torch.save(x,left/'sample_000000/embedding.pt');torch.save(x,right/'sample_000000/embedding.pt')
    torch.save(x,left/'sample_000000/layer_00.pt');torch.save(x+0.5,right/'sample_000000/layer_00.pt')
    result=m.compare_layers(left,right)
    assert result['records'][0]['status']=='exact'
    assert result['records'][1]['rms']==0.5

def test_select_expanded_short_probes():
    import pytest
    m=load('diagnose_mellum_mismatch')
    run=Path('/workspace/results/mellum_validation/20261111T132520303449Z')
    data=m.prepare(run,['h_0','h_1'])
    assert [r['name'] for r in data['records']]==['h_0_prefix_0','h_0_prefix_14','h_1_prefix_0','h_1_prefix_9']
    assert [len(r['input_ids'][0]) for r in data['records']]==[152,166,142,151]
    with pytest.raises(ValueError,match='Unknown cases'):m.prepare(run,['missing'])
