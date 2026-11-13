import importlib.util
from pathlib import Path
import sys
import json

SCRIPTS=Path(__file__).parents[1];sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('decode_runner',SCRIPTS/'run_mellum_decode_validation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_forced_baseline_uses_aligned_prefill_inputs():
    data=m.prepare_inputs(m.DEFAULT_PREFILL,8)
    assert len(data['records'])==4
    assert [len(r['input_ids'][0]) for r in data['records']]==[32,16,17,1023]
    assert {r['name'] for r in data['records']}=={'chinese_inventory','raw_16','raw_17','window_1023'}
    targeted=m.prepare_inputs(m.DEFAULT_PREFILL,2,True)
    assert [r['name'] for r in targeted['records']]==['window_1023']

def test_legacy_trace_recovery_does_not_change_history_or_saved_logits():
    source=Path('/workspace/results/mellum_decode_validation/20261112T123830320864Z/hf/reference.json')
    before=source.read_bytes();original=json.loads(before)
    fixed,changes=m.normalize_reference_traces(original)
    assert source.read_bytes()==before and len(changes)==4
    for old,new in zip(original['records'],fixed['records']):
        assert old['input_ids']==new['input_ids'] and old['generated_ids']==new['generated_ids']
        assert new['decode_steps'][0]['input_ids']==new['input_ids'][0]
        for a,b in zip(old['decode_steps'],new['decode_steps']):
            assert a['cached_logits']==b['cached_logits'] and a['cached_sha256']==b['cached_sha256']
            assert a['uncached_logits']==b['uncached_logits'] and a['uncached_sha256']==b['uncached_sha256']


def test_normal_failure_case_can_be_selected_only_with_exact_prefill():
    source=Path('/workspace/results/mellum_validation/20261112T173200224689Z')
    data=m.prepare_inputs(source,2,True,'python_add')
    assert [r['name'] for r in data['records']]==['python_add']
    assert len(data['records'][0]['input_ids'][0])==24
