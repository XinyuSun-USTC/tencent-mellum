import importlib.util
import json
from pathlib import Path
import sys
import copy
import pytest

SCRIPTS=Path(__file__).parents[1]
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('prefill_study',SCRIPTS/'run_mellum_prefill_study.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_fixed_prefill_inputs_keep_history_and_remove_stale_outputs():
    manifest,data=m.prepare_inputs(m.DEFAULT_RUN)
    original=json.loads((m.DEFAULT_RUN/'hf/reference.json').read_text())
    assert len(data['records'])==len(manifest['sample_names'])==4
    for got,ref in zip(data['records'],original['records']):
        for key in ['name','input_ids','attention_mask','position_ids','generation_settings']:
            assert got[key]==ref[key]
        assert got['generation_settings']['max_new_tokens']==1
        assert 'generated_ids' not in got and 'prefill_logits' not in got

def test_reference_reuse_checks_fixed_protocol_and_backend():
    previous=Path('/workspace/results/mellum_prefill_study/20261112T095954484085Z')
    _,inputs=m.prepare_inputs(m.DEFAULT_RUN)
    sha=m.digest(previous/'inputs.json')
    hf,metadata=m.reuse_reference(previous,inputs,sha)
    assert hf==previous/'hf'
    assert metadata['environment']['experts_implementation']=='grouped_mm'
    assert metadata['tensor_files_sha256']
    with pytest.raises(ValueError,match='different inputs'):m.reuse_reference(previous,inputs,'wrong')
    altered=copy.deepcopy(inputs);altered['records'][0]['position_ids'][0][0]=99
    with pytest.raises(ValueError,match='protocol differs'):m.reuse_reference(previous,altered,sha)
