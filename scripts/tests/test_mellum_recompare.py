import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).parents[1]
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('recompare',SCRIPTS/'recompare_mellum_prefill.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_metadata_recovery_preserves_originals_and_numeric_fields(tmp_path):
    source=tmp_path/'old.json';target=tmp_path/'new.json'
    rows=[{'name':origin,'prompt_source':origin,'input_ids':[[7,8]],'generated_ids':[[9]],
           'prefill_logits':'values.pt','prefill_logits_sha256':'kept'} for origin in ['raw_code_boundary','new_developer_prompt']]
    source.write_text(json.dumps({'records':rows}));before=source.read_bytes()
    provenance=m.repaired_copy(source,target);data=json.loads(target.read_text())
    assert source.read_bytes()==before and len(provenance['changes'])==2
    assert [r['enable_thinking'] for r in data['records']]==[False,True]
    for old,new in zip(rows,data['records']):
        assert new['input_ids']==old['input_ids'] and new['generated_ids']==old['generated_ids']
        assert new['prefill_logits_sha256']==old['prefill_logits_sha256']
        assert Path(new['prefill_logits'])==tmp_path/'values.pt'

def test_metadata_recovery_refuses_unknown_origin(tmp_path):
    source=tmp_path/'old.json'
    source.write_text(json.dumps({'records':[{'name':'unknown','prompt_source':'unknown'}]}))
    with pytest.raises(ValueError,match='Cannot infer'):m.repaired_copy(source,tmp_path/'new.json')
