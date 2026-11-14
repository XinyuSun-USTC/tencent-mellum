import json
from pathlib import Path
import shutil
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_mellum_swa_hf_check import prepare

SOURCE=Path('/workspace/results/mellum_swa_optimization/20261114T121938893923Z')

class Tokenizer:
    def decode(self,ids,skip_special_tokens=False):return 'Decoded display only'


def test_prepare_preserves_abba_ids_without_forcing_or_reencoding():
    data=prepare(SOURCE,Tokenizer())
    cases=json.loads((SOURCE/'a0/plan.json').read_text())['cases']
    for row,case in zip(data['records'],cases):
        assert row['input_ids']==[case['input_ids']]
        assert row['generation_settings']=={'do_sample':False,'max_new_tokens':64,'eos_token_id':28,'pad_token_id':0}
        assert row['position_ids']==[list(range(case['input_tokens']))]
        assert not any('forced' in key or 'generated' in key for key in row)


def test_prepare_rejects_changed_source_plan(tmp_path):
    shutil.copy2(SOURCE/'summary.json',tmp_path/'summary.json')
    for stage in ['a0','b0','b1','a1']:
        (tmp_path/stage).mkdir();shutil.copy2(SOURCE/stage/'plan.json',tmp_path/stage/'plan.json')
    path=tmp_path/'b1/plan.json';data=json.loads(path.read_text());data['cases'][0]['input_ids'][0]+=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='plans differ'):prepare(tmp_path,Tokenizer())
