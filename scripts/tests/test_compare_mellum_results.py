import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch

spec=importlib.util.spec_from_file_location('mellum_compare',Path(__file__).parents[1]/'compare_mellum_results.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SOURCE=Path('/workspace/results/atom_step0_batch_ab/hf_reference')

@pytest.fixture
def candidate(tmp_path):
    shutil.copytree(SOURCE,tmp_path/'candidate')
    return tmp_path/'candidate/reference.json'

def change(path,fn):
    data=json.loads(path.read_text());fn(data);path.write_text(json.dumps(data))

def test_same_reference(candidate):
    assert m.compare(SOURCE/'reference.json',candidate)['status']=='passed'

@pytest.mark.parametrize('field,value',[
    ('rendered_prompt','wrong'),('input_ids',[[1,2,3]]),('attention_mask',[[0]]),
    ('position_ids',[[7]]),('generation_settings',{'do_sample':True}),
    ('generated_ids',[[1]]),('generated_ids',[]),
])
def test_input_and_token_differences(candidate,field,value):
    change(candidate,lambda d:d['records'][0].update({field:value}))
    result=m.compare(SOURCE/'reference.json',candidate)
    assert result['status']=='failed'
    assert field in result['records'][1]['mismatches'] or any(field in r['mismatches'] for r in result['records'])

@pytest.mark.parametrize('kind',['logits','nan','hash','missing_record','duplicate','config'])
def test_artifact_failures(candidate,kind):
    data=json.loads(candidate.read_text())
    if kind in ['logits','nan','hash']:
        record=data['records'][0];p=candidate.parent/record['prefill_logits']
        t=torch.load(p,map_location='cpu',weights_only=True).clone()
        t[0]=float('nan') if kind=='nan' else t[0]+1
        torch.save(t,p)
        if kind!='hash':record['prefill_logits_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    elif kind=='missing_record':data['records'].pop()
    elif kind=='duplicate':data['records'].append(copy.deepcopy(data['records'][0]))
    else:data['model_config_sha256']='wrong'
    candidate.write_text(json.dumps(data))
    if kind in ['duplicate','hash']:
        with pytest.raises(ValueError):m.compare(SOURCE/'reference.json',candidate)
    else:assert m.compare(SOURCE/'reference.json',candidate)['status']=='failed'


def test_token_stats_counts_lengths_and_common_prefix():
    stats=m.token_stats([[1,2,3,4]],[[1,9,3]])
    assert stats=={'compared_positions':4,'matching_positions':2,'common_prefix_tokens':1,'token_match_fraction':0.5}
    assert m.token_stats([[1]],[])['compared_positions']==1
