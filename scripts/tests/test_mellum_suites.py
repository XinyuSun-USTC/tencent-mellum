import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('suite_builder',Path(__file__).parents[1]/'prepare_mellum_suite.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tokenizer:
    # Deterministic stand-in to check lengths/composition without HF import.
    def encode(self,text,add_special_tokens=False):return [ord(c)%95 for c in text]
    def decode(self,tokens,skip_special_tokens=False):return str(tokens)
    def apply_chat_template(self,messages,**kwargs):return '<user>'+messages[0]['content']+'<assistant>'

@pytest.mark.parametrize('suite,lengths',[('window',[1023,1024,1025]),('long',[8191,8192,8193,16384])])
def test_fixed_lengths(suite,lengths):
    data=m.build(Tokenizer(),suite)
    assert [len(r['input_ids'][0]) for r in data['records']]==lengths
    for r in data['records']:
        n=len(r['input_ids'][0])
        assert r['attention_mask']==[[1]*n]
        assert r['position_ids']==[list(range(n))]
    assert data==m.build(Tokenizer(),suite)

def test_short_is64_and_source_unmodified():
    path=Path('/workspace/results/atom_step0_batch_ab/hf_reference/inputs.json')
    before=path.read_bytes();data=m.build(Tokenizer(),'short')
    assert len(data['records'])==2
    assert all(r['generation_settings']['max_new_tokens']==64 for r in data['records'])
    assert path.read_bytes()==before


def test_expanded_has_real_code_sources_and_long64():
    data=m.build(Tokenizer(),'expanded')
    assert len(data['records'])==11
    assert len({r['name'] for r in data['records']})==11
    assert all(r['generation_settings']['max_new_tokens']==64 for r in data['records'])
    assert [len(r['input_ids'][0]) for r in data['records'][-3:]]==[1025,8192,16384]
    assert sum(r['prompt_source'].startswith('HumanEval/') for r in data['records'])==4
    assert 'not_official' in data['scope']

def test_prefill_suite_is_one_token_and_within_supported_window():
    data=m.build(Tokenizer(),'prefill')
    assert len(data['records'])==10
    assert len({r['name'] for r in data['records']})==10
    assert [len(r['input_ids'][0]) for r in data['records'][-4:]]==[16,17,1023,1024]
    assert sum(r['prompt_source']=='new_developer_prompt' for r in data['records'])==6
    for r in data['records']:
        n=len(r['input_ids'][0])
        assert 0<n<=1024 and r['generation_settings']['max_new_tokens']==1
        assert r['attention_mask']==[[1]*n] and r['position_ids']==[list(range(n))]
        assert isinstance(r['enable_thinking'],bool)
        assert r['enable_thinking']==(r['prompt_source']!='raw_code_boundary')
    assert data==m.build(Tokenizer(),'prefill')

def test_missing_template_metadata_is_rejected_before_inference():
    data=m.build(Tokenizer(),'prefill');m.validate_inputs(data)
    del data['records'][0]['enable_thinking']
    with pytest.raises(ValueError,match='enable_thinking'):m.validate_inputs(data)
