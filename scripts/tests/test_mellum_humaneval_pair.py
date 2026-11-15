"""Functional pair independent of token equality, full task accounting."""
import json,sys,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_mellum_humaneval_pair import paired_scores,validate_records,extract_all,generators


def rows(passed):return [{'task_id':f'HumanEval/{i}','passed':i in passed} for i in range(164)]


def test_two_percentage_points_means_three_tasks_allowed_not_four():
    assert paired_scores(rows(set(range(100))),rows(set(range(97))))['bf16_2pp_local_check']
    assert not paired_scores(rows(set(range(100))),rows(set(range(96))))['bf16_2pp_local_check']
    answer=paired_scores(rows({1,2}),rows({2,3}))
    assert answer['absolute_difference_pp']==0 and answer['candidate_gains']==answer['candidate_losses']==1
    assert paired_scores(rows(set()),rows(set()))['bf16_2pp_local_check'] is None


def test_missing_or_duplicate_task_scores_never_count_as_full164():
    with pytest.raises(ValueError):paired_scores(rows({1})[:-1],rows({1}))
    duplicate=rows({1});duplicate[-1]=duplicate[0]
    with pytest.raises(ValueError):paired_scores(duplicate,rows({1}))


def inputs():
    row={'source_id':'HumanEval/0','input_ids':[[1,2]],'enable_thinking':False,
         'generation_settings':{'max_new_tokens':2048}}
    return {'records':[row]}


def test_different_generated_tokens_are_accepted_but_protocol_drift_is_not():
    i=inputs();data={'records':[{**i['records'][0],'generated_ids':[[999,888]]}]}
    validate_records(i,data)  # No reference-generated-token equality field exists.
    data['records'][0]['enable_thinking']=True
    with pytest.raises(ValueError):validate_records(i,data)


def test_code_extraction_preserves_full_task_and_truncation_counts(tmp_path):
    class Tokenizer:
        def decode(self,ids,**kwargs):return '```python\ndef f():\n    return 42\n```'
    i=inputs();data={'records':[{**i['records'][0],'generated_ids':[[9]*2048]}]}
    task={'task_id':'HumanEval/0','entry_point':'f','prompt':'def f():\n    """Return a value."""\n'}
    value=extract_all(i,data,[task],Tokenizer(),tmp_path,'candidate')
    assert value['failures']==0 and value['budget_hits']==1
    sample=json.loads((tmp_path/'candidate_samples.jsonl').read_text())
    assert sample['task_id']=='HumanEval/0' and 'return 42' in sample['completion']


def test_parallel_producers_keep_both_exit_statuses_without_token_gate(tmp_path):
    commands={'hf':[sys.executable,'-c','print("reference complete")'],
              'candidate':[sys.executable,'-c','raise SystemExit(2)']}
    value=generators(commands,tmp_path,time.monotonic()+10)
    assert value['hf']['completed'] and not value['candidate']['completed']
    assert value['candidate']['returncode']==2


def test_HF_export_normalizes_successful_loading_report_sets(monkeypatch,tmp_path):
    import types,torch
    import run_mellum_humaneval_pair as pair
    real_tensor=torch.tensor
    class Model:
        config=types.SimpleNamespace(_experts_implementation='grouped_mm')
        def to(self,*args):return self
        def eval(self):return self
        def generate(self,input_ids,**kwargs):return torch.cat([input_ids,real_tensor([[28]])],dim=1)
    class Loader:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            return Model(),{'missing_keys':set(),'unexpected_keys':set(),'mismatched_keys':set(),
                            'error_msgs':[],'metadata_keys':{'a','b'}}
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(__version__='5.10.1',MellumForCausalLM=Loader))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *a:(100*1024**3,192*1024**3))
    monkeypatch.setattr(torch,'tensor',lambda *args,**kwargs:real_tensor(*args,**{k:v for k,v in kwargs.items() if k!='device'}))
    data=inputs();data['records'][0].update(attention_mask=[[1,1]],generation_settings={'max_new_tokens':2048})
    pair.hf_worker(tmp_path,data)
    output=json.loads((tmp_path/'generation.json').read_text())
    assert output['status']=='complete' and output['loading_info']['missing_keys']==[]
    assert output['loading_info']['metadata_keys']==['a','b']
