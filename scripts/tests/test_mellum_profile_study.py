import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
import torch

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
from run_mellum_profile_study import build_plans,percentile,summarize,real_cases
from profile_mellum import prepare_inputs,generate_request
from mellum_runtime_config import PROJECT
from analyze_mellum_profile import analyze_events


def test_expanded_matrix_has_explicit_budgets_and_one_token_tpot_is_undefined():
    synthetic=prepare_inputs([512,1024,4096,8192,16384])
    real=[dict(name=f'real_{i}',input_ids=[1]*32,input_tokens=32) for i in range(4)]
    plans=build_plans(synthetic,real)
    assert {k:len(v['jobs']) for k,v in plans.items()}==dict(timing=19,trace=9,detailed=1)
    assert sum(j['repeats'] for j in plans['timing']['jobs'])==132
    assert sum(j['repeats']*j['output_tokens'] for j in plans['timing']['jobs'])==22617
    b=[j for j in plans['timing']['jobs'] if j['case']['input_tokens']==1024 and j['output_tokens']==256]
    assert b[0]['repeats']==50
    row=dict(output_tokens=1,wall_seconds=1,request_ttft_seconds=1,request_tpot_seconds=0,
             request_latency_seconds=1,output_tokens_per_wall_second=1)
    stats=summarize([row]*5)
    assert stats['request_tpot_seconds']['median'] is None
    assert stats['wall_seconds']['p99'] is None
    assert summarize([row]*50)['wall_seconds']['p99']==1
    assert percentile([1,2,3,4,5],.5)==3


def test_generate_request_rejects_truncated_output_and_keeps_sampling_protocol():
    recorded=[]
    def sample(**kwargs):recorded.append(kwargs);return kwargs
    row=dict(token_ids=[2,3],ttft=.1,tpot=.2,latency=.3)
    engine=types.SimpleNamespace(generate=lambda *args:[row])
    result=generate_request(engine,dict(input_ids=[1]),2,sample)
    assert result['output_tokens']==2 and result['tpot_defined']
    assert recorded==[dict(temperature=0,max_tokens=2,ignore_eos=True)]
    with pytest.raises(RuntimeError):generate_request(engine,dict(input_ids=[1]),3,sample)


def test_self_cpu_duration_and_semantic_operator_not_runtime_overwrite():
    def e(name,ts,dur,cat,ext):
        return dict(name=name,ts=ts,dur=dur,cat=cat,ph='X',pid=1,tid=1,args={'External id':ext})
    events=[e('prefill[bs=1]',0,100,'user_annotation',1),
            e('mellum/attention_swa/layer_03',1,90,'user_annotation',2),
            e('mellum/sdpa/shared',5,50,'user_annotation',3),
            e('aten::outer',6,40,'cpu_op',4),e('aten::inner',10,10,'cpu_op',5),
            e('hipLaunchKernel',11,1,'cuda_runtime',5),
            e('kernel',200,20,'kernel',5)]
    result=analyze_events(events)
    operators={r['name']:r for r in result['top_cpu_operators']}
    assert operators['aten::outer']['self_cpu_event_us']==30
    assert operators['aten::inner']['self_cpu_event_us']==10
    assert result['top_launch_operators'][0]['name']=='aten::inner'
    assert result['phase_layer_module_kernel_us']['prefill']['layer_03']['sdpa']==20
    assert result['phase_instances']['prefill[bs=1]']['kernel_count']==1


def test_granular_function_wrappers_preserve_values_and_restore(monkeypatch):
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    labels=load('labels',PROJECT/'ATOM/atom/model_config/mellum_profile.py')
    attention=load('attention',PROJECT/'ATOM/atom/model_config/mellum_attention.py')
    atom=types.ModuleType('atom');atom.__path__=[]
    package=types.ModuleType('atom.model_config');package.__path__=[];package.mellum_attention=attention;atom.model_config=package
    monkeypatch.setitem(sys.modules,'atom',atom);monkeypatch.setitem(sys.modules,'atom.model_config',package)
    monkeypatch.setitem(sys.modules,'atom.model_config.mellum_attention',attention)
    monkeypatch.setenv('MELLUM_PROFILE_GRANULAR','1')
    original=torch.nn.functional.scaled_dot_product_attention
    q=torch.randn(1,2,4,8)
    expected=original(q,q,q,is_causal=True)
    handles=labels.install_profile_labels(types.SimpleNamespace(model=types.SimpleNamespace(layers=[])))
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            got=torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)
        assert torch.equal(got,expected)
        assert 'mellum/sdpa/shared' in {e.key for e in prof.key_averages()}
    finally:
        for handle in handles:handle.remove()
    assert torch.nn.functional.scaled_dot_product_attention is original


def test_telemetry_errors_are_recorded_without_changing_clocks(monkeypatch,tmp_path):
    import mellum_profile_telemetry as telemetry
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=1,stdout='not-json',stderr='no device')
    monkeypatch.setattr(telemetry.subprocess,'run',fake)
    sampler=telemetry.TelemetrySampler(tmp_path/'telemetry.jsonl')
    sampler.sample_once()
    row=json.loads((tmp_path/'telemetry.jsonl').read_text())
    assert row['returncode']==1 and row['stdout']=='not-json'
    assert not any('set' in arg for arg in calls[0])


def test_real_prompt_selection_reports_exclusions_without_truncation(tmp_path):
    rows=[dict(id='short',prompt='a'*100),dict(id='long',prompt='b'*200),dict(id='excluded',prompt='c'*17000)]
    for name in ['scenario_a_prompts.jsonl','scenario_c_prompts.jsonl']:
        (tmp_path/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    tokenizer=types.SimpleNamespace(apply_chat_template=lambda messages,**kwargs:messages[0]['content'],
                                    encode=lambda text,**kwargs:list(range(len(text))))
    cases,sources=real_cases(tokenizer,tmp_path)
    assert len(cases)==4 and {case['input_tokens'] for case in cases}=={100,200}
    assert all(row['excluded_for_length']==1 for row in sources)
    assert all(case['source_id']!='excluded' for case in cases)


def test_resume_validates_completed_results_and_archives_failed_attempt(tmp_path):
    from run_mellum_profile_study import completed_stage,archive_attempt
    from mellum_runtime_config import CANDIDATE
    folder=tmp_path/'timing';folder.mkdir()
    plan=dict(mode='timing',jobs=[dict(case=dict(name='tiny',input_tokens=1),output_tokens=2,repeats=1)])
    (folder/'plan.json').write_text(json.dumps(plan))
    result=dict(status='timing_collected',mode='timing',settings=CANDIDATE,
                jobs=[dict(name='tiny',input_tokens=1,output_tokens=2,rows=[dict(token_ids=[2,3])])])
    (folder/'worker_results.json').write_text(json.dumps(result))
    assert completed_stage(folder)
    result['jobs'][0]['rows'][0]['token_ids']=[2]
    (folder/'worker_results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):completed_stage(folder)
    result['status']='failed';previous=json.dumps(result)
    (folder/'worker_results.json').write_text(previous)
    (tmp_path/'timing.log').write_text('failure log')
    archive_attempt(tmp_path,'timing')
    old=next((tmp_path/'previous_attempts').iterdir())
    assert (old/'worker_results.json').read_text()==previous
    assert (old/'parent_stage.log').read_text()=='failure log'
    assert list(folder.iterdir())==[folder/'plan.json']


def test_resume_rejects_changed_plan_before_any_gpu_action(tmp_path):
    import hashlib
    from run_mellum_profile_study import resume_study,fingerprints
    from mellum_runtime_config import MODEL
    folder=tmp_path/'timing';folder.mkdir()
    plan=dict(model_config_sha256=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest())
    path=folder/'plan.json';path.write_text(json.dumps(plan))
    summary=dict(source_sha256=fingerprints(),plan_sha256=dict(timing=hashlib.sha256(path.read_bytes()).hexdigest()),case_counts=dict(timing=1))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    args=types.SimpleNamespace(resume=tmp_path,detailed=True,prepare_only=True)
    assert resume_study(args)==0
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='plan changed'):resume_study(args)
