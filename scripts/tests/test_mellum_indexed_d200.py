"""Follow-up uses the stated developer position gate, not CPD-drop screening."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_indexed_d200 as study
from run_mellum_batch_campaign import write


def groups(position=.91,sequence=.5):
    return {name:{'records':200,'matching_positions':int(position*12800),'compared_positions':12800,
                  'token_match_fraction':position,'exact_sequences':int(sequence*200),'exact_sequence_fraction':sequence}
            for name in ('all','below_1024','at_least_1024')}


def test_position_pass_keeps_sequence_and_length_warnings_separate():
    value=groups();value['at_least_1024']['token_match_fraction']=.88
    result=study.decision(value)
    assert result['developer_position_90_percent_passed']
    assert not result['developer_sequence_90_percent_passed']
    assert result['long_position_fraction']==.88
    assert not study.decision(groups(.899))['developer_position_90_percent_passed']
    assert study.decision(groups(.9))['developer_position_90_percent_passed']


def test_exactly_one_worker_and_no_downstream_performance(tmp_path):
    cases=[{'name':str(i),'budget':64} for i in range(200)];calls=[]
    def fake(stage,command,log,timeout):
        calls.append(stage);log.write_text('ok');out=log.parent
        write(out/'results.json',{'status':'complete','batch_verified':True,'loading_audit':{},'environment':{},'wall_seconds':123})
        write(out/'D200_r0.json',{'records':[{'name':c['name'],'actual_ids':[1]*64} for c in cases],
                               'completions':list(range(200)),'groups':groups(),'actual_batches':{'decode':32}})
        (out/'moe_calls.jsonl').write_text(json.dumps({'phase':'decode','variant':'indexed_decode','backend':'indexed_triton','requests':32})+'\n')
    manifest={'gpu':0,'max_workers':2,'jobs':{},'provenance':{'baselines':{}}}
    result=study.execute(tmp_path,manifest,cases,NS(budget_minutes=20,job_timeout=900),fake)
    assert calls==['indexed_D200'] and result['status']=='developer_position_passed'
    assert result['indexed_decode_calls_verified']==1


def test_execution_failure_is_not_scored_as_accuracy_failure(tmp_path):
    def fake(stage,command,log,timeout):log.write_text('No visible GPU');raise RuntimeError('No GPU')
    manifest={'gpu':0,'max_workers':2,'jobs':{},'provenance':{'baselines':{}}}
    result=study.execute(tmp_path,manifest,[],NS(budget_minutes=20,job_timeout=900),fake)
    assert result['status']=='blocked_environment' and 'groups' not in result


def test_completed_result_reuse_does_not_start_second_gpu(tmp_path):
    calls=[];cases=[{'name':str(i),'budget':64} for i in range(200)]
    def fake(stage,command,log,timeout):
        calls.append(stage);log.write_text('ok');out=log.parent
        write(out/'results.json',{'status':'complete','batch_verified':True,'loading_audit':{},'environment':{},'wall_seconds':123})
        write(out/'D200_r0.json',{'records':[{'name':c['name'],'actual_ids':[1]*64} for c in cases],
                               'completions':list(range(200)),'groups':groups(.8),'actual_batches':{'decode':32}})
        (out/'moe_calls.jsonl').write_text(json.dumps({'phase':'decode','variant':'indexed_decode','backend':'indexed_triton','requests':32})+'\n')
    manifest={'gpu':0,'max_workers':2,'jobs':{},'provenance':{'baselines':{}}}
    args=NS(budget_minutes=20,job_timeout=900)
    assert study.execute(tmp_path,manifest,cases,args,fake)['status']=='developer_position_below_90'
    assert study.execute(tmp_path,manifest,cases,args,fake)['status']=='developer_position_below_90'
    assert len(calls)==1
