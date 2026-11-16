import json,sys,asyncio
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import followup_screened_mellum as f
from diagnose_stream_cleanup import reproduce


def fixture(tmp_path,kind='kv'):
    source=tmp_path/'project';model=source/'ATOM/atom/models/mellum.py';model.parent.mkdir(parents=True);model.write_text('measured source')
    screen=tmp_path/'screen';screen.mkdir();artifact=screen/'measured.txt';artifact.write_text('real measured artifact')
    profile='r_reference_kv_once' if kind=='kv' else 'r_expert_decode2'
    candidate=screen/'candidate.json';candidate.write_text(json.dumps({'profile':profile,'source_sha256':{str(model):f.digest(model)}}))
    plan={'profile':profile,'candidate_plan':str(candidate),'candidate_plan_sha256':f.digest(candidate)} if kind=='kv' else {'plans':{'2':{'path':str(candidate),'sha256':f.digest(candidate)}}}
    result={'status':'screen_passed_full_acceptance_pending','source_unchanged':True,'generation':{'developer200_gate':True},
        'candidates':{'2':{'developer200_gate':True},'4':{'developer200_gate':False}},'paired_perf':{'R1':{}},'artifact_sha256':{'measured.txt':f.digest(artifact)}}
    (screen/'manifest.json').write_text(json.dumps(plan));(screen/'summary.json').write_text(json.dumps(result))
    return source,screen,profile


def test_followup_requires_actual_pass_and_unchanged_evidence(tmp_path):
    source,screen,profile=fixture(tmp_path)
    candidate,result=f.read_screen(screen,profile,source)
    assert candidate['profile']==profile
    (screen/'measured.txt').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):f.read_screen(screen,profile,source)


def test_requested_expert_group_must_have_passed(tmp_path):
    source,screen,profile=fixture(tmp_path,'expert')
    f.read_screen(screen,profile,source)
    with pytest.raises(ValueError,match='did not pass'):f.read_screen(screen,'r_expert_decode4',source)


def test_running_screen_cannot_trigger_full_followup(tmp_path):
    source,screen,profile=fixture(tmp_path)
    p=screen/'summary.json';d=json.loads(p.read_text());d['status']='started';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='completed passing'):f.read_screen(screen,profile,source)


def test_cpu_cleanup_diagnostic_normal_completion_is_a_positive_control():
    result=asyncio.run(reproduce())
    assert result['cases']['normal']['cleanup_calls']==1
    assert set(result['cases'])=={'normal','close_after_role','cancel_while_waiting'}
    assert result['source_sha256']


def test_full_capacity_requires_same_accepted_source_and_scheduler_size(tmp_path):
    root=tmp_path/'source';directory=tmp_path/'accepted';directory.mkdir();output=directory/'output.json';output.write_text('measured')
    profile='r_expert_decode2';manifest=directory/'manifest.json';summary=directory/'summary.json'
    manifest.write_text(json.dumps({'profile':profile,'source_root':str(root),'batch_size':32}))
    summary.write_text(json.dumps({'status':'local_candidate_checks_passed','artifact_sha256':{'output.json':f.digest(output)}}))
    assert f.read_accepted_run(directory,profile,root)['batch_size']==32
    with pytest.raises(ValueError,match='source worktree'):f.read_accepted_run(directory,profile,tmp_path/'different_source')
    data=json.loads(manifest.read_text());data['batch_size']=1;manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='capacity differs'):f.read_accepted_run(directory,profile,root)


def test_combination_uses_its_flat_manifest_and_requires_both_feature_receipts(tmp_path):
    source,screen,_=fixture(tmp_path)
    profile='r_expert_decode4_kv_once'
    candidate=json.loads((screen/'candidate.json').read_text());candidate['profile']=profile
    (screen/'manifest.json').write_text(json.dumps(candidate))
    assert f.read_screen(screen,profile,source)[0]['profile']==profile
    target=screen/'candidate';target.mkdir()
    events=[{'kind':'kv_write','phase':phase,'backend':'aiter_reference_kv_once','rope_backend':'torch','resident_inverse':False} for phase in ['prefill','decode']]
    (target/'native_calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
    groups=[{'phase':'decode','requests':32,'decode_group_size':4,'expert_groups':8}]
    (target/'expert_groups.jsonl').write_text(json.dumps(groups[0])+'\n')
    assert f.verify_feature(screen,profile)==3
    events[0]['backend']='different'
    (target/'native_calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
    with pytest.raises(ValueError,match='KV feature receipt'):f.verify_feature(screen,profile)
