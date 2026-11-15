"""Bounded phase diagnosis and no full-model MoE auto-promotion."""
import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_native_diagnostics as campaign
from run_mellum_batch_campaign import write


@pytest.mark.parametrize('passed,jobs,status',[(False,2,'no_split_candidate_above_90'),(True,4,'split_candidates_measured')])
def test_only_passing_phase_gets_performance_and_resume_skips(tmp_path,monkeypatch,passed,jobs,status):
    calls=[]
    def fake(name,cmd,log,timeout):
        plan=json.loads((log.parent/'plan.json').read_text());calls.append((name,plan));log.write_text('CPU mock')
        score=.95 if passed and name=='native_decode_D200' else .8
        val={'status':'complete','batch_verified':True,'cohorts':{c:{'groups':{'all':{'token_match_fraction':score}},'wall_seconds':[1.]} for c in plan['cohorts']}}
        write(log.parent/'results.json',val)
        if 'D200' in plan['cohorts']:write(log.parent/'D200_r0.json',{'records':[{}]*200,'completions':[{}]*200})
    monkeypatch.setattr(campaign,'verify_phases',lambda *a:{'verified':True})
    m={'track':'attention','gpu':1,'provenance':{},'jobs':{},'max_workers':5}
    data={k:[{'input_ids':[1],'budget':64}] for k in ('S32','L32','D200')};args=NS(budget_minutes=45,job_timeout=900)
    result=campaign.execute(tmp_path,m,data,args,fake)
    assert result['status']==status and len(calls)==jobs
    assert calls[0][1]['runtime_overrides']['MELLUM_ATTENTION_EXECUTION']=='native_prefill'
    assert calls[1][1]['runtime_overrides']['MELLUM_ATTENTION_EXECUTION']=='native_decode'
    campaign.execute(tmp_path,m,data,args,fake);assert len(calls)==jobs


def test_moe_runs_only_two_fixed_replays_not_engine_or_D200(tmp_path):
    plans=[]
    def fake(name,cmd,log,timeout):
        p=json.loads((log.parent/'plan.json').read_text());plans.append(p);log.write_text('CPU mock')
        write(log.parent/'results.json',{'status':'complete','all_fixed_repeats_exact':p['sorting']=='ck'})
    m={'track':'moe','gpu':0,'provenance':{},'micro_inputs':[],'jobs':{},'max_workers':2}
    value=campaign.execute(tmp_path,m,{},NS(budget_minutes=15,job_timeout=600),fake)
    assert value['status']=='diagnostic_complete' and [p['sorting'] for p in plans]==['opus','ck']
    assert all(p['kind']=='fixed_moe' and 'cohorts' not in p for p in plans)


def test_phase_evidence_checks_reference_half_and_no_extra_native_half(tmp_path):
    rows=[{'kind':'attention','phase':'prefill','requests':3,'backend':'reference_ragged_prefill'},
          {'kind':'attention','phase':'decode','requests':32,'backend':'paged_attention_triton'}]
    p=tmp_path/'native_calls.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
    assert campaign.verify_phases(tmp_path,'native_decode')
    with pytest.raises(ValueError,match='not actually'):campaign.verify_phases(tmp_path,'native_prefill')


def test_metadata_callable_description_preserves_actual_kernel_selection():
    from functools import partial
    from mellum_fused_repeat import callable_info
    def kernel(*args,**kwargs):pass
    info=callable_info(partial(kernel,kernelName='test_kernel',splitk=2))
    assert info['keywords']=={'kernelName':'test_kernel','splitk':'2'} and 'kernel' in info['callable']
