"""Bounded GPU1 orchestration; no GPU needed."""
import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_prefill_optimization as campaign
from run_mellum_batch_campaign import write


def result(plan,score=1.):
    return {'status':'complete','stable':True,'batch_verified':True,'cohorts':{
        c:{'groups':{'all':{'token_match_fraction':score}},'wall_seconds':[1.],
           'repeat_files':[c+'_r0.json']} for c in plan['cohorts']}}


def run(tmp_path,monkeypatch,stop=None):
    calls=[]
    def fake(name,cmd,log,timeout):
        plan=json.loads((log.parent/'plan.json').read_text());calls.append((name,plan))
        log.write_text('CPU mock')
        value=result(plan,.8 if name==stop else 1.)
        write(log.parent/'results.json',value)
    monkeypatch.setattr(campaign,'exact_to_cached',lambda *a:{'records':32,'exact_sequences':32})
    monkeypatch.setattr(campaign,'audit_backend',lambda *a:True)
    monkeypatch.setattr(campaign,'analyze_timing',lambda *a:{'performance_eligible':True,'cohorts':{
        c:{'cpd_seconds':[2.]*4,'cpd_median_seconds':2.,'speedup':2.} for c in ('S32','L32')}})
    manifest={'gpu':1,'jobs':{},'max_workers':4,'provenance':{'cached_shared_routing':{c:{} for c in ('S32','L32','D200')}}}
    args=NS(budget_minutes=60,job_timeout=900)
    data={c:[] for c in ('S32','L32','K4','D200')}
    summary=campaign.execute(tmp_path,manifest,data,args,fake)
    return summary,calls,manifest,args,data,fake


def test_complete_batch_uses_gpu1_dedicated_port_R_control_and_bounded_ABBA(tmp_path,monkeypatch):
    summary,calls,m,args,data,fake=run(tmp_path,monkeypatch)
    assert summary['status']=='candidate_measured'
    assert [c[0] for c in calls]==['screen_R','screen_candidate','D200_candidate','timing']
    assert all(p['gpu']==1 and p['engine_port']==8126 for _,p in calls)
    assert calls[0][1]['control']['moe_optimization']=='shared_routing'
    modes=calls[-1][1]['cohort_variants']
    assert list(modes.values()).count('shared_routing')==8
    assert list(modes.values()).count(campaign.VARIANT)==8
    count=len(calls)
    campaign.execute(tmp_path,m,data,args,fake)
    assert len(calls)==count


@pytest.mark.parametrize('stage,status,count',[
    ('screen_candidate','screen_stopped',2),('D200_candidate','developer_position_below_90',3)])
def test_accuracy_failure_stops_dependent_jobs(tmp_path,monkeypatch,stage,status,count):
    summary,calls,*_=run(tmp_path,monkeypatch,stage)
    assert summary['status']==status and len(calls)==count


def test_actual_backend_audit_requires_candidate_prefill_and_unchanged_decode(tmp_path):
    rows=[{'phase':'prefill','requests':2,'variant':campaign.VARIANT,'backend':'requestwise_gemm_shared_elementwise'},
          {'phase':'decode','requests':32,'variant':campaign.VARIANT,'backend':'requestwise_gemm_shared_route'}]
    p=tmp_path/'moe_calls.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
    assert campaign.audit_backend(tmp_path)
    rows[1]['backend']='indexed_triton';p.write_text('\n'.join(map(json.dumps,rows)))
    assert not campaign.audit_backend(tmp_path)
