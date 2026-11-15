"""Exactly one reduction candidate, staged promotion and resume."""
import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_isolated_moe as campaign
from run_mellum_batch_campaign import write


@pytest.mark.parametrize('stable,accuracy,status,jobs',[(False,.99,'fixed_replay_stopped',1),
    (True,.8,'developer_position_below_90',2),(True,.96,'candidate_measured',4)])
def test_local_then_generation_then_performance_gates(tmp_path,monkeypatch,stable,accuracy,status,jobs):
    calls=[]
    def fake(name,cmd,log,timeout):
        p=json.loads((log.parent/'plan.json').read_text());calls.append((name,p));log.write_text('CPU mock')
        if name=='fixed_replay':v={'status':'complete','all_fixed_repeats_exact':stable,'records':[{'finite':True}]*3}
        else:
            v={'status':'complete','batch_verified':True,'cohorts':{c:{'wall_seconds':[2. if name=='performance_R' else 1.],
                'groups':{'all':{'token_match_fraction':accuracy}}} for c in p['cohorts']}}
            if name=='D200_candidate':write(log.parent/'D200_r0.json',{'records':[{}]*200,'completions':[{}]*200})
        write(log.parent/'results.json',v)
    monkeypatch.setattr(campaign,'verify_backend',lambda *a:{'logical_topk':8,'native_topk':1})
    m={'provenance':{},'micro_inputs':[],'jobs':{},'max_workers':4}
    data={c:[{'input_ids':[1],'budget':64}] for c in ('S32','L32','D200')}
    args=NS(budget_minutes=45,job_timeout=900)
    value=campaign.execute(tmp_path,m,data,args,fake)
    assert value['status']==status and len(calls)==jobs
    assert calls[0][1]['variant']=='isolated'
    if jobs>1:
        assert calls[1][1]['runtime_overrides']=={'MELLUM_MOE_BACKEND':'fused_isolated'}
        assert calls[1][1]['control']['groups']=='L' and 'moe_optimization' not in calls[1][1]['control']
    campaign.execute(tmp_path,m,data,args,fake);assert len(calls)==jobs


def test_no_HF_distance_gate_but_fixed_replay_requires_finite_outputs():
    assert campaign.replay_ok({'all_fixed_repeats_exact':True,'records':[{'finite':True,'vs_reference':{'relative_l2':.01}}]*3})
    assert not campaign.replay_ok({'all_fixed_repeats_exact':True,'records':[{'finite':False}]*3})


def test_backend_evidence_requires_exact_route_expansion_and_no_shared_native_reduction(tmp_path):
    row={'kind':'moe','backend':'isolated_fused_experts','native_topk':1,'logical_topk':8,
         'rows':32,'expanded_rows':256,'quant_method':'UnquantizedFusedMoEMethod',
         'shuffled_w13':True,'shuffled_w2':True}
    rows=[{**row,'phase':'prefill','requests':2},{**row,'phase':'decode','requests':32}]
    p=tmp_path/'native_calls.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
    assert campaign.verify_backend(tmp_path)['native_topk']==1
    rows[1]['native_topk']=8;p.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='combination'):campaign.verify_backend(tmp_path)
