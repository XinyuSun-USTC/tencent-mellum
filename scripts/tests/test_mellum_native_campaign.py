"""Two isolated native tracks; speed-first gating and explicit correctness budget."""
import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_native_campaign as campaign
from run_mellum_batch_campaign import write


def test_configuration_replaces_E_for_fused_and_reference_bridge_for_native_attention():
    control,env=campaign.configuration('moe',True)
    assert control['groups']=='L' and 'moe_optimization' not in control
    assert env=={'MELLUM_MOE_BACKEND':'fused_torch_route'}
    control,env=campaign.configuration('attention',True)
    assert control['moe_optimization']=='shared_routing'
    assert env['MELLUM_SDPA_DECODE']=='0' and env['MELLUM_REQUEST_BATCHING']=='0'
    assert campaign.configuration('moe',False)[1]=={}


def test_perf256_not_scored_against_64_reference_and_counts_original_inputs():
    rows=[{'name':str(i),'input_ids':[i,7],'budget':16,'expected_ids':[3]*16} for i in range(32)]
    data=campaign.performance_inputs({'S32':rows,'L32':rows})
    assert len(data['S1'])==1 and len(data['L32'])==32
    assert all(r['budget']==256 for cases in data.values() for r in cases)
    assert data['S32'][17]['input_ids']==rows[17]['input_ids'] and rows[17]['budget']==16


@pytest.mark.parametrize('slow,accuracy,status,jobs',[(True,1.,'no_material_speedup',2),
    (False,.8,'developer_position_below_90',3),(False,.95,'candidate_eligible_for_combination',3)])
def test_speed_gate_before_full200_and_position_only_accuracy_gate(tmp_path,monkeypatch,slow,accuracy,status,jobs):
    calls=[]
    def fake(name,cmd,log,timeout):
        plan=json.loads((log.parent/'plan.json').read_text());calls.append((name,plan));log.write_text('CPU mock')
        value={'status':'complete','batch_verified':True,'stable':True,'cohorts':{c:{'wall_seconds':[2.,2.] if name=='performance_R' or slow else [1.,1.],
                'groups':{'all':{'token_match_fraction':accuracy}}} for c in plan['cohorts']}}
        write(log.parent/'results.json',value)
        if name=='D200_candidate':write(log.parent/'D200_r0.json',{'records':[{}]*200,'completions':[{}]*200})
    monkeypatch.setattr(campaign,'inspect_native',lambda *a:{'decode_max_requests':32})
    rows=[{'input_ids':[1],'budget':64}]*32
    data={c:rows for c in ('S32','L32','D200')}
    m={'gpu':0,'track':'moe','jobs':{},'max_workers':3,'provenance':{}}
    args=NS(budget_minutes=60,job_timeout=1800)
    result=campaign.execute(tmp_path,m,data,args,fake)
    assert result['status']==status and len(calls)==jobs
    assert calls[0][1]['engine_port']==8226 and calls[0][1]['performance_only']
    if jobs==3:assert not calls[-1][1]['performance_only'] and calls[-1][1]['repeats']==1
    campaign.execute(tmp_path,m,data,args,fake)
    assert len(calls)==jobs  # hash-verified successful jobs reused


def test_native_backend_evidence_cannot_hide_SDPA_fallback(tmp_path):
    rows=[{'kind':'attention','phase':'prefill','requests':3,'backend':'aiter_varlen'},
          {'kind':'attention','phase':'decode','requests':32,'backend':'paged_attention_triton'}]
    p=tmp_path/'native_calls.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
    assert campaign.inspect_native(tmp_path,'attention')['decode_max_requests']==32
    rows[1]['backend']='ragged_decode';p.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='fallback'):campaign.inspect_native(tmp_path,'attention')


def test_256_output_records_have_no_invented_accuracy_score():
    from run_mellum_batch_campaign import performance_records
    rows,groups=performance_records([{'name':'x','input_ids':[1,2],'budget':256,'expected_ids':[3]*64}],
                                     [{'token_ids':[4]*256}])
    assert len(rows[0]['actual_ids'])==256 and groups=={'performance_only':{'requests':1,'output_tokens':256}}
    with pytest.raises(ValueError,match='budget/count'):
        performance_records([{'budget':256}],[])
    with pytest.raises(ValueError,match='budget/count'):
        performance_records([{'budget':256}],[{'token_ids':[4]*255}])


def test_fused_backend_audit_requires_loaded_shuffled_unquantized_weights(tmp_path):
    row={'kind':'moe','phase':'decode','requests':32,'backend':'FusedMoE',
         'quant_method':'UnquantizedFusedMoEMethod','shuffled_w13':True,'shuffled_w2':True}
    rows=[{**row,'phase':'prefill','requests':2},row]
    p=tmp_path/'native_calls.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
    assert campaign.inspect_native(tmp_path,'moe')['decode_max_requests']==32
    rows[1]['shuffled_w2']=False;p.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='quantization/backend'):campaign.inspect_native(tmp_path,'moe')
