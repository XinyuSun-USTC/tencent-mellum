import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import profile_settings
import screen_mellum_reference_kv as s
from run_mellum_batch_campaign import write


def test_profile_changes_only_cache_write():
    before=profile_settings('r');after=profile_settings(s.CANDIDATE)
    assert {k:v for k,v in after.items() if k!='MELLUM_DIRECT_KV_WRITE'}=={k:v for k,v in before.items() if k!='MELLUM_DIRECT_KV_WRITE'}
    assert before['MELLUM_DIRECT_KV_WRITE']=='0' and after['MELLUM_DIRECT_KV_WRITE']=='1'
    assert after['MELLUM_MOE_BACKEND']=='torch' and after['MELLUM_ATTENTION_EXECUTION']=='reference'


def fake_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'current',lambda *a:None)
    directory=tmp_path/'candidate_screen';directory.mkdir()
    return {'gpu':1,'scope':'test','with_perf':False}


def test_cache_failure_stops_before_generation(tmp_path,monkeypatch):
    plan=fake_plan(tmp_path,monkeypatch);seen=[]
    def execute(name,args,log,timeout):
        seen.append(name);write(log.parent/'results.json',{'status':'failed','cases':[]})
    result=s.execute(tmp_path,plan,120,execute)
    assert seen==['cache_ab'] and result['status']=='failed'
    assert (tmp_path/'summary.json').exists() and result['source_unchanged']


def test_generation_gate_does_not_start_perf_or_invent_he(tmp_path,monkeypatch):
    plan=fake_plan(tmp_path,monkeypatch);plan['with_perf']=True;seen=[]
    def execute(name,args,log,timeout):
        seen.append(name)
        if name=='cache_ab':write(log.parent/'results.json',{'status':'passed','cases':[]})
    monkeypatch.setattr(s,'evaluate_candidate',lambda *a:{'developer200_gate':False,'long':{'all':{'position_match_fraction':.5}}})
    result=s.execute(tmp_path,plan,120,execute)
    assert seen==['cache_ab','generation'] and result['status']=='developer200_below_90'
    assert 'paired_perf' not in result and 'humaneval' not in result


def test_screen_pass_remains_distinct_from_final_acceptance(tmp_path,monkeypatch):
    plan=fake_plan(tmp_path,monkeypatch)
    def execute(name,args,log,timeout):
        if name=='cache_ab':write(log.parent/'results.json',{'status':'passed','cases':[]})
    monkeypatch.setattr(s,'evaluate_candidate',lambda *a:{'developer200_gate':True})
    result=s.execute(tmp_path,plan,120,execute)
    assert result['status']=='screen_passed_full_acceptance_pending'
    assert json.loads((tmp_path/'summary.json').read_text())['status']==result['status']


def test_single_token_v_is_not_misclassified_as_a_cache_failure():
    class V:
        def __init__(self,n,contiguous):self.shape=(n,4,128);self.contiguous=contiguous
        def is_contiguous(self):return self.contiguous
    assert s.packed_v_layout_covered(V(1,True))
    assert s.packed_v_layout_covered(V(6,False))
    assert not s.packed_v_layout_covered(V(6,True))
