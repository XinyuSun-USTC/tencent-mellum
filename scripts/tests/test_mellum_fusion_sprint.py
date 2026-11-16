import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import profile_settings
import run_mellum_fusion_sprint as study


def test_profiles_keep_graph_and_isolate_changes():
    names=['sprint_graph','sprint_graph_router','sprint_graph_rope','sprint_graph_qk']
    for n in names:
        s=profile_settings(n)
        assert s['MELLUM_DECODE_GRAPH']=='1' and s['MELLUM_BATCH_CAMPAIGN_CONFIG']==''
        assert s['MELLUM_NATIVE_ROUTING']==str(int(n.endswith('_router')))
        assert s['MELLUM_FUSED_ROPE']==str(int(n.endswith('_rope')))
        assert s['MELLUM_FUSED_QK_ROPE']==str(int(n.endswith('_qk')))
    assert study.TRACKS['routing']==['sprint_graph','sprint_graph_router','sprint_graph_router_repeat','sprint_graph_repeat']


def test_failure_does_not_cancel_repeats(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'verify_backend',lambda *a:None);calls=[]
    def fake(out,m,*a):
        calls.append((out.name,m['profile']))
        return {'status':'stage_failed' if out.name=='sprint_graph_router' else 'diagnostic_complete'}
    monkeypatch.setattr(study.capacity,'execute',fake)
    jobs={n:{'profile':n.removesuffix('_repeat')} for n in study.TRACKS['routing']}
    r=study.execute(tmp_path,{'track':'routing','scope':'test','source_sha256':{}},jobs,600)
    assert len(calls)==4 and calls[-1][1]=='sprint_graph' and r['status']=='complete_with_failures'
