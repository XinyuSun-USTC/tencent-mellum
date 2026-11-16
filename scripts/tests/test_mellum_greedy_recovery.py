import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import AGREEMENT,profile_settings
import run_mellum_greedy_recovery as recovery


def test_diagnostic_profiles_only_change_declared_flags():
    base=profile_settings('sprint_graph_qk_router_prefix_metadata')
    for name,flags in AGREEMENT.items():
        s=profile_settings(name)
        assert s=={**base,**flags} and s['MELLUM_DECODE_GRAPH']=='1'
    assert profile_settings('agree_qk_round')['MELLUM_NORM_BACKEND']=='aiter'
    assert profile_settings('agree_norms_route')['MELLUM_NATIVE_ROUTING']=='0'


def test_recovery_worker_never_runs_humaneval(tmp_path,monkeypatch):
    calls=[]
    (tmp_path/'manifest.json').write_text(json.dumps({'diagnostic_suites':['tokens','long'],'diagnostic_batch':1}))
    monkeypatch.setattr(recovery.acceptance,'worker',lambda *a,**kw:calls.append(kw))
    monkeypatch.setattr(sys,'argv',['recovery','--worker',str(tmp_path)])
    recovery.main()
    assert calls==[{'suites':['tokens','long'],'batch_size':1}]
