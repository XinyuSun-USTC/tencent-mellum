import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_optimization_sprint as study
from mellum_execution_profiles import profile_settings
import serve_mellum


def test_sprint_isolation():
    for name in ['sprint_clean','sprint_graph','sprint_unpacked','sprint_router']:
        s=profile_settings(name)
        assert s['MELLUM_BATCH_CAMPAIGN_CONFIG']==''
        assert s['MELLUM_NORM_BACKEND']=='aiter' and s['MELLUM_RESIDENT_ROPE']=='1'
        assert s['MELLUM_DECODE_GRAPH']==str(int(name=='sprint_graph'))
        assert s['MELLUM_NATIVE_ROUTING']==str(int(name=='sprint_router'))
        assert s['MELLUM_QKV_BACKEND']==('separate' if name=='sprint_unpacked' else 'packed')
    assert json.loads(profile_settings('fast_combined')['MELLUM_BATCH_CAMPAIGN_CONFIG'])['groups']==''


def test_launcher_graph_only_when_explicit(tmp_path,monkeypatch):
    for name in ['sprint_clean','sprint_graph']:
        out=tmp_path/name
        monkeypatch.setattr(sys,'argv',['serve','--profile',name,'--batching','--prepare-only','--output',str(out)])
        serve_mellum.main();d=json.loads((out/'launch.json').read_text());cmd=d['command']
        assert ('--enforce-eager' not in cmd)==(name=='sprint_graph')
        if name=='sprint_graph':
            assert json.loads(cmd[cmd.index('--cudagraph-capture-sizes')+1])==[1,2,4,8,16,32]


def test_failure_continues_to_independent_candidate(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'verify_backend',lambda *args:None)
    seen=[]
    def execute(out,*args):
        seen.append(out.name)
        if out.name=='sprint_unpacked':raise RuntimeError('kernel failed')
        return {'status':'diagnostic_complete'}
    monkeypatch.setattr(study.capacity,'execute',execute)
    jobs={n:{} for n in study.TRACKS['kernels']}
    d=study.execute(tmp_path,{'track':'kernels','scope':'test','source_sha256':{}},jobs,600)
    assert seen==list(jobs) and d['status']=='complete_with_failures'
