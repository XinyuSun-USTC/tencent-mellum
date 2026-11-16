import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_full_campaign as c
from run_mellum_batch_campaign import write


def test_profile_forwarding_failure_continuation_and_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'current',lambda *a:None);monkeypatch.setattr(c,'audit_backend',lambda *a:None)
    profile='sprint_graph_qk';(tmp_path/profile).mkdir()
    plan={'gpu':0,'profiles':[profile],'scope':'test','template':{'stages':{s:[{'name':s}] for s in ['A','B','C']}},'jobs':{}}
    seen=[]
    def execute(name,command,log,**kw):
        directory=Path(log).parent;job=json.loads((directory/'plan.json').read_text());seen.append(job)
        assert job['profile']==profile and job['server_port']==8451
        if job['stage']=='B':raise RuntimeError('owned stage failed')
        write(directory/'results.json',{'status':'complete','cases':{'case':{'metrics':{'complete_without_failures':True}}}})
    first=c.execute(tmp_path,plan,600,execute)
    assert first['status']=='complete_with_failures' and [s['stage'] for s in seen]==['A','B','C']
    seen.clear()
    second=c.execute(tmp_path,plan,600,execute)
    assert [s['stage'] for s in seen]==['B'] and (tmp_path/profile/'B/attempt_1').is_dir()
    (tmp_path/profile/'A/results.json').write_text('tampered')
    import pytest
    with pytest.raises(ValueError,match='artifact changed'):c.execute(tmp_path,plan,600,execute)
