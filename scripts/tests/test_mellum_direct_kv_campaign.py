import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_direct_kv as study
from run_mellum_batch_campaign import write


def test_layout_probe_failure_stops_before_http(tmp_path,monkeypatch):
    def fake(stage,cmd,log,timeout):write(log.parent/'results.json',{'status':'failed'})
    monkeypatch.setattr(study,'run_stage',fake)
    monkeypatch.setattr(study.capacity,'execute',lambda *a: (_ for _ in ()).throw(AssertionError('HTTP must not start')))
    s=study.execute(tmp_path,{'scope':'test','source_sha256':{}},{},1200)
    assert s['status']=='failed' and 'probe failed' in s['traceback']


def test_http_requires_actual_direct_write_in_both_phases(tmp_path,monkeypatch):
    def fake(stage,cmd,log,timeout):write(log.parent/'results.json',{'status':'passed'})
    def capacity(out,*args):
        for stage in ['A_small','C_small']:
            p=out/stage/'server/native_calls.jsonl';p.parent.mkdir(parents=True)
            p.write_text(''.join(json.dumps({'kind':'kv_write','phase':phase,'backend':'aiter_reference_kv_once'})+'\n' for phase in ['prefill','decode']))
        return {'status':'diagnostic_complete'}
    monkeypatch.setattr(study,'run_stage',fake);monkeypatch.setattr(study.capacity,'execute',capacity)
    s=study.execute(tmp_path,{'scope':'test','source_sha256':{}},{},1200)
    assert s['status']=='complete'
