import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_completion_benchmarks as b
from serve_mellum_step0_snapshot import describe
import pytest


def test_primary_requires_real_pass_before_any_benchmark(tmp_path,monkeypatch):
    (tmp_path/'summary.json').write_text(json.dumps({'status':'local_candidate_checks_not_all_passed'}))
    monkeypatch.setattr(sys,'argv',['bench','--track','primary','--acceptance-run',str(tmp_path)])
    with pytest.raises(ValueError,match='has not passed'):b.main()


def test_single_slot_revalidation_preserves_real_failures(tmp_path,monkeypatch):
    (tmp_path/'plan.json').write_text(json.dumps({'row':'primary','gpu':0,'server_port':1234,'engine_port':1235,
        'batch_size':1,'cases':[{'name':'A','scenario':'A','number':512,'expected':[]}]}))
    def stage(directory):
        (directory/'results.json').write_text(json.dumps({'status':'complete','cases':{'A':{'metrics':{'actual_decode_max_batch':1}}}}))
    monkeypatch.setattr(b.base,'stage_worker',stage)
    monkeypatch.setattr(b.base,'inspect_result',lambda *a:{'complete_without_failures':False,'validation_errors':['Real output count differs']})
    b.worker(tmp_path)
    value=json.loads((tmp_path/'results.json').read_text())['cases']['A']['metrics']
    assert value['actual_decode_max_batch']==1 and value['engine_scheduler_slots']==1
    assert not value['complete_without_failures'] and value['validation_errors']==['Real output count differs']


def test_historical_snapshot_is_exact_tag_and_not_current_reference(tmp_path):
    d=describe(Path('/workspace/tencent-mellum-step0-repro'),0,'127.0.0.1',8751,9726,tmp_path)
    assert d['snapshot_tag']=='step0-local-validated' and d['settings']['MELLUM_REQUEST_BATCHING']=='0'
    with pytest.raises(ValueError):describe(b.PROJECT,0,'127.0.0.1',8751,9726,tmp_path)
