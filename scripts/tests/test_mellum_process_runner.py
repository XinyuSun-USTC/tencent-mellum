from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
from mellum_process_runner import run_stage

def test_stage_stops_when_worker_death_is_logged(tmp_path):
    code="import time; print('proc died unexpectedly (exitcode=1)',flush=True); time.sleep(30)"
    with pytest.raises(RuntimeError,match='worker died'):
        run_stage('test',[sys.executable,'-c',code],tmp_path/'fatal.log',timeout=5)

def test_stage_has_bounded_wait(tmp_path):
    with pytest.raises(TimeoutError,match='exceeded'):
        run_stage('test',[sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'timeout.log',timeout=0.2)

def test_stage_success_returns_log(tmp_path):
    assert 'done' in run_stage('test',[sys.executable,'-c',"print('done')"],tmp_path/'ok.log',timeout=5)
