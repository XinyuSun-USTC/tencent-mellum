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


def test_exited_parent_does_not_leave_owned_child_running(tmp_path):
    import os,time
    pidfile=tmp_path/'child.pid'
    code="import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
    run_stage('orphan',[sys.executable,'-c',code,str(pidfile)],tmp_path/'orphan.log',timeout=5)
    pid=int(pidfile.read_text());deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':return
        time.sleep(.01)
    pytest.fail('Owned orphan still running')
