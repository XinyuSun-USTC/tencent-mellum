"""Run an owned diagnostic subprocess with bounded wait and group cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import time


def run_stage(stage,command,log_path,timeout=300,heartbeat=15,allowed_exit_codes=(0,),with_code=False):
    log_path=Path(log_path)
    print('STAGE',stage,'log',log_path,flush=True)
    start=time.monotonic();last=start
    with log_path.open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                now=time.monotonic()
                with log_path.open('rb') as stream:
                    stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536))
                    tail=stream.read().decode(errors='replace')
                if 'proc died unexpectedly' in tail:
                    raise RuntimeError(f'{stage}: worker died; inspect {log_path}')
                if now-start>timeout:
                    raise TimeoutError(f'{stage}: exceeded {timeout}s; inspect {log_path}')
                if now-last>=heartbeat:
                    print(f'STAGE {stage}: {int(now-start)}s elapsed, log {size} bytes',flush=True);last=now
                time.sleep(0.25)
            if child.returncode not in allowed_exit_codes:
                raise RuntimeError(f'{stage}: exited {child.returncode}; inspect {log_path}')
            content=log_path.read_text()
            return (child.returncode,content) if with_code else content
        finally:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait(timeout=3)
