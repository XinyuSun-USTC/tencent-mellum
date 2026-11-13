"""Coarse external SMI samples; never modifies power or clock settings."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import threading
import time


class TelemetrySampler:
    def __init__(self,path,interval=15):
        self.path=Path(path);self.interval=interval
        self.stop=threading.Event();self.thread=None

    def sample_once(self):
        command=['/opt/rocm/bin/rocm-smi','--showuse','--showmemuse','--showclocks','--showtemp','--showpower','--json']
        start=time.monotonic()
        row=dict(utc=datetime.now(timezone.utc).isoformat(),time_ns=time.time_ns(),command=command,
                 scope='All SMI-reported devices; mapping to HIP-visible device is not assumed')
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=5)
            row.update(returncode=result.returncode,stderr=result.stderr)
            try:row['data']=json.loads(result.stdout)
            except json.JSONDecodeError:row['stdout']=result.stdout
        except Exception as error:row['error']=repr(error)
        row['sample_seconds']=time.monotonic()-start
        with self.path.open('a') as stream:stream.write(json.dumps(row)+'\n')

    def _run(self):
        while not self.stop.is_set():
            self.sample_once()
            self.stop.wait(self.interval)

    def start_sampling(self):
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=6)
