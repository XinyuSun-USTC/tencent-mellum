"""Owned launcher for an unmodified historical Step0 worktree; no optimizations."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from mellum_runtime_config import CANDIDATE,MODEL


def describe(snapshot,gpu,host,http_port,engine_port,out):
    git=lambda *args:subprocess.check_output(['git','-C',str(snapshot),*args],text=True).strip()
    head=git('rev-parse','HEAD');expected=git('rev-parse','step0-local-validated^{commit}')
    if head!=expected or git('status','--porcelain'):raise ValueError('Require clean exact Step0 tag worktree')
    sources=[snapshot/'ATOM/atom/models/mellum.py',snapshot/'ATOM/atom/model_config/mellum_attention.py',snapshot/'ATOM/atom/entrypoints/openai/api_server.py']
    settings={**CANDIDATE,'HIP_VISIBLE_DEVICES':str(gpu),
        'PYTHONPATH':str(snapshot/'ATOM'),'MELLUM_LOAD_AUDIT_PATH':str(out/'loading_audit.json'),
        'MELLUM_PROFILE_LABELS':'0','MELLUM_REQUEST_BATCHING':'0'}
    cmd=['/opt/venv/bin/python','-m','atom.entrypoints.openai.api_server','--model',str(MODEL),
         '--host',host,'--server-port',str(http_port),'--port',str(engine_port),'-tp','1','-dp','1',
         '--enforce-eager','--level','0','--kv_cache_dtype','bf16','--max-model-len','16640',
         '--max-num-batched-tokens','16640','--max-num-seqs','1','--gpu-memory-utilization','0.35',
         '--no-enable_prefix_caching','--no-enable_chunked_prefill','--default-chat-template-kwargs','{"enable_thinking":true}']
    return {'snapshot_commit':head,'snapshot_tag':'step0-local-validated','settings':settings,'command':cmd,
        'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        'scope':'Historical Step0 math/weights; HTTP launcher outside snapshot, single scheduler slot',
        'warning':'Client A concurrency remains32; single slot queues requests. Not the later R batching baseline.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,default=Path('/workspace/tencent-mellum-step0-repro'))
    p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--server-port',type=int,default=8751);p.add_argument('--engine-port',type=int,default=9726)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if not 1<=a.server_port<=65535 or not 1<=a.engine_port<=65535 or a.server_port==a.engine_port:p.error('Distinct valid ports required')
    a.output.mkdir(parents=True,exist_ok=True);d=describe(a.snapshot,a.gpu,a.host,a.server_port,a.engine_port,a.output)
    (a.output/'launch.json').write_text(json.dumps(d,indent=2)+'\n')
    if a.prepare_only:print('RESULT historical_step0_prepared output',a.output);return
    for key in list(os.environ):
        if key.startswith('MELLUM_'):del os.environ[key]
    os.environ.update(d['settings']);os.execv(d['command'][0],d['command'])
if __name__=='__main__':main()
