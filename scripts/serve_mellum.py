"""One-command frozen Mellum serving, reference by default, no diagnostic dumps."""
import argparse,json,os,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import apply_serving,PROJECT,MODEL


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--server-port',type=int,default=8051);p.add_argument('--engine-port',type=int,default=8026)
    p.add_argument('--max-num-seqs',type=int,default=1);p.add_argument('--max-model-len',type=int,default=16640)
    p.add_argument('--max-batched-tokens',type=int,default=16640);p.add_argument('--implementation',choices=['reference','ck','triton'],default='reference')
    p.add_argument('--batching',action='store_true');p.add_argument('--batch-audit',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.max_num_seqs<1 or a.max_model_len<1 or a.max_batched_tokens<a.max_model_len:p.error('Invalid sequence/token limits')
    if a.max_num_seqs>1 and not a.batching:p.error('Multiple sequences require --batching')
    if a.engine_port==a.server_port:p.error('Engine and HTTP ports must differ')
    out=a.output or Path('/workspace/results/mellum_serving')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=True)
    settings=apply_serving(a.gpu,out,a.batching,a.implementation,a.batch_audit)
    command=['/opt/venv/bin/python','-m','atom.entrypoints.openai.api_server','--model',str(MODEL),
        '--host',a.host,'--server-port',str(a.server_port),'--port',str(a.engine_port),'-tp','1','-dp','1',
        '--enforce-eager','--level','0','--kv_cache_dtype','bf16','--max-model-len',str(a.max_model_len),
        '--max-num-batched-tokens',str(a.max_batched_tokens),'--max-num-seqs',str(a.max_num_seqs),
        '--gpu-memory-utilization','0.35','--no-enable_prefix_caching','--no-enable_chunked_prefill',
        '--default-chat-template-kwargs','{"enable_thinking":true}']
    source=[PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'ATOM/atom/model_config/mellum_batch.py',
            PROJECT/'ATOM/atom/model_config/mellum_attention.py',PROJECT/'ATOM/atom/entrypoints/openai/api_server.py',
            PROJECT/'scripts/mellum_runtime_config.py',Path(__file__)]
    (out/'launch.json').write_text(json.dumps({'settings':settings,'command':command,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source}},indent=2)+'\n')
    if a.prepare_only:print('RESULT prepared output',out);return
    print('SERVE output',out,flush=True);os.execv(command[0],command)
if __name__=='__main__':main()
