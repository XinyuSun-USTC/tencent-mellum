"""One-command frozen Mellum serving, reference by default, no diagnostic dumps."""
import argparse,json,os,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import apply_serving,PROJECT,MODEL
from mellum_execution_profiles import PROFILES,profile_settings


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--server-port',type=int,default=8051);p.add_argument('--engine-port',type=int,default=8026)
    p.add_argument('--max-num-seqs',type=int,default=1);p.add_argument('--max-model-len',type=int,default=16640)
    p.add_argument('--max-batched-tokens',type=int,default=16640);p.add_argument('--implementation',choices=['reference','ck','triton'],default='reference')
    p.add_argument('--batching',action='store_true');p.add_argument('--batch-audit',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--phase-audit',action='store_true',help='Diagnostic EngineCore host timestamps; not final performance mode')
    p.add_argument('--disable-thinking',action='store_true')
    p.add_argument('--campaign-config',type=Path)
    p.add_argument('--optimization-points',nargs='*',type=int,choices=[1,2,3],default=None,help='Explicit delivery point override; effective settings recorded in launch.json')
    p.add_argument('--profile',choices=PROFILES,help='Explicit experimental execution profile; requires --batching')
    a=p.parse_args()
    if a.optimization_points is not None and (not a.profile or not a.profile.startswith('delivery_step')):p.error('--optimization-points requires a delivery_step profile')
    if a.profile and (not a.batching or a.campaign_config or a.implementation!='reference'):
        p.error('--profile requires --batching and cannot combine with campaign-config/implementation')
    if a.max_num_seqs<1 or a.max_model_len<1 or a.max_batched_tokens<a.max_model_len:p.error('Invalid sequence/token limits')
    if a.max_num_seqs>1 and not a.batching:p.error('Multiple sequences require --batching')
    if a.engine_port==a.server_port:p.error('Engine and HTTP ports must differ')
    out=a.output or Path('/workspace/results/mellum_serving')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=True)
    settings=apply_serving(a.gpu,out,a.batching,a.implementation,a.batch_audit)
    if a.profile:
        if a.optimization_points is None:settings.update(profile_settings(a.profile))
        else:
            from mellum_delivery_matrix import point_settings
            settings.update(point_settings(a.optimization_points))
        if a.batch_audit:settings['MELLUM_NATIVE_AUDIT_PATH']=str(out/'native_calls.jsonl')
        os.environ.update(settings)
    if a.phase_audit:settings['MELLUM_ENGINE_PHASE_AUDIT']=str(out/'engine_steps.jsonl')
    settings['MELLUM_FASTPATH_AUDIT_PATH']=str(out/'fastpaths.json')
    os.environ.update(settings)
    if a.campaign_config:
        config=json.loads(a.campaign_config.read_text())
        if config.get('observe'):raise ValueError('HTTP campaign must disable observer')
        settings['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(config)
        os.environ.update(settings)
    command=['/opt/venv/bin/python','-m','atom.entrypoints.openai.api_server','--model',str(MODEL),
        '--host',a.host,'--server-port',str(a.server_port),'--port',str(a.engine_port),'-tp','1','-dp','1',
        '--enforce-eager','--level','0','--kv_cache_dtype','bf16','--max-model-len',str(a.max_model_len),
        '--max-num-batched-tokens',str(a.max_batched_tokens),'--max-num-seqs',str(a.max_num_seqs),
        '--gpu-memory-utilization','0.35','--no-enable_prefix_caching','--no-enable_chunked_prefill',
        '--default-chat-template-kwargs',json.dumps({'enable_thinking':not a.disable_thinking},separators=(',',':'))]
    if settings.get('MELLUM_PREFILL_METADATA_REUSE')=='1':
        settings['MELLUM_METADATA_AUDIT_PATH']=str(out/'metadata.jsonl')
        os.environ.update(settings)
    if settings.get('MELLUM_PREFIX_COMPAT')=='1':
        settings['MELLUM_PREFIX_AUDIT_PATH']=str(out/'prefix.jsonl')
        if settings.get('MELLUM_PREFIX_ENABLED')=='1':
            command[command.index('--no-enable_prefix_caching')]='--enable_prefix_caching'
        chunk=int(settings.get('MELLUM_CHUNK_BUDGET','0'))
        if chunk:
            command[command.index('--no-enable_chunked_prefill')]='--enable_chunked_prefill'
            command[command.index('--max-num-batched-tokens')+1]=str(chunk)
        os.environ.update(settings)
    if settings.get('MELLUM_REFERENCE_PREFIX')=='1':
        command[command.index('--no-enable_prefix_caching')]='--enable_prefix_caching'
    if settings.get('MELLUM_REFERENCE_CHUNKED')=='1':
        command[command.index('--no-enable_chunked_prefill')]='--enable_chunked_prefill'
        command[command.index('--max-num-batched-tokens')+1]=settings['MELLUM_CHUNK_BUDGET']
    if settings.get('MELLUM_DECODE_GRAPH')=='1':
        command.remove('--enforce-eager')
        command.extend(['--cudagraph-capture-sizes','[1,2,4,8,16,32]'])
        settings['MELLUM_GRAPH_AUDIT_PATH']=str(out/'graph_replays.jsonl')
        os.environ.update(settings)
    source=[PROJECT/'scripts/mellum_execution_profiles.py',PROJECT/'ATOM/atom/model_config/mellum_native.py',PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'ATOM/atom/model_config/mellum_batch.py',
            PROJECT/'ATOM/atom/model_config/mellum_attention.py',PROJECT/'ATOM/atom/entrypoints/openai/api_server.py',
            PROJECT/'scripts/mellum_runtime_config.py',PROJECT/'ATOM/atom/model_engine/engine_core.py',PROJECT/'ATOM/atom/model_engine/mellum_service_audit.py',Path(__file__)]
    (out/'launch.json').write_text(json.dumps({'profile':a.profile,'enabled_optimization_points':a.optimization_points if a.optimization_points is not None else next((item['points'] for item in __import__('mellum_delivery_matrix').ROWS.values() if item['profile']==a.profile),None),'acceptance':'Experimental profile; do not infer final correctness/performance acceptance from launcher availability','settings':settings,'command':command,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source}},indent=2)+'\n')
    if a.prepare_only:print('RESULT prepared output',out);return
    print('SERVE output',out,flush=True);os.execv(command[0],command)
if __name__=='__main__':main()
