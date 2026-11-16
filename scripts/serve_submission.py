"""Submission entrypoint: explicit correctness policy independent of component toggles."""
import argparse,hashlib,importlib.metadata as metadata,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT,MODEL,apply_serving
from mellum_execution_profiles import profile_settings


def build(args,out):
    preset=json.loads((PROJECT/'configs/submission'/f'{args.mode}.json').read_text())
    versions={p:metadata.version(p) for p in ['torch','transformers','triton','safetensors']}
    expected=json.loads((PROJECT/'configs/submission/environment.json').read_text())
    if versions!=expected:raise RuntimeError('Use the supplied frozen inference interpreter; package versions differ from environment.json')
    if not (args.model/'config.json').exists() or not (args.model/'model.safetensors.index.json').exists():raise ValueError('Model checkpoint directory missing')
    seq=args.max_num_seqs or preset['max_num_seqs'];length=args.max_model_len or preset['max_model_len']
    tokens=args.max_batched_tokens or preset['max_num_batched_tokens']
    if seq<1 or length<1 or tokens<length:raise ValueError('Invalid sequence or token limits')
    settings=apply_serving(args.gpu,out,True,'reference',args.audit)
    settings.update(profile_settings(preset['profile']))
    for flag,key,value in [('disable_graph','MELLUM_DECODE_GRAPH','0'),('disable_qk_fusion','MELLUM_FUSED_QK_ROPE','0'),
        ('disable_native_routing','MELLUM_NATIVE_ROUTING','0'),('disable_prefix','MELLUM_PREFIX_ENABLED','0'),
        ('disable_metadata_reuse','MELLUM_PREFILL_METADATA_REUSE','0'),('separate_qkv','MELLUM_QKV_BACKEND','separate'),
        ('disable_native_moe','MELLUM_MOE_BACKEND','torch'),('reference_norm','MELLUM_NORM_BACKEND','reference'),
        ('disable_direct_kv','MELLUM_DIRECT_KV_WRITE','0'),('disable_resident_rope','MELLUM_RESIDENT_ROPE','0')]:
        if getattr(args,flag):settings[key]=value
    if args.disable_native_attention:
        settings.update(MELLUM_ATTENTION_EXECUTION='reference',MELLUM_SDPA_DECODE='1',MELLUM_REQUEST_BATCHING='1',MELLUM_DIRECT_KV_WRITE='0')
    settings['MELLUM_FASTPATH_AUDIT_PATH']=str(out/'fastpaths.json')
    if args.audit:
        settings.update(MELLUM_NATIVE_AUDIT_PATH=str(out/'native_calls.jsonl'),MELLUM_GRAPH_AUDIT_PATH=str(out/'graph_replays.jsonl'),
            MELLUM_PREFIX_AUDIT_PATH=str(out/'prefix.jsonl'),MELLUM_METADATA_AUDIT_PATH=str(out/'metadata.jsonl'))
    command=[sys.executable,'-m','atom.entrypoints.openai.api_server','--model',str(args.model),'--host',args.host,
        '--server-port',str(args.port),'--port',str(args.engine_port),'-tp','1','-dp','1','--level','0','--kv_cache_dtype','bf16',
        '--max-model-len',str(length),'--max-num-batched-tokens',str(tokens),'--max-num-seqs',str(seq),
        '--gpu-memory-utilization',str(preset['gpu_memory_utilization']),
        '--enable_prefix_caching' if settings.get('MELLUM_PREFIX_ENABLED')=='1' else '--no-enable_prefix_caching',
        '--no-enable_chunked_prefill','--default-chat-template-kwargs','{"enable_thinking":true}']
    if settings.get('MELLUM_DECODE_GRAPH')=='1':command+=['--cudagraph-capture-sizes','[1,2,4,8,16,32]']
    else:command+=['--enforce-eager']
    source=[PROJECT/'scripts/serve_submission.py',PROJECT/'scripts/mellum_execution_profiles.py',PROJECT/'scripts/mellum_runtime_config.py',
        PROJECT/'configs/submission'/f'{args.mode}.json',PROJECT/'ATOM/atom/models/mellum.py']
    return {'mode':args.mode,'preset':preset,'versions':versions,'settings':settings,'command':command,
        'declared_gpus':1,'effective_max_num_seqs':seq,'component_overrides':[f for f in ['disable_graph','disable_qk_fusion','disable_native_routing','disable_prefix','disable_metadata_reuse','separate_qkv','disable_native_moe','disable_native_attention','reference_norm','disable_direct_kv','disable_resident_rope'] if getattr(args,f)],
        'source_sha256':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'note':'Mode selects a documented runtime policy; it is not Step0 or an optimization-point claim. Accuracy and speed results must use the same effective configuration.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['conservative','aggressive'],default='conservative')
    p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--model',type=Path,default=MODEL)
    p.add_argument('--host',default='0.0.0.0');p.add_argument('--port','--server-port',dest='port',type=int,default=8000)
    p.add_argument('--engine-port',type=int,default=8026);p.add_argument('--max-num-seqs',type=int)
    p.add_argument('--max-model-len',type=int);p.add_argument('--max-batched-tokens',type=int)
    p.add_argument('--audit',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--output',type=Path)
    for f in ['disable-graph','disable-qk-fusion','disable-native-routing','disable-prefix','disable-metadata-reuse','separate-qkv','disable-native-moe','disable-native-attention','reference-norm','disable-direct-kv','disable-resident-rope']:p.add_argument('--'+f,action='store_true')
    args=p.parse_args()
    if not 1<=args.port<=65535 or not 1<=args.engine_port<=65535 or args.port==args.engine_port:p.error('Distinct valid HTTP and engine ports required')
    out=args.output or PROJECT/'results/submission_server'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True);d=build(args,out);(out/'launch.json').write_text(json.dumps(d,indent=2)+'\n')
    print('MODE',args.mode,'STATUS',d['preset']['status'],'LOG_DIR',out,flush=True)
    if args.dry_run:return
    os.environ.update(d['settings']);os.execv(d['command'][0],d['command'])
if __name__=='__main__':main()
