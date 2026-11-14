"""First real ATOM Mellum load/greedy check. Run in the frozen interpreter."""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

ROOT=Path('/workspace')
sys.path.insert(0,str(ROOT/'ATOM'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=0,help='Physical GPU selected through HIP_VISIBLE_DEVICES')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--dump-layers',action='store_true')
    p.add_argument('--dump-stages',action='store_true')
    p.add_argument('--moe-backend',choices=['torch','aiter'],default='torch')
    p.add_argument('--swa-prefill-impl',choices=['ck','triton'],default='ck')
    p.add_argument('--swa-prefill-opt',action='store_true',help='Opt-in local SWA prefill; retain reference other paths')
    p.add_argument('--prefill-backend',choices=['aiter','sdpa_diagnostic','sdpa_rope_diagnostic','sdpa_cpu_rope_diagnostic'],default='aiter')
    p.add_argument('--moe-replay-inputs',type=Path,help='Diagnostic only: replace observed layer-zero MoE inputs')
    p.add_argument('--moe-replay-routing',choices=['native','rounded_native','hf'],default='native')
    p.add_argument('--max-batched-tokens',type=int)
    p.add_argument('--reference',type=Path,default=ROOT/'results/atom_step0_batch_ab/hf_reference/reference.json')
    p.add_argument('--stage-layer',type=int,choices=range(28),default=0)
    p.add_argument('--qkv-backend',choices=['packed','separate'],default='packed')
    p.add_argument('--forced-decode',action='store_true',help='Diagnostic only: save true logits, force reference token history')
    p.add_argument('--reference-kv-rope',action='store_true')
    p.add_argument('--sdpa-decode',action='store_true')
    p.add_argument('--decode-layer-capture',action='store_true')
    p.add_argument('--torch-router',action='store_true')
    p.add_argument('--hf-generation-experts',action='store_true',help='Match HF generate grouped prefill/batched decode experts')
    p.add_argument('--dump-generation-logits',action='store_true')
    a=p.parse_args()
    if a.swa_prefill_impl!='ck' and not a.swa_prefill_opt:p.error('--swa-prefill-impl requires --swa-prefill-opt')
    if a.swa_prefill_opt and a.prefill_backend!='sdpa_cpu_rope_diagnostic':p.error('--swa-prefill-opt requires CPU RoPE prefill')
    if a.hf_generation_experts and a.moe_backend!='torch':p.error('--hf-generation-experts requires torch MoE')
    if a.torch_router and a.moe_backend!='torch':p.error('--torch-router requires --moe-backend torch')
    if a.forced_decode and a.dump_generation_logits:p.error('Passive generation capture cannot be combined with forced decode')
    if a.sdpa_decode and not a.reference_kv_rope:p.error('--sdpa-decode requires --reference-kv-rope')
    if a.decode_layer_capture and not (a.forced_decode and a.sdpa_decode):p.error('Layer capture requires forced SDPA decode')
    if not a.moe_replay_inputs:os.environ.pop('MELLUM_MOE_REPLAY_INPUT_DIR',None)
    os.environ.pop('MELLUM_MOE_REPLAY_ROUTING',None)
    os.environ.pop('MELLUM_FORCED_DECODE_PLAN',None)
    os.environ.pop('MELLUM_FORCED_DECODE_DIR',None)
    os.environ.pop('MELLUM_FREE_LOGITS_DIR',None)
    if a.moe_replay_routing!='native' and not a.moe_replay_inputs:raise ValueError('Routing intervention requires MoE input replay')
    out=ROOT/'results/atom_mellum_gpu'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    settings={'HIP_VISIBLE_DEVICES':str(a.gpu),'ATOM_LOADER_USE_THREADPOOL':'0',
       'MELLUM_HF_GENERATION_EXPERTS':'1' if a.hf_generation_experts else '0',
       'MELLUM_MOE_BACKEND':a.moe_backend,'MELLUM_QKV_BACKEND':a.qkv_backend,
       'MELLUM_ROUTER_BACKEND':'torch' if a.torch_router else 'aiter',
       'MELLUM_PREFILL_BACKEND':a.prefill_backend,'MELLUM_SWA_PREFILL_OPT':'1' if a.swa_prefill_opt else '0','MELLUM_SWA_PREFILL_IMPL':a.swa_prefill_impl,'MELLUM_STAGE_LAYER':str(a.stage_layer),
       'MELLUM_REFERENCE_KV_ROPE':'1' if a.reference_kv_rope else '0',
       'MELLUM_SDPA_DECODE':'1' if a.sdpa_decode else '0',
       'MELLUM_DECODE_LAYER_CAPTURE':'1' if a.decode_layer_capture else '0',
       'ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION':'0','ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION':'0',
       'ATOM_USE_UNIFIED_ATTN':'0','AITER_LOG_LEVEL':'WARNING',
       'MELLUM_LOAD_AUDIT_PATH':str(out/'loading_audit.json'),
       'PYTHONPATH':str(ROOT/'ATOM')+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),
       'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    os.environ.update(settings)
    result={'status':'started','settings':settings,'records':[],'phase':'preflight'}
    result.update(sampling_mode='forced_history' if a.forced_decode else 'normal',ignore_eos=a.forced_decode)
    if a.dump_generation_logits:
        settings['MELLUM_FREE_LOGITS_DIR']=str(out/'generation_logits')
        os.environ['MELLUM_FREE_LOGITS_DIR']=settings['MELLUM_FREE_LOGITS_DIR']
    def save():(out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    reference=json.loads(a.reference.read_text())
    if a.forced_decode:
        if a.moe_replay_inputs:raise ValueError('Cannot combine forced decode with input replay')
        settings.update(MELLUM_FORCED_DECODE_PLAN=str(a.reference.resolve()),MELLUM_FORCED_DECODE_DIR=str(out/'decode'))
        os.environ.update(settings)
        result['scope']='forced_history_diagnostic_not_generation_acceptance'
    if a.moe_replay_inputs:
        if a.stage_layer!=0:raise ValueError('Existing MoE replay inputs are layer zero only')
        if not (a.dump_stages and a.dump_layers):raise ValueError('MoE replay requires stage/layer capture')
        for i,record in enumerate(reference['records']):
            if record['generation_settings']['max_new_tokens']!=1:raise ValueError('MoE replay requires one-token probes')
            if not (a.moe_replay_inputs/f'sample_{i:06d}'/'moe_input.pt').is_file():raise FileNotFoundError(a.moe_replay_inputs)
        settings['MELLUM_MOE_REPLAY_INPUT_DIR']=str(a.moe_replay_inputs.resolve())
        settings['MELLUM_MOE_REPLAY_ROUTING']=a.moe_replay_routing
        os.environ['MELLUM_MOE_REPLAY_ROUTING']=a.moe_replay_routing
        os.environ['MELLUM_MOE_REPLAY_INPUT_DIR']=settings['MELLUM_MOE_REPLAY_INPUT_DIR']
        result['scope']='diagnostic_intervention_not_generation_validation'
    required=max(len(r['input_ids'][0])+r['generation_settings']['max_new_tokens'] for r in reference['records'])
    max_len=max(256,((required+15)//16)*16)
    os.environ['MELLUM_PREFILL_LOGITS_DIR']=str(out/'logits')
    settings['MELLUM_PREFILL_LOGITS_DIR']=str(out/'logits')
    if a.dump_layers:
        os.environ['MELLUM_LAYER_DUMP_DIR']=str(out/'layers')
        settings['MELLUM_LAYER_DUMP_DIR']=str(out/'layers')
    if a.dump_stages:
        if not a.dump_layers:raise ValueError('--dump-stages requires --dump-layers for sample indices')
        os.environ['MELLUM_STAGE_DUMP_DIR']=str(out/'stages')
        settings['MELLUM_STAGE_DUMP_DIR']=str(out/'stages')
    result.update(reference=str(a.reference.resolve()),max_model_len=max_len)
    engine=None
    try:
        import torch,transformers
        result['environment']={'python':sys.executable,'transformers':transformers.__version__,
          'torch':torch.__version__,'visible_gpus':torch.cuda.device_count()}
        if transformers.__version__!='5.2.0':raise RuntimeError('Use the frozen /opt/venv interpreter')
        if not torch.cuda.is_available():
            result['status']='blocked_no_gpu';save();print('No GPU exposed; no checkpoint loaded. Results:',out);return 2
        free,total=torch.cuda.mem_get_info(0)
        if free < 80*1024**3:raise RuntimeError('First run requires at least80GiB free; no checkpoint loaded')
        if a.preflight_only:
            result['status']='preflight_passed';save();print(out);return 0
        from atom.model_engine.arg_utils import EngineArgs
        from atom import SamplingParams
        result['phase']='engine_initialization';save()
        engine=EngineArgs(model='/data/models/Mellum2.1-12B-A2.5B-Thinking',
           tensor_parallel_size=1,data_parallel_size=1,enforce_eager=True,level=0,
           kv_cache_dtype='bf16',max_model_len=max_len,max_num_batched_tokens=max(max_len,a.max_batched_tokens or max_len),
           max_num_seqs=1,gpu_memory_utilization=0.35,
           enable_prefix_caching=False,enable_chunked_prefill=False).create_engine()
        audit=json.loads((out/'loading_audit.json').read_text())
        if audit['source_tensors']!=5631 or audit['missing'] or audit['unexpected']:raise RuntimeError('Incomplete loading audit')
        result['loading_audit']=audit;result['phase']='generation';save()
        candidate={**reference,'producer':'atom','records':[],'environment':result['environment']}
        import hashlib
        for index,record in enumerate(reference['records']):
            tokens=record['input_ids'][0]
            outputs=engine.generate([tokens],SamplingParams(temperature=0,max_tokens=record['generation_settings']['max_new_tokens'],ignore_eos=a.forced_decode))
            if len(outputs)!=1:raise RuntimeError('Missing engine output')
            output=outputs[0]
            got=output['token_ids'];want=record['generated_ids'][0]
            if a.forced_decode and got!=want:raise RuntimeError('Forced engine history did not match plan')
            row={'name':record['name'],'input_ids':tokens,'generated_ids':got,
                 'hf_generated_ids':want,'token_exact':got==want,'text':output['text']}
            result['records'].append(row);save()
            logits=out/'logits'/f'prefill_{index:06d}.pt'
            if not logits.exists():raise RuntimeError('Expected per-request prefill logits were not saved')
            candidate['records'].append({**record,'generated_ids':[got],
                'prefill_logits':str(logits.resolve()),
                'prefill_logits_sha256':hashlib.sha256(logits.read_bytes()).hexdigest()})
            if a.dump_generation_logits:
                files=[out/'generation_logits'/f'sample_{index:06d}'/f'step_{step:03d}_logits.pt' for step in range(len(got))]
                candidate['records'][-1]['generation_logits']=[{'path':str(f.resolve()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]
            (out/'candidate.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+'\n')
            print(row['name'],'token exact:',row['token_exact'],'text:',repr(row['text']),flush=True)
        result['status']=('forced_decode_complete' if a.forced_decode else 'moe_replay_complete' if a.moe_replay_inputs else
                          'short_greedy_passed' if all(r['token_exact'] for r in result['records']) else 'token_mismatch')
        save();print('RESULT',result['status'],'output',out)
        return 0 if result['status'] in ['short_greedy_passed','moe_replay_complete','forced_decode_complete'] else 1
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save()
        print(result['traceback'],file=sys.stderr);print('Results:',out,file=sys.stderr);return 1
    finally:
        if engine is not None:engine.close()

if __name__=='__main__':
    multiprocessing.set_start_method('spawn',force=True)
    raise SystemExit(main())
