"""Bounded CPU-planned C-shaped profiling; diagnostic, not official C arrivals or scores."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,apply_serving,engine_options
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha,load_rows,chat_ids
from mellum_process_runner import run_stage
from profile_mellum import wait_receipt
from analyze_mellum_profile import analyze_file,render_markdown

PROFILE='r_swa4_abort_prefix_chunk1024_d4'

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def verify(plan):
    for path,h in plan['sha256'].items():
        if sha(path)!=h:raise ValueError('Source/input changed: '+path)

def prepare(out,gpu,track):
    from transformers import AutoTokenizer
    source=Path('/data/datasets/scenario_c_prompts.jsonl')
    tokenizer=AutoTokenizer.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking',local_files_only=True)
    rows=load_rows(source)[:16]
    prompts=[chat_ids(tokenizer,[{'role':'user','content':r['prompt']}]) for r in rows]
    if any(not 4000<=len(p)<=16384 for p in prompts):raise ValueError('Unexpected C input length')
    files=[source,Path(__file__).resolve(),PROJECT/'scripts/mellum_runtime_config.py',PROJECT/'scripts/mellum_execution_profiles.py']
    files += [PROJECT/'scripts'/name for name in ['profile_mellum.py','analyze_mellum_profile.py','mellum_perf_protocol.py','mellum_process_runner.py']]
    files+=list((PROJECT/'ATOM/atom/model_config').glob('mellum*.py'))
    files+=list((PROJECT/'ATOM/atom/model_engine').glob('*.py'))+[PROJECT/'ATOM/atom/models/mellum.py']
    plan={'gpu':gpu,'track':track,'profile':PROFILE,'prompts':prompts,'tokens':16,
          'sha256':{str(p):sha(p) for p in files},
          'scope':'C-shaped controlled burst, not Poisson, not C60/C200 capacity or correctness acceptance. Profile timings are perturbed.'}
    write(out/'plan.json',plan);return plan

def workload(engine,plan,SamplingParams,out):
    # Each pass starts with drained queues; model/prefix state remains warm.
    audit_path=out/'engine_steps.jsonl'
    offset=len(audit_path.read_text().splitlines()) if audit_path.exists() else 0
    initial=16 if plan['track']=='decode' else 4
    injected=plan['track']=='decode';rounds=0;completed={};start=time.monotonic()
    engine.add_request(plan['prompts'][:initial],SamplingParams(temperature=0,max_tokens=16,ignore_eos=True))
    while not engine.is_finished():
        completed.update(engine.io_processor.postprocess(engine.step()));rounds+=1
        rows=[json.loads(x) for x in audit_path.read_text().splitlines()[offset:]] if audit_path.exists() else []
        if not injected and sum(r.get('phase')=='decode' for r in rows)>=4:
            engine.add_request(plan['prompts'][4:8],SamplingParams(temperature=0,max_tokens=16,ignore_eos=True));injected=True
    expected=16 if plan['track']=='decode' else 8
    if not injected or len(completed)!=expected or any(len(v['token_ids'])!=16 for v in completed.values()):
        raise ValueError('Incomplete controlled workload')
    return {'wall_s':time.monotonic()-start,'completed':expected,'output_tokens':expected*16,'frontend_steps':rounds,'actual_decode_steps':sum(r.get('phase')=='decode' for r in rows),'actual_prefill_steps':sum(r.get('phase')=='prefill' for r in rows)}

def worker(out):
    plan=json.loads((out/'plan.json').read_text());verify(plan)
    env=apply_serving(plan['gpu'],out,True,'reference',True);env.update(profile_settings(PROFILE))
    env.update(MELLUM_PROFILE_LABELS='1',MELLUM_PROFILE_GRANULAR='1',
               MELLUM_PROFILE_CONTROL_DIR=str(out/'control'),MELLUM_ENGINE_PHASE_AUDIT=str(out/'engine_steps.jsonl'))
    os.environ.update(env);sys.path.insert(0,str(PROJECT/'ATOM'))
    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs
    options=engine_options(16640,out/'traces');options.update(max_num_seqs=32,enable_prefix_caching=True,
        enable_chunked_prefill=True,max_num_batched_tokens=1024,port=14826+100*plan['gpu'])
    engine=None;result={'status':'started','profile':PROFILE,'options':options,'scope':plan['scope']}
    try:
        engine=EngineArgs(**options).create_engine()
        audit=json.loads((out/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:
            raise ValueError('Loading audit failed')
        result['warmup']=workload(engine,plan,SamplingParams,out)
        result['unprofiled']=[workload(engine,plan,SamplingParams,out) for _ in range(2)]
        engine.start_profile();result['started']=wait_receipt(out/'control','started',1,60)
        result['profiled']=workload(engine,plan,SamplingParams,out)
        engine.stop_profile();result['stopped']=wait_receipt(out/'control','stopped',1,300)
        traces=list((out/'traces').rglob('*.pt.trace.json.gz'))
        if len(traces)!=1:raise ValueError('Expected one rank0 GPU trace')
        analysis=analyze_file(traces[0])
        if not analysis['gpu_kernel_count'] or not analysis['module_annotation_count']:
            raise ValueError('Missing GPU kernels or module attribution')
        write(out/'trace_analysis.json',analysis);(out/'trace_analysis.md').write_text(render_markdown(analysis))
        result.update(status='bottleneck_profile_collected',trace=str(traces[0]),trace_sha256=sha(traces[0]))
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    finally:
        if engine is not None:engine.close()
        verify(plan);result['source_unchanged']=True
        write(out/'worker_results.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--track',choices=['decode','mixed'],default='decode');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.worker:worker(a.worker);return
    out=Path('/workspace/results/mellum_c_bottleneck_profile')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    prepare(out,a.gpu,a.track)
    if a.prepare_only:print('RESULT prepared output',out);return
    run_stage(a.track,[sys.executable,str(Path(__file__).resolve()),'--worker',str(out)],out/'worker.log',timeout=1200)
    result=json.loads((out/'worker_results.json').read_text())
    result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}
    write(out/'summary.json',result);print('RESULT',result['status'],'output',out)
if __name__=='__main__':main()
