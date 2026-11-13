"""Expanded bs1 timing surface and granular traces, without optimizing model math."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import random
import shutil
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from mellum_runtime_config import PROJECT, MODEL, CANDIDATE, apply_candidate, engine_options
from profile_mellum import prepare_inputs, save, generate_request, capture_request
from mellum_process_runner import run_stage
from mellum_profile_telemetry import TelemetrySampler


LENGTHS = [512, 1024, 4096, 8192, 16384]


def percentile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError('No samples')
    position = (len(values)-1)*q
    left = int(position); right = min(left+1, len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def summarize(rows):
    keys = ['wall_seconds','request_ttft_seconds','request_tpot_seconds','request_latency_seconds','output_tokens_per_wall_second']
    result = {}
    for key in keys:
        values = [r[key] for r in rows if key!='request_tpot_seconds' or r['output_tokens']>1]
        if not values:
            result[key]=dict(n=0,mean=None,median=None,p90=None,p99=None,minimum=None,maximum=None,coefficient_of_variation=None)
            continue
        mean = statistics.mean(values)
        result[key] = dict(n=len(values),mean=mean,median=percentile(values,.5),
                           p90=percentile(values,.9),p99=percentile(values,.99) if len(values)>=50 else None,
                           minimum=min(values),maximum=max(values),
                           coefficient_of_variation=statistics.pstdev(values)/mean if mean else None)
    result['scope'] = 'Empirical offline request statistics; no HTTP, open-loop, or official scores'
    return result


def real_cases(tokenizer,dataset_root=Path('/data/datasets')):
    cases=[];sources=[]
    for short, name in [('a','scenario_a_prompts.jsonl'),('c','scenario_c_prompts.jsonl')]:
        path=dataset_root/name
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        eligible=[]
        for index,row in enumerate(rows):
            text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,
                                               add_generation_prompt=True,enable_thinking=True)
            ids=tokenizer.encode(text,add_special_tokens=False)
            if 0<len(ids)<=16384:eligible.append((len(ids),index,row,ids))
        eligible.sort(key=lambda x:(x[0],x[1]));chosen=set()
        if len(eligible)<2:raise ValueError(f'Insufficient eligible real prompts: {name}')
        for label,q in [('median',.5),('long',.9)]:
            position=int((len(eligible)-1)*q)
            while eligible[position][1] in chosen:position=(position+1)%len(eligible)
            n,index,row,ids=eligible[position];chosen.add(index)
            cases.append(dict(name=f'real_{short}_{label}',input_ids=ids,input_tokens=n,
                source=str(path),source_id=row.get('id',index),source_index=index,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                derivation='Untruncated supplied prompt with fixed chat template/thinking',
                intended_source_quantile=q))
        sources.append(dict(path=str(path),total_rows=len(rows),eligible_rows=len(eligible),
                            excluded_for_length=len(rows)-len(eligible)))
    return cases,sources


def build_plans(synthetic, real, repeats=5, b_repeats=50, seed=20261113):
    timing=[]
    for case in synthetic:
        for tokens in [1,64,256]:
            count=b_repeats if case['input_tokens']==1024 and tokens==256 else repeats
            timing.append(dict(case=case,output_tokens=tokens,repeats=count))
    timing += [dict(case=c,output_tokens=256,repeats=3) for c in real]
    random.Random(seed).shuffle(timing)
    return dict(timing=dict(mode='timing',jobs=timing),
                trace=dict(mode='trace',jobs=[dict(case=c,output_tokens=16,repeats=1) for c in synthetic+real]),
                detailed=dict(mode='detailed',jobs=[dict(case=next(c for c in synthetic if c['input_tokens']==1024),output_tokens=4,repeats=1)]))


def study_worker(output):
    plan=json.loads((output/'plan.json').read_text());mode=plan['mode']
    settings=apply_candidate(plan['gpu'],output,detailed=mode=='detailed')
    # Timings have neither profiler nor module hooks; trace jobs add only observations.
    import os
    overrides={'MELLUM_PROFILE_LABELS':'0' if mode=='timing' else '1',
               'MELLUM_PROFILE_GRANULAR':'0' if mode=='timing' else '1',
               'MELLUM_PROFILE_MEMORY_STATS':'0' if mode=='timing' else '1'}
    os.environ.update(overrides);settings.update(overrides)
    sys.path.insert(0,str(PROJECT/'ATOM'))
    result=dict(status='started',mode=mode,settings=settings,jobs=[],
                scope='Expanded sequential bs1 diagnostic, not formal A/B/C performance')
    save(output/'worker_results.json',result);engine=None
    try:
        import torch,transformers
        if transformers.__version__!='5.2.0':raise RuntimeError('Use frozen /opt/venv')
        result['environment']=dict(torch=torch.__version__,hip=torch.version.hip,transformers=transformers.__version__)
        if not torch.cuda.is_available():
            result['status']='blocked_no_gpu';save(output/'worker_results.json',result);return 2
        props=torch.cuda.get_device_properties(0)
        result['gpu']=dict(name=props.name,arch=getattr(props,'gcnArchName','unknown'),total_memory_bytes=props.total_memory)
        if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require 80GiB free before loading')
        from atom.model_engine.arg_utils import EngineArgs
        from atom import SamplingParams
        opts=engine_options(plan['engine_required_tokens'],output/'traces')
        if mode=='timing':opts['torch_profiler_dir']=None
        result['engine_options']=opts;save(output/'worker_results.json',result)
        print('STUDY engine_initialization',mode,flush=True)
        engine=EngineArgs(**opts).create_engine()
        audit=json.loads((output/'loading_audit.json').read_text())
        if audit['source_tensors']!=5631 or audit['successful_parts']!=5631 or audit['missing'] or audit['unexpected']:
            raise RuntimeError('Incomplete loading audit')
        result['loading_audit']=audit
        for sequence,job in enumerate(plan['jobs'],1):
            case=job['case'];tokens=job['output_tokens']
            print('STUDY warmup',mode,case['name'],'output',tokens,flush=True)
            generate_request(engine,case,tokens,SamplingParams)
            if mode=='timing':
                rows=[]
                for repeat in range(job['repeats']):
                    print('STUDY timing',case['name'],tokens,repeat+1,'/',job['repeats'],flush=True)
                    rows.append(generate_request(engine,case,tokens,SamplingParams))
                    # Incremental progress survives a later worker/timeout failure.
                    save(output/'progress.json',dict(job=sequence,case=case['name'],output_tokens=tokens,completed=rows))
                entry=dict(name=case['name'],input_tokens=case['input_tokens'],output_tokens=tokens,
                           rows=rows,statistics=summarize(rows))
            else:
                paired=generate_request(engine,case,tokens,SamplingParams)
                print('STUDY capture',mode,case['name'],flush=True)
                capture=capture_request(engine,case,tokens,output,sequence,SamplingParams)
                if paired['token_ids']!=capture['profiled_request']['token_ids']:
                    raise RuntimeError('Profiling changed greedy tokens in paired request')
                entry=dict(name=case['name'],input_tokens=case['input_tokens'],output_tokens=tokens,
                    paired_unprofiled=paired,**capture,
                    profile_wall_ratio=capture['profiled_request']['wall_seconds']/paired['wall_seconds'],
                    profile_is_not_benchmark=True)
            result['jobs'].append(entry);save(output/'worker_results.json',result)
            print('STUDY job_complete',mode,case['name'],tokens,flush=True)
        result['status']='timing_collected' if mode=='timing' else 'profile_collected'
        save(output/'worker_results.json',result);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save(output/'worker_results.json',result)
        print(result['traceback'],flush=True);return 1
    finally:
        if engine is not None:engine.close()


def render_study(output):
    lines=['# Expanded bs1 study','','离线单请求开发计时，不是正式A/B/C成绩；小样本不用于推断p99。','',
           '| 输入/样本 | 输入token | 输出token | n | TTFT中位ms | TPOT中位ms | wall中位s |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    timing=json.loads((output/'timing/worker_results.json').read_text())
    for job in timing['jobs']:
        stats=job['statistics'];tpot=stats['request_tpot_seconds']['median'];tpot_text='N/A' if tpot is None else f'{tpot*1000:.3f}';lines.append(f"| {job['name']} | {job['input_tokens']} | {job['output_tokens']} | {len(job['rows'])} | {stats['request_ttft_seconds']['median']*1000:.3f} | {tpot_text} | {stats['wall_seconds']['median']:.3f} |")
    (output/'study_summary.md').write_text('\n'.join(lines)+'\n')


def fingerprints():
    paths=list((PROJECT/'scripts').glob('*profile*.py'))+[PROJECT/'scripts/mellum_runtime_config.py',PROJECT/'ATOM/atom/model_config/mellum_profile.py',PROJECT/'ATOM/atom/model_engine/model_runner.py']
    return {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def completed_stage(folder):
    path=folder/'worker_results.json'
    if not path.exists():return False
    result=json.loads(path.read_text())
    plan=json.loads((folder/'plan.json').read_text())
    expected='timing_collected' if plan['mode']=='timing' else 'profile_collected'
    if result['status']!=expected:return False
    if result.get('mode')!=plan['mode'] or any(result.get('settings',{}).get(k)!=v for k,v in CANDIDATE.items()):
        raise ValueError('Completed runtime configuration mismatch')
    if len(result['jobs'])!=len(plan['jobs']):raise ValueError('Incomplete supposedly successful stage')
    for actual,wanted in zip(result['jobs'],plan['jobs']):
        if actual['name']!=wanted['case']['name'] or actual['input_tokens']!=wanted['case']['input_tokens'] or actual['output_tokens']!=wanted['output_tokens']:
            raise ValueError('Completed job does not match frozen plan')
        if plan['mode']=='timing':
            if len(actual['rows'])!=wanted['repeats'] or any(len(r['token_ids'])!=wanted['output_tokens'] for r in actual['rows']):
                raise ValueError('Invalid completed timing observations')
        else:
            trace=Path(actual['trace'])
            if not trace.exists() or hashlib.sha256(trace.read_bytes()).hexdigest()!=actual['trace_sha256']:
                raise ValueError('Completed trace missing or changed')
    return True


def archive_attempt(output,name):
    folder=output/name;log=output/(name+'.log')
    if not log.exists() and not any(p.name!='plan.json' for p in folder.iterdir()):return
    saved=output/'previous_attempts'/(name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    saved.parent.mkdir(exist_ok=True);folder.rename(saved);folder.mkdir()
    shutil.copyfile(saved/'plan.json',folder/'plan.json')
    if log.exists():log.rename(saved/'parent_stage.log')


def execute_study(output,summary,stages,args):
    telemetry=TelemetrySampler(output/'telemetry.jsonl',args.telemetry_interval)
    if args.telemetry:telemetry.start_sampling()
    try:
        for name in stages:
            if completed_stage(output/name):
                print('STUDY reused completed stage',name,flush=True)
                summary.setdefault('reused_stages',[]).append(name)
                continue
            archive_attempt(output,name)
            summary['phase']=name;save(output/'summary.json',summary)
            code,_=run_stage(name,['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker-output',str(output/name)],output/(name+'.log'),timeout=args.stage_timeout,allowed_exit_codes=(0,2),with_code=True)
            if code:
                summary['status']='blocked_no_gpu';save(output/'summary.json',summary)
                print('RESULT blocked_no_gpu output',output,flush=True);return code
        render_study(output);summary['status']='study_partially_collected' if summary.get('skipped_stages') else 'study_collected';save(output/'summary.json',summary)
        print('RESULT',summary['status'],'output',output,flush=True);return 0
    except Exception:
        summary.update(status='failed',traceback=traceback.format_exc());save(output/'summary.json',summary)
        print(summary['traceback'],flush=True);print('RESULT failed output',output,flush=True);return 1
    finally:
        if args.telemetry:telemetry.close()


def resume_study(args):
    output=args.resume.resolve();summary=json.loads((output/'summary.json').read_text())
    if summary['source_sha256']!=fingerprints():raise ValueError('Resume rejected: profiling source fingerprints changed')
    for name,expected in summary['plan_sha256'].items():
        path=output/name/'plan.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Resume rejected: plan changed')
        if json.loads(path.read_text())['model_config_sha256']!=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest():
            raise ValueError('Resume rejected: model config changed')
    stages=list(summary['case_counts'])
    summary.pop('skipped_stages',None)
    if not args.detailed and 'detailed' in stages:
        stages.remove('detailed');summary['skipped_stages']=['detailed']
    if args.prepare_only:
        print('RESULT resume_verified output',output,flush=True);return 0
    previous=output/'resume_history';previous.mkdir(exist_ok=True)
    shutil.copyfile(output/'summary.json',previous/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'))
    return execute_study(output,summary,stages,args)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--b-repeats',type=int,default=50)
    parser.add_argument('--seed',type=int,default=20261113)
    parser.add_argument('--real-prompts',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--detailed',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--stage-timeout',type=float,default=5400)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',type=Path,help='Resume saved stages with unchanged profiling sources/plans')
    parser.add_argument('--telemetry',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--telemetry-interval',type=float,default=15)
    parser.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_output:return study_worker(args.worker_output)
    if args.repeats<1 or args.b_repeats<1 or args.stage_timeout<=0 or args.telemetry_interval<=0:parser.error('Budgets must be positive')
    if args.resume:return resume_study(args)
    synthetic=prepare_inputs(LENGTHS);real=[];sources=[]
    if args.real_prompts:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        real,sources=real_cases(tokenizer)
    plans=build_plans(synthetic,real,args.repeats,args.b_repeats,args.seed)
    if not args.detailed:del plans['detailed']
    output=Path('/workspace/results/mellum_profile_study')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');output.mkdir(parents=True)
    cfg=json.loads((MODEL/'config.json').read_text())
    common=dict(seed=args.seed,model_structure={k:cfg.get(k) for k in ['num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','head_dim','num_experts','num_experts_per_tok','moe_intermediate_size','layer_types']},gpu=args.gpu,engine_required_tokens=max(c['input_tokens'] for c in synthetic+real)+256,
                model_config_sha256=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest())
    for name,plan in plans.items():
        folder=output/name;folder.mkdir();save(folder/'plan.json',dict(**plan,**common))
    fingerprint_paths=list((PROJECT/'scripts').glob('*profile*.py'))+[PROJECT/'scripts/mellum_runtime_config.py',PROJECT/'ATOM/atom/model_config/mellum_profile.py',PROJECT/'ATOM/atom/model_engine/model_runner.py']
    summary=dict(telemetry_enabled=args.telemetry,telemetry_interval_seconds=args.telemetry_interval,status='prepared',output=str(output),dataset_sources=sources,
        case_counts={k:len(v['jobs']) for k,v in plans.items()},
        timed_requests=sum(j['repeats'] for j in plans['timing']['jobs']),
        timed_output_tokens=sum(j['repeats']*j['output_tokens'] for j in plans['timing']['jobs']),
        git_head=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(PROJECT),'status','--short'],text=True),
        source_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fingerprint_paths})
    summary['plan_sha256']={name:hashlib.sha256((output/name/'plan.json').read_bytes()).hexdigest() for name in plans}
    save(output/'summary.json',summary)
    print('STUDY plan',summary['case_counts'],'timed requests',summary['timed_requests'],'timed output tokens',summary['timed_output_tokens'],flush=True)
    if args.prepare_only:
        print('RESULT prepared output',output,flush=True);return 0
    return execute_study(output,summary,list(plans),args)


if __name__=='__main__':
    multiprocessing.set_start_method('spawn',force=True)
    raise SystemExit(main())
