"""Bounded single-command campaign: real batching, natural history, cached HF.

No inference backend defaults are changed. Execute workers only in the frozen
runtime; --prepare-only and orchestration tests require no GPU.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
from mellum_runtime_config import PROJECT, MODEL, apply_serving, engine_options
from mellum_process_runner import run_stage
from validate_mellum_batch200 import prepare, drive_window, summarize, read_batches

MAX_JOBS = 14
CONTROLS = {
    'CP': {'groups':'LE','phases':['prefill']},
    'CD': {'groups':'LE','phases':['decode']},
    'CPD': {'groups':'LE','phases':['prefill','decode']},
}


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    files=list((PROJECT/'ATOM/atom').rglob('*.py'))
    files += [PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'ATOM/atom/model_engine/model_runner.py',
              PROJECT/'ATOM/atom/model_engine/scheduler.py',PROJECT/'ATOM/atom/model_engine/llm_engine.py']
    files += [Path(__file__).resolve()]+[PROJECT/'scripts'/n for n in (
        'mellum_runtime_config.py','mellum_process_runner.py','validate_mellum_batch200.py',
        'validate_mellum_batch.py','validate_mellum_http.py','serve_mellum.py','profile_mellum.py','analyze_mellum_profile.py')]
    files += [p for p in [PROJECT/'scripts/run_mellum_moe_optimization.py',PROJECT/'scripts/mellum_moe_micro.py',PROJECT/'scripts/run_mellum_indexed_d200.py',PROJECT/'scripts/run_mellum_native_campaign.py',PROJECT/'scripts/run_mellum_native_diagnostics.py',PROJECT/'scripts/mellum_fused_repeat.py'] if p.exists()]
    files += [MODEL/'config.json', MODEL/'model.safetensors.index.json']
    return {str(p):sha(p) for p in sorted(set(files))}


def datasets():
    cases, provenance=prepare()
    from validate_mellum_batch import prepare as small
    k4=[]
    for c in small():
        k4.append({**c,'prompt_source':'cached_K4','source_id':c['name']})
        if sha(c['hf_source'])!=c['hf_sha256']: raise ValueError('K4 reference changed')
    short=[c for c in cases if c['prompt_source']=='HumanEval'][:32]
    long=[c for c in cases if len(c['input_ids'])>=1024]
    if len(short)!=32 or len(long)!=32: raise ValueError('Cohort differs')
    def limit(rows,n):return [{**r,'expected_ids':r['expected_ids'][:n],'budget':n} for r in rows]
    targets={short[i]['name'] for i in (2,13,21)}|{long[0]['name']}
    source=json.loads((Path(provenance['source_run'])/'hf/reference.json').read_text())
    probes={r['name']:r['generation_logits'][:8] for r in source['records'] if r['name'] in targets}
    for entries in probes.values():
        if len(entries)!=8:raise ValueError('Cached target logits missing')
        for item in entries:
            if sha(item['path'])!=item['sha256']:raise ValueError('Cached target logits checksum differs')
    provenance['probe_logits']=probes
    return {'S32':limit(short,16),'L32':limit(long,16),'K4':k4,
            'D200':limit(cases,64)},provenance


def score(result):
    return min(result['cohorts'][n]['groups']['all']['token_match_fraction'] for n in ('S32','L32'))


def eligible(result, baseline):
    if not result or not result.get('stable') or not result.get('batch_verified'):return False
    a=[result['cohorts'][n]['groups']['all']['token_match_fraction'] for n in ('S32','L32')]
    b=[baseline['cohorts'][n]['groups']['all']['token_match_fraction'] for n in ('S32','L32')]
    return all(x>=y for x,y in zip(a,b)) and any(x>y for x,y in zip(a,b))


def select_candidates(results, baseline):
    return [name for name in sorted(results,key=lambda n:(-score(results[n]),results[n]['wall_seconds']))
            if eligible(results[name],baseline)][:2]


def performance_records(cases, outputs):
    """Fixed-output timing records, deliberately not an HF accuracy score."""
    if len(cases)!=len(outputs) or any(len(o['token_ids'])!=r['budget'] for r,o in zip(cases,outputs)):
        raise ValueError('Performance request output budget/count differs')
    rows=[{'name':r['name'],'actual_ids':o['token_ids'],'input_tokens':len(r['input_ids']),
           'finish_reason':o.get('finish_reason')} for r,o in zip(cases,outputs)]
    return rows,{'performance_only':{'requests':len(rows),'output_tokens':sum(len(o['token_ids']) for o in outputs)}}


def worker(out):
    plan=json.loads((out/'plan.json').read_text())
    settings=apply_serving(plan['gpu'],out,True,'reference',True)
    overrides=plan.get('runtime_overrides',{})
    if set(overrides)-{'MELLUM_MOE_BACKEND','MELLUM_ATTENTION_EXECUTION','MELLUM_SDPA_DECODE','MELLUM_REQUEST_BATCHING'}:
        raise ValueError('Unsupported native runtime override')
    settings.update(overrides)
    if plan.get('native_audit'):settings['MELLUM_NATIVE_AUDIT_PATH']=str(out/'native_calls.jsonl')
    os.environ.update(settings)
    cfg=dict(plan['control'])
    if plan.get('observe'):cfg['observe']=str(out/'observe')
    if plan.get('cohort_variants'):
        cfg['moe_switch']=str(out/'moe_switch.json')
        write(out/'moe_switch.json',{'variant':cfg['moe_optimization']})
    if plan.get('moe_audit'):cfg['moe_audit']=str(out/'moe_calls.jsonl')
    if plan.get('timing_comparison'):
        cfg['runtime_switch']=str(out/'runtime_switch.json');write(out/'runtime_switch.json',{'enabled':True})
    cfg['capture_bytes']=int(plan.get('capture_bytes',768*1024**2))
    if cfg.get('groups') or cfg.get('observe'):
        settings['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(cfg);os.environ.update(settings)
    if plan.get('profile'):
        settings.update(MELLUM_PROFILE_CONTROL_DIR=str(out/'control'),MELLUM_PROFILE_LABELS='1')
        os.environ.update(settings)
    sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch, transformers
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require 80GiB free before model loading')
    if transformers.__version__!='5.2.0' or torch.__version__!=plan['provenance']['reference_environment']['torch']:
        raise RuntimeError('Frozen runtime differs')
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    required=max(len(r['input_ids'])+r['budget']+4 for cases in plan['cohorts'].values() for r in cases)
    options=engine_options(max(4192,required),None)
    options.update(torch_profiler_dir=str(out/'traces') if plan.get('profile') else None,max_num_batched_tokens=16384,max_num_seqs=plan['batch'],port=plan.get('engine_port',8026))
    result={'probe_logits':plan['provenance'].get('probe_logits',{}),'status':'started','control':cfg,'settings':settings,'engine_options':options,'cohorts':{},
            'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'hip':torch.version.hip,
                           'device':torch.cuda.get_device_name(0)}}
    write(out/'results.json',result);write(out/'observe_control.json',{});engine=None
    try:
        engine=EngineArgs(**options).create_engine()
        audit=json.loads((out/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:
            raise RuntimeError('Loading audit differs')
        result['loading_audit']=audit
        start=time.monotonic();stable=True;verified=True
        for cohort,cases in plan['cohorts'].items():
            actual_window=plan.get('cohort_windows',{}).get(cohort,plan['batch'])
            variant=plan.get('cohort_variants',{}).get(cohort,cfg.get('moe_optimization'))
            if plan.get('cohort_variants'):write(out/'moe_switch.json',{'variant':variant})
            if plan.get('warmup'):
                drive_window(engine,[{**r,'budget':2} for r in cases],lambda r:SamplingParams(temperature=0,max_tokens=2),actual_window,lambda *a:None)
            if plan.get('timing_comparison'):
                write(out/'runtime_switch.json',{'enabled':not cohort.endswith('_baseline')})
                drive_window(engine,[{**r,'budget':2} for r in cases],lambda r:SamplingParams(temperature=0,max_tokens=2),actual_window,lambda *a:None)
            repeats=[]
            for repeat in range(plan['repeats']):
                stage=f'{cohort}_r{repeat}';(out/'batches.jsonl').unlink(missing_ok=True)
                targets=set(plan.get('targets',[]))
                def submitted(mapping):
                    write(out/'observe_control.json',{str(k):{'stage':stage,'name':r['name'],'input_tokens':len(r['input_ids'])}
                          for k,r in mapping.items() if r['name'] in targets})
                def progress(done,*unused):
                    write(out/'progress.json',{'stage':stage,'completed':len(done),'total':len(cases),'elapsed':time.monotonic()-start})
                submitted({});t=time.monotonic()
                outputs,subs,comps=drive_window(engine,cases,lambda r:SamplingParams(temperature=0,max_tokens=r['budget'],ignore_eos=plan.get('performance_only',False)),
                                               actual_window,progress,submitted if plan.get('observe') else None)
                torch.cuda.synchronize();elapsed=time.monotonic()-t
                if plan.get('performance_only'):
                    rows,groups=performance_records(cases,outputs)
                else:rows,groups=summarize(cases,outputs)
                auditfile=out/(stage+'_batches.jsonl');auditfile.write_bytes((out/'batches.jsonl').read_bytes())
                batches=read_batches(auditfile)
                observed=batches['decode']['max_batch_size']
                if observed!=min(actual_window,len(cases)):verified=False
                record={'moe_variant':variant,'outputs':outputs,'control_enabled':not (plan.get('timing_comparison') and cohort.endswith('_baseline')),
                        'records':rows,'groups':groups,'actual_batches':batches,'wall_seconds':elapsed,
                        'submissions':subs,'completions':comps}
                write(out/(stage+'.json'),record);repeats.append(record)
                print(stage,'position_match',groups.get('all',{}).get('token_match_fraction','performance_not_scored'),'wall',round(elapsed,2),flush=True)
            stable=stable and all([r['actual_ids'] for r in v['records']]==[r['actual_ids'] for r in repeats[0]['records']] for v in repeats[1:])
            result['cohorts'][cohort]={'groups':repeats[0]['groups'],'repeat_files':[f'{cohort}_r{i}.json' for i in range(len(repeats))],
                                     'wall_seconds':[r['wall_seconds'] for r in repeats]}
            write(out/'results.json',result)
        result.update(status='complete',stable=stable,batch_verified=verified,wall_seconds=time.monotonic()-start)
        write(out/'results.json',result)
        # Timing above excludes profiler and probes in this branch. Collect a separate
        # short trace only after timings, within this same owned GPU worker.
        if plan.get('profile'):
            sample=plan['cohorts']['S32_candidate']
            write(out/'runtime_switch.json',{'enabled':True})
            from profile_mellum import wait_receipt
            from analyze_mellum_profile import analyze_file,render_markdown
            engine.start_profile();wait_receipt(out/'control','started',1,60)
            drive_window(engine,[{**r,'budget':2} for r in sample],lambda r:SamplingParams(temperature=0,max_tokens=2),
                         plan['batch'],lambda *a:None)
            engine.stop_profile();wait_receipt(out/'control','stopped',1,180)
            traces=list((out/'traces').rglob('*.pt.trace.json.gz'))
            if len(traces)!=1:raise RuntimeError('Expected one actual ModelRunner trace')
            analysis=analyze_file(traces[0])
            if not analysis['gpu_kernel_count'] or not analysis['module_annotation_count']:
                raise RuntimeError('Trace is missing GPU kernels or module labels')
            write(out/'trace_analysis.json',analysis)
            (out/'trace_analysis.md').write_text(render_markdown(analysis))
            if traces[0].stat().st_size>768*1024**2:
                result['trace_omitted_storage_cap']={'bytes':traces[0].stat().st_size,'sha256':sha(traces[0])}
                traces[0].unlink()
            else:result['trace']=str(traces[0].relative_to(out))
            write(out/'results.json',result)
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());write(out/'results.json',result);raise
    finally:
        if engine is not None:engine.close()


def difference(out, jobs, data):
    """Compare only actual identical histories, including the first fork step."""
    result={'scope':'bs1/bs32 natural histories, first 8 positions, repeat0; numerical differences are not acceptance gates',
            'pairs':[],'unresolved':[]}
    left=jobs.get('A1');right=jobs.get('A32')
    if not left or not right:return {**result,'unresolved':['Missing successful A1/A32']}
    import torch
    refs={r['name']:r for rows in data.values() for r in rows}
    for file in sorted((out/'jobs/A1/observe').glob('*_r0__*.json')):
        other=out/'jobs/A32/observe'/file.name
        if not other.exists():continue
        a=json.loads(file.read_text());b=json.loads(other.read_text())
        if a['history_sha256']!=b['history_sha256']:continue
        expected=refs[a['name']];history=expected['input_ids']+expected['expected_ids'][:a['step']]
        if hashlib.sha256(json.dumps(history).encode()).hexdigest()!=a['history_sha256']:continue
        av=torch.load(file.with_suffix('.pt'),map_location='cpu',weights_only=True)
        bv=torch.load(other.with_suffix('.pt'),map_location='cpu',weights_only=True)
        diffs=[]
        for name in av:
            if name not in bv:continue
            x,y=av[name].float(),bv[name].float()
            if x.shape!=y.shape:raise ValueError('Observed tensor shapes differ')
            if not torch.equal(x,y):
                diffs.append({'boundary':name,'max_abs':(x-y).abs().max().item(),
                              'relative_l2':((x-y).norm()/x.norm().clamp_min(1e-20)).item()})
        item={'first_relative_l2_over_1e_3':next((d for d in diffs if d['relative_l2']>1e-3),None),
              'name':a['name'],'step':a['step'],'phase':a['phase'],'history_sha256':a['history_sha256'],
              'first_nonexact':diffs[0] if diffs else None,'differences':diffs,
              'mapping_left':a,'mapping_right':b}
        if '29.logits' in av and '29.logits' in bv:
            x,y=av['29.logits'].flatten().float(),bv['29.logits'].flatten().float()
            values,indices=x.topk(2);vv,ii=y.topk(2)
            item['logits']={'left_top2':indices.tolist(),'left_values':values.tolist(),'left_margin':(values[0]-values[1]).item(),
                            'right_top2':ii.tolist(),'right_values':vv.tolist(),'right_margin':(vv[0]-vv[1]).item(),
                            'top1_flip':indices[0].item()!=ii[0].item(),'reference_token':expected['expected_ids'][a['step']]}
        entries=jobs['A1'].get('probe_logits',{}).get(a['name'])
        if entries and '29.logits' in av and '29.logits' in bv:
            saved=entries[a['step']]
            if sha(saved['path'])!=saved['sha256']:raise ValueError('HF probe changed during campaign')
            hf=torch.load(saved['path'],map_location='cpu',weights_only=True).float().flatten()
            for label,tensor in [('bs1',av['29.logits']),('bs32',bv['29.logits'])]:
                values=tensor.float().flatten();delta=values-hf;topv,topi=hf.topk(2)
                item[label+'_versus_hf']={'max_abs':delta.abs().max().item(),'mean_shift':delta.mean().item(),
                    'centered_rms':(delta-delta.mean()).square().mean().sqrt().item(),
                    'top1_exact':values.argmax().item()==hf.argmax().item(),'hf_top2':topi.tolist(),'hf_values':topv.tolist()}
        result['pairs'].append(item)
    if not result['pairs']:result['unresolved'].append('No same-HF-history probe pairs captured')
    result['unresolved'].append('Only selected target rows observed; absence of observed cache corruption is not a proof of all cache ownership')
    return result


def compare_cache(root):
    import torch
    pairs=[]
    for file in sorted(root.glob('*_b1_r0__*.json')):
        other=root/file.name.replace('_b1_r0__','_b32_r0__')
        if not other.exists():continue
        a,b=json.loads(file.read_text()),json.loads(other.read_text())
        if a['history_sha256']!=b['history_sha256']:continue
        av=torch.load(file.with_suffix('.pt'),weights_only=True,map_location='cpu')
        bv=torch.load(other.with_suffix('.pt'),weights_only=True,map_location='cpu')
        delta={}
        for key,value in av.items():
            if '.cache_' in key and key in bv:
                delta[key]={'exact':torch.equal(value,bv[key]),'max_abs':(value.float()-bv[key].float()).abs().max().item()}
        pairs.append({'name':a['name'],'step':a['step'],'cache':delta})
    return {'scope':'Only identical natural histories in target layer actual paged cache','pairs':pairs}


class Runner:
    def __init__(self,out,manifest,budget,job_timeout,executor=run_stage,worker_script=None):
        self.out=out;self.manifest=manifest;self.deadline=time.monotonic()+budget-manifest.get('used_seconds',0)
        self.job_timeout=job_timeout;self.executor=executor;self.limited=False;self.blocked=False
        self.results={};self.launched=0
        self.worker_script=worker_script or Path(__file__).resolve()

    def job(self,name,plan,http=False):
        if self.blocked:return None
        if self.manifest.get('fingerprint') and fingerprint()!=self.manifest['fingerprint']:
            self.blocked=True
            self.manifest['source_error']='Runtime/model source changed during campaign'
            write(self.out/'manifest.json',self.manifest)
            return None
        directory=self.out/'jobs'/name;directory.mkdir(parents=True,exist_ok=True)
        entry=self.manifest['jobs'].get(name)
        if entry and entry.get('status')=='complete':
            if entry['plan']!=plan:raise ValueError('Resume job configuration differs: '+name)
            if any(not (directory/p).exists() or sha(directory/p)!=v for p,v in entry['artifacts'].items()):
                raise ValueError('Resume artifact differs: '+name)
            value=json.loads((directory/entry['result_file']).read_text());self.results[name]=value;return value
        remaining=self.deadline-time.monotonic()
        if remaining<30 or self.manifest.get('started_workers',0)>=self.manifest.get('max_workers',MAX_JOBS):
            self.limited=True;return None
        attempt=1
        while (directory/f'attempt_{attempt}').exists():attempt+=1
        if (directory/'plan.json').exists():
            archive=directory/f'attempt_{attempt}';archive.mkdir()
            for p in list(directory.iterdir()):
                if p!=archive and not p.name.startswith('attempt_'):p.rename(archive/p.name)
        write(directory/'plan.json',plan)
        self.manifest['jobs'][name]={'status':'running','plan':plan,'attempt':attempt}
        self.manifest['started_workers']=self.manifest.get('started_workers',0)+1
        write(self.out/'manifest.json',self.manifest);self.launched+=1
        started=time.monotonic()
        with (self.out/'runner.log').open('a') as stream:stream.write(f'{datetime.now(timezone.utc).isoformat()} START {name} attempt {attempt}\n')
        command=['/opt/venv/bin/python',str(self.worker_script),'--worker',str(directory)]
        result_file='results.json'
        if http:
            write(directory/'control.json',plan['control']);result_file='http/summary.json'
            command=['/opt/venv/bin/python',str(PROJECT/'scripts/validate_mellum_http.py'),'--gpu',str(plan['gpu']),
                     '--max-seqs','4','--campaign-config',str(directory/'control.json'),'--output',str(directory/'http')]
        try:
            self.executor(name,command,directory/'worker.log',timeout=min(self.job_timeout,remaining))
            if self.manifest.get('fingerprint') and fingerprint()!=self.manifest['fingerprint']:
                self.blocked=True;raise ValueError('Source changed during worker')
            value=json.loads((directory/result_file).read_text())
            if value['status'] not in ('complete','http_smoke_completed'):raise ValueError('Worker incomplete')
            artifacts={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file() and 'attempt_' not in str(p.relative_to(directory))}
            self.manifest['jobs'][name].update(status='complete',artifacts=artifacts,result_file=result_file)
            self.results[name]=value;return value
        except Exception as exc:
            log=(directory/'worker.log').read_text() if (directory/'worker.log').exists() else ''
            if isinstance(exc,TimeoutError) and time.monotonic()>=self.deadline:self.limited=True
            fatal=any(x in log.lower() for x in ('out of memory','hip error','hsa_status_error','no visible gpu','require 80gib','frozen runtime differs','address already in use'))
            self.blocked=self.blocked or fatal
            self.manifest['jobs'][name].update(status='failed',error=str(exc),fatal_environment=fatal)
            return None
        finally:
            self.manifest['used_seconds']=self.manifest.get('used_seconds',0)+time.monotonic()-started
            write(self.out/'manifest.json',self.manifest)
            with (self.out/'runner.log').open('a') as stream:stream.write(f"{datetime.now(timezone.utc).isoformat()} END {name} {self.manifest['jobs'][name]['status']}\n")


def analyze(out,runner,decisions,first):
    rows=[]
    for name,result in runner.results.items():
        for cohort,value in result.get('cohorts',{}).items():
            score_=value['groups']['all'];rows.append({'job':name,'cohort':cohort,**score_,
                'wall_seconds':sum(value['wall_seconds'])/len(value['wall_seconds']),
                'stable':result['stable'],'batch_verified':result['batch_verified']})
    with (out/'matrix.csv').open('w') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    passing=[n for n,r in runner.results.items() if n.startswith('C_') and r.get('batch_verified') and
             r['cohorts']['D200']['groups']['all']['token_match_fraction']>=.9]
    status='blocked_environment' if runner.blocked else 'partial_budget_limited' if runner.limited else 'complete' if passing and decisions.get('regressions_complete') else 'complete_with_unresolved'
    summary={'status':status,'scope':'Developer-set campaign, not official public-set or HumanEval acceptance',
             'passing_developer200':passing,'decisions':decisions,'jobs':runner.manifest['jobs'],'matrix':rows,
             'capture_truncated':[str(p.relative_to(out)) for p in out.glob('jobs/*/observe/capture_truncated')],
             'remaining':'Official public-set interpretation, full HumanEval and formal A/C remain separate; no defaults changed'}
    write(out/'summary.json',summary);write(out/'first_difference.json',first)
    lines=['# 批量准确性自动实验批次', '',f'状态：`{status}`。开发集结果不等于官方验收；未更改默认配置。','',
           '| job | 集合 | 位置一致率 | 序列一致率 | 平均秒 |','| --- | --- | --- | --- | --- |']
    for r in rows:lines.append(f"| {r['job']} | {r['cohort']} | {r['token_match_fraction']:.2%} | {r['exact_sequence_fraction']:.2%} | {r['wall_seconds']:.2f} |")
    lines += ['', '## 分支与未解决项','', '```json',json.dumps(decisions,ensure_ascii=False,indent=2),'```','',
              '审计与探针开启的时间不能用于正式加速比。中间张量不精确仅作定位，不是新增验收门槛。',
              '控制若按请求调用算子，属于通用正确性策略，可能显著降低吞吐；必须用独立无探针计时判断代价。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def execute(out,manifest,data,args,executor=run_stage):
    runner=Runner(out,manifest,args.budget_minutes*60,args.job_timeout,executor)
    base={'gpu':manifest['gpu'],'provenance':manifest['provenance'],'control':{},'batch':32,'repeats':2,
          'cohorts':{k:data[k] for k in ('S32','L32')},'observe':False}
    targets=[data['S32'][i]['name'] for i in (2,13,21)]+[data['L32'][0]['name']]
    decisions={'budget_workers':14,'selected':[],'skipped':[], 'regressions_complete':False}
    first={}
    try:
        for bs in (1,2,8,32):runner.job('A'+str(bs),{**base,'batch':bs,'observe':bs in (1,32),'targets':targets if bs in (1,32) else []})
        a1=runner.results.get('A1');b0=runner.results.get('A32')
        first=difference(out,runner.results,data) if a1 and b0 else {'unresolved':['Missing A extremes']}
        a1ok=a1 and a1.get('stable') and a1.get('batch_verified') and all(a1['cohorts'][k]['groups']['all']['token_match_fraction']==1 for k in ('S32','L32'))
        if not a1ok:
            decisions['skipped'].append('Dependent shape controls: bs1 reference reproduction/stability not verified')
        elif not b0 or not b0.get('stable') or not b0.get('batch_verified'):
            decisions['skipped'].append('Dependent controls: bs32 stability or actual batch not verified')
        else:
            candidates={}
            for name,control in CONTROLS.items():
                value=runner.job(name,{**base,'control':control})
                if value:candidates[name]=value
            successful=[n for n in CONTROLS if n in candidates]
            phase=CONTROLS[max(successful,key=lambda n:score(candidates[n]))]['phases'] if successful else ['prefill','decode']
            decisions['isolation_phases']=phase
            for name,group in (('CL','L'),('CE','E')):
                control={'groups':group,'phases':phase};value=runner.job(name,{**base,'control':control})
                if value:candidates[name]=value
            # One preimplemented outside-L/E control, selected only from a witnessed
            # first boundary; no open-ended flag search.
            boundaries=[p['first_nonexact']['boundary'] for p in first.get('pairs',[]) if p.get('first_nonexact')]
            group='N' if any('norm' in n for n in boundaries) else 'H' if boundaries and all(n=='29.logits' for n in boundaries) else None
            if group:
                name='CX';control={'groups':'LE'+group,'phases':['prefill','decode']}
                value=runner.job(name,{**base,'control':control})
                if value:candidates[name]=value
            elif any('attention_context' in n for n in boundaries):
                layer=int(next(n.split('.')[0] for n in boundaries if 'attention_context' in n))
                cachecohorts={f'{k}_b{bs}':data[k] for bs in (1,32) for k in ('S32','L32')}
                cache=runner.job('CX_cache',{**base,'control':{'groups':'','phases':[],'cache_layers':[layer]},
                    'cohorts':cachecohorts,'cohort_windows':{k:(1 if '_b1' in k else 32) for k in cachecohorts},
                    'observe':True,'targets':targets,'repeats':1})
                decisions['cache_followup']={'layer':layer,'completed':bool(cache)}
                if cache:first['cache_followup']=compare_cache(out/'jobs/CX_cache/observe')
            else:decisions['skipped'].append('CX: first observed difference does not justify norm/head/cache control')
            choices=select_candidates(candidates,b0)
            if score(b0)>=.9:choices=['A32']+choices[:1]
            decisions['selected']=choices
            for name in choices:
                control={} if name=='A32' else runner.manifest['jobs'][name]['plan']['control']
                runner.job('C_'+name,{**base,'control':control,'cohorts':{'D200':data['D200']},'repeats':1})
            passing=[name for name in choices if (r:=runner.results.get('C_'+name)) and r['batch_verified'] and r['cohorts']['D200']['groups']['all']['token_match_fraction']>=.9]
            if passing:
                selected=max(passing,key=lambda n:runner.results['C_'+n]['cohorts']['D200']['groups']['all']['token_match_fraction'])
                control=runner.manifest['jobs']['C_'+selected]['plan']['control']
                decisions['regression_candidate']=selected
                timing=runner.job('D_timing_trace',{**base,'control':control,'cohorts':{'K4':data['K4'],'S32_baseline':data['S32'],'S32_candidate':data['S32']},'profile':True,'timing_comparison':True})
                http=runner.job('D_http',{**base,'control':control,'cohorts':{}},http=True)
                decisions['regressions_complete']=bool(timing and timing['stable'] and timing['batch_verified']
                    and timing['cohorts']['K4']['groups']['all']['token_match_fraction']>=.9 and http and not http['review_required'])
            else:decisions['skipped'].append('D: no batch32 candidate meets developer200 position threshold')
    finally:summary=analyze(out,runner,decisions,first)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--worker',type=Path)
    p.add_argument('--budget-minutes',type=float,default=90);p.add_argument('--job-timeout',type=float,default=1200)
    args=p.parse_args()
    if args.worker:worker(args.worker);return 0
    if not 0<args.budget_minutes<=90 or args.job_timeout<=0:p.error('Budget must be positive and <=90 minutes')
    data,provenance=datasets();sources=fingerprint()
    inputs_hash=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    out=args.resume or Path('/workspace/results/mellum_batch_campaign')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if (manifest['fingerprint']!=sources or manifest['inputs_sha256']!=inputs_hash or manifest['gpu']!=args.gpu
                or manifest['provenance']!=provenance):
            raise ValueError('Resume requires unchanged mathematical/runtime source, input and GPU')
    else:
        manifest={'git_head':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'gpu':args.gpu,'fingerprint':sources,
                  'inputs_sha256':inputs_hash,'provenance':provenance,'jobs':{},'max_workers':14,
                  'capture_max_bytes':3*1024**3,'budget_seconds':args.budget_minutes*60}
        write(out/'manifest.json',manifest);write(out/'inputs.json',data)
    if args.prepare_only:
        write(out/'schedule.json',{'A':[1,2,8,32],'B':['CP','CD','CPD','CL','CE','optional CX'],
              'C':'at most 2 improved stable candidates, D200 natural64','D':'passing candidate timing/trace+HTTP',
              'cohort_sizes':{k:len(v) for k,v in data.items()},'max_workers':14,'gpu_not_started':True})
        print('RESULT campaign_prepared output',out);return 0
    summary=execute(out,manifest,data,args)
    print('RESULT',summary['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
