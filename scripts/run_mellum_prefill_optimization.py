"""GPU1 bounded prefill candidate: R control, screen, D200, optional ABBA.

Run from the isolated worktree; never edits the GPU0 assessment checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from mellum_runtime_config import PROJECT
from run_mellum_batch_campaign import Runner, datasets, fingerprint, sha, write, worker
from run_mellum_moe_optimization import control, abba_plan, analyze_timing
from mellum_process_runner import run_stage

VARIANT='prefill_shared_elementwise'
PREVIOUS=Path('/workspace/results/mellum_moe_optimization/20261115T121305387577Z')


def prepare():
    data, provenance=datasets()
    pins=json.loads((PROJECT/'artifacts/performance/14_moe_routing_and_indexed_decode/campaign_manifest_compact.json').read_text())
    cached={}
    for job,cohorts in [('screen_shared_routing',('S32','L32')),('D200_shared_routing',('D200',))]:
        for c in cohorts:
            rel=f'{c}_r0.json';path=PREVIOUS/'jobs'/job/rel
            if hashlib.sha256(json.dumps(data[c],sort_keys=True).encode()).hexdigest()!=pins['jobs'][job]['plan']['cohorts'][c]['sha256']:
                raise ValueError('Cached R cohort inputs/budgets differ')
            digest=pins['jobs'][job]['artifacts'][rel]
            if sha(path)!=digest:raise ValueError('Cached R artifact changed')
            record=json.loads(path.read_text())
            if [r['name'] for r in record['records']]!=[r['name'] for r in data[c]]:
                raise ValueError('Cached R input order differs')
            cached[c]={'path':str(path),'sha256':digest}
    provenance['cached_shared_routing']=cached
    return data,provenance


def screen_ok(result):
    return bool(result and result['stable'] and result['batch_verified'] and all(
        result['cohorts'][c]['groups']['all']['token_match_fraction']>=.9 for c in ('S32','L32')))


def exact_to_cached(directory,cohort,item):
    if sha(item['path'])!=item['sha256']:raise ValueError('Cached R changed during experiment')
    previous=json.loads(Path(item['path']).read_text())['records']
    current=json.loads((directory/f'{cohort}_r0.json').read_text())['records']
    if [r['name'] for r in current]!=[r['name'] for r in previous]:raise ValueError('Output alignment differs')
    return {'records':len(current),'exact_sequences':sum(a['actual_ids']==b['actual_ids'] for a,b in zip(current,previous))}


def audit_backend(directory):
    rows=[json.loads(x) for x in (directory/'moe_calls.jsonl').read_text().splitlines()]
    pre=[r for r in rows if r['phase']=='prefill'];dec=[r for r in rows if r['phase']=='decode']
    if not pre or not dec or not any(r['requests']>1 for r in pre):return False
    return all(r['variant']==VARIANT and r['backend']==('requestwise_gemm_shared_elementwise' if r['phase']=='prefill' else 'requestwise_gemm_shared_route') for r in rows)


def execute(out,manifest,data,args,executor=run_stage):
    runner=Runner(out,manifest,args.budget_minutes*60,args.job_timeout,executor,Path(__file__).resolve())
    base={'gpu':manifest['gpu'],'engine_port':8126,'provenance':manifest['provenance'],
          'control':control('shared_routing'),'batch':32,'repeats':2,
          'cohorts':{c:data[c] for c in ('K4','S32','L32')},'observe':False,'moe_audit':True}
    summary={'status':'started','decisions':{},'scope':'Developer workload, no formal ABC or full HumanEval; inference defaults unchanged'}
    d=summary['decisions'];cached=manifest['provenance']['cached_shared_routing']
    try:
        reference=runner.job('screen_R',base)
        if not reference or not reference['stable'] or not reference['batch_verified']:
            summary['status']='reference_failed';return summary
        d['reference_replay']={c:exact_to_cached(out/'jobs/screen_R',c,cached[c]) for c in ('S32','L32')}
        if any(r['records']!=r['exact_sequences'] for r in d['reference_replay'].values()):
            summary['status']='reference_replay_differed';return summary
        candidate=runner.job('screen_candidate',{**base,'control':control(VARIANT)})
        if not screen_ok(candidate):
            summary['status']='screen_stopped';return summary
        if candidate['cohorts']['K4']['groups']['all']['token_match_fraction']<.9 or not audit_backend(out/'jobs/screen_candidate'):
            summary['status']='screen_execution_invalid';return summary
        full=runner.job('D200_candidate',{**base,'control':control(VARIANT),'cohorts':{'D200':data['D200']},'repeats':1})
        if not full or not full['batch_verified'] or not audit_backend(out/'jobs/D200_candidate'):
            summary['status']='D200_incomplete';return summary
        d['D200_groups']=full['cohorts']['D200']['groups']
        d['D200_vs_R']=exact_to_cached(out/'jobs/D200_candidate','D200',cached['D200'])
        if d['D200_groups']['all']['token_match_fraction']<.9:
            summary['status']='developer_position_below_90';return summary
        cohorts,modes=abba_plan(data,VARIANT)
        modes={k:('shared_routing' if v=='cpd' else v) for k,v in modes.items()}
        timing=runner.job('timing',{**base,'control':control(VARIANT),'cohorts':cohorts,'cohort_variants':modes,
                                   'repeats':1,'warmup':True,'moe_audit':False})
        if not timing:summary['status']='timing_incomplete';return summary
        # Existing analyzer labels A as CPD; here A is explicitly R.
        metrics=analyze_timing(out/'jobs/timing',timing)
        for row in metrics['cohorts'].values():
            row['R_seconds']=row.pop('cpd_seconds');row['R_median_seconds']=row.pop('cpd_median_seconds')
        summary['timing']=metrics
        summary['status']='candidate_measured' if metrics['performance_eligible'] else 'no_material_speedup'
        return summary
    finally:
        summary['jobs']={k:{x:v for x,v in j.items() if x!='plan'} for k,j in manifest['jobs'].items()}
        summary['budget_limited']=runner.limited;summary['blocked']=runner.blocked
        if runner.blocked:summary['status']='blocked_environment'
        elif runner.limited:summary['status']='partial_budget_limited'
        write(out/'summary.json',summary)
        (out/'report.md').write_text('# MoE prefill bounded candidate\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--worker',type=Path)
    p.add_argument('--budget-minutes',type=float,default=60);p.add_argument('--job-timeout',type=float,default=900)
    a=p.parse_args()
    if a.worker:worker(a.worker);return 0
    if a.gpu!=1:p.error('This isolated campaign is restricted to physical GPU1 while GPU0 runs ABC')
    if not 0<a.budget_minutes<=60 or a.job_timeout<=0:p.error('Positive budget <=60min required')
    data,prov=prepare();sources=fingerprint()
    digest=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    out=a.resume or Path('/workspace/results/mellum_prefill_optimization')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    expected={'gpu':a.gpu,'fingerprint':sources,'inputs_sha256':digest,'provenance':prov}
    if a.resume:
        m=json.loads((out/'manifest.json').read_text())
        if any(m[k]!=v for k,v in expected.items()):raise ValueError('Resume source/input/GPU differs')
    else:
        m={**expected,'max_workers':4,'jobs':{},'git_head':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()}
        write(out/'manifest.json',m);write(out/'inputs.json',data)
    if a.prepare_only:
        write(out/'schedule.json',{'gpu_not_started':True,'jobs_max':4,'GPU':1,'engine_port':8126,'budget_minutes':a.budget_minutes,
            'jobs':['screen_R','screen_candidate','conditional D200_candidate','conditional R/candidate ABBAABBA'],
            'screen_policy':'Each S32/L32 position >=90%, repeated stable, real batch32; no CPD-relative-drop gate',
            'D200_policy':'Overall position>=90%; sequence/length/source groups separately reported',
            'scope':'No HumanEval, official ABC, HTTP server, profiler or environment upgrades'})
        print('RESULT prefill_campaign_prepared output',out);return 0
    result=execute(out,m,data,a);print('RESULT',result['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
