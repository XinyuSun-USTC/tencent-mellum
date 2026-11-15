"""Bounded MoE optimization: cached HF, CPD control, ABBA and service regression."""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
import time
import traceback
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT
from run_mellum_batch_campaign import (Runner,datasets,fingerprint,sha,write,worker as engine_worker)
from mellum_process_runner import run_stage
from mellum_moe_micro import inputs as micro_inputs,worker as micro_worker

PREVIOUS=Path('/workspace/results/mellum_batch_campaign/20261115T043721420603Z')
VARIANTS=('shared_routing','indexed_decode')
MAX_WORKERS=12


def control(variant):
    return {'groups':'LE','phases':['prefill','decode'],'moe_optimization':variant}


def prepare():
    data,provenance=datasets()
    pins=json.loads((PROJECT/'artifacts/performance/13_batch_accuracy_campaign/campaign_manifest_compact.json').read_text())
    historical={}
    for rel in ('jobs/CPD/S32_r0.json','jobs/CPD/L32_r0.json','jobs/C_CPD/D200_r0.json'):
        _,job,artifact=rel.split('/')
        file=PREVIOUS/rel;expected=pins['jobs'][job]['artifacts'][artifact]
        if sha(file)!=expected:raise ValueError('Historical CPD output differs from archived checksum')
        historical[rel]={'path':str(file),'sha256':expected}
    provenance['historical_cpd']={'run':str(PREVIOUS),'git_head':pins['git_head'],'files':historical}
    return data,provenance,micro_inputs()


def screen_ok(result,reference):
    if not result or not reference or not result['stable'] or not result['batch_verified']:return False
    if 'K4' in result['cohorts'] and result['cohorts']['K4']['groups']['all']['token_match_fraction']<.9:return False
    return all(result['cohorts'][c]['groups']['all']['token_match_fraction']>=max(.9,reference['cohorts'][c]['groups']['all']['token_match_fraction']-.02)
               for c in ('S32','L32'))


def d200_ok(result,reference):
    if not result or not reference or not result['batch_verified']:return False
    a=result['cohorts']['D200']['groups'];b=reference['cohorts']['D200']['groups']
    return (a['all']['token_match_fraction']>=max(.9,b['all']['token_match_fraction']-.005)
            and all(a[c]['token_match_fraction']>=.9 for c in ('below_1024','at_least_1024')))


def abba_plan(data,variant):
    cohorts={};modes={}
    for i,letter in enumerate('ABBAABBA'):
        for name in ('S32','L32'):
            key=f'{name}_{letter}{i}';cohorts[key]=data[name];modes[key]='cpd' if letter=='A' else variant
    return cohorts,modes


def analyze_timing(directory,result):
    report={'cohorts':{},'stable':True,'batch_verified':result['batch_verified']}
    for cohort in ('S32','L32'):
        collected={'A':[],'B':[]};tokens={'A':[],'B':[]}
        for key,value in result['cohorts'].items():
            if not key.startswith(cohort+'_'):continue
            letter=key.split('_')[-1][0]
            collected[letter].extend(value['wall_seconds'])
            for file in value['repeat_files']:
                record=json.loads((directory/file).read_text())
                tokens[letter].append([r['actual_ids'] for r in record['records']])
        if any(len(collected[k])!=4 for k in ('A','B')):raise ValueError('Incomplete ABBA sequence')
        stable=all(all(t==values[0] for t in values[1:]) for values in tokens.values())
        report['stable']=report['stable'] and stable
        a,b=statistics.median(collected['A']),statistics.median(collected['B'])
        report['cohorts'][cohort]={'cpd_seconds':collected['A'],'candidate_seconds':collected['B'],
                                  'cpd_median_seconds':a,'candidate_median_seconds':b,'speedup':a/b,'stable':stable}
    speeds=[r['speedup'] for r in report['cohorts'].values()]
    report['geometric_mean_speedup']=(speeds[0]*speeds[1])**.5
    report['performance_eligible']=(report['stable'] and report['batch_verified'] and min(speeds)>=.95
                                    and report['geometric_mean_speedup']>=1.05)
    return report


def historical_match(directory,provenance,stage):
    files=provenance['historical_cpd']['files'];answer={}
    for cohort in ('D200',) if stage=='D200' else ('S32','L32'):
        key=f'jobs/C_CPD/{cohort}_r0.json' if cohort=='D200' else f'jobs/CPD/{cohort}_r0.json'
        previous=json.loads(Path(files[key]['path']).read_text())
        current=json.loads((directory/f'{cohort}_r0.json').read_text())
        old={r['name']:r['actual_ids'] for r in previous['records']}
        answer[cohort]={'exact_sequences':sum(r['actual_ids']==old[r['name']] for r in current['records']),
                        'records':len(current['records'])}
    return answer


def render(out,runner,decisions,timing):
    matrix=[]
    for name,result in runner.results.items():
        for cohort,value in result.get('cohorts',{}).items():
            group=value['groups']['all']
            matrix.append({'job':name,'cohort':cohort,**group,'mean_wall_seconds':statistics.mean(value['wall_seconds'])})
    if matrix:
        with (out/'matrix.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
    best=decisions.get('best')
    status=('blocked_environment' if runner.blocked else 'partial_budget_limited' if runner.limited else
            'complete' if best and decisions.get('service_passed') and decisions.get('profile_passed') else 'complete_with_unresolved')
    summary={'status':status,'decisions':decisions,'timing':timing,'matrix':matrix,
             'jobs':{name:{k:v for k,v in job.items() if k!='plan'} for name,job in runner.manifest['jobs'].items()},
             'scope':'Developer200 and fixed S32/L32 ABBA only; no official A/C or full HumanEval; defaults unchanged',
             'correctness_policy':{'small_cohort_min':.9,'small_allowed_drop':.02,'D200_min':.9,
                                   'D200_allowed_drop_vs_current_CPD':.005,'D200_each_length_group_min':.9},
             'performance_policy':{'min_geomean_speedup':1.05,'min_each_cohort_speedup':.95}}
    write(out/'summary.json',summary)
    lines=['# MoE候选自动实验','','状态：`'+status+'`。结果仅限本地开发负载，默认配置未切换。','',
           '| job | 集合 | 位置匹配 | 完整序列 | wall均值秒 |','| --- | --- | --- | --- | --- |']
    for row in matrix:lines.append(f"| {row['job']} | {row['cohort']} | {row['token_match_fraction']:.2%} | {row['exact_sequence_fraction']:.2%} | {row['mean_wall_seconds']:.3f} |")
    lines+=['','## 自动决策','','```json',json.dumps(decisions,ensure_ascii=False,indent=2),'```','',
            '## ABBA','','```json',json.dumps(timing,ensure_ascii=False,indent=2),'```','',
            'CPD只是本地正确性对照，不能用其较慢的耗时替代题目官方Baseline放大成绩。']
    (out/'report.md').write_text('\n'.join(lines)+'\n');return summary


def execute(out,manifest,data,args,executor=run_stage):
    runner=Runner(out,manifest,args.budget_minutes*60,args.job_timeout,executor,Path(__file__).resolve())
    base={'kind':'engine','gpu':manifest['gpu'],'provenance':manifest['provenance'],'control':control('cpd'),
          'batch':32,'repeats':2,'cohorts':{k:data[k] for k in ('K4','S32','L32')},'observe':False,'moe_audit':True}
    decisions={'skipped':[],'selected_d200':[],'passing_d200':[]};timing={}
    try:
        micro=runner.job('micro',{'kind':'micro','gpu':manifest['gpu'],'micro_inputs':manifest['micro_inputs'],
                                'provenance':manifest['provenance']})
        b0=runner.job('screen_B0',{**base,'control':{},'moe_audit':False})
        cpd=runner.job('screen_CPD',base)
        if cpd:decisions['historical_screen_match']=historical_match(out/'jobs/screen_CPD',manifest['provenance'],'screen')
        screens={}
        for variant in VARIANTS:
            if variant=='indexed_decode':
                record=next((r for r in micro.get('records',[]) if r['name']=='decode_32'),None) if micro else None
                if not record or not micro.get('variants',{}).get(variant,{}).get('valid'):
                    decisions['skipped'].append('indexed_decode: micro validation unavailable/failed');continue
                speeds=record['median_ms'];memory=record['peak_additional_allocated_bytes']
                if speeds['cpd']/speeds[variant]<1.05 and memory[variant]>=memory['cpd']*.5:
                    decisions['skipped'].append('indexed_decode: neither local speed nor allocation benefit');continue
            value=runner.job('screen_'+variant,{**base,'control':control(variant)})
            if value:screens[variant]=value
        # Revalidate refactored CPD on full200 before judging candidates against it.
        valid_cpd=(cpd and cpd['stable'] and cpd['batch_verified'] and all(cpd['cohorts'][c]['groups']['all']['token_match_fraction']>=.9 for c in ('S32','L32')))
        if valid_cpd:
            reference=runner.job('D200_CPD',{**base,'cohorts':{'D200':data['D200']},'repeats':1})
            if reference:decisions['historical_D200_match']=historical_match(out/'jobs/D200_CPD',manifest['provenance'],'D200')
            choices=[v for v,r in screens.items() if screen_ok(r,cpd)][:2]
            decisions['selected_d200']=choices
            for variant in choices:
                result=runner.job('D200_'+variant,{**base,'control':control(variant),'cohorts':{'D200':data['D200']},'repeats':1})
                if d200_ok(result,reference):decisions['passing_d200'].append(variant)
            for variant in decisions['passing_d200']:
                cohorts,modes=abba_plan(data,variant)
                result=runner.job('timing_'+variant,{**base,'control':control(variant),'cohorts':cohorts,
                      'cohort_variants':modes,'repeats':1,'warmup':True,'moe_audit':False})
                if result:
                    timing[variant]=analyze_timing(out/('jobs/timing_'+variant),result)
                    write(out/('timing_'+variant+'.json'),timing[variant])
            winners=[v for v,r in timing.items() if r['performance_eligible']]
            if winners:
                best=max(winners,key=lambda v:timing[v]['geometric_mean_speedup']);decisions['best']=best
                profile=runner.job('profile_'+best,{**base,'control':control(best),
                    'cohorts':{'S32_candidate':data['S32'],'L32_candidate':data['L32']},
                    'repeats':1,'profile':True,'warmup':True})
                decisions['profile_passed']=bool(profile and profile.get('trace'))
                http=runner.job('HTTP_'+best,{**base,'control':control(best),'cohorts':{}},http=True)
                decisions['service_passed']=bool(http and not http['review_required'])
            else:decisions['skipped'].append('Profile/HTTP: no candidate has eligible ABBA benefit')
        else:decisions['skipped'].append('Dependent D200/performance: current CPD screening failed')
    finally:summary=render(out,runner,decisions,timing)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--worker',type=Path)
    p.add_argument('--budget-minutes',type=float,default=90);p.add_argument('--job-timeout',type=float,default=1200)
    args=p.parse_args()
    if args.worker:
        plan=json.loads((args.worker/'plan.json').read_text())
        if plan['kind']=='micro':
            try:micro_worker(args.worker,plan)
            except Exception:
                previous=json.loads((args.worker/'results.json').read_text()) if (args.worker/'results.json').exists() else {}
                write(args.worker/'results.json',{**previous,'status':'failed','traceback':traceback.format_exc()});raise
        else:engine_worker(args.worker)
        return 0
    if not 0<args.budget_minutes<=90 or args.job_timeout<=0:p.error('Positive total budget <=90 minutes required')
    data,provenance,micro=prepare();sources=fingerprint();input_sha=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    out=args.resume or Path('/workspace/results/mellum_moe_optimization')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if any(manifest[k]!=value for k,value in [('fingerprint',sources),('inputs_sha256',input_sha),('provenance',provenance),('micro_inputs',micro),('gpu',args.gpu)]):
            raise ValueError('Resume source/input/runtime configuration differs')
    else:
        manifest={'version':1,'gpu':args.gpu,'created_utc':datetime.now(timezone.utc).isoformat(),
                  'git_head':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
                  'max_workers':MAX_WORKERS,'fingerprint':sources,'inputs_sha256':input_sha,'provenance':provenance,
                  'micro_inputs':micro,'jobs':{}}
        write(out/'manifest.json',manifest);write(out/'inputs.json',data)
    if args.prepare_only:
        write(out/'schedule.json',{'max_workers':12,'budget_seconds':args.budget_minutes*60,
              'stages':['micro','B0/CPD/shared/indexed screening (2 repeats)','CPD and <=2 candidates D200',
                        '<=2 candidates ABBAABBA S32/L32','winner real child trace and HTTP'],
              'cohorts':{k:len(v) for k,v in data.items()},'gpu_not_started':True})
        print('RESULT moe_campaign_prepared output',out);return 0
    result=execute(out,manifest,data,args)
    print('RESULT',result['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
