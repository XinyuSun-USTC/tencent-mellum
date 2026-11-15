"""One indexed-decode D200 run, judged by explicit developer position >=90%.

No micro gate, CPD-relative drop gate, HF regeneration, timing sweep or deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT
from run_mellum_batch_campaign import Runner,sha,write,fingerprint,worker
from run_mellum_moe_optimization import control
from validate_mellum_batch200 import prepare as prepare_hf
from mellum_process_runner import run_stage

PREVIOUS=Path('/workspace/results/mellum_moe_optimization/20261115T121305387577Z')
ARCHIVE=PROJECT/'artifacts/performance/14_moe_routing_and_indexed_decode'


def prepare():
    cases,provenance=prepare_hf();cases=[{**r,'budget':64} for r in cases]
    pinned=json.loads((ARCHIVE/'campaign_manifest_compact.json').read_text())
    # This follow-up measures exactly the already screened mathematical variant.
    # Entry-point/orchestration files may change; inference Python and runtime
    # config/checkpoint metadata must still match the completed MoE campaign.
    preserved={p:v for p,v in pinned['fingerprint'].items() if p.startswith(str(PROJECT/'ATOM')+'/')
               or p.startswith('/data/models/') or p==str(PROJECT/'scripts/mellum_runtime_config.py')}
    for path,value in preserved.items():
        if sha(path)!=value:raise ValueError('Screened candidate mathematical source differs: '+path)
    case_sha=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    if case_sha!=pinned['jobs']['D200_shared_routing']['plan']['cohorts']['D200']['sha256']:
        raise ValueError('D200 inputs/budgets differ from completed campaign')
    expected=control('indexed_decode')
    old=pinned['jobs']['screen_indexed_decode']['plan']['control']
    if old!=expected:raise ValueError('Indexed candidate configuration differs')
    baselines={}
    for name in ('CPD','shared_routing'):
        rel=f'jobs/D200_{name}/D200_r0.json';path=PREVIOUS/rel
        digest=pinned['jobs']['D200_'+name]['artifacts']['D200_r0.json']
        if sha(path)!=digest:raise ValueError('Cached baseline output hash differs')
        baseline=json.loads(path.read_text())
        if [r['name'] for r in baseline['records']]!=[c['name'] for c in cases]:raise ValueError('Baseline case order differs')
        baselines[name]={'path':str(path),'sha256':digest,'groups':baseline['groups']}
    provenance.update(screened_run=str(PREVIOUS),screened_git_head=pinned['git_head'],
                      preserved_source_sha256=preserved,baselines=baselines)
    return cases,provenance


def decision(groups):
    overall=groups['all']
    return {'developer_position_90_percent_passed':overall['token_match_fraction']>=.9,
            'developer_sequence_90_percent_passed':overall['exact_sequence_fraction']>=.9,
            'long_position_fraction':groups['at_least_1024']['token_match_fraction'],
            'short_position_fraction':groups['below_1024']['token_match_fraction'],
            'policy':'Only aggregate position >=90% determines this follow-up branch; sequence and length groups reported separately. No CPD-relative drop gate.',
            'scope':'Unconfirmed developer200, not official public-set or HumanEval acceptance'}


def execute(out,manifest,cases,args,executor=run_stage):
    runner=Runner(out,manifest,args.budget_minutes*60,args.job_timeout,executor,Path(__file__).resolve())
    plan={'gpu':manifest['gpu'],'provenance':manifest['provenance'],'control':control('indexed_decode'),
          'batch':32,'repeats':1,'cohorts':{'D200':cases},'observe':False,'moe_audit':True}
    summary={'status':'started','scope':'One normal greedy64 D200 run, frozen indexed candidate; no performance/functional acceptance claim'}
    try:
        value=runner.job('indexed_D200',plan)
        if value is None:
            summary['status']='blocked_environment' if runner.blocked else 'partial_budget_limited' if runner.limited else 'failed'
        elif not value['batch_verified']:
            summary.update(status='execution_invalid',reason='Requested actual decode batch32 not verified')
        else:
            run=json.loads((out/'jobs/indexed_D200/D200_r0.json').read_text())
            if len(run['records'])!=200 or len(run['completions'])!=200:raise ValueError('Incomplete D200')
            audit=[json.loads(line) for line in (out/'jobs/indexed_D200/moe_calls.jsonl').read_text().splitlines()]
            indexed=[r for r in audit if r['phase']=='decode']
            if not indexed or not any(r['requests']==32 for r in indexed) or any(
                    r['variant']!='indexed_decode' or r['backend']!='indexed_triton' for r in indexed):
                raise ValueError('Actual indexed decode backend/batch32 audit failed')
            policy=decision(run['groups']);comparisons={}
            for name,baseline in manifest['provenance']['baselines'].items():
                if sha(baseline['path'])!=baseline['sha256']:raise ValueError('Baseline changed during run')
                saved=json.loads(Path(baseline['path']).read_text());old={r['name']:r for r in saved['records']}
                rows=[]
                for row in run['records']:
                    reference=old[row['name']]
                    rows.append({'name':row['name'],'matching_positions_delta_vs_hf':row['matching_positions']-reference['matching_positions'],
                                 'output_exact_to_baseline':row['actual_ids']==reference['actual_ids']})
                comparisons[name]={'baseline_groups':baseline['groups'],'records':rows,
                    'position_percentage_point_change':100*(run['groups']['all']['token_match_fraction']-baseline['groups']['all']['token_match_fraction']),
                    'output_exact_sequences':sum(r['output_exact_to_baseline'] for r in rows),
                    'improved_cases':sum(r['matching_positions_delta_vs_hf']>0 for r in rows),
                    'regressed_cases':sum(r['matching_positions_delta_vs_hf']<0 for r in rows)}
            write(out/'baseline_comparison.json',comparisons)
            summary.update(status='developer_position_passed' if policy['developer_position_90_percent_passed'] else 'developer_position_below_90',
                           groups=run['groups'],decision=policy,baseline_comparison={k:{n:v for n,v in r.items() if n!='records'} for k,r in comparisons.items()},
                           indexed_decode_calls_verified=len(indexed),actual_batches=run['actual_batches'],loading_audit=value['loading_audit'],environment=value['environment'],
                           generation_wall_seconds=value['wall_seconds'],
                           note='One run; stable field is not repeated stability. Audit-enabled wall is not an ABBA speedup.')
    except Exception:
        summary.update(status='analysis_failed',traceback=traceback.format_exc());raise
    finally:
        summary['jobs']={k:{n:v for n,v in r.items() if n!='plan'} for k,r in manifest['jobs'].items()}
        write(out/'summary.json',summary)
        lines=['# Indexed decode：一次D200补证','','状态：`'+summary['status']+'`。默认配置未切换。','',
               '| 分组 | 位置匹配 | 完整序列 |','| --- | --- | --- |']
        for group,row in summary.get('groups',{}).items():
            if row['compared_positions']:lines.append(f"| {group} | {row['matching_positions']}/{row['compared_positions']} = {row['token_match_fraction']:.4%} | {row['exact_sequences']}/{row['records']} = {row['exact_sequence_fraction']:.4%} |")
        lines+=['','本轮仅按开发集总体token位置≥90%判该分支；完整序列、长短分层仍单列。未使用原CPD下降2pp/0.5pp筛选。',
                '没有新增微基准、ABBA、HTTP、16k或HumanEval，不等于官方验收或自动启用候选。']
        (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--budget-minutes',type=float,default=20);p.add_argument('--job-timeout',type=float,default=900)
    args=p.parse_args()
    if args.worker:worker(args.worker);return 0
    if not 0<args.budget_minutes<=20 or args.job_timeout<=0:p.error('Positive total budget <=20 minutes required')
    cases,provenance=prepare();sources=fingerprint()
    input_sha=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    out=args.resume or Path('/workspace/results/mellum_indexed_d200')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if any(manifest[k]!=v for k,v in [('gpu',args.gpu),('fingerprint',sources),('inputs_sha256',input_sha),('provenance',provenance)]):
            raise ValueError('Resume source/input/configuration differs')
    else:
        manifest={'version':1,'gpu':args.gpu,'git_head':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
                  'fingerprint':sources,'inputs_sha256':input_sha,'provenance':provenance,
                  'max_workers':2,'jobs':{},'note':'One job, no automatic retry; at most one explicit resume retry within cumulative budget'}
        write(out/'manifest.json',manifest);write(out/'inputs.json',{'cases':cases})
    if args.prepare_only:
        write(out/'schedule.json',{'GPU_jobs':1,'records':len(cases),'tokens_per_case':64,'batch':32,'gpu_not_started':True,
                                 'control':control('indexed_decode'),'policy':'Aggregate developer token positions >=90%; all strata and whole-sequence fraction also reported'})
        print('RESULT indexed_d200_prepared output',out);return 0
    result=execute(out,manifest,cases,args)
    print('RESULT',result['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
