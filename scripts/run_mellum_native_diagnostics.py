"""One bounded split study per GPU. No full ABC, no broad numerical search."""
import argparse,hashlib,json,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT
from run_mellum_batch_campaign import Runner,datasets,fingerprint,write,worker as engine_worker
from run_mellum_native_campaign import R
from mellum_moe_micro import inputs as micro_inputs
from mellum_process_runner import run_stage

MODES=('native_prefill','native_decode')


def phase_overrides(mode):
    if mode not in MODES:raise ValueError('Unknown split mode')
    return {'MELLUM_ATTENTION_EXECUTION':mode,'MELLUM_SDPA_DECODE':'0','MELLUM_REQUEST_BATCHING':'0'}


def verify_phases(directory,mode):
    rows=[json.loads(l) for l in (directory/'native_calls.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['kind']=='attention']
    pre=[r for r in rows if r['phase']=='prefill'];dec=[r for r in rows if r['phase']=='decode']
    if not pre or not dec or not any(r['requests']==32 for r in dec):raise ValueError('Split real batch32 evidence missing')
    if mode=='native_prefill':
        ok=all(r['backend']=='aiter_varlen' for r in pre) and all(r['backend']=='reference_ragged_decode' for r in dec)
    else:
        ok=all(r['backend']=='reference_ragged_prefill' for r in pre) and all('paged_attention' in r['backend'] for r in dec)
    if not ok:raise ValueError('Requested native/reference phases not actually executed')
    return {'prefill_backends':sorted({r['backend'] for r in pre}),'decode_backends':sorted({r['backend'] for r in dec})}


def execute(out,m,data,args,executor=run_stage):
    runner=Runner(out,m,args.budget_minutes*60,args.job_timeout,executor,Path(__file__).resolve())
    summary={'status':'started','track':m['track'],'scope':'Bounded diagnosis only; no formal ABC/HumanEval or default changes'}
    try:
        if m['track']=='moe':
            results={}
            for sorting in ('opus','ck'):
                value=runner.job('fixed_'+sorting,{'kind':'fixed_moe','gpu':0,'sorting':sorting,'micro_inputs':m['micro_inputs'],'provenance':m['provenance']})
                if value:results[sorting]={k:v for k,v in value.items() if k!='kernel_names'}
            summary['fixed_replay']=results
            summary['status']='diagnostic_complete' if len(results)==2 else 'diagnostic_incomplete'
            return summary
        base={'gpu':m['gpu'],'engine_port':8426,'provenance':m['provenance'],'control':dict(R),'batch':32,
              'observe':False,'moe_audit':False,'native_audit':True,'repeats':1,'warmup':False,'cohorts':{'D200':data['D200']}}
        passing=[];summary['phase_results']={}
        for mode in MODES:
            value=runner.job(mode+'_D200',{**base,'runtime_overrides':phase_overrides(mode)})
            if value and value['batch_verified']:
                path=out/'jobs'/(mode+'_D200')
                evidence=verify_phases(path,mode)
                records=json.loads((path/'D200_r0.json').read_text())
                if len(records['records'])!=200 or len(records['completions'])!=200:raise ValueError('D200 count incomplete')
                groups=value['cohorts']['D200']['groups']
                summary['phase_results'][mode]={'groups':groups,'execution':evidence}
                if groups['all']['token_match_fraction']>=.9:passing.append(mode)
        summary['passing_modes']=passing
        if not passing:
            summary['status']='no_split_candidate_above_90' if len(summary['phase_results'])==2 else 'split_incomplete'
            return summary
        # Only passing halves consume performance budget. Same 256 tokens, one repetition.
        perf={n:[{**r,'budget':256} for r in data[n]] for n in ('S32','L32')}
        ref=runner.job('performance_R',{**base,'cohorts':perf,'performance_only':True,'warmup':True,'native_audit':False})
        summary['performance']={}
        if ref and ref['batch_verified']:
            for mode in passing:
                val=runner.job(mode+'_performance',{**base,'runtime_overrides':phase_overrides(mode),'cohorts':perf,
                    'performance_only':True,'warmup':True})
                if val and val['batch_verified']:
                    summary['performance'][mode]={n:{'R_seconds':ref['cohorts'][n]['wall_seconds'][0],
                        'candidate_seconds':val['cohorts'][n]['wall_seconds'][0],
                        'speedup':ref['cohorts'][n]['wall_seconds'][0]/val['cohorts'][n]['wall_seconds'][0]} for n in perf}
        summary['status']='split_candidates_measured' if len(summary['performance'])==len(passing) else 'split_performance_incomplete'
        return summary
    except Exception:
        summary.update(status='failed',traceback=traceback.format_exc());return summary
    finally:
        if runner.blocked:summary['status']='blocked_environment'
        elif runner.limited:summary['status']='partial_budget_limited'
        summary['jobs']={k:{x:v for x,v in j.items() if x!='plan'} for k,j in m['jobs'].items()}
        write(out/'summary.json',summary)
        (out/'report.md').write_text('# Bounded native diagnosis\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['moe','attention']);p.add_argument('--worker',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--budget-minutes',type=float);p.add_argument('--job-timeout',type=float)
    a=p.parse_args()
    if a.worker:
        plan=json.loads((a.worker/'plan.json').read_text())
        if plan.get('kind')=='fixed_moe':
            from mellum_fused_repeat import worker
            worker(a.worker,plan)
        else:engine_worker(a.worker)
        return 0
    if not a.track:p.error('--track required')
    max_minutes=15 if a.track=='moe' else 45
    if a.budget_minutes is None:a.budget_minutes=max_minutes
    if a.job_timeout is None:a.job_timeout=600 if a.track=='moe' else 900
    if not 0<a.budget_minutes<=max_minutes or a.job_timeout<=0:p.error('Budget must be positive and <=track limit')
    data,prov=datasets();micro=micro_inputs() if a.track=='moe' else []
    sources=fingerprint()
    digest=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    out=a.resume or Path('/workspace/results/mellum_native_'+a.track+'_diagnosis')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    expected={'track':a.track,'gpu':0 if a.track=='moe' else 1,'fingerprint':sources,'inputs_sha256':digest,'provenance':prov,'micro_inputs':micro}
    if a.resume:
        m=json.loads((out/'manifest.json').read_text())
        if any(m[k]!=v for k,v in expected.items()):raise ValueError('Resume source/input/track differs')
    else:
        m={**expected,'jobs':{},'max_workers':2 if a.track=='moe' else 5,
           'git_head':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()}
        write(out/'manifest.json',m);write(out/'inputs.json',data)
    if a.prepare_only:
        write(out/'schedule.json',{'GPU_not_started':True,'budget_minutes':a.budget_minutes,'max_workers':m['max_workers'],
            'stages':['fixed layer0 inputs/routes x12: Opus sorting','same with CK sorting'] if a.track=='moe' else
                     ['native_prefill only D200','native_decode only D200','if any >=90%, R and passing halves greedy256 S32/L32'],
            'scope':'No whole-model MoE rerun, no candidate combination, no full ABC/HumanEval, no tolerance changes'})
        print('RESULT native_diagnosis_prepared output',out);return 0
    result=execute(out,m,data,a);print('RESULT',result['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
