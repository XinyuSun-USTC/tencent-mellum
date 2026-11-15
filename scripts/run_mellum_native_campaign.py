"""Independent native MoE/attention pilots on GPU0/1; no full ABC rerun.

Performance is natural greedy256 with ignore_eos as in assessment. Accuracy
uses normal greedy64/EOS and the fixed HF200. No bitwise/internal tensor gate.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import traceback
from mellum_runtime_config import PROJECT
from run_mellum_batch_campaign import Runner,datasets,fingerprint,write,sha,worker
from mellum_process_runner import run_stage


R={'groups':'LE','phases':['prefill','decode'],'moe_optimization':'shared_routing'}


def configuration(track,candidate):
    if not candidate:return dict(R),{}
    if track=='moe':return {'groups':'L','phases':['prefill','decode']},{'MELLUM_MOE_BACKEND':'fused_torch_route'}
    if track=='attention':return dict(R),{'MELLUM_ATTENTION_EXECUTION':'native',
        'MELLUM_SDPA_DECODE':'0','MELLUM_REQUEST_BATCHING':'0'}
    raise ValueError('Unknown track')


def performance_inputs(data):
    # Distinct request names, actual original token IDs. Fixed same inputs for R/candidate.
    return {'S1':[{**data['S32'][0],'budget':256}],
            'S32':[{**r,'budget':256} for r in data['S32']],
            'L32':[{**r,'budget':256} for r in data['L32']]}


def inspect_native(directory,track):
    rows=[json.loads(l) for l in (directory/'native_calls.jsonl').read_text().splitlines()]
    kind='moe' if track=='moe' else 'attention'
    selected=[r for r in rows if r['kind']==kind]
    if not selected or not any(r['phase']=='decode' and r['requests']==32 for r in selected):
        raise ValueError('No actual native decode32 execution evidence')
    if not any(r['phase']=='prefill' for r in selected):raise ValueError('No native prefill evidence')
    if track=='moe' and any(r['backend']!='FusedMoE' or r['quant_method']!='UnquantizedFusedMoEMethod' or not r.get('shuffled_w13') or not r.get('shuffled_w2') for r in selected):
        raise ValueError('Unexpected FusedMoE quantization/backend')
    if track=='attention' and any((r['backend']!='aiter_varlen' if r['phase']=='prefill' else 'paged_attention' not in r['backend']) for r in selected):
        raise ValueError('Reference attention fallback detected')
    return {'calls':len(selected),'backends':sorted({r['backend'] for r in selected}),
            'decode_max_requests':max(r['requests'] for r in selected if r['phase']=='decode')}


def performance_summary(reference,candidate):
    answer={}
    for name in ('S1','S32','L32'):
        a=reference['cohorts'][name]['wall_seconds'];b=candidate['cohorts'][name]['wall_seconds']
        if len(a)!=2 or len(b)!=2:raise ValueError('Performance requires two measured repetitions per configuration')
        answer[name]={'R_seconds':a,'candidate_seconds':b,'speedup':statistics.median(a)/statistics.median(b)}
    answer['batch_geomean_speedup']=(answer['S32']['speedup']*answer['L32']['speedup'])**.5
    # A pilot, not official scoring. Do not impose a relative-accuracy-drop policy.
    answer['speed_eligible']=answer['batch_geomean_speedup']>=1.10 and min(answer[n]['speedup'] for n in ('S32','L32'))>=.95
    return answer


def execute(out,m,data,args,executor=run_stage):
    runner=Runner(out,m,args.budget_minutes*60,args.job_timeout,executor,Path(__file__).resolve())
    base={'gpu':m['gpu'],'engine_port':8226 if m['track']=='moe' else 8326,'provenance':m['provenance'],
          'control':dict(R),'batch':32,'observe':False,'moe_audit':False,'repeats':2,
          'cohorts':performance_inputs(data),'cohort_windows':{'S1':1},'performance_only':True,
          'warmup':True,'native_audit':False}
    summary={'status':'started','track':m['track'],'scope':'Native developer pilot; no formal ABC/HumanEval acceptance, no defaults changed'}
    try:
        ref=runner.job('performance_R',base)
        if not ref or not ref['batch_verified']:
            summary['status']='reference_failed';return summary
        control,overrides=configuration(m['track'],True)
        candidate=runner.job('performance_candidate',{**base,'control':control,'runtime_overrides':overrides,'native_audit':True})
        if not candidate or not candidate['batch_verified']:
            summary['status']='candidate_execution_failed';return summary
        summary['native_execution']=inspect_native(out/'jobs/performance_candidate',m['track'])
        summary['generation_repeat_stability']={'R':ref.get('stable'),'candidate':candidate.get('stable')}
        summary['performance']=performance_summary(ref,candidate)
        if not summary['performance']['speed_eligible']:
            summary['status']='no_material_speedup';return summary
        accuracy=runner.job('D200_candidate',{**base,'control':control,'runtime_overrides':overrides,
            'cohorts':{'D200':data['D200']},'cohort_windows':{},'performance_only':False,
            'warmup':False,'native_audit':True,'repeats':1})
        if not accuracy or not accuracy['batch_verified']:
            summary['status']='accuracy_execution_failed';return summary
        summary['accuracy_native_execution']=inspect_native(out/'jobs/D200_candidate',m['track'])
        summary['D200_groups']=accuracy['cohorts']['D200']['groups']
        records=json.loads((out/'jobs/D200_candidate/D200_r0.json').read_text())
        if len(records['records'])!=200 or len(records['completions'])!=200:raise ValueError('D200 incomplete')
        summary['status']='candidate_eligible_for_combination' if summary['D200_groups']['all']['token_match_fraction']>=.9 else 'developer_position_below_90'
        return summary
    except Exception:
        summary.update(status='failed',traceback=traceback.format_exc())
        return summary
    finally:
        if runner.limited:summary['status']='partial_budget_limited'
        elif runner.blocked:summary['status']='blocked_environment'
        summary['jobs']={k:{x:v for x,v in j.items() if x!='plan'} for k,j in m['jobs'].items()}
        write(out/'summary.json',summary)
        (out/'report.md').write_text('# Native execution pilot\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['moe','attention']);p.add_argument('--gpu',type=int)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--budget-minutes',type=float,default=60);p.add_argument('--job-timeout',type=float,default=1800)
    a=p.parse_args()
    if a.worker:worker(a.worker);return 0
    if not a.track:p.error('--track required')
    expected_gpu=0 if a.track=='moe' else 1
    if a.gpu is None:a.gpu=expected_gpu
    if a.gpu!=expected_gpu:p.error('moe uses GPU0, attention uses GPU1 for parallel isolation')
    if not 0<a.budget_minutes<=60 or a.job_timeout<=0:p.error('Positive budget<=60min required')
    data,provenance=datasets();sources=fingerprint()
    digest=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    out=a.resume or Path('/workspace/results/mellum_native_'+a.track)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    expected={'gpu':a.gpu,'track':a.track,'fingerprint':sources,'inputs_sha256':digest,'provenance':provenance}
    if a.resume:
        m=json.loads((out/'manifest.json').read_text())
        if any(m[k]!=v for k,v in expected.items()):raise ValueError('Resume inputs/source/GPU/track differs')
    else:
        m={**expected,'max_workers':3,'jobs':{},'git_head':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()}
        write(out/'manifest.json',m);write(out/'inputs.json',data)
    if a.prepare_only:
        write(out/'schedule.json',{'GPU_not_started':True,'track':a.track,'gpu':a.gpu,'jobs_max':3,
            'jobs':['R greedy256 S1/S32/L32 x2','candidate same performance inputs x2','conditional D200 greedy64'],
            'budget_minutes':a.budget_minutes,'performance_policy':'batch geomean>=1.10x and each>=.95x; pilot only',
            'correctness_policy':'D200 aggregate positions>=90%, sequences/length/source groups reported separately',
            'comparison_limit':'Two repetitions, separate loaded processes; exploratory speedup, not paired ABBA or official score'})
        print('RESULT native_campaign_prepared output',out);return 0
    result=execute(out,m,data,a);print('RESULT',result['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
