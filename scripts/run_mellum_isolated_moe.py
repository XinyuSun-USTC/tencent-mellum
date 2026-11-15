"""One MoE reduction fix: fixed replay -> D200 -> conditional greedy256 timing."""
import argparse,hashlib,json,subprocess,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT
from run_mellum_batch_campaign import Runner,datasets,fingerprint,write,worker as engine_worker
from mellum_moe_micro import inputs as micro_inputs
from mellum_process_runner import run_stage
from run_mellum_native_campaign import R

CANDIDATE={'groups':'L','phases':['prefill','decode']}
OVERRIDES={'MELLUM_MOE_BACKEND':'fused_isolated'}


def replay_ok(value):
    return bool(value and len(value.get('records',[]))==3 and value.get('all_fixed_repeats_exact') and
                all(r['finite'] for r in value['records']))


def verify_backend(directory):
    rows=[json.loads(l) for l in (directory/'native_calls.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['kind']=='moe']
    if not rows or not any(r['phase']=='prefill' for r in rows) or not any(r['phase']=='decode' and r['requests']==32 for r in rows):
        raise ValueError('Isolated actual prefill/decode32 evidence missing')
    if any(r['backend']!='isolated_fused_experts' or r['native_topk']!=1 or r['logical_topk']!=8 or
           r['expanded_rows']!=r['rows']*8 or r['quant_method']!='UnquantizedFusedMoEMethod' or
           not r['shuffled_w13'] or not r['shuffled_w2'] for r in rows):
        raise ValueError('Unexpected expert combination or weight layout')
    return {'calls':len(rows),'native_topk':1,'logical_topk':8,'decode_max_requests':max(r['requests'] for r in rows if r['phase']=='decode')}


def execute(out,m,data,args,executor=run_stage):
    runner=Runner(out,m,args.budget_minutes*60,args.job_timeout,executor,Path(__file__).resolve())
    summary={'status':'started','scope':'Single default-off MoE candidate, no formal ABC/HumanEval acceptance'}
    base={'gpu':0,'engine_port':8526,'provenance':m['provenance'],'control':CANDIDATE,'batch':32,
          'observe':False,'moe_audit':False,'native_audit':True,'runtime_overrides':OVERRIDES,
          'repeats':1,'warmup':False,'cohorts':{'D200':data['D200']}}
    try:
        micro=runner.job('fixed_replay',{'kind':'fixed_moe','gpu':0,'sorting':'opus','variant':'isolated',
                            'micro_inputs':m['micro_inputs'],'provenance':m['provenance']})
        if not replay_ok(micro):
            summary.update(status='fixed_replay_stopped',replay_summary=micro);return summary
        summary['replay']={k:v for k,v in micro.items() if k!='kernel_names'}
        full=runner.job('D200_candidate',base)
        if not full or not full['batch_verified']:
            summary['status']='D200_incomplete';return summary
        summary['execution']=verify_backend(out/'jobs/D200_candidate')
        record=json.loads((out/'jobs/D200_candidate/D200_r0.json').read_text())
        if len(record['records'])!=200 or len(record['completions'])!=200:raise ValueError('D200 incomplete')
        summary['D200_groups']=full['cohorts']['D200']['groups']
        if summary['D200_groups']['all']['token_match_fraction']<.9:
            summary['status']='developer_position_below_90';return summary
        perf={'S1':[{**data['S32'][0],'budget':256}],
              **{n:[{**r,'budget':256} for r in data[n]] for n in ('S32','L32')}}
        plan={**base,'cohorts':perf,'cohort_windows':{'S1':1},'performance_only':True,'warmup':True}
        reference=runner.job('performance_R',{**plan,'control':dict(R),'runtime_overrides':{},'native_audit':False})
        candidate=runner.job('performance_candidate',plan) if reference and reference['batch_verified'] else None
        if not reference or not candidate or not candidate['batch_verified']:
            summary['status']='performance_incomplete';return summary
        summary['performance']={n:{'R_seconds':reference['cohorts'][n]['wall_seconds'][0],
            'candidate_seconds':candidate['cohorts'][n]['wall_seconds'][0],
            'speedup':reference['cohorts'][n]['wall_seconds'][0]/candidate['cohorts'][n]['wall_seconds'][0]} for n in perf}
        summary['batch_geomean_speedup']=(summary['performance']['S32']['speedup']*summary['performance']['L32']['speedup'])**.5
        eligible=summary['batch_geomean_speedup']>=1.10 and min(summary['performance'][n]['speedup'] for n in ('S32','L32'))>=.95
        summary['status']='candidate_measured' if eligible else 'correctness_screen_passed_no_material_speedup'
        return summary
    except Exception:
        summary.update(status='failed',traceback=traceback.format_exc());return summary
    finally:
        if runner.blocked:summary['status']='blocked_environment'
        elif runner.limited:summary['status']='partial_budget_limited'
        summary['jobs']={k:{x:v for x,v in j.items() if x!='plan'} for k,j in m['jobs'].items()}
        write(out/'summary.json',summary)
        (out/'report.md').write_text('# Isolated MoE reduction\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--budget-minutes',type=float,default=45);p.add_argument('--job-timeout',type=float,default=900)
    a=p.parse_args()
    if a.worker:
        plan=json.loads((a.worker/'plan.json').read_text())
        if plan.get('kind')=='fixed_moe':
            from mellum_fused_repeat import worker
            worker(a.worker,plan)
        else:engine_worker(a.worker)
        return 0
    if not 0<a.budget_minutes<=45 or a.job_timeout<=0:p.error('Positive budget<=45min required')
    data,prov=datasets();inputs=micro_inputs();sources=fingerprint()
    digest=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    out=a.resume or Path('/workspace/results/mellum_isolated_moe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True)
    expected={'gpu':0,'fingerprint':sources,'inputs_sha256':digest,'provenance':prov,'micro_inputs':inputs}
    if a.resume:
        m=json.loads((out/'manifest.json').read_text())
        if any(m[k]!=v for k,v in expected.items()):raise ValueError('Resume source/input differs')
    else:
        m={**expected,'max_workers':4,'jobs':{},'git_head':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()}
        write(out/'manifest.json',m);write(out/'inputs.json',data)
    if a.prepare_only:
        write(out/'schedule.json',{'GPU_not_started':True,'gpu':0,'max_workers':4,'budget_minutes':a.budget_minutes,
            'stages':['12-repeat fixed input/route replay','if finite and repeatable: D200 greedy64',
                      'if developer positions>=90%: R and candidate greedy256 S1/S32/L32'],
            'policy':'Fixed-repeat equality tests removal of observed repeat variation; no HF tensor-distance gate. No full ABC/HumanEval.'})
        print('RESULT isolated_moe_prepared output',out);return 0
    result=execute(out,m,data,a);print('RESULT',result['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
