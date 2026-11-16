"""Reference decode-expert request groups2/4; no native experts or math fusion."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_candidate_acceptance as acc
import run_mellum_abc_baseline as perf
from screen_mellum_reference_kv import perf_worker
from mellum_process_runner import run_stage
from mellum_perf_protocol import sha
from run_mellum_batch_campaign import write


def worker(out):
    plan=json.loads((out/'manifest.json').read_text());original=acc.runtime
    def runtime(profile,gpu,target):
        env,opts=original(profile,gpu,target)
        cfg=json.loads(env['MELLUM_BATCH_CAMPAIGN_CONFIG']);cfg['moe_audit']=str(target/'expert_groups.jsonl')
        env['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(cfg,sort_keys=True);os.environ.update(env)
        return env,opts
    acc.runtime=runtime
    acc.worker(out,suites=['tokens','long'],batch_size=32)


def inspect(out,plan,size):
    result={}
    for suite in ['tokens','long']:
        result[suite]=acc.token_comparison(json.loads(Path(plan['references'][suite]).read_text()),json.loads((out/(suite+'_generation.json')).read_text()))
    target=out/'candidate';load=json.loads((target/'loading_audit.json').read_text())
    if load!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise ValueError('Weight loading incomplete')
    rows=[json.loads(l) for l in (target/'expert_groups.jsonl').read_text().splitlines()]
    if not rows or not any(r['phase']=='decode' and r['requests']==32 for r in rows):raise ValueError('Real batch32 not observed')
    for r in rows:
        group=size if r['phase']=='decode' else 1
        if r['decode_group_size']!=group or r['expert_groups']!=(r['requests']+group-1)//group or r['backend']!='requestwise_gemm_shared_route':
            raise ValueError('Unexpected actual expert grouping')
    actual=json.loads((target/'fastpaths.json').read_text())
    if actual['moe_classes']!={'MellumSparseMoeBlock':28} or actual['norm_backend']!='reference' or actual['prefill_functions']!={'ragged_prefill':28}:
        raise ValueError('Reference backend changed')
    result.update(group_size=size,weight_audit=load,actual_paths=actual,actual_group_events=len(rows),
        developer200_gate=result['tokens']['all']['at_least_90_percent'])
    return result


def prepare(out,gpu,groups,with_perf):
    plans={}
    for size in groups:
        directory=out/f'group{size}';directory.mkdir();p=acc.prepare(directory,f'r_expert_decode{size}',gpu)
        p['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
        helper=Path(perf_worker.__code__.co_filename).resolve();p['source_sha256'][str(helper)]=sha(helper)
        write(directory/'manifest.json',p)
        plans[str(size)]={'path':str(directory/'manifest.json'),'sha256':sha(directory/'manifest.json')}
    result={'gpu':gpu,'groups':groups,'plans':plans,'with_perf':with_perf,
        'scope':'Decode expert grouping only; full developer200 and long. Optional reduced A32/B4 output32, not full acceptance or C200'}
    if with_perf:
        directory=out/'perf_template';directory.mkdir();m=perf.prepare(directory,gpu,False);m['stages'].pop('smoke');m['stages'].pop('C')
        m['stages']['A'][0].update(number=32,expected=m['stages']['A'][0]['expected'][:32]);m['stages']['B'][0]['number']=4
        result['perf_template']=m;write(directory/'manifest.json',m)
    write(out/'manifest.json',result);return result


def current(out,plan):
    for p in plan['plans'].values():
        if sha(p['path'])!=p['sha256']:raise ValueError('Group plan changed')
        acc.current(json.loads(Path(p['path']).read_text()))
    if plan['with_perf']:perf.current_matches(plan['perf_template'],out/'perf_template')


def execute(out,plan,budget):
    end=time.monotonic()+budget;summary={'status':'started','scope':plan['scope'],'candidates':{}};passed=[]
    def stage(name,args,log,cap):
        current(out,plan);remaining=end-time.monotonic()
        if remaining<30:raise TimeoutError('Screen budget exhausted')
        run_stage(name,[sys.executable,str(Path(__file__).resolve()),*args],log,timeout=min(cap,remaining));current(out,plan)
    try:
        for size in plan['groups']:
            directory=out/f'group{size}';p=json.loads((directory/'manifest.json').read_text())
            try:
                stage(f'group{size}',['--worker',str(directory)],directory/'worker.log',1800)
                value=inspect(directory,p,size);summary['candidates'][str(size)]=value
                if value['developer200_gate']:passed.append(size)
                print('GROUP',size,'position_agreement',value['tokens']['all']['position_match_fraction'],'16k',value['long']['16k']['position_match_fraction'],flush=True)
            except Exception:summary['candidates'][str(size)]={'status':'failed','traceback':traceback.format_exc()}
            write(out/'summary.json',summary)
        if passed and plan['with_perf']:
            jobs=[('R1','r')]+[(f'G{g}_1',f'r_expert_decode{g}') for g in passed]+[(f'G{g}_2',f'r_expert_decode{g}') for g in reversed(passed)]+[('R2','r')]
            summary['paired_perf']={}
            for name,profile in jobs:
                directory=out/'paired_perf'/name;directory.mkdir(parents=True)
                job={'stage':'reduced_expert_group','gpu':plan['gpu'],'profile':profile,
                    'cases':sum(plan['perf_template']['stages'].values(),[]),'timeout':min(900,max(1,end-time.monotonic())),
                    'server_port':8851+plan['gpu']*100,'engine_port':11426+plan['gpu']*100}
                write(directory/'plan.json',job)
                stage(name,['--perf-worker',str(directory)],directory/'worker.log',900)
                value=json.loads((directory/'results.json').read_text())
                if value['status']!='complete' or any(not v['metrics']['complete_without_failures'] for v in value['cases'].values()):raise ValueError('Reduced HTTP workload failed')
                summary['paired_perf'][name]=value;write(out/'summary.json',summary)
        summary.update(status='screen_passed_full_acceptance_pending' if passed else 'no_candidate_above_90',passing_groups=passed)
    except Exception:summary.update(status='failed',traceback=traceback.format_exc())
    finally:
        current(out,plan);summary['source_unchanged']=True
        summary['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
        write(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--groups',type=int,choices=[2,4],nargs='+',default=[2,4]);p.add_argument('--with-perf',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--perf-worker',type=Path);p.add_argument('--budget-minutes',type=float,default=60);a=p.parse_args()
    if a.worker:worker(a.worker);return
    if a.perf_worker:perf_worker(a.perf_worker);return
    if len(set(a.groups))!=len(a.groups) or not 1<=a.budget_minutes<=90:p.error('Unique groups and budget1..90min required')
    out=Path('/workspace/results/mellum_expert_group_screen')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan=prepare(out,a.gpu,a.groups,a.with_perf)
    if a.prepare_only:print('RESULT expert_groups_prepared output',out);return
    result=execute(out,plan,a.budget_minutes*60);print('RESULT',result['status'],'output',out)
if __name__=='__main__':main()
