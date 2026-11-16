"""Compose only the separately screened decode-group4 and reference KV-once."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_candidate_acceptance as acc
import run_mellum_final_greedy_confirmation as confirm
import run_mellum_abc_baseline as perf
from screen_mellum_reference_kv import perf_worker
from screen_mellum_expert_groups import inspect as inspect_groups
from mellum_process_runner import run_stage
from mellum_perf_protocol import sha
from run_mellum_batch_campaign import write
PROFILE='r_expert_decode4_kv_once'
SCREENS=[Path('/workspace/results/mellum_reference_kv_screen/20261116T190628531443Z'),
         Path('/workspace/results/mellum_expert_group_screen/20261116T190635371969Z')]


def verify_screens():
    for root in SCREENS:
        value=json.loads((root/'summary.json').read_text())
        if value['status']!='screen_passed_full_acceptance_pending' or not value['source_unchanged']:raise ValueError('Prerequisite screen incomplete')
        for name,h in value['artifact_sha256'].items():
            if sha(root/name)!=h:raise ValueError('Prerequisite artifact changed')


def prepare(out,gpu,with_perf):
    verify_screens();plan=confirm.prepare(out,PROFILE,gpu,32)
    plan['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    helper=Path(perf_worker.__code__.co_filename).resolve();plan['source_sha256'][str(helper)]=sha(helper)
    group=SCREENS[1]/'group4';original=json.loads((SCREENS[1]/'summary.json').read_text())
    plan['group4_baseline']={}
    for suite in ['tokens','long']:
        file=group/(suite+'_generation.json');expected=original['artifact_sha256'][str(file.relative_to(SCREENS[1]))]
        if sha(file)!=expected:raise ValueError('Group4 comparison changed')
        plan['group4_baseline'][suite]=str(file);plan['external_sha256'][str(file)]=expected
    plan.update(with_perf=with_perf,scope='Combined group4 + reference KV once; full200/long and optional reduced A32/B4 output32. Full HumanEval and C200 pending.')
    if with_perf:
        template=out/'perf_template';template.mkdir();m=perf.prepare(template,gpu,False);m['stages'].pop('smoke');m['stages'].pop('C')
        m['stages']['B'][0]['number']=4;m['stages']['A'][0].update(number=32,expected=m['stages']['A'][0]['expected'][:32])
        plan['perf_template']=m;write(template/'manifest.json',m)
    write(out/'manifest.json',plan);return plan


def current(out,plan):
    confirm.current(plan)
    if plan['with_perf']:perf.current_matches(plan['perf_template'],out/'perf_template')


def worker(out):
    original=acc.runtime
    def runtime(profile,gpu,target):
        settings,opts=original(profile,gpu,target)
        cfg=json.loads(settings['MELLUM_BATCH_CAMPAIGN_CONFIG']);cfg['moe_audit']=str(target/'expert_groups.jsonl')
        settings['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(cfg,sort_keys=True);os.environ.update(settings)
        return settings,opts
    acc.runtime=runtime;acc.worker(out,suites=['tokens','long'],batch_size=32)


def execute(out,plan,budget):
    end=time.monotonic()+budget;original=confirm.run_stage
    def generation(name,cmd,log,**kwargs):
        return original(name,[sys.executable,str(Path(__file__).resolve()),'--worker',str(out)],log,**kwargs)
    confirm.run_stage=generation
    result=confirm.execute(out,plan,max(1,end-time.monotonic()))
    try:
        current(out,plan)
        if result['status']!='full200_agreement_passed':return result
        inspected=inspect_groups(out,plan,4);result['generation']=inspected
        events=[json.loads(l) for l in (out/'candidate/native_calls.jsonl').read_text().splitlines()]
        writes=[r for r in events if r.get('kind')=='kv_write']
        if {r['phase'] for r in writes}!={'prefill','decode'} or any(r.get('rope_backend')!='torch' or r.get('resident_inverse') for r in writes):raise ValueError('Combined KV receipt missing')
        result['kv_events']=writes
        for suite in ['tokens','long']:
            result[suite+'_vs_group4']=acc.token_comparison(json.loads(Path(plan['group4_baseline'][suite]).read_text()),json.loads((out/(suite+'_generation.json')).read_text()))['all']
        if plan['with_perf']:
            result['paired_perf']={}
            for name,profile in [('G1','r_expert_decode4'),('K1',PROFILE),('K2',PROFILE),('G2','r_expert_decode4')]:
                current(out,plan);remaining=end-time.monotonic()
                if remaining<30:raise TimeoutError('Combination screen budget exhausted')
                directory=out/'paired_perf'/name;directory.mkdir(parents=True)
                job={'stage':'reduced_combination','gpu':plan['gpu'],'profile':profile,
                    'cases':sum(plan['perf_template']['stages'].values(),[]),'timeout':min(900,remaining),
                    'server_port':9451+plan['gpu']*100,'engine_port':12026+plan['gpu']*100}
                write(directory/'plan.json',job)
                run_stage(name,[sys.executable,str(Path(__file__).resolve()),'--perf-worker',str(directory)],directory/'worker.log',timeout=job['timeout'])
                value=json.loads((directory/'results.json').read_text())
                if value['status']!='complete' or any(not c['metrics']['complete_without_failures'] for c in value['cases'].values()):raise ValueError('Reduced workload failed')
                result['paired_perf'][name]=value;write(out/'summary.json',result)
        result['status']='screen_passed_full_acceptance_pending'
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    finally:
        current(out,plan);result['scope']=plan['scope'];result['source_unchanged']=True
        result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
        write(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--with-perf',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=40)
    p.add_argument('--worker',type=Path);p.add_argument('--perf-worker',type=Path);a=p.parse_args()
    if a.worker:worker(a.worker);return
    if a.perf_worker:perf_worker(a.perf_worker);return
    if not 1<=a.budget_minutes<=60:p.error('Budget1..60min')
    out=Path('/workspace/results/mellum_reference_combination')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan=prepare(out,a.gpu,a.with_perf)
    if a.prepare_only:print('RESULT combination_prepared output',out);return
    result=execute(out,plan,a.budget_minutes*60);print('RESULT',result['status'],'output',out)
if __name__=='__main__':main()
