"""Single-GPU controlled reference decode screen: metadata and SWA groups2/4."""
import argparse,json,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import run_mellum_candidate_acceptance as acc
import run_mellum_final_greedy_confirmation as confirm
import run_mellum_c_service_diagnosis as service
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
CANDIDATES=('r_decode_meta','r_swa_decode2','r_swa_decode4')


def prepare(out,gpu):
    plan={'gpu':gpu,'source_sha256':{str(Path(__file__).resolve()):sha(__file__)},'generation':{},'timing':{}}
    for name in CANDIDATES:
        path=out/name/'generation';path.mkdir(parents=True);p=confirm.prepare(path,name,gpu,32)
        plan['generation'][name]={'path':str(path),'sha256':sha(path/'manifest.json')}
        plan['source_sha256'].update(p['source_sha256'])
    for label,name in [('R1','r_expert_decode4_kv_once'),*[(n,n) for n in CANDIDATES],('R2','r_expert_decode4_kv_once')]:
        path=out/'timing'/label;path.mkdir(parents=True)
        p=service.prepare(path,SimpleNamespace(track='decode',gpu=gpu,profile=name,batches=[8,32],tokens=64,observe_seconds=15))
        plan['timing'][label]={'path':str(path),'sha256':sha(path/'manifest.json')};plan['source_sha256'].update(p['source_sha256'])
    plan['scope']='D200/long screening and isolated decode8/32 output64; no HumanEval or formal C acceptance'
    write(out/'manifest.json',plan);return plan


def current(plan):
    for name,h in plan['source_sha256'].items():
        if sha(name)!=h:raise ValueError('Frozen source changed: '+name)
    for category in ['generation','timing']:
        for item in plan[category].values():
            if sha(Path(item['path'])/'manifest.json')!=item['sha256']:raise ValueError('Subplan changed')


def timing(item,remaining):
    target=Path(item['path'])
    run_stage(target.name,[sys.executable,str(Path(service.__file__).resolve()),'--track','decode','--decode-worker',str(target)],target/'worker.log',timeout=min(1200,remaining))
    value=json.loads((target/'decode_results.json').read_text())
    if value['status']!='isolated_decode_measured':raise ValueError('Timing incomplete')
    return value


def execute(out,plan,budget):
    end=time.monotonic()+budget;result={'status':'started','candidates':{},'timing':{},'scope':plan['scope']}
    try:
        current(plan);result['timing']['R1']=timing(plan['timing']['R1'],max(1,end-time.monotonic()))
        original=confirm.run_stage
        def generation(name,cmd,log,**kwargs):
            target=log.parent
            return original(name,[sys.executable,str(Path(__file__).resolve()),'--generation-worker',str(target)],log,**kwargs)
        confirm.run_stage=generation
        for name in CANDIDATES:
            current(plan);record={};result['candidates'][name]=record
            try:
                target=Path(plan['generation'][name]['path']);sub=json.loads((target/'manifest.json').read_text())
                value=confirm.execute(target,sub,min(900,max(1,end-time.monotonic())));record['generation']=value
                if value['status']!='full200_agreement_passed':record['status']='generation_gate_failed';continue
                if name!='r_decode_meta':
                    rows=[json.loads(x) for x in (target/'candidate/native_calls.jsonl').read_text().splitlines()]
                    actual=[r for r in rows if r.get('kind')=='swa_decode_groups' and r.get('group_size')==int(name[-1]) and r.get('group_calls',0)>0]
                    if not actual:raise ValueError('No actual grouped SWA observed')
                    record['actual_grouped_calls']=actual
                record['timing']=timing(plan['timing'][name],max(1,end-time.monotonic()));record['status']='screened_not_full_acceptance'
            except Exception:record.update(status='failed',traceback=traceback.format_exc())
            finally:current(plan);write(out/'summary.json',result)
        result['timing']['R2']=timing(plan['timing']['R2'],max(1,end-time.monotonic()))
        result['status']='decode_group_screen_complete'
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    current(plan);result['source_unchanged']=True
    result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--budget-minutes',type=float,default=60);p.add_argument('--prepare-only',action='store_true');p.add_argument('--generation-worker',type=Path);a=p.parse_args()
    if a.generation_worker:acc.worker(a.generation_worker,suites=['tokens','long'],batch_size=32);return
    if not 10<=a.budget_minutes<=90:p.error('Budget10..90min')
    out=Path('/workspace/results/mellum_swa_decode_groups')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan=prepare(out,a.gpu)
    if a.prepare_only:print('RESULT prepared output',out);return
    r=execute(out,plan,a.budget_minutes*60);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
