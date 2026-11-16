"""Bounded greedy200 diagnostics: no HF regeneration, HE scoring or full ABC."""
import argparse,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_candidate_acceptance as acceptance
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from mellum_execution_profiles import profile_settings
from run_mellum_full_campaign import audit_backend
from run_mellum_batch_campaign import write
TRACKS={'norm':[(n,n,32) for n in ['sprint_graph_qk_router_prefix_metadata','agree_qk_round','agree_norms','agree_norms_route']],
        'systems':[(n,n,32) for n in ['sprint_graph_qk_router_prefix_metadata','agree_no_prefix','agree_route','agree_projection']]+
                  [('single','sprint_graph_qk_router_prefix_metadata',1),('reference_control','r',1)]}


def prepare(out,track):
    jobs={};gpu=0 if track=='norm' else 1
    for name,profile,batch in TRACKS[track]:
        directory=out/name;directory.mkdir();plan=acceptance.prepare(directory,profile,gpu)
        plan.update(diagnostic_batch=batch,diagnostic_suites=['tokens','long']);write(directory/'manifest.json',plan)
        jobs[name]={'profile':profile,'batch':batch,'plan_sha256':sha(directory/'manifest.json')}
    m={'track':track,'jobs':jobs,'source':str(Path(__file__).resolve()),'source_sha256':sha(__file__),
       'scope':'Greedy200 and long regression only; no HE or throughput acceptance'}
    write(out/'manifest.json',m);return m


def current(out,m):
    if sha(m['source'])!=m['source_sha256']:raise ValueError('Recovery source changed')
    for name,job in m['jobs'].items():
        if sha(out/name/'manifest.json')!=job['plan_sha256']:raise ValueError('Job configuration changed')
        acceptance.current(json.loads((out/name/'manifest.json').read_text()))


def execute(out,m,budget):
    end=time.monotonic()+budget;result={'status':'started','scope':m['scope'],'jobs':{}}
    for name,job in m['jobs'].items():
        current(out,m)
        # Slow reference control runs only when no faster systems candidate passes90%.
        if name=='reference_control' and any(x.get('tokens',{}).get('all',{}).get('at_least_90_percent') for x in result['jobs'].values()):
            result['jobs'][name]={'status':'skipped_fast_candidate_passed'};continue
        remain=end-time.monotonic()
        if remain<60:result['status']='budget_limited';break
        directory=out/name
        try:
            run_stage(name,['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(directory)],
                directory/'worker.log',timeout=min(1200 if name=='reference_control' else 900,remain))
            (directory/'server').symlink_to(directory/'candidate',target_is_directory=True)
            audit_backend(directory,job['profile'],'greedy_diagnostic')
            actual=json.loads((directory/'candidate/fastpaths.json').read_text());wanted=profile_settings(job['profile'])
            norm=wanted.get('MELLUM_NORM_BACKEND','reference')
            if actual['norm_count']!=113 or actual['norm_backend']!=norm:raise ValueError('Actual norm backend differs')
            current(out,m);plan=json.loads((directory/'manifest.json').read_text());value={'status':'complete'}
            for suite in ['tokens','long']:
                value[suite]=acceptance.token_comparison(json.loads(Path(plan['references'][suite]).read_text()),json.loads((directory/(suite+'_generation.json')).read_text()))
            value['runtime']=json.loads((directory/'runtime.json').read_text())
            result['jobs'][name]=value
            print(name,'position_agreement',value['tokens']['all']['position_match_fraction'],
                'long',value['long']['all']['position_match_fraction'],'batch',job['batch'],flush=True)
        except Exception:result['jobs'][name]={'status':'failed','traceback':traceback.format_exc()}
        write(out/'summary.json',result)
    if result['status']=='started':result['status']='diagnostic_complete' if all(x['status'] in ['complete','skipped_fast_candidate_passed'] for x in result['jobs'].values()) else 'complete_with_failures'
    current(out,m);result['source_unchanged']=True
    result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=TRACKS);p.add_argument('--worker',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=45);a=p.parse_args()
    if a.worker:
        plan=json.loads((a.worker/'manifest.json').read_text())
        acceptance.worker(a.worker,suites=plan['diagnostic_suites'],batch_size=plan['diagnostic_batch']);return
    if not a.track:p.error('--track required')
    if not 1<=a.budget_minutes<=90:p.error('Budget1..90min')
    out=Path('/workspace/results/mellum_greedy_recovery')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+a.track);out.mkdir(parents=True)
    m=prepare(out,a.track)
    if a.prepare_only:print('RESULT greedy_recovery_prepared output',out);return
    r=execute(out,m,a.budget_minutes*60);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
