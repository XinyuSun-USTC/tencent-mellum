"""Bounded GPU1 HTTP capacity diagnostic; never a formal A512/C200 score."""
import argparse,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_abc_baseline as base
from mellum_execution_profiles import PROFILES
from mellum_perf_protocol import load_rows,sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write


def select_c(planned,count=20):
    # Evenly spaced length ranks include shortest and longest, independently of outputs.
    ordered=sorted(planned,key=lambda x:(x['tokens'],x['id']))
    if len(ordered)<count:raise ValueError('Insufficient C records')
    return [ordered[round(i*(len(ordered)-1)/(count-1))] for i in range(count)]


def prepare(out,gpu,profile,prefix_caching=False):
    m=base.prepare(out,gpu,True)
    native=json.loads((out/'inputs/adapter_manifest.json').read_text())
    a=native['scenarios']['A'];c=native['scenarios']['C']
    selected=select_c(c['planned_requests']);byid={r['id']:r for r in load_rows(c['source'])}
    source=out/'inputs/C_length_stratified20.jsonl'
    source.write_text(''.join(json.dumps(byid[r['id']],ensure_ascii=False)+'\n' for r in selected))
    m['stages'].update(A_small=[{'name':'A64','scenario':'A','number':64,'source':a['source'],'expected':a['planned_requests'][:64]}],
        C_small=[{'name':'C20','scenario':'C','number':20,'source':str(source),'expected':selected}])
    m.update(profile=profile,prefix_caching=prefix_caching,server_port=8251 if gpu==0 else 8151,engine_port=8926 if gpu==0 else 8726,
        claim='Diagnostic smoke/B2/A8/C2 then A64/concurrency32 and length-stratified C20/rate0.3. Not formal ABC, not a HumanEval acceptance result.',
        limitations=['Concurrent HF on another GPU may contend for host resources; final measurements require idle environment.',
                    'C20 is length-stratified, not the full C200 arrival/queue distribution.',
                    'No official throughput ratios or formal p99 acceptance inferred.'])
    m['additional_source_sha256']={}
    for name in ('run_mellum_candidate_capacity.py','mellum_execution_profiles.py'):
        p=base.PROJECT/'scripts'/name;m['additional_source_sha256'][str(p)]=sha(p)
    m['input_sha256'][str(source)]=sha(source)
    write(out/'manifest.json',m);return m


def current_matches(m,out):
    base.current_matches(m,out)
    if any(sha(p)!=v for p,v in m['additional_source_sha256'].items()):raise ValueError('Capacity/profile source changed')


def execute(out,m,budget,executor=run_stage):
    deadline=time.monotonic()+budget;result={'status':'started','scope':m['claim'],'limitations':m['limitations'],'stages':{}}
    for stage,cases in m['stages'].items():
        try:
            current_matches(m,out)
            remaining=deadline-time.monotonic()
            if remaining<60:result['status']='budget_limited';break
            directory=out/stage;directory.mkdir()
            timeout=min(900 if stage=='smoke' else 1200,remaining)
            write(directory/'plan.json',{'stage':stage,'gpu':m['gpu'],'cases':cases,'timeout':timeout,
                'profile':m['profile'],'server_port':m['server_port'],'engine_port':m['engine_port'],
                'prefix_caching':m.get('prefix_caching',False)})
            executor(stage,['/opt/venv/bin/python',str(base.PROJECT/'scripts/run_mellum_abc_baseline.py'),'--stage',str(directory)],
                directory/'stage.log',timeout=timeout)
            current_matches(m,out)
            value=json.loads((directory/'results.json').read_text());result['stages'][stage]=value
            # Every stage is a service/protocol gate, including partial-success loads.
            if value['status']!='complete' or not value['cases'] or any(not v['metrics']['complete_without_failures'] for v in value['cases'].values()):
                result['status']='service_or_capacity_gate_failed';break
            result['status']='diagnostic_complete'
        except Exception:
            result.update(status='stage_failed',traceback=traceback.format_exc());break
        finally:write(out/'summary.json',result)
    result['source_unchanged']=all(sha(p)==v for p,v in {**m['source_sha256'],**m['additional_source_sha256']}.items())
    if not result['source_unchanged']:result['status']='source_changed'
    result['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='summary.json'}
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--profile',choices=PROFILES,default='fused_moe');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--prefix-caching',action='store_true')
    p.add_argument('--budget-minutes',type=float,default=35);a=p.parse_args()
    if not 1<=a.budget_minutes<=45:p.error('Budget must be 1..45 minutes')
    if a.prefix_caching and a.profile not in ('r','fused_moe','fused_moe_native_attention','fused_moe_native_batched'):p.error('Prefix study currently supports r/fused_moe/fused_moe_native_attention only')
    out=Path('/workspace/results/mellum_candidate_capacity')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    m=prepare(out,a.gpu,a.profile,a.prefix_caching)
    if a.prepare_only:print('RESULT capacity_prepared output',out);return 0
    r=execute(out,m,a.budget_minutes*60);print('RESULT',r['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
