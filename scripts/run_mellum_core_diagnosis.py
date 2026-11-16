"""Small 2x2 expert/attention-stack diagnosis; no claim of public200 acceptance."""
import argparse,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_candidate_acceptance as a
from mellum_execution_profiles import CORE_PROFILES,PHASE_PROFILES
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write


def select(rows):
    names=set()
    for source,count in [('HumanEval',12),('supplied_scenario_a',8)]:
        group=sorted([r for r in rows if r['prompt_source']==source],key=lambda r:(len(r['input_ids'][0]),r['name']))
        if len(group)<count:raise ValueError('Insufficient diagnostic records')
        names.update(group[round(i*(len(group)-1)/(count-1))]['name'] for i in range(count))
    return [r for r in rows if r['name'] in names]


def prepare(out,gpu,profiles=CORE_PROFILES):
    m={'gpu':gpu,'scope':'20 fixed length/source-stratified diagnostics, not public200 acceptance','jobs':{},
       'source':str(Path(__file__).resolve()),'source_sha256':sha(__file__)}
    for name in profiles:
        d=out/name;d.mkdir();p=a.prepare(d,name,gpu)
        ref=json.loads(Path(p['references']['tokens']).read_text());ref['records']=select(ref['records'])
        f=d/'diagnostic_reference.json';write(f,ref)
        p['references']['tokens']=str(f);p['external_sha256'][str(f)]=sha(f)
        inp=json.loads((d/'tokens_inputs.json').read_text());keep={r['name'] for r in ref['records']}
        inp['records']=[r for r in inp['records'] if r['name'] in keep];write(d/'tokens_inputs.json',inp)
        p['input_sha256'][str(d/'tokens_inputs.json')]=sha(d/'tokens_inputs.json')
        p.update(diagnostic_batch=1,diagnostic_suites=['tokens']);write(d/'manifest.json',p)
        m['jobs'][name]={'manifest_sha256':sha(d/'manifest.json')}
    write(out/'manifest.json',m);return m


def current(out,m):
    if sha(m['source'])!=m['source_sha256']:raise ValueError('Diagnostic source changed')
    for name,v in m['jobs'].items():
        f=out/name/'manifest.json'
        if sha(f)!=v['manifest_sha256']:raise ValueError('Job source changed')
        a.current(json.loads(f.read_text()))


def execute(out,m,budget):
    deadline=time.monotonic()+budget;result={'status':'started','scope':m['scope'],'jobs':{}}
    for name in m['jobs']:
        current(out,m);remaining=deadline-time.monotonic()
        if remaining<60:result['status']='budget_limited';break
        d=out/name
        try:
            run_stage(name,['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(d)],d/'worker.log',timeout=min(600,remaining))
            p=json.loads((d/'manifest.json').read_text());r=a.token_comparison(json.loads(Path(p['references']['tokens']).read_text()),json.loads((d/'tokens_generation.json').read_text()))
            audit=json.loads((d/'candidate/fastpaths.json').read_text());load=json.loads((d/'candidate/loading_audit.json').read_text())
            expected='MellumPhaseMoeBlock' if name.startswith('expert_') else 'MellumSparseMoeBlock' if name in PHASE_PROFILES else 'MellumFusedMoeBlock' if name[6]=='n' else 'MellumSparseMoeBlock'
            prefill=('native_prefill' if name=='phase_prefill' else 'ragged_prefill') if name in PHASE_PROFILES else ('native_prefill' if name[5]=='n' else 'ragged_prefill')
            if audit['moe_classes']!={expected:28} or audit['prefill_functions']!={prefill:28} or load['successful_parts']!=5631:raise ValueError('Actual core paths differ')
            if name.startswith('phase_'):
                calls=[json.loads(x) for x in (d/'candidate/native_calls.jsonl').read_text().splitlines()]
                expected_decode='reference_ragged_decode' if name=='phase_prefill' else 'paged_attention'
                if not any(x.get('kind')=='attention' and x.get('phase')=='decode' and expected_decode in x.get('backend','') for x in calls):
                    raise ValueError('Actual attention phase did not match selected native/reference decode')
            if name.startswith('expert_'):
                calls=[json.loads(x) for x in (d/'candidate/native_calls.jsonl').read_text().splitlines()]
                for phase in ['prefill','decode']:
                    wanted='FusedMoE' if name=='expert_'+phase else 'MellumTorchExperts'
                    if not any(x.get('kind')=='moe_phase' and x.get('phase')==phase and x.get('backend')==wanted and x.get('reference_loaded_parts')==192 for x in calls):
                        raise ValueError('Expert phase switch or canonical weight copy audit differs')
            result['jobs'][name]={'status':'complete','tokens':r,'backend_audit':audit}
            print(name,'diagnostic_position_agreement',r['all']['position_match_fraction'],flush=True)
        except Exception:result['jobs'][name]={'status':'failed','traceback':traceback.format_exc()}
        write(out/'summary.json',result)
    if result['status']=='started':result['status']='diagnostic_complete' if all(v['status']=='complete' for v in result['jobs'].values()) else 'complete_with_failures'
    current(out,m);result['source_unchanged']=True
    result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--worker',type=Path)
    p.add_argument('--profiles',choices=CORE_PROFILES+PHASE_PROFILES,nargs='+',default=list(CORE_PROFILES))
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=25);v=p.parse_args()
    if v.worker:a.worker(v.worker,suites=['tokens'],batch_size=1);return
    if not 1<=v.budget_minutes<=45:p.error('Budget1..45min')
    out=Path('/workspace/results/mellum_core_diagnosis')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    m=prepare(out,v.gpu,v.profiles)
    if v.prepare_only:print('RESULT core_diagnosis_prepared output',out);return
    r=execute(out,m,v.budget_minutes*60);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
