"""Last full200/long confirmation: selected phase candidate or reference batching."""
import argparse,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_candidate_acceptance as a
from mellum_execution_profiles import PROFILES
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write


def prepare(out,profile,gpu,batch):
    p=a.prepare(out,profile,gpu);p.update(confirmation_batch=batch,confirmation_suites=['tokens','long'],
        orchestrator=str(Path(__file__).resolve()),orchestrator_sha256=sha(__file__))
    write(out/'manifest.json',p);return p


def current(p):
    a.current(p)
    if sha(p['orchestrator'])!=p['orchestrator_sha256']:raise ValueError('Confirmation source changed')


def execute(out,p,budget):
    r={'status':'started','profile':p['profile'],'batch':p['confirmation_batch'],
       'scope':'Original full developer200 and long group; not HumanEval or throughput acceptance'}
    try:
        current(p)
        run_stage('full200_long',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(out)],out/'worker.log',timeout=budget)
        current(p)
        for suite in ['tokens','long']:
            r[suite]=a.token_comparison(json.loads(Path(p['references'][suite]).read_text()),json.loads((out/(suite+'_generation.json')).read_text()))
        load=json.loads((out/'candidate/loading_audit.json').read_text())
        if load!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise ValueError('Loading audit failed')
        r['runtime']=json.loads((out/'runtime.json').read_text());r['batch_evidence']=json.loads((out/'batch_evidence.json').read_text())
        if r['batch_evidence']['decode']['max_batch_size']!=p['confirmation_batch']:raise ValueError('Actual decode batch differs from confirmation setting')
        if p['profile'].startswith('expert_'):
            events=[json.loads(x) for x in (out/'candidate/native_calls.jsonl').read_text().splitlines()]
            for phase in ['prefill','decode']:
                expected='FusedMoE' if p['profile']=='expert_'+phase else 'MellumTorchExperts'
                if not any(x.get('kind')=='moe_phase' and x.get('phase')==phase and x.get('backend')==expected and x.get('reference_loaded_parts')==192 for x in events):raise ValueError('Actual expert phase audit differs')
        r['status']='full200_agreement_passed' if r['tokens']['all']['at_least_90_percent'] else 'full200_agreement_below_90'
    except Exception:r.update(status='failed',traceback=traceback.format_exc())
    current(p);r['source_unchanged']=True
    r['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
    write(out/'summary.json',r);return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1],default=0)
    parser.add_argument('--profile',choices=PROFILES);parser.add_argument('--batch',type=int,choices=[1,32],default=1)
    parser.add_argument('--worker',type=Path);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--budget-minutes',type=float,default=30);v=parser.parse_args()
    if v.worker:
        p=json.loads((v.worker/'manifest.json').read_text());current(p)
        a.worker(v.worker,suites=p['confirmation_suites'],batch_size=p['confirmation_batch']);return
    if not v.profile:parser.error('--profile required')
    if not 1<=v.budget_minutes<=60:parser.error('Budget1..60min')
    out=Path('/workspace/results/mellum_final_greedy_confirmation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    p=prepare(out,v.profile,v.gpu,v.batch)
    if v.prepare_only:print('RESULT confirmation_prepared output',out);return
    r=execute(out,p,v.budget_minutes*60);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
