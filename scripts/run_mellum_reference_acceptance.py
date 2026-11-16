"""Complete reference-profile acceptance with explicit scheduler batch size.

Reuses verified HF outputs; generates every candidate record and scores all164
HumanEval tasks. No pass declaration is made before every check completes.
"""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_candidate_acceptance as acceptance
from run_mellum_batch_campaign import write
from mellum_perf_protocol import sha


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],required=True)
 p.add_argument('--batch-size',type=int,choices=[1,32],default=1)
 p.add_argument('--budget-minutes',type=float,default=120);p.add_argument('--prepare-only',action='store_true')
 p.add_argument('--worker',type=Path);a=p.parse_args()
 if a.worker:
  plan=json.loads((a.worker/'manifest.json').read_text())
  acceptance.worker(a.worker,suites=['humaneval','tokens','long'],batch_size=plan['reference_batch_size']);return
 if not 1<=a.budget_minutes<=120:p.error('Budget1..120min')
 out=Path('/workspace/results/mellum_reference_acceptance')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_batch'+str(a.batch_size))
 out.mkdir(parents=True);plan=acceptance.prepare(out,'r',a.gpu)
 plan.update(reference_batch_size=a.batch_size,mode='reference_acceptance',
             claim='Only a completed local_candidate_checks_passed result qualifies this tested configuration')
 plan['source_sha256'][str(Path(__file__).resolve())]=sha(__file__);write(out/'manifest.json',plan)
 if a.prepare_only:print('RESULT reference_acceptance_prepared output',out);return
 original=acceptance.run_stage
 def run(name,command,log,**kwargs):
  if name=='candidate':command=[sys.executable,str(Path(__file__).resolve()),'--gpu',str(a.gpu),'--worker',str(out)]
  return original(name,command,log,**kwargs)
 acceptance.run_stage=run
 result=acceptance.evaluate(out,plan,a.budget_minutes*60)
 result['reference_batch_size']=a.batch_size;write(out/'summary.json',result)
 print('RESULT',result['status'],'output',out,flush=True)
 if result['status']!='local_candidate_checks_passed':raise SystemExit(1)
if __name__=='__main__':main()
