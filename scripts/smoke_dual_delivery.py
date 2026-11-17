"""Exercise both modes through the actual root launcher, including HTTP and SSE."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
from smoke_delivery import run
from mellum_runtime_config import PROJECT
from mellum_submission_modes import ROWS,MODES

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=0)
 p.add_argument('--mode',choices=MODES);p.add_argument('--all-rows',action='store_true');a=p.parse_args()
 out=PROJECT/'results/dual_mode_smoke'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
 results={}
 for mode in ([a.mode] if a.mode else MODES):
  for row in (ROWS if a.all_rows else ['step0123']):
   key=mode+'_'+row;print('STAGE',key,'log',out/key/'server.log',flush=True)
   results[key]=run(row,a.gpu,out/key,300,route=mode)
   launchpath=out/key/'server/launch.json'
   if launchpath.exists():
    launch=json.loads(launchpath.read_text());expected={'mode':mode,'points':list(ROWS[row])}
    results[key]['configuration_identity']=launch.get('submission')
    if launch.get('submission')!=expected:results[key].update(status='failed',configuration_error='Mode/points differ from requested row')
   else:results[key].update(status='failed',configuration_error='Launch evidence missing')
   summary={'status':'smoke_passed' if all(r['status']=='smoke_passed' for r in results.values()) else 'smoke_failed','results':results}
   (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
   print(key,results[key]['status'],flush=True)
 print('RESULT',summary['status'],'output',out)
 return 0 if summary['status']=='smoke_passed' else 1
if __name__=='__main__':raise SystemExit(main())
