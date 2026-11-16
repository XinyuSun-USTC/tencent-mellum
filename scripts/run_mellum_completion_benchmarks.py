"""Full ABC supplementation: tested primary policy or historical cumulative rows.

No result is synthesized. Primary measurement requires a completed acceptance.
Historical Step0 uses an unchanged tag checkout. All client A runs keep parallel32;
internal decode batch1 is allowed when the declared engine has one scheduler slot.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_abc_baseline as base
from mellum_runtime_config import PROJECT
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write


def worker(directory):
 plan=json.loads((directory/'plan.json').read_text());original=subprocess.Popen
 def launch(command,*args,**kwargs):
  if any(str(x).endswith('/serve_mellum.py') for x in command):
   common=['--gpu',str(plan['gpu']),'--server-port',str(plan['server_port']),'--engine-port',str(plan['engine_port']),'--output',str(directory/'server')]
   if plan['row']=='historical_step0':
    command=[sys.executable,str(PROJECT/'scripts/serve_mellum_step0_snapshot.py'),*common]
   elif plan['row']=='primary':
    command=[str(PROJECT/'start_server.sh'),'--mode','conservative','--max-num-seqs',str(plan['batch_size']),'--audit',*common]
  return original(command,*args,**kwargs)
 base.subprocess.Popen=launch
 try:base.stage_worker(directory)
 finally:base.subprocess.Popen=original
 # Re-read the actual client DB for the single-slot policies. Parallel32 is a
 # client workload parameter; the assessment does not require decode batch32.
 if plan['row'] in ('primary','historical_step0'):
  result=json.loads((directory/'results.json').read_text())
  for spec in plan['cases']:
   if spec['scenario']!='A':continue
   old=result['cases'][spec['name']]['metrics']
   measured=base.inspect_result(directory/spec['name']/'client','A',spec['number'],spec['expected'])
   measured.update(actual_decode_max_batch=old.get('actual_decode_max_batch'),engine_scheduler_slots=plan['batch_size'],
    validation_note='Client parallel32 preserved; actual decode batch32 is not required for a single-slot engine')
   result['cases'][spec['name']]['metrics']=measured
  write(directory/'results.json',result)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--track',choices=['primary','cumulative'])
 p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--acceptance-run',type=Path)
 p.add_argument('--budget-minutes',type=float,default=240);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path)
 p.add_argument('--wait-minutes',type=float,default=0);p.add_argument('--after-run',type=Path)
 a=p.parse_args()
 if a.worker:worker(a.worker);return
 if not a.track:p.error('--track required')
 if not 1<=a.budget_minutes<=480:p.error('Budget1..480min')
 if not 0<=a.wait_minutes<=180:p.error('Wait0..180min')
 batch=1
 prerequisite=a.acceptance_run if a.track=='primary' else a.after_run
 if prerequisite and a.wait_minutes>0:
  waiting_until=time.monotonic()+a.wait_minutes*60
  while not (prerequisite/'summary.json').exists():
   if time.monotonic()>waiting_until:raise TimeoutError('Prerequisite did not finish; no GPU benchmark started')
   print('WAIT prerequisite',prerequisite,flush=True);time.sleep(15)
 if a.track=='primary':
  if not a.acceptance_run:p.error('Primary requires completed --acceptance-run')
  checked=json.loads((a.acceptance_run/'summary.json').read_text())
  if checked['status']!='local_candidate_checks_passed':raise ValueError('Primary acceptance has not passed')
  manifest=json.loads((a.acceptance_run/'manifest.json').read_text())
  if manifest['profile']!='r':raise ValueError('Require reference profile')
  batch=manifest['reference_batch_size']
  for f,h in checked['artifact_sha256'].items():
   if sha(a.acceptance_run/f)!=h:raise ValueError('Acceptance result changed')
  rows=[('primary','r',batch)]
 else:
  from serve_mellum_step0_snapshot import describe
  describe(Path('/workspace/tencent-mellum-step0-repro'),a.gpu,'127.0.0.1',8951,10226,Path('/tmp/step0-preflight'))
  rows=[('historical_step0',None,1),('plus_1_native','sprint_clean',32),
        ('plus_1_2_graph_fusion','sprint_graph_qk_router',32),
        ('plus_1_2_3_cache','sprint_graph_qk_router_prefix_metadata',32)]
 out=Path('/workspace/results/mellum_completion_benchmarks')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+a.track)
 out.mkdir(parents=True);template=out/'template';template.mkdir();prepared=base.prepare(template,a.gpu,False);prepared['stages'].pop('smoke')
 source={str(f):sha(f) for f in [Path(__file__).resolve(),PROJECT/'scripts/serve_mellum_step0_snapshot.py',PROJECT/'scripts/serve_submission.py']}
 def current():
  base.current_matches(prepared,template)
  if any(sha(f)!=h for f,h in source.items()):raise ValueError('Supplement source changed')
 manifest={'track':a.track,'gpu':a.gpu,'rows':rows,'source_sha256':source,'template':prepared,
  'acceptance_run':str(a.acceptance_run) if a.acceptance_run else None,'jobs':{},
  'scope':'Full A512/B50/C200; explicit historical Step0 or acceptance-gated primary; not hidden-set median'}
 manifest['execution_order']=[r[0] for r in (rows[1:]+rows[:1] if a.track=='cumulative' else rows)]
 write(out/'manifest.json',manifest)
 if a.prepare_only:print('RESULT completion_benchmarks_prepared output',out);return
 end=time.monotonic()+a.budget_minutes*60
 for row,profile,slots in (rows[1:]+rows[:1] if a.track=='cumulative' else rows):
  for stage,cases in prepared['stages'].items():
   current();remaining=end-time.monotonic()
   if remaining<60:break
   directory=out/row/stage;directory.mkdir(parents=True)
   job={'row':row,'profile':profile,'batch_size':slots,'stage':stage,'cases':cases,'gpu':a.gpu,
    'server_port':8951+a.gpu*100,'engine_port':10226+a.gpu*100,'timeout':remaining}
   write(directory/'plan.json',job);key=row+'/'+stage;manifest['jobs'][key]={'status':'running'};write(out/'manifest.json',manifest)
   try:
    run_stage(key,[sys.executable,str(Path(__file__).resolve()),'--worker',str(directory)],directory/'stage.log',timeout=remaining)
    current();value=json.loads((directory/'results.json').read_text())
    valid=value['status']=='complete' and all(c['metrics']['complete_without_failures'] for c in value['cases'].values())
    manifest['jobs'][key]={'status':'complete' if valid else 'requests_failed','results':value}
   except Exception:manifest['jobs'][key]={'status':'failed','traceback':traceback.format_exc()}
   write(out/'manifest.json',manifest)
 current();expected=len(rows)*3
 status='complete' if len(manifest['jobs'])==expected and all(j['status']=='complete' for j in manifest['jobs'].values()) else 'incomplete_or_failed'
 summary={**manifest,'status':status,'source_unchanged':True,'artifact_sha256':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
 write(out/'summary.json',summary);print('RESULT',status,'output',out,flush=True)
if __name__=='__main__':main()
