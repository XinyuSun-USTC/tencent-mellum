"""Incremental Step0 A completion or controlled third-point accuracy isolation."""
import argparse,json,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
import run_mellum_abc_baseline as perf
import run_mellum_candidate_acceptance as acc

OLD=Path('/workspace/results/mellum_delivery_matrix/20261117T003512204779Z')
CASES={
 'control':('delivery_step012',{}),
 'swa_meta':('r_swa_decode4',{}),
 'prefix':('r_prefix_only',{}),
 'chunk':('r_chunk1024_d4',{}),
 'prefix_chunk':('r_chunk1024_d4',{'MELLUM_REFERENCE_PREFIX':'1','MELLUM_PREFIX_ENABLED':'1'}),
 'full':('delivery_step0123',{}),
}

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def verify(plan):
 for p,h in plan['sha256'].items():
  if sha(p)!=h:raise ValueError('Frozen source changed: '+p)

def prepare(out,track,gpu):
 old=json.loads((OLD/'manifest.json').read_text());equivalent={}
 for path,h in old['source_sha256'].items():
  p=Path(path)
  try:relative=p.relative_to('/workspace/tencent-mellum-delivery-matrix-fix')
  except ValueError:continue
  new=PROJECT/relative
  if sha(new)!=h:raise ValueError('Runtime differs from prior matrix: '+str(relative))
  equivalent[str(relative)]=h
 plan={'track':track,'gpu':gpu,'sha256':{**perf.sources(),str(Path(__file__).resolve()):sha(__file__)},
       'runtime_equivalence_to_old_matrix':equivalent,'prior_run':str(OLD),
       'scope':'A-only full rerun or token-only numerical diagnosis; no reuse of truncated A work as completed requests'}
 if track=='step0-a':
  previous=json.loads((OLD/'performance/manifest.json').read_text())
  d=out/'A';d.mkdir();spec=previous['template']['stages']['A']
  write(d/'plan.json',{'stage':'A','gpu':gpu,'profile':'delivery_step0','cases':spec,
       'timeout':14400,'server_port':8651+gpu*100,'engine_port':10226+gpu*100})
  plan['retained_evidence']={}
  for name in ['performance/delivery_step0/B/results.json','performance/delivery_step0/C/results.json','step0/summary.json']:
   f=OLD/name;plan['retained_evidence'][str(f)]=sha(f)
 else:
  for name,(profile,overrides) in CASES.items():
   d=out/name;d.mkdir();case=acc.prepare(d,profile,gpu)
   case['diagnostic_overrides']=overrides
   case['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
   write(d/'manifest.json',case)
 write(out/'manifest.json',plan);return plan

def accuracy_worker(out):
 case=json.loads((out/'manifest.json').read_text());original=acc.runtime
 def runtime(profile,gpu,target):
  import os
  settings,opts=original(profile,gpu,target)
  settings.update(case['diagnostic_overrides']);os.environ.update(settings)
  opts['enable_prefix_caching']=settings.get('MELLUM_PREFIX_ENABLED')=='1'
  return settings,opts
 acc.runtime=runtime
 acc.worker(out,suites=['tokens','long'],batch_size=32)

def execute(out,plan):
 verify(plan);result={'track':plan['track'],'cases':{},'source_unchanged':False}
 names=['A'] if plan['track']=='step0-a' else list(CASES)
 for name in names:
  d=out/name
  try:
   if plan['track']=='step0-a':
    for f,h in plan['retained_evidence'].items():
     if sha(f)!=h:raise ValueError('Prior evidence changed')
    cmd=[sys.executable,str(PROJECT/'scripts/run_mellum_abc_baseline.py'),'--stage',str(d)]
    run_stage('step0_A_only',cmd,d/'stage.log',timeout=14500)
    row=json.loads((d/'results.json').read_text());result['cases'][name]=row
   else:
    run_stage(name,[sys.executable,str(Path(__file__).resolve()),'--gpu',str(plan['gpu']),'--track','accuracy','--worker',str(d)],d/'worker.log',timeout=1200)
    case=json.loads((d/'manifest.json').read_text());acc.current(case)
    row={}
    for suite in ['tokens','long']:
     row[suite]=acc.token_comparison(json.loads(Path(case['references'][suite]).read_text()),json.loads((d/(suite+'_generation.json')).read_text()))
    row['profile']=case['profile'];row['overrides']=case['diagnostic_overrides']
    row['differences_from_control']={}
    for suite in ['tokens','long']:
     control=out/'control'/(suite+'_generation.json')
     if control.exists():
      left=json.loads(control.read_text())['records'];right=json.loads((d/(suite+'_generation.json')).read_text())['records']
      changes=[]
      for x,y in zip(left,right):
       if x['name']!=y['name']:raise ValueError('Control record identity differs')
       a,b=x['generated_ids'][0],y['generated_ids'][0]
       if a!=b:
        first=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v),min(len(a),len(b)))
        changes.append({'name':x['name'],'input_tokens':len(x['input_ids'][0]),'first_difference_0based':first})
      row['differences_from_control'][suite]=changes
    audit=d/'candidate/batches.jsonl'
    events=[json.loads(x) for x in audit.read_text().splitlines()] if audit.exists() else []
    row['actual_cached_prefill_batches']=sum(r.get('phase')=='prefill' and any(t>0 for t in r.get('cached_tokens') or []) for r in events)

    result['cases'][name]=row
  except Exception:result['cases'][name]={'status':'failed','traceback':traceback.format_exc()}
  verify(plan);write(out/'progress.json',result)
 verify(plan);result.update(status='followup_measured',source_unchanged=True)
 result['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'summary.json'}
 write(out/'summary.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--track',choices=['step0-a','accuracy'],required=True)
 p.add_argument('--gpu',type=int,choices=[0,1],required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path)
 a=p.parse_args()
 if a.worker:accuracy_worker(a.worker);return
 out=Path('/workspace/results/mellum_final_followup')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+a.track);out.mkdir(parents=True)
 plan=prepare(out,a.track,a.gpu)
 if a.prepare_only:print('RESULT followup_prepared output',out);return
 r=execute(out,plan);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
