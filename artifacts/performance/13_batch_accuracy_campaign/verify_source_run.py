import hashlib,json,sys
from pathlib import Path
root=Path('/workspace/results/mellum_batch_campaign/20261115T043721420603Z')
m=json.loads((root/'manifest.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
errors=[];checked=0
for name,job in m['jobs'].items():
 if job['status']!='complete':errors.append((name,'not complete'))
 for rel,expected in job['artifacts'].items():
  p=root/'jobs'/name/rel
  if not p.exists() or digest(p)!=expected:errors.append((name,rel))
  checked+=1
for path,expected in m['fingerprint'].items():
 if not Path(path).exists() or digest(path)!=expected:errors.append(('source',path))
repeat_checks=[]
for name,job in m['jobs'].items():
 if name=='D_http':continue
 plan=job['plan'];result=json.loads((root/'jobs'/name/'results.json').read_text())
 for cohort,cases in plan['cohorts'].items():
  actual=[]
  for i in range(plan['repeats']):
   run=json.loads((root/'jobs'/name/f'{cohort}_r{i}.json').read_text())
   if len(run['records'])!=len(cases) or len(run['completions'])!=len(cases):errors.append((name,cohort,'missing'))
   for case,row in zip(cases,run['records']):
    if case['name']!=row['name'] or case['expected_ids']!=row['expected_ids']:errors.append((name,cohort,'identity'))
    if len(row['actual_ids'])!=case['budget']:errors.append((name,cohort,'length'))
   actual.append([r['actual_ids'] for r in run['records']])
  if len(actual)>1:
   same=all(x==actual[0] for x in actual[1:]);repeat_checks.append({'job':name,'cohort':cohort,'equal':same})
   if not same:errors.append((name,cohort,'unstable'))
source=Path(m['provenance']['source_run'])/'hf/reference.json'
if digest(source)!=m['provenance']['reference_sha256']:errors.append(('reference','sha'))
summary={'source_run':str(root),'git_head':m['git_head'],'artifact_files_verified':checked,
         'source_files_verified':len(m['fingerprint']),'repeated_cohorts':repeat_checks,
         'cached_reference_sha256':digest(source),'errors':errors,
         'started_workers':m['started_workers'],'worker_seconds':m['used_seconds'],
         'note':'D200 is one run per candidate; its stable field alone is not a repeated stability result.'}
Path('/tmp/mellum_campaign_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if errors:sys.exit(1)
