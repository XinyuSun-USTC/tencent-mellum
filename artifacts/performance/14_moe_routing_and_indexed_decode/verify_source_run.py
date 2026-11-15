import hashlib,json,sys
from pathlib import Path
root=Path('/workspace/results/mellum_moe_optimization/20261115T121305387577Z')
m=json.loads((root/'manifest.json').read_text());errors=[]
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
checked=0;repeats=[]
for name,job in m['jobs'].items():
 if job['status']!='complete':errors.append([name,'incomplete'])
 for rel,value in job['artifacts'].items():
  path=root/'jobs'/name/rel
  if not path.exists() or digest(path)!=value:errors.append([name,rel,'artifact_hash'])
  checked+=1
 plan=job['plan']
 if name=='micro' or name.startswith('HTTP'):continue
 for cohort,cases in plan['cohorts'].items():
  actual=[]
  for i in range(plan['repeats']):
   r=json.loads((root/'jobs'/name/f'{cohort}_r{i}.json').read_text());actual.append([v['actual_ids'] for v in r['records']])
   if len(r['records'])!=len(cases) or len(r['completions'])!=len(cases):errors.append([name,cohort,'count'])
   for case,row in zip(cases,r['records']):
    if row['name']!=case['name'] or row['expected_ids']!=case['expected_ids'] or len(row['actual_ids'])!=case['budget']:
     errors.append([name,cohort,'identity_budget'])
  if len(actual)>1:
   same=all(v==actual[0] for v in actual[1:]);repeats.append({'job':name,'cohort':cohort,'equal':same})
   if not same:errors.append([name,cohort,'unstable'])
for path,value in m['fingerprint'].items():
 if not Path(path).exists() or digest(path)!=value:errors.append(['source',path])
reference=Path(m['provenance']['source_run'])/'hf/reference.json'
if digest(reference)!=m['provenance']['reference_sha256']:errors.append(['hf_reference','hash'])
a=json.loads((root/'jobs/D200_CPD/D200_r0.json').read_text())
b=json.loads((root/'jobs/D200_shared_routing/D200_r0.json').read_text())
cpd={r['name']:r['actual_ids'] for r in a['records']}
exact=sum(r['actual_ids']==cpd[r['name']] for r in b['records'])
if exact!=200:errors.append(['shared_vs_cpd','not_exact'])
phases={}
for name in ['D200_CPD','D200_shared_routing']:
 r=json.loads((root/'jobs'/name/'D200_r0.json').read_text());phases[name]=r['actual_batches']
 r2=json.loads((root/'jobs'/name/'results.json').read_text())
 if r2['loading_audit']!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:
  errors.append([name,'load_audit'])
http=root/'jobs/HTTP_shared_routing/http/server/batches.jsonl'
http_rows=[json.loads(x) for x in http.read_text().splitlines()]
report={'source_run':str(root),'git_head':m['git_head'],'artifact_files_verified':checked,'source_files_verified':len(m['fingerprint']),
 'worker_seconds':m['used_seconds'],'started_workers':m['started_workers'],'repeat_checks':repeats,
 'shared_routing_vs_current_CPD_exact_sequences':exact,'D200_actual_batches':phases,
 'HTTP_actual_decode_max':max(r['batch_size'] for r in http_rows if r['phase']=='decode'),
 'cached_reference_sha256':digest(reference),'errors':errors,
 'note':'D200 one run per configuration; numerical probe completion is not HF accuracy acceptance'}
Path('/tmp/mellum_moe_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['repeat_checks','D200_actual_batches']},indent=2))
if errors:sys.exit(1)
