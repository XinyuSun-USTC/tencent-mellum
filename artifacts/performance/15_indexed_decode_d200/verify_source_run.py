import hashlib,json
from pathlib import Path
root=Path('/workspace/results/mellum_indexed_d200/20261115T133034290474Z');manifest=json.loads((root/'manifest.json').read_text());errors=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checked=0
for name,job in manifest['jobs'].items():
 for rel,digest in job['artifacts'].items():
  if sha(root/'jobs'/name/rel)!=digest:errors.append(['artifact',name,rel])
  checked+=1
for p,digest in manifest['fingerprint'].items():
 if sha(p)!=digest:errors.append(['source',p])
for p,digest in manifest['provenance']['preserved_source_sha256'].items():
 if sha(p)!=digest:errors.append(['screened_source',p])
for name,data in manifest['provenance']['baselines'].items():
 if sha(data['path'])!=data['sha256']:errors.append(['baseline',name])
reference=Path(manifest['provenance']['source_run'])/'hf/reference.json'
if sha(reference)!=manifest['provenance']['reference_sha256']:errors.append(['HF_reference'])
r=json.loads((root/'jobs/indexed_D200/D200_r0.json').read_text());cases=manifest['jobs']['indexed_D200']['plan']['cohorts']['D200']
assert len(r['records'])==len(r['completions'])==len(cases)==200
for case,row in zip(cases,r['records']):
 if case['name']!=row['name'] or case['expected_ids']!=row['expected_ids']:errors.append(['case',case['name']])
lengths={};prefix={};fork={}
for row in r['records']:
 n=len(row['actual_ids']);lengths[str(n)]=lengths.get(str(n),0)+1
 pos=row['first_difference_zero_based'];label='exact' if pos is None else '0-15' if pos<16 else '16-31' if pos<32 else '32-47' if pos<48 else '48-63'
 fork[label]=fork.get(label,0)+1
for n in [16,32,64]:
 matches=sum(sum(a==b for a,b in zip(row['actual_ids'][:n],row['expected_ids'][:n])) for row in r['records'])
 prefix[str(n)]={'matching_positions':matches,'compared_positions':200*n,'fraction':matches/(200*n)}
report={'source_run':str(root),'git_head':manifest['git_head'],'artifact_files_verified':checked,'source_files_verified':len(manifest['fingerprint']),
 'screened_candidate_source_files_verified':len(manifest['provenance']['preserved_source_sha256']),
 'requests_completed':len(r['completions']),'output_length_histogram':lengths,'prefix_match':prefix,
 'first_difference_histogram':fork,'worker_seconds':manifest['used_seconds'],'errors':errors}
Path('/tmp/indexed_d200_integrity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert not errors
