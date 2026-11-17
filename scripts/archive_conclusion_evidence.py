from pathlib import Path
import json,hashlib,shutil
ROOT=Path('/workspace/tencent-mellum-delivery-r32')
out=ROOT/'artifacts/submission/conclusion_evidence';out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
index={'scope':'Lightweight conclusion evidence: original summaries, derived analyses and reproducible token comparison; excludes tensors/full runtime logs','files':{},'sources':{}}
def copy(source,target):
 p=out/target;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,p)
 index['files'][str(p.relative_to(ROOT))]={'sha256':sha(p),'original':str(source)}
for category,names in {
 'mellum_c_chunk_screen':['20261116T233809387315Z','20261116T233817696068Z'],
 'mellum_c_bottleneck_profile':['20261117T000857196743Z','20261117T000905500998Z'],
 'mellum_delivery_matrix':['20261117T003512204779Z','20261117T003519808987Z']}.items():
 for name in names:
  source=Path('/workspace/results')/category/name;s=json.loads((source/'summary.json').read_text())
  for f,h in s['artifact_sha256'].items():
   if sha(source/f)!=h:raise ValueError('Changed original '+str(source/f))
  label=Path(category)/name
  copy(source/'summary.json',label/'summary.json')
  index['sources'][str(source)]={'verified_original_files':len(s['artifact_sha256']),'summary_sha256':sha(source/'summary.json')}
  for f in ['trace_analysis.json','trace_analysis.md','runtime.json','manifest.json']:
   p=source/f
   if p.exists():
    if p.stat().st_size<300000:copy(p,label/f)
    else:
     d=json.loads(p.read_text());d.pop('prompts',None)
     target=out/label/(f+'.compact.json');target.write_text(json.dumps(d,indent=2)+'\n')
     index['files'][str(target.relative_to(ROOT))]={'sha256':sha(target),'original':str(p),'original_sha256':sha(p),'transformation':'Prompt arrays omitted; raw source hashes retained'}
copy(Path('/workspace/results/mellum_c_capacity_analysis/20261116T235646021547Z/analysis.json'),Path('integrated_c_analysis.json'))
matrix=json.loads((ROOT/'artifacts/submission/final_r32_matrix/matrix.json').read_text())
for row,info in matrix['rows'].items():
 source=Path(info['acceptance_run']);manifest=json.loads((source/'manifest.json').read_text())
 copy(source/'summary.json',Path('final_acceptance')/row/'summary.json')
 # Old acceptance workers and recent matrix use the same generation filenames.
 for suite in ['tokens','long']:
  reference=Path(manifest['references'][suite]);candidate=source/(suite+'_generation.json')
  if not candidate.exists():raise ValueError('Candidate missing '+str(candidate))
  left=json.loads(reference.read_text())['records'];right=json.loads(candidate.read_text())['records']
  records=[]
  if len(left)!=len(right):raise ValueError('Record count differs')
  for a,b in zip(left,right):
   if a['name']!=b['name'] or a['input_ids']!=b['input_ids']:raise ValueError('Protocol differs')
   records.append({'name':a['name'],'source_id':a.get('source_id'),'input_tokens':len(a['input_ids'][0]),
     'input_ids_sha256':hashlib.sha256(json.dumps(a['input_ids'],separators=(',',':')).encode()).hexdigest(),
     'reference_ids':a['generated_ids'][0],'candidate_ids':b['generated_ids'][0]})
  target=out/'final_acceptance'/row/(suite+'_token_pairs.json');target.parent.mkdir(parents=True,exist_ok=True)
  payload={'scope':'Recompute positional and exact-sequence agreement without tensor dumps',
    'reference_sha256':sha(reference),'candidate_sha256':sha(candidate),'records':records}
  target.write_text(json.dumps(payload,indent=2)+'\n')
  index['files'][str(target.relative_to(ROOT))]={'sha256':sha(target),'transformation':'Keep names, input identity, reference and candidate output IDs'}
 for f in ['candidate_score/score.json','candidate_score/samples.jsonl_results.jsonl']:
  p=source/f
  if p.exists():copy(p,Path('final_acceptance')/row/f)
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
print('archived',len(index['files']),'files; bytes',sum((ROOT/p).stat().st_size for p in index['files']))
