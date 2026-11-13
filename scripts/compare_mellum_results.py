"""Offline comparison of version-1 Mellum artifacts; no model or GPU loading."""
import argparse
import hashlib
import json
import math
from pathlib import Path

INPUT_FIELDS=['name','rendered_prompt','input_ids','attention_mask','position_ids','generation_settings','enable_thinking','logits_semantics']
def token_stats(reference,candidate):
    total=matching=prefix=0
    for index in range(max(len(reference),len(candidate))):
        left=reference[index] if index<len(reference) else []
        right=candidate[index] if index<len(candidate) else []
        total+=max(len(left),len(right))
        matching+=sum(a==b for a,b in zip(left,right))
        for a,b in zip(left,right):
            if a!=b:break
            prefix+=1
    return {'compared_positions':total,'matching_positions':matching,
            'common_prefix_tokens':prefix,'token_match_fraction':matching/total if total else None}

def load(path):
    d=json.loads(path.read_text())
    if d.get('schema_version')!=1:raise ValueError('Unsupported artifact schema')
    names=[r['name'] for r in d['records']]
    if not names or len(set(names))!=len(names):raise ValueError('Empty or duplicate record names')
    for r in d['records']:
        for field in INPUT_FIELDS+['generated_ids','prefill_logits','prefill_logits_sha256']:
            if field not in r:raise ValueError('Missing field '+field)
    return d

def compare(left_path,right_path,atol=0.0,rtol=0.0):
    import torch
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol,rtol)<0:raise ValueError('Invalid tolerance')
    left,right=load(left_path),load(right_path)
    result={'status':'passed','atol':atol,'rtol':rtol,'records':[],'mismatches':[]}
    if left['model_config_sha256']!=right['model_config_sha256']:result['mismatches'].append('model_config_sha256')
    l={r['name']:r for r in left['records']};r={r['name']:r for r in right['records']}
    if set(l)!=set(r):result['mismatches'].append('record_names')
    for name in sorted(set(l)&set(r)):
        a,b=l[name],r[name];row={'name':name,'mismatches':[],'token_stats':token_stats(a['generated_ids'],b['generated_ids'])}
        for f in INPUT_FIELDS:
            if a[f]!=b[f]:row['mismatches'].append(f)
        if a['generated_ids']!=b['generated_ids']:
            row['mismatches'].append('generated_ids')
            row['first_token_difference']=next(({'batch':i,'position':j} for i,(x,y) in enumerate(zip(a['generated_ids'],b['generated_ids'])) for j in range(max(len(x),len(y))) if j>=len(x) or j>=len(y) or x[j]!=y[j]),{'batch_shape_difference':True})
        tensors=[]
        for path,record in [(left_path,a),(right_path,b)]:
            file=path.parent/record['prefill_logits']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=record['prefill_logits_sha256']:raise ValueError('Logits artifact hash mismatch')
            tensors.append(torch.load(file,map_location='cpu',weights_only=True).float())
        x,y=tensors
        if x.shape!=y.shape:row['mismatches'].append('logits_shape')
        elif not torch.isfinite(x).all() or not torch.isfinite(y).all():row['mismatches'].append('nonfinite_logits')
        else:
            delta=x-y;row.update(max_abs_error=delta.abs().max().item(),rms_error=delta.square().mean().sqrt().item(),top1_exact=bool(torch.equal(x.argmax(-1),y.argmax(-1))))
            if not torch.allclose(x,y,atol=atol,rtol=rtol):row['mismatches'].append('logits_values')
        if row['mismatches']:result['mismatches'].append(name)
        result['records'].append(row)
    stats=[r['token_stats'] for r in result['records']]
    positions=sum(x['compared_positions'] for x in stats)
    matches=sum(x['matching_positions'] for x in stats)
    result['token_summary']={'compared_positions':positions,'matching_positions':matches,'token_match_fraction':matches/positions if positions else None,'exact_sequences':sum(l[n]['generated_ids']==r[n]['generated_ids'] for n in set(l)&set(r)),'reference_records':len(l)}
    if result['mismatches']:result['status']='failed'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--atol',type=float,default=0);p.add_argument('--rtol',type=float,default=0)
    a=p.parse_args();result=compare(a.reference,a.candidate,a.atol,a.rtol)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
