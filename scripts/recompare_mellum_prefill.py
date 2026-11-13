"""Offline recovery of the prefill suite's missing template metadata; originals stay intact."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from diagnose_mellum_projection_math import digest
from compare_mellum_results import compare
from compare_mellum_layers import compare_layers

ROOT=Path('/workspace')
def repaired_copy(source,target):
    data=json.loads(source.read_text());changes=[]
    for row in data['records']:
        if 'enable_thinking' not in row:
            origin=row.get('prompt_source')
            if origin not in ['raw_code_boundary','new_developer_prompt','existing_window_boundary']:
                raise ValueError('Cannot infer template metadata for '+row['name'])
            row['enable_thinking']=origin!='raw_code_boundary'
            changes.append({'name':row['name'],'field':'enable_thinking','value':row['enable_thinking'],'basis':origin})
        path=Path(row['prefill_logits'])
        if not path.is_absolute():row['prefill_logits']=str((source.parent/path).resolve())
    target.write_text(json.dumps(data,indent=2)+'\n')
    return {'source':str(source.resolve()),'source_sha256':digest(source),'derived_sha256':digest(target),'changes':changes}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();source=a.run.resolve()
    import torch
    torch.set_num_threads(4)
    atom=Path(json.loads((source/'summary.json').read_text())['atom_run'])
    out=ROOT/'results/mellum_recomparison'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    provenance=[repaired_copy(source/'hf/reference.json',out/'reference.json'),repaired_copy(atom/'candidate.json',out/'candidate.json')]
    comparison=compare(out/'reference.json',out/'candidate.json')
    layers=compare_layers(source/'hf/layers',atom/'layers')
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (out/'layers_comparison.json').write_text(json.dumps(layers,indent=2)+'\n')
    bad=[r for r in layers['records'] if r['status']!='exact']
    result={'status':'offline_recomparison_complete','source_run':str(source),'atom_run':str(atom),'provenance':provenance,
        'comparison_status':comparison['status'],'token_summary':comparison['token_summary'],
        'layer_summary':{'compared_tensors':len(layers['records']),'nonexact_tensors':len(bad),'first_nonexact':bad[:1]},
        'scope':'Metadata-only derived copies; no new inference, token IDs/logits/tensors unchanged'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))
    print('RESULT offline_recomparison_complete output',out)
if __name__=='__main__':main()
