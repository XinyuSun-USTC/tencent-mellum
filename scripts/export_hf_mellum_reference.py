"""Convert a completed, audited HF smoke run into the shared comparison contract.

This reads saved CPU logits and tokens only; it does not rerun GPU inference.
New runs can be produced with validate_hf_mellum_gpu.py first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--from-smoke',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.from_smoke.resolve();d=json.loads(source.read_text())
    if d['status']!='short_input_smoke_passed':raise ValueError('Only completed HF runs may be exported')
    if any(d['loading_info'].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Loading audit is not clean')
    if not d['results'] or not all(r['cache_greedy_exact'] for r in d['results']):raise ValueError('Cache check failed')
    a.output.mkdir(parents=True,exist_ok=False)
    reference={'schema_version':1,'producer':'hf','model_config_sha256':d['input_file_hashes']['config.json'],
               'source_run':str(source),'source_sha256':sha(source),'environment':d['environment'],'records':[]}
    inputs={'schema_version':1,'model_config_sha256':reference['model_config_sha256'],'records':[]}
    for index,r in enumerate(d['results']):
        sample={k:r[k] for k in ['name','rendered_prompt','input_ids','attention_mask','generation_settings','enable_thinking']}
        position=[]
        for row in sample['attention_mask']:
            n=0;pos=[]
            for bit in row:
                n+=bit;pos.append(max(n-1,0))
            position.append(pos)
        sample['position_ids']=position
        inputs['records'].append(sample)
        logits=source.parent/(r['name']+'_prefill_logits.pt')
        target=a.output/f'prefill_{index}.pt';shutil.copyfile(logits,target)
        record={**sample,'generated_ids':r['generated_ids'],'prefill_logits':target.name,
                'prefill_logits_sha256':sha(target),'logits_semantics':'last_prompt_position_full_vocab'}
        reference['records'].append(record)
    (a.output/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
    reference['inputs_sha256']=sha(a.output/'inputs.json')
    (a.output/'reference.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n')
    print('Exported',len(reference['records']),'records to',a.output)
if __name__=='__main__':main()
