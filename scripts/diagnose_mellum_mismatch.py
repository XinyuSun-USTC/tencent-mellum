"""Prepare same-prefix one-token probes from a failed run, then trace both engines."""
import argparse
from datetime import datetime,timezone
import copy
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace')
def prepare(run,cases=None):
    hf=json.loads((run/'hf/reference.json').read_text())
    meta=json.loads((run/'summary.json').read_text())
    atom=json.loads((Path(meta['atom_run'])/'candidate.json').read_text())
    candidates={r['name']:r for r in atom['records']}
    probes=[]
    if cases:
        missing=set(cases)-{r['name'] for r in hf['records']}
        if missing:raise ValueError(f'Unknown cases: {sorted(missing)}')
    for record in hf['records']:
        if cases and record['name'] not in cases:continue
        want=record['generated_ids'][0];got=candidates[record['name']]['generated_ids'][0]
        first=next((j for j in range(min(len(want),len(got))) if want[j]!=got[j]),None)
        for count in sorted({0,first} if first is not None else {0}):
            row=copy.deepcopy(record);tokens=record['input_ids'][0]+want[:count]
            row.update(name=f"{record['name']}_prefix_{count}",input_ids=[tokens],
                attention_mask=[[1]*len(tokens)],position_ids=[list(range(len(tokens)))],
                rendered_prompt=record['rendered_prompt'])
            # raw IDs are authoritative; rendered_prompt is the original user
            # prompt, while forced_prefix explicitly records appended tokens.
            row['forced_prefix_ids']=want[:count]
            row['source_hf_cached_next_token_id']=want[count] if count<len(want) else None
            row['generation_settings']['max_new_tokens']=1
            for key in ['generated_ids','prefill_logits','prefill_logits_sha256','logits_semantics']:row.pop(key,None)
            probes.append(row)
    return {'schema_version':1,'suite':'diagnostic','model_config_sha256':hf['model_config_sha256'],
            'source_run':str(run),'records':probes}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true');p.add_argument('--stages',action='store_true')
    p.add_argument('--cases',nargs='+',help='Select named inputs before building probes')
    a=p.parse_args();data=prepare(a.run.resolve(),a.cases)
    out=ROOT/'results/mellum_diagnostic_inputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    file=out/'inputs.json';file.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Same-prefix probes:',[(r['name'],len(r['input_ids'][0])) for r in data['records']],flush=True)
    if a.prepare_only:return 0
    return subprocess.call(['/opt/venv/bin/python',str(ROOT/'scripts/run_mellum_validation.py'),
          '--suite','diagnostic','--inputs',str(file),'--dump-layers','--gpu',str(a.gpu)]+(['--dump-stages'] if a.stages else []))
if __name__=='__main__':raise SystemExit(main())
