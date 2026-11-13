"""Replay saved HF layer-zero MoE inputs through the native ATOM model."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
from compare_mellum_layers import compare_layers,compare_weights

ROOT=Path('/workspace')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--routing',choices=['native','rounded_native','hf'],default='native')
    p.add_argument('--routing-study',action='store_true',help='Run rounded-native and HF routing sequentially')
    a=p.parse_args();source=a.run.resolve();reference=source/'hf/reference.json';stages=source/'hf/stages'
    if a.routing_study:
        for mode in ['rounded_native','hf']:
            print('ROUTING',mode,flush=True)
            code=subprocess.call(['/opt/venv/bin/python',str(Path(__file__).resolve()),'--run',str(source),'--gpu',str(a.gpu),'--routing',mode])
            if code:return code
        return 0
    if not reference.is_file() or not stages.is_dir():raise FileNotFoundError(source)
    out=ROOT/'results/mellum_moe_replay'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    cmd=['/opt/venv/bin/python',str(ROOT/'scripts/check_atom_mellum.py'),'--reference',str(reference),
         '--dump-layers','--dump-stages','--moe-replay-inputs',str(stages),'--gpu',str(a.gpu),
         '--moe-replay-routing',a.routing]
    print('Replaying saved HF inputs; engine log:',out/'atom.log',flush=True)
    with (out/'atom.log').open('w') as log:
        code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
    log=(out/'atom.log').read_text();matches=re.findall(r'RESULT moe_replay_complete output (\S+)',log)
    result={'status':'failed','source_run':str(source),'returncode':code,'routing':a.routing,
            'scope':'Diagnostic intervention; generation tokens are not correctness evidence'}
    if code==0 and len(matches)==1:
        atom=Path(matches[0]);result['atom_run']=str(atom)
        rows=compare_layers(stages,atom/'stages')['records']
        result['records']=[r for r in rows if r['stage'] in ['moe_input','router_logits','moe_output']]
        result['weights']=compare_weights(stages/'weights',atom/'stages/weights')
        result['status']='diagnostic_complete'
        if any(r['status']!='exact' for r in result['records'] if r['stage']=='moe_input'):
            result['status']='replay_input_mismatch'
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in result.get('records',[]):print(row['sample'],row['stage'],row['status'],row.get('relative_l2'),flush=True)
    print('RESULT',result['status'],'output',out,flush=True)
    return 0 if result['status']=='diagnostic_complete' else 1
if __name__=='__main__':raise SystemExit(main())
