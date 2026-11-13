"""Replay real fixed router logits through HF's torch operations and AITER topk.

No model/weights/engine allocation. This is a diagnostic, not an acceptance test.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

ROOT=Path('/workspace')

def hf_route(logits,topk=8):
    import torch
    probs=logits.float().softmax(-1)
    weights,ids=probs.topk(topk,dim=-1)
    weights=weights/weights.sum(-1,keepdim=True)
    return weights.to(logits.dtype),ids

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu);os.environ['AITER_LOG_LEVEL']='WARNING'
    a.output.mkdir(parents=True,exist_ok=False);report={'status':'started','records':[]}
    def save():(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import torch
        if not torch.cuda.is_available():report['status']='blocked_no_gpu';save();return 2
        from aiter import topk_softmax
        source=ROOT/'results/mellum_validation/20261111T120512707283Z'
        atom=Path(json.loads((source/'summary.json').read_text())['atom_run'])/'stages'
        roots={'hf':source/'hf/stages','atom':atom}
        for producer,root in roots.items():
            files=sorted(root.glob('sample_*/router_logits.pt'))
            if len(files)!=2:
                raise ValueError(f'Expected two saved router inputs for {producer}, found {len(files)} in {root}')
            for file in files:
                logits=torch.load(file,map_location='cpu',weights_only=True).to('cuda')
                expected_weights,expected_ids=hf_route(logits)
                probs=logits.float().softmax(-1);boundary=probs.topk(9,dim=-1).values
                runs=[]
                for repeat in range(3):
                    weights=torch.empty((len(logits),8),device='cuda',dtype=torch.float32)
                    ids=torch.empty((len(logits),8),device='cuda',dtype=torch.int32)
                    indices=torch.empty_like(ids)
                    # Same call as native topK.py for Mellum: no shared experts.
                    topk_softmax(weights,ids,indices,logits,True,0,'')
                    same_set=torch.equal(ids.long().sort(-1).values,expected_ids.sort(-1).values)
                    different=(ids.long().sort(-1).values!=expected_ids.sort(-1).values).any(-1)
                    ideal=probs.gather(-1,ids.long());ideal=ideal/ideal.sum(-1,keepdim=True)
                    quantized_ideal=ideal.to(logits.dtype).float()
                    record={'repeat':repeat,'expert_set_exact':same_set,'different_set_rows':different.sum().item(),
                            'native_weights_vs_hf_rounded_max_abs':(weights-quantized_ideal).abs().max().item(),
                            'native_ids':ids.cpu().tolist(),'hf_ids':expected_ids.cpu().tolist(),
                            'native_weights_dtype':str(weights.dtype),'hf_weights_dtype':str(expected_weights.dtype)}
                    runs.append(record)
                report['records'].append({'producer':producer,'sample':file.parent.name,'tokens':len(logits),
                    'top8_boundary_tie_rows':(boundary[:,7]==boundary[:,8]).sum().item(),
                    'native_ids_repeat_stable':all(r['native_ids']==runs[0]['native_ids'] for r in runs),'runs':runs})
                save();print(producer,file.parent.name,'sets exact',runs[0]['expert_set_exact'],'ties',report['records'][-1]['top8_boundary_tie_rows'],flush=True)
        report.update(status='diagnostic_complete',scope='Same-logits routing only; no claim about full MoE or correctness acceptance')
        save();print('RESULT diagnostic_complete output',a.output,flush=True);return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());save();print(report['traceback']);return 1
if __name__=='__main__':raise SystemExit(main())
