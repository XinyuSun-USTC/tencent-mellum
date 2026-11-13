"""Canonical layer-zero expert replay: isolate activation and reduction rounding."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import traceback

ROOT=Path('/workspace')
MODEL=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')

def expert_activation(gate,up,fused=False):
    import torch.nn.functional as F
    if fused:return (F.silu(gate.float())*up.float()).to(gate.dtype)
    return F.silu(gate)*up

def error(actual,expected):
    import torch
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('Nonfinite expert replay output')
    a,b=actual.float(),expected.float();d=a-b
    return {'exact':bool((actual==expected).all()),'max_abs':d.abs().max().item(),
            'relative_l2':(d.norm()/b.norm().clamp_min(1e-12)).item()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--official-backends',action='store_true',help='Use the dedicated HF interpreter to replay grouped_mm and eager experts')
    a=p.parse_args();os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu)
    out=ROOT/'results/mellum_expert_math'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    result={'status':'started','scope':'Canonical first-layer diagnostic, not engine acceptance','records':[]}
    def save():(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open
        if not torch.cuda.is_available():result['status']='blocked_no_gpu';save();return 2
        torch.set_grad_enabled(False)
        index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
        def load(expert,name):
            key=f'model.layers.0.mlp.experts.{expert}.{name}_proj.weight'
            with safe_open(MODEL/index[key],framework='pt',device='cpu') as f:return f.get_tensor(key).to('cuda')
        source=ROOT/'results/mellum_validation/20261111T140735437405Z/hf/stages'
        replay=json.loads((ROOT/'results/mellum_moe_replay/20261111T155808236095Z/results.json').read_text())
        native=Path(replay['atom_run'])/'stages'
        if a.official_backends:
            import transformers
            if transformers.__version__!='5.10.1':raise RuntimeError('Official replay requires dedicated HF 5.10.1 interpreter')
            from transformers import MellumConfig
            from transformers.models.mellum.modeling_mellum import MellumExperts
            config=MellumConfig.from_pretrained(MODEL,local_files_only=True)
            with torch.device('meta'):experts=MellumExperts(config)
            experts.gate_up_proj=torch.nn.Parameter(torch.stack([torch.cat([load(e,'gate'),load(e,'up')]) for e in range(64)]),requires_grad=False)
            experts.down_proj=torch.nn.Parameter(torch.stack([load(e,'down') for e in range(64)]),requires_grad=False)
            result['transformers']=transformers.__version__
        for sample in sorted(source.glob('sample_*')):
            x=torch.load(sample/'moe_input.pt',weights_only=True,map_location='cpu').to('cuda')
            logits=torch.load(sample/'router_logits.pt',weights_only=True,map_location='cpu').to('cuda')
            weights,ids=logits.float().softmax(-1).topk(8,dim=-1)
            weights=(weights/weights.sum(-1,keepdim=True)).to(logits.dtype)
            targets={'hf':torch.load(sample/'moe_output.pt',weights_only=True,map_location='cpu').to('cuda'),
                     'atom_hf_router':torch.load(native/sample.name/'moe_output.pt',weights_only=True,map_location='cpu').to('cuda')}
            if a.official_backends:
                rows={}
                for mode in ['grouped_mm','eager']:
                    config._experts_implementation=mode
                    value=experts(x,ids,weights)
                    rows[mode]={target:error(value,expected) for target,expected in targets.items()}
                    torch.save(value.cpu(),out/(sample.name+'_'+mode+'.pt'))
                result['records'].append({'sample':sample.name,'official_backends':rows});save()
                print(sample.name,json.dumps(rows),flush=True)
                continue
            sums={f'{act}_{reduction}':torch.zeros_like(x,dtype=x.dtype if reduction=='bf16_sum' else torch.float32)
                  for act in ['hf_activation','fused_activation'] for reduction in ['bf16_sum','fp32_sum','fp32_weight_sum']}
            for e in range(64):
                # Same top-k-position-major token order as MellumExperts.forward.
                slot,token=torch.where(ids.T==e)
                if not len(token):continue
                gate_up=torch.cat([load(e,'gate'),load(e,'up')],dim=0)
                gate,up=F.linear(x[token],gate_up).chunk(2,-1)
                down=load(e,'down')
                for fused in [False,True]:
                    activation=expert_activation(gate,up,fused)
                    projected=F.linear(activation,down)
                    contribution=projected*weights[token,slot,None]
                    act='fused_activation' if fused else 'hf_activation'
                    sums[act+'_bf16_sum'].index_add_(0,token,contribution)
                    sums[act+'_fp32_sum'].index_add_(0,token,contribution.float())
                    sums[act+'_fp32_weight_sum'].index_add_(0,token,projected.float()*weights[token,slot,None].float())
                    if e==int(ids[0,0]):
                        dest=out/sample.name;dest.mkdir(exist_ok=True)
                        torch.save({'expert':e,'token':token.cpu(),'slot':slot.cpu(),'gate':gate.cpu(),'up':up.cpu(),
                                    'activation':activation.cpu(),'down':projected.cpu(),'weighted':contribution.cpu()},
                                   dest/(act+'_intermediates.pt'))
            rows={name:{target:error(value.to(x.dtype),expected) for target,expected in targets.items()}
                  for name,value in sums.items()}
            result['records'].append({'sample':sample.name,'variants':rows});save()
            print(sample.name,json.dumps(rows),flush=True)
        if len(result['records'])!=4:raise ValueError('Expected four diagnostic inputs')
        result['status']='diagnostic_complete';save();print('RESULT diagnostic_complete output',out,flush=True);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();print(result['traceback']);return 1
if __name__=='__main__':raise SystemExit(main())
