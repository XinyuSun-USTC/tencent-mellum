"""Small kernel geometry/math smoke; not model acceptance or HF evaluation."""
import argparse,json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ATOM'))
    import torch
    from types import SimpleNamespace as NS
    from atom.model_config.mellum_fused_rope import fused_qk_rope
    from atom.model_config.mellum_attention import reference_position_rope
    torch.manual_seed(7);rows=[]
    for n in [1,33,1024]:
        packed=torch.randn(n,40,128,device='cuda',dtype=torch.bfloat16)
        q,k=packed[:,:32],packed[:,32:36];before=packed.clone()
        pos=(torch.arange(n,device='cuda',dtype=torch.long)*131).remainder(131072)
        inv=1./(500000.**(torch.arange(0,128,2,device='cuda').float()/128))
        for normalize in [False,True]:
            norms=tuple(NS(weight=torch.randn(128,device='cuda',dtype=torch.bfloat16),eps=1e-6) for _ in range(2)) if normalize else None
            def norm(x,w):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).bfloat16()
            qr,kr=(norm(q,norms[0].weight),norm(k,norms[1].weight)) if normalize else (q,k)
            expected=reference_position_rope(qr,kr,pos,inv,1.1)
            actual=fused_qk_rope(q,k,pos,inv,1.1,norms)
            metrics=[]
            for x,y in zip(actual,expected):
                if x.shape!=y.shape or not torch.isfinite(x).all():raise ValueError('Invalid fused kernel output')
                rel=((x.float()-y.float()).norm()/y.float().norm()).item()
                # Broad smoke guard catches indexing/sign/norm errors; not an acceptance threshold.
                if rel>.05:raise ValueError(f'Kernel math smoke failed: relative_l2={rel}')
                metrics.append({'max_abs':(x.float()-y.float()).abs().max().item(),'relative_l2':rel})
            if not torch.equal(before,packed):raise ValueError('Input storage mutated')
            rows.append({'tokens':n,'norm':normalize,'metrics':metrics})
    a.output.write_text(json.dumps({'status':'kernel_smoke_passed','scope':'Not model correctness acceptance','cases':rows},indent=2)+'\n')
    print('RESULT kernel_smoke_passed',flush=True)
if __name__=='__main__':main()
