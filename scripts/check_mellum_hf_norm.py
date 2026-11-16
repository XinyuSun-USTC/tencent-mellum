"""GPU-only local math smoke for HF-compatible norm; not model acceptance."""
import argparse,json,os,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu);sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ATOM'))
    import torch
    from atom.model_config.mellum_hf_norm import HFCompatibleRMSNorm
    torch.manual_seed(17);results=[]
    with torch.inference_mode():
        for dim in [128,2304]:
            norm=HFCompatibleRMSNorm(dim).to(device='cuda',dtype=torch.bfloat16)
            norm.weight.copy_(torch.randn_like(norm.weight))
            for rows in [1,32,1024]:
                x=torch.randn(rows,dim*2,device='cuda',dtype=torch.bfloat16)[:,::2];original=x.clone()
                for has_res in [False,True]:
                    residual=torch.randn_like(x) if has_res else None
                    combined=x if residual is None else x+residual
                    expected=norm.weight*(combined.float()*torch.rsqrt(combined.float().square().mean(-1,keepdim=True)+norm.eps)).bfloat16()
                    actual=norm(x,residual);y=actual[0] if has_res else actual
                    rel=float((y.float()-expected.float()).norm()/expected.float().norm())
                    if not torch.isfinite(y).all() or rel>.01:raise ValueError('HF Norm local math smoke failed')
                    if has_res and not torch.equal(actual[1],combined):raise ValueError('Residual BF16 boundary mismatch')
                    if not torch.equal(x,original):raise ValueError('Input mutated')
                    results.append({'dim':dim,'rows':rows,'residual':has_res,'relative_l2':rel,'max_abs':float((y.float()-expected.float()).abs().max())})
    a.output.write_text(json.dumps({'status':'kernel_smoke_passed','scope':'Local math only, not greedy200 or HE acceptance','cases':results},indent=2)+'\n')
    print('RESULT kernel_smoke_passed',flush=True)
if __name__=='__main__':main()
