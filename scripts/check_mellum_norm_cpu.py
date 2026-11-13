"""Replay captured norm inputs and compare the Mellum correction with official HF."""
import ast
import json
from pathlib import Path
import torch
from transformers.models.mellum.modeling_mellum import MellumRMSNorm as HFNorm

ROOT=Path('/workspace');run=ROOT/'results/mellum_validation/20261111T115534143208Z'
atom=Path(json.loads((run/'summary.json').read_text())['atom_run'])/'stages'
hf=run/'hf/stages'
tree=ast.parse((ROOT/'ATOM/atom/models/mellum.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MellumRMSNorm')
namespace={'torch':torch,'nn':torch.nn}
exec(compile('from __future__ import annotations\n'+ast.unparse(node),'mellum_norm','exec'),namespace)
Norm=namespace['MellumRMSNorm']
torch.set_num_threads(2)
weight=torch.load(hf/'weights/input_norm.pt',weights_only=True)
corrected=Norm(weight.numel()).to(dtype=weight.dtype)
reference=HFNorm(weight.numel()).to(dtype=weight.dtype)
with torch.no_grad():corrected.weight.copy_(weight);reference.weight.copy_(weight)
rows=[]
for sample in sorted(hf.glob('sample_*')):
    x=torch.load(sample/'input_norm_input.pt',weights_only=True)
    old=torch.load(atom/sample.name/'input_norm.pt',weights_only=True)
    captured=torch.load(sample/'input_norm.pt',weights_only=True)
    f=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
    trunc=(f*weight.float()).contiguous().view(torch.int32).bitwise_and(-65536).view(torch.float32).to(x.dtype)
    new=corrected(x);expected=reference(x)
    torch.testing.assert_close(new,expected,rtol=0,atol=0)
    residual=x*0.125
    y,combined=corrected(x,residual)
    torch.testing.assert_close(y,reference(x+residual),rtol=0,atol=0)
    torch.testing.assert_close(combined,x+residual,rtol=0,atol=0)
    rows.append({'sample':sample.name,'old_atom_equals_fp32_product_truncation':torch.equal(old,trunc),
        'old_truncation_match_fraction':(old==trunc).float().mean().item(),
        'corrected_equals_official_cpu':torch.equal(new,expected),
        'corrected_vs_captured_hf_gpu_max_abs':(new.float()-captured.float()).abs().max().item(),
        'old_atom_vs_captured_hf_gpu_max_abs':(old.float()-captured.float()).abs().max().item()})
assert not torch.cuda.is_initialized()
out=ROOT/'results/atom_mellum_norm_fix';out.mkdir(exist_ok=True)
(out/'cpu_replay.json').write_text(json.dumps({'rows':rows,'cuda_initialized':False,'scope':'CPU formula replay, GPU end-to-end regression still pending'},indent=2)+'\n')
print(json.dumps(rows,indent=2))
