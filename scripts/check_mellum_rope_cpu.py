"""Compare actual AITER cache-building source with official HF, without importing AITER/GPU."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import types
import torch
from transformers import AutoConfig
from transformers.models.mellum.modeling_mellum import MellumRotaryEmbedding

root=Path('/workspace')
spec=importlib.util.spec_from_file_location('mellum_rope_helpers',root/'ATOM/atom/model_config/mellum_utils.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
source=Path('/app/aiter-test/aiter/rotary_embedding.py')
tree=ast.parse(source.read_text())
names={'RotaryEmbedding','YaRNScalingRotaryEmbedding','_yarn_find_correction_dim','_yarn_find_correction_range','_yarn_linear_ramp_mask','_yarn_get_mscale'}
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
namespace={'torch':torch,'math':math,'nn':torch.nn,'dtypes':types.SimpleNamespace(fp32=torch.float32)}
code='from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[]))
exec(compile(code,str(source),'exec'),namespace)
torch.set_num_threads(2)
config=AutoConfig.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking',local_files_only=True)
ref=MellumRotaryEmbedding(config)
pos=torch.tensor([[0,1,1023,1024,8191,8192,8193,16383]])
result={}
for kind in ['sliding_attention','full_attention']:
    p=helpers.aiter_rope_parameters(config.rope_parameters[kind])
    if p['rope_type']=='default':
        obj=namespace['RotaryEmbedding'](128,128,16384,p['rope_theta'],True,torch.float32)
    else:
        obj=namespace['YaRNScalingRotaryEmbedding'](128,128,p['original_max_position_embeddings'],p['rope_theta'],True,p['factor'],torch.float32,attn_factor=p['attn_factor'],beta_fast=p['beta_fast'],beta_slow=p['beta_slow'])
    with torch.inference_mode():hc,hs=ref(torch.zeros(1,1,128),pos,kind)
    ac=obj.cos_cache[pos.flatten()].reshape(-1,64).repeat(1,2).unsqueeze(0)
    ass=obj.sin_cache[pos.flatten()].reshape(-1,64).repeat(1,2).unsqueeze(0)
    torch.testing.assert_close(ac,hc,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(ass,hs,atol=2e-6,rtol=1e-5)
    result[kind]={'cos_max_abs':(ac-hc).abs().max().item(),'sin_max_abs':(ass-hs).abs().max().item(),'parameters':p}
assert not torch.cuda.is_initialized()
out=root/'results/atom_step0_batch_c';out.mkdir(parents=True,exist_ok=True)
(out/'rope_cpu.json').write_text(json.dumps({'status':'passed','positions':pos.tolist(),'results':result,'scope':'FP32 cache construction only; no GPU rotation kernels','cuda_initialized':False},indent=2)+'\n')
print(json.dumps(result,indent=2))
