"""Replay the first divergent layer's post-attention tail using canonical weights."""
import argparse
import ast
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import traceback
from diagnose_mellum_projection_math import compare,digest
from diagnose_mellum_attention_math import import_native_rope

ROOT=Path('/workspace')

def own_norm_class():
    path=ROOT/'ATOM/atom/models/mellum.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MellumRMSNorm')
    import torch
    scope={'torch':torch,'nn':torch.nn}
    exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(path),'exec'),scope)
    return scope['MellumRMSNorm']


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu);source=a.run.resolve()
    out=ROOT/'results/mellum_layer_tail'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    report={'status':'started','source_run':str(source),'script_sha256':digest(__file__)}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import torch,transformers
        import torch.nn.functional as F
        if transformers.__version__!='5.10.1':raise ValueError('Use dedicated HF interpreter')
        meta=json.loads((source/'results.json').read_text());first=meta['first_nonexact_state']
        if first['step']!=1 or not re.fullmatch('layer_[0-9]+',first['stage']):raise ValueError('Expected a divergent decoder layer')
        layer=int(first['stage'][6:]);report['layer']=layer
        if layer==0:raise ValueError('This study expects an earlier aligned layer')
        atom=Path(meta['atom_run'])/'decode/sample_000000';hf=source/'hf/sample_000000'
        hidden_cpu=torch.load(hf/f'step_001_layer_{layer-1:02d}.pt',weights_only=True,map_location='cpu')
        if not torch.equal(hidden_cpu,torch.load(atom/f'step_001_layer_{layer-1:02d}.pt',weights_only=True,map_location='cpu')):
            raise ValueError('Layer input differs')
        hf_context=torch.load(hf/f'layer_{layer:02d}_attention.pt',weights_only=True,map_location='cpu')
        atom_context=torch.load(atom/f'layer_{layer:02d}_attention.pt',weights_only=True,map_location='cpu')
        for key in ['query','keys','values','sdpa_output']:
            if not torch.equal(hf_context[key],atom_context[key]):raise ValueError('Attention not aligned: '+key)
        report['preconditions']={'layer_input_exact':True,'attention_qkv_context_exact':True,
            'hf_input_sha256':digest(hf/f'step_001_layer_{layer-1:02d}.pt'),
            'hf_attention_sha256':digest(hf/f'layer_{layer:02d}_attention.pt')}
        if a.prepare_only:
            report['status']='prepared';save();print('RESULT prepared output',out,flush=True);return 0
        if not torch.cuda.is_available():report['status']='blocked_no_gpu';save();return 2
        from safetensors import safe_open
        from transformers import MellumConfig
        from transformers.models.mellum.modeling_mellum import MellumRMSNorm,MellumExperts
        import_native_rope()
        from aiter.tuned_gemm import tgemm
        spec=importlib.util.spec_from_file_location('experts',ROOT/'ATOM/atom/model_config/mellum_experts.py')
        experts=importlib.util.module_from_spec(spec);spec.loader.exec_module(experts)
        model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');cfg=MellumConfig.from_pretrained(model,local_files_only=True)
        cfg._experts_implementation='grouped_mm'
        index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        def load(name):
            with safe_open(model/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name).to('cuda')
        def tensor(path):return torch.load(path,weights_only=True,map_location='cpu').to('cuda')
        hidden=tensor(hf/f'step_001_layer_{layer-1:02d}.pt')
        if not torch.equal(hidden,tensor(atom/f'step_001_layer_{layer-1:02d}.pt')):raise ValueError('Layer input differs')
        ha=torch.load(hf/f'layer_{layer:02d}_attention.pt',weights_only=True,map_location='cpu')
        aa=torch.load(atom/f'layer_{layer:02d}_attention.pt',weights_only=True,map_location='cpu')
        for key in ['query','keys','values','sdpa_output']:
            if not torch.equal(ha[key],aa[key]):raise ValueError('Attention input/context not aligned: '+key)
        context=ha['sdpa_output'].to('cuda').reshape(1,-1)
        prefix=f'model.layers.{layer}.'
        ow=load(prefix+'self_attn.o_proj.weight');nw=load(prefix+'post_attention_layernorm.weight');gw=load(prefix+'mlp.gate.weight')
        hn=MellumRMSNorm(cfg.hidden_size,cfg.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
        an=own_norm_class()(cfg.hidden_size,cfg.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
        hn.weight.data.copy_(nw);an.weight.data.copy_(nw)
        torch.set_grad_enabled(False)
        with torch.device('meta'):he=MellumExperts(cfg);ae=experts.MellumTorchExperts(cfg)
        print('STAGE load canonical experts for layer',layer,'(no full model)',flush=True)
        gate_up=torch.stack([torch.cat([load(prefix+f'mlp.experts.{e}.gate_proj.weight'),load(prefix+f'mlp.experts.{e}.up_proj.weight')]) for e in range(cfg.num_experts)])
        down=torch.stack([load(prefix+f'mlp.experts.{e}.down_proj.weight') for e in range(cfg.num_experts)])
        he.gate_up_proj=torch.nn.Parameter(gate_up,requires_grad=False);he.down_proj=torch.nn.Parameter(down,requires_grad=False)
        ae.w13_weight=torch.nn.Parameter(gate_up,requires_grad=False);ae.w2_weight=torch.nn.Parameter(down,requires_grad=False)
        print('STAGE replay O projection, norm, router and MoE',flush=True)
        ho=F.linear(context.reshape(1,1,-1),ow).reshape(1,-1)
        ao=tgemm.mm(context,ow,None,otype=torch.bfloat16)
        hr=hidden+ho;ar=hidden+ao
        hnorm=hn(hr.reshape(1,1,-1)).reshape(1,-1);anorm=an(ao,hidden)[0]
        hg=F.linear(hnorm,gw);ag=tgemm.mm(anorm,gw,None,otype=torch.bfloat16)
        def route(g):
            w,i=g.float().softmax(-1).topk(cfg.num_experts_per_tok,-1);w/=w.sum(-1,keepdim=True)
            return w.to(g.dtype),i
        hw,hi=route(hg);aw,ai=route(ag)
        hm=he(hnorm,hi,hw);am=ae(anorm,ag)
        same_input=ae(hnorm,hg)
        hf_expected=tensor(hf/f'step_001_layer_{layer:02d}.pt');atom_expected=tensor(atom/f'step_001_layer_{layer:02d}.pt')
        report.update(comparisons={'o_projection':compare(ao,ho),'residual':compare(ar,hr),
            'post_norm':compare(anorm,hnorm),'router_logits':compare(ag,hg),
            'moe_same_reference_input':compare(same_input,hm),'moe_natural_input':compare(am,hm),
            'hf_layer_replay':compare(hr+hm,hf_expected),'atom_layer_replay':compare(ar+am,atom_expected)},
            routing={'hf_ids':hi.cpu().tolist(),'atom_ids':ai.cpu().tolist(),'hf_weights':hw.float().cpu().tolist(),'atom_weights':aw.float().cpu().tolist()})
        torch.save({'hf_o':ho.cpu(),'atom_o':ao.cpu(),'hf_post_norm':hnorm.cpu(),'atom_post_norm':anorm.cpu(),
            'hf_router':hg.cpu(),'atom_router':ag.cpu(),'hf_moe':hm.cpu(),'atom_moe':am.cpu()},out/'intermediates.pt')
        report['interpretation_gate']='inspect_components' if report['comparisons']['hf_layer_replay']['exact'] and report['comparisons']['atom_layer_replay']['exact'] else 'resolve_reconstruction_mismatch_first'
        report['status']='diagnostic_complete';save();print(json.dumps(report,indent=2));print('RESULT diagnostic_complete output',out,flush=True);return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());save();print(report['traceback']);print('RESULT failed output',out,flush=True);return 1
if __name__=='__main__':raise SystemExit(main())
