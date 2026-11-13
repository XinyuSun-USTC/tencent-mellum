"""Reconstruct HF layer-zero Q/K/V only and compare saved decode snapshots."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import types
import traceback
from diagnose_mellum_projection_math import compare,digest,check_tensor

ROOT=Path('/workspace')

def prepare_source(source):
    import torch
    run=json.loads((source/'results.json').read_text())
    if run['status'] not in ['forced_history_logits_exact','forced_history_logits_differ']:
        raise ValueError('Source baseline is incomplete')
    if run['atom_configuration'].get('MELLUM_SDPA_DECODE')!='1':raise ValueError('Source has no paired SDPA decode snapshots')
    atom=Path(run['atom_run']);reference=json.loads(Path(run['atom_configuration']['MELLUM_FORCED_DECODE_PLAN']).read_text())
    files=[]
    for i,record in enumerate(reference['records']):
        file=atom/'decode'/f'sample_{i:06d}'/'first_decode_attention.pt'
        saved=torch.load(file,map_location='cpu',weights_only=True);n=len(record['input_ids'][0])+1
        for key,shape in [('query',(1,32,128)),('keys',(n,4,128)),('values',(n,4,128)),
                          ('sdpa_output',(1,32,128)),('native_output',(1,32,128))]:
            check_tensor(saved[key],shape,key)
        if saved['logical_start']!=0 or saved['logical_end']!=n:raise ValueError('Unexpected snapshot visible range')
        files.append(file)
    if len(files)!=4:raise ValueError('Expected four paired snapshots')
    return reference,atom,files

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu);source=a.run.resolve()
    out=ROOT/'results/mellum_decode_boundary'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    report={'status':'started','source_run':str(source),'script_sha256':digest(__file__),'records':[],
            'scope':'First-layer first-decode Q/K/V reconstruction only; no full-model execution'}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import torch,transformers
        import torch.nn.functional as F
        if transformers.__version__!='5.10.1':raise ValueError('Use dedicated HF 5.10.1')
        reference,atom,files=prepare_source(source)
        report['snapshots']=[{'path':str(f),'sha256':digest(f)} for f in files]
        if a.prepare_only:
            report['status']='prepared';save();print('RESULT prepared output',out,flush=True);return 0
        if not torch.cuda.is_available():report['status']='blocked_no_gpu';save();return 2
        from safetensors import safe_open
        from transformers import MellumConfig,DynamicCache
        from transformers.models.mellum.modeling_mellum import MellumRMSNorm,MellumRotaryEmbedding,apply_rotary_pos_emb
        from transformers.masking_utils import create_sliding_window_causal_mask
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');cfg=MellumConfig.from_pretrained(model,local_files_only=True)
        cfg._attn_implementation='sdpa'
        if digest(model/'config.json')!=reference['model_config_sha256']:raise ValueError('Config hash differs')
        index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        def load(name,device='cuda'):
            with safe_open(model/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name).to(device)
        embedding=load('model.embed_tokens.weight','cpu')
        norms=[]
        for name,dim in [('input_layernorm',cfg.hidden_size),('self_attn.q_norm',cfg.head_dim),('self_attn.k_norm',cfg.head_dim)]:
            norm=MellumRMSNorm(dim,cfg.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
            norm.weight.data.copy_(load('model.layers.0.'+name+'.weight'));norms.append(norm)
        norm,qn,kn=norms
        weights={name:load('model.layers.0.self_attn.'+name+'_proj.weight') for name in ['q','k','v']}
        rotary=MellumRotaryEmbedding(cfg).to('cuda')
        def project(ids,positions):
            hidden=F.embedding(torch.tensor([ids]),embedding).to('cuda')
            hidden=norm(hidden);n=len(ids)
            q=qn(F.linear(hidden,weights['q']).view(1,n,cfg.num_attention_heads,cfg.head_dim)).transpose(1,2)
            k=kn(F.linear(hidden,weights['k']).view(1,n,cfg.num_key_value_heads,cfg.head_dim)).transpose(1,2)
            v=F.linear(hidden,weights['v']).view(1,n,cfg.num_key_value_heads,cfg.head_dim).transpose(1,2)
            cos,sin=rotary(hidden,positions,layer_type='sliding_attention')
            q,k=apply_rotary_pos_emb(q,k,cos,sin)
            return hidden,q,k,v
        with torch.inference_mode():
            for i,record in enumerate(reference['records']):
                prompt=record['input_ids'][0];n=len(prompt)
                _,_,pk,pv=project(prompt,torch.arange(n,device='cuda')[None])
                token=record['generated_ids'][0][0];pos=torch.tensor([[n]],device='cuda')
                hidden,q,k,v=project([token],pos)
                cache=DynamicCache(config=cfg);cache.update(pk,pv,0)
                mask=create_sliding_window_causal_mask(config=cfg,inputs_embeds=hidden,
                    attention_mask=torch.ones(1,n+1,device='cuda',dtype=torch.long),past_key_values=cache,position_ids=pos)
                k,v=cache.update(k,v,0)
                output,_=sdpa_attention_forward(types.SimpleNamespace(num_key_value_groups=cfg.num_attention_heads//cfg.num_key_value_heads,is_causal=True),
                    q,k,v,mask,scaling=cfg.head_dim**-0.5)
                file=atom/'decode'/f'sample_{i:06d}'/'first_decode_attention.pt'
                saved=torch.load(file,map_location='cpu',weights_only=True)
                refs={'query':q[0].transpose(0,1).cpu(),'keys':k[0].transpose(0,1).cpu(),
                      'values':v[0].transpose(0,1).cpu(),'sdpa_output':output.reshape(1,cfg.num_attention_heads,cfg.head_dim).cpu()}
                row={'name':record['name'],'logical_context_tokens':n+1,'snapshot_sha256':digest(file),
                    'mask_is_none':mask is None,'mask_all_true':None if mask is None else bool(mask.all().item()),
                    'comparisons':{key:compare(saved[key],value) for key,value in refs.items()}}
                report['records'].append(row);save();print(record['name'],json.dumps(row),flush=True)
                torch.save({**refs,'mask':None if mask is None else mask.cpu()},out/f'sample_{i:06d}_reference.pt')
        report['status']='diagnostic_complete';save();print('RESULT diagnostic_complete output',out,flush=True);return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());save();print(report['traceback']);print('RESULT failed output',out,flush=True);return 1
if __name__=='__main__':raise SystemExit(main())
