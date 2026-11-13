"""Same-QKV first-layer RoPE/SDPA replay, using the dedicated HF interpreter."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
import sys
from pathlib import Path
import traceback
import types
from diagnose_mellum_expert_math import error

ROOT=Path('/workspace')
def import_native_rope():
    # The frozen interpreter finds this editable source through its .pth file;
    # the isolated HF environment does not execute that editable installation.
    source=Path('/app/aiter-test')
    if not (source/'aiter/rotary_embedding.py').is_file():raise FileNotFoundError(source)
    sys.path.insert(0,str(source))
    # Reuse the container's existing extensions rather than triggering a build.
    sys.path.insert(1,str(source/'aiter/jit'))
    from aiter.rotary_embedding import get_rope
    return get_rope

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu)
    out=ROOT/'results/mellum_attention_math'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    report={'status':'started','scope':'Same-QKV first-layer replay; no engine acceptance','records':[]}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import torch,transformers
        if transformers.__version__!='5.10.1':raise RuntimeError('Use the dedicated HF interpreter')
        if not torch.cuda.is_available():report['status']='blocked_no_gpu';save();return 2
        from transformers import MellumConfig
        from transformers.models.mellum.modeling_mellum import MellumRotaryEmbedding,apply_rotary_pos_emb
        from transformers.masking_utils import create_sliding_window_causal_mask
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        get_rope=import_native_rope()
        from aiter import flash_attn_varlen_func
        from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
        report['aiter_source']='/app/aiter-test'
        spec=importlib.util.spec_from_file_location('rope_bridge',ROOT/'ATOM/atom/model_config/mellum_utils.py')
        bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
        cfg=MellumConfig.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking',local_files_only=True)
        cfg._attn_implementation='sdpa'
        rotary=MellumRotaryEmbedding(cfg).to('cuda')
        params=bridge.aiter_rope_parameters(cfg.rope_parameters['sliding_attention'])
        with torch.device('cuda'):
            native=get_rope(128,rotary_dim=128,max_position=256,base=params['rope_theta'],rope_scaling=params,dtype=torch.bfloat16)
        source=ROOT/'results/mellum_validation/20261111T140735437405Z/hf/stages'
        atom=ROOT/'results/atom_mellum_gpu/20261111T140748118711Z/stages'
        torch.set_grad_enabled(False)
        for sample in sorted(source.glob('sample_*')):
            def load(name,root=sample):return torch.load(root/(name+'.pt'),weights_only=True,map_location='cpu').to('cuda')
            qflat,kflat,vflat=load('q_norm'),load('k_norm'),load('v_proj');n=len(qflat)
            pos=torch.arange(n,device='cuda');q=qflat.view(1,n,32,128).transpose(1,2);k=kflat.view(1,n,4,128).transpose(1,2)
            v=vflat.view(1,n,4,128).transpose(1,2)
            cos,sin=rotary(q,pos[None],layer_type='sliding_attention')
            hq,hk=apply_rotary_pos_emb(q,k,cos,sin)
            aq,ak=native(pos,qflat.clone(),kflat.clone())
            aq=aq.view(1,n,32,128).transpose(1,2);ak=ak.view(1,n,4,128).transpose(1,2)
            # Actual layer-zero sliding-attention branch in PagedAttentionImpl.rope_cache.
            fq=qflat.view(n,32,128).clone();fk=kflat.view(n,4,128).clone()
            blocks=(n+15)//16
            kc=torch.empty(blocks,4,16,16,8,device='cuda',dtype=q.dtype)
            vc=torch.empty(blocks,4,128,16,device='cuda',dtype=q.dtype)
            fq,fk,_,_=fused_qk_rope_reshape_and_cache(fq,fk,vflat.view(n,4,128),kc,vc,
                pos,pos,native.cos_cache,native.sin_cache,None,None,native.is_neox_style,
                flash_layout=False,apply_scale=False,offs=None,q_out=fq,k_out=fk,output_zeros=False)
            fq=fq[None].transpose(1,2);fk=fk[None].transpose(1,2)
            mask=create_sliding_window_causal_mask(config=cfg,inputs_embeds=torch.zeros(1,n,cfg.hidden_size,device='cuda',dtype=q.dtype),
                attention_mask=torch.ones(1,n,device='cuda',dtype=torch.long),past_key_values=None,position_ids=pos[None])
            module=types.SimpleNamespace(num_key_value_groups=8,is_causal=True)
            contexts={}
            for name,rq,rk in [('hf_rope',hq,hk),('aiter_rope',aq,ak),('atom_fused_rope',fq,fk)]:
                value,_=sdpa_attention_forward(module,rq,rk,v,mask,scaling=128**-0.5)
                contexts[name]=value.reshape(n,-1)
            cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
            for name,rq,rk in [('atom_prefill_hf_rope',hq,hk),('atom_prefill_fused_rope',fq,fk)]:
                contexts[name]=flash_attn_varlen_func(
                    rq.transpose(1,2).reshape(n,32,128).contiguous(),
                    rk.transpose(1,2).reshape(n,4,128).contiguous(),vflat.view(n,4,128).contiguous(),
                    cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=n,max_seqlen_k=n,min_seqlen_q=n,
                    dropout_p=0.0,softmax_scale=128**-0.5,causal=True,window_size=(1024,0,0),sink_ptr=None).reshape(n,-1)
            expected=load('attention_context');old_atom=load('attention_context',atom/sample.name)
            row={'sample':sample.name,'mask_is_none':mask is None,'q_rope':error(aq,hq),'k_rope':error(ak,hk),
                 'fused_q_rope':error(fq,hq),'fused_k_rope':error(fk,hk),
                 'contexts':{name:{'hf':error(value,expected),'old_atom':error(value,old_atom)} for name,value in contexts.items()}}
            report['records'].append(row);save();print(sample.name,json.dumps(row),flush=True)
            torch.save({'hf_q':hq.cpu(),'hf_k':hk.cpu(),'aiter_q':aq.cpu(),'aiter_k':ak.cpu(),
                        'fused_q':fq.cpu(),'fused_k':fk.cpu()},out/(sample.name+'_rope.pt'))
        if len(report['records'])!=4:raise ValueError('Expected four probes')
        report['status']='diagnostic_complete';save();print('RESULT diagnostic_complete output',out,flush=True);return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());save();print(report['traceback']);return 1
if __name__=='__main__':raise SystemExit(main())
