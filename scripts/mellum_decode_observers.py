"""Temporary HF observation only; preserve SDPA registry and model arithmetic."""
from contextlib import contextmanager
from pathlib import Path
import torch

@contextmanager
def observe_decode_layers(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    capture={'active':False,'step':-1,'path':None};handles=[]
    def hook(name):
        def save(module,args,output):
            if not capture['active']:return
            value=output[0] if isinstance(output,tuple) else output
            torch.save(value.detach().reshape(-1,value.shape[-1]).cpu(),
                Path(capture['path'])/f"step_{capture['step']:03d}_{name}.pt")
        return save
    handles.append(model.model.embed_tokens.register_forward_hook(hook('embedding')))
    handles.append(model.model.norm.register_forward_hook(hook('final_norm')))
    for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(hook(f'layer_{i:02d}')))
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    def sdpa(module,query,key,value,attention_mask,**kwargs):
        output=original(module,query,key,value,attention_mask,**kwargs)
        if capture['active'] and capture['step']==1:
            target=Path(capture['path'])/f'layer_{module.layer_idx:02d}_attention.pt'
            torch.save({'query':query[0].transpose(0,1).detach().cpu(),
                'keys':key[0].transpose(0,1).detach().cpu(),'values':value[0].transpose(0,1).detach().cpu(),
                'sdpa_output':output[0].reshape(query.shape[2],query.shape[1],query.shape[3]).detach().cpu(),
                'mask':None if attention_mask is None else attention_mask.detach().cpu()},target)
        return output
    ALL_ATTENTION_FUNCTIONS.register('sdpa',sdpa)
    try:yield capture
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
        for handle in handles:handle.remove()
