"""Compare official HF manual cached forward and generate on one model instance."""
import argparse
from datetime import datetime,timezone
import json,os
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case',default='python_add');a=p.parse_args()
    import torch,transformers
    from transformers import MellumForCausalLM
    from export_mellum_decode_reference import export_case
    from mellum_decode_observers import observe_decode_layers
    if transformers.__version__!='5.10.1':raise ValueError('Use HF reference interpreter')
    data=json.loads((a.run/'inputs.json').read_text());row=next(r for r in data['records'] if r['name']==a.case)
    out=Path('/workspace/results/mellum_hf_generation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    model=MellumForCausalLM.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='grouped_mm').to('cuda').eval()
    manual=out/'manual';manual.mkdir();generated=out/'generate';generated.mkdir()
    calls=[];handles=[]
    def tensor_info(t):
        return {'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'values':t.detach().cpu().tolist()}
    def before(module,args,kwargs):
        step=len(calls);capture.update(active=True,step=step,path=generated)
        cache=kwargs.get('past_key_values')
        calls.append({'step':step,'inputs':{k:tensor_info(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)},
                      'experts_implementation':model.config._experts_implementation,'cache_type':type(cache).__name__,'cache_layers':[] if cache is None else [type(x).__name__ for x in cache.layers],
                      'scalar_kwargs':{k:v for k,v in kwargs.items() if v is None or isinstance(v,(str,int,float,bool))}})
    with torch.inference_mode(),observe_decode_layers(model) as capture:
        ref=export_case(model,row,2,manual,0,capture)
        handles.append(model.register_forward_pre_hook(before,with_kwargs=True))
        try:
            settings={**row['generation_settings'],'max_new_tokens':2}
            g=model.generate(input_ids=torch.tensor(row['input_ids'],device='cuda'),attention_mask=torch.tensor(row['attention_mask'],device='cuda'),**settings,use_cache=True,return_dict_in_generate=True,output_logits=True)
        finally:
            for h in handles:h.remove()
            capture['active']=False
    results=[]
    def compare(x,y):
        x=x.float();y=y.float();return {'exact':torch.equal(x,y),'max_abs':float((x-y).abs().max())}
    for step,v in enumerate(g.logits):
        torch.save(v[0].cpu(),generated/f'step_{step:03d}_logits.pt')
        results.append({'step':step,**compare(v[0].cpu(),torch.load(ref['decode_steps'][step]['cached_logits'],weights_only=True))})
    states=[]
    for f in sorted((manual/'sample_000000').glob('step_*_*.pt')):
        other=generated/f.name
        if other.exists():states.append({'file':f.name,**compare(torch.load(f,weights_only=True),torch.load(other,weights_only=True))})
    attention=[]
    for f in sorted((manual/'sample_000000').glob('layer_*_attention.pt')):
        left=torch.load(f,weights_only=True);right=torch.load(generated/f.name,weights_only=True)
        for key in ['query','keys','values','sdpa_output']:
            attention.append({'file':f.name,'stage':key,**compare(left[key],right[key])})
    states.sort(key=lambda r:(r['file'][5:8], 0 if 'embedding' in r['file'] else 29 if 'final_norm' in r['file'] else int(r['file'].split('_')[-1].split('.')[0])+1))
    report={'attention':attention,'first_nonexact_attention':next((r for r in attention if not r['exact']),None),'status':'diagnostic_complete','calls':calls,'logits':results,'first_nonexact_state':next((r for r in states if not r['exact']),None),'states':states,'manual_tokens':ref['generated_ids'],'generate_tokens':g.sequences[:,len(row['input_ids'][0]):].cpu().tolist()}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT diagnostic_complete output',out,flush=True)
if __name__=='__main__':main()
