"""HF cached and uncached logits under the same forced token history."""
import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import traceback
from prepare_mellum_suite import validate_inputs
from validate_hf_mellum_gpu import write_result


def export_case(model,record,steps,out,index,capture=None):
    import torch
    device=next(model.parameters()).device
    history=list(record['input_ids'][0]);cache=None;forced=[];observations=[]
    path=out/f'sample_{index:06d}';path.mkdir()
    for step in range(steps):
        ids=history if step==0 else [history[-1]]
        positions=list(range(len(history))) if step==0 else [len(history)-1]
        args={'input_ids':torch.tensor([ids],device=device),
              'attention_mask':torch.ones(1,len(history),device=device,dtype=torch.long),
              'position_ids':torch.tensor([positions],device=device),'logits_to_keep':1}
        if capture is not None:capture.update(active=step in (0,1),step=step,path=path)
        cached=model(**args,past_key_values=cache,use_cache=True)
        if capture is not None:capture['active']=False
        cache=cached.past_key_values;logits=cached.logits[0,-1]
        uncached=model(input_ids=torch.tensor([history],device=device),
            attention_mask=args['attention_mask'],position_ids=torch.arange(len(history),device=device)[None],
            use_cache=False,logits_to_keep=1).logits[0,-1]
        if not torch.isfinite(logits).all() or not torch.isfinite(uncached).all():raise ValueError('Nonfinite HF logits')
        cached_file=path/f'step_{step:03d}_cached.pt';full_file=path/f'step_{step:03d}_uncached.pt'
        torch.save(logits.detach().cpu(),cached_file);torch.save(uncached.detach().cpu(),full_file)
        physical={}
        for layer in [0,min(3,len(cache.layers)-1)]:
            keys=getattr(cache.layers[layer],'keys',None)
            physical[str(layer)]=None if keys is None else list(keys.shape)
        token=int(logits.argmax().item());forced.append(token)
        observations.append({'step':step,'input_ids':list(ids),'positions':list(positions),
            'logical_context_tokens':len(history),'forced_token':token,'hf_cache_key_shapes':physical,
            'cached_logits':str(cached_file.resolve()),'uncached_logits':str(full_file.resolve()),
            'cached_sha256':hashlib.sha256(cached_file.read_bytes()).hexdigest(),
            'uncached_sha256':hashlib.sha256(full_file.read_bytes()).hexdigest()})
        history.append(token)
    row=copy.deepcopy(record);row['generation_settings']['max_new_tokens']=steps
    row.update(generated_ids=[forced],decode_steps=observations,
        prefill_logits=observations[0]['cached_logits'],prefill_logits_sha256=observations[0]['cached_sha256'],
        logits_semantics='last_prompt_position_full_vocab')
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--dump-first-decode',action='store_true');a=p.parse_args()
    if a.steps<2:p.error('At least two steps required')
    os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu);os.environ['HF_HUB_OFFLINE']='1'
    a.output.mkdir(parents=True);report={'status':'started'}
    def save():write_result(a.output,report)
    try:
        import torch,transformers
        if transformers.__version__!='5.10.1':raise ValueError('Use dedicated HF 5.10.1')
        inputs=json.loads(a.inputs.read_text());validate_inputs(inputs)
        if not torch.cuda.is_available():report['status']='blocked_no_gpu';save();return 2
        model_path=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
        if hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest()!=inputs['model_config_sha256']:
            raise ValueError('Config hash differs')
        from transformers import MellumForCausalLM
        model,info=MellumForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='sdpa',experts_implementation='grouped_mm',output_loading_info=True)
        if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('HF load audit failed')
        model=model.to('cuda').eval()
        reference={**inputs,'producer':'hf','records':[],
            'environment':{'transformers':transformers.__version__,'torch':torch.__version__,
                           'experts_implementation':model.config._experts_implementation},
            'scope':'Forced history; EOS may be fed beyond normal stopping, not free generation acceptance'}
        report['loading_info']=info;save()
        from mellum_decode_observers import observe_decode_layers
        with torch.inference_mode(), (observe_decode_layers(model) if a.dump_first_decode else nullcontext(None)) as capture:
            for i,record in enumerate(inputs['records']):
                reference['records'].append(export_case(model,record,a.steps,a.output,i,capture))
                (a.output/'reference.json').write_text(json.dumps(reference,indent=2)+'\n')
                print(record['name'],'cached/uncached',a.steps,'steps saved',flush=True)
        report['status']='passed';save();return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());save();print(report['traceback']);return 1
if __name__=='__main__':raise SystemExit(main())
