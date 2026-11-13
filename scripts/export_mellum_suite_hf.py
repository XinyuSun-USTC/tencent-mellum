"""HF GPU reference for a fixed-input suite; emits the schema-v1 comparison artifact."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import hashlib
import json
from pathlib import Path
import traceback
from validate_hf_mellum_gpu import write_result
from prepare_mellum_suite import validate_inputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--dump-layers',action='store_true')
    p.add_argument('--dump-stages',action='store_true')
    p.add_argument('--dump-generation-logits',action='store_true')
    p.add_argument('--stage-layer',type=int,choices=range(28),default=0)
    a=p.parse_args()
    if a.dump_stages and not a.dump_layers:p.error('--dump-stages requires --dump-layers')
    os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu)
    a.output.mkdir(parents=True,exist_ok=False)
    inputs=json.loads(a.inputs.read_text());validate_inputs(inputs)
    report={'status':'started','stage_layer':a.stage_layer,'records':[]}
    try:
        import torch,transformers
        if transformers.__version__!='5.10.1':raise RuntimeError('Use the dedicated HF reference interpreter')
        if not torch.cuda.is_available():report['status']='blocked_no_gpu';write_result(a.output,report);return 2
        if torch.cuda.mem_get_info(0)[0]<60*1024**3:raise RuntimeError('Require60GiB free for long-input diagnostics')
        from transformers import MellumForCausalLM
        model_path=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
        actual_hash=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest()
        if inputs['model_config_sha256']!=actual_hash:raise ValueError('Input/config hash differs')
        model,info=MellumForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='grouped_mm',output_loading_info=True)
        report['loading_info']=info
        report['experts_implementation']=model.config._experts_implementation
        write_result(a.output,report)
        if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError('HF loading audit failed')
        model=model.to('cuda:0').eval()
        reference={**inputs,'producer':'hf','environment':{'transformers':transformers.__version__,'torch':torch.__version__,
                   'experts_implementation':model.config._experts_implementation},'records':[]}
        expert_calls=[]
        def expert_call(module,args,kwargs):
            expert_calls.append(model.config._experts_implementation)
        expert_handle=model.register_forward_pre_hook(expert_call,with_kwargs=True)
        capture={'enabled':False,'path':None}
        handles=[]
        if a.dump_layers:
            def hook(name):
                def save_hook(module,args,output):
                    if capture['enabled']:
                        value=output[0] if isinstance(output,tuple) else output
                        if value.ndim==3:value=value[0]
                        torch.save(value.detach().cpu(),capture['path']/(name+'.pt'))
                return save_hook
            handles.append(model.model.embed_tokens.register_forward_hook(hook('embedding')))
            handles.append(model.model.norm.register_forward_hook(hook('final_norm')))
            for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(hook(f'layer_{i:02d}')))
        if a.dump_stages:
            import importlib.util
            spec=importlib.util.spec_from_file_location('stage_observers','/workspace/ATOM/atom/model_config/mellum_diagnostics.py')
            observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
            def stage_path():
                return a.output/'stages'/capture['path'].name if capture['enabled'] else None
            handles.extend(observer.install_stage_hooks(model.model.layers[a.stage_layer],stage_path,'hf'))
        for index,record in enumerate(inputs['records']):
            if a.dump_layers:
                capture['path']=a.output/'layers'/f'sample_{index:06d}'
                capture['path'].mkdir(parents=True)

            ids=torch.tensor(record['input_ids'],device='cuda',dtype=torch.long)
            mask=torch.tensor(record['attention_mask'],device='cuda',dtype=torch.long)
            position=torch.tensor(record['position_ids'],device='cuda',dtype=torch.long)
            with torch.inference_mode():
                capture['enabled']=a.dump_layers
                logits=model(input_ids=ids,attention_mask=mask,position_ids=position,use_cache=False,logits_to_keep=1).logits[0,-1]
                capture['enabled']=False
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite prefill logits')
                file=a.output/(record['name']+'_prefill.pt');torch.save(logits.cpu(),file)
                expert_calls.clear()
                generation=model.generate(input_ids=ids,attention_mask=mask,**record['generation_settings'],use_cache=True,
                    **({'return_dict_in_generate':True,'output_logits':True} if a.dump_generation_logits else {}))
                generated=generation.sequences if a.dump_generation_logits else generation
            row={**record,'generation_experts_by_forward':list(expert_calls),'generated_ids':generated[:,ids.shape[1]:].cpu().tolist(),
                 'prefill_logits':file.name,'prefill_logits_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                 'logits_semantics':'last_prompt_position_full_vocab'}
            if a.dump_generation_logits:
                files=[]
                for step,value in enumerate(generation.logits):
                    file=a.output/(record['name']+f'_generation_{step:03d}.pt');torch.save(value[0].detach().cpu(),file)
                    files.append({'path':str(file.resolve()),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
                row['generation_logits']=files
            reference['records'].append(row)
            report['records'].append({'name':record['name'],'input_tokens':ids.shape[1],'generated_ids':row['generated_ids']})
            (a.output/'reference.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n')
            write_result(a.output,report);print(record['name'],'reference saved',flush=True)
        report['status']='passed';write_result(a.output,report);return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());write_result(a.output,report)
        print(report['traceback']);return 1
if __name__=='__main__':raise SystemExit(main())
