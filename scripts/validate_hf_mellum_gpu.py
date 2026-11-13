"""Single-GPU BF16 HF checkpoint smoke test; no ATOM imports or remote downloads."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import traceback

MODEL = Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
ROOT = Path('/workspace')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def json_default(value):
    # HF loading reports contain sets, including non-empty failure reports.
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported diagnostic value: {type(value).__name__}')


def write_result(out, result):
    # Serialize before touching disk; replace atomically to retain the last
    # valid report if serialization or writing fails.
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + '\n'
    temporary = out / 'results.json.tmp'
    temporary.write_text(payload)
    temporary.replace(out / 'results.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='Logical GPU index; default selects the most free memory')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        parser.error('Use 1..64 tokens for the initial short-input smoke test')
    out = ROOT/'results/hf_mellum_gpu'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    result = {'status': 'started', 'argv': sys.argv, 'output_dir': str(out), 'phase': 'preflight',
              'script_sha256': digest(Path(__file__)), 'results': [],
              'scope': 'Short-input smoke and HF internal consistency; not ATOM or official acceptance'}
    def save():
        write_result(out, result)
    try:
        import torch
        import transformers
        if transformers.__version__ != '5.10.1' or not str(Path(transformers.__file__).resolve()).startswith('/workspace/venvs/hf-reference/'):
            raise RuntimeError('Run with /workspace/venvs/hf-reference/bin/python (local Transformers5.10.1 required)')
        result['environment'] = {'python':sys.executable, 'transformers':transformers.__version__,
             'transformers_file':transformers.__file__, 'torch':torch.__version__, 'torch_file':torch.__file__,
             'hip':torch.version.hip, 'visible_device_count':torch.cuda.device_count(),
             'packages':{n:metadata.version(n) for n in ['torch','transformers','huggingface-hub','tokenizers','safetensors']}}
        result['source_hashes'] = {str(p.relative_to(Path(transformers.__file__).parent)):digest(p) for p in [
            Path(transformers.__file__).parent/'models/mellum/modeling_mellum.py',
            Path(transformers.__file__).parent/'models/mellum/configuration_mellum.py']}
        if not torch.cuda.is_available():
            result.update(status='blocked_no_gpu', reason='No accessible ROCm GPU; checkpoint was not loaded')
            save(); print(json.dumps(result,ensure_ascii=False,indent=2)); return 2
        result['gpus'] = []
        for i in range(torch.cuda.device_count()):
            free,total = torch.cuda.mem_get_info(i)
            result['gpus'].append({'index':i,'name':torch.cuda.get_device_name(i),'free_bytes':free,'total_bytes':total})
        selected = args.gpu if args.gpu is not None else max(result['gpus'],key=lambda g:g['free_bytes'])['index']
        if not 0 <= selected < len(result['gpus']):raise ValueError('GPU index out of range')
        if result['gpus'][selected]['free_bytes'] < 40*1024**3:raise RuntimeError('Require >=40 GiB free on selected GPU; no checkpoint loaded')
        result['selected_gpu'] = selected
        if args.preflight_only:
            result['status']='preflight_passed';save();print(json.dumps(result,ensure_ascii=False,indent=2));return 0
        from transformers import AutoTokenizer, MellumForCausalLM
        torch.manual_seed(20261111)
        torch.cuda.set_device(selected)
        device=torch.device('cuda',selected)
        result['input_file_hashes']={n:digest(MODEL/n) for n in ['config.json','model.safetensors.index.json','generation_config.json','chat_template.jinja','tokenizer_config.json']}
        index=json.loads((MODEL/'model.safetensors.index.json').read_text())
        expected_parameters=index['metadata']['total_size']//2
        result['phase']='loading_checkpoint';save()
        print('Loading local BF16 checkpoint on CPU, then moving to',device,flush=True)
        start=time.monotonic()
        # No device_map/accelerate and no changes to frozen dependencies.
        model,info=MellumForCausalLM.from_pretrained(MODEL, local_files_only=True,
                  dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
        result['loading_info']={k:v for k,v in info.items()}
        save()
        for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:
            if info.get(key):raise RuntimeError(f'Checkpoint audit failed: {key}; inspect results.json')
        result['parameter_count']=sum(p.numel() for p in model.parameters())
        if result['parameter_count']!=expected_parameters:raise RuntimeError('Parameter count differs from BF16 index payload element count')
        if any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise RuntimeError('Non-BF16 parameter found')
        model=model.to(device).eval()
        torch.cuda.synchronize()
        result['load_seconds']=time.monotonic()-start
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        tests=[('python_add','Write a Python function add(a, b) that returns their sum.'),
               ('arithmetic','What is 17 + 25?')]
        result['phase']='short_input_inference';save()
        for name,prompt in tests:
            messages=[{'role':'user','content':prompt}]
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            inputs=tokenizer(rendered,add_special_tokens=False,return_tensors='pt').to(device)
            settings={'do_sample':False,'max_new_tokens':args.max_new_tokens,
                      'eos_token_id':model.generation_config.eos_token_id,'pad_token_id':tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.generation_config.eos_token_id}
            if not isinstance(settings['pad_token_id'],int):settings['pad_token_id']=28
            row={'name':name,'messages':messages,'enable_thinking':True,'rendered_prompt':rendered,
                 'input_ids':inputs.input_ids.cpu().tolist(),'attention_mask':inputs.attention_mask.cpu().tolist(),
                 'generation_settings':settings,'attention_backend':'sdpa'}
            with torch.inference_mode():
                prefill=model(**inputs,use_cache=False,logits_to_keep=1).logits
                if not torch.isfinite(prefill).all():raise RuntimeError('Non-finite prefill logits')
                values,indices=prefill[0,-1].float().topk(5)
                row['prefill_top5']={'ids':indices.cpu().tolist(),'logits':values.cpu().tolist()}
                torch.save(prefill[0,-1].detach().cpu(),out/f'{name}_prefill_logits.pt')
                started=time.monotonic()
                cached=model.generate(**inputs,**settings,use_cache=True)
                torch.cuda.synchronize();row['cached_generation_seconds']=time.monotonic()-started
                uncached=model.generate(**inputs,**settings,use_cache=False)
                torch.cuda.synchronize()
            start_token=inputs.input_ids.shape[1]
            row['generated_ids']=cached[:,start_token:].cpu().tolist()
            row['uncached_generated_ids']=uncached[:,start_token:].cpu().tolist()
            row['cache_greedy_exact']=torch.equal(cached,uncached)
            row['generated_text']=tokenizer.decode(cached[0,start_token:],skip_special_tokens=False)
            result['results'].append(row);save()
            print(name,'cache exact:',row['cache_greedy_exact'],'text:',repr(row['generated_text']),flush=True)
            if not row['cache_greedy_exact']:raise RuntimeError('Cached/uncached greedy divergence; retain outputs and diagnose before retry')
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        result.update(status='short_input_smoke_passed',phase='complete');save()
        print('RESULT',result['status'],'output',out,flush=True)
        return 0
    except Exception:
        original_traceback = traceback.format_exc()
        result.update(status='failed', traceback=original_traceback)
        print(original_traceback, file=sys.stderr)
        try:
            save()
            print('Saved diagnostic:', out, file=sys.stderr)
        except Exception:
            print('JSON diagnostic save failed:', traceback.format_exc(), file=sys.stderr)
            try:
                (out / 'failure_traceback.txt').write_text(original_traceback)
            except Exception:
                print('Text fallback also failed; original traceback is above.', file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())
