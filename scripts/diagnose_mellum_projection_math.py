"""Fixed-input projection study with CPU preflight and an explicit decision gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path('/workspace')
DEFAULT_RUN = ROOT/'results/mellum_validation/20261111T140735437405Z'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_tensor(value, shape, name):
    import torch
    if tuple(value.shape) != tuple(shape) or value.dtype != torch.bfloat16:
        raise ValueError(f'{name}: expected BF16 {tuple(shape)}, got {value.dtype} {tuple(value.shape)}')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name}: nonfinite tensor')


def prepare(source):
    """Read only CPU tensors; return verified provenance and replay inputs."""
    import torch
    source = Path(source).resolve()
    reference_path = source/'hf/reference.json'
    reference = json.loads(reference_path.read_text())
    config_path = Path('/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json')
    config = json.loads(config_path.read_text())
    if digest(config_path) != reference['model_config_sha256']:
        raise ValueError('Reference/model config hash mismatch')
    meta = json.loads((source/'summary.json').read_text())
    if meta.get('stage_layer', 0) != 0:
        raise ValueError('This study currently accepts layer-zero captures only')
    stages = source/'hf/stages'
    records = reference['records']
    if len(records) != 4:
        raise ValueError('Expected the four fixed layer-zero probes')
    h = config['hidden_size']; d = config['head_dim']
    qdim = config['num_attention_heads']*d
    kvdim = config['num_key_value_heads']*d
    shapes = {'q_proj': (qdim,h), 'k_proj': (kvdim,h), 'v_proj': (kvdim,h),
              'o_proj': (h,qdim), 'router': (config['num_experts'],h)}
    hashes = {}
    def load(path, shape):
        value = torch.load(path, weights_only=True, map_location='cpu')
        check_tensor(value, shape, str(path))
        hashes[str(path)] = digest(path)
        return value
    weights = {name: load(stages/'weights'/(name+'.pt'), shape) for name,shape in shapes.items()}
    inputs = []
    for index, record in enumerate(records):
        sample = f'sample_{index:06d}'; n = len(record['input_ids'][0])
        widths = {'input_norm':h, 'q_proj':qdim, 'k_proj':kvdim, 'v_proj':kvdim,
                  'attention_context':qdim, 'attention_output':h,
                  'moe_input':h, 'router_logits':config['num_experts']}
        tensors = {name:load(stages/sample/(name+'.pt'), (n,width)) for name,width in widths.items()}
        inputs.append((sample, record['name'], tensors))
    manifest = {'source_run':str(source), 'reference_sha256':digest(reference_path),
                'source_reference_environment':reference.get('environment',{}),
                'model_config_sha256':digest(config_path), 'stage_layer':0,
                'stage_layer_evidence':'explicit metadata' if 'stage_layer' in meta else 'legacy default layer-zero capture',
                'sample_names':[{'sample':s,'name':name,'tokens':len(t['input_norm'])} for s,name,t in inputs],
                'tensor_files_sha256':hashes, 'weights_verified_by_this_study':list(weights),
                'scope':'Saved layer-zero QKV/O/router only; no lm_head, full model, cache or decode claim'}
    return manifest, weights, inputs


def compare(actual, expected):
    import torch
    from diagnose_mellum_expert_math import error
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Replay output shape/dtype mismatch')
    result = error(actual, expected)
    result['different_elements'] = int(torch.count_nonzero(actual != expected).item())
    result['elements'] = actual.numel()
    return result


def repeat_output(function, repeats):
    import torch
    # Snapshot the first result even if a backend reuses an output buffer.
    first = function().clone()
    checks = [torch.equal(first, function()) for _ in range(repeats-1)]
    return first, all(checks)


def summarize(records):
    if not records:
        raise ValueError('No projection records')
    reference_exact = all(
        all(v['exact'] for v in r['comparisons']['torch_separate'].values()) and
        all(r['comparisons'][name]['torch']['exact'] for name in ['o_proj','router'])
        for r in records)
    stable = all(all(r['repeat_stability'].values()) for r in records)
    return {'reference_replay_exact':reference_exact, 'all_repeats_exact':stable,
            'torch_packing_changes_outputs':any(not v['exact'] for r in records for v in r['packing_effect'].values()),
            'aiter_same_shape_changes_outputs':any(not v['exact'] for r in records for v in r['kernel_effect'].values()),
            'interpretation_gate':'inspect_projection_differences' if reference_exact and stable else 'resolve_reference_or_repeat_instability_first',
            'acceptance':'No correctness threshold or production backend is changed by this diagnostic'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeats', type=int, default=2)
    a = p.parse_args()
    if a.repeats < 2:p.error('--repeats must be at least 2')
    os.environ['HIP_VISIBLE_DEVICES'] = str(a.gpu)
    out = ROOT/'results/mellum_projection_math'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    report = {'status':'started','records':[], 'script_sha256':digest(__file__),
              'python':sys.executable, 'repeats':a.repeats}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(4)
        manifest, cpu_weights, inputs = prepare(a.source_run)
        report['manifest'] = manifest
        report['environment'] = {'torch':torch.__version__, 'hip':torch.version.hip,
                                 'cuda_initialized_before_replay':torch.cuda.is_initialized()}
        if a.prepare_only:
            report['status']='prepared';save();print('RESULT prepared output',out,flush=True);return 0
        if not torch.cuda.is_available():
            report['status']='blocked_no_gpu';save();print('RESULT blocked_no_gpu output',out,flush=True);return 2
        from diagnose_mellum_attention_math import import_native_rope
        import_native_rope()
        from aiter.tuned_gemm import tgemm
        report['environment'].update(aiter_source='/app/aiter-test',gpu_name=torch.cuda.get_device_name(0))
        weights={name:value.to('cuda') for name,value in cpu_weights.items()}
        names=['q_proj','k_proj','v_proj']; sizes=[weights[name].shape[0] for name in names]
        packed=torch.cat([weights[name] for name in names])
        with torch.inference_mode():
            for sample, name, tensors in inputs:
                tensors={key:value.to('cuda') for key,value in tensors.items()}
                x=tensors['input_norm']; rows={}; stability={}
                def run(label, function):
                    first, stability[label]=repeat_output(function,a.repeats)
                    return first
                tp=run('torch_packed',lambda:F.linear(x,packed))
                ap=run('aiter_packed',lambda:tgemm.mm(x,packed,None,otype=torch.bfloat16))
                ts={key:run('torch_'+key,lambda key=key:F.linear(x,weights[key])) for key in names}
                tps=dict(zip(names,tp.split(sizes,-1))); aps=dict(zip(names,ap.split(sizes,-1)))
                for label,outputs in [('torch_packed',tps),('aiter_packed',aps),('torch_separate',ts)]:
                    rows[label]={key:compare(value,tensors[key]) for key,value in outputs.items()}
                for key,input_name,target in [('o_proj','attention_context','attention_output'),('router','moe_input','router_logits')]:
                    z=tensors[input_name]
                    tv=run('torch_'+key,lambda:F.linear(z,weights[key]))
                    av=run('aiter_'+key,lambda:tgemm.mm(z,weights[key],None,otype=torch.bfloat16))
                    rows[key]={'torch':compare(tv,tensors[target]),'aiter':compare(av,tensors[target])}
                row={'sample':sample,'name':name,'comparisons':rows,'repeat_stability':stability,
                     'packing_effect':{key:compare(tps[key],ts[key]) for key in names},
                     'kernel_effect':{key:compare(aps[key],tps[key]) for key in names}}
                report['records'].append(row);save()
                print(sample,name,'torch separate exact:',all(v['exact'] for v in rows['torch_separate'].values()),flush=True)
        report['summary']=summarize(report['records'])
        report['status']='diagnostic_complete';save()
        print(json.dumps(report['summary']),flush=True)
        print('RESULT diagnostic_complete output',out,flush=True);return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());save()
        print(report['traceback']);print('RESULT failed output',out,flush=True);return 1
if __name__=='__main__':raise SystemExit(main())
