"""Capture two real first-layer QKV samples and replay bounded attention controls."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from mellum_runtime_config import PROJECT
from mellum_process_runner import run_stage


def oracle(q,k,v,window,scale,block=128):
    """Explicit FP32 matmul/softmax, contiguous Q-head groups mapped to KV heads."""
    import torch
    n,h,d=q.shape;groups=h//k.shape[1];parts=[]
    kh=k.float().repeat_interleave(groups,1).transpose(0,1)
    vh=v.float().repeat_interleave(groups,1).transpose(0,1)
    keys=torch.arange(n,device=q.device)
    for start in range(0,n,block):
        end=min(n,start+block);queries=torch.arange(start,end,device=q.device)
        mask=(keys[None]<=queries[:,None])&(keys[None]>=queries[:,None]-window+1)
        scores=(q[start:end].float().transpose(0,1)@kh.transpose(-1,-2))*scale
        probs=scores.masked_fill(~mask[None],float('-inf')).softmax(-1)
        parts.append((probs@vh).transpose(0,1))
    return torch.cat(parts)


def metrics(a,b):
    import torch
    delta=a.float()-b.float()
    return {'exact':torch.equal(a,b),'max_abs':delta.abs().max().item(),
            'relative_l2':(delta.norm()/b.float().norm().clamp_min(1e-30)).item()}


def replay(output):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.set_grad_enabled(False)
    spec=importlib.util.spec_from_file_location('swa',PROJECT/'ATOM/atom/model_config/mellum_attention.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for path in sorted((output/'qkv').glob('*.pt')):
        data=torch.load(path,weights_only=True,map_location='cpu');q,k,v=[data[x].cuda() for x in ['q','k','v']]
        n=q.shape[0];w=data['window'];scale=data['scale'];cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
        mask=torch.ones(n,n,device='cuda',dtype=torch.bool).tril().triu(1-w)[None,None]
        ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],
            k.repeat_interleave(8,1).transpose(0,1)[None],v.repeat_interleave(8,1).transpose(0,1)[None],
            attn_mask=mask,scale=scale)[0].transpose(0,1).contiguous()
        captured=data['reference_output'].cuda()
        if not torch.equal(ref,captured):raise RuntimeError('Captured QKV cannot reproduce actual reference output exactly')
        high=oracle(q,k,v,w,scale);candidate=module.aiter_swa_prefill(q,k,v,cu,w,scale)
        # Window +/-1 controls test bounds; not candidate tuning or acceptance.
        controls={str(ww):module.aiter_swa_prefill(q,k,v,cu,ww,scale) for ww in [w-1,w+1]}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as profile:
            module.aiter_swa_prefill(q,k,v,cu,w,scale)
            torch.cuda.synchronize()
        trace=output/(path.stem+'_candidate_trace.json.gz')
        profile.export_chrome_trace(str(trace))
        kernels=sorted({event.name for event in profile.events() if str(event.device_type).endswith('CUDA')})
        row={'name':path.stem,'candidate_trace':str(trace),'kernel_names':kernels,'qkv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'reference_replay_exact':True,'reference_vs_fp32':metrics(ref,high),
            'candidate_vs_reference':metrics(candidate,ref),'candidate_vs_fp32':metrics(candidate,high),
            'reference_vs_fp32_rounded_bf16':metrics(ref,high.to(ref.dtype)),
            'candidate_vs_fp32_rounded_bf16':metrics(candidate,high.to(candidate.dtype)),
            'last_position_candidate_vs_reference':metrics(candidate[-1],ref[-1]),
            'window_controls_vs_reference':{ww:metrics(value,ref) for ww,value in controls.items()}}
        if n==w:
            # At N=W local and full causal have identical visibility.
            full=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],
                k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,scale=scale,enable_gqa=True)[0].transpose(0,1)
            row['full_sdpa_vs_reference']=metrics(full,ref)
        rows.append(row);(output/'replay.json').write_text(json.dumps(rows,indent=2)+'\n')
    if len(rows)!=2:raise RuntimeError('Expected two QKV samples')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,default=Path('/workspace/results/mellum_swa_hf_check/20261114T134033129558Z'))
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true');p.add_argument('--replay',type=Path)
    a=p.parse_args()
    if a.replay:
        os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu);replay(a.replay);return 0
    out=Path('/workspace/results/mellum_swa_layer')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    s=json.loads((a.source_run/'summary.json').read_text())
    if s['status']!='diagnostic_complete' or s['hf_vs_baseline']['generation_summary']['nonexact_positions']:
        raise ValueError('Require completed exact HF/Step0 source')
    source=Path(s['candidate_validation_run'])/'hf/reference.json';data=json.loads(source.read_text());records=[]
    for row in data['records']:
        if len(row['input_ids'][0]) not in [1024,4096]:continue
        row={**row,'generation_settings':{**row['generation_settings'],'max_new_tokens':1},'generated_ids':[row['generated_ids'][0][:1]],
             'prefill_logits':str((source.parent/row['prefill_logits']).resolve())}
        row.pop('generation_logits',None);records.append(row)
    if len(records)!=2:raise ValueError('Missing source inputs')
    (out/'reference.json').write_text(json.dumps({**data,'records':records},indent=2)+'\n')
    result={'status':'prepared','source_run':str(a.source_run),'scope':'First-layer real QKV diagnostic, not accuracy acceptance',
        'source_reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),PROJECT/'ATOM/atom/model_config/mellum_swa_capture.py',PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'ATOM/atom/model_config/mellum_attention.py']}}
    def save():(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    if a.prepare_only:print('RESULT prepared output',out);return 0
    try:
        os.environ['MELLUM_SWA_QKV_CAPTURE_DIR']=str(out/'qkv')
        try:
            run_stage('capture',['/opt/venv/bin/python',str(PROJECT/'scripts/check_atom_mellum.py'),
                '--reference',str(out/'reference.json'),'--gpu',str(a.gpu),'--qkv-backend','separate',
                '--prefill-backend','sdpa_cpu_rope_diagnostic','--reference-kv-rope','--sdpa-decode',
                '--torch-router','--hf-generation-experts'],out/'capture.log',timeout=900)
        finally:os.environ.pop('MELLUM_SWA_QKV_CAPTURE_DIR',None)
        run_stage('replay',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--gpu',str(a.gpu),'--replay',str(out)],out/'replay.log',timeout=900)
        result['status']='diagnostic_complete';save();print('RESULT diagnostic_complete output',out);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise

if __name__=='__main__':raise SystemExit(main())
