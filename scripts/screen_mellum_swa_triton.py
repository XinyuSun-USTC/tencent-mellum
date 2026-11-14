"""One bounded GPU screening of Triton vs CK vs SDPA on saved real QKV."""
import argparse,hashlib,importlib.util,json,os,statistics,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT
from mellum_process_runner import run_stage
from diagnose_mellum_swa_layer import metrics


def load_module(name,file):
    spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def boundary_probe(torch,call):
    n=1025;q=torch.zeros(n,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.zeros(n,4,128,device='cuda',dtype=torch.bfloat16);v=torch.zeros_like(k)
    values=torch.tensor([1,2,4,8],device='cuda',dtype=v.dtype)[:,None].expand(4,128)
    v[0]=values
    out=call(q,k,v,torch.tensor([0,n],device='cuda',dtype=torch.int32),1024,128**-.5)
    if not torch.isfinite(out).all():raise RuntimeError('Nonfinite probe output')
    torch.testing.assert_close(out[0],values.repeat_interleave(8,0),rtol=0,atol=0)
    # Token 0 is included at query1023 and excluded at query1024.
    torch.testing.assert_close(out[1023],values.repeat_interleave(8,0)/1024,rtol=.02,atol=1e-6)
    if not torch.equal(out[1024],torch.zeros_like(out[1024])):raise RuntimeError('Window includes an out-of-window key')
    return {'gqa_first_position_exact':True,'left_boundary_included':True,'outside_window_excluded_exact':True}


def worker(out):
    plan=json.loads((out/'plan.json').read_text());os.environ['HIP_VISIBLE_DEVICES']=str(plan['gpu']);os.environ['AITER_LOG_LEVEL']='WARNING'
    import torch
    import triton
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    props=torch.cuda.get_device_properties(0)
    if 'gfx942' not in props.gcnArchName:raise RuntimeError('This screening targets allocated gfx942 hardware')
    torch.set_grad_enabled(False)
    base=load_module('swa_reference',PROJECT/'ATOM/atom/model_config/mellum_attention.py')
    alternative=load_module('swa_triton',PROJECT/'ATOM/atom/model_config/mellum_swa_triton.py')
    report={'status':'started','environment':{'torch':torch.__version__,'triton':triton.__version__,'hip':torch.version.hip,'gpu':props.name,'arch':props.gcnArchName},'records':[]}
    def save():(out/'screening.json').write_text(json.dumps(report,indent=2)+'\n')
    save();report['boundary_probe']=boundary_probe(torch,alternative.triton_swa_prefill);save()
    for item in plan['qkv']:
        path=Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('QKV hash differs')
        data=torch.load(path,map_location='cpu',weights_only=True);q,k,v=[data[key].cuda() for key in ['q','k','v']]
        n=data['input_tokens'];window=data['window'];scale=data['scale'];cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
        if q.shape!=(n,32,128) or k.shape!=(n,4,128) or window!=1024:raise ValueError('Unexpected captured shape/window')
        def reference():
            mask=torch.ones(n,n,device=q.device,dtype=torch.bool).tril().triu(1-window)[None,None]
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],
                k.repeat_interleave(8,1).transpose(0,1)[None],v.repeat_interleave(8,1).transpose(0,1)[None],
                attn_mask=mask,scale=scale)[0].transpose(0,1).contiguous()
        calls={'sdpa':reference,'ck':lambda:base.aiter_swa_prefill(q,k,v,cu,window,scale),
               'triton':lambda:alternative.triton_swa_prefill(q,k,v,cu,window,scale)}
        outputs={name:fn() for name,fn in calls.items()}
        if not torch.equal(outputs['sdpa'],data['reference_output'].cuda()):raise RuntimeError('Reference replay is not exact')
        for name,value in outputs.items():
            if value.shape!=q.shape or not torch.isfinite(value).all():raise RuntimeError('Invalid '+name+' output')
        for fn in calls.values():
            for _ in range(3):fn()
        torch.cuda.synchronize();timings={name:[] for name in calls};names=list(calls)
        for repeat in range(10):
            order=names[repeat%3:]+names[:repeat%3]
            for name in order:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();calls[name]();end.record();end.synchronize();timings[name].append(start.elapsed_time(end))
        medians={name:statistics.median(values) for name,values in timings.items()}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            calls['triton']();torch.cuda.synchronize()
        trace=out/(f'ctx_{n}_triton_trace.json.gz');profile.export_chrome_trace(str(trace))
        row={'input_tokens':n,'qkv_sha256':item['sha256'],'reference_replay_exact':True,
             'ck_vs_sdpa':metrics(outputs['ck'],outputs['sdpa']),
             'triton_vs_sdpa':metrics(outputs['triton'],outputs['sdpa']),
             'triton_last_position_vs_sdpa':metrics(outputs['triton'][-1],outputs['sdpa'][-1]),
             'timing_ms':timings,'median_ms':medians,'triton_speedup_vs_sdpa':medians['sdpa']/medians['triton'],
             'triton_speedup_vs_ck':medians['ck']/medians['triton'],
             'trace':str(trace),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
             'kernel_names':sorted({e.name for e in profile.events() if str(e.device_type).endswith('CUDA')})}
        report['records'].append(row);save();print('SCREEN',n,'relative_l2',row['triton_vs_sdpa']['relative_l2'],'median_ms',medians,flush=True)
    report['status']='screening_complete';report['interpretation']='Local screening only; no acceptance or production backend switch';save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,default=Path('/workspace/results/mellum_swa_layer/20261114T145253662684Z'))
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path)
    a=p.parse_args()
    if a.worker:worker(a.worker);return 0
    source=json.loads((a.source_run/'summary.json').read_text())
    if source['status']!='diagnostic_complete':raise ValueError('Require completed real-QKV source')
    rows=json.loads((a.source_run/'replay.json').read_text());qkv=[]
    for row in rows:
        f=a.source_run/'qkv'/(row['name']+'.pt')
        if not row['reference_replay_exact'] or hashlib.sha256(f.read_bytes()).hexdigest()!=row['qkv_sha256']:raise ValueError('Source capture invalid')
        qkv.append({'path':str(f.resolve()),'sha256':row['qkv_sha256']})
    if sorted(row['name'] for row in rows)!=['ctx_1024','ctx_4096']:raise ValueError('Expected exactly 1K/4K captures')
    out=Path('/workspace/results/mellum_swa_triton_screen')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    files=[Path(__file__),PROJECT/'ATOM/atom/model_config/mellum_swa_triton.py',PROJECT/'ATOM/atom/model_config/mellum_attention.py',
           Path('/app/aiter-test/aiter/ops/triton/attention/mha.py'),Path('/app/aiter-test/aiter/ops/triton/_triton_kernels/attention/mha.py')]
    plan={'gpu':a.gpu,'source_run':str(a.source_run),'qkv':qkv,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result={'status':'prepared','scope':'One alternative, two saved real QKV samples; no model load or backend switch'}
    def save():(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    if a.prepare_only:print('RESULT prepared output',out);return 0
    try:
        run_stage('screen',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(out)],out/'screen.log',timeout=900)
        result['status']='screening_complete';save();print('RESULT screening_complete output',out);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise
if __name__=='__main__':raise SystemExit(main())
