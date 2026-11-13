"""Opt-in SWA candidate: kernel diagnostics then unprofiled ABBA engine timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from mellum_runtime_config import PROJECT, apply_candidate, engine_options
from mellum_process_runner import run_stage
from profile_mellum import prepare_inputs, generate_request


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def micro(output):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(PROJECT/'ATOM'))
    # Load the lightweight file directly; avoid importing the engine in this stage.
    import importlib.util
    spec=importlib.util.spec_from_file_location('swa_math',PROJECT/'ATOM/atom/model_config/mellum_attention.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(20261113)
    rows=[]
    with torch.inference_mode():
        for n in [1023,1024,1025,4096,16384]:
            q=torch.randn(n,32,128,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(n,4,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
            def reference():
                mask=torch.ones(n,n,device='cuda',dtype=torch.bool).tril().triu(-1023)[None,None]
                return F.scaled_dot_product_attention(q.transpose(0,1)[None],
                    k.repeat_interleave(8,1).transpose(0,1)[None],v.repeat_interleave(8,1).transpose(0,1)[None],
                    attn_mask=mask,scale=128**-0.5)[0].transpose(0,1).contiguous()
            def candidate():return module.aiter_swa_prefill(q,k,v,cu,1024,128**-0.5)
            left=reference();right=candidate();torch.cuda.synchronize()
            if right.shape!=left.shape or not torch.isfinite(right).all():raise RuntimeError('Invalid candidate output')
            diff=(right.float()-left.float());timings={'reference':[],'candidate':[]}
            for fn in [reference,candidate]:
                for _ in range(3):fn()
            for repeat in range(10):
                for name,fn in ([('reference',reference),('candidate',candidate)] if repeat%2==0 else [('candidate',candidate),('reference',reference)]):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();fn();end.record();end.synchronize();timings[name].append(start.elapsed_time(end))
            row={'input_tokens':n,'max_abs':diff.abs().max().item(),
                 'relative_l2':(diff.norm()/left.float().norm().clamp_min(1e-30)).item(),
                 'timing_ms':timings,'speedup_median':statistics.median(timings['reference'])/statistics.median(timings['candidate']),
                 'scope':'Random QKV local-window diagnostic; not model acceptance'}
            rows.append(row);save(output/'micro.json',rows);print(n,row['relative_l2'],row['speedup_median'],flush=True)


def worker(output):
    plan=json.loads((output/'plan.json').read_text());settings=apply_candidate(plan['gpu'],output)
    settings.update(MELLUM_SWA_PREFILL_OPT=str(plan['candidate']),MELLUM_PROFILE_LABELS='0')
    import os
    os.environ.update(settings)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    if plan['mode']=='micro':micro(output);return
    if torch.cuda.mem_get_info(0)[0] < 80*1024**3:
        raise RuntimeError('Require 80GiB free before model loading')
    import transformers
    if transformers.__version__!='5.2.0':raise RuntimeError('Use frozen interpreter')
    sys.path.insert(0,str(PROJECT/'ATOM'))
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    options=engine_options(16640,output/'unused_traces');options['torch_profiler_dir']=None
    props=torch.cuda.get_device_properties(0)
    engine=None;result={'environment':{'torch':torch.__version__,'hip':torch.version.hip,'transformers':transformers.__version__,'gpu':props.name,'arch':getattr(props,'gcnArchName','unknown')},'settings':settings,'engine_options':options,'rows':[],'status':'started'}
    try:
        engine=EngineArgs(**options).create_engine()
        audit=json.loads((output/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:
            raise RuntimeError('Loading audit differs')
        result['loading_audit']=audit
        for case in plan['cases']:
            for budget in ([256] if case['input_tokens']==1024 else [1,64]):
                generate_request(engine,case,budget,SamplingParams)
                for repeat in range(3):
                    row=generate_request(engine,case,budget,SamplingParams)
                    row.update(name=case['name'],input_tokens=case['input_tokens'],repeat=repeat)
                    result['rows'].append(row);save(output/'worker_results.json',result)
        result['status']='timing_collected';save(output/'worker_results.json',result)
    finally:
        if engine is not None:engine.close()


def summarize(output):
    groups={}
    for stage in ['a0','b0','b1','a1']:
        data=json.loads((output/stage/'worker_results.json').read_text())
        for r in data['rows']:
            groups.setdefault((r['name'],r['output_tokens']),{}).setdefault('candidate' if stage.startswith('b') else 'reference',[]).append(r)
    rows=[]
    for (name,budget),variants in groups.items():
        left=variants['reference'];right=variants['candidate'];matches=sum(a==b for a,b in zip(left[0]['token_ids'],right[0]['token_ids']))
        item={'name':name,'output_tokens':budget,'reference_repeats_exact':all(x['token_ids']==left[0]['token_ids'] for x in left),
              'candidate_repeats_exact':all(x['token_ids']==right[0]['token_ids'] for x in right),
              'greedy_matching_positions':matches,'greedy_match_fraction':matches/budget,'timings':{}}
        for metric in ['request_ttft_seconds','request_tpot_seconds','wall_seconds']:
            if budget==1 and metric=='request_tpot_seconds':continue
            a=statistics.median(x[metric] for x in left);b=statistics.median(x[metric] for x in right)
            item['timings'][metric]={'reference_median':a,'candidate_median':b,'speedup':a/b,
                'reference_samples':[x[metric] for x in left],'candidate_samples':[x[metric] for x in right]}
        rows.append(item)
    return {'status':'candidate_measured','rows':rows,'scope':'Offline ABBA bs1, not official A/B/C or HF acceptance; review numerical and generation diagnostics before adoption'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--micro-only',action='store_true')
    a=p.parse_args()
    if a.worker:
        try:worker(a.worker);return 0
        except Exception: (a.worker/'error.txt').write_text(traceback.format_exc());raise
    out=Path('/workspace/results/mellum_swa_optimization')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    cases=prepare_inputs([1024,4096,16384])
    sources=[Path(__file__),PROJECT/'ATOM/atom/model_config/mellum_attention.py',PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'scripts/mellum_runtime_config.py',PROJECT/'scripts/profile_mellum.py']
    result={'status':'prepared','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True),'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'order':['micro'] if a.micro_only else ['micro','a0','b0','b1','a1']}
    for stage in result['order']:
        dest=out/stage;dest.mkdir();save(dest/'plan.json',{'gpu':a.gpu,'mode':'micro' if stage=='micro' else 'timing','candidate':int(stage.startswith('b')),'cases':cases})
    save(out/'summary.json',result)
    if a.prepare_only:print('RESULT prepared output',out);return 0
    try:
        for stage in result['order']:
            run_stage(stage,['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(out/stage)],out/(stage+'.log'),timeout=1800)
        if a.micro_only:
            result['status']='kernel_diagnostic_complete';save(out/'summary.json',result)
            print('RESULT kernel_diagnostic_complete output',out,flush=True);return 0
        result.update(summarize(out));save(out/'summary.json',result)
        print('RESULT candidate_measured output',out,flush=True);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save(out/'summary.json',result);raise

if __name__=='__main__':raise SystemExit(main())
