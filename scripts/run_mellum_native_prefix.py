"""GPU0 native-prefix campaign: rectangular mask probes, model on/off, HTTP."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,apply_serving
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
import run_mellum_prefix_probe as probe
import run_mellum_candidate_capacity as capacity

PROFILE='fused_moe_native_batched'


def kernel_worker(out):
    apply_serving(0,out);sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch,aiter
    from atom.model_config.mellum_prefix import suffix_mask
    if not torch.cuda.is_available():raise RuntimeError('GPU0 unavailable')
    results=[]
    # Uniform attention gives an analytic mask/GQA oracle. Values differ by
    # request and head, making wrong offsets or cross-request mixing visible.
    for window in [-1,4,1024]:
        qlens=[1,3];klens=[1039,1041] if window==1024 else [19,25]
        q=torch.zeros(sum(qlens),32,128,device='cuda',dtype=torch.bfloat16)
        k=torch.zeros(sum(klens),4,128,device='cuda',dtype=torch.bfloat16)
        parts=[]
        for i,n in enumerate(klens):
            if window==1024:
                # Spike exactly at the first query's left boundary: keeping
                # or dropping one extra key changes the mean by 0.25.
                v=torch.zeros(n,4,128,device='cuda',dtype=torch.float32)
                left=n-qlens[i]-window+1
                v[:left]=128;v[left]=256
            else:
                v=(torch.arange(n,device='cuda',dtype=torch.float32)[:,None,None]*2
                    +torch.arange(4,device='cuda')[None,:,None]*16+i*128).expand(n,4,128)
            parts.append(v.to(torch.bfloat16).contiguous())
        v=torch.cat(parts);qb=torch.tensor([0,1,4],device='cuda',dtype=torch.int32);kb=torch.tensor([0,klens[0],sum(klens)],device='cuda',dtype=torch.int32)
        output=aiter.flash_attn_varlen_func(q=q,k=k,v=v,cu_seqlens_q=qb,cu_seqlens_k=kb,
            max_seqlen_q=3,max_seqlen_k=max(klens),dropout_p=0.,softmax_scale=128**-.5,causal=True,
            window_size=(window-1,0,0) if window>0 else (-1,-1,0),return_lse=False,return_attn_probs=False)
        expected=[]
        for nq,nk,values in zip(qlens,klens,parts):
            mask=suffix_mask(nq,nk,window,'cuda')[0,0]
            for row in mask:
                expected.append(values[row].float().mean(0).repeat_interleave(8,dim=0))
        expected=torch.stack(expected).to(torch.bfloat16)
        finite=bool(torch.isfinite(output).all());err=float((output.float()-expected.float()).abs().max())
        # This is a structural kernel probe, not an assessment accuracy threshold.
        # Half-unit error is enough for BF16 rounding at these values, while
        # a one-key window shift changes their mean by at least one unit.
        tolerance=.01 if window==1024 else .51
        results.append({'window':window,'finite':finite,'max_abs':err,'structural_check':finite and err<=tolerance})
    write(out/'results.json',{'status':'passed' if all(r['structural_check'] for r in results) else 'failed',
        'cases':results,'scope':'Synthetic bottom-right/GQA/window structural check, not formal accuracy acceptance'})
    if not all(r['structural_check'] for r in results):raise ValueError('Rectangular native kernel structural check failed')


def prepare(out):
    p=out/'probe';p.mkdir();pm=probe.prepare(p,PROFILE)
    c=out/'capacity';c.mkdir();cm=capacity.prepare(c,0,PROFILE,True)
    sources={**pm['source_sha256'],**cm['source_sha256'],**cm['additional_source_sha256'],str(Path(__file__).resolve()):sha(__file__)}
    plan={'gpu':0,'profile':PROFILE,'source_sha256':sources,'probe_plan_sha256':sha(p/'plan.json'),
          'capacity_manifest_sha256':sha(c/'manifest.json'),'stages':['kernel','off','on','HTTP_smoke_A64_C20'],
          'scope':'Native prefix integration diagnostic; no HF regeneration or token-equality gate.'}
    write(out/'manifest.json',plan);return plan,pm,cm


def current(plan):
    if any(sha(p)!=v for p,v in plan['source_sha256'].items()):raise ValueError('Campaign source changed')


def execute(out,plan,pm,cm,budget):
    deadline=time.monotonic()+budget;result={'status':'started','scope':plan['scope']}
    def remaining(cap):
        current(plan);n=deadline-time.monotonic()
        if n<60:raise TimeoutError('Native prefix campaign budget exhausted')
        return min(cap,n)
    try:
        kernel=out/'kernel';kernel.mkdir()
        run_stage('kernel',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--kernel-worker',str(kernel)],kernel/'worker.log',timeout=remaining(300))
        result['kernel']=json.loads((kernel/'results.json').read_text())
        if result['kernel']['status']!='passed':raise ValueError('Structural kernel gate failed')
        for name,enabled in [('off',False),('on',True)]:
            directory=out/'probe'/name;directory.mkdir();cmd=['/opt/venv/bin/python',str(PROJECT/'scripts/run_mellum_prefix_probe.py'),'--worker',str(directory)]
            if enabled:cmd+=['--prefix']
            run_stage(name,cmd,directory/'worker.log',timeout=remaining(600))
        off=json.loads((out/'probe/off/results.json').read_text());on=json.loads((out/'probe/on/results.json').read_text())
        if off['status']!='complete' or on['status']!='complete' or on['observed_cached_tokens']<=0:raise ValueError('No real prefix reuse or incomplete probe')
        native=[json.loads(x) for x in (out/'probe/on/native_calls.jsonl').read_text().splitlines()]
        if not any(r['kind']=='attention' and r['backend']=='aiter_varlen' and r['phase']=='prefill' for r in native):raise ValueError('Native prefill audit missing')
        events=on['prefix_events']
        if not events or any(r.get('backend')!='aiter_varlen_cached' for r in events):raise ValueError('Cached prefill silently used reference path')
        pairs=[(l,r) for a,b in zip(off['groups'],on['groups']) for l,r in zip(a['records'],b['records'])]
        result['probe']={'cached_tokens':on['observed_cached_tokens'],'exact_sequences':sum(l['generated_ids']==r['generated_ids'] for l,r in pairs),
            'sequences':len(pairs),'note':'Free-generation equality is reported, not a gate for numerically different native paths.'}
        write(out/'summary.json',result)
        measured=capacity.execute(out/'capacity',cm,remaining(2700))
        result['capacity_status']=measured['status'];result['status']='complete' if measured['status']=='diagnostic_complete' else 'capacity_review_required'
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    result['source_unchanged']=all(sha(p)==v for p,v in plan['source_sha256'].items())
    if not result['source_unchanged']:result['status']='source_changed'
    result['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'summary.json'}
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--kernel-worker',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--budget-minutes',type=float,default=45);a=p.parse_args()
    if a.kernel_worker:kernel_worker(a.kernel_worker);return 0
    if not 1<=a.budget_minutes<=60:p.error('Budget must be1..60 minutes')
    out=Path('/workspace/results/mellum_native_prefix')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan,pm,cm=prepare(out)
    if a.prepare_only:print('RESULT native_prefix_prepared output',out);return 0
    s=execute(out,plan,pm,cm,a.budget_minutes*60);print('RESULT',s['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
