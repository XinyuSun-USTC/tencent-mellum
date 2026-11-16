"""GPU1 direct reference KV write: tiny layout probe then HTTP capacity."""
import argparse,json,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,apply_serving
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
import run_mellum_candidate_capacity as capacity
PROFILE='fused_moe_native_batched_direct_kv'


def kernel_worker(out):
    apply_serving(1,out);sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch
    from types import SimpleNamespace as NS
    from atom.model_config.mellum_direct_kv import install_direct_kv_rope
    from atom.model_config.mellum_attention import reference_position_rope,gather_reference_kv
    if not torch.cuda.is_available():raise RuntimeError('GPU1 unavailable')
    torch.manual_seed(74);results=[]
    for window in [-1,1024]:
        n=4;h=4;d=128;bs=16;x=8
        kc=torch.full((3,h,d//x,bs,x),-7,device='cuda',dtype=torch.bfloat16)
        vc=torch.full((3,h,d,bs),-9,device='cuda',dtype=torch.bfloat16)
        q=torch.randn(n,32,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(n,h,d,device='cuda',dtype=torch.bfloat16)
        packed=torch.randn(n,h*d*3,device='cuda',dtype=torch.bfloat16);v=packed[:,h*d:2*h*d].reshape(n,h,d)
        slots=torch.tensor([15,16,-1,33],device='cuda',dtype=torch.int64);positions=torch.tensor([15,16,17,33],device='cuda')
        inv=1/(500000**(torch.arange(0,d,2,device='cpu').float()/d))
        ctx=NS(attn_metadata=NS(has_cached=False,slot_mapping=slots),kv_cache_data={'layer_1':NS(k_cache=kc,v_cache=vc,k_scale=None,v_scale=None)},context=NS(is_prefill=True,is_dummy_run=True))
        impl=NS(layer_num=1,use_flash_layout=False,kv_cache_dtype='bf16',q_norm=None,k_norm=None,sliding_window=window,head_dim=d,num_heads=32,num_kv_heads=h)
        install_direct_kv_rope(impl,inv,1.)
        actual=impl.rope_cache(q,k,v,None,positions,ctx);expected=reference_position_rope(q,k,positions,inv,1.)
        keys,values=gather_reference_kv(kc,vc,torch.tensor([0,1,2],device='cuda'),0,48,impl._cache_format)
        valid=slots>=0;where=slots[valid].long();untouched=torch.ones(48,device='cuda',dtype=torch.bool);untouched[where]=False
        checks={'query_exact':torch.equal(actual[0],expected[0]),'keys_exact':torch.equal(keys[where],expected[1][valid]),
                'values_exact':torch.equal(values[where],v[valid]),'untouched':bool((keys[untouched]==-7).all() and (values[untouched]==-9).all()),
                'strided_v_tested':not v.is_contiguous()}
        results.append({'window':window,'layout':impl._cache_format,'checks':checks})
    good=all(all(r['checks'].values()) for r in results)
    write(out/'results.json',{'status':'passed' if good else 'failed','cases':results,'scope':'Cache copy/layout semantics only, no free-generation equality requirement'})
    if not good:raise ValueError('Direct KV kernel/layout check failed')


def prepare(out):
    c=out/'capacity';c.mkdir();m=capacity.prepare(c,1,PROFILE)
    sources={**m['source_sha256'],**m['additional_source_sha256'],str(Path(__file__).resolve()):sha(__file__)}
    plan={'gpu':1,'profile':PROFILE,'source_sha256':sources,'scope':'Direct reference KV write diagnostic; no HF regeneration or full ABC'}
    write(out/'manifest.json',plan);return plan,m


def execute(out,plan,m,budget):
    deadline=time.monotonic()+budget;s={'status':'started','scope':plan['scope']}
    def remaining(cap):
        if any(sha(f)!=h for f,h in plan['source_sha256'].items()):raise ValueError('Direct KV source changed')
        n=deadline-time.monotonic()
        if n<60:raise TimeoutError('Direct KV campaign budget exhausted')
        return min(cap,n)
    try:
        k=out/'kernel';k.mkdir();run_stage('kv_kernel',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--kernel-worker',str(k)],k/'worker.log',timeout=remaining(300))
        s['kernel']=json.loads((k/'results.json').read_text())
        if s['kernel']['status']!='passed':raise ValueError('Direct KV probe failed')
        c=capacity.execute(out/'capacity',m,remaining(2100));s['capacity_status']=c['status']
        s['status']='complete' if c['status']=='diagnostic_complete' else 'capacity_review_required'
        if c['status']=='diagnostic_complete':
            for stage in ['A_small','C_small']:
                rows=[json.loads(x) for x in (out/'capacity'/stage/'server/native_calls.jsonl').read_text().splitlines()]
                writes=[x for x in rows if x['kind']=='kv_write']
                if {x['phase'] for x in writes}!={'prefill','decode'} or any(x['backend']!='aiter_reference_kv_once' for x in writes):raise ValueError('Actual direct KV phase audit missing')
    except Exception:s.update(status='failed',traceback=traceback.format_exc())
    s['source_unchanged']=all(sha(f)==h for f,h in plan['source_sha256'].items())
    if not s['source_unchanged']:s['status']='source_changed'
    s['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'summary.json'}
    write(out/'summary.json',s);return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--kernel-worker',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=40);a=p.parse_args()
    if a.kernel_worker:kernel_worker(a.kernel_worker);return 0
    if not 1<=a.budget_minutes<=60:p.error('Budget must be1..60 minutes')
    out=Path('/workspace/results/mellum_direct_kv')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan,m=prepare(out)
    if a.prepare_only:print('RESULT direct_kv_prepared output',out);return 0
    s=execute(out,plan,m,a.budget_minutes*60);print('RESULT',s['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
