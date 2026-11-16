"""Isolated reference-KV-once screen; unchanged reference math and actual cache A/B.

Default: cache semantics/timing then developer200 + long regression. Optional
paired reduced HTTP timings use output32, not the formal ABC output256 workload.
"""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace as NS
from mellum_runtime_config import PROJECT,MODEL,apply_serving
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
import run_mellum_candidate_acceptance as acceptance
import run_mellum_abc_baseline as perf
CANDIDATE='r_reference_kv_once'
PRIMARY=PROJECT/'artifacts/submission/primary_acceptance_batch32'


def settings(gpu,out,candidate=True):
    env=apply_serving(gpu,out,True,'reference',True)
    env.update(profile_settings(CANDIDATE if candidate else 'r'))
    if any(env.get(k,'0')!='0' for k in ['MELLUM_RESIDENT_ROPE','MELLUM_FUSED_ROPE','MELLUM_FUSED_QK_ROPE','MELLUM_DECODE_GRAPH','MELLUM_PREFIX_ENABLED']):
        raise ValueError('Reference-KV screen forbids unrelated fast paths')
    if env['MELLUM_MOE_BACKEND']!='torch' or env['MELLUM_ATTENTION_EXECUTION']!='reference' or env['MELLUM_QKV_BACKEND']!='separate':
        raise ValueError('Reference path must remain unchanged')
    os.environ.update(env);return env


def packed_v_layout_covered(v):
    # Singleton leading dimensions can legally be contiguous despite a packed
    # source. Multi-token cases must exercise a genuinely strided V view.
    return v.shape[0]==1 or not v.is_contiguous()


def cache_worker(out,gpu):
    settings(gpu,out,False);sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch
    from aiter.rotary_embedding import get_rope
    from atom.model_ops.attention_mha import PagedAttentionImpl
    from atom.model_config.mellum_utils import aiter_rope_parameters
    from atom.model_config.mellum_attention import install_reference_kv_rope,cpu_inverse_frequency,gather_reference_kv
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    torch.set_grad_enabled(False);torch.manual_seed(615)
    cfg=json.loads((MODEL/'config.json').read_text());qh=cfg['num_attention_heads'];kh=cfg['num_key_value_heads'];dim=cfg['head_dim'];block=16;x=8
    results=[]
    for kind,window in [('full_attention',-1),('sliding_attention',cfg['sliding_window'])]:
        params=aiter_rope_parameters(cfg['rope_parameters'][kind])
        with torch.device('cuda'):
            rotary=get_rope(dim,rotary_dim=dim,max_position=16448,base=params['rope_theta'],rope_scaling=params,dtype=torch.bfloat16)
        def create(direct):
            os.environ['MELLUM_DIRECT_KV_WRITE']='1' if direct else '0'
            with torch.device('cuda'):
                impl=PagedAttentionImpl(qh,dim,dim**-.5,kh,None,sliding_window=window,kv_cache_dtype='bf16',layer_num=1,rotary_emb=rotary)
            install_reference_kv_rope(impl);return impl
        old,new=create(False),create(True)
        for n in [1,6,32,1024]:
            blocks=(n+31)//block+1;capacity=blocks*block
            seed=torch.randn(n,(qh+2*kh)*dim,device='cuda',dtype=torch.bfloat16)
            positions=torch.arange(n,device='cuda',dtype=torch.int64)
            slots=torch.randperm(capacity,device='cuda')[:n].to(torch.int64)
            if n==6:
                positions=torch.tensor([0,15,16,1023,1024,16384],device='cuda',dtype=torch.int64)
                slots=torch.tensor([15,16,-1,0,33,47],device='cuda',dtype=torch.int64)
            def allocate():
                packed=seed.clone();q,k,v=packed.split([qh*dim,kh*dim,kh*dim],dim=-1)
                q=q.reshape(n,qh,dim);k=k.reshape(n,kh,dim);v=v.reshape(n,kh,dim)
                kc=torch.full((blocks,kh,dim//x,block,x),-7,device='cuda',dtype=torch.bfloat16)
                vc=torch.full((blocks,kh,dim,block),-9,device='cuda',dtype=torch.bfloat16)
                ctx=NS(attn_metadata=NS(has_cached=False,slot_mapping=slots),kv_cache_data={'layer_1':NS(k_cache=kc,v_cache=vc,k_scale=None,v_scale=None)},context=NS(is_prefill=True,is_dummy_run=True))
                return packed,(q,k,v,packed,positions,ctx),kc,vc
            p0,a0,k0,v0=allocate();p1,a1,k1,v1=allocate()
            y0=old.rope_cache(*a0);y1=new.rope_cache(*a1);torch.cuda.synchronize()
            keys,values=gather_reference_kv(k1,v1,torch.arange(blocks,device='cuda'),0,capacity,new._cache_format)
            valid=slots>=0;written=slots[valid];untouched=torch.ones(capacity,device='cuda',dtype=torch.bool);untouched[written]=False
            checks={'query_exact':torch.equal(y0[0],y1[0]),'returned_key_exact':torch.equal(y0[1],y1[1]),
                'full_k_cache_exact':torch.equal(k0,k1),'full_v_cache_exact':torch.equal(v0,v1),
                'layout_exact':old._cache_format==new._cache_format,
                'gathered_keys_exact':torch.equal(keys[written],y1[1][valid]),
                'gathered_values_exact':torch.equal(values[written],a1[2][valid]),
                'untouched_slots':bool((keys[untouched]==-7).all() and (values[untouched]==-9).all()),
                'candidate_input_unchanged':torch.equal(p1,seed),'packed_v_layout_covered':packed_v_layout_covered(a1[2])}
            # Identical input reset is included in BOTH timings to avoid repeated
            # in-place rotations in the reference kernel changing input values.
            def measure(impl,packed,args):
                for _ in range(3):packed.copy_(seed);impl.rope_cache(*args)
                torch.cuda.synchronize();samples=[]
                for _ in range(3):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):packed.copy_(seed);impl.rope_cache(*args)
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
                return sorted(samples)[1]
            old_ms=measure(old,p0,a0);new_ms=measure(new,p1,a1)
            row={'kind':kind,'tokens':n,'window':window,'cache_layout':new._cache_format,'v_is_contiguous':a1[2].is_contiguous(),'checks':checks,'old_ms':old_ms,'new_ms':new_ms,'local_speedup':old_ms/new_ms}
            results.append(row);write(out/'results.json',{'status':'running','cases':results});print('CACHE',kind,n,'checks',all(checks.values()),'ratio',old_ms/new_ms,flush=True)
            if not all(checks.values()):raise ValueError('Reference cache A/B mismatch; no model test launched')
    write(out/'results.json',{'status':'passed','cases':results,'scope':'Actual old/new RoPE-cache and full physical-cache comparison; timings include identical input reset; not model speed'})


def generation_worker(out):
    acceptance.worker(out,suites=['tokens','long'],batch_size=32)


def perf_worker(out):
    # Reduced timing changes only output budget; the database inspector checks32.
    original_command,original_inspect=perf.command,perf.inspect_result
    def command(*args,**kwargs):
        cmd=original_command(*args,**kwargs)
        for flag in ['--max-tokens','--min-tokens']:cmd[cmd.index(flag)+1]='32'
        return cmd
    def inspect(*args,**kwargs):return original_inspect(*args,**{**kwargs,'budget':32})
    perf.command=command;perf.inspect_result=inspect
    perf.stage_worker(out)
    result=json.loads((out/'results.json').read_text())
    result.update(scope='Reduced A32/B4 HTTP timing with32 output tokens; not formal ABC',output_budget=32)
    write(out/'results.json',result)


def prepare(out,gpu,with_perf):
    candidate=out/'candidate_screen';candidate.mkdir();p=acceptance.prepare(candidate,CANDIDATE,gpu)
    p['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    old_summary=json.loads((PRIMARY/'summary.json').read_text())
    original=Path(json.loads((PROJECT/'artifacts/submission/index.json').read_text())['sources']['primary_acceptance_batch32'])
    baselines={}
    for suite in ['tokens','long']:
        name=suite+'_generation.json';f=PRIMARY/name
        if not f.exists():f=original/name
        if sha(f)!=old_summary['artifact_sha256'][name]:raise ValueError('Primary comparison output checksum differs')
        baselines[suite]=str(f);p['external_sha256'][str(f)]=sha(f)
    write(candidate/'manifest.json',p)
    plan={'gpu':gpu,'profile':CANDIDATE,'candidate_plan':str(candidate/'manifest.json'),'candidate_plan_sha256':sha(candidate/'manifest.json'),
          'source_sha256':p['source_sha256'],'primary_generations':baselines,'with_perf':with_perf,'scope':'One variable: reference KV write once. Full developer200/long; no HE rerun or formal ABC at this screening stage'}
    if with_perf:
        template=out/'perf_template';template.mkdir();m=perf.prepare(template,gpu,False);m['stages'].pop('smoke')
        stages=m['stages'];stages['B'][0]['number']=4;stages['A'][0]['number']=32;stages['A'][0]['expected']=stages['A'][0]['expected'][:32]
        # No C screen here: cached KV reuse is disabled and a full C run is
        # reserved for a promoted combination. A covers mixed prefill/decode.
        stages.pop('C');write(template/'manifest.json',m);plan['perf_template']=m
    write(out/'manifest.json',plan);return plan


def current(out,plan):
    if sha(plan['candidate_plan'])!=plan['candidate_plan_sha256']:raise ValueError('Candidate plan changed')
    acceptance.current(json.loads(Path(plan['candidate_plan']).read_text()))
    if 'perf_template' in plan:perf.current_matches(plan['perf_template'],out/'perf_template')


def evaluate_candidate(out,plan):
    directory=out/'candidate_screen';p=json.loads(Path(plan['candidate_plan']).read_text());result={}
    for suite in ['tokens','long']:
        result[suite]=acceptance.token_comparison(json.loads(Path(p['references'][suite]).read_text()),json.loads((directory/(suite+'_generation.json')).read_text()))
        baseline=json.loads(Path(plan['primary_generations'][suite]).read_text())
        result[suite+'_vs_primary']=acceptance.token_comparison(baseline,json.loads((directory/(suite+'_generation.json')).read_text()))['all']
    target=directory/'candidate';load=json.loads((target/'loading_audit.json').read_text())
    if load!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise ValueError('Loading incomplete')
    actual=json.loads((target/'fastpaths.json').read_text())
    if (actual['norm_backend']!='reference' or actual['separate_qkv_count']!=28
            or actual['moe_classes']!={'MellumSparseMoeBlock':28}
            or actual['prefill_functions']!={'ragged_prefill':28}
            or actual['resident_inverse_frequency']):raise ValueError('Unrelated path changed')
    events=[json.loads(l) for l in (target/'native_calls.jsonl').read_text().splitlines()]
    writes=[r for r in events if r.get('kind')=='kv_write']
    if {r['phase'] for r in writes}!={'prefill','decode'} or any(r['backend']!='aiter_reference_kv_once' or r.get('rope_backend')!='torch' or r.get('resident_inverse') for r in writes):
        raise ValueError('Actual reference-only direct-KV phase evidence missing')
    result.update(loading_audit=load,actual_paths=actual,kv_write_events=writes,
        developer200_gate=result['tokens']['all']['at_least_90_percent'],
        note='Long and16k separately reported; no new internal bitwise gate. HumanEval and formal performance remain pending.')
    return result


def execute(out,plan,budget,executor=run_stage):
    deadline=time.monotonic()+budget;summary={'status':'started','scope':plan['scope']};current(out,plan)
    def stage(name,args,log,cap):
        current(out,plan);remaining=deadline-time.monotonic()
        if remaining<10:raise TimeoutError('Screen budget exhausted')
        executor(name,[sys.executable,str(Path(__file__).resolve()),'--gpu',str(plan['gpu']),*args],log,timeout=min(cap,remaining));current(out,plan)
    try:
        cache=out/'cache';cache.mkdir();stage('cache_ab',['--worker','cache','--output',str(cache)],cache/'worker.log',600)
        summary['cache']=json.loads((cache/'results.json').read_text())
        if summary['cache']['status']!='passed':raise ValueError('Cache A/B failed')
        directory=out/'candidate_screen';stage('generation',['--worker','generation','--output',str(directory)],directory/'worker.log',1800)
        summary['generation']=evaluate_candidate(out,plan)
        write(out/'summary.json',summary)
        if not summary['generation']['developer200_gate']:
            summary['status']='developer200_below_90';return summary
        if plan['with_perf']:
            summary['paired_perf']={}
            for name,profile in [('R1','r'),('K1',CANDIDATE),('K2',CANDIDATE),('R2','r')]:
                directory=out/'paired_perf'/name;directory.mkdir(parents=True)
                job={'stage':'reduced_reference_kv','gpu':plan['gpu'],'profile':profile,
                    'cases':sum(plan['perf_template']['stages'].values(),[]),'timeout':min(900,max(1,deadline-time.monotonic())),
                    'server_port':8651+plan['gpu']*100,'engine_port':11226+plan['gpu']*100}
                write(directory/'plan.json',job)
                stage(name,['--worker','perf','--output',str(directory)],directory/'worker.log',900)
                value=json.loads((directory/'results.json').read_text())
                if value['status']!='complete' or any(not v['metrics']['complete_without_failures'] for v in value['cases'].values()):raise ValueError('Reduced HTTP workload incomplete')
                summary['paired_perf'][name]=value;write(out/'summary.json',summary)
        summary['status']='screen_passed_full_acceptance_pending'
    except Exception:summary.update(status='failed',traceback=traceback.format_exc())
    finally:
        current(out,plan);summary['source_unchanged']=True
        summary['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
        write(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--with-perf',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=40)
    p.add_argument('--worker',choices=['cache','generation','perf']);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.worker:
        a.output.mkdir(parents=True,exist_ok=True)
        if a.worker=='cache':cache_worker(a.output,a.gpu)
        elif a.worker=='generation':generation_worker(a.output)
        else:perf_worker(a.output)
        return
    if not 1<=a.budget_minutes<=90:p.error('Budget1..90min')
    out=Path('/workspace/results/mellum_reference_kv_screen')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan=prepare(out,a.gpu,a.with_perf)
    if a.prepare_only:print('RESULT reference_kv_screen_prepared output',out);return
    r=execute(out,plan,a.budget_minutes*60);print('RESULT',r['status'],'output',out)
    if r['status'] not in ('screen_passed_full_acceptance_pending',):raise SystemExit(1)
if __name__=='__main__':main()
