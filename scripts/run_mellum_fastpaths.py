"""Bounded two-track profiling/basic-fastpath campaign, no HF regeneration."""
import argparse,copy,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT,apply_serving,engine_options
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
from profile_mellum import prepare_inputs,wait_receipt
from analyze_mellum_profile import analyze_file,render_markdown
import run_mellum_candidate_capacity as capacity
BASE='fused_moe_native_batched_direct_kv'
TRACKS={'resident':[BASE,'fast_resident'],'kernels':[BASE,'fast_norm','fast_packed','fast_combined']}


def trace_worker(out):
    settings=apply_serving(0,out,True,'reference',True);settings.update(profile_settings(BASE))
    settings.update(MELLUM_PROFILE_LABELS='1',MELLUM_PROFILE_CONTROL_DIR=str(out/'control'),
        MELLUM_FASTPATH_AUDIT_PATH=str(out/'fastpaths.json'),MELLUM_NATIVE_AUDIT_PATH=str(out/'native_calls.jsonl'))
    os.environ.update(settings);sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    if not torch.cuda.is_available():raise RuntimeError('GPU0 unavailable')
    if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require80GiB free')
    plan=json.loads((out.parent/'manifest.json').read_text())
    case=plan['trace_inputs'][0]
    if sha(PROJECT/case['source'])!=case['source_sha256']:raise ValueError('Trace input source changed')
    opts=engine_options(1056,out/'traces');opts.update(max_num_seqs=32,max_num_batched_tokens=16384,port=9226)
    result={'status':'started','cases':[],'scope':'Short diagnostic traces; profile time is not throughput'};engine=None
    try:
        engine=EngineArgs(**opts).create_engine()
        def generate(count,tokens):
            start=time.monotonic();rows=engine.generate([case['input_ids'] for _ in range(count)],
                [SamplingParams(temperature=0,max_tokens=tokens,ignore_eos=True) for _ in range(count)])
            if len(rows)!=count or any(len(r['token_ids'])!=tokens for r in rows):raise ValueError('Incomplete trace generation')
            return time.monotonic()-start
        for sequence,count in enumerate([1,32],1):
            generate(count,8);unprofiled=generate(count,16)
            before=set((out/'traces').rglob('*.pt.trace.json.gz'));active=False
            try:
                engine.start_profile();active=True;wait_receipt(out/'control','started',sequence,60)
                profiled=generate(count,8)
            finally:
                if active:engine.stop_profile()
            wait_receipt(out/'control','stopped',sequence,300)
            traces=set((out/'traces').rglob('*.pt.trace.json.gz'))-before
            if len(traces)!=1:raise ValueError('Expected one rank0 trace')
            trace=next(iter(traces));analysis=analyze_file(trace)
            if not analysis['gpu_kernel_count'] or not analysis['module_annotation_count']:raise ValueError('Trace lacks actual GPU/module evidence')
            write(out/f'bs{count}_analysis.json',analysis);(out/f'bs{count}_analysis.md').write_text(render_markdown(analysis))
            result['cases'].append({'batch':count,'context':1024,'trace_decode_tokens':8,'unprofiled_16_token_seconds':unprofiled,
                'profiled_8_token_seconds':profiled,'trace':str(trace),'trace_sha256':sha(trace)})
            write(out/'results.json',result)
        result['status']='complete';write(out/'results.json',result)
    except Exception:result.update(status='failed',traceback=traceback.format_exc());write(out/'results.json',result);raise
    finally:
        if engine is not None:engine.close()


def prepare(out,track):
    gpu=0 if track=='resident' else 1
    template=out/'template';template.mkdir();tm=capacity.prepare(template,gpu,BASE)
    jobs={}
    for name in TRACKS[track]:
        directory=out/name;directory.mkdir();(directory/'inputs').symlink_to(template/'inputs',target_is_directory=True)
        m=copy.deepcopy(tm);m['profile']=name;write(directory/'manifest.json',m);jobs[name]=m
    source={**tm['source_sha256'],**tm['additional_source_sha256'],str(Path(__file__).resolve()):sha(__file__),
        str(PROJECT/'scripts/profile_mellum.py'):sha(PROJECT/'scripts/profile_mellum.py'),
        str(PROJECT/'scripts/analyze_mellum_profile.py'):sha(PROJECT/'scripts/analyze_mellum_profile.py')}
    m={'track':track,'gpu':gpu,'profiles':TRACKS[track],'source_sha256':source,
       'scope':'Basic fastpath small-load study, fixed inference environment, no HF/public200/full ABC',
       'trace_inputs':prepare_inputs([1024]) if track=='resident' else None,
       'input_sha256':{**tm['input_sha256'],str(template/'inputs/adapter_manifest.json'):tm['adapter_manifest_sha256']}}
    write(out/'manifest.json',m);return m,jobs


def current(plan):
    if any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Campaign source changed')
    if any(sha(p)!=h for p,h in plan.get('input_sha256',{}).items()):raise ValueError('Campaign input changed')


def verify_backend(out,profile):
    for stage in ['smoke','A_small','C_small']:
        file=out/stage/'server/fastpaths.json';a=json.loads(file.read_text())
        wanted=profile_settings(profile)
        expected_norm=wanted.get('MELLUM_NORM_BACKEND','reference');packed=wanted.get('MELLUM_QKV_BACKEND','separate')=='packed'
        if a['norm_count']!=113 or a['qkv_count']!=28 or a['norm_backend']!=expected_norm:raise ValueError('Actual norm/model backend audit differs')
        if a['norm_classes']!=({'RMSNorm':113} if expected_norm=='aiter' else {'MellumRMSNorm':113}):raise ValueError('Actual norm classes differ')
        if a['separate_qkv_count']!=(0 if packed else 28):raise ValueError('Actual packed/separate projection audit differs')
        if wanted.get('MELLUM_RESIDENT_ROPE')=='1':
            rows=[json.loads(x) for x in (out/stage/'server/native_calls.jsonl').read_text().splitlines()]
            kv=[x for x in rows if x['kind']=='kv_write']
            if not kv or any(not x.get('resident_inverse') or not x.get('inverse_device','').startswith('cuda') for x in kv):raise ValueError('GPU-resident inverse frequency not observed')


def execute(out,plan,jobs,budget):
    deadline=time.monotonic()+budget;s={'status':'started','scope':plan['scope'],'jobs':{}}
    if plan['track']=='resident':
        directory=out/'trace';directory.mkdir()
        try:
            current(plan);remaining=deadline-time.monotonic()
            if remaining<60:raise TimeoutError('Insufficient trace budget')
            run_stage('short_trace',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--trace-worker',str(directory)],directory/'worker.log',timeout=min(900,remaining))
            s['trace_status']=json.loads((directory/'results.json').read_text())['status']
        except Exception:s.update(trace_status='failed',trace_error=traceback.format_exc())
    for name,m in jobs.items():
        current(plan);remaining=deadline-time.monotonic()
        if remaining<60:s['status']='budget_limited';break
        # Independent failures remain visible; skip combinations containing a crashed component.
        if name=='fast_combined' and any(s['jobs'].get(x,{}).get('status')!='diagnostic_complete' for x in ['fast_norm','fast_packed']):
            s['jobs'][name]={'status':'skipped_component_failure'};continue
        try:
            measured=capacity.execute(out/name,m,remaining)
            if measured['status']=='diagnostic_complete':verify_backend(out/name,name)
            s['jobs'][name]={'status':measured['status'],'summary':str(out/name/'summary.json')}
        except Exception:s['jobs'][name]={'status':'failed','traceback':traceback.format_exc()}
        write(out/'summary.json',s)
    if s['status']=='started':s['status']='complete' if all(x['status']=='diagnostic_complete' for x in s['jobs'].values()) and len(s['jobs'])==len(jobs) and s.get('trace_status','complete')=='complete' else 'complete_with_failures'
    current(plan);s['source_unchanged']=True
    s['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'summary.json'}
    write(out/'summary.json',s);return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=TRACKS);p.add_argument('--trace-worker',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=60);a=p.parse_args()
    if a.trace_worker:trace_worker(a.trace_worker);return 0
    if not a.track:p.error('--track required')
    if not 1<=a.budget_minutes<=90:p.error('Budget must be1..90min')
    out=Path('/workspace/results/mellum_fastpaths')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+a.track);out.mkdir(parents=True)
    plan,jobs=prepare(out,a.track)
    if a.prepare_only:print('RESULT fastpaths_prepared output',out);return 0
    s=execute(out,plan,jobs,a.budget_minutes*60);print('RESULT',s['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
