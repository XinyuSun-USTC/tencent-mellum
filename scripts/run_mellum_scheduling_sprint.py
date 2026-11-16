"""Bounded two-track profiling/basic-fastpath campaign, no HF regeneration."""
import argparse,copy,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
import run_mellum_candidate_capacity as capacity
BASE='fused_moe_native_batched_direct_kv'
TRACKS={'prefix':['sprint_graph','sprint_graph_prefix'],
        'chunk':['sprint_graph','sprint_graph_chunk2048','sprint_graph_chunk4096']}


def prepare(out,track):
    gpu=0 if track=='prefix' else 1
    template=out/'template';template.mkdir();tm=capacity.prepare(template,gpu,BASE)
    jobs={}
    for name in TRACKS[track]:
        directory=out/name;directory.mkdir();(directory/'inputs').symlink_to(template/'inputs',target_is_directory=True)
        m=copy.deepcopy(tm);m['profile']=name
        m['stages'].pop('smoke',None)
        write(directory/'manifest.json',m);jobs[name]=m
    source={**tm['source_sha256'],**tm['additional_source_sha256'],str(Path(__file__).resolve()):sha(__file__),
        str(PROJECT/'scripts/profile_mellum.py'):sha(PROJECT/'scripts/profile_mellum.py'),
        str(PROJECT/'scripts/analyze_mellum_profile.py'):sha(PROJECT/'scripts/analyze_mellum_profile.py')}
    m={'track':track,'gpu':gpu,'profiles':TRACKS[track],'source_sha256':source,
       'scope':'Optimization sprint: performance screening only; final correctness deferred',
       'trace_inputs':None,
       'input_sha256':{**tm['input_sha256'],str(template/'inputs/adapter_manifest.json'):tm['adapter_manifest_sha256']}}
    write(out/'manifest.json',m);return m,jobs


def current(plan):
    if any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Campaign source changed')
    if any(sha(p)!=h for p,h in plan.get('input_sha256',{}).items()):raise ValueError('Campaign input changed')


def verify_backend(out,profile):
    for stage in ['A_small','C_small']:
        file=out/stage/'server/fastpaths.json';a=json.loads(file.read_text())
        if profile.startswith('sprint_graph'):
            receipts=[json.loads(x) for x in (out/stage/'server/graph_replays.jsonl').read_text().splitlines()]
            if not receipts or not all(x['event']=='replayed' for x in receipts):raise ValueError('No actual decode graph replay')
        if stage=='C_small' and profile in ('sprint_graph_prefix','sprint_graph_chunk2048','sprint_graph_chunk4096'):
            events=[json.loads(x) for x in (out/stage/'server/prefix.jsonl').read_text().splitlines()]
            if not events or not any(sum(x['cached_tokens'])>0 for x in events):raise ValueError('No cached suffix processing observed')
        wanted=profile_settings(profile)
        expected_norm=wanted.get('MELLUM_NORM_BACKEND','reference');packed=wanted.get('MELLUM_QKV_BACKEND','separate')=='packed'
        if a['norm_count']!=113 or a['qkv_count']!=28 or a['norm_backend']!=expected_norm:raise ValueError('Actual norm/model backend audit differs')
        if a['norm_classes']!=({'RMSNorm':113} if expected_norm=='aiter' else {'MellumRMSNorm':113}):raise ValueError('Actual norm classes differ')
        if a['separate_qkv_count']!=(0 if packed else 28):raise ValueError('Actual packed/separate projection audit differs')
        if profile=='sprint_router':
            events=[json.loads(x) for x in (out/stage/'server/native_calls.jsonl').read_text().splitlines()]
            if not any(x.get('kind')=='moe' and x.get('routing')=='aiter_native' for x in events):raise ValueError('Native routing not observed')
        if wanted.get('MELLUM_RESIDENT_ROPE')=='1':
            rows=[json.loads(x) for x in (out/stage/'server/native_calls.jsonl').read_text().splitlines()]
            kv=[x for x in rows if x['kind']=='kv_write']
            if not kv or any(not x.get('resident_inverse') or not x.get('inverse_device','').startswith('cuda') for x in kv):raise ValueError('GPU-resident inverse frequency not observed')


def execute(out,plan,jobs,budget):
    deadline=time.monotonic()+budget;s={'status':'started','scope':plan['scope'],'jobs':{}}
    for name,m in jobs.items():
        current(plan);remaining=deadline-time.monotonic()
        if remaining<60:s['status']='budget_limited';break
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
    p=argparse.ArgumentParser();p.add_argument('--track',choices=TRACKS)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=60);a=p.parse_args()
    if not a.track:p.error('--track required')
    if not 1<=a.budget_minutes<=90:p.error('Budget must be1..90min')
    out=Path('/workspace/results/mellum_scheduling_sprint')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+a.track);out.mkdir(parents=True)
    plan,jobs=prepare(out,a.track)
    if a.prepare_only:print('RESULT sprint_prepared output',out);return 0
    s=execute(out,plan,jobs,a.budget_minutes*60);print('RESULT',s['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
