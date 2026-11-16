"""Assessment-scale ABC for explicit profiles; independent failure/resume, no HF."""
import argparse,copy,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_abc_baseline as base
from mellum_execution_profiles import PROFILES,profile_settings
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write


def prepare(out,gpu,profiles):
    template=out/'template';template.mkdir();m=base.prepare(template,gpu,False)
    m['stages'].pop('smoke')
    plan={'gpu':gpu,'profiles':profiles,'template':m,'jobs':{},'source':str(Path(__file__).resolve()),
        'source_sha256':sha(__file__),'scope':'Full local A512/B50/C200; not judge hidden-set median',
        'measurement_note':'Record whether other GPU was active; final idle-machine repetitions are separate'}
    for profile in profiles:
        directory=out/profile;directory.mkdir();(directory/'inputs').symlink_to(template/'inputs',target_is_directory=True)
    write(out/'manifest.json',plan);return plan


def current(out,plan):
    if sha(plan['source'])!=plan['source_sha256']:raise ValueError('Campaign source changed')
    base.current_matches(plan['template'],out/'template')


def audit_backend(directory,profile,stage):
    server=directory/'server';audit=json.loads((server/'loading_audit.json').read_text())
    if audit['source_tensors']!=5631 or audit['successful_parts']!=5631 or audit['missing'] or audit['unexpected']:
        raise ValueError('Checkpoint loading incomplete')
    wanted=profile_settings(profile)
    if wanted.get('MELLUM_DECODE_GRAPH')=='1':
        events=[json.loads(x) for x in (server/'graph_replays.jsonl').read_text().splitlines()]
        if not events:raise ValueError('No graph replay')
    if wanted.get('MELLUM_NATIVE_ROUTING')=='1' or wanted.get('MELLUM_FUSED_QK_ROPE')=='1':
        events=[json.loads(x) for x in (server/'native_calls.jsonl').read_text().splitlines()]
        if wanted.get('MELLUM_NATIVE_ROUTING')=='1' and not any(x.get('routing')=='aiter_native' for x in events):raise ValueError('No native routing')
        if wanted.get('MELLUM_FUSED_QK_ROPE')=='1' and not any(x.get('rope_backend')=='triton_qk_norm_rope' for x in events):raise ValueError('No QK fusion')


def execute(out,plan,budget,executor=run_stage):
    deadline=time.monotonic()+budget
    for profile in plan['profiles']:
        for stage,cases in plan['template']['stages'].items():
            current(out,plan);key=profile+'/'+stage;old=plan['jobs'].get(key,{})
            directory=out/profile/stage
            if old.get('status')=='complete':
                if any(sha(directory/f)!=h for f,h in old['artifacts'].items()):raise ValueError('Completed job artifact changed')
                continue
            remaining=deadline-time.monotonic()
            if remaining<60:return finish(out,plan,'budget_limited')
            directory.mkdir(exist_ok=True)
            if (directory/'plan.json').exists():
                n=1
                while (directory/f'attempt_{n}').exists():n+=1
                archive=directory/f'attempt_{n}';archive.mkdir()
                for f in list(directory.iterdir()):
                    if f!=archive and not f.name.startswith('attempt_'):f.rename(archive/f.name)
            job={'stage':stage,'gpu':plan['gpu'],'profile':profile,'cases':cases,
                 'timeout':min(base.LIMITS[stage],remaining),'server_port':8451+plan['gpu']*100,
                 'engine_port':9426+plan['gpu']*100}
            write(directory/'plan.json',job);plan['jobs'][key]={'status':'running'};write(out/'manifest.json',plan)
            try:
                executor(key,['/opt/venv/bin/python',str(base.PROJECT/'scripts/run_mellum_abc_baseline.py'),
                    '--stage',str(directory)],directory/'stage.log',timeout=job['timeout'])
                current(out,plan);value=json.loads((directory/'results.json').read_text())
                if value['status']!='complete' or not value['cases'] or any(not c['metrics']['complete_without_failures'] for c in value['cases'].values()):
                    raise ValueError('Incomplete requests or stage failed; retain metrics')
                audit_backend(directory,profile,stage)
                plan['jobs'][key]={'status':'complete','artifacts':{str(f.relative_to(directory)):sha(f)
                    for f in directory.rglob('*') if f.is_file() and not any(s.startswith('attempt_') for s in f.relative_to(directory).parts)}}
            except Exception:plan['jobs'][key]={'status':'failed','traceback':traceback.format_exc()}
            write(out/'manifest.json',plan)
    return finish(out,plan,'complete' if all(v['status']=='complete' for v in plan['jobs'].values()) else 'complete_with_failures')


def finish(out,plan,status):
    current(out,plan)
    rows={};metrics={}
    for key,job in plan['jobs'].items():
        f=out/key/'results.json'
        rows[key]={'status':job['status'],'results':json.loads(f.read_text()) if f.exists() else None}
    official=json.loads(Path('/data/datasets/baseline.json').read_text());comparisons={}
    for key,row in rows.items():
        profile,stage=key.split('/')
        for case in (row['results'] or {}).get('cases',{}).values():
            m=case.get('metrics',{})
            metrics.setdefault(profile,{})[stage]={k:m.get(k) for k in ['requests','successes','failures','output_tokens_per_s_gpu','ttft_p50_s','ttft_p99_s','tpot_p50_s','tpot_p99_s','complete_without_failures']}
            if row['status']!='complete' or not m.get('complete_without_failures'):continue
            if stage=='A':comparisons.setdefault(profile,{})['A_throughput_ratio_vs_official']=m['output_tokens_per_s_gpu']/official['scenario_a']['tokens_per_s_gpu']
            if stage=='C':comparisons.setdefault(profile,{}).update(C_ttft_p99_latency_ratio_vs_official=m['ttft_p99_s']/official['scenario_c']['ttft_p99_s'],C_tpot_p99_latency_ratio_vs_official=m['tpot_p99_s']/official['scenario_c']['tpot_p99_s'])
    write(out/'manifest.json',plan)
    result={'metrics':metrics,'comparisons':comparisons,'status':status,'scope':plan['scope'],'jobs':rows,'source_unchanged':True,
            'artifact_sha256':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}}
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--profiles',choices=PROFILES,nargs='+',required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--budget-minutes',type=float,default=90);a=p.parse_args()
    if len(set(a.profiles))!=len(a.profiles):p.error('Duplicate profiles')
    if not 1<=a.budget_minutes<=180:p.error('Budget1..180min')
    out=a.resume or Path('/workspace/results/mellum_full_campaign')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if a.resume:
        plan=json.loads((out/'manifest.json').read_text());current(out,plan)
        if plan['gpu']!=a.gpu or plan['profiles']!=a.profiles:raise ValueError('Resume configuration changed')
    else:out.mkdir(parents=True);plan=prepare(out,a.gpu,a.profiles)
    if a.prepare_only:print('RESULT full_campaign_prepared output',out);return
    r=execute(out,plan,a.budget_minutes*60);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
