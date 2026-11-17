"""Single-command HTTP A/B/C baseline with the audited EvalScope release.

Dataset compatibility is explicit; no vendor code, model math or metrics patched.
"""
import argparse,collections,hashlib,json,math,os,signal,socket,sqlite3,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
from mellum_runtime_config import PROJECT,MODEL
from mellum_perf_protocol import literal_command,sha,load_rows
from mellum_process_runner import run_stage
from mellum_execution_profiles import PROFILES
from run_mellum_batch_campaign import write

CLIENT=Path('/workspace/venvs/evalscope-perf/bin/python')
ADAPTER=PROJECT/'scripts/evalscope_mellum_adapter.py'
AUDIT=Path('/workspace/results/mellum_perf_client/20261115T144518825402Z')
CONFIG=PROJECT/'configs/mellum/shared_routing.json'
LIMITS={'smoke':900,'B':1800,'A':5400,'C':1800}


def cpu_environment():
    env=dict(os.environ);env.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    env.pop('PYTHONPATH',None);return env


def sources():
    paths=list((PROJECT/'ATOM/atom').rglob('*.py'))+[CONFIG,ADAPTER,Path(__file__).resolve(),PROJECT/'start_server.sh',PROJECT/'configs/submission/environment.json']
    paths += [PROJECT/'scripts'/name for name in ('mellum_delivery_matrix.py','smoke_delivery.py','serve_delivery.py','run_delivery_matrix.py','serve_mellum.py','mellum_execution_profiles.py','mellum_runtime_config.py','mellum_perf_protocol.py','mellum_process_runner.py','setup_mellum_perf_client.py')]
    return {str(p):sha(p) for p in sorted(set(paths))}


def validate_client(audit_run=AUDIT):
    audit_run=Path(audit_run)
    setup=json.loads((audit_run/'summary.json').read_text());audit=setup['audit']
    if setup['status']!='client_audit_complete' or not setup['frozen_runtime_distributions_unchanged']:
        raise ValueError('Require successful independent client audit')
    if audit['versions']['evalscope']!='1.15.0' or not all(audit['stream_probe']['checks'][k] for k in
            ('success','reasoning_text_preserved','usage_counts_preserved','first_observable_output_timed','last_content_observed')):
        raise ValueError('Pinned client/reasoning stream audit invalid')
    for record in audit['source_sha256'].values():
        if sha(record['path'])!=record['sha256']:raise ValueError('Audited client source changed')
    from setup_mellum_perf_client import frozen_state
    if frozen_state()!=json.loads((audit_run/'frozen_after.json').read_text()):raise ValueError('Frozen inference distribution state changed')
    actual=subprocess.check_output([str(CLIENT),'-B','-c',"from importlib.metadata import version; print(version('evalscope'),version('transformers'))"],env=cpu_environment(),text=True).strip()
    if actual!='1.15.0 5.10.1':raise ValueError('Installed client versions changed')
    return {'audit_run':str(audit_run),'audit_summary_sha256':sha(audit_run/'summary.json'),'versions':audit['versions'],
            'source_sha256':{r['path']:r['sha256'] for r in audit['source_sha256'].values()}}


def command(scenario,directory,number,source=None,server_port=8051):
    # Literal PDF command is retained separately in the manifest. Changes here:
    # namespaced prompt-field plugin for A/C, paths for output, deterministic seed.
    cmd=literal_command(scenario,f'http://127.0.0.1:{server_port}/v1/chat/completions',directory)
    cmd[:1]=[str(CLIENT),str(ADAPTER)]
    cmd[cmd.index('--number')+1]=str(number)
    # EvalScope 1.15 uses BooleanOptionalAction, unlike the PDF's value syntax.
    stream=cmd.index('--stream')
    if cmd[stream+1]=='True':del cmd[stream+1]
    if scenario in ('A','C'):
        old=cmd.index('--dataset');cmd[old+1]='mellum_assessment_jsonl'
        if '--dataset-path' in cmd:cmd[cmd.index('--dataset-path')+1]=str(source)
        else:cmd[old+2:old+2]=['--dataset-path',str(source)]
    cmd += ['--outputs-dir',str(directory),'--no-timestamp','--seed','42']
    return cmd


def percentile(values,p):
    if not values:return None
    ordered=sorted(values);return ordered[max(0,math.ceil(len(values)*p/100)-1)]


def inspect_result(directory,scenario,count,expected,budget=256):
    databases=list(Path(directory).rglob('benchmark_data.db'))
    summaries=list(Path(directory).rglob('benchmark_summary.json'))
    percentiles=list(Path(directory).rglob('benchmark_percentile.json'))
    if len(databases)!=1 or len(summaries)!=1:raise ValueError('Expected one complete EvalScope DB and summary')
    with sqlite3.connect(f'file:{databases[0]}?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row;rows=[dict(r) for r in con.execute('SELECT * FROM result')]
    errors=[]
    if len(rows)!=count:errors.append('Recorded request count differs from required number')
    successes=[r for r in rows if r['success']];observed=[]
    for row in rows:
        body=json.loads(row['request']);messages=body.get('messages')
        if not isinstance(messages,list) or len(messages)!=1 or messages[0].get('role')!='user':
            errors.append('Unexpected request messages');continue
        digest=hashlib.sha256(messages[0]['content'].encode()).hexdigest()
        observed.append(digest)
        expected_lengths={r['prompt_sha256']:r.get('tokens') for r in expected}
        if row['success'] and expected_lengths.get(digest) is not None and row['prompt_tokens']!=expected_lengths[digest]:
            errors.append('Server/client rendered input token lengths differ')
        if body.get('max_tokens')!=budget or body.get('min_tokens')!=budget or body.get('stream') is not True or body.get('ignore_eos') is not True or body.get('temperature')!=0:
            errors.append('Request generation settings differ')
    if scenario in ('A','C') and collections.Counter(observed)!=collections.Counter(r['prompt_sha256'] for r in expected):
        errors.append('Actual prompt multiset differs from native filtered/cycled manifest')
    if any(r['completion_tokens']!=budget for r in successes):errors.append('Successful reply token count differs from 256')
    if any(not r['is_stream'] for r in rows):errors.append('Non-stream request recorded')
    if scenario=='B' and any(r['prompt_tokens']!=1024 for r in successes):errors.append('B input length differs from 1024')
    vendor=json.loads(summaries[0].read_text())
    metrics={'requests':len(rows),'successes':len(successes),'failures':len(rows)-len(successes),
        'ttft_p50_s':percentile([r['first_chunk_latency'] for r in successes],50),
        'ttft_p99_s':percentile([r['first_chunk_latency'] for r in successes],99),
        'tpot_p50_s':percentile([r['time_per_output_token'] for r in successes],50),
        'tpot_p99_s':percentile([r['time_per_output_token'] for r in successes],99),
        'output_tokens_per_s_gpu':vendor.get('Output Throughput (tok/s)'),
        'vendor_summary':vendor,'vendor_percentiles':json.loads(percentiles[0].read_text()) if len(percentiles)==1 else None,
        'request_start_span_s':max((r['start_time'] for r in rows),default=0)-min((r['start_time'] for r in rows),default=0),
        'validation_errors':sorted(set(errors)),
        'complete_without_failures':len(rows)==count and len(successes)==count and not errors,
        'tail_note':'Client vendor percentiles are success-only; failure counts retained. Do not treat partial-success tails as a valid complete score.'}
    return metrics


def wait_client(process,server,logpath,deadline):
    last=time.monotonic()
    while process.poll() is None:
        if server.poll() is not None:raise RuntimeError('Owned inference server exited')
        serverlog=logpath.parent.parent/'server.log'
        if serverlog.exists():
            with serverlog.open('rb') as stream:
                stream.seek(0,2);stream.seek(max(0,stream.tell()-65536));tail=stream.read().decode(errors='replace')
            if any(message in tail.lower() for message in ('proc died unexpectedly','out of memory','hip error')):
                raise RuntimeError('Inference worker failure; inspect server.log')
        if time.monotonic()>deadline:raise TimeoutError('Stage client exceeded remaining budget')
        if time.monotonic()-last>=15:
            print('CLIENT running, log bytes',logpath.stat().st_size,'server alive',flush=True);last=time.monotonic()
        time.sleep(.25)
    if process.returncode:raise RuntimeError('EvalScope client exited '+str(process.returncode))


def stage_worker(directory):
    plan=json.loads((directory/'plan.json').read_text());server=None;client=None
    result={'status':'started','cases':{},'scope':'Pinned local assessment-scale reproduction; adapter and arrival normalization disclosed'}
    write(directory/'results.json',result)
    deadline=time.monotonic()+plan['timeout']-35
    try:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',plan.get('server_port',8051)))==0:raise RuntimeError('HTTP port already in use; refuse unowned server')
        servercmd=['/opt/venv/bin/python',str(PROJECT/'scripts/serve_mellum.py'),'--gpu',str(plan['gpu']),
                   '--batching','--batch-audit','--max-num-seqs',str(__import__('mellum_delivery_matrix').slots(plan.get('profile'))),'--max-model-len','16640','--max-batched-tokens','16640',
                   '--output',str(directory/'server'),'--server-port',str(plan.get('server_port',8051)),
                   '--engine-port',str(plan.get('engine_port',8026))]
        if plan.get('profile'):servercmd+=['--profile',plan['profile']]
        else:servercmd+=['--campaign-config',str(CONFIG)]
        # Inherit this owned stage process group, so outer timeout cleanup reaches
        # server/model workers even when their parent has exited unexpectedly.
        with (directory/'server.log').open('w') as log:server=subprocess.Popen(servercmd,stdout=log,stderr=subprocess.STDOUT)
        opener=build_opener(ProxyHandler({}));started=time.monotonic();last=started
        while True:
            if server.poll() is not None:raise RuntimeError('Server exited during initialization')
            try:
                with opener.open(f"http://127.0.0.1:{plan.get('server_port',8051)}/health",timeout=2) as response:
                    if response.status==200:break
            except Exception:pass
            if time.monotonic()-started>300:raise TimeoutError('Server readiness exceeded 300 seconds')
            if time.monotonic()-last>15:print('Waiting for owned server',directory/'server.log',flush=True);last=time.monotonic()
            time.sleep(.5)
        result['server_command']=servercmd;result['server_load_audit']=json.loads((directory/'server/loading_audit.json').read_text())
        if result['server_load_audit']!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:
            raise ValueError('Model loading audit differs')
        for spec in plan['cases']:
            name=spec['name'];out=directory/name;out.mkdir()
            cmd=command(spec['scenario'],out/'client',spec['number'],spec.get('source'),plan.get('server_port',8051))
            result['cases'][name]={'command':cmd,'status':'running'};write(directory/'results.json',result)
            auditpath=directory/'server/batches.jsonl';offset=auditpath.stat().st_size if auditpath.exists() else 0
            logfile=out/'client.log';print('CLIENT',name,'log',logfile,flush=True)
            with logfile.open('w') as log:client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=cpu_environment())
            wait_client(client,server,logfile,deadline)
            measured=inspect_result(out/'client',spec['scenario'],spec['number'],spec.get('expected',[]))
            if auditpath.exists():
                with auditpath.open('rb') as stream:stream.seek(offset);batchbytes=stream.read()
                (out/'batches.jsonl').write_bytes(batchbytes)
                batches=[json.loads(line) for line in batchbytes.decode().splitlines() if line.strip()]
                measured['actual_decode_max_batch']=max((r['batch_size'] for r in batches if r['phase']=='decode'),default=0)
                if spec['scenario']=='A' and measured['actual_decode_max_batch']!=min(32,spec['number']):
                    measured['validation_errors'].append('Requested actual A decode concurrency not observed');measured['complete_without_failures']=False
            result['cases'][name].update(status='complete',metrics=measured);write(directory/'results.json',result)
            print('CLIENT complete',name,'success',measured['successes'],'/',measured['requests'],flush=True)
            if plan['stage']=='smoke' and not measured['complete_without_failures']:
                result['status']='smoke_failed';write(directory/'results.json',result);return
        result['status']='complete';write(directory/'results.json',result)
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());write(directory/'results.json',result);raise
    finally:
        for proc in (client,server):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)


def prepare(out,gpu,smoke_only,audit_run=AUDIT):
    client=validate_client(audit_run)
    run_stage('CPU_adapter',[str(CLIENT),str(ADAPTER),'--prepare',str(out/'inputs')],out/'adapter_prepare.log',timeout=180)
    native=json.loads((out/'inputs/adapter_manifest.json').read_text());a=native['scenarios']['A'];c=native['scenarios']['C']
    rows=load_rows(c['source']);byid={r['id']:r for r in rows}
    longest=max(c['planned_requests'],key=lambda r:r['tokens']);selected=[c['planned_requests'][0],longest]
    if selected[0]['id']==selected[1]['id']:selected[0]=c['planned_requests'][1]
    smokeC=out/'inputs/C_smoke.jsonl';smokeC.write_text(''.join(json.dumps(byid[r['id']])+'\n' for r in selected))
    def spec(name,scenario,number,source=None,expected=None):return {'name':name,'scenario':scenario,'number':number,'source':source,'expected':expected or []}
    stages={'smoke':[spec('smoke_B','B',2),spec('smoke_A','A',8,a['source'],a['planned_requests'][:8]),
                     spec('smoke_C','C',2,str(smokeC),selected)]}
    if not smoke_only:
        stages.update(B=[spec('B','B',50)],A=[spec('A','A',512,a['source'],a['planned_requests'])],
                      C=[spec('C','C',200,c['source'],c['planned_requests'])])
    manifest={'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'gpu':gpu,'smoke_only':smoke_only,
        'source_sha256':sources(),'client':client,'adapter_manifest_sha256':sha(out/'inputs/adapter_manifest.json'),
        'input_sha256':{**{x['source']:x['source_sha256'] for x in (a,c)},str(smokeC):sha(smokeC),
                          str(MODEL/'config.json'):sha(MODEL/'config.json'),
                          str(MODEL/'model.safetensors.index.json'):sha(MODEL/'model.safetensors.index.json'),
                          '/data/datasets/baseline.json':sha('/data/datasets/baseline.json')},'stages':stages,'jobs':{},
        'literal_commands':{s:literal_command(s,'http://127.0.0.1:8051/v1/chat/completions',out) for s in 'ABC'},
        'differences':['A/C prompt-field dataset adapter, using native prepare_messages filter/wrapping',
          'A native eligible requests cycle to512; explicit ID/token manifest retained',
          'Seed42 and output paths recorded; inference defaults not changed',
          'PDF --stream True translated to EvalScope1.15 --stream; streaming remains enabled',
          'EvalScope1.15 open-loop exponentials are rescaled to total n/r, not strict unconditioned Poisson'],
        'claim':'Local assessment-scale reproduction, not asserted identical to unspecified judge EvalScope version'}
    write(out/'manifest.json',manifest);return manifest


def current_matches(manifest,out):
    if manifest['source_sha256']!=sources():raise ValueError('Benchmark/model source changed')
    validate_client(manifest['client']['audit_run'])
    if sha(out/'inputs/adapter_manifest.json')!=manifest['adapter_manifest_sha256']:raise ValueError('Input manifest changed')
    for path,digest in manifest['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Original scenario source changed')


def execute(out,manifest,budget,executor=run_stage):
    deadline=time.monotonic()+budget-manifest.get('used_seconds',0);results={};status='complete';error=None
    for stage,specs in manifest['stages'].items():
        current_matches(manifest,out)
        directory=out/stage;directory.mkdir(exist_ok=True)
        old=manifest['jobs'].get(stage)
        if old and old['status']=='complete':
            if any(sha(directory/rel)!=v for rel,v in old['artifacts'].items()):raise ValueError('Completed stage artifact changed')
            value=json.loads((directory/'results.json').read_text());results[stage]=value
        else:
            remaining=deadline-time.monotonic()
            if remaining<60:status='partial_budget_limited';break
            if (directory/'plan.json').exists():
                n=1
                while (directory/f'attempt_{n}').exists():n+=1
                archive=directory/f'attempt_{n}';archive.mkdir()
                for file in list(directory.iterdir()):
                    if file!=archive and not file.name.startswith('attempt_'):file.rename(archive/file.name)
            timeout=min(remaining,LIMITS[stage]);plan={'stage':stage,'gpu':manifest['gpu'],'cases':specs,'timeout':timeout,
                'profile':manifest.get('profile'),'server_port':manifest.get('server_port',8051),'engine_port':manifest.get('engine_port',8026)}
            write(directory/'plan.json',plan);manifest['jobs'][stage]={'status':'running'};write(out/'manifest.json',manifest)
            started=time.monotonic()
            try:
                executor(stage,['/opt/venv/bin/python',str(Path(__file__).resolve()),'--stage',str(directory)],directory/'stage.log',timeout=timeout)
                current_matches(manifest,out)
                value=json.loads((directory/'results.json').read_text());results[stage]=value
                manifest['jobs'][stage].update(status='complete',artifacts={str(f.relative_to(directory)):sha(f) for f in directory.rglob('*')
                    if f.is_file() and 'attempt_' not in str(f.relative_to(directory))})
            except Exception as exc:
                status='partial_budget_limited' if time.monotonic()>=deadline else 'stage_failed';error=str(exc)
                manifest['jobs'][stage].update(status='failed',error=error)
                if (directory/'results.json').exists():results[stage]=json.loads((directory/'results.json').read_text())
                break
            finally:
                manifest['used_seconds']=manifest.get('used_seconds',0)+time.monotonic()-started;write(out/'manifest.json',manifest)
        if stage=='smoke' and (value['status']!='complete' or any(not r['metrics']['complete_without_failures'] for r in value['cases'].values())):
            status='smoke_failed';break
    formal={k:v for k,v in results.items() if k!='smoke'}
    if status=='complete':
        if manifest['smoke_only']:status='smoke_passed'
        elif len(formal)!=3 or any(v['status']!='complete' or any(not c['metrics']['complete_without_failures'] for c in v['cases'].values()) for v in formal.values()):status='complete_with_failures'
    baseline=json.loads(Path('/data/datasets/baseline.json').read_text());comparisons={}
    for stage,value in formal.items():
        for case in value.get('cases',{}).values():
            measured=case.get('metrics',{})
            if not measured.get('complete_without_failures'):continue
            if stage=='A':comparisons['A_throughput_ratio_vs_official']=measured['output_tokens_per_s_gpu']/baseline['scenario_a']['tokens_per_s_gpu']
            if stage=='C':comparisons.update(C_ttft_p99_latency_ratio_vs_official=measured['ttft_p99_s']/baseline['scenario_c']['ttft_p99_s'],
                                             C_tpot_p99_latency_ratio_vs_official=measured['tpot_p99_s']/baseline['scenario_c']['tpot_p99_s'])
    summary={'status':status,'error':error,'stages':results,'comparisons':comparisons,'scope':manifest['claim'],
             'protocol_differences':manifest['differences'],'note':'No score invented from null target_g fields; failures and partial-success tails preserved.'}
    write(out/'summary.json',summary)
    lines=['# '+(manifest.get('profile') or 'R')+' 单卡A/B/C本地复测','','状态：`'+status+'`。这是注明客户端适配与版本差异的本地题目负载复现。','',
           '| 阶段 | 成功/请求 | tokens/s/GPU | TTFT p50秒 | TTFT p99秒 | TPOT p50秒 | TPOT p99秒 |','| --- | --- | --- | --- | --- | --- | --- |']
    for stage,value in results.items():
        for name,case in value.get('cases',{}).items():
            r=case.get('metrics',{})
            if r:lines.append(f"| {name} | {r['successes']}/{r['requests']} | {r['output_tokens_per_s_gpu']} | {r['ttft_p50_s']} | {r['ttft_p99_s']} | {r['tpot_p50_s']} | {r['tpot_p99_s']} |")
    lines+=['','失败/预算不完整的阶段不作为有效完整成绩。p99仅成功请求时不能忽略失败率。','',json.dumps(comparisons,indent=2)]
    (out/'report.md').write_text('\n'.join(lines)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--stage',type=Path)
    p.add_argument('--profile',choices=PROFILES)
    p.add_argument('--server-port',type=int,default=8051);p.add_argument('--engine-port',type=int,default=8026)
    p.add_argument('--client-audit',type=Path,default=AUDIT)
    p.add_argument('--budget-minutes',type=float,default=180);a=p.parse_args()
    if a.stage:stage_worker(a.stage);return 0
    if not 1<=a.server_port<=65535 or not 1<=a.engine_port<=65535 or a.server_port==a.engine_port:p.error('Distinct valid HTTP/engine ports required')
    if not 0<a.budget_minutes<=180:p.error('Total budget must be positive and <=180 minutes')
    out=a.resume or Path('/workspace/results/mellum_abc_optimized' if a.profile else '/workspace/results/mellum_abc_baseline')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=True)
    if a.resume:
        manifest=json.loads((out/'manifest.json').read_text());current_matches(manifest,out)
        if (manifest['gpu']!=a.gpu or manifest['smoke_only']!=a.smoke_only or manifest.get('profile')!=a.profile
            or manifest.get('server_port',8051)!=a.server_port or manifest.get('engine_port',8026)!=a.engine_port):raise ValueError('Resume options differ')
    else:
        manifest=prepare(out,a.gpu,a.smoke_only,a.client_audit)
        manifest.update(profile=a.profile,server_port=a.server_port,engine_port=a.engine_port,
            measurement_note='Single local full-load reproduction; concurrent GPU jobs may contend for shared host resources. Final idle-machine repetitions are separate.')
        write(out/'manifest.json',manifest)
    if a.prepare_only:print('RESULT abc_baseline_prepared output',out);return 0
    result=execute(out,manifest,a.budget_minutes*60)
    print('RESULT',result['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
