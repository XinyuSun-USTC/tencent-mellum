"""Bounded GPU diagnostics: isolated batch decode or live disconnect observation."""
import argparse,json,os,re,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,build_opener,ProxyHandler
from mellum_runtime_config import PROJECT,apply_serving,engine_options
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha,chat_ids,load_rows
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
PROFILE='r_expert_decode4_kv_once'


def current(plan):
    for name,h in plan['source_sha256'].items():
        if sha(name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in plan['external_sha256'].items():
        if sha(name)!=h:raise ValueError('Frozen asset changed: '+name)


def load_steps(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def delivered_rate(events,cutoff):
    eligible=[x for x in events if x['time_ns']>cutoff]
    elapsed=(max((x['time_ns'] for x in eligible),default=cutoff)-cutoff)/1e9
    tokens=sum(len(x['token_ids']) for x in eligible)
    return {'delivered_tokens':tokens,'seconds_after_last_prefill':elapsed,
            'delivered_tokens_per_second':tokens/elapsed if elapsed else None,
            'required_C_output_rate':76.8,'scope':'Observed callback delivery after last prefill; includes host/transport effects. Not official C throughput or a universal upper bound.'}


def disconnect_evidence(steps,target,closed):
    owned=[r for r in steps if target in r['request_ids']]
    later=[r for r in owned if r['start_ns']>closed]
    return {'engine_steps_started_after_close':len(later),
            'inflight_steps_spanning_close':sum(r['start_ns']<=closed<r['end_ns'] for r in owned),
            'continued_host_step_seconds':sum(r['end_ns']-r['start_ns'] for r in later)/1e9,
            'engine_completion_observed':any(target in r['finished_request_ids'] for r in steps)}


def prepare(out,args):
    source=Path('/data/datasets/scenario_c_prompts.jsonl');rows=load_rows(source)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking',local_files_only=True,trust_remote_code=False)
    prompts=[chat_ids(tokenizer,[{'role':'user','content':r['prompt']}]) for r in rows[:32]]
    if any(not 4000<=len(x)<=16384 for x in prompts):raise ValueError('Unexpected actual C prompt lengths')
    files=list((PROJECT/'ATOM/atom/model_config').glob('mellum*.py'))+[
        PROJECT/'ATOM/atom/model_engine/engine_core.py',PROJECT/'ATOM/atom/model_engine/mellum_service_audit.py',
        PROJECT/'ATOM/atom/model_engine/scheduler.py',PROJECT/'ATOM/atom/entrypoints/openai/serving_chat.py',
        PROJECT/'ATOM/atom/entrypoints/openai/api_server.py',PROJECT/'scripts/mellum_execution_profiles.py',
        PROJECT/'scripts/serve_mellum.py',PROJECT/'scripts/mellum_runtime_config.py',Path(__file__).resolve()]
    plan={'track':args.track,'gpu':args.gpu,'profile':args.profile,'batches':args.batches,'tokens':args.tokens,
          'observe_seconds':args.observe_seconds,'prompts':prompts,'http_prompt':rows[0]['prompt'],
          'source_sha256':{str(p):sha(p) for p in files},'external_sha256':{str(source):sha(source)},
          'scope':'Diagnostic isolated decode or live disconnect, no correctness/capacity promotion; no model math changed.'}
    write(out/'manifest.json',plan);return plan


def decode_worker(out):
    plan=json.loads((out/'manifest.json').read_text());current(plan)
    env=apply_serving(plan['gpu'],out,True,'reference',True);env.update(profile_settings(plan['profile']))
    env['MELLUM_ENGINE_PHASE_AUDIT']=str(out/'engine_steps.jsonl')
    env['MELLUM_NATIVE_AUDIT_PATH']=str(out/'native_calls.jsonl');os.environ.update(env)
    sys.path.insert(0,str(PROJECT/'ATOM'))
    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.mellum_service_audit import summarize_steps
    options=engine_options(16640,None);options.update(max_num_seqs=32,port=13826+100*plan['gpu'],torch_profiler_dir=None)
    engine=None;result={'status':'started','profile':plan['profile'],'options':options,'cases':[]}
    try:
        engine=EngineArgs(**options).create_engine()
        # Exclude one short untimed warmup; no prefix cache is enabled.
        engine.generate([plan['prompts'][0][:1024]],SamplingParams(temperature=0,max_tokens=8,ignore_eos=True))
        for batch in plan['batches']:
            receipts=[];before=len(load_steps(out/'engine_steps.jsonl'));start=time.monotonic_ns();outputs={}
            def callback(i):
                def receive(value):receipts.append({'request_index':i,'time_ns':time.monotonic_ns(),'token_ids':[int(t) for t in value.output_tokens],'finished':bool(value.finished)})
                return receive
            engine.add_request(plan['prompts'][:batch],SamplingParams(temperature=0,max_tokens=plan['tokens'],ignore_eos=True),stream_callback=[callback(i) for i in range(batch)])
            while not engine.is_finished():outputs.update(engine.io_processor.postprocess(engine.step()))
            end=time.monotonic_ns()
            # The output may arrive before the EngineCore receipt is written.
            receipt_deadline=time.monotonic()+5
            while True:
                steps=load_steps(out/'engine_steps.jsonl')[before:]
                finished={i for r in steps for i in r['finished_request_ids']}
                if set(outputs)<=finished:break
                if time.monotonic()>receipt_deadline:raise TimeoutError('Final engine receipt missing')
                time.sleep(.01)
            if len(outputs)!=batch or any(len(o['token_ids'])!=plan['tokens'] for o in outputs.values()):raise ValueError('Incomplete isolated generation')
            prefills=[r for r in steps if r['phase']=='prefill']
            if not prefills:raise ValueError('No real prefill audit')
            cutoff=max(r['end_ns'] for r in prefills)
            write(out/f'decode_batch{batch}_callbacks.json',receipts);write(out/f'decode_batch{batch}_steps.json',steps)
            row={'batch':batch,'input_lengths':[len(x) for x in plan['prompts'][:batch]],'output_tokens':batch*plan['tokens'],
                 'wall_seconds':(end-start)/1e9,'phase_summary':summarize_steps(steps),'delivery':delivered_rate(receipts,cutoff)}
            result['cases'].append(row);write(out/'decode_results.json',result);print('batch',batch,'delivery',row['delivery'],flush=True)
        audit=json.loads((out/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise ValueError('Loading audit failed')
        result['status']='isolated_decode_measured'
    except Exception:result.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        if engine is not None:engine.close()
        write(out/'decode_results.json',result)


def stop_owned(proc):
    if proc is None:return
    # The server inherits the diagnostic worker's owned process group.
    # Outer run_stage kills any remaining model descendants on every exit.
    if proc.poll() is None:proc.terminate()
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill();proc.wait(timeout=5)


def disconnect_case(out,plan,mode):
    out.mkdir(parents=True,exist_ok=True);port=19751+plan['gpu']*100;engine_port=14926+plan['gpu']*100
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Refuse unowned occupied port')
    command=[sys.executable,str(PROJECT/'scripts/serve_mellum.py'),'--gpu',str(plan['gpu']),'--batching','--profile',plan['profile'],
             '--max-num-seqs','32','--max-model-len','16640','--max-batched-tokens','16640','--phase-audit',
             '--server-port',str(port),'--engine-port',str(engine_port),'--output',str(out/'server')]
    proc=None;response=None;result={'mode':mode,'status':'started','server_command':command}
    opener=build_opener(ProxyHandler({}));url=f'http://127.0.0.1:{port}'
    try:
        with (out/'server.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+240;heartbeat=time.monotonic()
            while True:
                if proc.poll() is not None:raise RuntimeError('Server exited before readiness')
                try:
                    with opener.open(url+'/health',timeout=2) as r:
                        if r.status==200:break
                except Exception:pass
                if time.monotonic()>deadline:raise TimeoutError('Readiness timeout')
                if time.monotonic()-heartbeat>15:print('WAIT',mode,out/'server.log',flush=True);heartbeat=time.monotonic()
                time.sleep(.5)
            with opener.open(url+'/v1/models',timeout=5) as r:model=json.load(r)['data'][0]['id']
            payload={'model':model,'messages':[{'role':'user','content':plan['http_prompt']}],'temperature':0,
                     'max_tokens':8 if mode=='normal' else 256,'ignore_eos':True,'stream':True,
                     'stream_options':{'include_usage':True}}
            started=time.monotonic_ns();response=opener.open(Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=90)
            frames=[];done=False;token_frame=False
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data:'):continue
                data=line[5:].strip();frames.append({'time_ns':time.monotonic_ns(),'payload':data})
                if data=='[DONE]':done=True;break
                value=json.loads(data)
                token_frame=any((c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning_content')) for c in value.get('choices',[]))
                if mode=='close_after_role' or (mode=='close_after_token' and token_frame):break
            closed=time.monotonic_ns();response.close();response=None
            if mode=='normal' and not done:raise ValueError('Normal completion positive control failed')
            if mode=='close_after_token' and not token_frame:raise ValueError('Did not observe a content token frame')
            result.update(client_start_ns=started,client_close_ns=closed,done_received=done,frames=frames)
            time.sleep(plan['observe_seconds'])
            text=(out/'server.log').read_text();ids=[int(x) for x in re.findall(r'API: Created request_id=\S+, seq_id=(\d+)',text)]
            if len(ids)!=1:raise ValueError('Expected exactly one owned HTTP request')
            steps=load_steps(out/'server/engine_steps.jsonl');target=ids[0]
            evidence=disconnect_evidence(steps,target,closed)
            result.update(status='disconnect_observed',request_id=target,**evidence,
                          verdict='engine_work_continued_after_disconnect' if mode!='normal' and evidence['engine_steps_started_after_close'] else 'no_post_close_step_observed',
                          scope='Live transport close, no abort repair. Observed subsequent engine steps, not exact GPU wasted time or full C causal contribution.')
            audit=json.loads((out/'server/loading_audit.json').read_text())
            if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise ValueError('Loading audit failed')
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    finally:
        if response is not None:response.close()
        stop_owned(proc);write(out/'results.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--track',choices=['decode','disconnect'],required=True)
    p.add_argument('--profile',choices=['r_expert_decode4_kv_once','r_decode_meta','r_swa_decode2','r_swa_decode4'],default=PROFILE)
    p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--batches',type=int,nargs='+',default=[1,4,8,16,32])
    p.add_argument('--tokens',type=int,choices=[64,128,256],default=128);p.add_argument('--observe-seconds',type=int,default=15)
    p.add_argument('--budget-minutes',type=float,default=40);p.add_argument('--prepare-only',action='store_true');p.add_argument('--decode-worker',type=Path);p.add_argument('--disconnect-worker',type=Path)
    a=p.parse_args()
    if a.decode_worker:decode_worker(a.decode_worker);return
    if a.disconnect_worker:
        job=json.loads((a.disconnect_worker/'plan.json').read_text());current(job)
        r=disconnect_case(a.disconnect_worker,job,job['mode']);current(job)
        raise SystemExit(0 if r['status']=='disconnect_observed' else 1)
    if any(b not in [1,4,8,16,32] for b in a.batches):p.error('Supported batch sizes1/4/8/16/32')
    if not 1<=a.observe_seconds<=60 or not 5<=a.budget_minutes<=90:p.error('Observation1..60sec, budget5..90min')
    out=Path('/workspace/results/mellum_c_service_diagnosis')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+a.track);out.mkdir(parents=True)
    plan=prepare(out,a)
    if a.prepare_only:print('RESULT prepared output',out);return
    result={'status':'started','scope':plan['scope']};end=time.monotonic()+a.budget_minutes*60
    try:
        current(plan)
        if a.track=='decode':
            run_stage('isolated_decode',[sys.executable,str(Path(__file__).resolve()),'--track','decode','--decode-worker',str(out)],out/'worker.log',timeout=max(1,end-time.monotonic()))
            result['decode']=json.loads((out/'decode_results.json').read_text());result['status']=result['decode']['status']
        else:
            result['cases']={}
            # Each live case has an independent engine: no abandoned request contaminates the next case.
            for mode in ['normal','close_after_role','close_after_token']:
                if end-time.monotonic()<60:raise TimeoutError('Insufficient budget for another owned server')
                target=out/mode;target.mkdir()
                job={k:plan[k] for k in ['gpu','profile','observe_seconds','http_prompt','source_sha256','external_sha256']};job['mode']=mode
                write(target/'plan.json',job)
                try:
                    run_stage(mode,[sys.executable,str(Path(__file__).resolve()),'--track','disconnect','--disconnect-worker',str(target)],target/'worker.log',timeout=min(420,end-time.monotonic()),allowed_exit_codes=(0,1))
                    result['cases'][mode]=json.loads((target/'results.json').read_text())
                except Exception:result['cases'][mode]={'status':'failed','traceback':traceback.format_exc()}
                write(out/'summary.json',result)
            result['status']='disconnect_diagnosis_complete' if all(c['status']=='disconnect_observed' for c in result['cases'].values()) else 'disconnect_diagnosis_partial'
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    current(plan);result['source_unchanged']=True
    result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
    write(out/'summary.json',result);print('RESULT',result['status'],'output',out)
if __name__=='__main__':main()
