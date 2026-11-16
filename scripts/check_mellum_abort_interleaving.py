"""Verify an immutable abort candidate with a surviving request and pool reuse."""
import argparse,hashlib,json,re,socket,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,build_opener,ProxyHandler
from mellum_process_runner import run_stage


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def checked_candidate(source,run):
    manifest=json.loads((run/'manifest.json').read_text());verdict=json.loads((run/'summary.json').read_text())
    if manifest['profile']!='r_abort' or verdict['status']!='disconnect_diagnosis_complete':raise ValueError('Require completed r_abort check')
    if str(source/'ATOM/atom/model_engine/engine_core.py') not in manifest['source_sha256']:raise ValueError('Wrong source tree')
    for name,h in manifest['source_sha256'].items():
        if sha(name)!=h:raise ValueError('Measured candidate source changed: '+name)
    for name,h in verdict['artifact_sha256'].items():
        if sha(run/name)!=h:raise ValueError('Prior result changed: '+name)
    for name in ['close_after_role','close_after_token']:
        case=verdict['cases'][name]
        if not case.get('abort_receipts') or case['engine_steps_started_after_abort']!=0:raise ValueError('Prior abort did not pass')
    return manifest


class Stream:
    def __init__(self,response):self.response=response;self.frames=[];self.text='';self.usage=None;self.request_id=None;self.done=False
    def advance(self,stop_after_role=False,stop_after_token=False):
        for raw in self.response:
            line=raw.decode().strip()
            if not line.startswith('data:'):continue
            payload=line[5:].strip();self.frames.append({'time_ns':time.monotonic_ns(),'payload':payload})
            if payload=='[DONE]':self.done=True;return
            d=json.loads(payload)
            if d.get('id'):self.request_id=d['id']
            if d.get('usage') is not None:self.usage=d['usage']
            text=''.join((c.get('delta',{}).get('content') or '')+(c.get('delta',{}).get('reasoning_content') or '') for c in d.get('choices',[]))
            self.text+=text
            if stop_after_role or (stop_after_token and text):return
        if not self.done:raise ValueError('SSE ended without DONE or requested stopping point')
    def finished(self,budget):
        if not self.done or not self.text or not self.usage or self.usage.get('completion_tokens')!=budget:raise ValueError('Surviving/recovery output or usage incomplete')
        return {'request_id':self.request_id,'text':self.text,'usage':self.usage,'done':self.done,'frames':self.frames}


def lookup_seq(log,request_id):
    ids=re.findall(r'API: Created request_id='+re.escape(request_id)+r', seq_id=(\d+)',log)
    if len(ids)!=1:raise ValueError('Cannot uniquely map HTTP request to engine sequence')
    return int(ids[0])


def verify_lifetime(rows,cancel_id,survivor_id,closed):
    receipts=[r for r in rows if r.get('kind')=='abort' and cancel_id in r.get('removed',[])]
    if not receipts:raise ValueError('Canceled request was not removed')
    last=receipts[-1]
    if last['running']<1:raise ValueError('No surviving request remained active when cancellation was processed')
    later=[r for r in rows if r.get('kind')=='engine_step' and r['start_ns']>last['time_ns']]
    if any(cancel_id in r['request_ids'] for r in later):raise ValueError('Canceled request scheduled after removal')
    if not any(survivor_id in r['request_ids'] for r in later):raise ValueError('No survivor execution after removal')
    if not any(survivor_id in r.get('finished_request_ids',[]) for r in rows):raise ValueError('No survivor completion receipt')
    return {'abort_receipts':receipts,'close_to_removal_seconds':(last['time_ns']-closed)/1e9,
            'post_abort_survivor_steps':sum(survivor_id in r['request_ids'] for r in later),
            'post_abort_canceled_steps':0,'scope':'Actual survivor and pool-reuse smoke; no full C capacity or model-accuracy claim'}


def worker(out):
    plan=json.loads((out/'plan.json').read_text());source=Path(plan['source']);mode=plan['mode']
    for name,h in plan['source_sha256'].items():
        if sha(name)!=h:raise ValueError('Frozen source changed')
    port=20751+100*plan['gpu'];engine_port=15926+100*plan['gpu']
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Refuse unowned server port')
    command=[sys.executable,str(source/'scripts/serve_mellum.py'),'--gpu',str(plan['gpu']),'--batching','--profile','r_abort',
        '--max-num-seqs','32','--max-model-len','16640','--max-batched-tokens','16640','--phase-audit',
        '--server-port',str(port),'--engine-port',str(engine_port),'--output',str(out/'server')]
    result={'status':'started','mode':mode,'command':command};proc=None;responses=[]
    opener=build_opener(ProxyHandler({}));base=f'http://127.0.0.1:{port}'
    def request(model,prompt,budget):
        data={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':budget,'ignore_eos':True,
              'stream':True,'stream_options':{'include_usage':True}}
        response=opener.open(Request(base+'/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=90)
        responses.append(response);return Stream(response)
    try:
        with (out/'server.log').open('w') as log:
            # Parent run_stage owns this group and cleans all descendants.
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+240
            while True:
                if proc.poll() is not None:raise RuntimeError('Server exited before ready')
                try:
                    with opener.open(base+'/health',timeout=2) as response:
                        if response.status==200:break
                except Exception:pass
                if time.monotonic()>deadline:raise TimeoutError('Readiness timeout')
                time.sleep(.5)
            with opener.open(base+'/v1/models',timeout=5) as response:model=json.load(response)['data'][0]['id']
            control=request(model,plan['prompts'][0],plan['survivor_tokens']);control.advance();result['control']=control.finished(plan['survivor_tokens']);control.response.close()
            survivor=request(model,plan['prompts'][0],plan['survivor_tokens']);survivor.advance(stop_after_token=True)
            canceled=request(model,plan['prompts'][1],256)
            canceled.advance(stop_after_role=mode=='role',stop_after_token=mode=='token')
            closed=time.monotonic_ns();canceled.response.close()
            survivor.advance();result['survivor']=survivor.finished(plan['survivor_tokens']);survivor.response.close()
            result['survivor_text_equals_solo']=survivor.text==control.text
            # Exercise allocation/generation after cancellation and normal completion.
            recovery=request(model,'Write a Python function add(a, b) that returns their sum.',8)
            recovery.advance();result['recovery']=recovery.finished(8);recovery.response.close()
            deadline=time.monotonic()+5
            while True:
                text=(out/'server.log').read_text();cancel_id=lookup_seq(text,canceled.request_id);survivor_id=lookup_seq(text,survivor.request_id)
                audit=out/'server/engine_steps.jsonl';rows=[json.loads(x) for x in audit.read_text().splitlines()]
                recovery_id=lookup_seq(text,recovery.request_id)
                finished={i for r in rows for i in r.get('finished_request_ids',[])}
                if {survivor_id,recovery_id}<=finished:break
                if time.monotonic()>deadline:raise TimeoutError('Final receipt missing')
                time.sleep(.01)
            result['lifetime']=verify_lifetime(rows,cancel_id,survivor_id,closed)
            wrong=[r for r in rows if r.get('kind')=='abort' and set(r.get('requested',[]))-{cancel_id}]
            if wrong:raise ValueError('A surviving/control request was also sent to abort')
            final=next(r for r in reversed(rows) if recovery_id in r.get('finished_request_ids',[]))
            if final['waiting'] or final['running']:raise ValueError('Requests remain after recovery completion')
            result['final_waiting_running']=[final['waiting'],final['running']]
            result.update(status='abort_interleaving_passed',cancel_id=cancel_id,survivor_id=survivor_id,close_ns=closed,
                canceled_frames=canceled.frames,source_unchanged=True)
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    finally:
        for response in responses:response.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        write(out/'results.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('/workspace/tencent-mellum-disconnect-abort'))
    p.add_argument('--verified-run',type=Path,default=Path('/workspace/results/mellum_c_service_diagnosis/20261116T223755052494Z_disconnect'))
    p.add_argument('--gpu',type=int,choices=[0,1],default=1);p.add_argument('--survivor-tokens',type=int,choices=[64,128,256],default=128)
    p.add_argument('--budget-minutes',type=float,default=20);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.worker:raise SystemExit(0 if worker(a.worker)['status']=='abort_interleaving_passed' else 1)
    if not 5<=a.budget_minutes<=40:p.error('Budget5..40min')
    source=a.source_root.resolve();original=checked_candidate(source,a.verified_run)
    rows=[json.loads(x) for x in Path('/data/datasets/scenario_c_prompts.jsonl').read_text().splitlines()]
    out=Path('/workspace/results/mellum_abort_interleaving')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    sources={**original['source_sha256'],str(Path(__file__).resolve()):sha(__file__)}
    plan={'source':str(source),'verified_run':str(a.verified_run),'verified_summary_sha256':sha(a.verified_run/'summary.json'),
          'gpu':a.gpu,'survivor_tokens':a.survivor_tokens,'prompts':[r['prompt'] for r in rows[:2]],'source_sha256':sources,
          'dataset_sha256':sha('/data/datasets/scenario_c_prompts.jsonl'),'scope':'Paired active/canceled chat requests and fresh allocation; not full correctness or C acceptance'}
    write(out/'manifest.json',plan)
    if a.prepare_only:print('RESULT prepared output',out);return
    result={'status':'started','cases':{},'scope':plan['scope']};end=time.monotonic()+a.budget_minutes*60
    for mode in ['role','token']:
        target=out/mode;target.mkdir();write(target/'plan.json',{**plan,'mode':mode})
        try:
            run_stage('abort_interleaving_'+mode,[sys.executable,str(Path(__file__).resolve()),'--worker',str(target)],target/'worker.log',timeout=min(480,max(1,end-time.monotonic())),allowed_exit_codes=(0,1))
            result['cases'][mode]=json.loads((target/'results.json').read_text())
        except Exception:result['cases'][mode]={'status':'failed','traceback':traceback.format_exc()}
        write(out/'summary.json',result)
    for name,h in sources.items():
        if sha(name)!=h:raise ValueError('Source changed during test')
    result['status']='abort_interleaving_passed' if all(x['status']=='abort_interleaving_passed' for x in result['cases'].values()) else 'abort_interleaving_failed'
    result['source_unchanged']=True;result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
    write(out/'summary.json',result);print('RESULT',result['status'],'output',out)
if __name__=='__main__':main()
