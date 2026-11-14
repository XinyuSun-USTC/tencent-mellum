"""Owned short HTTP smoke test; real generation, SSE and concurrent requests."""
import argparse,concurrent.futures,json,os,signal,socket,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,build_opener,ProxyHandler
from mellum_runtime_config import PROJECT,MODEL
from validate_mellum_batch import prepare


def parse_sse(lines):
    text=[];usage=None;done=False;chat={'content':'','reasoning_content':''}
    for line in lines:
        if not line.startswith('data:'):continue
        data=line[5:].strip()
        if data=='[DONE]':done=True;continue
        item=json.loads(data)
        if item.get('usage') is not None:usage=item['usage']
        for choice in item.get('choices',[]):
            text.append(choice.get('text') or '')
            delta=choice.get('delta',{})
            for key in chat:chat[key]+=delta.get(key) or ''
    return {'text':''.join(text),'usage':usage,'done':done,'chat':chat}


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--server-port',type=int,default=8051)
    p.add_argument('--engine-port',type=int,default=8026);p.add_argument('--max-seqs',type=int,choices=[1,4],default=1);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();out=Path('/workspace/results/mellum_http_validation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    cases=prepare()[:2]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    for c in cases:
        if tokenizer.encode(c['rendered_prompt'])!=c['input_ids']:raise ValueError('HTTP prompt re-encoding differs from HF input')
    command=['/opt/venv/bin/python',str(PROJECT/'scripts/serve_mellum.py'),'--gpu',str(a.gpu),
        '--server-port',str(a.server_port),'--engine-port',str(a.engine_port),'--max-num-seqs',str(a.max_seqs),
        '--max-model-len','1152','--max-batched-tokens','4096','--output',str(out/'server')]
    if a.max_seqs>1:command+=['--batching','--batch-audit']
    result={'status':'prepared','scope':'Small HTTP smoke, not A/B/C performance or public accuracy acceptance',
            'command':command,'cases':cases,'requests':[]}
    def save():(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    if a.prepare_only:print('RESULT prepared output',out);return 0
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',a.server_port))==0:raise RuntimeError('HTTP port already occupied; refuse reuse')
    opener=build_opener(ProxyHandler({}));base=f'http://127.0.0.1:{a.server_port}'
    def call(path,payload=None):
        req=Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=180) as response:
            if payload and payload.get('stream'):return parse_sse([line.decode().strip() for line in response])
            return json.loads(response.read())
    def completion(c,stream=False):
        data=call('/v1/completions',{'model':str(MODEL),'prompt':c['rendered_prompt'],'temperature':0,
            'max_tokens':c['budget'],'ignore_eos':False,'stream':stream})
        text=data['text'] if stream else data['choices'][0]['text']
        expected=tokenizer.decode(c['expected_ids'],skip_special_tokens=True)
        return {'case':c['name'],'stream':stream,'text':text,'expected_text':expected,'text_exact':text==expected,
            'usage':data.get('usage'),'done':data.get('done') if stream else None,
            'usage_matches':data.get('usage',{}).get('completion_tokens')==len(c['expected_ids'])}
    proc=None
    try:
        with (out/'server.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+300;heartbeat=time.monotonic()
            while True:
                if proc.poll() is not None:raise RuntimeError('Server exited; see server.log')
                try:call('/health');break
                except Exception:
                    if time.monotonic()>deadline:raise TimeoutError('Server startup timeout')
                    if time.monotonic()-heartbeat>15:print('HTTP waiting for server; log',out/'server.log',flush=True);heartbeat=time.monotonic()
                    time.sleep(.5)
            for c in cases:
                for streaming in [False,True]:result['requests'].append(completion(c,streaming));save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                result['concurrent_requests']=list(pool.map(completion,cases))
            payload={'model':str(MODEL),'messages':[{'role':'user','content':'What is 17 + 25?'}],
                'temperature':0,'max_tokens':16,'chat_template_kwargs':{'enable_thinking':True}}
            chat=call('/v1/chat/completions',payload);stream=call('/v1/chat/completions',{**payload,'stream':True})
            message=chat['choices'][0]['message'];wanted={k:message.get(k) or '' for k in ['content','reasoning_content']}
            result['chat']={'message':wanted,'stream_message':stream['chat'],'stream_done':stream['done'],
                'stream_matches':wanted==stream['chat'],'usage':chat.get('usage')}
            if not any(wanted.values()):raise RuntimeError('Empty chat generation')
            result['status']='http_smoke_completed'
            result['review_required']=any(not (x['text_exact'] and x['usage_matches'] and (not x['stream'] or x['done'])) for x in result['requests']+result['concurrent_requests']) or not result['chat']['stream_matches'] or not stream['done']
            save()
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    print('RESULT http_smoke_completed output',out);return 0
if __name__=='__main__':raise SystemExit(main())
