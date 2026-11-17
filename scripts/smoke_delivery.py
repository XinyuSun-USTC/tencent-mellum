"""Owned HTTP/SSE smoke using exact cumulative delivery row options."""
import argparse,json,os,signal,socket,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,build_opener,ProxyHandler
PROJECT=Path(__file__).resolve().parents[1]
ROWS=('step0','step01','step012','step0123')

def ports(mode,gpu):return 18751+gpu*100+ROWS.index(mode),19426+gpu*100+ROWS.index(mode)


def parse_sse(lines):
    done=False;usage=None;text=''
    for line in lines:
        if not line.startswith('data:'):continue
        payload=line[5:].strip()
        if payload=='[DONE]':done=True;continue
        d=json.loads(payload)
        if d.get('usage') is not None:usage=d['usage']
        for c in d.get('choices',[]):
            delta=c.get('delta',{});text+=(delta.get('content') or '')+(delta.get('reasoning_content') or '')+(c.get('text') or '')
    return {'done':done,'usage':usage,'text':text}


def run(mode,gpu,out,timeout,route=None):
    out.mkdir(parents=True);port,engine=ports(mode,gpu)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Refuse unowned occupied smoke port')
    command=[str(PROJECT/'start_server.sh'),'--row',mode,'--gpu',str(gpu),'--port',str(port),
        '--engine-port',str(engine),'--output',str(out/'server')]
    if route is not None:command+=['--mode',route]
    result={'route':route,'mode':mode,'status':'started','command':command,'scope':'Submission-entrypoint load/HTTP/SSE smoke only; not correctness or ABC'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');proc=None
    opener=build_opener(ProxyHandler({}));url=f'http://127.0.0.1:{port}'
    def call(path,payload=None):
        req=Request(url+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=60) as response:
            return parse_sse([line.decode().strip() for line in response]) if payload and payload.get('stream') else json.loads(response.read())
    try:
        with (out/'server.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            end=time.monotonic()+timeout;beat=time.monotonic()
            while True:
                if proc.poll() is not None:raise RuntimeError('Submission server exited; inspect server.log')
                try:call('/health');break
                except Exception:
                    if time.monotonic()>end:raise TimeoutError('Submission server startup timed out')
                    if time.monotonic()-beat>15:print('WAIT',mode,'server log',out/'server.log',flush=True);beat=time.monotonic()
                    time.sleep(.5)
            models=call('/v1/models');model=models['data'][0]['id']
            payload={'model':model,'messages':[{'role':'user','content':'Write a Python function add(a, b) that returns their sum.'}],
                     'temperature':0,'max_tokens':8,'ignore_eos':True}
            raw=call('/v1/chat/completions',payload)
            message=raw['choices'][0]['message'];text=(message.get('content') or '')+(message.get('reasoning_content') or '')
            if not text or raw.get('usage',{}).get('completion_tokens')!=8:raise ValueError('Non-stream response/usage invalid')
            stream=call('/v1/chat/completions',{**payload,'stream':True,'stream_options':{'include_usage':True}})
            if not stream['done'] or not stream['text'] or not stream['usage'] or stream['usage'].get('completion_tokens')!=8:raise ValueError('Streaming response/usage invalid')
            audit=json.loads((out/'server/loading_audit.json').read_text())
            if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise ValueError('Checkpoint loading audit invalid')
            result.update(status='smoke_passed',models=models,nonstream=raw,stream=stream,loading_audit=audit)
    except Exception:result.update(status='failed',traceback=traceback.format_exc())
    finally:
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            if proc.poll() is None:proc.wait(timeout=5)
            released=False;deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                with socket.socket() as check:
                    check.settimeout(.2)
                    if check.connect_ex(('127.0.0.1',port))!=0:released=True;break
                time.sleep(.1)
            result['http_port_released']=released
            if not released:result.update(status='failed',cleanup_error='Owned HTTP port remains bound after process-group cleanup')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--modes',nargs='+',choices=['step0','step01','step012','step0123'],default=['step0123'])
    p.add_argument('--startup-timeout',type=int,default=240);p.add_argument('--output',type=Path);a=p.parse_args()
    out=a.output or PROJECT/'results/submission_smoke'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    results={}
    for mode in a.modes:
        print('STAGE submission_smoke',mode,'log',out/mode/'server.log',flush=True)
        try:results[mode]=run(mode,a.gpu,out/mode,a.startup_timeout)
        except Exception:
            results[mode]={'status':'failed','traceback':traceback.format_exc()}
            (out/mode).mkdir(exist_ok=True)
            (out/mode/'result.json').write_text(json.dumps(results[mode],indent=2)+'\n')
        print(mode,results[mode]['status'],flush=True)
        (out/'summary.json').write_text(json.dumps({'status':'running','results':results},indent=2)+'\n')
    d={'status':'smoke_passed' if all(r['status']=='smoke_passed' for r in results.values()) else 'smoke_failed','results':results,
       'scope':'Actual delivery launcher, Exact row context/slots, 8 generated tokens; not complete acceptance'}
    (out/'summary.json').write_text(json.dumps(d,indent=2)+'\n');print('RESULT',d['status'],'output',out)
    return 0 if d['status']=='smoke_passed' else 1
if __name__=='__main__':raise SystemExit(main())
