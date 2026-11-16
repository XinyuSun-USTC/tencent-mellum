"""CPU coroutine-close reproduction using actual serving function bodies.

Text parsers are inert stubs: this tests callback lifetime, not content parsing,
ASGI transport behavior, or actual engine cancellation. No GPU import occurs.
"""
import ast,asyncio,hashlib,json,time,argparse
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def load_chat():
    source=ROOT/'ATOM/atom/entrypoints/openai/serving_chat.py';tree=ast.parse(source.read_text())
    names={'_prompt_token_count','create_chat_chunk','stream_chat_response'}
    body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    class Parser:
        def process(self,*a):return []
        def flush(self):return []
    env={'asyncio':asyncio,'json':json,'time':time,'ReasoningFilter':Parser,'ToolCallStreamParser':Parser,
         'CHAT_COMPLETION_CHUNK_OBJECT':'chat.completion.chunk','STREAM_DONE_MESSAGE':'data: [DONE]\n\n'}
    exec(compile('from __future__ import annotations\n'+ast.unparse(ast.Module(body=body,type_ignores=[])),str(source),'exec'),env)
    return env['stream_chat_response'],source


async def reproduce():
    stream,source=load_chat();results={}
    for mode in ['normal','close_after_role','cancel_while_waiting']:
        calls=[];q=asyncio.Queue()
        generator=stream('test','model',[1],q,123,None,lambda *a:calls.append(a))
        await anext(generator)
        if mode=='normal':
            await q.put({'text':'','token_ids':[],'finished':True})
            async for _ in generator:pass
        elif mode=='close_after_role':await generator.aclose()
        else:
            task=asyncio.create_task(anext(generator));await asyncio.sleep(0);task.cancel()
            try:await task
            except asyncio.CancelledError:pass
            await generator.aclose()
        results[mode]={'cleanup_calls':len(calls),'cleanup_called':bool(calls)}
    return {'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':results,
        'status':'cleanup_gap_reproduced' if results['normal']['cleanup_called'] and not all(r['cleanup_called'] for r in results.values()) else 'review_required',
        'scope':'Actual coroutine body; inert content parsers. Does not measure the live C failure contribution or provide engine abort.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    out=a.output or ROOT/'results/stream_cleanup_cpu'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    result=asyncio.run(reproduce());(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('RESULT',result['status'],'output',out)
if __name__=='__main__':main()
