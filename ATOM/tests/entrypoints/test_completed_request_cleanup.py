"""Exercise actual async generator bodies without importing GPU dependencies."""
import ast,asyncio,time,types
from pathlib import Path
import pytest


def load_functions():
    source=Path(__file__).parents[2]/'atom/entrypoints/openai/api_server.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name in
           ['generate_async','generate_async_multimodal','generate_async_fanout']]
    def test_loop():
        real=asyncio.get_running_loop()
        async def inline_executor(executor,fn):return fn()
        return types.SimpleNamespace(run_in_executor=inline_executor,call_soon_threadsafe=real.call_soon_threadsafe)
    ns={'asyncio':types.SimpleNamespace(Queue=asyncio.Queue,get_running_loop=test_loop),'time':time}
    exec(compile('from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),ns)
    return ns


@pytest.mark.parametrize('mode',['text','multimodal','fanout'])
def test_completed_http_requests_release_only_their_frontend_entries(mode):
    ns=load_functions();requests={99:'other active request'}
    class IO:
        def preprocess(self,*args,stream_callback=None,**kwargs):
            seq=types.SimpleNamespace(id=1,num_prompt_tokens=6,callback=stream_callback)
            requests[seq.id]=seq;return seq
        def preprocess_fanout(self,*args,stream_callbacks=None,**kwargs):
            seqs=[]
            for i,cb in enumerate(stream_callbacks,1):
                seq=types.SimpleNamespace(id=i,num_prompt_tokens=6,callback=cb);requests[i]=seq;seqs.append(seq)
            return seqs
    io=IO();io.requests=requests
    def add(seqs):
        for seq in seqs:seq.callback(types.SimpleNamespace(output_tokens=[7],finished=True,finish_reason='length'))
    ns['engine']=types.SimpleNamespace(io_processor=io,core_mgr=types.SimpleNamespace(add_request=add))
    ns['tokenizer']=types.SimpleNamespace(decode=lambda *a,**k:'answer')
    async def run():
        if mode=='fanout':
            result=await ns['generate_async_fanout']('prompt',types.SimpleNamespace(n=2),'r')
            assert len(result)==2
        else:
            gen=ns['generate_async']('prompt',object(),'r') if mode=='text' else ns['generate_async_multimodal']([1],{},object(),'r')
            assert len([x async for x in gen])==1
    asyncio.run(run());assert requests=={99:'other active request'}
