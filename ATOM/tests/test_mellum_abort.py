import ast,asyncio,json,os,queue,time
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus
from conftest import MockConfig
ROOT=Path(__file__).resolve().parents[1]


def test_abort_waiting_and_running_is_idempotent_and_frees_blocks(seq_factory):
    sched=Scheduler(MockConfig(num_kvcache_blocks=100,max_num_seqs=4,max_num_batched_tokens=100))
    active=seq_factory([1,2,3,4]);queued=seq_factory([5,6,7,8]);sched.add(active);sched.schedule();sched.add(queued)
    active.is_partial_prefill=True;sched._partial_prefill_count=1
    free_before=len(sched.block_manager.free_block_ids_set)
    assert set(sched.abort_requests([active.id,queued.id]))=={active.id,queued.id}
    assert not sched.running and not sched.waiting and sched._partial_prefill_count==0
    assert len(sched.block_manager.free_block_ids_set)>free_before
    assert active.is_finished and queued.is_finished
    assert sched.abort_requests([active.id,queued.id])==[]


def test_engine_queue_barrier_precedes_actual_running_kv_release(seq_factory):
    source=ROOT/'atom/model_engine/engine_core.py';tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineCore')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='pull_and_process_input_queue')
    env={'SequenceStatus':SequenceStatus,'EngineCoreRequestType':NS(ABORT='abort'),'logger':NS(debug=lambda *x:None),'time':time}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
    sched=Scheduler(MockConfig(num_kvcache_blocks=100));seq=seq_factory([1,2,3]);sched.add(seq);sched.schedule()
    order=[];original=sched.abort_requests
    def abort(ids):order.append('free');return original(ids)
    sched.abort_requests=abort
    pending=queue.Queue();pending.put(('abort',[seq.id]))
    core=NS(scheduler=sched,input_queue=pending,_mellum_abort_enabled=True,_mellum_phase_audit=None,
            runner_mgr=NS(call_func=lambda *x,**k:order.append('barrier')))
    assert env['pull_and_process_input_queue'](core) is False
    assert order==['barrier','free'] and not sched.running


def load_stream():
    source=ROOT/'atom/entrypoints/openai/serving_chat.py';tree=ast.parse(source.read_text())
    names={'_prompt_token_count','create_chat_chunk','stream_chat_response'}
    functions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    class Parser:
        def process(self,*args):return []
        def flush(self):return []
    env={'os':os,'asyncio':asyncio,'json':json,'time':time,'ReasoningFilter':Parser,'ToolCallStreamParser':Parser,
         'CHAT_COMPLETION_CHUNK_OBJECT':'chat.completion.chunk','STREAM_DONE_MESSAGE':'data: [DONE]\n\n'}
    exec(compile('from __future__ import annotations\n'+ast.unparse(ast.Module(body=functions,type_ignores=[])),str(source),'exec'),env)
    return env['stream_chat_response']


@pytest.mark.parametrize('mode',['normal','close_after_role','cancel_while_waiting'])
def test_actual_chat_generator_finally_requests_abort_only_for_early_exit(monkeypatch,mode):
    monkeypatch.setenv('MELLUM_ABORT_ON_DISCONNECT','1');stream=load_stream()
    async def run():
        calls=[];q=asyncio.Queue()
        generator=stream('req','model',[1],q,7,None,lambda *args,**kw:calls.append(kw))
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
        return calls
    assert asyncio.run(run())==[{'abort':mode!='normal'}]
