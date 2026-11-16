import json
from types import SimpleNamespace as NS
from atom.model_engine.mellum_service_audit import record_step,summarize_steps


def test_phase_summary_excludes_decode_interleaved_before_last_prefill():
    def row(phase,a,b,count):
        return {'kind':'engine_step','phase':phase,'start_ns':a,'end_ns':b,'request_ids':[1,2],'decode_tokens':count}
    rows=[row('prefill',0,10,0),row('decode',10,20,2),row('prefill',20,30,0),row('decode',30,50,2)]
    result=summarize_steps(rows)
    assert result['decode_steps']==2 and result['post_last_prefill_decode_steps']==1
    assert result['decode_scheduled_token_work']==2
    assert result['prefill_host_step_seconds']==20/1e9


def test_actual_receipt_has_phase_ids_and_finished_state(tmp_path):
    path=tmp_path/'steps.jsonl';batch=NS(total_tokens_num_prefill=0,total_tokens_num_decode=2,req_ids=[8,9],num_scheduled_tokens=[1,1],context_lens=[8001,16001])
    sched=NS(waiting=[1,2],running=[3])
    record_step(path,batch,sched,1,2,[NS(id=8)])
    row=json.loads(path.read_text())
    assert row['phase']=='decode' and row['request_ids']==[8,9]
    assert row['finished_request_ids']==[8] and row['waiting']==2
