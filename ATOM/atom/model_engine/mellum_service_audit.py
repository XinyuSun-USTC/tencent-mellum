"""Default-off host-side EngineCore receipts; no CUDA synchronize or model edits."""
import json,time
from pathlib import Path


def record_step(path,batch,scheduler,start_ns,forward_done_ns,finished):
    end=time.monotonic_ns()
    row={'kind':'engine_step','start_ns':start_ns,'forward_done_ns':forward_done_ns,'end_ns':end,
         'phase':'prefill' if batch.total_tokens_num_prefill else 'decode',
         'request_ids':[int(x) for x in batch.req_ids],
         'scheduled_tokens':[int(x) for x in batch.num_scheduled_tokens],
         'context_lengths':[int(x) for x in batch.context_lens],
         'prefill_tokens':int(batch.total_tokens_num_prefill),'decode_tokens':int(batch.total_tokens_num_decode),
         'waiting':len(scheduler.waiting),'running':len(scheduler.running),
         'finished_request_ids':[int(x.id) for x in finished]}
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('a') as stream:stream.write(json.dumps(row)+'\n')


def summarize_steps(rows):
    rows=[r for r in rows if r.get('kind')=='engine_step']
    if not rows:return {'steps':0}
    prefills=[r for r in rows if r['phase']=='prefill'];last_prefill=max((r['end_ns'] for r in prefills),default=0)
    steady=[r for r in rows if r['phase']=='decode' and r['start_ns']>=last_prefill]
    elapsed=sum((r['end_ns']-r['start_ns']) for r in steady)/1e9
    span=(steady[-1]['end_ns']-steady[0]['start_ns'])/1e9 if steady else 0
    work=sum(r['decode_tokens'] for r in steady)
    return {'steps':len(rows),'prefill_steps':len(prefills),'decode_steps':sum(r['phase']=='decode' for r in rows),
        'post_last_prefill_decode_steps':len(steady),'decode_scheduled_token_work':work,
        'steady_host_step_seconds':elapsed,'steady_host_span_seconds':span,
        'scheduled_token_work_per_host_span_s':work/span if span else None,
        'max_decode_batch':max((len(r['request_ids']) for r in steady),default=0),
        'prefill_host_step_seconds':sum(r['end_ns']-r['start_ns'] for r in prefills)/1e9,
        'note':'Host call/postprocess timing, not GPU kernel timing. Scheduled decode work may include deferred extra forwards; not delivered-output throughput. Instrumentation can perturb timing.'}
