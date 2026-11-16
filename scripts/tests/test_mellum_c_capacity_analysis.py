import sys,json,sqlite3
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_mellum_c_capacity import database,batches,analyze


def test_failed_partial_streams_remain_in_denominator(tmp_path):
    p=tmp_path/'data.db';c=sqlite3.connect(p)
    c.execute('CREATE TABLE result(success,start_time,completed_time,inter_token_latencies)')
    c.executemany('INSERT INTO result VALUES(?,?,?,?)',[(1,0,20,'[0.1,0.2]'),(0,1,181,'[0.4,0.6]'),(0,2,182,'[]')]);c.commit();c.close()
    d=database(p)
    assert (d['requests'],d['successes'],d['failures'])==(3,1,2)
    assert d['failed_with_intervals']==1 and d['failed_without_intervals']==1
    assert d['failed_elapsed_min_s']==180


def test_batch_order_does_not_invent_timestamps(tmp_path):
    p=tmp_path/'batches.jsonl'
    rows=[{'phase':'prefill','batch_size':1,'lengths':[512],'cached_tokens':[0]},
          {'phase':'prefill','batch_size':1,'lengths':[512],'cached_tokens':[512]},
          {'phase':'decode','batch_size':2,'lengths':[11,22]}]
    p.write_text(''.join(json.dumps(x)+'\n' for x in rows));d=batches(p)
    assert d['cached_prefill_batches']==1 and d['max_consecutive_prefill_batches']==2
    assert d['max_total_prefill_tokens']==512 and 'no timestamps' in d['note']


def test_running_evidence_is_rejected(tmp_path):
    (tmp_path/'summary.json').write_text('{"status":"started"}')
    with pytest.raises(ValueError,match='completed run'):analyze(tmp_path)


def test_abort_receipts_detect_post_removal_scheduling(tmp_path):
    from analyze_mellum_c_capacity import lifecycle
    p=tmp_path/'engine_steps.jsonl'
    rows=[{'kind':'engine_step','request_ids':[1,2],'start_ns':1},
          {'kind':'abort','removed':[2],'kv_free_before':10,'kv_free_after':30},
          {'kind':'engine_step','request_ids':[1],'start_ns':3}]
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    d=lifecycle(p)
    assert d['lifecycle_receipts_consistent'] and d['net_blocks_released_at_abort']==20
    rows.append({'kind':'engine_step','request_ids':[2],'start_ns':4})
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert not lifecycle(p)['lifecycle_receipts_consistent']
