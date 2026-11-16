import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_mellum_c_prefix_log import parse


def test_real_prefix_hit_is_not_sum_of_chunk_progress():
    log='''[atom 01:00:00] Request 0 arrived, input tokens: 9, pending requests: 1
[atom 01:00:01] Request 1 arrived, input tokens: 8000, pending requests: 1
[atom 01:00:02] Scheduled prefill batch: 1 reqs, 1024 new tokens (cached: [3808], new: [1024]), req_ids: (1,)
[atom 01:00:03] Scheduled prefill batch: 1 reqs, 1024 new tokens (cached: [4832], new: [1024]), req_ids: (1,)
[atom 01:00:04] Request 2 arrived, input tokens: 9000, pending requests: 2
'''
    r=parse(log)
    assert r['arrivals']==2 and r['observed_admitted']==1 and r['not_observed_admitted']==1
    assert r['first_admission_cached_tokens']==3808
    assert r['requests'][0]['subsequent_chunk_cached_sum_not_prefix_hits']==4832
    assert r['admission_wait_p50_s']==1


def test_chunking_without_shared_prefix_has_zero_external_reuse():
    r=parse('''[atom 23:59:59] Request 1 arrived, input tokens: 8000
[atom 00:00:00] Scheduled prefill batch: 1 reqs, 1024 new tokens (cached: [0], new: [1024]), req_ids: (1,)
[atom 00:00:01] Scheduled prefill batch: 1 reqs, 1024 new tokens (cached: [1024], new: [1024]), req_ids: (1,)
''')
    assert r['first_admission_cached_tokens']==0 and r['admission_wait_max_s']==1


def test_invalid_geometry_and_reordered_timestamps_rejected():
    with pytest.raises(ValueError,match='progress'):
        parse('''[atom 01:00:00] Request 1 arrived, input tokens: 8000
[atom 01:00:01] Scheduled prefill batch: 1 reqs, 1024 new tokens (cached: [7900], new: [1024]), req_ids: (1,)
''')
    with pytest.raises(ValueError,match='Out-of-order'):
        parse('[atom 01:01:00] hello\n[atom 01:00:59] hello')
