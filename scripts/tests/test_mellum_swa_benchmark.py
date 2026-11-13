import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_mellum_swa import summarize


def test_summary_retains_samples_and_distinguishes_token_drift(tmp_path):
    for stage in ['a0','b0','b1','a1']:
        dest=tmp_path/stage;dest.mkdir()
        candidate=stage.startswith('b')
        row={'name':'ctx_4096','output_tokens':2,'token_ids':[1,3 if candidate else 2],
             'request_ttft_seconds':1 if candidate else 2,'request_tpot_seconds':.1,'wall_seconds':1.1 if candidate else 2.1}
        (dest/'worker_results.json').write_text(json.dumps({'rows':[row,row]}))
    result=summarize(tmp_path)['rows'][0]
    assert result['greedy_match_fraction']==.5
    assert result['reference_repeats_exact'] and result['candidate_repeats_exact']
    assert result['timings']['request_ttft_seconds']['speedup']==2
    assert len(result['timings']['wall_seconds']['candidate_samples'])==4
