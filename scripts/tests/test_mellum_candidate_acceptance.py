import sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_mellum_candidate_acceptance import token_comparison,validate_candidate


def record(name,ids,length=2):
    return {'name':name,'input_ids':[[1]*length],'enable_thinking':False,
            'generation_settings':{'max_new_tokens':64,'do_sample':False},'generated_ids':[ids]}


def test_position_fraction_and_long_group_are_separate():
    reference={'records':[record('short',[1,2,3]),record('16k',[4,5],16384)]}
    candidate=copy.deepcopy(reference);candidate['records'][0]['generated_ids']=[[1]]
    d=token_comparison(reference,candidate)
    assert d['all']['matching_positions']==3 and d['all']['compared_positions']==5
    assert d['all']['position_match_fraction']==.6 and d['16k']['position_match_fraction']==1


def test_missing_duplicate_or_reordered_inputs_cannot_pass():
    a={'records':[record('a',[1]),record('b',[2])]}
    for rows in [a['records'][:1],a['records'][::-1],[a['records'][0],a['records'][0]]]:
        with pytest.raises(ValueError):validate_candidate(a,{'records':rows})


def test_invalid_token_budget_is_rejected():
    a={'records':[record('a',[1])]};b=copy.deepcopy(a);b['records'][0]['generated_ids']=[[1]*65]
    with pytest.raises(ValueError):validate_candidate(a,b)


def test_final_profile_runtime_keeps_graph_prefix_and_audits(tmp_path,monkeypatch):
    import run_mellum_candidate_acceptance as c
    monkeypatch.setattr(c.os,'environ',dict(c.os.environ))
    s,o=c.runtime('sprint_graph_qk_router_prefix_metadata',1,tmp_path)
    assert not o['enforce_eager'] and o['enable_prefix_caching'] and not o['enable_chunked_prefill']
    assert s['MELLUM_NATIVE_ROUTING']==s['MELLUM_FUSED_QK_ROPE']==s['MELLUM_PREFILL_METADATA_REUSE']=='1'
    for key in ['MELLUM_NATIVE_AUDIT_PATH','MELLUM_GRAPH_AUDIT_PATH','MELLUM_PREFIX_AUDIT_PATH','MELLUM_METADATA_AUDIT_PATH']:
        assert Path(s[key]).parent==tmp_path
