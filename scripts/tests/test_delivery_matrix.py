import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_delivery_matrix import ROWS,DEFAULT_ROW,slots
from mellum_execution_profiles import profile_settings
from serve_delivery import command
from run_delivery_matrix import render

def test_cumulative_rows_and_default_match():
    assert DEFAULT_ROW=='step0123'
    assert [v['points'] for v in ROWS.values()]==[[],[1],[1,2],[1,2,3]]
    assert slots('delivery_step0')==32
    assert slots('initial_single_reference')==1
    for row,parent in [('step01','r_expert_decode4'),('step012','r_expert_decode4_kv_once'),('step0123','r_swa4_abort_prefix_chunk1024_d4')]:
        assert profile_settings(ROWS[row]['profile'])==profile_settings(parent)
    assert profile_settings('delivery_step0')==profile_settings('r')
    base=json.loads(profile_settings('delivery_step0')['MELLUM_BATCH_CAMPAIGN_CONFIG'])
    one=json.loads(profile_settings('delivery_step01')['MELLUM_BATCH_CAMPAIGN_CONFIG'])
    assert one=={**base,'decode_expert_group_size':4}
    assert profile_settings('initial_single_reference')['MELLUM_BATCH_CAMPAIGN_CONFIG']==''
    c=command(DEFAULT_ROW,0,8000,8026,Path('/tmp/example'),True)
    assert c[c.index('--profile')+1]=='submission_performance_123' and '--prepare-only' in c

def test_failed_c_never_displays_success_only_tail(tmp_path):
    (tmp_path/'performance').mkdir()
    (tmp_path/'performance/summary.json').write_text(json.dumps({'metrics':{'delivery_step0123':{'C':{'requests':200,'successes':43,'complete_without_failures':False,'ttft_p99_s':1,'tpot_p99_s':.1}}}}))
    render(tmp_path,{'rows':['step0123']})
    text=(tmp_path/'matrix.md').read_text()
    assert '43/200' in text and '1000.000' not in text and '待测或未完成' in text


def test_each_smoke_row_has_disjoint_ports_on_both_cards():
    from smoke_delivery import ports
    all_ports=[p for gpu in (0,1) for row in ROWS for p in ports(row,gpu)]
    assert len(set(all_ports))==len(all_ports)


def test_independent_point_switches_match_cumulative_profiles():
    from mellum_delivery_matrix import point_settings
    for row in ROWS.values():assert point_settings(row['points'])==profile_settings(row['profile'])
    only_two=point_settings([2]);base=profile_settings('r')
    assert only_two=={**base,'MELLUM_DIRECT_KV_WRITE':'1'}
    no_two=point_settings([1,3])
    assert no_two['MELLUM_DIRECT_KV_WRITE']=='0'
    assert no_two['MELLUM_REFERENCE_CHUNKED']=='1'
    c=command(DEFAULT_ROW,0,8000,8026,Path('/tmp/example'),True,disabled=[1,3],mode="precision")
    assert c[c.index('--profile')+1]=='submission_precision_2'
