import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import ABLATION_BASE,ABLATIONS,profile_settings
from serve_mellum_step0_snapshot import describe
from prepare_mellum_ablation_matrix import prepare


def test_off_switches_preserve_every_other_setting():
    anchor=profile_settings(ABLATION_BASE)
    for name,overrides in ABLATIONS.items():
        assert profile_settings(name)=={**anchor,**overrides}
    assert profile_settings('ablate_norms')['MELLUM_FUSED_QK_ROPE']=='0'
    assert profile_settings('ablate_prefix')['MELLUM_NATIVE_ROUTING']=='1'


def test_snapshot_launcher_has_no_batch_bridge_or_fast_flags(tmp_path):
    snapshot=Path('/workspace/tencent-mellum-step0-repro')
    d=describe(snapshot,0,'127.0.0.1',8751,9726,tmp_path)
    assert d['snapshot_tag']=='step0-local-validated' and d['settings']['PYTHONPATH']==str(snapshot/'ATOM')
    cmd=d['command'];assert cmd[cmd.index('--max-num-seqs')+1]=='1' and '--enforce-eager' in cmd
    assert d['settings']['MELLUM_MOE_BACKEND']=='torch' and d['settings']['MELLUM_SDPA_DECODE']=='1'
    assert 'MELLUM_BATCH_CAMPAIGN_CONFIG' not in d['settings']
    with pytest.raises(ValueError):describe(Path(__file__).resolve().parents[2],0,'127.0.0.1',8751,9726,tmp_path)


def test_matrix_does_not_mislabel_component_table_as_three_points(tmp_path):
    p=prepare(tmp_path)
    assert len(p['rows'])==8 and p['anchor_acceptance'].startswith('pending')
    assert p['historical_step0']['max_num_seqs']==1
    assert all(r['status']=='not_run' and not r['historical_step0'] for r in p['rows'])
    assert 'torch_experts' in p['excluded_single_switches']
