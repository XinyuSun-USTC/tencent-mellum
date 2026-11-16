import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import profile_settings


def test_combination_adds_only_previously_screened_kv_flag_to_group4():
    base=profile_settings('r_expert_decode4');combined=profile_settings('r_expert_decode4_kv_once')
    assert combined=={**base,'MELLUM_DIRECT_KV_WRITE':'1'}
    config=json.loads(combined['MELLUM_BATCH_CAMPAIGN_CONFIG'])
    assert config['decode_expert_group_size']==4 and config['moe_optimization']=='shared_routing'
    assert combined['MELLUM_MOE_BACKEND']=='torch' and combined['MELLUM_ATTENTION_EXECUTION']=='reference'
    assert combined.get('MELLUM_SWA_PREFILL_OPT')=='0'
