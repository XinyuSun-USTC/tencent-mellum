import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import CORE_PROFILES,profile_settings
from run_mellum_core_diagnosis import select


def test_two_by_two_keeps_inputs_norm_routing_and_projection_fixed():
    for name in CORE_PROFILES:
        s=profile_settings(name);native=name[5]=='n';experts=name[6]=='n'
        assert s['MELLUM_NORM_BACKEND']=='reference' and s['MELLUM_QKV_BACKEND']=='separate'
        assert s['MELLUM_FUSED_QK_ROPE']==s['MELLUM_DECODE_GRAPH']==s['MELLUM_PREFIX_ENABLED']==s['MELLUM_NATIVE_ROUTING']=='0'
        assert s['MELLUM_MOE_BACKEND']==('fused_torch_route' if experts else 'torch')
        assert s['MELLUM_ATTENTION_EXECUTION']==('native' if native else 'reference')
        assert s['MELLUM_REQUEST_BATCHING']==('0' if native else '1')
        assert s['MELLUM_SDPA_DECODE']==('0' if native else '1')


def test_subset_selection_depends_on_source_and_length_not_expected_outputs():
    rows=[{'name':str(i),'prompt_source':'HumanEval' if i<164 else 'supplied_scenario_a','input_ids':[[1]*(i+2)],'generated_ids':[[7]]} for i in range(200)]
    chosen=select(rows);names=[r['name'] for r in chosen]
    for r in rows:r['generated_ids']=[[999]]
    assert len(chosen)==20 and [r['name'] for r in select(rows)]==names
    assert sum(r['prompt_source']=='HumanEval' for r in chosen)==12


def test_attention_phase_controls_change_only_dispatch_mode():
    base=profile_settings('core_nt')
    for name,mode in [('phase_prefill','native_prefill'),('phase_decode','native_decode')]:
        s=profile_settings(name)
        assert s=={**base,'MELLUM_ATTENTION_EXECUTION':mode}
        assert s['MELLUM_MOE_BACKEND']=='torch' and s['MELLUM_FUSED_QK_ROPE']=='0'


def test_expert_phase_profiles_only_change_expert_block():
    base=profile_settings('core_tn')
    for name,phase in [('expert_prefill','prefill'),('expert_decode','decode')]:
        assert profile_settings(name)=={**base,'MELLUM_MOE_BACKEND':'phase_hybrid','MELLUM_NATIVE_EXPERT_PHASE':phase}
