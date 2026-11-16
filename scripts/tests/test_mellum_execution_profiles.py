import json,os,subprocess,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import PROFILES,profile_settings


@pytest.mark.parametrize('name',PROFILES)
def test_profiles_preserve_required_model_semantics_and_do_not_split_fused_experts(name):
    env=profile_settings(name);cfg=json.loads(env['MELLUM_BATCH_CAMPAIGN_CONFIG'])
    if env['MELLUM_MOE_BACKEND']!='torch':assert cfg['groups']==('' if name in ('fused_moe_native_batched','fused_moe_native_batched_direct_kv') or name.startswith('fast_') else 'L') and 'moe_optimization' not in cfg
    else:assert cfg['groups']=='LE' and cfg['moe_optimization']=='shared_routing'
    assert env['MELLUM_REFERENCE_KV_ROPE']=='1' and env['MELLUM_PREFILL_BACKEND']=='sdpa_cpu_rope_diagnostic'
    if env['MELLUM_ATTENTION_EXECUTION']=='native':
        assert env['MELLUM_SDPA_DECODE']=='0' and env['MELLUM_REQUEST_BATCHING']=='0'
        assert env['MELLUM_SWA_PREFILL_OPT']=='0'
    else:assert env['MELLUM_SDPA_DECODE']==env['MELLUM_REQUEST_BATCHING']=='1'


def test_launcher_profile_prepare_records_exact_environment_and_no_GPU_import(tmp_path):
    script=Path(__file__).resolve().parents[1]/'serve_mellum.py'
    p=subprocess.run([sys.executable,str(script),'--profile','fused_moe_swa_ck','--batching','--max-num-seqs','32',
        '--prepare-only','--output',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    launch=json.loads((tmp_path/'launch.json').read_text())
    assert launch['profile']=='fused_moe_swa_ck'
    assert launch['settings']['MELLUM_MOE_BACKEND']=='fused_torch_route'
    assert launch['settings']['MELLUM_SWA_PREFILL_OPT']=='1'
    assert launch['settings']['PYTHONPATH'].endswith('/ATOM')
    assert '--enforce-eager' in launch['command'] and '--no-enable_prefix_caching' in launch['command']


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):profile_settings('unknown')


def test_prefix_structural_estimate_excludes_unshared_tail_and_block_remainder():
    from analyze_mellum_prefix_potential import analyze
    value=analyze({'x':[[1,2,3,4,5],[1,2,3,4,6]],'y':[[9,8,7]]},block_size=2)
    assert value['total_input_tokens']==13
    assert value['group_common_prefix_reuse_tokens']==4
    assert value['groups'][0]['common_prefix_tokens']==4
    assert value['groups'][1]['ideal_reusable_tokens_after_first']==0


def test_launcher_disables_thinking_only_when_explicit_and_records_native_audit(tmp_path):
    script=Path(__file__).resolve().parents[1]/'serve_mellum.py'
    p=subprocess.run([sys.executable,str(script),'--profile','fused_moe','--batching','--batch-audit',
        '--disable-thinking','--prepare-only','--output',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    data=json.loads((tmp_path/'launch.json').read_text());cmd=data['command']
    assert json.loads(cmd[cmd.index('--default-chat-template-kwargs')+1])=={'enable_thinking':False}
    assert data['settings']['MELLUM_NATIVE_AUDIT_PATH']==str(tmp_path/'native_calls.jsonl')


def test_batched_projection_candidate_releases_only_requestwise_linear_controls():
    a=profile_settings('fused_moe_native_attention');b=profile_settings('fused_moe_native_batched')
    key='MELLUM_BATCH_CAMPAIGN_CONFIG'
    assert {k:v for k,v in a.items() if k!=key}=={k:v for k,v in b.items() if k!=key}
    old=json.loads(a[key]);new=json.loads(b[key]);assert old['groups']=='L' and new['groups']==''
    assert old['phases']==new['phases']==['prefill','decode']


def test_direct_kv_profile_only_adds_the_direct_write_switch():
    a=profile_settings('fused_moe_native_batched');b=profile_settings('fused_moe_native_batched_direct_kv')
    assert a['MELLUM_DIRECT_KV_WRITE']=='0' and b['MELLUM_DIRECT_KV_WRITE']=='1'
    assert {k:v for k,v in a.items() if k!='MELLUM_DIRECT_KV_WRITE'}=={k:v for k,v in b.items() if k!='MELLUM_DIRECT_KV_WRITE'}
