import json,os,subprocess,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import PROFILES,profile_settings


@pytest.mark.parametrize('name',PROFILES)
def test_profiles_preserve_required_model_semantics_and_do_not_split_fused_experts(name):
    env=profile_settings(name);cfg=json.loads(env['MELLUM_BATCH_CAMPAIGN_CONFIG'])
    if env['MELLUM_MOE_BACKEND']!='torch':assert cfg['groups']==('' if name=='fused_moe_native_batched' else 'L') and 'moe_optimization' not in cfg
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


def test_prefix_service_flag_enables_engine_and_records_compatibility(tmp_path):
    script=Path(__file__).resolve().parents[1]/'serve_mellum.py'
    p=subprocess.run([sys.executable,str(script),'--profile','fused_moe','--batching',
        '--prefix-caching','--gpu','0','--server-port','8251','--engine-port','8926',
        '--prepare-only','--output',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    data=json.loads((tmp_path/'launch.json').read_text())
    assert '--enable_prefix_caching' in data['command'] and '--no-enable_prefix_caching' not in data['command']
    assert data['settings']['MELLUM_PREFIX_COMPAT']=='1'
    assert data['settings']['MELLUM_PREFIX_AUDIT_PATH']==str(tmp_path/'prefix.jsonl')


def test_prefix_service_rejects_unvalidated_native_attention_combination(tmp_path):
    script=Path(__file__).resolve().parents[1]/'serve_mellum.py'
    p=subprocess.run([sys.executable,str(script),'--profile','native_attention','--batching',
        '--prefix-caching','--prepare-only','--output',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode!=0 and 'currently requires' in p.stderr


def test_native_prefix_service_disables_old_bridge_and_enables_cache(tmp_path):
    script=Path(__file__).resolve().parents[1]/'serve_mellum.py'
    p=subprocess.run([sys.executable,str(script),'--profile','fused_moe_native_attention','--batching',
        '--prefix-caching','--gpu','0','--server-port','8251','--engine-port','8926',
        '--prepare-only','--output',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    d=json.loads((tmp_path/'launch.json').read_text())
    assert '--enable_prefix_caching' in d['command']
    assert d['settings']['MELLUM_PREFIX_COMPAT']=='1'
    assert d['settings']['MELLUM_REQUEST_BATCHING']==d['settings']['MELLUM_SDPA_DECODE']=='0'


def test_batched_prefix_launcher_keeps_native_attention_and_releases_linear_control(tmp_path):
    script=Path(__file__).resolve().parents[1]/'serve_mellum.py'
    p=subprocess.run([sys.executable,str(script),'--profile','fused_moe_native_batched','--batching',
        '--prefix-caching','--gpu','0','--prepare-only','--output',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    d=json.loads((tmp_path/'launch.json').read_text());e=d['settings']
    assert e['MELLUM_PREFIX_COMPAT']=='1' and e['MELLUM_ATTENTION_EXECUTION']=='native'
    assert json.loads(e['MELLUM_BATCH_CAMPAIGN_CONFIG'])['groups']==''
    assert '--enable_prefix_caching' in d['command']
