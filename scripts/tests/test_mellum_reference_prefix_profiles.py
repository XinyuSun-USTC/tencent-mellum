import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import profile_settings


def test_decode_quota_changes_only_service_count():
    for size in [1024,2048]:
        base=profile_settings(f'r_chunk{size}_d4');value=profile_settings(f'r_chunk{size}_d1')
        assert value=={**base,'MELLUM_C_DECODE_STEPS':'1'}


def test_prefix_changes_only_prefix_bridge_and_cache_enablement():
    for candidate,parent in [('r_prefix_only','r_expert_decode4_kv_once'),('r_prefix_chunk1024_d1','r_chunk1024_d1'),('r_prefix_chunk2048_d1','r_chunk2048_d1')]:
        base=profile_settings(parent);value=profile_settings(candidate)
        assert value=={**base,'MELLUM_REFERENCE_PREFIX':'1','MELLUM_PREFIX_ENABLED':'1'}
        assert value['MELLUM_DIRECT_KV_WRITE']=='1' and value['MELLUM_SDPA_DECODE']=='1'
    assert profile_settings('r_prefix_only').get('MELLUM_REFERENCE_CHUNKED','0')=='0'
    assert profile_settings('r_prefix_only').get('MELLUM_C_DECODE_STEPS','0')=='0'


def test_prefix_only_launcher_enables_cache_without_chunking(tmp_path):
    import json,subprocess
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,str(root/'scripts/serve_mellum.py'),'--batching','--profile','r_prefix_only','--max-num-seqs','32','--prepare-only','--output',str(tmp_path)],check=True,capture_output=True)
    value=json.loads((tmp_path/'launch.json').read_text())
    assert '--enable_prefix_caching' in value['command'] and '--no-enable_chunked_prefill' in value['command']
    assert value['settings']['MELLUM_REFERENCE_PREFIX']=='1'
    assert value['settings']['MELLUM_ATTENTION_EXECUTION']=='reference'


def test_combination_launcher_keeps_reference_math_and_bounded_chunk(tmp_path):
    import json,subprocess
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,str(root/'scripts/serve_mellum.py'),'--batching','--profile','r_prefix_chunk1024_d1','--max-num-seqs','32','--prepare-only','--output',str(tmp_path)],check=True,capture_output=True)
    value=json.loads((tmp_path/'launch.json').read_text())
    assert '--enable_prefix_caching' in value['command'] and '--enable_chunked_prefill' in value['command']
    assert value['command'][value['command'].index('--max-num-batched-tokens')+1]=='1024'
    assert value['settings']['MELLUM_C_DECODE_STEPS']=='1'
    assert value['settings']['MELLUM_SDPA_DECODE']=='1' and value['settings']['MELLUM_QKV_BACKEND']=='separate'
