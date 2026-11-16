import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import profile_settings


def test_integration_has_only_selected_changes_and_prefix_is_separate():
    base=profile_settings('r_swa_decode4');plain=profile_settings('r_swa4_abort')
    assert plain=={**base,'MELLUM_ABORT_ON_DISCONNECT':'1'}
    prefix=profile_settings('r_swa4_abort_prefix')
    assert prefix=={**plain,'MELLUM_REFERENCE_PREFIX':'1','MELLUM_PREFIX_ENABLED':'1'}
    chunk=profile_settings('r_swa4_abort_prefix_chunk1024_d4')
    assert chunk=={**prefix,'MELLUM_REFERENCE_CHUNKED':'1','MELLUM_CHUNK_BUDGET':'1024','MELLUM_C_DECODE_STEPS':'4'}
