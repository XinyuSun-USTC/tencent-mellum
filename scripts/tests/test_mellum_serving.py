import json,sys,os
import pytest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_runtime_config import apply_serving
from validate_mellum_http import parse_sse
import serve_mellum


@pytest.fixture(autouse=True)
def isolate_process_environment():
    saved=dict(os.environ)
    yield
    os.environ.clear();os.environ.update(saved)


def test_serving_clears_forcing_and_disables_profiling(monkeypatch,tmp_path):
    monkeypatch.setenv('MELLUM_FORCED_DECODE_PLAN','unwanted')
    monkeypatch.setenv('MELLUM_SWA_QKV_CAPTURE_DIR','unwanted')
    settings=apply_serving(0,tmp_path,True,'reference')
    import os
    assert 'MELLUM_FORCED_DECODE_PLAN' not in os.environ and 'MELLUM_SWA_QKV_CAPTURE_DIR' not in os.environ
    assert settings['MELLUM_REQUEST_BATCHING']=='1' and settings['MELLUM_SWA_PREFILL_OPT']=='0'
    assert settings['MELLUM_PROFILE_LABELS']=='0' and 'MELLUM_PROFILE_CONTROL_DIR' not in settings


def test_server_prepare_records_fixed_configuration(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['server','--output',str(tmp_path),'--prepare-only','--batching','--max-num-seqs','4'])
    serve_mellum.main();data=json.loads((tmp_path/'launch.json').read_text())
    assert '--no-enable_prefix_caching' in data['command'] and '--no-enable_chunked_prefill' in data['command']
    assert data['settings']['MELLUM_REQUEST_BATCHING']=='1'
    assert '--torch-profiler-dir' not in data['command']


def test_sse_usage_and_done_without_counting_empty_chunks_as_tokens():
    lines=['data: '+json.dumps({'choices':[{'text':'abc'}]}),'data: '+json.dumps({'choices':[], 'usage':{'completion_tokens':2}}),'data: [DONE]']
    result=parse_sse(lines)
    assert result['text']=='abc' and result['usage']['completion_tokens']==2 and result['done']
