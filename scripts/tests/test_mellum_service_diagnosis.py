import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_mellum_c_service_diagnosis import delivered_rate


def test_delivery_rate_uses_delivered_tokens_after_cutoff_not_scheduled_work():
    events=[{'time_ns':5,'token_ids':[1,2,3]},{'time_ns':20,'token_ids':[4,5]},{'time_ns':30,'token_ids':[6]}]
    result=delivered_rate(events,10)
    assert result['delivered_tokens']==3 and result['seconds_after_last_prefill']==20/1e9
    assert delivered_rate(events,40)['delivered_tokens_per_second'] is None


def test_phase_audit_is_explicit_default_off_launcher_option(tmp_path):
    import subprocess
    root=Path(__file__).resolve().parents[2]
    for enabled in [False,True]:
        dest=tmp_path/str(enabled)
        cmd=[sys.executable,str(root/'scripts/serve_mellum.py'),'--batching','--profile','r_expert_decode4_kv_once','--prepare-only','--output',str(dest)]
        if enabled:cmd+=['--phase-audit']
        subprocess.run(cmd,check=True,capture_output=True)
        value=json.loads((dest/'launch.json').read_text())
        assert ('MELLUM_ENGINE_PHASE_AUDIT' in value['settings'])==enabled



def test_disconnect_receipt_separates_inflight_from_new_steps():
    from run_mellum_c_service_diagnosis import disconnect_evidence
    rows=[{'start_ns':0,'end_ns':20,'request_ids':[1],'finished_request_ids':[]},
          {'start_ns':21,'end_ns':30,'request_ids':[1],'finished_request_ids':[1]},
          {'start_ns':22,'end_ns':40,'request_ids':[2],'finished_request_ids':[]}]
    result=disconnect_evidence(rows,1,10)
    assert result['engine_steps_started_after_close']==1
    assert result['inflight_steps_spanning_close']==1
    assert result['continued_host_step_seconds']==9/1e9
    assert result['engine_completion_observed']
