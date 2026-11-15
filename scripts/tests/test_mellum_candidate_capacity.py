import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_candidate_capacity as study
from run_mellum_batch_campaign import write


def test_c_selection_includes_length_extremes_without_duplicate_ids():
    rows=[{'id':str(i),'tokens':200-i} for i in range(200)]
    selected=study.select_c(rows)
    assert len(selected)==len({r['id'] for r in selected})==20
    assert selected[0]['tokens']==1 and selected[-1]['tokens']==200


def test_custom_port_preserves_load_and_sampling_settings(tmp_path):
    cmd=study.base.command('C',tmp_path,20,'/data/test',8151)
    assert cmd[cmd.index('--url')+1]=='http://127.0.0.1:8151/v1/chat/completions'
    assert cmd[cmd.index('--rate')+1]=='0.3' and cmd[cmd.index('--max-tokens')+1]=='256'


def test_capacity_failure_stops_later_jobs_and_never_invents_official_ratio(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'current_matches',lambda *a:None)
    m={'claim':'diagnostic','limitations':[],'gpu':1,'profile':'fused_moe','server_port':8151,'engine_port':8726,
       'source_sha256':{},'additional_source_sha256':{},'stages':{'smoke':[],'A_small':[],'C_small':[]}}
    calls=[]
    def fake(name,cmd,log,timeout):
        calls.append(name);plan=json.loads((log.parent/'plan.json').read_text())
        assert plan['profile']=='fused_moe' and plan['gpu']==1 and plan['engine_port']==8726
        write(log.parent/'results.json',{'status':'complete','cases':{'x':{'metrics':{'complete_without_failures':name=='smoke'}}}})
    result=study.execute(tmp_path,m,2000,fake)
    assert calls==['smoke','A_small']
    assert result['status']=='service_or_capacity_gate_failed' and 'comparisons' not in result
