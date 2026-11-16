import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_native_prefix as study
from run_mellum_batch_campaign import write


def test_campaign_does_not_gate_performance_on_free_generation_token_equality(tmp_path,monkeypatch):
    (tmp_path/'probe').mkdir();(tmp_path/'capacity').mkdir();calls=[]
    def fake(stage,cmd,log,timeout):
        calls.append(stage)
        if stage=='kernel':write(log.parent/'results.json',{'status':'passed'});return
        enabled=stage=='on';record={'id':'a','generated_ids':[2 if enabled else 1]}
        write(log.parent/'results.json',{'status':'complete','groups':[{'records':[record]}],
            'observed_cached_tokens':16 if enabled else 0,
            'prefix_events':[{'backend':'aiter_varlen_cached'}] if enabled else []})
        if enabled:(log.parent/'native_calls.jsonl').write_text(json.dumps({'kind':'attention','backend':'aiter_varlen','phase':'prefill'})+'\n')
    def capacity(*args):calls.append('HTTP');return {'status':'diagnostic_complete'}
    monkeypatch.setattr(study,'run_stage',fake);monkeypatch.setattr(study.capacity,'execute',capacity)
    result=study.execute(tmp_path,{'scope':'test','source_sha256':{}},None,{},1200)
    assert result['status']=='complete' and result['probe']['exact_sequences']==0
    assert calls==['kernel','off','on','HTTP']


def test_structural_failure_never_launches_model_or_http(tmp_path,monkeypatch):
    def fake(stage,cmd,log,timeout):
        assert stage=='kernel';write(log.parent/'results.json',{'status':'failed'})
    monkeypatch.setattr(study,'run_stage',fake)
    monkeypatch.setattr(study.capacity,'execute',lambda *a: (_ for _ in ()).throw(AssertionError('HTTP must not start')))
    result=study.execute(tmp_path,{'scope':'test','source_sha256':{}},None,{},1200)
    assert result['status']=='failed' and 'Structural kernel gate' in result['traceback']
