import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_fastpaths as study
from run_mellum_batch_campaign import write
from mellum_execution_profiles import profile_settings


def test_single_changes_and_combination_do_not_enable_quant_or_graph():
    for name in ['fast_resident','fast_norm','fast_packed','fast_combined']:
        s=profile_settings(name);assert s['MELLUM_DIRECT_KV_WRITE']=='1' and s['MELLUM_MOE_BACKEND']=='fused_torch_route'
        assert s['MELLUM_NORM_BACKEND']==('aiter' if name in ['fast_norm','fast_combined'] else 'reference')
        assert s['MELLUM_QKV_BACKEND']==('packed' if name in ['fast_packed','fast_combined'] else 'separate')
        assert s['MELLUM_RESIDENT_ROPE']==('1' if name in ['fast_resident','fast_combined'] else '0')


def test_failed_norm_does_not_hide_packed_result_or_run_invalid_combination(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'verify_backend',lambda *a:None);calls=[]
    def fake(out,*a):calls.append(out.name);return {'status':'stage_failed' if out.name=='fast_norm' else 'diagnostic_complete'}
    monkeypatch.setattr(study.capacity,'execute',fake)
    plan={'track':'kernels','scope':'test','source_sha256':{}}
    jobs={n:{} for n in study.TRACKS['kernels']}
    result=study.execute(tmp_path,plan,jobs,2000)
    assert calls==[study.BASE,'fast_norm','fast_packed'] and result['status']=='complete_with_failures'
    assert result['jobs']['fast_combined']['status']=='skipped_component_failure'
