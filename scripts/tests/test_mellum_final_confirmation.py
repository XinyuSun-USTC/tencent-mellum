import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_final_greedy_confirmation as f


def test_prepare_keeps_original200_and_long_not_diagnostic_subset(tmp_path):
    p=f.prepare(tmp_path,'r',1,32)
    assert p['confirmation_batch']==32 and p['confirmation_suites']==['tokens','long']
    assert len(json.loads((tmp_path/'tokens_inputs.json').read_text())['records'])==200
    assert len(json.loads((tmp_path/'long_inputs.json').read_text())['records'])==4
    f.current(p)


def test_confirmation_worker_does_not_generate_humaneval(tmp_path,monkeypatch):
    (tmp_path/'manifest.json').write_text(json.dumps({'confirmation_batch':1,'confirmation_suites':['tokens','long']}))
    monkeypatch.setattr(f,'current',lambda *a:None);calls=[]
    monkeypatch.setattr(f.a,'worker',lambda *a,**kw:calls.append(kw))
    monkeypatch.setattr(sys,'argv',['confirm','--worker',str(tmp_path)])
    f.main();assert calls==[{'suites':['tokens','long'],'batch_size':1}]
