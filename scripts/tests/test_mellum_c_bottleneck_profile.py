import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from profile_mellum_c_bottleneck import workload


def test_mixed_injection_waits_for_actual_decode_receipts(tmp_path):
    class Engine:
        def __init__(self):self.n=0;self.calls=[];self.io_processor=self
        def add_request(self,prompts,params):self.calls.append((self.n,len(prompts)))
        def is_finished(self):return self.n>=22
        def step(self):
            self.n+=1
            with (tmp_path/'engine_steps.jsonl').open('a') as f:
                f.write(json.dumps({'phase':'prefill' if self.n<=3 else 'decode'})+'\n')
            return {i:{'token_ids':[1]*16} for i in range(8)} if self.n==22 else {}
        def postprocess(self,value):return value
    e=Engine();r=workload(e,{'track':'mixed','prompts':[[1]]*16},lambda **kw:kw,tmp_path)
    assert e.calls==[(0,4),(7,4)]
    assert r['completed']==8 and r['actual_prefill_steps']==3
