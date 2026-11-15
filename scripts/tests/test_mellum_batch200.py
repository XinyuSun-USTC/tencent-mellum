import json,sys,types
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate_mellum_batch200 import drive_window,summarize,read_batches


def cases(n):
    return [{'name':f'p{i}','input_ids':[i+1],'expected_ids':[i+1],
             'prompt_source':'HumanEval','source_id':str(i)} for i in range(n)]


class FakeEngine:
    def __init__(self):
        self.counter=10;self.max_pending=0
        self.io_processor=types.SimpleNamespace(requests={},postprocess=self.postprocess)
    def add_request(self,inputs,params):
        for ids in inputs:
            self.io_processor.requests[self.counter]=types.SimpleNamespace(id=self.counter,token_ids=list(ids))
            self.counter+=1
        self.max_pending=max(self.max_pending,len(self.io_processor.requests))
    def is_finished(self):return not self.io_processor.requests
    def step(self):return [self.io_processor.requests[max(self.io_processor.requests)]]
    def postprocess(self,seqs):
        result={}
        for seq in seqs:
            self.io_processor.requests.pop(seq.id)
            result[seq.id]={'token_ids':seq.token_ids,'finish_reason':'length'}
        return result


def test_completion_order_does_not_change_mapping_and_window_refills():
    engine=FakeEngine();data=cases(9);progress=[]
    outputs,submitted,completed=drive_window(engine,data,lambda row:None,3,lambda r,*a:progress.append(len(r)))
    assert engine.max_pending==3 and len(submitted)>1
    assert [r['token_ids'] for r in outputs]==[r['expected_ids'] for r in data]
    assert [r['index'] for r in completed]!=list(range(9))
    assert submitted[0]['outstanding']==3 and sum(len(r['names']) for r in submitted)==9
    assert progress[-1]==9


def test_unknown_completion_is_not_silently_accepted():
    engine=FakeEngine();engine.io_processor.postprocess=lambda seqs:{999:{'token_ids':[1]}}
    with pytest.raises(RuntimeError,match='Unknown'):drive_window(engine,cases(3),lambda row:None,2,lambda *args:None)


def test_eos_short_output_keeps_unmatched_reference_positions_in_denominator():
    data=cases(2);data[0]['expected_ids']=[1,2,3]
    rows,groups=summarize(data,[{'token_ids':[1],'finish_reason':'eos'},{'token_ids':[2],'finish_reason':'length'}])
    assert groups['all']['compared_positions']==4 and groups['all']['matching_positions']==2
    assert rows[0]['first_difference_zero_based']==1 and not rows[0]['exact']
    assert groups['at_least_1024']['token_match_fraction'] is None


def test_audit_distinguishes_actual_decode_batch_from_client_window(tmp_path):
    f=tmp_path/'audit.jsonl';f.write_text('\n'.join(json.dumps(r) for r in [
        {'phase':'prefill','batch_size':7},{'phase':'decode','batch_size':32},{'phase':'decode','batch_size':16}]))
    stats=read_batches(f)
    assert stats['prefill']['max_batch_size']==7 and stats['decode']['max_batch_size']==32
