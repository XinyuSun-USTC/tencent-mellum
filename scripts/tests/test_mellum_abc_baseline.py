"""Dataset-only adaptation, raw metric validation, stage gating and resume."""
import collections,json,sqlite3,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_abc_baseline as study
from run_mellum_batch_campaign import write


def make_database(root,requests,success=True,tokens=256):
    root.mkdir(parents=True);db=root/'benchmark_data.db'
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE result(request TEXT,success INTEGER,prompt_tokens INTEGER,completion_tokens INTEGER,first_chunk_latency REAL,time_per_output_token REAL,start_time REAL,completed_time REAL,is_stream INTEGER,latency REAL)')
        for i,content in enumerate(requests):
            body={'messages':[{'role':'user','content':content}],'max_tokens':256,'min_tokens':256,'stream':True,'ignore_eos':True,'temperature':0.0}
            con.execute('INSERT INTO result VALUES(?,?,?,?,?,?,?,?,?,?)',(json.dumps(body),int(success),1024,tokens,1.,.1,float(i),i+26.5,1,26.5))
    write(root/'benchmark_summary.json',{'Output Throughput (tok/s)':10})


def expected(values):
    import hashlib
    return [{'prompt_sha256':hashlib.sha256(x.encode()).hexdigest()} for x in values]


def test_actual_requests_checked_as_multiset_not_completion_order(tmp_path):
    make_database(tmp_path/'client',['two','one','one'])
    result=study.inspect_result(tmp_path/'client','A',3,expected(['one','two','one']))
    assert result['complete_without_failures'] and result['ttft_p99_s']==1
    assert not study.inspect_result(tmp_path/'client','A',3,expected(['one','two','other']))['complete_without_failures']


def test_failures_remain_in_denominator_and_not_validated_as_full_score(tmp_path):
    make_database(tmp_path/'client',['hello'],False)
    result=study.inspect_result(tmp_path/'client','B',1,[])
    assert result['failures']==1 and result['successes']==0 and not result['complete_without_failures']
    assert result['ttft_p99_s'] is None


def test_wrong_output_length_and_missing_rows_rejected(tmp_path):
    make_database(tmp_path/'client',['hello'],tokens=255)
    result=study.inspect_result(tmp_path/'client','B',2,[])
    assert len(result['validation_errors'])==2 and not result['complete_without_failures']


def test_literal_load_parameters_preserved_and_dataset_translation_explicit(tmp_path):
    for scenario,n in [('A',512),('B',50),('C',200)]:
        cmd=study.command(scenario,tmp_path,n,'/data/source.jsonl')
        assert cmd[:3]==[str(study.CLIENT),str(study.ADAPTER),'perf']
        assert cmd[cmd.index('--number')+1]==str(n)
        assert cmd[cmd.index('--max-tokens')+1]=='256'
        if scenario=='C':assert cmd[cmd.index('--rate')+1]=='0.3' and '--open-loop' in cmd
        if scenario in ('A','C'):assert cmd[cmd.index('--dataset')+1]=='mellum_assessment_jsonl'
        else:assert cmd[cmd.index('--dataset')+1]=='random' and cmd[cmd.index('--parallel')+1]=='1'


def metrics(ok=True):
    return {'complete_without_failures':ok,'successes':2 if ok else 1,'requests':2,'output_tokens_per_s_gpu':10,
            'ttft_p50_s':1,'ttft_p99_s':1,'tpot_p50_s':.1,'tpot_p99_s':.2}


def manifest(smoke_only=False):
    return {'gpu':0,'smoke_only':smoke_only,'stages':{'smoke':[],**({} if smoke_only else {c:[] for c in 'BAC'})},
            'jobs':{},'claim':'test','differences':[]}


def test_all_stages_and_completed_resume_skip_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'current_matches',lambda *a:None);calls=[]
    def fake(stage,command,log,timeout):
        calls.append(stage);log.write_text('ok');write(log.parent/'results.json',{'status':'complete','cases':{stage:{'metrics':metrics()}}})
    plan=manifest();result=study.execute(tmp_path,plan,1000,fake)
    assert calls==['smoke','B','A','C'] and result['status']=='complete'
    assert study.execute(tmp_path,plan,1000,fake)['status']=='complete' and len(calls)==4


def test_smoke_failure_stops_formal_stages(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'current_matches',lambda *a:None);calls=[]
    def fake(stage,command,log,timeout):
        calls.append(stage);log.write_text('ok');write(log.parent/'results.json',{'status':'smoke_failed','cases':{'x':{'metrics':metrics(False)}}})
    result=study.execute(tmp_path,manifest(),1000,fake)
    assert calls==['smoke'] and result['status']=='smoke_failed'


def test_budget_and_corrupted_resume_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'current_matches',lambda *a:None)
    assert study.execute(tmp_path,manifest(),1,lambda *a:pytest.fail())['status']=='partial_budget_limited'
    def fake(stage,command,log,timeout):
        log.write_text('ok');write(log.parent/'results.json',{'status':'complete','cases':{'x':{'metrics':metrics()}}})
    plan=manifest(True);assert study.execute(tmp_path,plan,1000,fake)['status']=='smoke_passed'
    (tmp_path/'smoke/results.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):study.execute(tmp_path,plan,1000,fake)


def test_percentile_matches_vendor_nearest_rank():
    assert study.percentile(list(range(1,51)),99)==50
    assert study.percentile(list(range(1,51)),50)==25


def test_server_client_tokenization_drift_is_not_accepted(tmp_path):
    make_database(tmp_path/'client',['hello'])
    planned=expected(['hello']);planned[0]['tokens']=1000
    result=study.inspect_result(tmp_path/'client','C',1,planned)
    assert not result['complete_without_failures']
    assert 'Server/client rendered input token lengths differ' in result['validation_errors']
