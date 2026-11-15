"""CPU contracts for controls, actual request mapping, DAG and durable resume."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_batch_campaign as campaign
from validate_mellum_batch200 import drive_window

spec=importlib.util.spec_from_file_location('campaign_math',Path(__file__).resolve().parents[2]/'ATOM/atom/model_config/mellum_campaign.py')
math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)


def model():
    class Norm(torch.nn.Module):
        def forward(self,x,residual=None):return x if residual is None else (x+residual,residual)
    layer=NS(input_layernorm=Norm(),post_attention_layernorm=Norm(),
             self_attn=NS(qkv_proj=torch.nn.Linear(3,3),o_proj=torch.nn.Linear(3,3),q_norm=Norm(),k_norm=Norm()),
             mlp=NS(gate=torch.nn.Linear(3,3),experts=torch.nn.Linear(3,3)))
    # self_attn and layer only need nn.Module for observer hooks.
    attn=torch.nn.Module()
    for k,v in vars(layer.self_attn).items():setattr(attn,k,v)
    layer.self_attn=attn
    real=torch.nn.Module()
    for k,v in vars(layer).items():setattr(real,k,v)
    return NS(model=NS(layers=[real],norm=Norm()),lm_head=torch.nn.Linear(3,3))


def context(prefill,positions):
    return NS(context=NS(is_dummy_run=False,is_prefill=prefill,positions=torch.tensor(positions)),
              attn_metadata=NS(context_lens=torch.tensor([4,5]),block_tables=torch.tensor([[9,10],[2,3]])))


def test_split_preserves_tuple_residual_dtype_and_row_order():
    x=torch.arange(18,dtype=torch.bfloat16).reshape(6,3);res=x+1;calls=[]
    def f(value,residual=None,scale=2):
        calls.append(len(value));return value*scale+residual,residual
    actual=math.split_call(f,(x,),{'residual':res,'scale':2},[0,2,6])
    assert calls==[2,4]
    assert all(torch.equal(a,b) for a,b in zip(actual,f(x,residual=res)))
    assert actual[0].dtype==torch.bfloat16


def test_controls_only_selected_phase_and_group():
    m=model();c=math.Campaign(m,{'groups':'L','phases':['decode']})
    linear=m.model.layers[0].self_attn.qkv_proj;calls=[]
    linear.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0])))
    # pre-hooks count only outer module calls; underlying forward count tested via
    # monkeypatched F.linear would be implementation-mirroring. Compare outputs.
    x=torch.randn(4,3);expected=torch.nn.functional.linear(x,linear.weight,linear.bias)
    c.begin(NS(req_ids=[9,2],num_scheduled_tokens=[1,3]),context(True,[0,0,1,2]),torch.arange(4))
    assert torch.allclose(linear(x),expected)
    assert c.phase=='prefill' and c.bounds==[0,1,4]
    c.begin(NS(req_ids=[2,9],num_scheduled_tokens=[1,1]),context(False,[3,4]),torch.tensor([1,2]))
    assert c.bounds==[0,1,2]
    assert torch.allclose(linear(x[:2]),expected[:2])
    c.end(torch.zeros(2,3));assert c.bounds is None


def test_observer_uses_real_mapping_history_and_is_output_neutral(tmp_path):
    root=tmp_path/'observe';m=model();c=math.Campaign(m,{'observe':str(root),'groups':''})
    campaign.write(tmp_path/'observe_control.json',{'8':{'stage':'S32_r0','name':'target','input_tokens':2}})
    batch=NS(req_ids=[5,8],num_scheduled_tokens=[1,2])
    c.begin(batch,context(True,[0,0,1]),torch.tensor([99,10,11]))
    x=torch.randn(3,3);before=x.clone();c.hook('00.hidden')(None,(),x)
    c.end(x);assert torch.equal(x,before)
    c.begin(NS(req_ids=[8,5],num_scheduled_tokens=[1,1]),context(False,[2,1]),torch.tensor([12,98]))
    c.hook('00.hidden')(None,(),x[:2]);c.end(x[:2])
    meta=json.loads((root/'S32_r0__target__01.json').read_text())
    assert meta['row']==0 and meta['request_id']==8 and meta['block_table']==[9,10]
    assert meta['history_sha256']==__import__('hashlib').sha256(json.dumps([10,11,12]).encode()).hexdigest()
    tensor=torch.load(root/'S32_r0__target__00.pt',weights_only=True)
    assert torch.equal(tensor['00.hidden.0'],x[2])


def test_bad_row_mapping_rejected():
    c=math.Campaign(model(),{})
    with pytest.raises(ValueError,match='real eager rows'):
        c.begin(NS(req_ids=[2],num_scheduled_tokens=[1]),context(False,[0]),torch.tensor([1,2]))


def test_capture_cap_and_dummy_do_not_change_output(tmp_path):
    campaign.write(tmp_path/'observe_control.json',{'1':{'stage':'S','name':'x','input_tokens':1}})
    c=math.Campaign(model(),{'observe':str(tmp_path/'observe'),'capture_bytes':1})
    c.begin(NS(req_ids=[1],num_scheduled_tokens=[1]),context(True,[0]),torch.tensor([1]))
    c.end(torch.zeros(1,3));assert (tmp_path/'observe/capture_truncated').exists()
    ctx=context(True,[0]);ctx.context.is_dummy_run=True
    c.begin(NS(req_ids=[1],num_scheduled_tokens=[1]),ctx,torch.tensor([1]));assert not c.bounds


def result(value=1,stable=True):
    group={'token_match_fraction':value,'exact_sequence_fraction':value,'records':32,
           'matching_positions':int(value*512),'compared_positions':512,'exact_sequences':int(value*32)}
    return {'status':'complete','stable':stable,'batch_verified':True,'wall_seconds':3,
            'cohorts':{n:{'groups':{'all':group},'wall_seconds':[3,3]} for n in ('S32','L32')}}


def test_selection_requires_both_cohorts_and_stability():
    baseline=result(.7);good=result(.9);bad=result(.8);bad['cohorts']['L32']['groups']['all']={**bad['cohorts']['L32']['groups']['all'],'token_match_fraction':.6}
    assert campaign.select_candidates({'good':good,'regressed':bad,'unstable':result(1,False)},baseline)==['good']


def test_runner_resume_verifies_complete_artifacts_and_plan(tmp_path):
    manifest={'jobs':{}};calls=[]
    def fake(stage,command,log,timeout):
        calls.append(stage);log.write_text('ok');campaign.write(log.parent/'results.json',result())
    runner=campaign.Runner(tmp_path,manifest,120,50,fake)
    assert runner.job('A1',{'x':1})['stable'];assert len(calls)==1
    assert runner.job('A1',{'x':1})['stable'];assert len(calls)==1
    with pytest.raises(ValueError,match='configuration differs'):runner.job('A1',{'x':2})
    (tmp_path/'jobs/A1/results.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact differs'):runner.job('A1',{'x':1})


def test_failure_isolation_fatal_environment_and_budget(tmp_path):
    def fake(stage,command,log,timeout):
        log.write_text('compile unavailable' if stage=='bad' else 'HIP error')
        raise RuntimeError('worker failed')
    r=campaign.Runner(tmp_path,{'jobs':{}},120,50,fake)
    assert r.job('bad',{}) is None and not r.blocked
    assert r.job('fatal',{}) is None and r.blocked
    assert r.job('later',{}) is None and r.launched==2
    r=campaign.Runner(tmp_path/'limited',{'jobs':{}},.001,50,fake)
    assert r.job('skip',{}) is None and r.limited and r.launched==0


def test_failed_attempt_preserved(tmp_path):
    count=[]
    def fake(stage,command,log,timeout):
        count.append(1);log.write_text('compile failure')
        if len(count)==1:raise RuntimeError('compile failure')
        campaign.write(log.parent/'results.json',result())
    r=campaign.Runner(tmp_path,{'jobs':{}},120,50,fake)
    assert r.job('retry',{}) is None
    assert r.job('retry',{})['stable']
    assert (tmp_path/'jobs/retry/attempt_1/worker.log').read_text()=='compile failure'


def test_dag_no_reference_reproduction_stops_dependent_controls(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'difference',lambda *a:{'pairs':[]})
    def fake(stage,command,log,timeout):
        log.write_text('ok');campaign.write(log.parent/'results.json',result(.7))
    args=NS(budget_minutes=2,job_timeout=50)
    data={k:[{}]*32 for k in ('S32','L32')}
    data['S32']=[{'name':str(i)} for i in range(32)];data['L32']=[{'name':str(i+32)} for i in range(32)]
    summary=campaign.execute(tmp_path,{'jobs':{},'gpu':0,'provenance':{}},data,args,fake)
    assert len(summary['jobs'])==4 and summary['status']=='complete_with_unresolved'


def test_head_control_selects_once_before_requestwise_gemm(monkeypatch):
    # The real head prefill uses global cu. Any call of the original on slices
    # would fail this contract; the control must select rows before native GEMM.
    from types import ModuleType
    aiter=ModuleType('aiter');tuned=ModuleType('aiter.tuned_gemm');seen=[]
    def mm(x,w,b):seen.append(x.clone());return torch.nn.functional.linear(x,w,b)
    tuned.tgemm=NS(mm=mm)
    monkeypatch.setitem(sys.modules,'aiter',aiter);monkeypatch.setitem(sys.modules,'aiter.tuned_gemm',tuned)
    m=model();head=m.lm_head;c=math.Campaign(m,{'groups':'H','phases':['prefill','decode']})
    x=torch.randn(5,3)
    c.begin(NS(req_ids=[5,9],num_scheduled_tokens=[2,3]),context(True,[0,1,0,1,2]),torch.arange(5))
    out=head(x)
    assert torch.equal(torch.cat(seen),x[[1,4]])
    assert torch.allclose(out,torch.nn.functional.linear(x[[1,4]],head.weight,head.bias))
    seen.clear();c.begin(NS(req_ids=[9,5],num_scheduled_tokens=[1,1]),context(False,[3,2]),torch.arange(2))
    assert head(x[:2]).shape==(2,3) and len(seen)==2


def test_full_dag_branching_and_worker_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'difference',lambda *a:{'pairs':[{'first_nonexact':{'boundary':'00.input_norm.0'}}]})
    def fake(stage,command,log,timeout):
        log.write_text('ok')
        if stage=='D_http':
            campaign.write(log.parent/'http/summary.json',{'status':'http_smoke_completed','review_required':False});return
        plan=json.loads((log.parent/'plan.json').read_text())
        value=result(1 if stage=='A1' or not stage.startswith('A') else .7)
        value['cohorts']={k:{'groups':{'all':result(1)['cohorts']['S32']['groups']['all']},'wall_seconds':[1,1]} for k in plan['cohorts']}
        if stage in ('A2','A8','A32'):value=result(.7)
        campaign.write(log.parent/'results.json',value)
    args=NS(budget_minutes=2,job_timeout=50)
    data={'S32':[{'name':str(i)} for i in range(32)],'L32':[{'name':str(i+32)} for i in range(32)],'K4':[{}]*4,'D200':[{}]*200}
    manifest={'jobs':{},'gpu':0,'provenance':{}}
    summary=campaign.execute(tmp_path,manifest,data,args,fake)
    assert len(summary['jobs'])==14 and manifest['started_workers']==14
    assert summary['status']=='complete' and len(summary['passing_developer200'])==2
    assert summary['decisions']['regressions_complete']


def test_worker_cap_persists_across_resume(tmp_path):
    r=campaign.Runner(tmp_path,{'jobs':{},'started_workers':14},120,50,lambda *a,**k:pytest.fail('must not start'))
    assert r.job('next',{}) is None and r.limited


def test_same_prefix_filter_excludes_postfork_and_reads_hf(tmp_path):
    left=tmp_path/'jobs/A1/observe';right=tmp_path/'jobs/A32/observe';left.mkdir(parents=True);right.mkdir(parents=True)
    hf=tmp_path/'hf.pt';torch.save(torch.tensor([3.,1.,0.]),hf)
    refs={'S32':[{'name':'p','input_ids':[10],'expected_ids':[0,1]}]}
    for step in (0,1):
        key=f'S32_r0__p__{step:02d}'
        history=[10]+[0]*step
        for directory in (left,right):
            digest=__import__('hashlib').sha256(json.dumps(history if directory==left or step==0 else [10,2]).encode()).hexdigest()
            campaign.write(directory/(key+'.json'),{'name':'p','phase':'prefill' if step==0 else 'decode','step':step,'history_sha256':digest})
            torch.save({'29.logits':torch.tensor([3.,1.,0.])},directory/(key+'.pt'))
    result_=campaign.difference(tmp_path,{'A1':{'probe_logits':{'p':[{'path':str(hf),'sha256':campaign.sha(hf)}]*2}},'A32':{'status':'complete'}},refs)
    assert len(result_['pairs'])==1 and result_['pairs'][0]['step']==0
    assert result_['pairs'][0]['bs1_versus_hf']['top1_exact']


def test_identity_published_before_async_dispatch():
    from test_mellum_batch200 import FakeEngine,cases
    class AsyncFake(FakeEngine):
        def __init__(self):
            super().__init__();self.published={};self.core_mgr=NS(add_request=self.dispatch)
        def dispatch(self,reqs):
            assert all(req.id in self.published for req in reqs)
        def add_request(self,inputs,params):
            before=set(self.io_processor.requests)
            super().add_request(inputs,params)
            self.core_mgr.add_request([self.io_processor.requests[k] for k in sorted(set(self.io_processor.requests)-before)])
    engine=AsyncFake()
    drive_window(engine,cases(5),lambda r:None,3,lambda *a:None,lambda mapping:engine.published.update(mapping))


def test_target_cache_observer_preserves_backend_and_uses_real_table(tmp_path,monkeypatch):
    from types import ModuleType
    gathered=[];module=ModuleType('atom.model_config.mellum_attention')
    def gather(k,v,table,start,end,format_):
        gathered.append((table.tolist(),start,end,format_));return k[start:end],v[start:end]
    module.gather_reference_kv=gather
    monkeypatch.setitem(sys.modules,'atom.model_config.mellum_attention',module)
    c=math.Campaign(model(),{'observe':str(tmp_path/'observe')})
    c.rows=[('target',0,1,{'context_length':5})];c.current={'target':{}}
    impl=NS(dispatch_backend=lambda ctx:lambda q,*a:q+1,sliding_window=2,_cache_format='NHD')
    c.install_cache(impl,3);ctx=context(False,[3,4]);q=torch.zeros(2,3);cache=torch.arange(18).reshape(6,3)
    actual=impl.dispatch_backend(ctx)(q,None,None,cache,cache,None,None,ctx)
    assert torch.equal(actual,q+1) and gathered==[([2,3],3,5,'NHD')]
    assert torch.equal(c.current['target']['03.cache_k'],cache[3:5])
