"""CPU wiring, mapping and audit checks; real GPU operators are not exercised."""
import ast
import json
from pathlib import Path
import types
from unittest.mock import patch
import pytest
import torch
from atom.model_config.mellum import MellumConfig
from atom.model_config.mellum_utils import aiter_rope_parameters, mellum_load_target

ROOT=Path(__file__).parents[1]
MODEL=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')

def classes():
    tree=ast.parse((ROOT/'atom/models/mellum.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) or isinstance(n,ast.FunctionDef) and n.name=='make_mellum_norm']
    ns={'nn':torch.nn,'torch':torch,'json':json,'Path':Path,'os':__import__('os'),
        'logging':__import__('logging'),'aiter_rope_parameters':aiter_rope_parameters,
        'mellum_load_target':mellum_load_target}
    exec(compile('from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])),'mellum_classes','exec'),ns)
    return ns

def test_registered_without_compile_decoration():
    tree=ast.parse((ROOT/'atom/model_engine/model_runner.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='support_model_arch_dict' for t in n.targets))
    assert ast.literal_eval(node.value)['MellumForCausalLM']=='atom.models.mellum.MellumForCausalLM'
    tree=ast.parse((ROOT/'atom/models/mellum.py').read_text())
    assert all(not n.decorator_list for n in tree.body if isinstance(n,ast.ClassDef))

def test_layer_specific_wiring():
    ns=classes();calls=[]
    ns.update(MellumAttention=lambda **kw:calls.append(kw) or torch.nn.Identity(),
              MellumSparseMoeBlock=lambda *a,**kw:torch.nn.Identity(),
              Qwen3MoeSparseMoeBlock=lambda *a,**kw:torch.nn.Identity(),
              RMSNorm=lambda *a,**kw:torch.nn.Identity(),ENABLE_ALLREDUCE_RMSNORM_FUSION=False)
    cfg=MellumConfig.from_pretrained(MODEL,local_files_only=True)
    atom=types.SimpleNamespace(hf_config=cfg,kv_cache_dtype='bf16',quant_config=None)
    for layer in range(28):ns['MellumDecoderLayer'](atom,layer,f'model.layers.{layer}')
    assert len(calls)==28
    assert [c['sliding_window'] for c in calls]==[1024,1024,1024,None]*7
    assert all(c['head_dim']==128 for c in calls)
    assert calls[3]['rope_scaling']['attn_factor']==pytest.approx(1)
    assert 'attention_factor' not in calls[3]['rope_scaling']
    assert cfg.rope_parameters['full_attention']['attention_factor']==1.2772588722239782

def test_attention_dimensions():
    ns=classes();seen=[]
    ns.update(get_tensor_model_parallel_world_size=lambda:1,
              QKVParallelLinear=lambda *a,**kw:torch.nn.Identity(),
              RowParallelLinear=lambda *a,**kw:torch.nn.Identity(),
              get_rope=lambda *a,**kw:torch.nn.Identity(),
              RMSNorm=lambda *a,**kw:torch.nn.Identity(),
              Attention=lambda *a,**kw:seen.append(kw) or torch.nn.Identity(),
              ENABLE_ALLREDUCE_RMSNORM_FUSION=False,ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=False)
    attn=ns['MellumAttention'](2304,32,4,head_dim=128,atom_config=types.SimpleNamespace(quant_config=None),sliding_window=1024)
    assert (attn.q_size,attn.kv_size)==(4096,512)
    assert seen[0]['per_layer_sliding_window']==1024

def test_all_index_names_have_distinct_load_slots():
    names=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    targets=[mellum_load_target(n) for n in names]
    assert len(targets)==len(set(targets))==5631
    assert sum(t[2] is not None for t in targets)==28*64*3
    assert sum(t[1] in {'q','k','v'} for t in targets)==28*3
    # Compare expert target patterns with the real FusedMoE mapping method.
    tree=ast.parse((ROOT/'atom/model_ops/moe.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FusedMoE')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='make_expert_params_mapping')
    fn.decorator_list=[];ns={}
    exec(compile('from __future__ import annotations\n'+ast.unparse(fn),'expert_map','exec'),ns)
    for target,source,expert,shard in ns['make_expert_params_mapping'](None,'gate_proj','down_proj','up_proj',64):
        name='model.layers.0.mlp.'+source+'weight'
        assert mellum_load_target(name)==('model.layers.0.mlp.'+target+'weight',shard,expert)

@pytest.mark.parametrize('amplitude',[1.0,1.2772588722239782,2.0])
def test_rope_amplitude(amplitude):
    import math
    p=aiter_rope_parameters({'rope_type':'yarn','factor':16,'attention_factor':amplitude})
    assert (1+0.1*math.log(16))*p['attn_factor']==pytest.approx(amplitude)

def test_audit_rejects_missing_expert(tmp_path):
    ns=classes();cls=ns['MellumForCausalLM']
    obj=cls.__new__(cls);torch.nn.Module.__init__(obj)
    obj.atom_config=types.SimpleNamespace(model=str(tmp_path))
    names=['model.layers.0.mlp.experts.0.gate_proj.weight','model.layers.0.mlp.experts.0.up_proj.weight']
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':dict.fromkeys(names,'shard')}))
    obj._loaded_mellum_parts={mellum_load_target(names[0])}
    with pytest.raises(RuntimeError,match='1 missing'):obj.validate_loaded_weights()
    obj._loaded_mellum_parts.add(mellum_load_target(names[1]));obj.validate_loaded_weights()

def test_audit_records_only_successful_loader_calls(monkeypatch):
    import sys
    ns=classes();cls=ns['MellumForCausalLM']
    obj=cls.__new__(cls);torch.nn.Module.__init__(obj)
    def default(param, weight):param.data.copy_(weight)
    fake=types.ModuleType('atom.model_loader.loader');fake.default_weight_loader=default
    monkeypatch.setitem(sys.modules,'atom.model_loader.loader',fake)
    direct=torch.nn.Parameter(torch.zeros(2));qkv=torch.nn.Parameter(torch.zeros(2));expert=torch.nn.Parameter(torch.zeros(2))
    def packed(param,weight,shard_id):
        if shard_id=='invalid':raise ValueError('bad shard')
    def moe(param,weight,weight_name='',shard_id='',expert_id=0):pass
    qkv.weight_loader=packed;expert.weight_loader=moe
    obj.named_parameters=lambda:iter([('model.norm.weight',direct),('model.layers.0.self_attn.qkv_proj.weight',qkv),('model.layers.0.mlp.experts.w13_weight',expert)])
    obj._install_load_audit()
    direct.weight_loader(direct,torch.ones(2))
    qkv.weight_loader(qkv,torch.ones(2),'q')
    expert.weight_loader(expert,torch.ones(2),'source','w3',7)
    assert len(obj._loaded_mellum_parts)==3
    assert ('model.layers.0.mlp.experts.w13_weight','w3',7) in obj._loaded_mellum_parts
    assert torch.equal(direct,torch.ones(2))
    with pytest.raises(ValueError):qkv.weight_loader(qkv,torch.ones(2),'invalid')
    assert len(obj._loaded_mellum_parts)==3

def test_prefill_diagnostic_skips_dummy_and_decode(tmp_path,monkeypatch):
    import sys
    ns=classes();cls=ns['MellumForCausalLM']
    obj=cls.__new__(cls);torch.nn.Module.__init__(obj)
    obj.lm_head=torch.nn.Linear(4,6,bias=False)
    context=types.SimpleNamespace(is_prefill=True,is_dummy_run=True,batch_size=1)
    fake=types.ModuleType('atom.utils.forward_context')
    fake.get_forward_context=lambda:types.SimpleNamespace(context=context)
    monkeypatch.setitem(sys.modules,'atom.utils.forward_context',fake)
    monkeypatch.setenv('MELLUM_PREFILL_LOGITS_DIR',str(tmp_path))
    x=torch.randn(3,4)
    expected=obj.lm_head(x)
    assert torch.equal(obj.compute_logits(x),expected)
    assert not list(tmp_path.glob('*.pt'))
    context.is_dummy_run=False
    obj.compute_logits(x)
    torch.testing.assert_close(torch.load(tmp_path/'prefill_000000.pt',weights_only=True),expected[-1])
    context.is_prefill=False;obj.compute_logits(x)
    assert len(list(tmp_path.glob('*.pt')))==1
    context.is_prefill=True;obj.compute_logits(x)
    assert (tmp_path/'prefill_000001.pt').exists()
    context.batch_size=2
    with pytest.raises(RuntimeError,match='batch_size=1'):obj.compute_logits(x)

def test_layer_dump_preserves_forward_and_materializes_residual(tmp_path,monkeypatch):
    import sys
    ns=classes();cls=ns['MellumModel']
    obj=cls.__new__(cls);torch.nn.Module.__init__(obj)
    obj.embed_tokens=torch.nn.Embedding(16,4)
    class Layer(torch.nn.Module):
        def __init__(self,i):super().__init__();self.layer_idx=i
        def forward(self,positions,hidden,residual,**kwargs):
            return hidden*0.25,(hidden if residual is None else residual)+0.5
    class Norm(torch.nn.Module):
        def forward(self,hidden,residual):return hidden+residual,None
    obj.layers=torch.nn.ModuleList([Layer(0),Layer(1)]);obj.norm=Norm()
    obj.start_layer=0;obj.end_layer=2
    ns['get_pp_group']=lambda:types.SimpleNamespace(is_first_rank=True,is_last_rank=True)
    context=types.SimpleNamespace(is_prefill=True,is_dummy_run=False,batch_size=1)
    module=types.ModuleType('atom.utils.forward_context');module.get_forward_context=lambda:types.SimpleNamespace(context=context)
    monkeypatch.setitem(sys.modules,'atom.utils.forward_context',module)
    ids=torch.tensor([1,2,3]);positions=torch.arange(3)
    monkeypatch.delenv('MELLUM_LAYER_DUMP_DIR',raising=False)
    expected=obj(ids,positions)
    monkeypatch.setenv('MELLUM_LAYER_DUMP_DIR',str(tmp_path))
    got=obj(ids,positions)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    sample=tmp_path/'sample_000000'
    torch.testing.assert_close(torch.load(sample/'final_norm.pt',weights_only=True),expected)
    torch.testing.assert_close(torch.load(sample/'layer_01.pt',weights_only=True),expected)
    context.is_dummy_run=True;obj(ids,positions)
    assert not (tmp_path/'sample_000001').exists()

def test_stage_observers_are_read_only_and_split_qkv(tmp_path,monkeypatch):
    monkeypatch.delenv('MELLUM_MOE_REPLAY_INPUT_DIR',raising=False)
    from atom.model_config.mellum_diagnostics import install_stage_hooks
    class Norm(torch.nn.Module):
        def __init__(self,size):super().__init__();self.weight=torch.nn.Parameter(torch.ones(size))
        def forward(self,x,residual=None):
            return x*self.weight if residual is None else (x*self.weight,residual)
    class MLP(torch.nn.Module):
        def __init__(self):super().__init__();self.gate=torch.nn.Linear(4,3,bias=False)
        def forward(self,x):self.gate(x);return x*2
    attn=types.SimpleNamespace(q_size=4,kv_size=2,qkv_proj=torch.nn.Linear(4,8,bias=False),
        q_norm=Norm(2),k_norm=Norm(2),o_proj=torch.nn.Linear(4,4,bias=False))
    layer=types.SimpleNamespace(self_attn=attn,input_layernorm=Norm(4),post_attention_layernorm=Norm(4),mlp=MLP())
    active=[None]
    handles=install_stage_hooks(layer,lambda:active[0],'atom')
    x=torch.randn(3,4);original=x.clone()
    layer.input_layernorm(x)
    assert not list(tmp_path.glob('**/*.pt'))
    active[0]=tmp_path/'sample_000000'
    y=layer.input_layernorm(x)
    q,k,v=attn.qkv_proj(y).split([4,2,2],dim=-1)
    attn.q_norm(q.reshape(3,2,2));attn.k_norm(k.reshape(3,1,2))
    z=attn.o_proj(q);layer.post_attention_layernorm(z,x);layer.mlp(x)
    torch.testing.assert_close(x,original,rtol=0,atol=0)
    torch.testing.assert_close(torch.load(active[0]/'q_proj.pt',weights_only=True),q)
    torch.testing.assert_close(torch.load(tmp_path/'weights/q_proj.pt',weights_only=True),attn.qkv_proj.weight[:4])
    assert (active[0]/'router_logits.pt').exists() and (active[0]/'moe_output.pt').exists()
    for handle in handles:handle.remove()

def test_moe_replay_skips_inactive_and_requires_same_shape_dtype(tmp_path):
    from atom.model_config.mellum_diagnostics import install_moe_input_replay
    module=torch.nn.Identity();active=[None]
    handle=install_moe_input_replay(module,lambda:active[0],tmp_path)
    x=torch.zeros(2,4,dtype=torch.bfloat16)
    assert torch.equal(module(x),x)
    active[0]=tmp_path/'sample_000000';active[0].mkdir()
    wanted=torch.ones_like(x);torch.save(wanted,active[0]/'moe_input.pt')
    assert torch.equal(module(x),wanted)
    assert torch.count_nonzero(x)==0
    torch.save(wanted.float(),active[0]/'moe_input.pt')
    with pytest.raises(ValueError,match='shape/dtype'):module(x)
    handle.remove()
    assert torch.equal(module(x),x)


def test_replay_routing_rounding_preserves_native_ids():
    from atom.model_config.mellum_diagnostics import make_replay_router
    logits=torch.zeros(1,4,dtype=torch.bfloat16)
    weights=torch.tensor([[0.123456,0.876544]])
    ids=torch.tensor([[3,1]],dtype=torch.int32)
    route=make_replay_router('rounded_native',lambda **kw:(weights,ids))
    got,index=route(None,logits,2,True)
    assert got.dtype==torch.float32 and index.dtype==torch.int32
    assert torch.equal(index,ids)
    assert torch.equal(got,weights.bfloat16().float())
    hf=make_replay_router('hf')
    logits=torch.tensor([[1.,2.,3.,4.]],dtype=torch.bfloat16)
    got,index=hf(None,logits,2,True)
    expected,chosen=logits.float().softmax(-1).topk(2)
    expected=(expected/expected.sum(-1,keepdim=True)).bfloat16().float()
    assert torch.equal(got,expected) and torch.equal(index,chosen.int())

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_mellum_norm_matches_official_reference_order(dtype):
    path=Path('/workspace/venvs/hf-reference/lib/python3.12/site-packages/transformers/models/mellum/modeling_mellum.py')
    if not path.exists():pytest.skip('Official reference environment is not installed')
    tree=ast.parse(path.read_text())
    ref=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MellumRMSNorm')
    ref.decorator_list=[];ns={'torch':torch,'nn':torch.nn}
    exec(compile('from __future__ import annotations\n'+ast.unparse(ref),'official_mellum_norm','exec'),ns)
    actual=classes()['MellumRMSNorm'](8).to(dtype=dtype)
    expected=ns['MellumRMSNorm'](8).to(dtype=dtype)
    generator=torch.Generator().manual_seed(321)
    weight=torch.randn(8,generator=generator).to(dtype)
    with torch.no_grad():actual.weight.copy_(weight);expected.weight.copy_(weight)
    x=torch.randn(2,3,8,generator=generator).to(dtype);before=x.clone()
    residual=torch.randn(2,3,8,generator=generator).to(dtype)
    torch.testing.assert_close(actual(x),expected(x),rtol=0,atol=0)
    y,updated=actual(x,residual)
    torch.testing.assert_close(y,expected(x+residual),rtol=0,atol=0)
    torch.testing.assert_close(updated,x+residual,rtol=0,atol=0)
    torch.testing.assert_close(x,before,rtol=0,atol=0)
