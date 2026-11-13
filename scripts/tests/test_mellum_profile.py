import gzip
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from mellum_runtime_config import CANDIDATE, apply_candidate, engine_options, PROJECT
from profile_mellum import prepare_inputs, wait_receipt
from analyze_mellum_profile import analyze_events, analyze_file


def load_labels():
    spec = importlib.util.spec_from_file_location('labels', PROJECT/'ATOM/atom/model_config/mellum_profile.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configuration_matches_validated_gpu_backend_and_clears_probes(monkeypatch, tmp_path):
    saved = json.loads((PROJECT/'artifacts/results/atom_mellum_gpu/20261113T105239431753Z/results.json').read_text())['settings']
    assert all(saved[k] == v for k,v in CANDIDATE.items())
    monkeypatch.setenv('MELLUM_FORCED_DECODE_PLAN', '/wrong/plan')
    monkeypatch.setenv('MELLUM_PREFILL_LOGITS_DIR', '/wrong/dump')
    # apply_candidate intentionally mutates the process environment; monkeypatch undo it.
    monkeypatch.setattr('os.environ', __import__('os').environ.copy())
    values = apply_candidate(1, tmp_path)
    import os
    assert 'MELLUM_FORCED_DECODE_PLAN' not in os.environ
    assert 'MELLUM_PREFILL_LOGITS_DIR' not in os.environ
    assert values['HIP_VISIBLE_DEVICES'] == '1'
    opts = engine_options(16384+64, tmp_path/'trace')
    assert opts['max_model_len'] == 16448 and opts['max_num_seqs'] == 1
    assert not opts['enable_chunked_prefill'] and opts['enforce_eager']


def test_inputs_are_fixed_exact_length_and_keep_long_suffix():
    first = prepare_inputs([1024,4096,16384])
    assert first == prepare_inputs([1024,4096,16384])
    assert [len(r['input_ids']) for r in first] == [1024,4096,16384]
    assert first[1]['derivation'] != 'unchanged'
    with pytest.raises(ValueError):prepare_inputs([2048])


def test_receipts_are_atomic_and_sequence_checked(monkeypatch, tmp_path):
    labels = load_labels()
    monkeypatch.setenv('MELLUM_PROFILE_CONTROL_DIR', str(tmp_path))
    labels.write_profile_receipt('started', 0, 1, tmp_path)
    assert wait_receipt(tmp_path, 'started', 1, .01)['sequence'] == 1
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(TimeoutError):wait_receipt(tmp_path, 'stopped', 1, .01)


def event(name, ts, dur, cat='user_annotation', ext=None, **other):
    data=dict(name=name,ts=ts,dur=dur,cat=cat,ph='X',pid=1,tid=1,args={})
    if ext is not None:data['args']['External id']=ext
    data.update(other)
    return data


def test_trace_attribution_uses_launch_correlation_not_gpu_time(tmp_path):
    # GPU launches occur after CPU ranges end; timestamp-containment would fail.
    events=[event('prefill[bs=1 tok=10]',0,10),
            event('mellum/attention_swa/layer_00',1,8),
            event('mellum/qkv/layer_00',2,2),
            event('aten::linear',2.1,1,'cpu_op',ext=7),
            event('eager_decode[bs=1]',20,10),
            event('mellum/moe_experts/layer_00',21,8),
            event('aten::bmm',22,1,'cpu_op',ext=8),
            event('gemm',100,5,'kernel',ext=7,tid=99),
            event('bmm',102,8,'kernel',ext=8,tid=100),
            event('unknown',120,1,'kernel',ext=999,tid=99),
            event('prefill[bs=1 tok=10]',100,10,'gpu_user_annotation',ext=1,pid=2,tid=0),
            event('mellum/qkv/layer_00',100,5,'gpu_user_annotation',ext=7,pid=2,tid=0)]
    result=analyze_events(events)
    assert result['phase_kernel_us']=={'prefill':5,'decode':8,'unattributed':1}
    assert result['phase_cpu_ranges']['prefill']['calls']==1
    assert result['module_annotation_count']==3
    assert result['phase_exclusive_module_kernel_us']['prefill']['qkv']==5
    assert result['exclusive_module_kernel_us']['qkv']==5
    assert result['inclusive_module_kernel_us']['attention_swa']==5
    assert result['summed_gpu_kernel_us']==14
    assert result['kernel_active_union_us']==11  # overlap must not be double-counted
    assert result['kernel_span_us']==21
    trace=tmp_path/'trace.json.gz'
    with gzip.open(trace,'wt') as f:json.dump({'traceEvents':events},f)
    assert analyze_file(trace)==result
    assert analyze_events([])['gpu_kernel_count']==0


def test_labels_preserve_outputs_and_balance_hook_lifetime():
    labels=load_labels()
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.sliding_window=4
            self.qkv_proj=torch.nn.Linear(3,3);self.o_proj=torch.nn.Linear(3,3)
        def forward(self,x):return self.o_proj(self.qkv_proj(x))
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.self_attn=Attention()
            self.input_layernorm=torch.nn.Identity();self.post_attention_layernorm=torch.nn.Identity()
            self.mlp=torch.nn.Module();self.mlp.gate=torch.nn.Linear(3,3);self.mlp.experts=torch.nn.Linear(3,3)
        def forward(self,x):
            x=self.self_attn(self.input_layernorm(x))
            return self.mlp.experts(self.mlp.gate(self.post_attention_layernorm(x)))
    layer=Layer().eval();wrapper=types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
    x=torch.randn(2,3);expected=layer(x)
    handles=labels.install_profile_labels(wrapper)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        got=layer(x)
    assert torch.equal(got,expected)
    assert 'mellum/attention_swa/layer_00' in {e.key for e in prof.key_averages()}
    for h in handles:h.remove()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in layer.modules())


def test_real_model_runner_profile_methods_acknowledge_and_remove_hooks(monkeypatch, tmp_path):
    import ast
    import logging
    import os
    import time
    from typing import Optional
    labels = load_labels()
    order = []
    class Handle:
        def remove(self):order.append('remove')
    def install(model):
        order.append('install');return [Handle()]
    monkeypatch.setattr(labels, 'install_profile_labels', install)
    monkeypatch.setitem(sys.modules, 'atom.model_config.mellum_profile', labels)
    monkeypatch.setenv('MELLUM_PROFILE_CONTROL_DIR', str(tmp_path/'control'))
    monkeypatch.setenv('MELLUM_PROFILE_LABELS', '1')
    class Profiler:
        def __init__(self, **kwargs):self.callback=kwargs['on_trace_ready']
        def __enter__(self):order.append('enter')
        def __exit__(self,*args):order.append('exit');self.callback(self)
        def export_chrome_trace(self,path):Path(path).write_text('{"traceEvents": []}')
    fake_prof=types.SimpleNamespace(profile=Profiler,ProfilerActivity=types.SimpleNamespace(CPU=0,CUDA=1))
    tree=ast.parse((PROJECT/'ATOM/atom/model_engine/model_runner.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunner')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('start_profiler','stop_profiler')]
    scope=dict(os=os,time=time,Optional=Optional,logger=logging.getLogger('test'),
               envs=types.SimpleNamespace(ATOM_PROFILER_MORE=False),torch_profiler=fake_prof,
               torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda device:order.append('sync'))))
    for method in methods:
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<ModelRunner profile methods>','exec'),scope)
    trace=tmp_path/'traces';trace.mkdir()
    runner=types.SimpleNamespace(profiler_dir=str(trace),profiler=None,model=object(),device='cuda:0',rank=0,
                                config=types.SimpleNamespace(model='mellum'))
    scope['start_profiler'](runner)
    assert wait_receipt(tmp_path/'control','started',1,.01)['kind']=='started'
    scope['start_profiler'](runner)  # idempotent while active
    assert order.count('install')==1
    scope['stop_profiler'](runner)
    assert wait_receipt(tmp_path/'control','stopped',1,.01)['traces']
    assert runner.profiler is None and not runner._mellum_profile_handles
    assert order==['sync','install','enter','sync','exit','remove']


def test_prepare_only_via_workspace_compatibility_path():
    import subprocess
    result=subprocess.run(['/opt/venv/bin/python','/workspace/scripts/profile_mellum.py',
                           '--prepare-only','--lengths','1024'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'RESULT prepared output' in result.stdout
