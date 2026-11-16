import sys,json
from types import SimpleNamespace as NS
import torch
import pytest
from atom.model_config.mellum_fastpaths import make_norm,write_audit


def test_native_norm_factory_preserves_parameter_key_shape_and_residual_api(monkeypatch):
    class Reference(torch.nn.Module):
        def __init__(self,dim,eps,fused_allreduce):super().__init__();self.weight=torch.nn.Parameter(torch.ones(dim));self.eps=eps
        def forward(self,x,residual=None):return x if residual is None else (x+residual,x+residual)
    class Native(Reference):pass
    monkeypatch.setitem(sys.modules,'atom.model_ops.layernorm',NS(RMSNorm=Native))
    monkeypatch.setenv('MELLUM_NORM_BACKEND','reference');a=make_norm(Reference,128,1e-6)
    monkeypatch.setenv('MELLUM_NORM_BACKEND','aiter');b=make_norm(Reference,128,1e-6)
    assert type(a) is Reference and type(b) is Native
    assert list(a.state_dict())==list(b.state_dict())==['weight'] and b.weight.shape==(128,)
    x=torch.randn(2,4,128);res=torch.randn_like(x);assert b(x).shape==x.shape and len(b(x,res))==2
    monkeypatch.setenv('MELLUM_NORM_BACKEND','unknown')
    with pytest.raises(ValueError):make_norm(Reference,128)


def test_constructor_audit_counts_actual_norms_and_separate_projection_attributes(monkeypatch,tmp_path):
    m=torch.nn.Module();m.model=torch.nn.Module();m.model.norm=torch.nn.LayerNorm(4)
    m.model.layer=torch.nn.Module();m.model.layer.q_norm=torch.nn.LayerNorm(4)
    m.model.layer.qkv_proj=torch.nn.Linear(4,4);m.model.layer.qkv_proj._mellum_projection_sizes=(2,1,1)
    p=tmp_path/'audit.json';monkeypatch.setenv('MELLUM_FASTPATH_AUDIT_PATH',str(p));write_audit(m)
    data=json.loads(p.read_text());assert data['norm_count']==2 and data['qkv_count']==data['separate_qkv_count']==1
