"""Exercise the real sampler class without importing GPU-only AITER kernels."""
import ast
from pathlib import Path
import pytest
import torch


def sampler_class():
    source=Path(__file__).parents[1]/'atom/model_ops/sampler.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Sampler')
    scope={'torch':torch,'nn':torch.nn,'SAMPLER_EPS':1e-10}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    return scope['Sampler']


@pytest.mark.parametrize('filtered',[False,True])
def test_greedy_ties_choose_first_index_without_random_kernel(filtered):
    sampler=sampler_class()()
    def forbidden(*args,**kwargs):raise AssertionError('Greedy reached random sampling')
    sampler._temperature_sample=forbidden
    sampler._topk_topp_sample=forbidden
    logits=torch.full((3,98304),-10.0,dtype=torch.bfloat16)
    logits[0,[8599,8943]]=37.25
    logits[1,[392,2516]]=34.25
    logits[2,123]=20
    result=sampler(logits,torch.full((3,),1e-10),
        top_ks=torch.tensor([2,2,2]) if filtered else None,all_greedy=True)
    assert result.dtype==torch.int32
    assert result.tolist()==[8599,392,123]


def test_non_greedy_unfiltered_keeps_temperature_dispatch():
    sampler=sampler_class()()
    sentinel=torch.tensor([7],dtype=torch.int)
    sampler._temperature_sample=lambda *args,**kwargs:sentinel
    assert sampler(torch.zeros(1,8),torch.ones(1),all_greedy=False) is sentinel
