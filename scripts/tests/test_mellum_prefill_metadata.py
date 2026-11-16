import sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atom.model_config.mellum_prefill_metadata import geometry


def context(cached=False):
    return NS(attn_metadata=NS(has_cached=cached,cu_seqlens_q=torch.tensor([0,2]),
        cu_seqlens_k=torch.tensor([0,4] if cached else [0,2]),max_seqlen_q=2,max_seqlen_k=4 if cached else 2),
        context=NS(positions=torch.tensor([2,3]) if cached else torch.tensor([0,1])))


def test_reuse_only_same_forward_and_reject_mutation(monkeypatch):
    monkeypatch.setenv('MELLUM_PREFILL_METADATA_REUSE','1');ctx=context();calls=[];original=torch.equal
    monkeypatch.setattr(torch,'equal',lambda *a:calls.append(1) or original(*a))
    a=geometry(ctx,2,2);assert geometry(ctx,2,2) is a and len(calls)==1
    assert geometry(context(),2,2)==a and len(calls)==2
    ctx.attn_metadata.cu_seqlens_k[1]=3
    with pytest.raises(ValueError):geometry(ctx,2,2)
    assert len(calls)==3


def test_default_does_not_memoize(monkeypatch):
    monkeypatch.delenv('MELLUM_PREFILL_METADATA_REUSE',raising=False);ctx=context()
    assert geometry(ctx,2,2)==([2],[2]) and not hasattr(ctx,'_mellum_prefill_geometry')


def test_cached_positions_change_invalidates_reuse(monkeypatch):
    monkeypatch.setenv('MELLUM_PREFIX_COMPAT','1');monkeypatch.setenv('MELLUM_PREFILL_METADATA_REUSE','1');ctx=context(True)
    a=geometry(ctx,2,4);assert a==([2],[4]) and geometry(ctx,2,4) is a
    ctx.context.positions[0]=0
    with pytest.raises(ValueError):geometry(ctx,2,4)


def test_inference_tensors_are_supported_without_cross_forward_cache(monkeypatch):
    monkeypatch.setenv('MELLUM_PREFILL_METADATA_REUSE','1')
    with torch.inference_mode():
        ctx=context();a=geometry(ctx,2,2)
        assert geometry(ctx,2,2) is a
        other=context();other.attn_metadata.cu_seqlens_k[1]=3
        with pytest.raises(ValueError):geometry(other,2,2)
