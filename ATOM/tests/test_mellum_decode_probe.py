import json
import types
import pytest
import torch
from atom.model_config.mellum_decode_probe import ForcedDecodeProbe,FreeGenerationProbe

def test_free_generation_capture_does_not_force_or_mutate_logits(tmp_path):
    p=FreeGenerationProbe(tmp_path)
    original=torch.arange(8,dtype=torch.bfloat16)[None];before=original.clone()
    p.observe(torch.tensor([1,2]),torch.tensor([0,1]),context(prefill=True))
    p.save(original,context(prefill=True))
    assert torch.equal(original,before)
    saved=torch.load(tmp_path/'sample_000000/step_000_logits.pt',weights_only=True)
    assert torch.equal(saved,original[0])
    p.observe(torch.tensor([7]),torch.tensor([2]),context())
    p.save(original+1,context())
    assert p.sample==0 and p.step==1 and not p.pending

def probe(tmp_path):
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'records':[{
        'name':'test','input_ids':[[1,2]],'generated_ids':[[3,4,5]]}]}))
    return ForcedDecodeProbe(plan,tmp_path/'out')

def context(prefill=False,dummy=False):
    return types.SimpleNamespace(batch_size=1,is_prefill=prefill,is_dummy_run=dummy)

def test_forcing_saves_true_logits_and_checks_history(tmp_path):
    p=probe(tmp_path);logits=torch.arange(8,dtype=torch.bfloat16)[None]
    assert p.force(logits,context(dummy=True)) is logits
    p.observe(torch.tensor([0]),torch.tensor([0]),context(dummy=True));assert p.sample==-1
    for step,(ids,pos) in enumerate([([1,2],[0,1]),([3],[2]),([4],[3])]):
        p.observe(torch.tensor(ids),torch.tensor(pos),context(prefill=step==0))
        got=p.force(logits,context(prefill=step==0))
        assert got.argmax().item()==[3,4,5][step]
        assert torch.isfinite(got).sum()==1
        saved=torch.load(tmp_path/'out/sample_000000'/f'step_{step:03d}_logits.pt',weights_only=True)
        assert torch.equal(saved,logits[0]) and saved.argmax().item()==7
    assert torch.equal(logits,torch.arange(8,dtype=torch.bfloat16)[None])

def test_forcing_rejects_bad_positions_and_unpaired_calls(tmp_path):
    p=probe(tmp_path)
    with pytest.raises(ValueError,match='Invalid'):p.force(torch.zeros(1,8),context())
    p.observe(torch.tensor([1,2]),torch.tensor([0,1]),context(prefill=True))
    with pytest.raises(ValueError,match='matching logits'):p.observe(torch.tensor([3]),torch.tensor([2]),context())
    p.force(torch.zeros(1,8),context())
    with pytest.raises(ValueError,match='token/position'):p.observe(torch.tensor([3]),torch.tensor([99]),context())

def test_one_terminal_forward_is_unscored_and_next_request_can_start(tmp_path):
    p=probe(tmp_path);logits=torch.zeros(1,8)
    for step,(ids,pos) in enumerate([([1,2],[0,1]),([3],[2]),([4],[3])]):
        p.observe(torch.tensor(ids),torch.tensor(pos),context(prefill=step==0));p.force(logits,context())
    p.observe(torch.tensor([5]),torch.tensor([4]),context())
    assert p.force(logits,context()) is logits
    terminal=tmp_path/'out/sample_000000/terminal_inputs.json'
    assert json.loads(terminal.read_text())['unscored_terminal']
    assert not (terminal.parent/'step_003_logits.pt').exists()
    p.records.append({'name':'next','input_ids':[[7]],'generated_ids':[[2,3]]})
    p.observe(torch.tensor([7]),torch.tensor([0]),context(prefill=True))
    assert p.sample==1 and p.step==0

def test_layer_hooks_materialize_residual_and_only_capture_active_steps(tmp_path):
    p=probe(tmp_path)
    class Layer(torch.nn.Module):
        def forward(self,x):return x*2,x
    backbone=types.SimpleNamespace(embed_tokens=torch.nn.Embedding(8,4),norm=torch.nn.Identity(),
                                   layers=torch.nn.ModuleList([Layer()]))
    handles=p.install_layer_hooks(backbone)
    p.observe(torch.tensor([1,2]),torch.tensor([0,1]),context(prefill=True))
    x=backbone.embed_tokens(torch.tensor([1,2]));backbone.layers[0](x);backbone.norm(x)
    saved=torch.load(tmp_path/'out/sample_000000/step_000_layer_00.pt',weights_only=True)
    torch.testing.assert_close(saved,x*3)
    p.force(torch.zeros(1,8),context())
    old=(tmp_path/'out/sample_000000/step_000_layer_00.pt').read_bytes()
    backbone.layers[0](x+1)
    assert (tmp_path/'out/sample_000000/step_000_layer_00.pt').read_bytes()==old
    for handle in handles:handle.remove()
