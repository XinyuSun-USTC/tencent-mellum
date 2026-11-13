import importlib.util
from pathlib import Path
import sys
import pytest
import torch
import transformers

SCRIPTS=Path(__file__).parents[1];sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('decode_hf',SCRIPTS/'export_mellum_decode_reference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def tiny_model():
    from transformers import MellumConfig,MellumForCausalLM
    cfg=MellumConfig(vocab_size=64,hidden_size=32,intermediate_size=32,moe_intermediate_size=16,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,num_experts=4,
        num_experts_per_tok=2,max_position_embeddings=64,sliding_window=4,
        layer_types=['sliding_attention','full_attention'],
        rope_parameters={'sliding_attention':{'rope_type':'default','rope_theta':10000},
                         'full_attention':{'rope_type':'default','rope_theta':10000}})
    cfg._attn_implementation='sdpa';cfg._experts_implementation='grouped_mm'
    return MellumForCausalLM(cfg).eval()

@pytest.mark.skipif(transformers.__version__!='5.10.1',reason='Run with dedicated HF interpreter')
def test_cached_export_handles_window_crossing_without_gpu(tmp_path):
    model=tiny_model()
    record={'name':'tiny','input_ids':[[1,2,3]],'generation_settings':{'max_new_tokens':1}}
    with torch.inference_mode():row=m.export_case(model,record,4,tmp_path,0)
    assert len(row['generated_ids'][0])==len(row['decode_steps'])==4
    assert [r['logical_context_tokens'] for r in row['decode_steps']]==[3,4,5,6]
    assert [r['positions'] for r in row['decode_steps']]==[[0,1,2],[3],[4],[5]]
    assert row['decode_steps'][0]['input_ids']==[1,2,3]
    for step in row['decode_steps']:
        assert torch.load(step['cached_logits'],weights_only=True).shape==(64,)
        assert torch.load(step['uncached_logits'],weights_only=True).shape==(64,)
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(transformers.__version__!='5.10.1',reason='Run with dedicated HF interpreter')
def test_hf_observers_restore_registry_and_do_not_change_logits(tmp_path):
    from mellum_decode_observers import observe_decode_layers
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    model=tiny_model();record={'name':'tiny','input_ids':[[1,2,3]],'generation_settings':{'max_new_tokens':1}}
    plain=tmp_path/'plain';observed=tmp_path/'observed';plain.mkdir();observed.mkdir()
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    with torch.inference_mode():
        expected=m.export_case(model,record,2,plain,0)
        with observe_decode_layers(model) as capture:
            actual=m.export_case(model,record,2,observed,0,capture)
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
    for a,b in zip(expected['decode_steps'],actual['decode_steps']):
        for field in ['cached_logits','uncached_logits']:
            torch.testing.assert_close(torch.load(a[field],weights_only=True),torch.load(b[field],weights_only=True),rtol=0,atol=0)
    for step in [0,1]:
        assert (observed/f'sample_000000/step_{step:03d}_layer_00.pt').exists()
        assert (observed/f'sample_000000/step_{step:03d}_final_norm.pt').exists()
    snap=torch.load(observed/'sample_000000/layer_00_attention.pt',weights_only=True)
    assert snap['keys'].shape==(4,2,8) and snap['query'].shape==(1,4,8)

@pytest.mark.skipif(transformers.__version__!='5.10.1',reason='Run with dedicated HF interpreter')
def test_generate_raw_logit_capture_preserves_greedy_tokens():
    model=tiny_model();ids=torch.tensor([[1,2,3]])
    options={'input_ids':ids,'attention_mask':torch.ones_like(ids),'max_new_tokens':3,
             'do_sample':False,'pad_token_id':0,'eos_token_id':28,'use_cache':True}
    with torch.inference_mode():
        plain=model.generate(**options)
        observed=model.generate(**options,return_dict_in_generate=True,output_logits=True)
    assert torch.equal(plain,observed.sequences)
    assert len(observed.logits)==observed.sequences.shape[1]-ids.shape[1]
    assert all(v.shape==(1,64) and torch.isfinite(v).all() for v in observed.logits)

@pytest.mark.skipif(transformers.__version__!='5.10.1',reason='Run with dedicated HF interpreter')
def test_atom_batched_experts_match_official_backend():
    spec=importlib.util.spec_from_file_location('atom_experts_math','/workspace/ATOM/atom/model_config/mellum_experts.py')
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    torch.manual_seed(73)
    model=tiny_model().to(torch.bfloat16)
    model.set_experts_implementation('batched_mm')
    official=model.model.layers[0].mlp
    candidate=math.MellumTorchExperts(model.config).to(torch.bfloat16)
    with torch.no_grad():
        candidate.w13_weight.copy_(official.experts.gate_up_proj)
        candidate.w2_weight.copy_(official.experts.down_proj)
    for tokens in [1,3]:
        hidden=torch.randn(tokens,32,dtype=torch.bfloat16)
        with torch.inference_mode():
            logits,weights,ids=official.gate(hidden)
            expected=official.experts(hidden,ids,weights)
            actual=candidate(hidden,logits,batched=True)
        assert torch.equal(actual,expected)

@pytest.mark.skipif(transformers.__version__!='5.10.1',reason='Run with dedicated HF interpreter')
def test_long_prefill_band_matches_official_sliding_window_mask():
    from transformers.masking_utils import sliding_window_causal_mask_function
    for n,window in [(4,4),(5,4),(17,8)]:
        query=torch.arange(n)[:,None];key=torch.arange(n)[None,:]
        expected=sliding_window_causal_mask_function(window)(0,0,query,key)
        actual=torch.ones(n,n,dtype=torch.bool).tril().triu(1-window)
        assert torch.equal(actual,expected)
