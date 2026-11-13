"""Random tiny-model checks for the official HF reference; CPU only, no real weights."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import traceback
import torch
import transformers
from transformers import AutoConfig, MellumConfig, MellumForCausalLM
from transformers.masking_utils import create_sliding_window_causal_mask
from safetensors.torch import save_file

ROOT = Path('/workspace')
OUT = ROOT / 'results/hf_mellum_cpu' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT.mkdir(parents=True)
checks = []
def check(name, fn):
    try:
        detail = fn()
        checks.append({'name': name, 'status': 'passed', 'detail': detail})
    except Exception:
        checks.append({'name': name, 'status': 'failed', 'traceback': traceback.format_exc()})
    print(name, checks[-1]['status'], flush=True)

def close(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    return {'max_abs_error': (a-b).abs().max().item(), 'elements': a.numel()}

assert transformers.__version__ == '5.10.1'
assert str(Path(transformers.__file__).resolve()).startswith('/workspace/venvs/hf-reference/')
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
torch.manual_seed(20261111)
actual = json.loads(Path('/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json').read_text())
tiny = copy.deepcopy(actual)
tiny.update(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
            max_position_embeddings=128, sliding_window=4, pad_token_id=96,
            bos_token_id=0, eos_token_id=95, layer_types=actual['layer_types'][:4],
            mlp_layer_types=['sparse'] * 4)
tiny['rope_parameters']['full_attention']['original_max_position_embeddings'] = 8
config = MellumConfig(**tiny)
config._attn_implementation = 'eager'
model = MellumForCausalLM(config).to(device='cpu', dtype=torch.float32).eval()
assert all(p.device.type == 'cpu' for p in model.parameters())
ids = torch.randint(1, 90, (1, 15), device='cpu')

@torch.inference_mode()
def cache_case(length, chunks):
    x = ids[:, :length]
    expected = model(x, use_cache=False).logits
    cache = None
    outputs = []
    offset = 0
    for width in chunks:
        part = model(x[:, offset:offset+width], past_key_values=cache, use_cache=True)
        cache = part.past_key_values
        outputs.append(part.logits)
        offset += width
    assert offset == length
    result = close(torch.cat(outputs, dim=1), expected)
    assert cache.get_seq_length() == length
    result['cache_layer_lengths'] = [layer.keys.shape[-2] for layer in cache.layers]
    return result

@torch.inference_mode()
def mask_case(window, length):
    cfg = copy.deepcopy(config)
    cfg.sliding_window = window
    embeds = torch.zeros(1, length, config.hidden_size)
    pos = torch.arange(length).unsqueeze(0)
    mask = create_sliding_window_causal_mask(config=cfg, inputs_embeds=embeds,
                                            attention_mask=None, past_key_values=None, position_ids=pos)
    q = torch.arange(length)[:, None]
    k = torch.arange(length)[None, :]
    allowed = (k <= q) & (k > q-window)
    torch.testing.assert_close(mask[0, 0] == 0, allowed)
    return {'last_query_visible_keys': allowed[-1].sum().item(), 'window': window, 'length': length}

@torch.inference_mode()
def backends():
    cfg = copy.deepcopy(config)
    cfg._attn_implementation = 'sdpa'
    other = MellumForCausalLM(cfg).eval()
    other.load_state_dict(model.state_dict(), strict=True)
    return close(other(ids, use_cache=False).logits, model(ids, use_cache=False).logits)

@torch.inference_mode()
def padding():
    short = ids[:, :7]
    batch = torch.cat([torch.cat([torch.full((1, 8), 96), short], dim=1), ids], dim=0)
    mask = batch.ne(96).long()
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    got = model(batch, attention_mask=mask, position_ids=positions, use_cache=False).logits
    result = close(got[:1, -7:], model(short, use_cache=False).logits)
    result['long_row'] = close(got[1:], model(ids, use_cache=False).logits)
    return result

@torch.inference_mode()
def greedy():
    x = ids[:, :6]
    a = model.generate(x, do_sample=False, max_new_tokens=8, use_cache=True)
    b = model.generate(x, do_sample=False, max_new_tokens=8, use_cache=False)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    return {'ids': a.tolist(), 'new_tokens': a.shape[-1]-x.shape[-1]}

@torch.inference_mode()
def checkpoint_roundtrip(unpacked):
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        folder = Path(folder)
        model.config.save_pretrained(folder)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if unpacked:
            converted = {}
            for key, value in state.items():
                if key.endswith('mlp.experts.gate_up_proj'):
                    prefix = key.removesuffix('gate_up_proj')
                    for expert in range(config.num_experts):
                        g, u = value[expert].chunk(2, dim=0)
                        converted[f'{prefix}{expert}.gate_proj.weight'] = g.contiguous()
                        converted[f'{prefix}{expert}.up_proj.weight'] = u.contiguous()
                elif key.endswith('mlp.experts.down_proj'):
                    prefix = key.removesuffix('down_proj')
                    for expert in range(config.num_experts):
                        converted[f'{prefix}{expert}.down_proj.weight'] = value[expert].contiguous()
                else:
                    converted[key] = value
            state = converted
        save_file(state, folder/'model.safetensors', metadata={'format': 'pt'})
        loaded, info = MellumForCausalLM.from_pretrained(folder, local_files_only=True,
                       attn_implementation='eager', output_loading_info=True, dtype=torch.float32)
        for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']:
            assert not info.get(key), (key, info.get(key))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        result = close(loaded(ids, use_cache=False).logits, model(ids, use_cache=False).logits)
        result['unpacked_source'] = unpacked
        return result

def roundtrip_config():
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        config.save_pretrained(folder)
        loaded = AutoConfig.from_pretrained(folder, local_files_only=True)
        for key in ['layer_types', 'mlp_layer_types', 'rope_parameters', 'head_dim', 'num_experts']:
            assert getattr(loaded, key) == getattr(config, key), key
    return {'config_class': type(loaded).__name__}

check('config_roundtrip', roundtrip_config)
for length in [3, 4, 5, 9, 15]:
    check(f'cache_tokenwise_len_{length}', lambda length=length: cache_case(length, [1]*length))
    check(f'cache_chunked_len_{length}', lambda length=length: cache_case(length, [length-1, 1]))
check('cache_chunk_cross_window', lambda: cache_case(15, [3, 6, 2, 4]))
for window, length in [(4, 3), (4, 4), (4, 5), (4, 9), (1024, 1023), (1024, 1024), (1024, 1025)]:
    check(f'mask_w{window}_len{length}', lambda window=window, length=length: mask_case(window, length))
check('eager_vs_sdpa', backends)
check('left_padded_batch', padding)
check('greedy_cache_on_off', greedy)
check('packed_checkpoint_roundtrip', lambda: checkpoint_roundtrip(False))
check('unpacked_experts_checkpoint_conversion', lambda: checkpoint_roundtrip(True))
assert not torch.cuda.is_initialized()
result = {'status': 'passed' if all(c['status']=='passed' for c in checks) else 'failed',
          'checks': checks, 'seed': 20261111, 'dtype': 'float32', 'device': 'cpu',
          'transformers': transformers.__version__, 'torch': torch.__version__,
          'cuda_initialized': torch.cuda.is_initialized(), 'parameters': sum(p.numel() for p in model.parameters()),
          'tiny_config': tiny, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'limits': 'Random tiny model; no real checkpoint, BF16 GPU, ATOM comparison or official acceptance.'}
(OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print('RESULT', result['status'], 'checks',len(checks),'output',OUT,flush=True)
raise SystemExit(0 if result['status']=='passed' else 1)
