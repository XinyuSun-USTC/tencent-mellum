"""Validate the reference environment without allocating a model or GPU tensors."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

OUT = Path('/workspace/results/hf_reference_environment')
OUT.mkdir(parents=True, exist_ok=True)
import torch
import transformers
result = {
    'python': sys.executable,
    'prefix': sys.prefix,
    'transformers': transformers.__version__,
    'transformers_file': transformers.__file__,
    'torch': torch.__version__,
    'torch_file': torch.__file__,
    'hip': torch.version.hip,
    'cuda_initialized': torch.cuda.is_initialized(),
    'status': 'incomplete',
}
if transformers.__version__ != '5.10.1' or not str(Path(transformers.__file__).resolve()).startswith('/workspace/venvs/hf-reference/'):
    result['reason'] = 'Pinned Transformers is not installed locally; shared 5.2.0 is not a usable reference.'
    (OUT / 'environment_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(2)
from transformers import AutoConfig, MellumConfig, MellumForCausalLM
config = AutoConfig.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking', local_files_only=True, trust_remote_code=False)
assert isinstance(config, MellumConfig)
assert len(config.layer_types) == 28
assert config.layer_types.count('sliding_attention') == 21
assert config.layer_types.count('full_attention') == 7
assert config.mlp_layer_types == ['sparse'] * 28
assert config.rope_parameters['full_attention']['rope_type'] == 'yarn'
assert not torch.cuda.is_initialized()
result.update(status='config_and_import_passed', config_class=type(config).__name__, model_class=MellumForCausalLM.__name__)
result['packages'] = {n: metadata.version(n) for n in ['transformers', 'torch', 'huggingface-hub', 'tokenizers', 'safetensors']}
(OUT / 'environment_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
