"""CPU-only support audit; no model construction or checkpoint tensor reads."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import transformers
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

model = Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
config = json.loads((model / 'config.json').read_text())
result = {
    'executable': sys.executable,
    'prefix': sys.prefix,
    'sys_path': sys.path,
    'transformers_metadata_version': importlib.metadata.version('transformers'),
    'transformers_runtime_version': transformers.__version__,
    'transformers_file': transformers.__file__,
    'config_mapping': CONFIG_MAPPING_NAMES.get('mellum'),
    'causal_lm_mapping': MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get('mellum'),
    'model_config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
    'model_identity': {k: config.get(k) for k in ['model_type', 'architectures', 'transformers_version', 'auto_map']},
    'checkpoint_python_files': sorted(p.name for p in model.glob('*.py')),
    'api_probe': {},
}
try:
    result['class_import'] = str(getattr(transformers, 'MellumForCausalLM'))
except Exception as error:
    result['class_import'] = {'error': type(error).__name__, 'message': str(error)}
try:
    cfg = AutoConfig.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    result['auto_config'] = {'class': type(cfg).__name__}
except Exception as error:
    result['auto_config'] = {'error': type(error).__name__, 'message': str(error)}
for module, names in {
    'transformers.configuration_utils': ['PreTrainedConfig'],
    'huggingface_hub.dataclasses': ['strict'],
    'transformers.integrations': ['use_experts_implementation', 'use_kernel_forward_from_hub', 'use_kernel_func_from_hub', 'use_kernelized_func'],
    'transformers.utils.generic': ['maybe_autocast', 'merge_with_config_defaults'],
    'transformers.utils.output_capturing': ['OutputRecorder', 'capture_outputs'],
    'transformers.modeling_utils': ['ALL_ATTENTION_FUNCTIONS', 'PreTrainedModel'],
    'transformers.cache_utils': ['DynamicCache'],
}.items():
    try:
        imported = importlib.import_module(module)
        for name in names:
            value = getattr(imported, name, None)
            entry = {'present': value is not None}
            if name == 'PreTrainedConfig':
                entry['has_post_init'] = hasattr(value, '__post_init__')
            if name == 'DynamicCache':
                entry['init_signature'] = str(inspect.signature(value.__init__))
            result['api_probe'][module + '.' + name] = entry
    except Exception as error:
        result['api_probe'][module] = {'error': type(error).__name__, 'message': str(error)}
import torch
result['cuda_initialized_at_end'] = torch.cuda.is_initialized()
print(json.dumps(result, ensure_ascii=False, indent=2))
