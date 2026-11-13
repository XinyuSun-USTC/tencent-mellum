"""Use real frozen HF configuration code; avoid engine/GPU construction."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import AutoConfig, PretrainedConfig
from atom.model_config.mellum import MellumConfig

MODEL = Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')

@pytest.fixture
def raw():
    return json.loads((MODEL/'config.json').read_text())

def test_real_config_and_roundtrip(raw, tmp_path):
    cfg = MellumConfig.from_dict(raw)
    for key, value in raw.items():
        if key not in {'dtype', 'transformers_version'}:
            assert getattr(cfg, key) == value, key
    assert str(cfg.dtype) == 'torch.bfloat16'
    cfg.save_pretrained(tmp_path)
    restored = MellumConfig.from_pretrained(tmp_path, local_files_only=True)
    for key in raw:
        if key != 'transformers_version':
            assert getattr(restored, key) == getattr(cfg, key), key
    assert raw['rope_parameters']['full_attention']['attention_factor'] == 1.2772588722239782

@pytest.mark.parametrize('change', [
    {'head_dim': 127}, {'num_attention_heads': 31}, {'num_experts_per_tok':65},
    {'layer_types':['full_attention']}, {'mlp_layer_types':['dense']*28},
    {'sliding_window':0}, {'num_hidden_layers':0}, {'num_mtp_layers':1},
    {'rope_parameters':{'full_attention':{'rope_type':'unknown'}}},
])
def test_invalid_config(raw, change):
    raw.update(change)
    with pytest.raises(ValueError): MellumConfig.from_dict(raw)

def test_nested_input_not_mutated(raw):
    before=copy.deepcopy(raw)
    cfg=MellumConfig.from_dict(raw)
    cfg.rope_parameters['full_attention']['factor']=2
    assert raw == before

def test_real_entry_function_and_existing_branch(raw):
    # Execute the actual function AST with real HF classes, without importing
    # atom.config's GPU/distributed dependencies (conftest stubs that module).
    source=Path(__file__).parents[1]/'atom/config.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_hf_config')
    ns={'PretrainedConfig':PretrainedConfig,'AutoConfig':AutoConfig,
        '_CONFIG_REGISTRY':{},'_MULTIMODAL_MODEL_TYPES':{},
        'is_vllm':lambda:False}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    with patch.object(PretrainedConfig,'get_config_dict',return_value=(raw,{})), patch.object(AutoConfig,'from_pretrained',side_effect=AssertionError('Mellum must not reach AutoConfig')):
        assert isinstance(ns['get_hf_config']('local'),MellumConfig)
    with patch.object(PretrainedConfig,'get_config_dict',return_value=({'model_type':'llama'},{})), patch.object(AutoConfig,'from_pretrained',return_value='existing') as original:
        assert ns['get_hf_config']('local') == 'existing'
        original.assert_called_once()
