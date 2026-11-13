# SPDX-License-Identifier: MIT
"""Opt-in observers and explicitly requested MoE input replay diagnostics."""
from pathlib import Path
import os
import torch


def make_replay_router(mode, native_router=None):
    """Controlled routing intervention; weights stay FP32 for AITER's ABI."""
    if mode not in ('rounded_native', 'hf'):
        raise ValueError(mode)
    def route(hidden_states, gating_output, topk, renormalize):
        if mode == 'rounded_native':
            weights, ids = native_router(gating_output=gating_output, topk=topk,
                                         renormalize=renormalize)
        else:
            weights, ids = torch.topk(gating_output.float().softmax(-1), topk, dim=-1)
            if renormalize:
                weights = weights / weights.sum(-1, keepdim=True)
        return weights.to(gating_output.dtype).float(), ids.to(torch.int32)
    return route


def install_moe_input_replay(module, capture_path, source):
    """Diagnostic intervention: replace only observed prefill MLP inputs."""
    source = Path(source)
    def replay(module, args):
        path = capture_path()
        if path is None:
            return None
        value = torch.load(source / Path(path).name / 'moe_input.pt',
                           map_location='cpu', weights_only=True)
        if value.shape != args[0].shape or value.dtype != args[0].dtype:
            raise ValueError('MoE replay input shape/dtype differs from current input')
        return (value.to(args[0].device), *args[1:])
    return module.register_forward_pre_hook(replay)


def install_stage_hooks(layer, capture_path, backend):
    """capture_path returns a sample path or None. Observe only layer zero."""
    handles = []
    if backend == 'atom' and os.getenv('MELLUM_MOE_REPLAY_INPUT_DIR'):
        handles.append(install_moe_input_replay(layer.mlp, capture_path,
                       os.environ['MELLUM_MOE_REPLAY_INPUT_DIR']))
        mode = os.getenv('MELLUM_MOE_REPLAY_ROUTING', 'native')
        if mode != 'native':
            from atom.model_ops.topK import rocm_aiter_topk_softmax
            layer.mlp.experts.custom_routing_function = make_replay_router(
                mode, rocm_aiter_topk_softmax)
    attn = layer.self_attn

    def save(name, value, heads=False):
        path = capture_path()
        if path is None:
            return
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if heads:
            value = value.reshape(-1, value.shape[-2] * value.shape[-1])
        else:
            value = value.reshape(-1, value.shape[-1])
        torch.save(value.detach().cpu(), path / (name + '.pt'))

    def weight(name, value):
        path = capture_path()
        if path is None:
            return
        target = Path(path).parent / 'weights' / (name + '.pt')
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value.detach().cpu(), target)

    def normal_hook(name):
        def observe(module, args, output):
            if capture_path() is None:
                return
            raw = args[0]
            if len(args) > 1 and args[1] is not None:
                raw = raw + args[1]
            save(name + '_input', raw)
            save(name, output[0] if isinstance(output, tuple) else output)
            weight(name, module.weight)
        return observe

    handles.append(layer.input_layernorm.register_forward_hook(normal_hook('input_norm')))
    handles.append(layer.post_attention_layernorm.register_forward_hook(normal_hook('post_norm')))
    for name in ['q_norm', 'k_norm']:
        def observe(module, args, output, name=name):
            save(name, output, heads=True)
            if capture_path() is not None:
                weight(name, module.weight)
        handles.append(getattr(attn, name).register_forward_hook(observe))
    if backend == 'atom':
        def packed(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            for name, chunk, w in zip(['q_proj', 'k_proj', 'v_proj'],
                value.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1),
                module.weight.split([attn.q_size, attn.kv_size, attn.kv_size], dim=0)):
                save(name, chunk)
                weight(name, w)
        handles.append(attn.qkv_proj.register_forward_hook(packed))
    else:
        for name in ['q_proj', 'k_proj', 'v_proj']:
            def project(module, args, output, name=name):
                save(name, output)
                weight(name, module.weight)
            handles.append(getattr(attn, name).register_forward_hook(project))
    def output_project(module, args, output):
        save('attention_context', args[0])
        save('attention_output', output[0] if isinstance(output, tuple) else output)
        weight('o_proj', module.weight)
    handles.append(attn.o_proj.register_forward_hook(output_project))
    def router(module, args, output):
        save('router_logits', output[0] if isinstance(output, tuple) else output)
        weight('router', module.weight)
    handles.append(layer.mlp.gate.register_forward_hook(router))
    def mlp(module, args, output):
        save('moe_input', args[0])
        save('moe_output', output)
    handles.append(layer.mlp.register_forward_hook(mlp))
    return handles
