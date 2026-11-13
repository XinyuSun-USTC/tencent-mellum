# SPDX-License-Identifier: MIT
"""Opt-in profiling labels and acknowledged worker lifecycle; no tensor edits."""
import json
import os
from pathlib import Path
import time
import torch


def write_profile_receipt(kind, rank, sequence, trace_dir):
    root = os.getenv('MELLUM_PROFILE_CONTROL_DIR')
    if not root:
        return
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'rank_{rank}_{kind}_{sequence:03d}.json'
    data = dict(kind=kind, rank=rank, sequence=sequence, pid=os.getpid(),
                time_ns=time.time_ns(), trace_dir=str(trace_dir),
                traces=[str(p.resolve()) for p in sorted(Path(trace_dir).glob('*.pt.trace.json.gz'))])
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, target)


def install_profile_labels(model):
    """Attach balanced record_function ranges only while a profiler is active."""
    handles = []

    def attach(module, label):
        stack = []

        def before(module, args):
            context = None
            if torch.autograd._profiler_enabled():
                context = torch.profiler.record_function(label)
                context.__enter__()
            stack.append(context)

        def after(module, args, output):
            context = stack.pop()
            if context is not None:
                context.__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))

    for index, layer in enumerate(model.model.layers):
        suffix = f'layer_{index:02d}'
        attn = layer.self_attn
        window = getattr(attn, 'sliding_window', None)
        # MellumAttention stores its effective window in the implementation.
        if window is None:
            window = getattr(getattr(attn, 'attn', None), 'impl', None)
            window = getattr(window, 'sliding_window', 0)
        attach(attn, f'mellum/attention_{"swa" if window and window > 0 else "full"}/{suffix}')
        attach(layer.mlp.experts, f'mellum/moe_experts/{suffix}')
        attach(layer.mlp.gate, f'mellum/router/{suffix}')
        attach(layer.input_layernorm, f'mellum/norm_input/{suffix}')
        attach(layer.post_attention_layernorm, f'mellum/norm_post/{suffix}')
        attach(attn.qkv_proj, f'mellum/qkv/{suffix}')
        attach(attn.o_proj, f'mellum/output_projection/{suffix}')
    return handles
