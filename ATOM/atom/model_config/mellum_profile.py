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
    if torch.cuda.is_initialized():
        try:
            data['allocator'] = dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peaks_reset_at_profile_start=os.getenv('MELLUM_PROFILE_MEMORY_STATS')=='1')
        except Exception as error:
            data['allocator_error'] = str(error)
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
    if os.getenv('MELLUM_PROFILE_GRANULAR', '0') == '1':
        from functools import wraps
        from atom.model_config import mellum_attention

        class Restore:
            def __init__(self, owner, name, previous):
                self.owner, self.name, self.previous = owner, name, previous
            def remove(self):
                setattr(self.owner, self.name, self.previous)

        def wrap(owner, name, label):
            original = getattr(owner, name)
            @wraps(original)
            def observed(*args, **kwargs):
                with torch.profiler.record_function(label):
                    return original(*args, **kwargs)
            setattr(owner, name, observed)
            handles.append(Restore(owner, name, original))

        for name in ['reference_position_rope', 'overwrite_reference_keys', 'gather_reference_kv']:
            wrap(mellum_attention, name, f'mellum/{name}/shared')
        wrap(torch.nn.functional, 'scaled_dot_product_attention', 'mellum/sdpa/shared')
        from atom.model_config import mellum_batch, mellum_grouped_decode
        wrap(mellum_batch, 'sdpa_cached_decode', 'mellum/requestwise_decode/shared')
        wrap(mellum_grouped_decode, 'grouped_swa', 'mellum/grouped_swa/shared')
    return handles
