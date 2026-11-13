"""Shared explicit Step0 runtime configuration for profiling and future serving."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODEL = Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
CANDIDATE = {
    'MELLUM_HF_GENERATION_EXPERTS': '1', 'MELLUM_MOE_BACKEND': 'torch',
    'MELLUM_QKV_BACKEND': 'separate', 'MELLUM_ROUTER_BACKEND': 'torch',
    'MELLUM_PREFILL_BACKEND': 'sdpa_cpu_rope_diagnostic',
    'MELLUM_REFERENCE_KV_ROPE': '1', 'MELLUM_SDPA_DECODE': '1',
    'MELLUM_STAGE_LAYER': '0', 'MELLUM_DECODE_LAYER_CAPTURE': '0',
    'ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION': '0',
    'ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION': '0', 'ATOM_USE_UNIFIED_ATTN': '0',
    'ATOM_LOADER_USE_THREADPOOL': '0', 'AITER_LOG_LEVEL': 'WARNING',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
}


def apply_candidate(gpu, output, detailed=False):
    # Clear inherited diagnostic/intervention settings before spawning workers.
    for key in list(os.environ):
        if key.startswith('MELLUM_'):
            del os.environ[key]
    settings = {**CANDIDATE, 'HIP_VISIBLE_DEVICES': str(gpu),
                'PYTHONPATH': str(PROJECT / 'ATOM'),
                'MELLUM_LOAD_AUDIT_PATH': str(Path(output) / 'loading_audit.json'),
                'MELLUM_PROFILE_LABELS': '1',
                'MELLUM_PROFILE_CONTROL_DIR': str(Path(output) / 'control'),
                'ATOM_PROFILER_MORE': '1' if detailed else '0'}
    os.environ.update(settings)
    return settings


def engine_options(required_tokens, trace_dir):
    max_len = max(256, ((required_tokens + 15) // 16) * 16)
    return dict(model=str(MODEL), tensor_parallel_size=1, data_parallel_size=1,
                enforce_eager=True, level=0, kv_cache_dtype='bf16',
                max_model_len=max_len, max_num_batched_tokens=max_len,
                max_num_seqs=1, gpu_memory_utilization=0.35,
                enable_prefix_caching=False, enable_chunked_prefill=False,
                torch_profiler_dir=str(trace_dir), mark_trace=False)
