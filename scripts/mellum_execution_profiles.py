"""Explicit, default-off serving profiles for existing measured execution routes."""
import json

FAST_PROFILES=('fast_resident','fast_norm','fast_packed','fast_combined')
PROFILES=FAST_PROFILES+('r','fused_moe','r_swa_ck','fused_moe_swa_ck','native_attention','fused_moe_native_attention','isolated_moe','fused_moe_native_batched','fused_moe_native_batched_direct_kv')


def profile_settings(name):
    if name not in PROFILES:raise ValueError('Unknown Mellum execution profile')
    fast=name in FAST_PROFILES
    fused=fast or name in ('fused_moe','fused_moe_swa_ck','fused_moe_native_attention','isolated_moe','fused_moe_native_batched','fused_moe_native_batched_direct_kv')
    native=fast or name in ('native_attention','fused_moe_native_attention','fused_moe_native_batched','fused_moe_native_batched_direct_kv')
    swa=name in ('r_swa_ck','fused_moe_swa_ck')
    control={'groups':'' if fast or name in ('fused_moe_native_batched','fused_moe_native_batched_direct_kv') else 'L' if fused else 'LE','phases':['prefill','decode']}
    if not fused:control['moe_optimization']='shared_routing'
    result={'MELLUM_DIRECT_KV_WRITE':'1' if fast or name=='fused_moe_native_batched_direct_kv' else '0',
            'MELLUM_MOE_BACKEND':'fused_isolated' if name=='isolated_moe' else 'fused_torch_route' if fused else 'torch',
            'MELLUM_ROUTER_BACKEND':'torch','MELLUM_PREFILL_BACKEND':'sdpa_cpu_rope_diagnostic',
            'MELLUM_REFERENCE_KV_ROPE':'1','MELLUM_SDPA_DECODE':'0' if native else '1',
            'MELLUM_REQUEST_BATCHING':'0' if native else '1',
            'MELLUM_ATTENTION_EXECUTION':'native' if native else 'reference',
            'MELLUM_SWA_PREFILL_OPT':'1' if swa else '0','MELLUM_SWA_PREFILL_IMPL':'ck',
            'MELLUM_BATCH_CAMPAIGN_CONFIG':json.dumps(control,sort_keys=True)}

    if fast:
        result.update(MELLUM_NORM_BACKEND='aiter' if name in ('fast_norm','fast_combined') else 'reference',
            MELLUM_QKV_BACKEND='packed' if name in ('fast_packed','fast_combined') else 'separate',
            MELLUM_RESIDENT_ROPE='1' if name in ('fast_resident','fast_combined') else '0')
    return result
