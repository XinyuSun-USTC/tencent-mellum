"""Explicit, default-off serving profiles for existing measured execution routes."""
import json

FAST_PROFILES=('fast_resident','fast_norm','fast_packed','fast_combined','sprint_clean','sprint_graph','sprint_unpacked','sprint_router','sprint_graph_router','sprint_graph_rope','sprint_graph_qk','sprint_graph_prefix','sprint_graph_chunk2048','sprint_graph_chunk4096')
DERIVED={
    'sprint_graph_qk_router_prefix':{'MELLUM_NATIVE_ROUTING':'1','MELLUM_PREFIX_COMPAT':'1','MELLUM_PREFIX_ENABLED':'1'},
    'sprint_graph_qk_router_prefix_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'1','MELLUM_NATIVE_ROUTING':'1','MELLUM_PREFIX_COMPAT':'1','MELLUM_PREFIX_ENABLED':'1'},
    'sprint_graph_qk_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'1'},
    'sprint_graph_qk_router_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'1','MELLUM_NATIVE_ROUTING':'1'},
    'sprint_graph_qk_prefix_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'1','MELLUM_PREFIX_COMPAT':'1','MELLUM_PREFIX_ENABLED':'1'},
    'sprint_graph_qk_chunk2048_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'1','MELLUM_PREFIX_COMPAT':'1','MELLUM_CHUNK_BUDGET':'2048'},
    'sprint_graph_qk_chunk4096_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'1','MELLUM_PREFIX_COMPAT':'1','MELLUM_CHUNK_BUDGET':'4096'},
    'sprint_graph_qk_router':{'MELLUM_NATIVE_ROUTING':'1'},
    'sprint_graph_qk_prefix':{'MELLUM_PREFIX_COMPAT':'1','MELLUM_PREFIX_ENABLED':'1'},
    'sprint_graph_qk_chunk2048':{'MELLUM_PREFIX_COMPAT':'1','MELLUM_CHUNK_BUDGET':'2048'},
    'sprint_graph_qk_chunk4096':{'MELLUM_PREFIX_COMPAT':'1','MELLUM_CHUNK_BUDGET':'4096'},
}
ABLATION_BASE='sprint_graph_qk_router_prefix_metadata'
ABLATIONS={
    'ablate_graph':{'MELLUM_DECODE_GRAPH':'0'},
    'ablate_qk':{'MELLUM_FUSED_QK_ROPE':'0'},
    'ablate_route':{'MELLUM_NATIVE_ROUTING':'0'},
    'ablate_prefix':{'MELLUM_PREFIX_ENABLED':'0'},
    'ablate_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'0'},
    'ablate_packed':{'MELLUM_QKV_BACKEND':'separate'},
    'ablate_norms':{'MELLUM_NORM_BACKEND':'reference','MELLUM_FUSED_QK_ROPE':'0'},
}
PROFILES=tuple(ABLATIONS)+tuple(DERIVED)+FAST_PROFILES+('r','fused_moe','r_swa_ck','fused_moe_swa_ck','native_attention','fused_moe_native_attention','isolated_moe','fused_moe_native_batched','fused_moe_native_batched_direct_kv')


def profile_settings(name):
    if name not in PROFILES:raise ValueError('Unknown Mellum execution profile')
    if name in ABLATIONS:
        result=profile_settings(ABLATION_BASE);result.update(ABLATIONS[name]);return result
    if name in DERIVED:
        result=profile_settings('sprint_graph_qk');result.update(DERIVED[name]);return result
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

    if name.startswith('sprint_'):
        result.update(MELLUM_BATCH_CAMPAIGN_CONFIG='',MELLUM_NORM_BACKEND='aiter',
            MELLUM_QKV_BACKEND='separate' if name=='sprint_unpacked' else 'packed',
            MELLUM_RESIDENT_ROPE='1',MELLUM_NATIVE_ROUTING='1' if name in ('sprint_router','sprint_graph_router') else '0',
            MELLUM_DECODE_GRAPH='1' if name in ('sprint_graph','sprint_graph_router','sprint_graph_rope','sprint_graph_qk','sprint_graph_prefix','sprint_graph_chunk2048','sprint_graph_chunk4096') else '0')
        result.update(MELLUM_FUSED_ROPE='1' if name=='sprint_graph_rope' else '0',
            MELLUM_FUSED_QK_ROPE='1' if name=='sprint_graph_qk' else '0')
        result.update(MELLUM_PREFIX_COMPAT='1' if name in ('sprint_graph_prefix','sprint_graph_chunk2048','sprint_graph_chunk4096') else '0',
            MELLUM_PREFIX_ENABLED='1' if name=='sprint_graph_prefix' else '0',
            MELLUM_CHUNK_BUDGET='2048' if name=='sprint_graph_chunk2048' else '4096' if name=='sprint_graph_chunk4096' else '0')
        return result
    if fast:
        result.update(MELLUM_NORM_BACKEND='aiter' if name in ('fast_norm','fast_combined') else 'reference',
            MELLUM_QKV_BACKEND='packed' if name in ('fast_packed','fast_combined') else 'separate',
            MELLUM_RESIDENT_ROPE='1' if name in ('fast_resident','fast_combined') else '0')
    return result
