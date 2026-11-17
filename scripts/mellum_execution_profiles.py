"""Explicit, default-off serving profiles for existing measured execution routes."""
import json
from mellum_submission_modes import PROFILES as SUBMISSION_PROFILES

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
AGREEMENT={
    'agree_qk_round':{'MELLUM_HF_QK_ROUNDING':'1'},
    'agree_norms':{'MELLUM_HF_QK_ROUNDING':'1','MELLUM_NORM_BACKEND':'reference'},
    'agree_norms_route':{'MELLUM_HF_QK_ROUNDING':'1','MELLUM_NORM_BACKEND':'reference','MELLUM_NATIVE_ROUTING':'0'},
    'agree_no_prefix':{'MELLUM_PREFIX_ENABLED':'0'},
    'agree_route':{'MELLUM_NATIVE_ROUTING':'0'},
    'agree_projection':{'MELLUM_QKV_BACKEND':'separate'},
}

CORE_PROFILES=('core_nn','core_nt','core_tn','core_tt')

PHASE_PROFILES=('phase_prefill','phase_decode','expert_prefill','expert_decode')
PROFILES=SUBMISSION_PROFILES+('initial_single_reference','delivery_step0','delivery_step01','delivery_step012','delivery_step0123')+PHASE_PROFILES+CORE_PROFILES+tuple(AGREEMENT)+tuple(DERIVED)+FAST_PROFILES+('r_swa4_abort','r_swa4_abort_prefix','r_swa4_abort_prefix_chunk1024_d4','r_abort','r_decode_meta','r_swa_decode2','r_swa_decode4','r_prefix_only','r_prefix_chunk1024_d1','r_prefix_chunk2048_d1','r_chunk1024_d1','r_chunk2048_d1','r_chunk512_d4','r_chunk1024_d4','r_chunk2048_d4','r_expert_decode4_kv_once','r_expert_decode2','r_expert_decode4','r_reference_kv_once','r','fused_moe','r_swa_ck','fused_moe_swa_ck','native_attention','fused_moe_native_attention','isolated_moe','fused_moe_native_batched','fused_moe_native_batched_direct_kv')


def profile_settings(name):
    if name not in PROFILES:raise ValueError('Unknown Mellum execution profile')
    if name in SUBMISSION_PROFILES:
        from mellum_submission_modes import from_profile
        return from_profile(name)
    if name=='initial_single_reference':
        result=profile_settings('r');result.update(MELLUM_BATCH_CAMPAIGN_CONFIG='',MELLUM_REQUEST_BATCHING='1');return result
    if name.startswith('delivery_step'):
        if name=='delivery_step0':
            return profile_settings('r')
        return profile_settings({'delivery_step01':'r_expert_decode4',
            'delivery_step012':'r_expert_decode4_kv_once',
            'delivery_step0123':'r_swa4_abort_prefix_chunk1024_d4'}[name])

    if name in ('r_swa4_abort','r_swa4_abort_prefix','r_swa4_abort_prefix_chunk1024_d4'):
        result=profile_settings('r_swa_decode4')
        result['MELLUM_ABORT_ON_DISCONNECT']='1'
        if name!='r_swa4_abort':result.update(MELLUM_REFERENCE_PREFIX='1',MELLUM_PREFIX_ENABLED='1')
        if name.endswith('chunk1024_d4'):result.update(MELLUM_REFERENCE_CHUNKED='1',MELLUM_CHUNK_BUDGET='1024',MELLUM_C_DECODE_STEPS='4')
        return result
    if name=='r_abort':
        result=profile_settings('r_expert_decode4_kv_once')
        result['MELLUM_ABORT_ON_DISCONNECT']='1'
        return result
    if name in ('r_decode_meta','r_swa_decode2','r_swa_decode4'):
        result=profile_settings('r_expert_decode4_kv_once')
        result['MELLUM_DECODE_META_ONCE']='1'
        if name!='r_decode_meta':result['MELLUM_SWA_DECODE_GROUP']=name[-1]
        return result
    if name in ('r_prefix_only','r_prefix_chunk1024_d1','r_prefix_chunk2048_d1'):
        parent='r_expert_decode4_kv_once' if name=='r_prefix_only' else name.replace('r_prefix_chunk','r_chunk')
        result=profile_settings(parent)
        result.update(MELLUM_REFERENCE_PREFIX='1',MELLUM_PREFIX_ENABLED='1')
        return result
    if name in ('r_chunk1024_d1','r_chunk2048_d1'):
        result=profile_settings(name[:-1]+'4')
        result['MELLUM_C_DECODE_STEPS']='1'
        return result
    if name in ('r_chunk512_d4','r_chunk1024_d4','r_chunk2048_d4'):
        result=profile_settings('r_expert_decode4_kv_once')
        result.update(MELLUM_REFERENCE_CHUNKED='1',MELLUM_CHUNK_BUDGET=name.split('_')[1][5:],MELLUM_C_DECODE_STEPS='4')
        return result
    if name=='r_expert_decode4_kv_once':
        result=profile_settings('r_expert_decode4')
        result['MELLUM_DIRECT_KV_WRITE']='1'
        return result
    if name in ('r_expert_decode2','r_expert_decode4'):
        result=profile_settings('r')
        config=json.loads(result['MELLUM_BATCH_CAMPAIGN_CONFIG'])
        config['decode_expert_group_size']=int(name[-1])
        result['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(config,sort_keys=True)
        return result
    if name=='r_reference_kv_once':
        result=profile_settings('r')
        result['MELLUM_DIRECT_KV_WRITE']='1'
        return result
    if name in PHASE_PROFILES:
        if name.startswith('expert_'):
            result=profile_settings('core_tn')
            result.update(MELLUM_MOE_BACKEND='phase_hybrid',MELLUM_NATIVE_EXPERT_PHASE='prefill' if name=='expert_prefill' else 'decode')
        else:
            result=profile_settings('core_nt')
            result['MELLUM_ATTENTION_EXECUTION']='native_prefill' if name=='phase_prefill' else 'native_decode'
        return result
    if name in CORE_PROFILES:
        native_attention=name[5]=='n';native_experts=name[6]=='n'
        result=profile_settings('agree_norms_route')
        result.update(MELLUM_DECODE_GRAPH='0',MELLUM_QKV_BACKEND='separate',MELLUM_FUSED_QK_ROPE='0',
            MELLUM_PREFIX_ENABLED='0',MELLUM_PREFIX_COMPAT='0',MELLUM_PREFILL_METADATA_REUSE='0',
            MELLUM_MOE_BACKEND='fused_torch_route' if native_experts else 'torch',
            MELLUM_ATTENTION_EXECUTION='native' if native_attention else 'reference',
            MELLUM_SDPA_DECODE='0' if native_attention else '1',MELLUM_REQUEST_BATCHING='0' if native_attention else '1',
            MELLUM_DIRECT_KV_WRITE='1' if native_attention else '0')
        return result
    if name in AGREEMENT:
        result=profile_settings('sprint_graph_qk_router_prefix_metadata');result.update(AGREEMENT[name]);return result
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
