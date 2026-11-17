"""Final dual-mode configuration, composed from the measured execution profiles."""
import json
MODES=('performance','precision')
ROWS={'step0':(), 'step01':(1,), 'step012':(1,2), 'step0123':(1,2,3)}
DEFAULT_MODE='performance'

def validate(mode,points):
    points=tuple(sorted(set(points)))
    if mode not in MODES or set(points)-{1,2,3}:raise ValueError('Unknown mode or optimization point')
    if mode=='performance' and (2 in points or 3 in points) and 1 not in points:
        raise ValueError('Performance points 2/3 require native computation point 1; disable dependent points too, or use --row step0')
    return points

def profile_name(mode,points):
    points=validate(mode,points)
    return 'submission_'+mode+'_'+(''.join(map(str,points)) or '0')

PROFILES=tuple(profile_name(mode,p) for mode in MODES for p in [(),(1,),(2,),(3,),(1,2),(1,3),(2,3),(1,2,3)]
               if mode!='performance' or not (set(p)&{2,3}) or 1 in p)

def settings(mode,points):
    from mellum_execution_profiles import profile_settings
    points=validate(mode,points)
    result=profile_settings('r')
    if mode=='precision':
        if 1 in points:
            config=json.loads(result['MELLUM_BATCH_CAMPAIGN_CONFIG']);config['decode_expert_group_size']=4
            result['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(config,sort_keys=True)
        if 2 in points:result['MELLUM_DIRECT_KV_WRITE']='1'
        if 3 in points:result.update(MELLUM_DECODE_META_ONCE='1',MELLUM_SWA_DECODE_GROUP='4')
        return result
    # Apply measured incremental dictionaries; point 2 depends on native point 1.
    native=profile_settings('sprint_clean');native.update(MELLUM_DIRECT_KV_WRITE='0',MELLUM_RESIDENT_ROPE='0')
    runtime=profile_settings('sprint_graph_qk_router')
    prefix=profile_settings('sprint_graph_qk_router_prefix_metadata')
    if 1 in points:result=native
    if 2 in points:result.update({k:v for k,v in runtime.items() if native.get(k)!=v})
    if 3 in points:result.update({k:v for k,v in prefix.items() if runtime.get(k)!=v})
    return result

def from_profile(name):
    _,mode,code=name.split('_')
    return settings(mode,() if code=='0' else tuple(map(int,code)))
