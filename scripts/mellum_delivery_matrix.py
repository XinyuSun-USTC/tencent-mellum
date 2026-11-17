"""Single source of truth for final cumulative rows and the delivery launcher."""
ROWS={
 'step0':{'profile':'delivery_step0','slots':32,'parent':None,'points':[]},
 'step01':{'profile':'delivery_step01','slots':32,'parent':'step0','points':[1]},
 'step012':{'profile':'delivery_step012','slots':32,'parent':'step01','points':[1,2]},
 'step0123':{'profile':'delivery_step0123','slots':32,'parent':'step012','points':[1,2,3]},
}
BASELINE_POLICY='r32-v2'
DEFAULT_ROW='step0123'

def slots(profile):
    if profile=='initial_single_reference':return 1
    return next((v['slots'] for v in ROWS.values() if v['profile']==profile),32)


def point_settings(points):
    """Compose independently selected optimization points on the Step0 baseline."""
    import json
    from mellum_execution_profiles import profile_settings
    points=set(points)
    if points-{1,2,3}:raise ValueError('Optimization points must be1/2/3')
    result=profile_settings('r')
    if 1 in points:
        config=json.loads(result['MELLUM_BATCH_CAMPAIGN_CONFIG'])
        config['decode_expert_group_size']=4
        result['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(config,sort_keys=True)
    if 2 in points:result['MELLUM_DIRECT_KV_WRITE']='1'
    if 3 in points:
        before=profile_settings('delivery_step012')
        after=profile_settings('delivery_step0123')
        result.update({k:v for k,v in after.items() if before.get(k)!=v})
    return result
