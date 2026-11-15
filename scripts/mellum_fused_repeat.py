"""Fixed-input/route layer0 replay, not a full-model correctness acceptance."""
import importlib,json,os,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
from mellum_runtime_config import PROJECT,MODEL
from run_mellum_batch_campaign import write,sha


def callable_info(fn):
    core=getattr(fn,'func',fn)
    return {'callable':getattr(core,'__module__','')+'.'+getattr(core,'__qualname__',type(core).__name__),
            'keywords':{str(k):str(v) for k,v in getattr(fn,'keywords',{}).items()}}


def worker(out,plan):
    os.environ.update(HIP_VISIBLE_DEVICES='0',AITER_LOG_LEVEL='WARNING',AITER_USE_CK_MOE_SORTING='1' if plan['sorting']=='ck' else '0')
    sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch,transformers
    from safetensors import safe_open
    from atom.model_config.mellum_experts import MellumTorchExperts
    from atom.model_config.mellum_native import hf_route,isolated_fused_experts
    from atom.model_ops.utils import shuffle_weights
    from aiter import ActivationType,QuantType
    fm=importlib.import_module('aiter.fused_moe')
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    if bool(fm._USE_CK_MOE_SORTING)!=(plan['sorting']=='ck'):raise RuntimeError('Requested sorting mode not selected')
    if transformers.__version__!='5.2.0' or torch.__version__!=plan['provenance']['reference_environment']['torch']:
        raise RuntimeError('Frozen runtime differs')
    if torch.cuda.mem_get_info(0)[0]<8*1024**3:raise RuntimeError('Require 8GiB free for layer replay')
    torch.set_grad_enabled(False)
    for item in plan['micro_inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Replay inputs changed')
    config=SimpleNamespace(**json.loads((MODEL/'config.json').read_text()))
    with torch.device('meta'):canonical=MellumTorchExperts(config)
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    def weight(expert,name):
        key=f'model.layers.0.mlp.experts.{expert}.{name}_proj.weight'
        with safe_open(MODEL/index[key],framework='pt',device='cpu') as f:return f.get_tensor(key)
    canonical.w13_weight=torch.nn.Parameter(torch.stack([torch.cat([weight(e,'gate'),weight(e,'up')]) for e in range(config.num_experts)]).cuda(),requires_grad=False)
    canonical.w2_weight=torch.nn.Parameter(torch.stack([weight(e,'down') for e in range(config.num_experts)]).cuda(),requires_grad=False)
    w13=torch.nn.Parameter(canonical.w13_weight.detach().clone(),requires_grad=False)
    w2=torch.nn.Parameter(canonical.w2_weight.detach().clone(),requires_grad=False)
    shuffle_weights(w13,w2)
    source=Path(plan['micro_inputs'][0]['path']).parent.parent
    samples=[(torch.load(source/f'sample_{i:06d}/moe_input.pt',map_location='cpu',weights_only=True).cuda(),
              torch.load(source/f'sample_{i:06d}/router_logits.pt',map_location='cpu',weights_only=True).cuda()) for i in range(4)]
    lengths=[7,19,31,11,23,17,29,13]
    cases=[('prefill_ragged',torch.cat([samples[i%4][0][:n] for i,n in enumerate(lengths)]),
             torch.cat([samples[i%4][1][:n] for i,n in enumerate(lengths)]),False)]
    for count in (1,32):
        cases.append((f'decode_{count}',torch.cat([samples[i%4][0][-1:] for i in range(count)]),
                      torch.cat([samples[i%4][1][-1:] for i in range(count)]),True))
    result={'status':'started','sorting':plan['sorting'],'variant':plan.get('variant','native'),'records':[],
            'frozen_source_sha256':{str(Path(fm.__file__)):sha(fm.__file__)},
            'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(0)},
            'scope':'Layer0 saved HF prefill vectors and decode-shaped last-row replay; fixed route. Not natural decode trajectories or full-model acceptance.'}
    write(out/'results.json',result)
    original_cfg=fm.get_2stage_cfgs;original_sort=fm.moe_sorting
    selected=[];sorting=[]
    def cfg(*args,**kwargs):
        value=original_cfg(*args,**kwargs)
        info={'token_key':int(args[0]),'run_1stage':bool(value.run_1stage),'block_m':int(value.block_m),
              'ksplit':int(value.ksplit),'stage1':callable_info(value.stage1),'stage2':callable_info(value.stage2)}
        if info not in selected:selected.append(info)
        return value
    def sorted_call(*args,**kwargs):
        value=original_sort(*args,**kwargs)
        counts=value[3].detach().cpu().tolist();n=int(counts[0])
        if not 0<=n<=value[0].numel():raise ValueError('Sorting valid count outside output buffer')
        # Only initialized token-id prefix, never uninitialized padding tail/weights.
        digest=hashlib.sha256(value[0][:n].detach().cpu().numpy().tobytes()).hexdigest()
        sorting.append({'num_valid_ids':counts,'token_ids_sha256':digest})
        return value
    fm.get_2stage_cfgs=cfg;fm.moe_sorting=sorted_call
    try:
        for name,x,logits,batched in cases:
            weights,ids=hf_route(x,logits,config.num_experts_per_tok,config.norm_topk_prob)
            frozen=(x.clone(),weights.clone(),ids.clone())
            def call():
                if plan.get('variant')=='isolated':return isolated_fused_experts(x,weights,ids,w13,w2)
                return fm.fused_moe(x,w13,w2,weights,ids,activation=ActivationType.Silu,quant_type=QuantType.No)
            call();torch.cuda.synchronize();selected.clear();sorting.clear()
            reference=canonical.forward_routed(x,weights.to(x.dtype),ids.long(),batched=batched)
            first=None;changes=[];max_abs=0.;finite=True
            for repeat in range(12):
                value=call();torch.cuda.synchronize()
                finite=finite and bool(torch.isfinite(value).all())
                if first is None:first=value.detach().clone()
                elif not torch.equal(value,first):changes.append(repeat)
                max_abs=max(max_abs,float((value.float()-first.float()).abs().max()))
            if not all(torch.equal(a,b) for a,b in zip((x,weights,ids),frozen)):raise ValueError('Fixed replay inputs mutated')
            delta=first.float()-reference.float()
            record={'name':name,'rows':len(x),'phase':'decode' if batched else 'prefill','repeats':12,
                    'different_repeat_indices':changes,'repeats_exact':not changes,'repeat_max_abs':max_abs,'finite':finite,
                    'vs_reference':{'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/reference.float().norm().clamp_min(1e-12))},
                    'selected_metadata':list(selected),'sorting_records':list(sorting),
                    'sorting_order_repeated':all(r['token_ids_sha256']==sorting[0]['token_ids_sha256'] and r['num_valid_ids'][0]==sorting[0]['num_valid_ids'][0] for r in sorting)}
            torch.save({'first':first.cpu(),'reference':reference.cpu(),'input':x.cpu(),'weights':weights.cpu(),'ids':ids.cpu()},out/(name+'.pt'))
            result['records'].append(record);write(out/'results.json',result)
            print(name,'repeats_exact',not changes,'repeat_max_abs',max_abs,'selected',selected,flush=True)
        # One bounded true GPU trace, separate from repeat checks and not used as timing.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            call();torch.cuda.synchronize()
        kernels=sorted({e.name for e in profiler.events() if str(e.device_type).endswith('CUDA')})
        if not kernels:raise RuntimeError('No actual GPU kernels observed')
        profiler.export_chrome_trace(str(out/'decode32_trace.json.gz'))
        result.update(status='complete',kernel_names=kernels,all_fixed_repeats_exact=all(r['repeats_exact'] for r in result['records']))
        write(out/'results.json',result)
    finally:
        fm.get_2stage_cfgs=original_cfg;fm.moe_sorting=original_sort
