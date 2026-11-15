"""Real layer-zero MoE micro study, isolated from the full-model workers."""
import json,os,sys,statistics
from pathlib import Path
from types import SimpleNamespace
from mellum_runtime_config import PROJECT,MODEL

SOURCE=Path('/workspace/results/mellum_validation/20261111T140735437405Z/hf/stages')


def inputs():
    from run_mellum_batch_campaign import sha
    return [{'path':str(SOURCE/f'sample_{i:06d}'/n),'sha256':sha(SOURCE/f'sample_{i:06d}'/n)}
            for i in range(4) for n in ('moe_input.pt','router_logits.pt')]


def worker(out,plan):
    os.environ['HIP_VISIBLE_DEVICES']=str(plan['gpu']);sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch,transformers
    from safetensors import safe_open
    from atom.model_config.mellum_experts import MellumTorchExperts
    from atom.model_config.mellum_moe_optimization import optimized_experts
    from atom.model_config.mellum_indexed_gemv import indexed_linear
    from run_mellum_batch_campaign import write,sha
    from diagnose_mellum_expert_math import error
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    if torch.__version__!=plan['provenance']['reference_environment']['torch'] or transformers.__version__!='5.2.0':
        raise RuntimeError('Frozen runtime differs')
    if torch.cuda.mem_get_info(0)[0]<8*1024**3:raise RuntimeError('Micro study requires 8GiB free')
    torch.set_grad_enabled(False)
    for item in plan['micro_inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Micro input hash mismatch')
    config=SimpleNamespace(**json.loads((MODEL/'config.json').read_text()))
    with torch.device('meta'):module=MellumTorchExperts(config)
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    def weight(e,name):
        key=f'model.layers.0.mlp.experts.{e}.{name}_proj.weight'
        with safe_open(MODEL/index[key],framework='pt',device='cpu') as stream:return stream.get_tensor(key)
    module.w13_weight=torch.nn.Parameter(torch.stack([torch.cat([weight(e,'gate'),weight(e,'up')]) for e in range(config.num_experts)]).cuda(),requires_grad=False)
    module.w2_weight=torch.nn.Parameter(torch.stack([weight(e,'down') for e in range(config.num_experts)]).cuda(),requires_grad=False)
    # Expert/row/boundary probe: asymmetric values, repeated experts, masked tail.
    # Exact single nonzero coordinate avoids making a numerical tolerance into
    # an acceptance threshold. Also exercises strided input rows and both repeats.
    x=torch.zeros(2,11,device='cuda',dtype=torch.bfloat16);x[:,0]=torch.tensor([1.,2.],device='cuda')
    w=torch.zeros(3,7,11,device='cuda',dtype=torch.bfloat16)
    for e in range(3):w[e,:,0]=torch.arange(1,8,device='cuda')+e*10
    ids=torch.tensor([[2,0],[1,2]],device='cuda')
    probe=indexed_linear(x,ids,w,2)
    expected=torch.stack([w[int(e),:,0]*x[i//2,0] for i,e in enumerate(ids.flatten())])
    boundary=bool(torch.equal(probe,expected))
    down=indexed_linear(x.repeat_interleave(2,0),ids,w,1)
    if not boundary or not torch.equal(down,expected):raise RuntimeError('Indexed expert semantic boundary probe failed')
    result={'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'hip':torch.version.hip,
                           'device':torch.cuda.get_device_name(0)},'status':'started','boundary_exact':True,'records':[],
            'scope':'Local MoE replay, not HF generation acceptance','variants':{}}
    write(out/'results.json',result)
    samples=[]
    for i in range(4):
        folder=SOURCE/f'sample_{i:06d}'
        x=torch.load(folder/'moe_input.pt',weights_only=True,map_location='cpu').cuda()
        logits=torch.load(folder/'router_logits.pt',weights_only=True,map_location='cpu').cuda()
        samples.append((x,logits))
    cases=[('prefill_ragged',torch.cat([samples[i%4][0][:n] for i,n in enumerate([7,19,31,11,23,17,29,13])]),
            torch.cat([samples[i%4][1][:n] for i,n in enumerate([7,19,31,11,23,17,29,13])]),[0,7,26,57,68,91,108,137,150],False)]
    for count in (1,8,32):
        cases.append((f'decode_{count}',torch.cat([samples[i%4][0][-1:] for i in range(count)]),
                      torch.cat([samples[i%4][1][-1:] for i in range(count)]),list(range(count+1)),True))
    for name,x,logits,bounds,batched in cases:
        calls={v:(lambda variant=v:optimized_experts(module,x,logits,bounds,batched,variant))
               for v in ('cpd','shared_routing','indexed_decode')}
        values={v:fn() for v,fn in calls.items()}
        differences={v:error(value,values['cpd']) for v,value in values.items()}
        for fn in calls.values():
            for _ in range(2):fn()
        times={v:[] for v in calls};peak={}
        for i in range(6):
            order=list(calls);order=order[i%3:]+order[:i%3]
            for variant in order:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=torch.cuda.memory_allocated()
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();value=calls[variant]();b.record();b.synchronize()
                times[variant].append(a.elapsed_time(b))
                peak[variant]=max(peak.get(variant,0),torch.cuda.max_memory_allocated()-before)
        entry={'name':name,'rows':len(x),'bounds':bounds,'phase':'decode' if batched else 'prefill',
               'versus_cpd':differences,'timing_ms':times,'median_ms':{v:statistics.median(t) for v,t in times.items()},
               'peak_additional_allocated_bytes':peak}
        result['records'].append(entry);write(out/'results.json',result)
        print('MICRO',name,'ms',entry['median_ms'],'error',differences,flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        name,x,logits,bounds,batched=cases[-1]
        optimized_experts(module,x,logits,bounds,batched,'indexed_decode');torch.cuda.synchronize()
    profiler.export_chrome_trace(str(out/'indexed_decode_trace.json.gz'))
    kernels=sorted({e.name for e in profiler.events() if str(e.device_type).endswith('CUDA')})
    if not any('_indexed_expert_gemv' in k for k in kernels):raise RuntimeError('Indexed kernel absent from actual GPU trace')
    for variant in ('shared_routing','indexed_decode'):
        result['variants'][variant]={'valid':True,'finite':True,'micro_speedup_decode32':
            result['records'][-1]['median_ms']['cpd']/result['records'][-1]['median_ms'][variant]}
    result.update(status='complete',kernel_names=kernels)
    write(out/'results.json',result)
