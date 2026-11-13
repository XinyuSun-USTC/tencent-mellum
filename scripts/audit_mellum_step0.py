"""Read-only Step0 evidence audit; does not start inference or execute generated code."""
import argparse
from datetime import datetime,timezone
import hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
from compare_mellum_generation import compare_generation
from compare_mellum_results import compare

ROOT=Path('/workspace')

def audit_run(run,expected_cases):
    summary=json.loads((run/'summary.json').read_text())
    atom=Path(summary['atom_run']);reference=run/'hf/reference.json';candidate=atom/'candidate.json'
    comparison=compare(reference,candidate)
    generation=compare_generation(reference,candidate)
    hf=json.loads(reference.read_text());actual=json.loads((atom/'results.json').read_text())
    audit=actual['loading_audit']
    cases=[]
    for row in hf['records']:
        count=len(row['generated_ids'][0]);calls=row.get('generation_experts_by_forward')
        if count!=64 or calls!=['grouped_mm']+['batched_mm']*(count-1):
            raise ValueError('Unexpected actual HF decode configuration or output count')
        cases.append({'name':row['name'],'input_tokens':len(row['input_ids'][0]),'output_tokens':count})
    expected={'MELLUM_QKV_BACKEND':'separate','MELLUM_MOE_BACKEND':'torch',
              'MELLUM_ROUTER_BACKEND':'torch','MELLUM_PREFILL_BACKEND':'sdpa_cpu_rope_diagnostic',
              'MELLUM_REFERENCE_KV_ROPE':'1','MELLUM_SDPA_DECODE':'1','MELLUM_HF_GENERATION_EXPERTS':'1'}
    passed=(comparison['status']=='passed' and generation['nonexact_positions']==0
        and generation['compared_positions']==expected_cases*64 and len(cases)==expected_cases
        and all(r['both_selected_are_argmax'] for r in generation['records'])
        and audit['source_tensors']==audit['successful_parts']==5631
        and not audit['missing'] and not audit['unexpected']
        and actual.get('sampling_mode')=='normal' and actual.get('ignore_eos') is False
        and 'MELLUM_FORCED_DECODE_PLAN' not in actual['settings']
        and all(actual['settings'].get(k)==v for k,v in expected.items()))
    return {'status':'passed' if passed else 'failed','source_run':str(run),'atom_run':str(atom),
            'comparison':comparison,'generation':{k:v for k,v in generation.items() if k!='records'},
            'cases':cases,'loading_audit':audit,'actual_configuration':expected,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [reference,candidate]}}

def capability_probes():
    import torch
    source=ROOT/'ATOM/atom/model_config/mellum_attention.py'
    spec=importlib.util.spec_from_file_location('mellum_acceptance_attention',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    results=[]
    for n in [17]:
        q=torch.zeros(n,1,2)
        ctx=SimpleNamespace(attn_metadata=SimpleNamespace(cu_seqlens_q=torch.tensor([0,n]),cu_seqlens_k=torch.tensor([0,n])))
        try:
            module.sdpa_single_prefill(SimpleNamespace(sliding_window=8,scale=0.5),q,q,q,None,None,None,None,ctx)
            results.append({'input_tokens':n,'window':8,'status':'cpu_beyond_window_supported'})
        except ValueError as e:results.append({'input_tokens':n,'status':'unsupported','reason':str(e)})
    try:
        module.sdpa_cached_decode(None,torch.zeros(2,1,2),None,None,None,None,None,None,
            SimpleNamespace(attn_metadata=SimpleNamespace(context_lens=torch.tensor([25,25]))))
        results.append({'requests':2,'status':'unexpectedly_not_rejected'})
    except ValueError as e:results.append({'requests':2,'status':'unsupported','reason':str(e)})
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    out=a.output or ROOT/'results/mellum_step0_acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    result={'status':'incomplete','claim_step0_passed':False,'scope':'Local evidence audit, not official evaluator',
        'checks':{},'gpu_access':Path('/dev/kfd').exists()}
    for label,run,count in [('short','20261112T193650834240Z',2),('diverse_window','20261112T213129468143Z',10)]:
        try:result['checks'][label]=audit_run(ROOT/'results/mellum_validation'/run,count)
        except Exception as e:result['checks'][label]={'status':'failed','error':repr(e)}
    result['capability_probes']=capability_probes()
    baseline=json.loads(Path('/data/datasets/baseline.json').read_text())
    result['acceptance_gates']={
        'single_gpu_short_correctness':{'status':'passed' if all(v['status']=='passed' for v in result['checks'].values()) else 'failed'},
        'greedy_200':{'status':'not_run','reason':'Only 12 developer prompts covered; no 200-prompt acceptance result or official evaluator located'},
        'long_16k':{'status':'not_verified','reason':'Long-prefill band mask implemented; no 16k GPU acceptance result for the enhanced code'},
        'humaneval':{'status':'not_run','reason':'No paired HF/ATOM pass@1 assessment; existing baseline is explicitly rejected','existing_reference_note':baseline.get('humaneval_note')},
        'service':{'status':'not_verified','reason':'No Mellum GPU-backed HTTP inference acceptance artifact; multi-request candidate decode is unsupported'},
    }
    result['optional_features']={'multi_gpu':'Not enabled; not a prerequisite for single GPU Step0','mtp':'Not enabled; not required unless used'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('RESULT step0_incomplete output',out,flush=True)
    for key,value in result['acceptance_gates'].items():print(key,value['status'],flush=True)
    return 1
if __name__=='__main__':raise SystemExit(main())
