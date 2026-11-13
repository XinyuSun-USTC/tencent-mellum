"""Run one developer suite sequentially in HF and frozen ATOM environments."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from compare_mellum_results import compare
from mellum_process_runner import run_stage

ROOT=Path('/workspace')

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['short','window','long','expanded','prefill','diagnostic'],default='short');p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--inputs',type=Path);p.add_argument('--dump-layers',action='store_true');p.add_argument('--dump-stages',action='store_true');p.add_argument('--max-batched-tokens',type=int);p.add_argument('--router-audit',action='store_true')
    p.add_argument('--swa-prefill-opt',action='store_true',help='Use Step0 settings plus local SWA candidate')
    p.add_argument('--prefill-backend',choices=['aiter','sdpa_diagnostic','sdpa_rope_diagnostic','sdpa_cpu_rope_diagnostic'],default='aiter')
    p.add_argument('--stage-layer',type=int,choices=range(28),default=0)
    p.add_argument('--qkv-backend',choices=['packed','separate'],default='packed')
    p.add_argument('--correctness-candidate',action='store_true',help='Explicit tested short TP1 correctness combination; default path unchanged')
    p.add_argument('--dump-generation-logits',action='store_true')
    p.add_argument('--stage-timeout',type=float,default=300,help='Maximum seconds per worker stage; heartbeat remains 15 seconds')
    a=p.parse_args()
    if a.stage_timeout<=0:p.error('--stage-timeout must be positive')
    if a.swa_prefill_opt:a.correctness_candidate=True
    if a.correctness_candidate:
        for option,field,wanted in [('--qkv-backend','qkv_backend','separate'),
                                    ('--prefill-backend','prefill_backend','sdpa_cpu_rope_diagnostic')]:
            explicit=any(arg==option or arg.startswith(option+'=') for arg in sys.argv[1:])
            if explicit and getattr(a,field)!=wanted:p.error('Candidate conflicts with '+option)
        a.qkv_backend='separate';a.prefill_backend='sdpa_cpu_rope_diagnostic';a.dump_generation_logits=True
    if a.suite=='diagnostic' and a.inputs is None:p.error('diagnostic requires --inputs')
    out=ROOT/'results/mellum_validation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True);status={'swa_prefill_opt':a.swa_prefill_opt,'correctness_candidate':a.correctness_candidate,'sampling_mode':'normal_no_forcing','qkv_backend':a.qkv_backend,'stage_layer':a.stage_layer,'prefill_backend':a.prefill_backend,'suite':a.suite,'status':'started','output':str(out)}
    def save():(out/'summary.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(stage,command):
        status.update(phase=stage);save()
        try:
            code,_=run_stage(stage,command,out/(stage+'.log'),
                allowed_exit_codes=(0,1) if stage=='atom' else (0,),with_code=True,timeout=a.stage_timeout)
        except Exception as error:
            status.update(status='stage_failed',error=str(error));save();return False
        if code:
            status.update(status='stage_failed',returncode=code);save();return False
        return True
    hf=str(ROOT/'venvs/hf-reference/bin/python');atom='/opt/venv/bin/python'
    inputs=out/'inputs.json'
    if a.inputs:
        import shutil
        shutil.copyfile(a.inputs,inputs)
    elif not run('prepare',[hf,str(ROOT/'scripts/prepare_mellum_suite.py'),'--suite',a.suite,'--output',str(inputs)]):return 1
    if a.correctness_candidate:
        data=json.loads(inputs.read_text())
        from prepare_mellum_suite import validate_inputs
        validate_inputs(data)
        if any(len(r['input_ids'][0])>16384 for r in data['records']):
            status['status']='unsupported_candidate_input';save()
            raise ValueError('This candidate currently supports at most 16384 prompt tokens')
    if a.prepare_only:
        status['status']='prepared';save();print('RESULT prepared',out);return 0
    if a.router_audit:
        if not run('router',[atom,str(ROOT/'scripts/check_mellum_router.py'),'--output',str(out/'router'),'--gpu',str(a.gpu)]):return 1
    if not run('hf',[hf,str(ROOT/'scripts/export_mellum_suite_hf.py'),'--inputs',str(inputs),'--output',str(out/'hf'),'--gpu',str(a.gpu),'--stage-layer',str(a.stage_layer)]+(['--dump-layers'] if a.dump_layers else [])+(['--dump-stages'] if a.dump_stages else [])+(['--dump-generation-logits'] if a.dump_generation_logits else [])):return 1
    atom_ok=run('atom',[atom,str(ROOT/'scripts/check_atom_mellum.py'),'--reference',str(out/'hf/reference.json'),'--gpu',str(a.gpu),'--qkv-backend',a.qkv_backend,'--prefill-backend',a.prefill_backend,'--stage-layer',str(a.stage_layer)]+(['--dump-layers'] if a.dump_layers else [])+(['--dump-stages'] if a.dump_stages else [])+(['--max-batched-tokens',str(a.max_batched_tokens)] if a.max_batched_tokens else [])+(['--reference-kv-rope','--sdpa-decode','--torch-router','--hf-generation-experts'] if a.correctness_candidate else [])+(['--swa-prefill-opt'] if a.swa_prefill_opt else [])+(['--dump-generation-logits'] if a.dump_generation_logits else []))
    # The engine creates its own unique run directory; get the exact path
    # from its final RESULT line rather than scanning for "latest".
    lines=(out/'atom.log').read_text().splitlines()
    final=next((line for line in reversed(lines) if line.startswith('RESULT ') and ' output ' in line),None)
    if final is None:status['status']='missing_atom_result';save();return 1
    atom_out=Path(final.rsplit(' output ',1)[1]);status['atom_run']=str(atom_out)
    if not (atom_out/'candidate.json').exists():
        status['status']='missing_atom_candidate';save();return 1
    try:
        comparison=compare(out/'hf/reference.json',atom_out/'candidate.json')
    except Exception as error:
        status.update(status='artifact_comparison_error',error=str(error));save();raise
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    # Zero tolerance is a diagnostic default, not an invented BF16 acceptance
    # threshold. Token equality and all non-numerical protocol checks gate.
    structural=[x for x in comparison['mismatches'] if x not in {r['name'] for r in comparison['records']}]
    critical=structural+[r['name'] for r in comparison['records'] if any(x!='logits_values' for x in r['mismatches']) or r.get('top1_exact') is False]
    if critical:status.update(status='comparison_failed',critical=critical)
    elif comparison['status']=='passed':status['status']='tokens_and_logits_exact'
    else:status['status']='tokens_exact_logits_review_required'
    if a.dump_layers:
        from compare_mellum_layers import compare_layers
        layers=compare_layers(out/'hf/layers',atom_out/'layers')
        (out/'layers_comparison.json').write_text(json.dumps(layers,indent=2)+'\n')
        nonexact=[r for r in layers['records'] if r['status']!='exact']
        status['layer_observation_summary']={'compared_tensors':len(layers['records']),
            'nonexact_tensors':len(nonexact),'first_nonexact':nonexact[:1],
            'scope':'Observed prefill states only; does not validate decode/KV'}
    if a.dump_stages:
        from compare_mellum_layers import compare_layers, compare_weights
        stages=compare_layers(out/'hf/stages',atom_out/'stages')
        stages['weights']=compare_weights(out/'hf/stages/weights',atom_out/'stages/weights')
        (out/'stages_comparison.json').write_text(json.dumps(stages,indent=2)+'\n')
    try:
        if a.correctness_candidate:
            observed=json.loads((atom_out/'results.json').read_text())
            expected={'MELLUM_HF_GENERATION_EXPERTS':'1','MELLUM_QKV_BACKEND':'separate','MELLUM_MOE_BACKEND':'torch','MELLUM_ROUTER_BACKEND':'torch',
                      'MELLUM_PREFILL_BACKEND':'sdpa_cpu_rope_diagnostic','MELLUM_REFERENCE_KV_ROPE':'1','MELLUM_SDPA_DECODE':'1'}
            expected['MELLUM_SWA_PREFILL_OPT']='1' if a.swa_prefill_opt else '0'
            if any(observed['settings'].get(k)!=v for k,v in expected.items()) or observed.get('sampling_mode')!='normal' or observed.get('ignore_eos') is not False:
                raise ValueError('Candidate actual configuration/normal sampling differs')
            if 'MELLUM_FORCED_DECODE_PLAN' in observed['settings']:raise ValueError('Unexpected forcing in normal generation')
            hf_actual=json.loads((out/'hf/reference.json').read_text())
            for row in hf_actual['records']:
                calls=row.get('generation_experts_by_forward')
                wanted=['grouped_mm']+['batched_mm']*(len(row['generated_ids'][0])-1)
                if calls!=wanted:raise ValueError('HF actual expert dispatch differs from candidate schedule')
        if a.dump_generation_logits:
            from compare_mellum_generation import compare_generation
            generation=compare_generation(out/'hf/reference.json',atom_out/'candidate.json')
            (out/'generation_comparison.json').write_text(json.dumps(generation,indent=2)+'\n')
            status['generation_logit_summary']={k:v for k,v in generation.items() if k!='records'}
            if generation['nonexact_positions'] and status['status']=='tokens_and_logits_exact':
                status['status']='tokens_exact_logits_review_required'
    except Exception as error:
        status.update(status='post_generation_comparison_error',error=str(error));save()
        print('RESULT',status['status'],'output',out,flush=True);return 1
    status['token_summary']=comparison.get('token_summary',{})
    status['logits_errors']={r['name']:r.get('max_abs_error') for r in comparison['records']};save()
    print('RESULT',status['status'],'output',out,flush=True)
    return 1 if critical or not atom_ok else 0
if __name__=='__main__':raise SystemExit(main())
