"""Forced-history baseline: compare true ATOM logits with cached HF at each step."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from prepare_mellum_suite import validate_inputs
from diagnose_mellum_projection_math import digest
from mellum_process_runner import run_stage

ROOT=Path('/workspace')
DEFAULT_PREFILL=ROOT/'results/mellum_validation/20261112T122035305179Z'

def normalize_reference_traces(reference):
    data=copy.deepcopy(reference);changes=[]
    for row in data['records']:
        prompt=row['input_ids'][0];forced=row['generated_ids'][0]
        for step in row['decode_steps']:
            s=step['step'];expected=prompt if s==0 else [forced[s-1]]
            if s==0 and step['input_ids']==prompt+forced:
                step['input_ids']=list(prompt)
                changes.append({'name':row['name'],'step':0,'field':'input_ids','reason':'legacy mutable-list trace snapshot'})
            if step['input_ids']!=expected:raise ValueError('HF trace input differs from fixed history')
            if step['positions']!=(list(range(len(prompt))) if s==0 else [len(prompt)+s-1]):
                raise ValueError('HF trace position differs')
            if step['logical_context_tokens']!=len(prompt)+s:raise ValueError('HF trace context length differs')
    return data,changes

def prepare_inputs(run,steps,layer_study=False,case=None):
    data=json.loads((run/'inputs.json').read_text());validate_inputs(data)
    wanted={case} if case else {'window_1023'} if layer_study else {'raw_16','raw_17','chinese_inventory','window_1023'}
    records=[r for r in data['records'] if r['name'] in wanted]
    if len(records)!=len(wanted):raise ValueError('Missing forced decode cases')
    summary=json.loads((run/'summary.json').read_text())
    if case:
        comparison=json.loads((run/'comparison.json').read_text())
        rows=[r for r in comparison['records'] if r['name']==case]
        if len(rows)!=1 or rows[0].get('max_abs_error')!=0.0:
            raise ValueError('Selected case lacks exact full prefill logits')
    elif summary['status']!='tokens_and_logits_exact' or summary['layer_observation_summary']['nonexact_tensors']!=0:
        raise ValueError('Source prefill run did not fully align')
    return {**data,'suite':'forced_decode','records':records,'scope':'Fixed history baseline, not generation acceptance'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefill-run',type=Path,default=DEFAULT_PREFILL)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--reuse-hf-run',type=Path)
    p.add_argument('--stage-timeout',type=float,default=300)
    p.add_argument('--reference-kv-rope',action='store_true')
    p.add_argument('--sdpa-decode',action='store_true')
    p.add_argument('--layer-study',action='store_true')
    p.add_argument('--case',help='One source case with verified exact prefill logits; requires fresh two-step layer study')
    p.add_argument('--torch-router',action='store_true')
    a=p.parse_args()
    if a.case and not a.layer_study:p.error('--case requires --layer-study')
    if a.sdpa_decode and not a.reference_kv_rope:p.error('--sdpa-decode requires --reference-kv-rope')
    if a.layer_study and (a.steps!=2 or not a.sdpa_decode or a.reuse_hf_run):p.error('--layer-study requires --steps 2 --sdpa-decode and a fresh HF export')
    if a.stage_timeout<=0:p.error('--stage-timeout must be positive')
    if not 2<=a.steps<=16:p.error('Use 2 to 16 steps for the first decode baseline')
    out=ROOT/'results/mellum_decode_validation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    report={'status':'started','source_prefill_run':str(a.prefill_run.resolve()),'steps':a.steps,
            'scope':'Only saved pre-forcing logits are compared; forced token equality is not acceptance','records':[]}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        inputs=prepare_inputs(a.prefill_run,a.steps,a.layer_study,a.case);file=out/'inputs.json';file.write_text(json.dumps(inputs,indent=2)+'\n')
        report['inputs_sha256']=digest(file);report['script_sha256']=digest(__file__);save()
        if a.prepare_only:
            report['status']='prepared';save();print('RESULT prepared output',out,flush=True);return 0
        def run(stage,command):
            return run_stage(stage,command,out/(stage+'.log'),timeout=a.stage_timeout)
        hf=out/'hf'
        reference_path=hf/'reference.json'
        if a.reuse_hf_run:
            old=a.reuse_hf_run.resolve();hf=old/'hf'
            old_report=json.loads((hf/'results.json').read_text())
            old_ref=json.loads((hf/'reference.json').read_text())
            if old_report['status']!='passed' or digest(old/'inputs.json')!=report['inputs_sha256']:
                raise ValueError('Cannot reuse incomplete/different HF reference')
            if old_ref['model_config_sha256']!=inputs['model_config_sha256'] or len(old_ref['records'])!=4:
                raise ValueError('Reused HF config/cases differ')
            env=old_ref['environment']
            if env['transformers']!='5.10.1' or env.get('experts_implementation')!='grouped_mm':
                raise ValueError('Reused HF backend differs')
            for row in old_ref['records']:
                if len(row['decode_steps'])!=a.steps or len(row['generated_ids'][0])!=a.steps:
                    raise ValueError('Reused HF step count differs')
                for step in row['decode_steps']:
                    for field,sha in [('cached_logits','cached_sha256'),('uncached_logits','uncached_sha256')]:
                        if digest(Path(step[field]))!=step[sha]:raise ValueError('Reused HF logits changed')
            normalized,changes=normalize_reference_traces(old_ref)
            reference_path=out/'reused_hf_reference.json'
            reference_path.write_text(json.dumps(normalized,indent=2)+'\n')
            report['hf_reference_provenance']={'source':str(hf/'reference.json'),
                'source_sha256':digest(hf/'reference.json'),'derived_sha256':digest(reference_path),'metadata_changes':changes}
            report['hf_reused_from']=str(old);save()
            print('STAGE hf: reused verified completed reference',old,flush=True)
        else:
            run('hf',['/workspace/venvs/hf-reference/bin/python',str(ROOT/'scripts/export_mellum_decode_reference.py'),
                '--inputs',str(file),'--output',str(hf),'--steps',str(a.steps),'--gpu',str(a.gpu)]+(['--dump-first-decode'] if a.layer_study else []))
        log=run('atom',['/opt/venv/bin/python',str(ROOT/'scripts/check_atom_mellum.py'),
            '--reference',str(reference_path),'--gpu',str(a.gpu),'--qkv-backend','separate',
            '--prefill-backend','sdpa_cpu_rope_diagnostic','--forced-decode']+(['--reference-kv-rope'] if a.reference_kv_rope else [])+(['--sdpa-decode'] if a.sdpa_decode else [])+(['--decode-layer-capture'] if a.layer_study else [])+(['--torch-router'] if a.torch_router else []))
        matches=re.findall(r'^RESULT forced_decode_complete output (\S+)',log,re.MULTILINE)
        if len(matches)!=1:raise ValueError('Missing forced decode completion artifact')
        atom=Path(matches[0]);report['atom_run']=str(atom)
        actual_run=json.loads((atom/'results.json').read_text())
        expected={'MELLUM_QKV_BACKEND':'separate','MELLUM_MOE_BACKEND':'torch',
                  'MELLUM_PREFILL_BACKEND':'sdpa_cpu_rope_diagnostic',
                  'MELLUM_REFERENCE_KV_ROPE':'1' if a.reference_kv_rope else '0',
                  'MELLUM_SDPA_DECODE':'1' if a.sdpa_decode else '0',
                  'MELLUM_DECODE_LAYER_CAPTURE':'1' if a.layer_study else '0',
                  'MELLUM_ROUTER_BACKEND':'torch' if a.torch_router else 'aiter',
                  'MELLUM_FORCED_DECODE_PLAN':str(reference_path.resolve())}
        if any(actual_run['settings'].get(k)!=v for k,v in expected.items()):
            raise ValueError('Unexpected actual ATOM diagnostic configuration')
        audit=actual_run['loading_audit']
        if audit['source_tensors']!=5631 or audit['successful_parts']!=5631 or audit['missing'] or audit['unexpected']:
            raise ValueError('Incomplete ATOM loading audit')
        report['atom_configuration']=expected;report['loading_audit']=audit
        report['forced_plan_sha256']=digest(reference_path)
        import torch
        from diagnose_mellum_projection_math import compare
        reference=json.loads(reference_path.read_text())
        for index,record in enumerate(reference['records']):
            for step in record['decode_steps']:
                s=step['step'];path=atom/'decode'/f'sample_{index:06d}'
                trace=json.loads((path/f'step_{s:03d}_inputs.json').read_text())
                for key in ['input_ids','positions','logical_context_tokens']:
                    if trace[key]!=step[key]:raise ValueError('Actual engine history/position mismatch')
                values=[]
                for label,sha in [('cached_logits','cached_sha256'),('uncached_logits','uncached_sha256')]:
                    f=Path(step[label])
                    if digest(f)!=step[sha]:raise ValueError('HF logits hash mismatch')
                    values.append(torch.load(f,weights_only=True,map_location='cpu'))
                af=path/f'step_{s:03d}_logits.pt';actual=torch.load(af,weights_only=True,map_location='cpu')
                report['records'].append({'name':record['name'],'step':s,'phase':trace['phase'],
                    'logical_context_tokens':trace['logical_context_tokens'],'history_verified':True,
                    'atom_vs_hf_cached':compare(actual,values[0]),'hf_cached_vs_uncached':compare(values[0],values[1]),
                    'atom_logits_sha256':digest(af),'hf_cache_key_shapes':step['hf_cache_key_shapes']})
        if len(report['records'])!=len(inputs['records'])*a.steps:raise ValueError('Incomplete decode observations')
        if a.layer_study:
            state_rows=[];attention_rows=[]
            config=json.loads(Path('/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json').read_text())
            tags=['embedding']+[f'layer_{i:02d}' for i in range(config['num_hidden_layers'])]+['final_norm']
            for step in [0,1]:
                for tag in tags:
                    af=atom/'decode/sample_000000'/f'step_{step:03d}_{tag}.pt'
                    rf=hf/'sample_000000'/af.name
                    state_rows.append({'step':step,'stage':tag,**compare(torch.load(af,weights_only=True,map_location='cpu'),torch.load(rf,weights_only=True,map_location='cpu'))})
            for i in range(config['num_hidden_layers']):
                af=atom/'decode/sample_000000'/f'layer_{i:02d}_attention.pt';rf=hf/'sample_000000'/af.name
                av=torch.load(af,weights_only=True,map_location='cpu');rv=torch.load(rf,weights_only=True,map_location='cpu')
                for tag in ['query','keys','values','sdpa_output']:
                    attention_rows.append({'layer':i,'stage':tag,**compare(av[tag],rv[tag])})
            (out/'layer_state_comparison.json').write_text(json.dumps(state_rows,indent=2)+'\n')
            (out/'attention_state_comparison.json').write_text(json.dumps(attention_rows,indent=2)+'\n')
            report['first_nonexact_state']=next((r for r in state_rows if not r['exact']),None)
            report['first_nonexact_attention']=next((r for r in attention_rows if not r['exact']),None)
        report['status']='forced_history_logits_exact' if all(r['atom_vs_hf_cached']['exact'] for r in report['records']) else 'forced_history_logits_differ'
        report['first_nonexact']=next((r for r in report['records'] if not r['atom_vs_hf_cached']['exact']),None)
        save();print('RESULT',report['status'],'output',out,flush=True);return 0
    except Exception:
        import traceback
        report.update(status='failed',traceback=traceback.format_exc());save();print(report['traceback']);print('RESULT failed output',out,flush=True);return 1
if __name__=='__main__':raise SystemExit(main())
