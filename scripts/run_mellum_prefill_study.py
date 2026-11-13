"""One HF export, then packed/separate QKV controls on fixed one-token probes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from diagnose_mellum_projection_math import DEFAULT_RUN, prepare, digest

ROOT=Path('/workspace')

def prepare_inputs(source):
    manifest,_,_=prepare(source)
    reference=json.loads((source/'hf/reference.json').read_text())
    records=[]
    for record in reference['records']:
        keys=['name','input_ids','attention_mask','position_ids','rendered_prompt','generation_settings',
              'forced_prefix_ids','source_hf_cached_next_token_id','enable_thinking']
        row={key:record[key] for key in keys if key in record}
        if row['generation_settings']['max_new_tokens']!=1:
            raise ValueError('Prefill study only accepts one-token probes')
        records.append(row)
    return manifest,{'schema_version':1,'suite':'diagnostic','scope':'fixed_prefill_only_not_free_generation',
                     'model_config_sha256':reference['model_config_sha256'],'records':records}

def reuse_reference(previous, inputs, inputs_sha256):
    previous=Path(previous).resolve()
    meta=json.loads((previous/'results.json').read_text())
    if meta['status']!='diagnostic_complete' or meta['inputs_sha256']!=inputs_sha256:
        raise ValueError('Reused run is incomplete or has different inputs')
    if digest(previous/'inputs.json')!=inputs_sha256:
        raise ValueError('Reused inputs changed since their run')
    hf=previous/'hf';reference=json.loads((hf/'reference.json').read_text())
    if reference['model_config_sha256']!=inputs['model_config_sha256']:
        raise ValueError('Reused reference model config differs')
    env=reference['environment']
    if env['transformers']!='5.10.1' or env.get('experts_implementation')!='grouped_mm':
        raise ValueError('Reused reference does not have the pinned HF expert backend')
    if len(reference['records'])!=len(inputs['records']):raise ValueError('Reused reference record count differs')
    for actual,expected in zip(reference['records'],inputs['records']):
        if any(actual[key]!=expected[key] for key in ['name','input_ids','position_ids','attention_mask','generation_settings']):
            raise ValueError('Reused reference protocol differs')
    if not (hf/'layers').is_dir() or not (hf/'stages/weights').is_dir():
        raise ValueError('Reused reference lacks observation artifacts')
    hashes={str(file):digest(file) for file in sorted(hf.rglob('*.pt'))}
    return hf,{'reference_run':str(previous),'reference_sha256':digest(hf/'reference.json'),
               'tensor_files_sha256':hashes,'environment':env}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--attention-study',action='store_true')
    group.add_argument('--cpu-rope-study',action='store_true')
    p.add_argument('--reference-run',type=Path,help='Reuse a completed compatible study HF reference')
    a=p.parse_args();source=a.source_run.resolve()
    out=ROOT/'results/mellum_prefill_study'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    result={'status':'started','source_run':str(source),'script_sha256':digest(__file__),
            'scope':'Fixed prefill control; no decode or Step0 acceptance claim','variants':{}}
    def save():(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        manifest,inputs=prepare_inputs(source);result['source_manifest']=manifest
        file=out/'inputs.json';file.write_text(json.dumps(inputs,indent=2)+'\n')
        result['inputs_sha256']=digest(file);save()
        hf=out/'hf'
        if a.reference_run:
            hf,provenance=reuse_reference(a.reference_run,inputs,result['inputs_sha256'])
            result['reference_reuse']=provenance
        variants=([('separate_sdpa_cpu_rope','separate','sdpa_cpu_rope_diagnostic')] if a.cpu_rope_study else
                  [('separate_sdpa','separate','sdpa_diagnostic'),
                   ('separate_sdpa_rope','separate','sdpa_rope_diagnostic')] if a.attention_study else
                  [('packed','packed','aiter'),('separate','separate','aiter')])
        result['planned_variants']=variants;save()
        if a.prepare_only:
            result['status']='prepared';save();print('RESULT prepared output',out,flush=True);return 0
        def run(stage,command):
            print('STAGE',stage,'log',out/(stage+'.log'),flush=True)
            with (out/(stage+'.log')).open('w') as log:
                code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
            return code,(out/(stage+'.log')).read_text()
        if not a.reference_run:
            code,_=run('hf',['/workspace/venvs/hf-reference/bin/python',str(ROOT/'scripts/export_mellum_suite_hf.py'),
                '--inputs',str(file),'--output',str(hf),'--gpu',str(a.gpu),'--dump-layers','--dump-stages'])
            if code:raise RuntimeError('HF export failed; see hf.log')
        from compare_mellum_results import compare
        from compare_mellum_layers import compare_layers,compare_weights
        for backend,qkv,prefill in variants:
            code,log=run(backend,['/opt/venv/bin/python',str(ROOT/'scripts/check_atom_mellum.py'),
                '--reference',str(hf/'reference.json'),'--gpu',str(a.gpu),'--qkv-backend',qkv,
                '--prefill-backend',prefill,'--moe-backend','torch','--stage-layer','0','--dump-layers','--dump-stages'])
            matches=re.findall(r'^RESULT (short_greedy_passed|token_mismatch) output (\S+)',log,re.MULTILINE)
            if code not in (0,1) or len(matches)!=1:
                raise RuntimeError(f'{backend} did not produce complete artifacts; see its log')
            atom=Path(matches[0][1]);actual=json.loads((atom/'results.json').read_text())
            for key,value in [('MELLUM_QKV_BACKEND',qkv),('MELLUM_PREFILL_BACKEND',prefill),('MELLUM_MOE_BACKEND','torch')]:
                if actual['settings'].get(key)!=value:raise ValueError(f'Actual {key} differs')
            audit=actual['loading_audit']
            if audit['source_tensors']!=5631 or audit['successful_parts']!=5631 or audit['missing'] or audit['unexpected']:
                raise ValueError('Incomplete loading audit')
            comparison=compare(hf/'reference.json',atom/'candidate.json')
            layers=compare_layers(hf/'layers',atom/'layers')
            stages=compare_layers(hf/'stages',atom/'stages')
            stages['weights']=compare_weights(hf/'stages/weights',atom/'stages/weights')
            for name,value in [('comparison',comparison),('layers',layers),('stages',stages)]:
                (out/(backend+'_'+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
            result['variants'][backend]={'atom_run':str(atom),'generation_status':matches[0][0],
                'configuration':{'qkv':qkv,'prefill':prefill,'moe':'torch'},'loading_audit':audit,
                'token_summary':comparison['token_summary'],
                'logits':[{'name':r['name'],'max_abs':r.get('max_abs_error'),'rms':r.get('rms_error')} for r in comparison['records']],
                'qkv_exact':all(r['status']=='exact' for r in stages['records'] if r['stage'] in ['q_proj','k_proj','v_proj']),
                'selected_weights_exact':all(r['exact'] for r in stages['weights'])}
            save();print(backend,json.dumps(result['variants'][backend]),flush=True)
        result['status']='diagnostic_complete';save();print('RESULT diagnostic_complete output',out,flush=True);return 0
    except Exception:
        import traceback
        result.update(status='failed',traceback=traceback.format_exc());save()
        print(result['traceback']);print('RESULT failed output',out,flush=True);return 1
if __name__=='__main__':raise SystemExit(main())
