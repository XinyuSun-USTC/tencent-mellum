"""Triangulate three unchanged ABBA prompts against HF, normal greedy 64 tokens."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
from mellum_process_runner import run_stage
from mellum_runtime_config import PROJECT, MODEL


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, tokenizer):
    summary=json.loads((source/'summary.json').read_text())
    if summary['status']!='candidate_measured':raise ValueError('Source ABBA is incomplete')
    plans=[source/stage/'plan.json' for stage in ['a0','b0','b1','a1']]
    cases=json.loads(plans[0].read_text())['cases']
    if [c['input_tokens'] for c in cases]!=[1024,4096,16384]:raise ValueError('Unexpected ABBA input lengths')
    for path in plans[1:]:
        if json.loads(path.read_text())['cases']!=cases:raise ValueError('ABBA plans differ')
    model_hash=digest(MODEL/'config.json');records=[]
    for c in cases:
        ids=c['input_ids'];n=c['input_tokens']
        if len(ids)!=n or hashlib.sha256(json.dumps(ids).encode()).hexdigest()!=c['token_ids_sha256']:
            raise ValueError('Source token IDs/hash differ')
        origin=PROJECT/c['source']
        if digest(origin)!=c['source_sha256']:raise ValueError('Original source file changed')
        data=json.loads(origin.read_text())
        if data['model_config_sha256']!=model_hash:raise ValueError('Model config differs')
        records.append({'name':c['name'],'input_ids':[ids],
            'rendered_prompt':tokenizer.decode(ids,skip_special_tokens=False),
            'attention_mask':[[1]*n],'position_ids':[list(range(n))],
            'enable_thinking':True,'prompt_source':'unchanged_ABBA_fixed_token_input',
            'generation_settings':{'do_sample':False,'max_new_tokens':64,'eos_token_id':28,'pad_token_id':0}})
    # Save only schema-v1 inputs. No forced histories or ABBA output tokens enter generation.
    return {'schema_version':1,'suite':'diagnostic','scope':'Three fixed ABBA inputs, normal greedy64, not public-set acceptance',
            'model_config_sha256':model_hash,'records':records}


def compare_artifacts(hf, atom, output):
    from compare_mellum_results import compare
    from compare_mellum_generation import compare_generation
    value=compare(hf,atom);generation=compare_generation(hf,atom)
    output.mkdir();(output/'comparison.json').write_text(json.dumps(value,indent=2)+'\n')
    (output/'generation_comparison.json').write_text(json.dumps(generation,indent=2)+'\n')
    return {'token_summary':value['token_summary'],'records':value['records'],
            'generation_summary':{k:v for k,v in generation.items() if k!='records'},
            'forks':[r for r in generation['records'] if r['hf_selected']!=r['atom_selected']]}


def validate_settings(atom, enabled):
    from mellum_runtime_config import CANDIDATE
    actual=json.loads((atom/'results.json').read_text())
    # Shared config also includes offline/logging settings; check mathematical backend settings here.
    expected={k:v for k,v in CANDIDATE.items() if k.startswith('MELLUM_')}
    expected['MELLUM_SWA_PREFILL_OPT']=str(enabled)
    if any(actual['settings'].get(k)!=v for k,v in expected.items()):raise ValueError('Actual backend settings differ')
    if actual['sampling_mode']!='normal' or actual['ignore_eos'] is not False:raise ValueError('Not normal generation')
    if 'MELLUM_FORCED_DECODE_PLAN' in actual['settings']:raise ValueError('Forced history is forbidden')
    audit=actual['loading_audit']
    if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:
        raise ValueError('Loading audit differs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,default=Path('/workspace/results/mellum_swa_optimization/20261114T121938893923Z'))
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true');p.add_argument('--stage-timeout',type=float,default=900)
    a=p.parse_args()
    if a.stage_timeout<=0:p.error('Stage timeout must be positive')
    out=Path('/workspace/results/mellum_swa_hf_check')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    result={'status':'started','source_run':str(a.source_run.resolve()),'output':str(out)}
    def save():(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        inputs=prepare(a.source_run,tokenizer);path=out/'inputs.json';path.write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
        result.update(inputs_sha256=digest(path),source_summary_sha256=digest(a.source_run/'summary.json'),
            source_plan_sha256={stage:digest(a.source_run/stage/'plan.json') for stage in ['a0','b0','b1','a1']},
            source_sha256={str(f):digest(f) for f in [Path(__file__),PROJECT/'scripts/compare_mellum_generation.py',PROJECT/'scripts/run_mellum_validation.py',PROJECT/'scripts/check_atom_mellum.py',PROJECT/'scripts/export_mellum_suite_hf.py',PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'ATOM/atom/model_config/mellum_attention.py']})
        save()
        if a.prepare_only:
            result['status']='prepared';save();print('RESULT prepared output',out);return 0
        code,log=run_stage('hf_and_candidate',['/opt/venv/bin/python',str(PROJECT/'scripts/run_mellum_validation.py'),
            '--suite','diagnostic','--inputs',str(path),'--swa-prefill-opt','--gpu',str(a.gpu),
            '--stage-timeout',str(a.stage_timeout)],out/'hf_and_candidate.log',timeout=a.stage_timeout*2+120,
            allowed_exit_codes=(0,1),with_code=True)
        matches=re.findall(r'^RESULT \S+ output (\S+)$',log,re.MULTILINE)
        if len(matches)!=1:raise RuntimeError('Candidate pipeline did not emit a comparison result')
        run=Path(matches[0]);meta=json.loads((run/'summary.json').read_text())
        if meta['status'] not in ['comparison_failed','tokens_exact_logits_review_required','tokens_and_logits_exact']:
            raise RuntimeError('Candidate pipeline failed before comparison')
        hf=run/'hf/reference.json';candidate=Path(meta['atom_run']);validate_settings(candidate,1)
        result['candidate_validation_run']=str(run)
        result['hf_vs_candidate']=compare_artifacts(hf,candidate/'candidate.json',out/'hf_vs_candidate');save()
        code,log=run_stage('baseline',['/opt/venv/bin/python',str(PROJECT/'scripts/check_atom_mellum.py'),
            '--reference',str(hf),'--gpu',str(a.gpu),'--qkv-backend','separate','--prefill-backend','sdpa_cpu_rope_diagnostic',
            '--reference-kv-rope','--sdpa-decode','--torch-router','--hf-generation-experts','--dump-generation-logits'],
            out/'baseline.log',timeout=a.stage_timeout,allowed_exit_codes=(0,1),with_code=True)
        matches=re.findall(r'^RESULT (?:short_greedy_passed|token_mismatch) output (\S+)$',log,re.MULTILINE)
        if len(matches)!=1:raise RuntimeError('Baseline did not produce generation artifacts')
        baseline=Path(matches[0]);validate_settings(baseline,0);result['baseline_run']=str(baseline)
        result['hf_vs_baseline']=compare_artifacts(hf,baseline/'candidate.json',out/'hf_vs_baseline')
        # ABBA ignores EOS and uses 256 tokens for 1K. Compare only the common
        # normal-generation prefix, and disclose any EOS truncation explicitly.
        for variant,atom in [('reference',baseline),('candidate',candidate)]:
            observed=json.loads((atom/'candidate.json').read_text())['records']
            stage='a0' if variant=='reference' else 'b0'
            old=json.loads((a.source_run/stage/'worker_results.json').read_text())['rows'];checks=[]
            for row in observed:
                n=len(row['input_ids'][0]);budget=256 if n==1024 else 64
                original=next(x for x in old if x['name']==row['name'] and x['output_tokens']==budget and x['repeat']==0)
                got=row['generated_ids'][0];checks.append({'name':row['name'],'normal_output_tokens':len(got),
                    'abba_budget':budget,'common_prefix_exact':got==original['token_ids'][:len(got)]})
            result[variant+'_vs_abba']=checks
        result['status']='diagnostic_complete';save();print('RESULT diagnostic_complete output',out);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise

if __name__=='__main__':raise SystemExit(main())
