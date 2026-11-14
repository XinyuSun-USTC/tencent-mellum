"""Reuse verified Step0 HF200 outputs; evaluate the opt-in SWA candidate."""
import argparse
from datetime import datetime,timezone
import hashlib,json,re,sys,traceback
from pathlib import Path
from mellum_runtime_config import PROJECT,MODEL
from mellum_process_runner import run_stage
from run_mellum_swa_hf_check import validate_settings


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--swa-prefill-impl',choices=['ck','triton'],default='ck');p.add_argument('--eligible-only',action='store_true')
    a=p.parse_args();source=Path('/workspace/results/mellum_validation/20261113T104458011376Z')
    out=(Path('/workspace/results')/('mellum_swa_32' if a.eligible_only else 'mellum_swa_200'))/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    meta=json.loads((source/'summary.json').read_text());data=json.loads((source/'hf/reference.json').read_text())
    if meta['status']!='tokens_and_logits_exact' or len(data['records'])!=200:raise ValueError('Invalid HF200 source')
    if data['model_config_sha256']!=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest():raise ValueError('Model differs')
    if data['environment']['transformers']!='5.10.1':raise ValueError('Reference version differs')
    if a.eligible_only:
        data['records']=[r for r in data['records'] if len(r['input_ids'][0])>=1024]
        data['scope']='developer200_activation_subset_diagnostic'
        data['selection']='Input length>=1024 subset of fixed developer200, original order'
    count=len(data['records'])
    for row in data['records']:
        if len(row['generated_ids'][0])!=64 or row['generation_settings']['max_new_tokens']!=64:raise ValueError('Invalid reference output budget')
        f=source/'hf'/row['prefill_logits']
        if hashlib.sha256(f.read_bytes()).hexdigest()!=row['prefill_logits_sha256']:raise ValueError('Reference tensor hash differs')
        row['prefill_logits']=str(f.resolve());row.pop('generation_logits',None)
    (out/'reference.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    result={'status':'prepared','source_run':str(source),'source_reference_sha256':hashlib.sha256((source/'hf/reference.json').read_bytes()).hexdigest(),
            'scope':'Developer activation subset' if a.eligible_only else 'Developer200, not confirmed official public set; no new logits equality gate',
            'swa_prefill_impl':a.swa_prefill_impl,'source_reference_records':200,'reference_records':count,'swa_eligible_records':sum(len(x['input_ids'][0])>=1024 for x in data['records'])}
    def save():(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    if a.prepare_only:print('RESULT prepared output',out);return 0
    try:
        _,log=run_stage('candidate',['/opt/venv/bin/python',str(PROJECT/'scripts/check_atom_mellum.py'),
            '--reference',str(out/'reference.json'),'--gpu',str(a.gpu),'--qkv-backend','separate',
            '--prefill-backend','sdpa_cpu_rope_diagnostic','--reference-kv-rope','--sdpa-decode',
            '--torch-router','--hf-generation-experts','--swa-prefill-opt','--swa-prefill-impl',a.swa_prefill_impl],out/'candidate.log',timeout=3600,allowed_exit_codes=(0,1),with_code=True)
        paths=re.findall(r'^RESULT (?:short_greedy_passed|token_mismatch) output (\S+)$',log,re.MULTILINE)
        if len(paths)!=1:raise RuntimeError('Candidate did not complete')
        atom=Path(paths[0]);validate_settings(atom,1)
        if json.loads((atom/'results.json').read_text())['settings'].get('MELLUM_SWA_PREFILL_IMPL')!=a.swa_prefill_impl:raise RuntimeError('Actual SWA implementation differs')
        from compare_mellum_results import compare
        comparison=compare(out/'reference.json',atom/'candidate.json');(out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        record_names={row['name'] for row in comparison['records']}
        if any(x not in record_names for x in comparison['mismatches']) or any(set(row['mismatches'])-{'generated_ids','logits_values'} for row in comparison['records']):
            raise RuntimeError('Structural/nonfinite comparison error; inspect comparison.json')
        names={row['name'] for row in data['records'] if len(row['input_ids'][0])>=1024}
        result['candidate_run']=str(atom);result['all']=comparison['token_summary']
        for label,eligible in [('swa_eligible',True),('below_window',False)]:
            rows=[x for x in comparison['records'] if (x['name'] in names)==eligible]
            positions=sum(x['token_stats']['compared_positions'] for x in rows);matching=sum(x['token_stats']['matching_positions'] for x in rows)
            exact=sum('generated_ids' not in x['mismatches'] for x in rows)
            result[label]={'records':len(rows),'matching_positions':matching,'compared_positions':positions,
                'token_match_fraction':matching/positions if positions else None,'exact_sequences':exact,'exact_sequence_fraction':exact/len(rows) if rows else None}
        result['all']['exact_sequence_fraction']=result['all']['exact_sequences']/count
        result['developer_90_percent_checks']={'token_position_interpretation':result['all']['token_match_fraction']>=.9,
            'whole_sequence_interpretation':result['all']['exact_sequence_fraction']>=.9,
            'note':'Activation subset is diagnostic, not a replacement public set' if a.eligible_only else 'Report both interpretations; local developer set does not certify official acceptance'}
        result['status']='developer_comparison_complete';save();print('RESULT developer_comparison_complete output',out);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise

if __name__=='__main__':raise SystemExit(main())
