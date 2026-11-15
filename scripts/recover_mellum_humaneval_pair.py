"""Repair only failed HF export; reuse the independently running ATOM output."""
import argparse,json,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,MODEL
from run_mellum_batch_campaign import write,sha
from run_mellum_humaneval_pair import validate_records,extract_all,paired_scores
from mellum_process_runner import run_stage


def wait_candidate(source,inputs,deadline):
    last=0
    while True:
        file=source/'candidate/generation.json'
        data=json.loads(file.read_text()) if file.exists() else {}
        if data.get('status')=='complete':validate_records(inputs,data);return data
        if (source/'candidate/failure.json').exists():raise RuntimeError('Original ATOM producer failed; saved HF remains reusable')
        status=source/'summary.json'
        if status.exists() and json.loads(status.read_text()).get('status') in ('failed','interrupted','partial_generation_or_scoring'):
            raise RuntimeError('Original run ended before complete candidate generation')
        if time.monotonic()>=deadline:raise TimeoutError('Waiting for original candidate exceeded overall recovery budget')
        if time.monotonic()-last>=15:
            print('WAIT original candidate',len(data.get('records',[])),'/164',flush=True);last=time.monotonic()
        time.sleep(.25)


def finalize_scores(out,hf,candidate):
    left=[json.loads(l) for l in (out/'hf_score/samples.jsonl_results.jsonl').read_text().splitlines()]
    right=[json.loads(l) for l in (out/'candidate_score/samples.jsonl_results.jsonl').read_text().splitlines()]
    score=paired_scores(left,right)
    for name in ('hf','candidate'):
        expected=json.loads((out/(name+'_score')/'score.json').read_text())['score']['pass@1']
        if abs(expected-score['pass_at_1'][name])>1e-12:raise ValueError('Official pass@1 disagrees with task verdicts')
    if hf['environment']['torch']!=candidate['environment']['torch']:raise ValueError('Torch versions differ')
    return {**score,'exact_full_sequences':sum(a['generated_ids']==b['generated_ids'] for a,b in zip(hf['records'],candidate['records']))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--budget-minutes',type=float,default=90);a=p.parse_args()
    if not 0<a.budget_minutes<=180:p.error('Positive recovery budget<=180min required')
    source=a.run.resolve();old=json.loads((source/'manifest.json').read_text())
    inputs=json.loads((source/'inputs.json').read_text());tasks=[json.loads(l) for l in (source/'problems.jsonl').read_text().splitlines()]
    if old['inputs_sha256']!=sha(source/'inputs.json') or len(inputs['records'])!=164 or len(tasks)!=164:
        raise ValueError('Original protocol/input hash differs or is incomplete')
    for f,h in old['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Original running source changed: '+f)
    if old['thinking'] or old['max_new_tokens']!=2048:raise ValueError('Recovery targets user-confirmed nonthinking2048 protocol')
    out=Path('/workspace/results/mellum_humaneval_recovery')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    for file in ('inputs.json','problems.jsonl'):shutil.copyfile(source/file,out/file)
    hf_source=Path('/workspace/venvs/hf-reference/lib/python3.12/site-packages/transformers/models/mellum/modeling_mellum.py')
    if sha(hf_source)!=old['hf_model_source_sha256']:raise ValueError('HF model source changed')
    files=[Path(__file__).resolve(),PROJECT/'scripts/run_mellum_humaneval_pair.py',PROJECT/'scripts/validate_hf_mellum_gpu.py',
           PROJECT/'scripts/run_mellum_humaneval.py',PROJECT/'scripts/score_mellum_humaneval.py',PROJECT/'scripts/humaneval_sandbox.py']
    current={str(f):sha(f) for f in files}
    m={**old,'status':'recovery_prepared','original_run':str(source),'original_manifest_sha256':sha(source/'manifest.json'),
       'recovery_source_sha256':current,'recovery_note':'Only HF loading-report serialization normalized; original candidate inputs/math/output untouched'}
    m.pop('completed_hf_sha256',None);write(out/'manifest.json',m)
    if a.prepare_only:print('RESULT recovery_prepared output',out);return 0
    deadline=time.monotonic()+a.budget_minutes*60;summary={'status':'started','original_run':str(source)}
    def remaining():
        n=deadline-time.monotonic()
        if n<=0:raise TimeoutError('Recovery total time budget exceeded')
        return n
    try:
        run_stage('hf_recovery',['/workspace/venvs/hf-reference/bin/python',str(PROJECT/'scripts/run_mellum_humaneval_pair.py'),
              '--worker','hf','--output',str(out/'hf')],out/'hf.log',timeout=remaining())
        hf=json.loads((out/'hf/generation.json').read_text());validate_records(inputs,hf)
        if hf.get('status')!='complete':raise ValueError('Recovered HF incomplete')
        m['completed_hf_sha256']=sha(out/'hf/generation.json');write(out/'manifest.json',m)
        candidate=wait_candidate(source,inputs,deadline)
        for f,h in old['source_sha256'].items():
            if sha(f)!=h:raise ValueError('Original candidate source changed while running')
        (out/'candidate').mkdir();write(out/'candidate/generation.json',candidate)
        m['original_candidate_generation_sha256']=sha(source/'candidate/generation.json');write(out/'manifest.json',m)
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        for name,data in [('hf',hf),('candidate',candidate)]:
            summary[name+'_extraction']=extract_all(inputs,data,tasks,tokenizer,out,name)
            prior=source/(name+'_score')
            can_reuse=(name=='candidate' and (prior/'score.json').exists() and (prior/'samples.jsonl_results.jsonl').exists()
                and (prior/'evaluator_manifest.json').exists() and (source/'candidate_samples.jsonl').exists()
                and sha(source/'candidate_samples.jsonl')==sha(out/'candidate_samples.jsonl')
                and json.loads((prior/'evaluator_manifest.json').read_text())==old['evaluator'])
            if can_reuse:
                shutil.copytree(prior,out/(name+'_score'));summary['candidate_score_reused']=True
            else:
                run_stage(name+'_score',['/opt/venv/bin/python',str(PROJECT/'scripts/score_mellum_humaneval.py'),
                    '--samples',str(out/(name+'_samples.jsonl')),'--problems',str(out/'problems.jsonl'),
                    '--output',str(out/(name+'_score'))],out/(name+'_score.log'),timeout=min(900,remaining()))
        summary.update(finalize_scores(out,hf,candidate),status='paired_functional_evaluation_complete')
        summary['protocol']=old['protocol'];summary['thinking']=False;summary['max_new_tokens']=2048
        summary['notes']=['No token-position gate; official functional score under explicit local protocol',
            'Public200/16k and final performance remain independent requirements; inspect extraction/truncation counts']
    except KeyboardInterrupt:summary['status']='interrupted'
    except Exception:summary.update(status='failed',traceback=traceback.format_exc())
    summary['recovery_source_unchanged']=all(sha(f)==h for f,h in current.items())
    if not summary['recovery_source_unchanged']:summary['status']='failed_source_changed'
    write(out/'summary.json',summary)
    (out/'report.md').write_text('# Recovered full HumanEval pair\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')
    print('RESULT',summary['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
