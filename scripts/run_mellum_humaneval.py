"""Paired complete-code HumanEval development assessment, not just greedy64."""
import argparse,ast,hashlib,json,re,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT,MODEL
from mellum_process_runner import run_stage
from run_mellum_swa_hf_check import validate_settings


def extract_completion(raw,problem):
    text=raw.replace('<|im_end|>','').replace('<|endoftext|>','')
    if '<think>' in text:
        if '</think>' not in text:return '', 'unfinished_thinking'
        text=text.split('</think>',1)[1]
    elif '</think>' in text:text=text.split('</think>',1)[1]
    blocks=re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```',text,re.DOTALL)
    if blocks:
        # Fixed rule, without looking at test outcomes: first block defining the entry point.
        matches=[b for b in blocks if re.search(r'^def\s+'+re.escape(problem['entry_point'])+r'\s*\(',b,re.MULTILINE)]
        text=(matches or blocks)[0]
    code=text.strip('\r\n')
    if not code.strip():return '', 'empty_answer'
    try:ast.parse(problem['prompt']+'\n'+code)
    except SyntaxError:return code,'invalid_python'
    return code,'extracted'


def build(tokenizer,tasks,budget,thinking):
    rows=[]
    for task in tasks:
        prompt='Complete this Python function. Return a complete Python implementation in a single Python code block.\n\n'+task['prompt']
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
        ids=tokenizer.encode(text,add_special_tokens=False);n=len(ids)
        rows.append({'name':task['task_id'].replace('/','_'),'source_id':task['task_id'],'rendered_prompt':text,
            'input_ids':[ids],'attention_mask':[[1]*n],'position_ids':[list(range(n))],'enable_thinking':thinking,
            'generation_settings':{'do_sample':False,'max_new_tokens':budget,'eos_token_id':28,'pad_token_id':0}})
    return {'schema_version':1,'suite':'diagnostic','scope':'Paired HumanEval code generation, local protocol not confirmed official',
            'model_config_sha256':hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),'records':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--disable-thinking',action='store_true');p.add_argument('--gpu',type=int,default=0);p.add_argument('--prepare-only',action='store_true');p.add_argument('--generate-only',action='store_true')
    a=p.parse_args()
    if not 64<=a.max_new_tokens<=8192:p.error('Generation budget must be 64..8192')
    out=Path('/workspace/results/mellum_humaneval')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    all_tasks=[json.loads(x) for x in Path('/data/datasets/HumanEval.jsonl').read_text().splitlines() if x.strip()]
    tasks=all_tasks if a.full else [all_tasks[i] for i in [0,1,10,100]]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    inputs=build(tokenizer,tasks,a.max_new_tokens,not a.disable_thinking)
    (out/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
    (out/'problems.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in tasks))
    result={'status':'prepared','scope':'Full164' if a.full else 'Pilot4_not_acceptance',
        'thinking':not a.disable_thinking,'max_new_tokens':a.max_new_tokens,'tasks':len(tasks),
        'swa_eligible_prompts':sum(len(x['input_ids'][0])>=1024 for x in inputs['records']),
        'dataset_sha256':hashlib.sha256(Path('/data/datasets/HumanEval.jsonl').read_bytes()).hexdigest(),
        'inputs_sha256':hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),
        'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),PROJECT/'scripts/score_mellum_humaneval.py',PROJECT/'scripts/humaneval_sandbox.py',PROJECT/'ATOM/atom/model_config/mellum_attention.py',PROJECT/'ATOM/atom/models/mellum.py']}}
    def save():(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    if a.prepare_only:print('RESULT prepared output',out);return 0
    if not a.generate_only and not Path('/workspace/results/humaneval_environment/manifest.json').exists():
        raise RuntimeError('Set up official evaluator first, or explicitly use --generate-only')
    try:
        if not a.generate_only:
            run_stage('evaluator_selftest',['/opt/venv/bin/python',str(PROJECT/'scripts/score_mellum_humaneval.py'),'--self-test'],out/'selftest.log',timeout=1500)
        run_stage('hf',['/workspace/venvs/hf-reference/bin/python',str(PROJECT/'scripts/export_mellum_suite_hf.py'),
            '--inputs',str(out/'inputs.json'),'--output',str(out/'hf'),'--gpu',str(a.gpu)],out/'hf.log',timeout=43200 if a.full else 3600)
        _,log=run_stage('candidate',['/opt/venv/bin/python',str(PROJECT/'scripts/check_atom_mellum.py'),
            '--reference',str(out/'hf/reference.json'),'--gpu',str(a.gpu),'--qkv-backend','separate',
            '--prefill-backend','sdpa_cpu_rope_diagnostic','--reference-kv-rope','--sdpa-decode',
            '--torch-router','--hf-generation-experts','--swa-prefill-opt'],out/'candidate.log',
            timeout=43200 if a.full else 3600,allowed_exit_codes=(0,1),with_code=True)
        matches=re.findall(r'^RESULT (?:short_greedy_passed|token_mismatch) output (\S+)$',log,re.MULTILINE)
        if len(matches)!=1:raise RuntimeError('Candidate generation incomplete')
        atom=Path(matches[0]);validate_settings(atom,1);result['candidate_run']=str(atom)
        dataset={t['task_id']:t for t in tasks}
        for name,file in [('hf',out/'hf/reference.json'),('candidate',atom/'candidate.json')]:
            records=json.loads(file.read_text())['records'];samples=[];details=[]
            for row in records:
                task=dataset[row['source_id']];ids=row['generated_ids'][0]
                raw=tokenizer.decode(ids,skip_special_tokens=False);code,status=extract_completion(raw,task)
                samples.append({'task_id':task['task_id'],'completion':code if status=='extracted' else "\n\nraise RuntimeError('completion extraction failed')\n"})
                details.append({'task_id':task['task_id'],'raw_text':raw,'completion':code,'extraction_status':status,
                    'output_tokens':len(ids),'hit_budget_without_eos':len(ids)>=a.max_new_tokens and (not ids or ids[-1]!=28)})
            (out/(name+'_samples.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in samples))
            (out/(name+'_extraction.json')).write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
            result[name+'_extraction']={'failures':sum(x['extraction_status']!='extracted' for x in details),
                'budget_hits':sum(x['hit_budget_without_eos'] for x in details)}
            save()
            if not a.generate_only:
                run_stage(name+'_score',['/opt/venv/bin/python',str(PROJECT/'scripts/score_mellum_humaneval.py'),
                    '--samples',str(out/(name+'_samples.jsonl')),'--problems',str(out/'problems.jsonl'),
                    '--output',str(out/(name+'_score'))],out/(name+'_score.log'),timeout=1500)
        if a.generate_only:result['status']='generation_and_extraction_complete_not_scored'
        else:
            scores={n:json.loads((out/(n+'_score')/'score.json').read_text())['score']['pass@1'] for n in ['hf','candidate']}
            result['scores']=scores;result['absolute_difference_percentage_points']=abs(scores['hf']-scores['candidate'])*100
            result['bf16_2pp_local_check']=result['absolute_difference_percentage_points']<=2 if a.full else None
            left=[json.loads(x) for x in (out/'hf_score/samples.jsonl_results.jsonl').read_text().splitlines()]
            right={x['task_id']:x for x in [json.loads(x) for x in (out/'candidate_score/samples.jsonl_results.jsonl').read_text().splitlines()]}
            result['pass_status_changes']=[{'task_id':x['task_id'],'hf':x['passed'],'candidate':right[x['task_id']]['passed']} for x in left if x['passed']!=right[x['task_id']]['passed']]
            result['interpretation_warnings']=['Local prompt/extraction/budget protocol, not confirmed official evaluator settings']
            if result['hf_extraction']['budget_hits'] or result['candidate_extraction']['budget_hits']:
                result['interpretation_warnings'].append('Generation budget reached: inspect incomplete thinking/code before interpreting scores')
            if scores['hf']==0:result['interpretation_warnings'].append('HF score is zero: pipeline/budget validity must be reviewed')
            result['status']='paired_evaluation_complete'
        save();print('RESULT',result['status'],'output',out);return 0
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
