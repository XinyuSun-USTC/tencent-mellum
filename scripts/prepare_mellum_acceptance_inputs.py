"""Freeze 200 developer prompts for generation agreement; not official acceptance data."""
import argparse,hashlib,json
from pathlib import Path
from prepare_mellum_suite import MODEL,validate_inputs


def build(tokenizer,human,scenario):
    tasks=[json.loads(x) for x in human.read_text().splitlines() if x.strip()]
    supplied=[json.loads(x) for x in scenario.read_text().splitlines() if x.strip()]
    if len(tasks)!=164 or len(supplied)<36:raise ValueError('Unexpected source dataset sizes')
    prompts=[('humaneval_'+str(i),'Complete this Python function. Return the implementation.\n\n'+r['prompt'],str(r['task_id']),'HumanEval') for i,r in enumerate(tasks)]
    prompts += [('scenario_a_'+str(i),r['prompt'],str(r.get('id',i)),'supplied_scenario_a') for i,r in enumerate(supplied[:36])]
    if len({x[1] for x in prompts})!=200:raise ValueError('Duplicate prompt texts')
    records=[]
    for name,prompt,source_id,origin in prompts:
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=tokenizer.encode(text,add_special_tokens=False);n=len(ids)
        if not 0<n<=16384:raise ValueError('Input outside supported validation bound')
        records.append({'name':name,'rendered_prompt':text,'input_ids':[ids],
            'attention_mask':[[1]*n],'position_ids':[list(range(n))],'enable_thinking':True,
            'prompt_source':origin,'source_id':source_id,
            'generation_settings':{'do_sample':False,'max_new_tokens':64,'eos_token_id':28,'pad_token_id':0}})
    return {'schema_version':1,'suite':'diagnostic','scope':'developer_200_generation_agreement_not_official_not_humaneval_pass1',
        'model_config_sha256':hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),
        'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [human,scenario]],
        'selection':'All 164 HumanEval prompts and first 36 supplied scenario A prompts in original file order',
        'records':records}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    data=build(tokenizer,Path('/data/datasets/HumanEval.jsonl'),Path('/data/datasets/scenario_a_prompts.jsonl'));validate_inputs(data)
    payload=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if a.output.exists() and a.output.read_text()!=payload:raise ValueError('Refuse overwrite of different fixed input data')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(payload)
    lengths=[len(r['input_ids'][0]) for r in data['records']]
    print('prepared',a.output,'cases',len(lengths),'token_lengths',min(lengths),max(lengths),flush=True)
if __name__=='__main__':main()
