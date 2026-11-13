"""Deterministic fixed-token developer sets. Not the official 200-prompt set."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
def validate_inputs(data):
    fields=['name','rendered_prompt','input_ids','attention_mask','position_ids','generation_settings','enable_thinking']
    records=data.get('records',[])
    if not records:raise ValueError('Empty input suite')
    names=[]
    for row in records:
        for field in fields:
            if field not in row:raise ValueError(f"Input {row.get('name')}: missing {field}")
        if not isinstance(row['enable_thinking'],bool):raise ValueError('enable_thinking must be bool')
        names.append(row['name'])
    if len(set(names))!=len(names):raise ValueError('Duplicate input names')

def build(tokenizer,suite):
    source=json.loads(Path('/workspace/results/atom_step0_batch_ab/hf_reference/inputs.json').read_text())
    if suite=='short':
        for record in source['records']:record['generation_settings']['max_new_tokens']=64
        return {**source,'suite':suite,'scope':'developer_regression_not_official'}
    if suite=='prefill':
        records=[]
        prompts=[('rotate_list','Write a Python function rotating a list to the right by k places.'),
                 ('stable_unique','Implement stable deduplication of a list of integers in Python.'),
                 ('missing_integer','Find the smallest missing nonnegative integer in an unsorted list.'),
                 ('parenthesized_math','Compute (137 - 58) * 6 and show the calculation.'),
                 ('chinese_inventory','仓库有六箱零件，每箱十五个，取走二十三个后还剩多少个？'),
                 ('identity_explanation','Explain why [] is [] evaluates to False in Python.')]
        def append(name,ids,text,origin):
            settings=dict(source['records'][0]['generation_settings']);settings['max_new_tokens']=1
            n=len(ids)
            if not 0<n<=1024:raise ValueError('Prefill inputs must fit the diagnostic sliding window')
            records.append({'name':name,'input_ids':[ids],'rendered_prompt':text,
                'enable_thinking':origin!='raw_code_boundary',
                'attention_mask':[[1]*n],'position_ids':[list(range(n))],
                'generation_settings':settings,'prompt_source':origin})
        for name,prompt in prompts:
            text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                              add_generation_prompt=True,enable_thinking=True)
            append(name,tokenizer.encode(text,add_special_tokens=False),text,'new_developer_prompt')
        body=tokenizer.encode('def increment(x):\n    return x + 1\n'*20,add_special_tokens=False)
        for n in [16,17]:
            ids=body[:n]
            if len(ids)!=n:raise ValueError('Insufficient raw boundary tokens')
            append(f'raw_{n}',ids,tokenizer.decode(ids,skip_special_tokens=False),'raw_code_boundary')
        for record in build(tokenizer,'window')['records'][:2]:
            ids=record['input_ids'][0]
            append(record['name'],ids,record['rendered_prompt'],'existing_window_boundary')
        return {'schema_version':1,'suite':suite,'scope':'prefill_developer_regression_not_official',
                'model_config_sha256':source['model_config_sha256'],'records':records}
    if suite=='expanded':
        records=[]
        humaneval=[json.loads(line) for line in Path('/data/datasets/HumanEval.jsonl').read_text().splitlines()]
        prompts=[(f'h_{i}', 'Complete this Python function. Return the implementation.\n\n'+humaneval[i]['prompt'],humaneval[i]['task_id']) for i in [0,1,10,100]]
        prompts += [('multiply','Compute 37 times 19 and explain your calculation.','custom_math'),
                    ('word_problem','A box holds 24 pencils. We have 7 boxes and give away 39 pencils. How many remain?','custom_math'),
                    ('binary_search','Write a Python binary_search function for a sorted list, returning -1 when absent.','custom_code'),
                    ('frequencies','Write a Python function counting each word in a string without collections.Counter.','custom_code')]
        for name,prompt,origin in prompts:
            text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            ids=tokenizer.encode(text,add_special_tokens=False)
            records.append({'name':name,'rendered_prompt':text,'input_ids':[ids],
                'attention_mask':[[1]*len(ids)],'position_ids':[list(range(len(ids)))],
                'enable_thinking':True,'prompt_source':origin,
                'generation_settings':{'do_sample':False,'max_new_tokens':64,'eos_token_id':28,'pad_token_id':0}})
        boundary=build(tokenizer,'window')['records'][-1:]+[r for r in build(tokenizer,'long')['records'] if len(r['input_ids'][0]) in [8192,16384]]
        for record in boundary:
            record['generation_settings']['max_new_tokens']=64
            record['prompt_source']='synthetic_boundary'
        records.extend(boundary)
        return {'schema_version':1,'suite':suite,'scope':'developer_regression_not_official',
                'model_config_sha256':source['model_config_sha256'],'records':records}
    lengths=[1023,1024,1025] if suite=='window' else [8191,8192,8193,16384]
    prefix=tokenizer.encode('<|im_start|>user\n',add_special_tokens=False)
    suffix=tokenizer.encode('\nQuestion: What is 17 + 25? Give the answer.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
    body=tokenizer.encode('\n'.join(f'# Record {i}: value = {i%997}; preserve this context for the final question.' for i in range(2000)),add_special_tokens=False)
    records=[]
    for length in lengths:
        count=length-len(prefix)-len(suffix)
        if count<0 or len(body)<count:raise ValueError('Insufficient deterministic filler')
        ids=prefix+body[:count]+suffix
        assert len(ids)==length
        records.append({'name':f'{suite}_{length}','rendered_prompt':tokenizer.decode(ids,skip_special_tokens=False),
           'input_ids':[ids],'attention_mask':[[1]*length],'position_ids':[list(range(length))],
           'enable_thinking':True,'generation_settings':{'do_sample':False,'max_new_tokens':8 if suite=='window' else 4,'eos_token_id':28,'pad_token_id':0}})
    return {'schema_version':1,'suite':suite,'scope':'developer_regression_not_official',
            'model_config_sha256':hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),'records':records}

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['short','window','long','expanded','prefill'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    data=build(t,a.suite);validate_inputs(data);a.output.parent.mkdir(parents=True,exist_ok=True)
    # Do not silently replace a different fixed dataset.
    payload=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if a.output.exists() and a.output.read_text()!=payload:raise FileExistsError(a.output)
    a.output.write_text(payload);print(a.suite,[(r['name'],len(r['input_ids'][0])) for r in data['records']])
if __name__=='__main__':main()
