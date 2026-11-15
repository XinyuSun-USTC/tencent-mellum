"""Assessment workload manifests; literal commands and proposed adapters stay separate."""
import hashlib,json
from pathlib import Path
from mellum_runtime_config import MODEL,PROJECT

DATA=Path('/data/datasets')
CLIENT=Path('/workspace/venvs/evalscope-perf/bin/evalscope')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate scenario IDs')
    if any(not isinstance(r.get('prompt'),str) or not r['prompt'] for r in rows):raise ValueError('Missing prompt')
    return rows


def chat_ids(tokenizer,messages):
    # New Transformers versions may return a BatchEncoding from tokenize=True;
    # len(that) counts keys, not tokens. Render text then explicitly encode.
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    return tokenizer.encode(rendered,add_special_tokens=False)


def prepare_datasets(output,tokenizer):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);answer={}
    for scenario,count in [('A',512),('C',200)]:
        source=DATA/f'scenario_{scenario.lower()}_prompts.jsonl';rows=load_rows(source)
        if len(rows)!=count:raise ValueError('Scenario row count differs from assessment')
        target=output/f'scenario_{scenario.lower()}_messages.jsonl';details=[]
        with target.open('w') as stream:
            for row in rows:
                messages=[{'role':'user','content':row['prompt']}]
                stream.write(json.dumps(messages,ensure_ascii=False)+'\n')
                raw=len(tokenizer.encode(row['prompt'],add_special_tokens=False));chat=len(chat_ids(tokenizer,messages))
                details.append({'id':row['id'],'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
                                'raw_tokens':raw,'chat_tokens':chat,'newlines':row['prompt'].count('\n'),
                                'shared_prefix_group':row.get('shared_prefix_group')})
        answer[scenario]={'source':str(source),'source_sha256':sha(source),'records':len(rows),
            'proposed_messages_file':str(target),'proposed_messages_sha256':sha(target),'rows':details,
            'raw_min_max':[min(x['raw_tokens'] for x in details),max(x['raw_tokens'] for x in details)],
            'chat_min_max':[min(x['chat_tokens'] for x in details),max(x['chat_tokens'] for x in details)],
            'A_range_valid_raw':sum(512<=r['raw_tokens']<=4096 for r in details),
            'A_range_valid_chat':sum(512<=r['chat_tokens']<=4096 for r in details),
            'note':'Derived messages preserve full prompt including newlines; metadata removed from API body. Proposed adaptation, not silently substituted into official commands.'}
    return answer


def literal_command(scenario,url,output):
    command=[str(CLIENT),'perf','--url',url,'--api','openai','--model',str(MODEL),'--tokenizer-path',str(MODEL)]
    if scenario=='A':command+=['--parallel','32','--number','512','--dataset',str(DATA/'scenario_a_prompts.jsonl'),
                              '--min-prompt-length','512','--max-prompt-length','4096']
    elif scenario=='B':command+=['--parallel','1','--number','50','--dataset','random',
                                '--min-prompt-length','1024','--max-prompt-length','1024']
    elif scenario=='C':command+=['--open-loop','--rate','0.3','--number','200','--dataset','line_by_line',
                                '--dataset-path',str(DATA/'scenario_c_prompts.jsonl'),'--total-timeout','180']
    else:raise ValueError('Unknown assessment scenario')
    command+=['--max-tokens','256','--min-tokens','256','--stream','True',
              '--extra-args','{"ignore_eos": true, "temperature": 0.0}','--name',{'A':'Scenario_A_N1','B':'Scenario_B_SingleStream','C':'Scenario_C_Poisson_N1'}[scenario]]
    return command


def prepare(output,tokenizer):
    result={'scope':'CPU protocol preparation only; no server/benchmark/GPU started','gpus':1,
        'model':str(MODEL),'datasets':prepare_datasets(Path(output)/'datasets',tokenizer),
        'literal_commands':{s:literal_command(s,'http://127.0.0.1:8051/v1/chat/completions',output) for s in 'ABC'},
        'candidate_config':str(PROJECT/'configs/mellum/shared_routing.json'),
        'candidate_config_sha256':sha(PROJECT/'configs/mellum/shared_routing.json'),
        'assessment_pdf_sha256':sha(PROJECT/'docs/requirements/2026 CSIG云一计算加速团队实习校招面试题.pdf'),
        'baseline':json.loads((DATA/'baseline.json').read_text()),
        'protocol_ready_for_full_gpu':False,
        'unresolved':['Validate installed release CLI, dataset semantics and reasoning-stream timing',
                      'A literal --dataset filepath compatibility',
                      'A length filtering/repetition and JSONL prompt handling must be explicit',
                      'B random actual token length; C per-request timeout and open-loop scheduling'],
        'server_command':['/opt/venv/bin/python',str(PROJECT/'scripts/serve_mellum.py'),'--batching','--max-num-seqs','32',
            '--max-model-len','16640','--max-batched-tokens','16640','--campaign-config',str(PROJECT/'configs/mellum/shared_routing.json')]}
    return result
