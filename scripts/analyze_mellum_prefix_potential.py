"""CPU-only structural prefix analysis; no cache hit or speed claims."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from mellum_runtime_config import MODEL
from mellum_perf_protocol import chat_ids


def prefix_length(rows):
    if not rows:return 0
    for i in range(min(map(len,rows))):
        if any(row[i]!=rows[0][i] for row in rows[1:]):return i
    return min(map(len,rows))


def analyze(groups,block_size=16):
    report=[]
    for name,items in sorted(groups.items()):
        common=prefix_length(items);blocks=common//block_size
        report.append({'group':name,'requests':len(items),'min_tokens':min(map(len,items)),
            'max_tokens':max(map(len,items)),'common_prefix_tokens':common,'common_full_blocks':blocks,
            'ideal_reusable_tokens_after_first':(len(items)-1)*blocks*block_size})
    total=sum(len(row) for items in groups.values() for row in items)
    reuse=sum(r['ideal_reusable_tokens_after_first'] for r in report)
    return {'groups':report,'total_input_tokens':total,'group_common_prefix_reuse_tokens':reuse,
            'group_common_prefix_reuse_fraction':reuse/total if total else 0,
            'scope':'Ideal reuse of the common prefix shared by every request in each metadata group, after first request. Excludes eviction/timing. Does not count deeper pairwise/subgroup sharing; not a universal cache-hit upper bound or measured speedup.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    source=Path('/data/datasets/scenario_c_prompts.jsonl');groups=defaultdict(list)
    for row in [json.loads(l) for l in source.read_text().splitlines()]:
        ids=chat_ids(t,[{'role':'user','content':row['prompt']}])
        if not isinstance(ids,list) or len(ids)<4000 or not isinstance(ids[0],int):raise ValueError('Unexpected C tokenization')
        groups[row['shared_prefix_group']].append(ids)
    result=analyze(groups);result.update(dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tokenizer_path=str(MODEL),thinking=True,block_size=16)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
