"""GPU0 bounded prefix on/off probe; no HF rerun, no formal acceptance claim."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,MODEL,apply_serving,engine_options
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha,load_rows,chat_ids
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write


def prepare(out,profile='r'):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    rows=load_rows('/data/datasets/scenario_c_prompts.jsonl');selected=[rows[0],rows[len(rows)//2],rows[-1]]
    cases=[{'id':r['id'],'input_ids':chat_ids(tokenizer,[{'role':'user','content':r['prompt']}])} for r in selected]
    # Keep representative real prompts; no precomputed outputs or special-case cache keys.
    groups=[{'name':'cold_first','indices':[0]},{'name':'same_prompt','indices':[0]},
            {'name':'shared_prefix_ragged','indices':[0,1,2]},{'name':'reverse_reuse','indices':[2,1,0]}]
    paths=list((PROJECT/'ATOM/atom').rglob('*.py'))+[Path(__file__),PROJECT/'scripts/mellum_runtime_config.py',PROJECT/'scripts/mellum_execution_profiles.py',PROJECT/'scripts/mellum_process_runner.py']
    m={'gpu':0,'profile':profile,'budget_tokens':16,'cases':cases,'groups':groups,
       'source_sha256':{str(p):sha(p) for p in paths},'dataset_sha256':sha('/data/datasets/scenario_c_prompts.jsonl'),
       'scope':'Prefix off/on self comparison on three real C prompts. Not public200/HumanEval or formal speedup.',
       'thinking':True,'chunked_prefill':False}
    write(out/'plan.json',m);return m


def check_sources(m):
    if any(sha(p)!=v for p,v in m['source_sha256'].items()):raise ValueError('Prefix probe source changed')


def worker(out,plan,enabled):
    settings=apply_serving(0,out,True,'reference',True);settings.update(profile_settings(plan['profile']))
    settings.update(MELLUM_PREFIX_COMPAT='1' if enabled else '0',MELLUM_PREFIX_AUDIT_PATH=str(out/'prefix.jsonl'),MELLUM_NATIVE_AUDIT_PATH=str(out/'native_calls.jsonl'))
    os.environ.update(settings);sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU0 unavailable')
    if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require 80GiB free on GPU0')
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    maxlen=max(len(c['input_ids']) for c in plan['cases'])+32
    options=engine_options(maxlen,None);options.update(torch_profiler_dir=None,max_num_seqs=4,
        enable_prefix_caching=enabled,enable_chunked_prefill=False,port=8826)
    result={'status':'started','prefix_enabled':enabled,'settings':settings,'options':options,'groups':[]};engine=None
    write(out/'results.json',result)
    try:
        engine=EngineArgs(**options).create_engine()
        audit=json.loads((out/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise ValueError('Loading audit failed')
        result['loading_audit']=audit
        for group in plan['groups']:
            cases=[plan['cases'][i] for i in group['indices']];start=time.monotonic()
            outputs=engine.generate([c['input_ids'] for c in cases],
                [SamplingParams(temperature=0,max_tokens=plan['budget_tokens'],ignore_eos=True) for c in cases])
            if len(outputs)!=len(cases) or any(len(o['token_ids'])!=plan['budget_tokens'] for o in outputs):raise ValueError('Invalid probe output count/length')
            result['groups'].append({'name':group['name'],'elapsed_seconds':time.monotonic()-start,
                'records':[{'id':c['id'],'input_tokens':len(c['input_ids']),'generated_ids':o['token_ids']} for c,o in zip(cases,outputs)]})
            write(out/'results.json',result);print(group['name'],'completed',len(cases),flush=True)
        rows=[json.loads(x) for x in (out/'prefix.jsonl').read_text().splitlines()] if (out/'prefix.jsonl').exists() else []
        result['prefix_events']=rows;result['observed_cached_tokens']=sum(sum(x['cached_tokens']) for x in rows)
        result['status']='complete';write(out/'results.json',result)
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());write(out/'results.json',result);raise
    finally:
        if engine is not None:engine.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path)
    p.add_argument('--profile',choices=['r','fused_moe_native_attention','fused_moe_native_batched'],default='r')
    p.add_argument('--prefix',action='store_true');p.add_argument('--budget-minutes',type=float,default=25);a=p.parse_args()
    if a.worker:
        plan=json.loads((a.worker.parent/'plan.json').read_text());worker(a.worker,plan,a.prefix);return 0
    if not 1<=a.budget_minutes<=40:p.error('Budget must be 1..40 minutes')
    out=Path('/workspace/results/mellum_prefix_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    m=prepare(out,a.profile)
    if a.prepare_only:print('RESULT prefix_probe_prepared output',out);return 0
    deadline=time.monotonic()+a.budget_minutes*60;s={'status':'started','scope':m['scope']}
    try:
        for name,enabled in [('off',False),('on',True)]:
            check_sources(m);directory=out/name;directory.mkdir();cmd=['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(directory)]
            if enabled:cmd+=['--prefix']
            remaining=deadline-time.monotonic()
            if remaining<60:raise TimeoutError('Insufficient remaining budget to start next prefix worker')
            run_stage(name,cmd,directory/'worker.log',timeout=remaining)
        check_sources(m);off=json.loads((out/'off/results.json').read_text());on=json.loads((out/'on/results.json').read_text())
        comparisons=[]
        for x,y in zip(off['groups'],on['groups']):
            for l,r in zip(x['records'],y['records']):
                assert l['id']==r['id']
                comparisons.append({'group':x['name'],'id':l['id'],'exact':l['generated_ids']==r['generated_ids']})
        hit=on['observed_cached_tokens']>0
        s.update(status='prefix_probe_passed' if hit and all(x['exact'] for x in comparisons) else 'prefix_probe_review_required',
            expected_native_variability=m['profile']=='fused_moe_native_attention',
            comparisons=comparisons,observed_cached_tokens=on['observed_cached_tokens'],timings={k:[{'name':x['name'],'seconds':x['elapsed_seconds']} for x in v['groups']] for k,v in [('off',off),('on',on)]})
    except Exception:s.update(status='failed',traceback=traceback.format_exc())
    s['source_unchanged']=all(sha(p)==v for p,v in m['source_sha256'].items());write(out/'summary.json',s)
    print('RESULT',s['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
