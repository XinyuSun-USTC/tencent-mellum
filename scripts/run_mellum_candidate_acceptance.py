"""One selected profile, one engine load, three local checks; verified HF reuse."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,MODEL,apply_serving,engine_options
from mellum_execution_profiles import PROFILES,profile_settings
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
from run_mellum_humaneval_pair import validate_records,extract_all,paired_scores,EVALUATOR
import run_mellum_abc_baseline as perf
HE=Path('/workspace/results/mellum_humaneval_recovery/20261115T213706098181Z')
TOKEN=Path('/workspace/results/mellum_validation/20261113T104458011376Z')
LONG=Path('/workspace/results/mellum_validation/20261113T103154043872Z')


def validate_candidate(inputs,candidate):
    expected=inputs['records'];actual=candidate['records']
    if len(actual)!=len(expected) or len({r['name'] for r in actual})!=len(actual):raise ValueError('Incomplete or duplicate records')
    for a,b in zip(expected,actual):
        if any(a.get(k)!=b.get(k) for k in ['name','source_id','input_ids','generation_settings','enable_thinking']):raise ValueError('Candidate input/protocol differs')
        ids=b['generated_ids']
        if len(ids)!=1 or len(ids[0])>a['generation_settings']['max_new_tokens'] or any(type(i) is not int or i<0 for i in ids[0]):raise ValueError('Invalid output IDs/budget')


def token_comparison(reference,candidate):
    validate_candidate(reference,candidate)
    rows=[]
    for a,b in zip(reference['records'],candidate['records']):
        x,y=a['generated_ids'][0],b['generated_ids'][0];n=max(len(x),len(y))
        rows.append({'name':a['name'],'input_tokens':len(a['input_ids'][0]),'compared_positions':n,
                     'matching_positions':sum(i==j for i,j in zip(x,y)),'exact':x==y})
    def group(selected):
        n=sum(r['compared_positions'] for r in selected);m=sum(r['matching_positions'] for r in selected)
        return {'records':len(selected),'matching_positions':m,'compared_positions':n,
                'position_match_fraction':m/n if n else None,'exact_sequences':sum(r['exact'] for r in selected),
                'at_least_90_percent':m/n>=.9 if n else False}
    return {'all':group(rows),'16k':group([r for r in rows if r['input_tokens']>=16000]),'records':rows}


def prepare(out,profile,gpu):
    references={};external={};cfg=sha(MODEL/'config.json')
    from validate_mellum_batch200 import prepare as verified_developer200
    verified_developer200()  # checks the archived reference checksum and greedy input metadata
    for name,source,count in [('tokens',TOKEN,200),('long',LONG,4)]:
        if json.loads((source/'summary.json').read_text())['status']!='tokens_and_logits_exact':raise ValueError('Unverified token reference')
        f=source/'hf/reference.json';d=json.loads(f.read_text())
        if len(d['records'])!=count or d['model_config_sha256']!=cfg or d['environment']['transformers']!='5.10.1':raise ValueError('Reference identity differs')
        if any(r['generation_settings']['do_sample'] or len(r['generated_ids'][0])!=64 for r in d['records']):raise ValueError('Reference protocol differs')
        if name=='tokens':
            pinned=json.loads((PROJECT/'artifacts/performance/07_representative_correctness/developer200/summary.json').read_text())['source_reference_sha256']
            if sha(f)!=pinned:raise ValueError('Archived HF200 checksum differs')
        references[name]=str(f);external[str(f)]=sha(f)
    manifest=json.loads((HE/'manifest.json').read_text());inp=json.loads((HE/'inputs.json').read_text());hf=json.loads((HE/'hf/generation.json').read_text())
    if (manifest['thinking'] or manifest['max_new_tokens']!=2048 or hf['status']!='complete' or len(inp['records'])!=164
            or sha(HE/'inputs.json')!=manifest['inputs_sha256'] or sha(HE/'hf/generation.json')!=manifest['completed_hf_sha256']
            or sha('/data/datasets/HumanEval.jsonl')!=manifest['dataset_sha256'] or hf['model_config_sha256']!=cfg):raise ValueError('HumanEval HF reuse identity/protocol differs')
    validate_records(inp,hf)
    if json.loads((HE/'summary.json').read_text())['status']!='paired_functional_evaluation_complete':raise ValueError('HF reference recovery incomplete')
    evaluator=json.loads(EVALUATOR.read_text())
    for f,h in evaluator['files_sha256'].items():
        if sha(Path(evaluator['source'])/f)!=h:raise ValueError('Official evaluator differs')
    references['humaneval']=str(HE/'hf/generation.json')
    hf_model=Path('/workspace/venvs/hf-reference/lib/python3.12/site-packages/transformers/models/mellum/modeling_mellum.py')
    if sha(hf_model)!=manifest['hf_model_source_sha256']:raise ValueError('HF reference model source changed')
    external[str(hf_model)]=sha(hf_model)
    for file,digest in evaluator['files_sha256'].items():external[str(Path(evaluator['source'])/file)]=digest
    for name in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
        f=MODEL/name
        if f.exists():external[str(f)]=sha(f)
    for f in [HE/'inputs.json',HE/'hf/generation.json',HE/'hf_score/score.json',HE/'hf_score/samples.jsonl_results.jsonl',EVALUATOR,Path('/data/datasets/HumanEval.jsonl')]:external[str(f)]=sha(f)
    bundles={}
    keep=('name','source_id','prompt_source','input_ids','attention_mask','position_ids','generation_settings','enable_thinking','rendered_prompt')
    for name,ref in references.items():
        raw=json.loads(Path(ref).read_text())
        for row in raw['records']:
            n=len(row['input_ids'][0])
            if row['attention_mask']!=[[1]*n] or row['position_ids']!=[list(range(n))] or row.get('forced_prefix_ids'):
                raise ValueError('Reference requires unsupported padding/forced history')
        bundles[name]={**raw,'records':[{k:r[k] for k in keep if k in r} for r in raw['records']]}
        for field in ['loading_info','environment']:bundles[name].pop(field,None)
        write(out/(name+'_inputs.json'),bundles[name])
    sources=perf.sources()
    for name in ['run_mellum_candidate_acceptance.py','run_mellum_humaneval_pair.py','run_mellum_humaneval.py','score_mellum_humaneval.py','humaneval_sandbox.py','validate_mellum_batch200.py','run_mellum_full_campaign.py']:
        f=PROJECT/'scripts'/name;sources[str(f)]=sha(f)
    plan={'profile':profile,'gpu':gpu,'references':references,'source_sha256':sources,'external_sha256':external,
          'input_sha256':{str(out/(n+'_inputs.json')):sha(out/(n+'_inputs.json')) for n in bundles},
          'scope':'Local developer200, length-boundary group and HumanEval; official hidden/public protocol not asserted identical',
          'protocol':'Reuse verified inputs and HF IDs; HE thinking disabled, 2048 tokens; no logits/tensor equality gate'}
    write(out/'manifest.json',plan);return plan


def current(plan):
    for group in ['source_sha256','external_sha256','input_sha256']:
        if any(sha(f)!=h for f,h in plan[group].items()):raise ValueError('Source/reference/input changed')


def runtime(profile,gpu,target):
    settings=apply_serving(gpu,target,True,'reference',True)
    settings.update(profile_settings(profile))
    settings.update(MELLUM_NATIVE_AUDIT_PATH=str(target/'native_calls.jsonl'),
        MELLUM_FASTPATH_AUDIT_PATH=str(target/'fastpaths.json'),MELLUM_GRAPH_AUDIT_PATH=str(target/'graph_replays.jsonl'),
        MELLUM_PREFIX_AUDIT_PATH=str(target/'prefix.jsonl'),MELLUM_METADATA_AUDIT_PATH=str(target/'metadata.jsonl'))
    os.environ.update(settings)
    opts=engine_options(16640,None);opts.update(max_num_seqs=32,max_num_batched_tokens=int(settings.get('MELLUM_CHUNK_BUDGET','0')) or 16640,
        port=9826+gpu*100,enforce_eager=settings.get('MELLUM_DECODE_GRAPH')!='1',
        enable_prefix_caching=settings.get('MELLUM_PREFIX_ENABLED')=='1',enable_chunked_prefill=int(settings.get('MELLUM_CHUNK_BUDGET','0'))>0,
        cudagraph_capture_sizes='[1,2,4,8,16,32]')
    return settings,opts


def worker(out,suites=None,batch_size=32):
    plan=json.loads((out/'manifest.json').read_text());current(plan)
    target=out/'candidate';target.mkdir();settings,opts=runtime(plan['profile'],plan['gpu'],target)
    batch_size=min(batch_size,__import__('mellum_delivery_matrix').slots(plan['profile']))
    opts['max_num_seqs']=batch_size
    sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch,transformers
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    from validate_mellum_batch200 import drive_window,read_batches
    if transformers.__version__!='5.2.0' or not torch.cuda.is_available():raise RuntimeError('Frozen GPU interpreter required')
    if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require80GiB free')
    write(out/'runtime.json',{'settings':settings,'engine_options':opts,'note':'Same profile switches as serving; raw token IDs avoid HTTP rendering ambiguity'})
    hf_environment=json.loads(Path(plan['references']['humaneval']).read_text())['environment']
    if torch.__version__!=hf_environment['torch']:raise ValueError('Reference and candidate Torch versions differ')
    engine=None
    try:
        engine=EngineArgs(**opts).create_engine()
        for name in (suites or ['tokens','long','humaneval']):
            inputs=json.loads((out/(name+'_inputs.json')).read_text());rows=inputs['records']
            cases=[{'name':r['name'],'input_ids':r['input_ids'][0],'budget':r['generation_settings']['max_new_tokens']} for r in rows]
            result={**inputs,'producer':'atom','status':'started','profile':plan['profile'],'records':[],
                'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'hip':torch.version.hip,'gpu':plan['gpu']}}
            def progress(done,*args):
                result['records']=[{**rows[i],'generated_ids':[done[i]['token_ids']]} for i in sorted(done)]
                write(out/(name+'_generation.json'),result);print(name,'completed',len(done),'/',len(rows),flush=True)
            outputs,_,_=drive_window(engine,cases,lambda c:SamplingParams(temperature=0,max_tokens=c['budget'],ignore_eos=False),batch_size,progress)
            result.update(status='complete',records=[{**r,'generated_ids':[v['token_ids']]} for r,v in zip(rows,outputs)])
            validate_candidate(inputs,result);write(out/(name+'_generation.json'),result)
        write(out/'batch_evidence.json',read_batches(target/'batches.jsonl'))
    finally:
        if engine is not None:engine.close()


def evaluate(out,plan,budget):
    deadline=time.monotonic()+budget;summary={'status':'started','profile':plan['profile'],'scope':plan['scope']}
    try:
        current(plan)
        run_stage('candidate',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(out)],out/'candidate.log',timeout=max(1,deadline-time.monotonic()))
        current(plan)
        for name in ['tokens','long']:
            summary[name]=token_comparison(json.loads(Path(plan['references'][name]).read_text()),json.loads((out/(name+'_generation.json')).read_text()))
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        tasks=[json.loads(x) for x in Path('/data/datasets/HumanEval.jsonl').read_text().splitlines() if x.strip()]
        generation=json.loads((out/'humaneval_generation.json').read_text());inputs=json.loads((out/'humaneval_inputs.json').read_text())
        summary['candidate_extraction']=extract_all(inputs,generation,tasks,tokenizer,out,'candidate')
        run_stage('candidate_score',['/opt/venv/bin/python',str(PROJECT/'scripts/score_mellum_humaneval.py'),
            '--samples',str(out/'candidate_samples.jsonl'),'--problems','/data/datasets/HumanEval.jsonl','--output',str(out/'candidate_score')],
            out/'candidate_score.log',timeout=min(1200,max(1,deadline-time.monotonic())))
        verdict=lambda f:[json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
        summary['humaneval']=paired_scores(verdict(HE/'hf_score/samples.jsonl_results.jsonl'),verdict(out/'candidate_score/samples.jsonl_results.jsonl'))
        for name,file in [('hf',HE/'hf_score/score.json'),('candidate',out/'candidate_score/score.json')]:
            if abs(json.loads(file.read_text())['score']['pass@1']-summary['humaneval']['pass_at_1'][name])>1e-12:raise ValueError('Official score and verdicts differ')
        # The same backend audit used by full ABC expects a server subdirectory.
        from run_mellum_full_campaign import audit_backend
        (out/'server').symlink_to(out/'candidate',target_is_directory=True)
        audit_backend(out,plan['profile'],'acceptance')
        wanted=profile_settings(plan['profile'])
        for switch,file,predicate in [('MELLUM_PREFIX_ENABLED','prefix.jsonl',lambda x:sum(x.get('cached_tokens',[]))>0),
                                      ('MELLUM_PREFILL_METADATA_REUSE','metadata.jsonl',lambda x:x.get('reused') is True)]:
            if wanted.get(switch)=='1' and not (switch=='MELLUM_PREFIX_ENABLED' and wanted.get('MELLUM_REFERENCE_PREFIX')=='1'):
                events=[json.loads(x) for x in (out/'candidate'/file).read_text().splitlines()]
                if not any(predicate(x) for x in events):raise ValueError('Required actual feature not observed: '+switch)
        if wanted.get('MELLUM_REFERENCE_PREFIX')=='1':
            events=[json.loads(x) for x in (out/'candidate/batches.jsonl').read_text().splitlines()]
            if not any(r.get('phase')=='prefill' and any(t>0 for t in r.get('cached_tokens') or []) for r in events):
                raise ValueError('No actual reference prefix/chunk cache use')
        summary['local_checks']={'developer200_90_percent':summary['tokens']['all']['at_least_90_percent'],
            'long_group_90_percent':summary['long']['all']['at_least_90_percent'],'16k_90_percent':summary['long']['16k']['at_least_90_percent'],
            'human_eval_absolute_2pp':summary['humaneval']['bf16_2pp_local_check']}
        summary['status']='local_candidate_checks_passed' if all(summary['local_checks'].values()) else 'local_candidate_checks_not_all_passed'
    except Exception:summary.update(status='failed',traceback=traceback.format_exc())
    current(plan);summary['source_unchanged']=True
    summary['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
    write(out/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=PROFILES);p.add_argument('--gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--worker',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--budget-minutes',type=float,default=60);a=p.parse_args()
    if a.worker:worker(a.worker);return
    if not a.profile:p.error('--profile required')
    if not 1<=a.budget_minutes<=120:p.error('Budget1..120min')
    out=Path('/workspace/results/mellum_candidate_acceptance')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan=prepare(out,a.profile,a.gpu)
    if a.prepare_only:print('RESULT acceptance_prepared output',out);return
    r=evaluate(out,plan,a.budget_minutes*60);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
