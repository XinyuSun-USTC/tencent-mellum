"""Full164 functional HumanEval, independent HF/GPU0 and ATOM/GPU1.

No token-position or tensor-equality gate. Uses the already pinned official
scorer and sandbox. Generation/extraction protocol is explicit and shared.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,MODEL,apply_serving,engine_options
from run_mellum_batch_campaign import write,sha,fingerprint
from run_mellum_humaneval import build,extract_completion
from mellum_process_runner import run_stage

EVALUATOR=Path('/workspace/results/humaneval_environment/manifest.json')


def validate_records(inputs,data):
    expected=inputs['records'];actual=data['records']
    if len(actual)!=len(expected) or len({r['source_id'] for r in actual})!=len(actual):
        raise ValueError('Incomplete/duplicate HumanEval generation')
    for a,b in zip(actual,expected):
        if any(a[k]!=b[k] for k in ('source_id','input_ids','generation_settings','enable_thinking')):
            raise ValueError('Generation protocol/input identity differs')
        ids=a['generated_ids']
        if len(ids)!=1 or len(ids[0])>b['generation_settings']['max_new_tokens']:
            raise ValueError('Invalid output budget/shape')


def attention_settings(attention):
    if attention not in ('reference','native'):raise ValueError('Unknown functional attention path')
    native=attention=='native'
    return {'MELLUM_ATTENTION_EXECUTION':attention,'MELLUM_SDPA_DECODE':'0' if native else '1',
            'MELLUM_REQUEST_BATCHING':'0' if native else '1'}


def validate_native_audit(rows,backend,attention):
    target='FusedMoE' if backend=='fused_torch_route' else 'isolated_fused_experts'
    moe=[r for r in rows if r['kind']=='moe']
    if not moe or any(r['backend']!=target for r in moe):raise ValueError('Candidate native MoE audit failed')
    if attention=='native':
        attn=[r for r in rows if r['kind']=='attention']
        if not any(r['phase']=='prefill' and r['backend']=='aiter_varlen' for r in attn):raise ValueError('Native prefill audit missing')
        if not any(r['phase']=='decode' and r['backend'].startswith('paged_attention') for r in attn):raise ValueError('Native decode audit missing')
        if any(r['backend']!='aiter_varlen' and not r['backend'].startswith('paged_attention') for r in attn):raise ValueError('Reference attention fallback observed')


def load_reference(run,manifest,inputs):
    old=json.loads((run/'manifest.json').read_text());file=run/'hf/generation.json'
    if old.get('hf_model_source_sha256')!=manifest['hf_model_source_sha256']:raise ValueError('HF model source differs from reused reference')
    if old['inputs_sha256']!=manifest['inputs_sha256'] or old.get('completed_hf_sha256')!=sha(file):raise ValueError('Reused HF input/completion hash differs')
    cached=json.loads(file.read_text());validate_records(inputs,cached)
    if cached.get('status')!='complete' or cached['environment']['transformers']!='5.10.1':raise ValueError('Reused HF incomplete/wrong environment')
    return cached


def projection_control(batched=False):
    return {'groups':'' if batched else 'L','phases':['prefill','decode']}


def atom_worker(out,inputs,backend,attention='reference',batched=False):
    settings=apply_serving(1,out,True,'reference',True)
    settings.update(MELLUM_MOE_BACKEND=backend,MELLUM_NATIVE_AUDIT_PATH=str(out/'native_calls.jsonl'),
        MELLUM_BATCH_CAMPAIGN_CONFIG=json.dumps(projection_control(batched)))
    settings.update(attention_settings(attention))
    os.environ.update(settings);sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch,transformers
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    from validate_mellum_batch200 import drive_window,read_batches
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    if transformers.__version__!='5.2.0':raise RuntimeError('Frozen ATOM interpreter required')
    if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require80GiB free on GPU1')
    cases=[{'name':r['name'],'input_ids':r['input_ids'][0],'budget':r['generation_settings']['max_new_tokens']} for r in inputs['records']]
    opts=engine_options(max(len(r['input_ids'])+r['budget'] for r in cases),None)
    opts.update(max_num_seqs=32,max_num_batched_tokens=16384,port=8626)
    result={**inputs,'producer':'atom','records':[],'settings':settings,
            'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'gpu':1}}
    engine=None
    try:
        engine=EngineArgs(**opts).create_engine()
        audit=json.loads((out/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:
            raise ValueError('Incomplete ATOM loading audit')
        result['loading_audit']=audit
        def progress(done,*args):
            result['records']=[{**inputs['records'][i],'generated_ids':[done[i]['token_ids']]} for i in sorted(done)]
            write(out/'generation.json',result)
            print('ATOM completed',len(done),'/',len(cases),flush=True)
        outputs,subs,comps=drive_window(engine,cases,
            lambda r:SamplingParams(temperature=0,max_tokens=r['budget'],ignore_eos=False),32,progress)
        result['records']=[{**row,'generated_ids':[value['token_ids']]} for row,value in zip(inputs['records'],outputs)]
        result.update(status='complete',submissions=subs,completions=comps,actual_batches=read_batches(out/'batches.jsonl'))
        if result['actual_batches']['decode']['max_batch_size']!=32:raise ValueError('Actual batch32 not observed')
        native=[json.loads(l) for l in (out/'native_calls.jsonl').read_text().splitlines()]
        validate_native_audit(native,backend,attention)
        validate_records(inputs,result);write(out/'generation.json',result)
    finally:
        if engine is not None:engine.close()


def hf_worker(out,inputs):
    os.environ.update(HIP_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch,transformers
    from transformers import MellumForCausalLM
    if transformers.__version__!='5.10.1':raise RuntimeError('Dedicated HF5.10.1 interpreter required')
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    if torch.cuda.mem_get_info(0)[0]<60*1024**3:raise RuntimeError('Require60GiB free on GPU0')
    model,info=MellumForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='sdpa',experts_implementation='grouped_mm',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Incomplete HF loading audit')
    # HF loading reports contain sets even on successful loads. Keep the same
    # normalized diagnostics used by our original HF validation utility.
    from validate_hf_mellum_gpu import json_default
    info=json.loads(json.dumps(info,default=json_default))
    model=model.to('cuda:0').eval()
    result={**inputs,'producer':'hf','status':'started','records':[],'loading_info':info,
            'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'gpu':0,
                           'experts_implementation':model.config._experts_implementation}}
    for r in inputs['records']:
        ids=torch.tensor(r['input_ids'],device='cuda:0',dtype=torch.long)
        mask=torch.tensor(r['attention_mask'],device='cuda:0',dtype=torch.long)
        with torch.inference_mode():value=model.generate(input_ids=ids,attention_mask=mask,use_cache=True,**r['generation_settings'])
        result['records'].append({**r,'generated_ids':value[:,ids.shape[1]:].cpu().tolist()})
        write(out/'generation.json',result);print('HF completed',len(result['records']),'/',len(inputs['records']),flush=True)
    result['status']='complete';validate_records(inputs,result);write(out/'generation.json',result)


def generators(commands,out,deadline):
    children={};streams={};result={};last=time.monotonic()
    try:
        for name,cmd in commands.items():
            streams[name]=(out/(name+'.log')).open('w')
            children[name]=subprocess.Popen(cmd,stdout=streams[name],stderr=subprocess.STDOUT,start_new_session=True)
            print('STAGE',name,'log',out/(name+'.log'),flush=True)
        while len(result)<len(children):
            if time.monotonic()>deadline:raise TimeoutError('Paired generation reached overall time budget')
            for name,child in children.items():
                if name not in result and child.poll() is not None:
                    result[name]={'returncode':child.returncode,'completed':child.returncode==0}
                    print('STAGE',name,'exited',child.returncode,flush=True)
            if time.monotonic()-last>=15:
                for name in children:
                    file=out/name/'generation.json';completed=0
                    if file.exists():
                        try:completed=len(json.loads(file.read_text())['records'])
                        except (ValueError,KeyError):pass
                    print(name,'completed',completed,'/164','log_bytes',(out/(name+'.log')).stat().st_size,flush=True)
                last=time.monotonic()
            time.sleep(.25)
        return result
    finally:
        for child in children.values():
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:pass
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            if child.poll() is None:child.wait(timeout=3)
        for stream in streams.values():stream.close()


def extract_all(inputs,generation,tasks,tokenizer,out,name):
    validate_records(inputs,generation);problems={r['task_id']:r for r in tasks};samples=[];details=[]
    for r in generation['records']:
        task=problems[r['source_id']];ids=r['generated_ids'][0]
        raw=tokenizer.decode(ids,skip_special_tokens=False);code,status=extract_completion(raw,task)
        samples.append({'task_id':task['task_id'],'completion':code if status=='extracted' else "\nraise RuntimeError('completion extraction failed')\n"})
        details.append({'task_id':task['task_id'],'raw_text':raw,'completion':code,'extraction_status':status,
            'output_tokens':len(ids),'budget_hit':len(ids)>=r['generation_settings']['max_new_tokens'] and (not ids or ids[-1]!=28)})
    (out/(name+'_samples.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in samples))
    write(out/(name+'_extraction.json'),details)
    return {'failures':sum(r['extraction_status']!='extracted' for r in details),'budget_hits':sum(r['budget_hit'] for r in details),
            'output_tokens':sum(r['output_tokens'] for r in details)}


def paired_scores(left,right):
    def keyed(rows):
        if len(rows)!=164 or len({r['task_id'] for r in rows})!=164:raise ValueError('Scoring requires all164 unique tasks')
        if any(not isinstance(r['passed'],bool) for r in rows):raise ValueError('Official verdict must be boolean')
        return {r['task_id']:r['passed'] for r in rows}
    a,b=keyed(left),keyed(right)
    if a.keys()!=b.keys():raise ValueError('Scored task identities differ')
    wins=sum(not a[k] and b[k] for k in a);losses=sum(a[k] and not b[k] for k in a)
    counts={'hf':sum(a.values()),'candidate':sum(b.values())}
    delta=abs(counts['candidate']-counts['hf'])/164*100
    return {'passed_tasks':counts,'pass_at_1':{k:v/164 for k,v in counts.items()},'absolute_difference_pp':delta,
            'candidate_gains':wins,'candidate_losses':losses,'bf16_2pp_local_check':delta<=2 if counts['hf'] else None,
            'changed_tasks':[{'task_id':k,'hf':a[k],'candidate':b[k]} for k in a if a[k]!=b[k]]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--thinking',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--budget-minutes',type=float,default=90)
    p.add_argument('--batched-projections',action='store_true')
    p.add_argument('--attention',choices=['reference','native'],default='reference')
    p.add_argument('--candidate',choices=['fused_torch_route','fused_isolated'],default='fused_torch_route')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--reuse-hf-run',type=Path)
    p.add_argument('--worker',choices=['hf','candidate']);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.worker:
        inputs=json.loads((a.output.parent/'inputs.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
        try:
            (hf_worker if a.worker=='hf' else lambda o,i:atom_worker(o,i,a.candidate,a.attention,a.batched_projections))(a.output,inputs)
        except Exception:
            write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise
        return 0
    if a.batched_projections and (a.attention!='native' or a.candidate!='fused_torch_route'):p.error('Batched projection follow-up requires native attention and fused_torch_route')
    if a.attention=='native' and not a.reuse_hf_run:p.error('Native functional follow-up requires --reuse-hf-run; no HF/GPU0 launch')
    if not 256<=a.max_new_tokens<=8192 or not 0<a.budget_minutes<=240:p.error('Budget tokens256..8192; time>0 <=240min')
    out=Path('/workspace/results/mellum_humaneval_pair')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    tasks=[json.loads(l) for l in Path('/data/datasets/HumanEval.jsonl').read_text().splitlines() if l.strip()]
    if len(tasks)!=164 or len({r['task_id'] for r in tasks})!=164:raise ValueError('Expected full164 HumanEval tasks')
    inputs=build(tokenizer,tasks,a.max_new_tokens,a.thinking);write(out/'inputs.json',inputs)
    (out/'problems.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
    if not EVALUATOR.exists():raise RuntimeError('Pinned evaluator setup missing')
    evaluator=json.loads(EVALUATOR.read_text())
    for f,h in evaluator['files_sha256'].items():
        if sha(Path(evaluator['source'])/f)!=h:raise ValueError('Official evaluator hash differs')
    sources=fingerprint();sources.update({str(PROJECT/'scripts'/f):sha(PROJECT/'scripts'/f) for f in
        ('run_mellum_humaneval_pair.py','run_mellum_humaneval.py','score_mellum_humaneval.py','humaneval_sandbox.py','validate_hf_mellum_gpu.py')})
    hf_source=Path('/workspace/venvs/hf-reference/lib/python3.12/site-packages/transformers/models/mellum/modeling_mellum.py')
    sources[str(hf_source)]=sha(hf_source)
    for file in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
        if (MODEL/file).exists():sources[str(MODEL/file)]=sha(MODEL/file)
    m={'status':'prepared','tasks':164,'thinking':a.thinking,'max_new_tokens':a.max_new_tokens,'candidate':a.candidate,'attention':a.attention,'batched_projections':a.batched_projections,
       'hf_model_source_sha256':sha(hf_source),'candidate_batch':32,'hf_gpu':0,'candidate_gpu':1,'inputs_sha256':sha(out/'inputs.json'),
       'dataset_sha256':sha('/data/datasets/HumanEval.jsonl'),'evaluator':evaluator,'source_sha256':sources,
       'protocol':'Local paired chat/full-code extraction protocol, official HumanEval scorer; not asserted judge-identical',
       'token_position_gate':False,'tensor_equality_gate':False,'time_budget_minutes':a.budget_minutes,
       'template_suffix':inputs['records'][0]['rendered_prompt'][-160:]}
    write(out/'manifest.json',m)
    cached=load_reference(a.reuse_hf_run,m,inputs) if a.reuse_hf_run else None
    if a.prepare_only:print('RESULT humaneval_pair_prepared output',out);return 0
    summary={'status':'started','protocol':m['protocol']};deadline=time.monotonic()+a.budget_minutes*60
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('Overall experiment time budget exceeded')
        return value
    try:
        run_stage('evaluator_selftest',['/opt/venv/bin/python',str(PROJECT/'scripts/score_mellum_humaneval.py'),'--self-test'],out/'selftest.log',timeout=min(600,remaining()))
        commands={'candidate':['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker','candidate','--candidate',a.candidate,'--attention',a.attention,'--output',str(out/'candidate')]}
        if a.batched_projections:commands['candidate'].append('--batched-projections')
        if a.reuse_hf_run:
            (out/'hf').mkdir();write(out/'hf/generation.json',cached);summary['reused_hf']=str(a.reuse_hf_run)
        else:commands['hf']=['/workspace/venvs/hf-reference/bin/python',str(Path(__file__).resolve()),'--worker','hf','--output',str(out/'hf')]
        summary['generation_stages']=generators(commands,out,deadline)
        for name in ('hf','candidate'):
            file=out/name/'generation.json'
            if not file.exists():continue
            gen=json.loads(file.read_text())
            if gen.get('status')!='complete':continue
            summary[name+'_extraction']=extract_all(inputs,gen,tasks,tokenizer,out,name)
            if name=='hf':m['completed_hf_sha256']=sha(file);write(out/'manifest.json',m)
            run_stage(name+'_score',['/opt/venv/bin/python',str(PROJECT/'scripts/score_mellum_humaneval.py'),
                '--samples',str(out/(name+'_samples.jsonl')),'--problems',str(out/'problems.jsonl'),
                '--output',str(out/(name+'_score'))],out/(name+'_score.log'),timeout=min(900,remaining()))
        if all((out/(n+'_score')/'samples.jsonl_results.jsonl').exists() for n in ('hf','candidate')):
            scores={n:[json.loads(l) for l in (out/(n+'_score')/'samples.jsonl_results.jsonl').read_text().splitlines()] for n in ('hf','candidate')}
            summary.update(paired_scores(scores['hf'],scores['candidate']),status='paired_functional_evaluation_complete')
            for name in ('hf','candidate'):
                official=json.loads((out/(name+'_score')/'score.json').read_text())['score']['pass@1']
                if abs(official-summary['pass_at_1'][name])>1e-12:raise ValueError('Official pass@1 and task verdicts differ')
            hf=json.loads((out/'hf/generation.json').read_text());atom=json.loads((out/'candidate/generation.json').read_text())
            if hf['environment']['torch']!=atom['environment']['torch']:raise ValueError('HF/ATOM Torch environment differs')
            summary['candidate_configuration']={'gpu':1,'attention':a.attention,'moe':a.candidate,'batched_projections':a.batched_projections,'prefix_caching':False}
            summary['exact_full_sequences']=sum(a['generated_ids']==b['generated_ids'] for a,b in zip(hf['records'],atom['records']))
            summary['interpretation_notes']=['Token differences do not gate functional scoring; public200 and16k remain independent requirements',
                'Local thinking/prompt/extraction/budget protocol; compare truncation and reference quality before interpreting pass@1',
                'Single generation per task; no repeated-run reliability claim for nondeterministic candidate']
        else:summary['status']='partial_generation_or_scoring'
        for f,h in sources.items():
            if sha(f)!=h:raise ValueError('Source changed during paired evaluation')
    except KeyboardInterrupt:summary.update(status='interrupted')
    except Exception:summary.update(status='failed',traceback=traceback.format_exc())
    # Keep a completed HF reference reusable even if its paired producer times out.
    file=out/'hf/generation.json'
    if file.exists():
        try:
            cached=json.loads(file.read_text())
            if cached.get('status')=='complete':
                validate_records(inputs,cached);m['completed_hf_sha256']=sha(file);write(out/'manifest.json',m)
        except Exception:summary['reference_validation_error']=traceback.format_exc()
    summary['source_unchanged']=all(sha(f)==h for f,h in sources.items())
    if not summary['source_unchanged']:summary.update(status='failed',source_error='Source changed during evaluation')
    write(out/'summary.json',summary)
    (out/'report.md').write_text('# Full HumanEval functional pair\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')
    print('RESULT',summary['status'],'output',out);return 0
if __name__=='__main__':raise SystemExit(main())
