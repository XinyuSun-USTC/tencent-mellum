"""Representative developer200 normal greedy64 under a real bounded B0 window."""
import argparse,hashlib,json,sys,time,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT,MODEL,apply_serving,engine_options
from mellum_process_runner import run_stage

SOURCE=Path('/workspace/results/mellum_validation/20261113T104458011376Z')


def prepare(source=SOURCE):
    status=json.loads((source/'summary.json').read_text());file=source/'hf/reference.json';data=json.loads(file.read_text())
    pinned=json.loads((PROJECT/'artifacts/performance/07_representative_correctness/developer200/summary.json').read_text())['source_reference_sha256']
    if hashlib.sha256(file.read_bytes()).hexdigest()!=pinned:raise ValueError('Cached HF reference differs from archived checksum')
    if status['status']!='tokens_and_logits_exact' or len(data['records'])!=200:raise ValueError('Require verified HF200 source')
    if data['model_config_sha256']!=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest():raise ValueError('Model config differs')
    if data['environment']['transformers']!='5.10.1':raise ValueError('Reference version differs')
    cases=[]
    for row in data['records']:
        settings=row['generation_settings'];ids=row['input_ids'][0];expected=row['generated_ids'][0]
        if settings['do_sample'] or settings['max_new_tokens']!=64 or len(expected)!=64:raise ValueError('Source is not greedy64')
        if row['attention_mask']!=[[1]*len(ids)] or row['position_ids']!=[list(range(len(ids)))]:raise ValueError('Source metadata differs')
        if row.get('forced_prefix_ids'):raise ValueError('Forced input is forbidden')
        cases.append({'name':row['name'],'input_ids':ids,'expected_ids':expected,
            'prompt_source':row['prompt_source'],'source_id':row['source_id']})
    if len({r['name'] for r in cases})!=200:raise ValueError('Source names are not unique')
    return cases,{'source_run':str(source),'reference_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'model_config_sha256':data['model_config_sha256'],'reference_environment':data['environment']}


def drive_window(engine,cases,params,window,on_progress,on_submit=None):
    """Map live seq IDs explicitly, replenish only after requests complete."""
    next_index=0;mapping={};results={};submissions=[];completions=[];start=time.monotonic()
    if engine.io_processor.requests:raise RuntimeError('Engine is not initially idle')
    def replenish():
        nonlocal next_index
        capacity=window-len(engine.io_processor.requests)
        if capacity<0:raise RuntimeError('Outstanding window exceeded')
        group=cases[next_index:next_index+capacity]
        if not group:return
        before=set(engine.io_processor.requests)
        if on_submit is None:
            engine.add_request([r['input_ids'] for r in group],[params(r) for r in group])
        else:
            # EngineCore runs asynchronously: publish observer identity BEFORE
            # dispatch, not after add_request (which can race first prefill).
            original=engine.core_mgr.add_request
            def dispatch(reqs):
                if len(reqs)!=len(group):raise RuntimeError('Pre-dispatch request count differs')
                anticipated=dict(mapping)
                for offset,(req,row) in enumerate(zip(reqs,group)):
                    if list(req.token_ids[:len(row['input_ids'])])!=row['input_ids']:
                        raise RuntimeError('Pre-dispatch input mapping differs')
                    anticipated[req.id]=next_index+offset
                on_submit({seq_id:cases[index] for seq_id,index in anticipated.items()})
                return original(reqs)
            engine.core_mgr.add_request=dispatch
            try:engine.add_request([r['input_ids'] for r in group],[params(r) for r in group])
            finally:engine.core_mgr.add_request=original
        ids=sorted(set(engine.io_processor.requests)-before)
        if len(ids)!=len(group):raise RuntimeError('Submission identity mismatch')
        for offset,(seq_id,row) in enumerate(zip(ids,group)):
            seq=engine.io_processor.requests[seq_id]
            if list(seq.token_ids[:len(row['input_ids'])])!=row['input_ids']:raise RuntimeError('Submission input order differs')
            mapping[seq_id]=next_index+offset
        next_index+=len(group)
        if len(engine.io_processor.requests)>window:raise RuntimeError('Outstanding window exceeded')
        submissions.append({'elapsed_seconds':time.monotonic()-start,'seq_ids':ids,
            'names':[r['name'] for r in group],'outstanding':len(engine.io_processor.requests)})
    replenish()
    while not engine.is_finished():
        seqs=engine.step();outputs=engine.io_processor.postprocess(seqs)
        for seq_id,value in outputs.items():
            if seq_id not in mapping or mapping[seq_id] in results:raise RuntimeError('Unknown or duplicate completion')
            index=mapping[seq_id];results[index]=value
            completions.append({'seq_id':seq_id,'name':cases[index]['name'],'index':index,'elapsed_seconds':time.monotonic()-start})
        replenish()
        if outputs:on_progress(results,submissions,completions)
    if len(results)!=len(cases) or next_index!=len(cases):raise RuntimeError('Requests missing at completion')
    return [results[i] for i in range(len(cases))],submissions,completions


def summarize(cases,outputs):
    if len(cases)!=len(outputs):raise ValueError('Missing comparison records')
    rows=[]
    for case,out in zip(cases,outputs):
        got=out['token_ids'];expected=case['expected_ids']
        if not isinstance(got,list) or len(got)>64:raise ValueError('Invalid output token budget')
        first=next((i for i in range(max(len(got),len(expected))) if i>=len(got) or i>=len(expected) or got[i]!=expected[i]),None)
        rows.append({'name':case['name'],'prompt_source':case['prompt_source'],'input_tokens':len(case['input_ids']),
            'actual_ids':got,'expected_ids':expected,'matching_positions':sum(a==b for a,b in zip(got,expected)),
            'compared_positions':max(len(got),len(expected)),'exact':got==expected,
            'first_difference_zero_based':first,'finish_reason':out.get('finish_reason')})
    groups={}
    for name,selection in [('all',rows),('below_1024',[r for r in rows if r['input_tokens']<1024]),
            ('at_least_1024',[r for r in rows if r['input_tokens']>=1024]),
            ('humaneval',[r for r in rows if r['prompt_source']=='HumanEval']),
            ('scenario_a',[r for r in rows if r['prompt_source']=='supplied_scenario_a'])]:
        count=len(selection);total=sum(r['compared_positions'] for r in selection);match=sum(r['matching_positions'] for r in selection)
        exact=sum(r['exact'] for r in selection)
        groups[name]={'records':count,'matching_positions':match,'compared_positions':total,
            'token_match_fraction':match/total if total else None,'exact_sequences':exact,
            'exact_sequence_fraction':exact/count if count else None}
    return rows,groups


def read_batches(path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {phase:{'calls_by_batch_size':dict(sorted(Counter(r['batch_size'] for r in rows if r['phase']==phase).items())),
                   'max_batch_size':max((r['batch_size'] for r in rows if r['phase']==phase),default=0)} for phase in ['prefill','decode']}


def worker(out):
    plan=json.loads((out/'plan.json').read_text());settings=apply_serving(plan['gpu'],out,True,'reference',True)
    sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch,transformers
    if not torch.cuda.is_available():raise RuntimeError('No visible GPU')
    if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require80GiB free before loading')
    if transformers.__version__!='5.2.0' or torch.__version__!=plan['provenance']['reference_environment']['torch']:
        raise RuntimeError('Frozen runtime differs from reference')
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    cases=plan['cases'];options=engine_options(max(len(r['input_ids']) for r in cases)+64,None)
    options.update(torch_profiler_dir=None,max_num_batched_tokens=16384,max_num_seqs=plan['max_seqs'],port=8026)
    result={'status':'started','scope':plan['scope'],'settings':settings,'engine_options':options,
            'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'hip':torch.version.hip}}
    def save():(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save();engine=None
    try:
        engine=EngineArgs(**options).create_engine();audit=json.loads((out/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise RuntimeError('Loading audit differs')
        result['loading_audit']=audit;(out/'batches.jsonl').unlink(missing_ok=True)
        def progress(results,submissions,completions):
            result['completed_requests']=len(results);save()
            (out/'request_progress.json').write_text(json.dumps({'submissions':submissions,'completions':completions},indent=2)+'\n')
            print('BATCH200 completed',len(results),'/',len(cases),flush=True)
        start=time.monotonic()
        outputs,submissions,completions=drive_window(engine,cases,lambda r:SamplingParams(temperature=0,max_tokens=64,ignore_eos=False),plan['max_seqs'],progress)
        result['generation_wall_seconds']=time.monotonic()-start
        records,groups=summarize(cases,outputs)
        (out/'comparison.json').write_text(json.dumps({'records':records,'groups':groups},indent=2)+'\n')
        batches=read_batches(out/'batches.jsonl')
        if batches['decode']['max_batch_size']!=plan['max_seqs']:raise RuntimeError('Requested decode batch size not observed')
        result.update(status='batch200_comparison_complete',groups=groups,actual_batches=batches,
            submitted_requests=sum(len(s['names']) for s in submissions),completed_requests=len(completions),
            developer_90_percent_checks={'token_position_interpretation':groups['all']['token_match_fraction']>=.9,
                'whole_sequence_interpretation':groups['all']['exact_sequence_fraction']>=.9,
                'scope':'Developer set only, not confirmed official public prompts; no logits equality gate'},
            note='Audit enabled; wall is not official A/B/C performance or a speedup claim')
        save()
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise
    finally:
        if engine is not None:engine.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--max-seqs',type=int,choices=[1,4,8,16,32],default=32)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.worker:worker(a.worker);return 0
    cases,provenance=prepare();out=Path('/workspace/results/mellum_batch200')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    files=[Path(__file__),PROJECT/'scripts/mellum_runtime_config.py',PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'ATOM/atom/model_config/mellum_attention.py',PROJECT/'ATOM/atom/model_config/mellum_batch.py',PROJECT/'ATOM/atom/model_config/mellum_experts.py',PROJECT/'ATOM/atom/model_config/mellum_projection.py']
    plan={'cases':cases,'provenance':provenance,'max_seqs':a.max_seqs,'gpu':a.gpu,
          'scope':'B0 developer200, normal greedy64, bounded outstanding window; no official performance/accuracy claim',
          'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if a.prepare_only:print('RESULT prepared output',out);return 0
    run_stage('batch200',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(out)],out/'batch200.log',timeout=1800)
    print('RESULT batch200_comparison_complete output',out);return 0
if __name__=='__main__':raise SystemExit(main())
