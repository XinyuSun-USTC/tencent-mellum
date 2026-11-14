"""Small reference B0 validation: bs1, ragged 2/4, reorder/reuse and late arrival."""
import argparse,json,hashlib,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from mellum_runtime_config import PROJECT,apply_serving,engine_options
from mellum_process_runner import run_stage


def prepare():
    cases=[]
    for path,selected in [(Path('/workspace/results/mellum_validation/20261113T104458011376Z/hf/reference.json'),[(0,16),(1,32)]),
                          (Path('/workspace/results/mellum_validation/20261114T112559495973Z/hf/reference.json'),[(0,8),(2,8)])]:
        source=json.loads(path.read_text())
        for index,budget in selected:
            row=source['records'][index]
            if len(row['generated_ids'][0])<budget:raise ValueError('HF source too short')
            cases.append({'name':row['name'],'input_ids':row['input_ids'][0],'rendered_prompt':row['rendered_prompt'],
                'budget':budget,'expected_ids':row['generated_ids'][0][:budget],
                'hf_source':str(path),'hf_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return cases


def audit_summary(path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    return {phase:sorted({r['batch_size'] for r in rows if r['phase']==phase}) for phase in ['prefill','decode']}


def worker(out):
    plan=json.loads((out/'plan.json').read_text());settings=apply_serving(plan['gpu'],out,True,'reference',True)
    sys.path.insert(0,str(PROJECT/'ATOM'))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No GPU visible')
    if torch.cuda.mem_get_info(0)[0]<80*1024**3:raise RuntimeError('Require 80GiB free before model loading')
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    def params(case):return SamplingParams(temperature=0,max_tokens=case['budget'],ignore_eos=False)
    options=engine_options(1152,None);options.update(torch_profiler_dir=None,max_num_batched_tokens=4096,max_num_seqs=plan['max_seqs'],port=8026)
    result={'status':'started','settings':settings,'engine_options':options,'stages':[]}
    def save():(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    engine=None
    try:
        engine=EngineArgs(**options).create_engine();audit=json.loads((out/'loading_audit.json').read_text())
        if audit!={'source_tensors':5631,'successful_parts':5631,'missing':[],'unexpected':[]}:raise RuntimeError('Loading audit failed')
        result['loading_audit']=audit
        def finish_stage(name,cases,outputs,start):
            if len(outputs)!=len(cases):raise RuntimeError('Lost outputs')
            rows=[]
            for case,output in zip(cases,outputs):
                got=output['token_ids'];wanted=case['expected_ids']
                if not got or len(got)>case['budget']:raise RuntimeError('Invalid output budget')
                rows.append({'name':case['name'],'input_tokens':len(case['input_ids']),'expected_ids':wanted,'actual_ids':got,
                    'exact':got==wanted,'matching_positions':sum(x==y for x,y in zip(got,wanted)),
                    'compared_positions':max(len(got),len(wanted)),'finish_reason':output['finish_reason']})
            phase={'name':name,'elapsed_seconds':time.monotonic()-start,'records':rows,
                   'actual_batches':audit_summary(out/'batches.jsonl')}
            result['stages'].append(phase);save();print(name,'exact',sum(r['exact'] for r in rows),'/',len(rows),'batch',phase['actual_batches'],flush=True)
        def generate(name,cases):
            (out/'batches.jsonl').unlink(missing_ok=True);start=time.monotonic()
            outputs=engine.generate([c['input_ids'] for c in cases],[params(c) for c in cases])
            # Preserve per-stage actual metadata; exclude model warmup.
            finish_stage(name,cases,outputs,start)
            if (out/'batches.jsonl').exists():(out/(name+'_batches.jsonl')).write_bytes((out/'batches.jsonl').read_bytes())
        cases=plan['cases']
        for i,c in enumerate(cases):generate('single_'+str(i),[c])
        generate('pair',cases[:2]);generate('ragged4',cases);generate('reverse_reuse',list(reversed(cases)))
        for count in [8,16,32]:
            if count<=plan['max_seqs']:generate('capacity_'+str(count),[cases[i%4] for i in range(count)])
        # Add two new requests after a shorter initial request finishes while
        # another initial request still has tokens to generate.
        (out/'batches.jsonl').unlink(missing_ok=True);start=time.monotonic();order=[];outputs={}
        def submit(group):
            before=set(engine.io_processor.requests)
            engine.add_request([c['input_ids'] for c in group],[params(c) for c in group])
            ids=sorted(set(engine.io_processor.requests)-before)
            if len(ids)!=len(group):raise RuntimeError('Submission identity mismatch')
            order.extend(zip(ids,group))
        submit(cases[:2]);inserted=False
        while not engine.is_finished():
            seqs=engine.step();outputs.update(engine.io_processor.postprocess(seqs))
            if seqs and not inserted:
                result['late_insertion_pending_requests']=len(engine.io_processor.requests)
                submit(cases[2:]);inserted=True
        if not inserted:raise RuntimeError('Late arrival stage did not insert requests')
        finish_stage('late_arrival',[c for _,c in order],[outputs[i] for i,_ in order],start)
        if (out/'batches.jsonl').exists():(out/'late_arrival_batches.jsonl').write_bytes((out/'batches.jsonl').read_bytes())
        if not any(any(n>1 for n in stage['actual_batches']['decode']) for stage in result['stages']):
            raise RuntimeError('No real multi-request decode observed')
        result['status']='batch_smoke_completed';result['accuracy_review_required']=any(not r['exact'] for s in result['stages'] for r in s['records']);save()
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());save();raise
    finally:
        if engine is not None:engine.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--max-seqs',type=int,choices=[4,8,16,32],default=4)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.worker:worker(a.worker);return 0
    out=Path('/workspace/results/mellum_batch_validation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan={'cases':prepare(),'gpu':a.gpu,'max_seqs':a.max_seqs,'scope':'Reference bridge small validation, not public-set or performance acceptance'}
    files=[Path(__file__),PROJECT/'ATOM/atom/models/mellum.py',PROJECT/'ATOM/atom/model_config/mellum_batch.py',PROJECT/'scripts/mellum_runtime_config.py']
    plan['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if a.prepare_only:print('RESULT prepared output',out);return 0
    run_stage('batch',['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker',str(out)],out/'batch.log',timeout=1200)
    print('RESULT batch_smoke_completed output',out);return 0
if __name__=='__main__':raise SystemExit(main())
