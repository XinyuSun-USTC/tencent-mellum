"""Bounded chunked-reference generation smoke then sustained C capacity screening."""
import argparse,json,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT,apply_serving,engine_options
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha,load_rows
from mellum_process_runner import run_stage
from run_mellum_batch_campaign import write
import run_mellum_abc_baseline as perf
PROFILES=('r_prefix_only','r_prefix_chunk1024_d1','r_prefix_chunk2048_d1','r_chunk1024_d1','r_chunk2048_d1','r_expert_decode4_kv_once','r_chunk512_d4','r_chunk1024_d4','r_chunk2048_d4')


def current(plan):
    for name,h in plan['source_sha256'].items():
        if sha(name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in plan['external_sha256'].items():
        if sha(name)!=h:raise ValueError('Reference/input changed: '+name)


def prepare(out,gpu,profiles,count):
    cases=[];external={}
    for file,indices in [(Path('/workspace/results/mellum_validation/20261113T104458011376Z/hf/reference.json'),[0]),
                         (Path('/workspace/results/mellum_validation/20261114T112559495973Z/hf/reference.json'),[0,2]),
                         (Path('/workspace/results/mellum_validation/20261113T103154043872Z/hf/reference.json'),[3])]:
        external[str(file)]=sha(file)
        data=json.loads(file.read_text())
        for i in indices:
            r=data['records'][i];cases.append({'name':r['name'],'input_ids':r['input_ids'][0],'expected_ids':r['generated_ids'][0][:8]})
    template=out/'template';template.mkdir();m=perf.prepare(template,gpu,False)
    case=m['stages']['C'][0];case['number']=count;case['expected']=case['expected'][:count]
    rows={r['id']:r for r in load_rows(case['source'])};subset=template/'inputs'/f'C{count}.jsonl'
    subset.write_text(''.join(json.dumps(rows[r['id']],ensure_ascii=False)+'\n' for r in case['expected']))
    case['source']=str(subset);external[str(subset)]=sha(subset)
    files=list((PROJECT/'ATOM/atom/model_config').glob('mellum*.py'))+[PROJECT/'ATOM/atom/model_engine/scheduler.py',PROJECT/'ATOM/atom/models/mellum.py',Path(__file__).resolve(),PROJECT/'scripts/mellum_execution_profiles.py',PROJECT/'scripts/serve_mellum.py',PROJECT/'scripts/run_mellum_abc_baseline.py']
    plan={'gpu':gpu,'profiles':profiles,'count':count,'cases':cases,'capacity_case':case,'source_sha256':{str(p):sha(p) for p in files},'external_sha256':external,
          'scope':f'Exploratory reference chunking; four-prompt smoke + C{count}, rate0.3 output256 timeout180. Not full correctness acceptance or idle judge median.'}
    write(out/'manifest.json',plan);return plan


def smoke_worker(directory):
    import os
    job=json.loads((directory/'plan.json').read_text());settings=apply_serving(job['gpu'],directory,True,'reference',True)
    settings.update(profile_settings(job['profile']));os.environ.update(settings)
    sys.path.insert(0,str(PROJECT/'ATOM'))
    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    chunked=settings.get('MELLUM_REFERENCE_CHUNKED')=='1'
    options=engine_options(16640,None);options.update(max_num_seqs=32,port=12826+100*job['gpu'],torch_profiler_dir=None,
             enable_prefix_caching=settings.get('MELLUM_REFERENCE_PREFIX')=='1',enable_chunked_prefill=chunked,max_num_batched_tokens=int(settings.get('MELLUM_CHUNK_BUDGET','16640')))
    engine=None;result={'status':'started','profile':job['profile'],'settings':settings,'options':options,'stages':[]}
    try:
        engine=EngineArgs(**options).create_engine()
        for label,cases in [(f'single_{i}',[c]) for i,c in enumerate(job['cases'])]+[('ragged4',job['cases']),('repeat_first',job['cases'][:1])]:
            audit_file=directory/'batches.jsonl'
            before=len(audit_file.read_text().splitlines()) if audit_file.exists() else 0
            start=time.monotonic();outputs=engine.generate([c['input_ids'] for c in cases],[SamplingParams(temperature=0,max_tokens=8,ignore_eos=True) for c in cases])
            if len(outputs)!=len(cases):raise ValueError('Missing smoke outputs')
            records=[]
            for c,o in zip(cases,outputs):
                ids=o['token_ids']
                if len(ids)!=8:raise ValueError('Incomplete generation')
                records.append({'name':c['name'],'input_tokens':len(c['input_ids']),'actual_ids':ids,'expected_ids':c['expected_ids'],'matching_positions':sum(a==b for a,b in zip(ids,c['expected_ids']))})
            actual=[json.loads(x) for x in audit_file.read_text().splitlines()[before:]]
            cache_rows=[r for r in actual if r['phase']=='prefill' and any(n>0 for n in r.get('cached_tokens') or [])]
            result['stages'].append({'name':label,'elapsed':time.monotonic()-start,'records':records,'cached_prefill_batches':len(cache_rows),'cached_token_counts':[r['cached_tokens'] for r in cache_rows]});write(directory/'results.json',result)
            print(label,'completed',flush=True)
        audits=[json.loads(x) for x in (directory/'batches.jsonl').read_text().splitlines()]
        cached=[r for r in audits if r['phase']=='prefill' and any(n>0 for n in r.get('cached_tokens') or [])]
        if (chunked or settings.get('MELLUM_REFERENCE_PREFIX')=='1') and not cached:raise ValueError('No actual cached prefill observed')
        if settings.get('MELLUM_REFERENCE_PREFIX')=='1':
            repeat=result['stages'][-1]['records'][0]
            first=result['stages'][0]['records'][0]
            if repeat['actual_ids']!=first['actual_ids']:raise ValueError('Repeated prefix generation differs from cold output')
            if result['stages'][-1]['cached_prefill_batches']<1:
                raise ValueError('No short repeated-prefix cache hit observed')
        result.update(status='chunk_generation_smoke_completed',cached_prefill_batches=len(cached),accuracy_note='Small smoke only; mismatches recorded, no official gate inferred')
    except Exception:result.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        write(directory/'results.json',result)
        if engine is not None:engine.close()


def execute(out,plan,budget):
    end=time.monotonic()+budget;result={'status':'started','scope':plan['scope'],'candidates':{}}
    for profile in plan['profiles']:
        current(plan);record={};result['candidates'][profile]=record
        try:
            d=out/profile/'smoke';d.mkdir(parents=True);write(d/'plan.json',{'gpu':plan['gpu'],'profile':profile,'cases':plan['cases']})
            run_stage(profile+'_smoke',[sys.executable,str(Path(__file__).resolve()),'--smoke-worker',str(d)],d/'worker.log',timeout=min(900,max(1,end-time.monotonic())))
            record['smoke']=json.loads((d/'results.json').read_text())
            if record['smoke']['status']!='chunk_generation_smoke_completed':raise ValueError('Smoke failed')
            d=out/profile/'capacity';d.mkdir();remaining=end-time.monotonic()
            if remaining<60:raise TimeoutError('Screen budget exhausted')
            write(d/'plan.json',{'stage':'C_chunk_screen','gpu':plan['gpu'],'profile':profile,'cases':[plan['capacity_case']],
                  'server_port':9551+100*plan['gpu'],'engine_port':13026+100*plan['gpu'],'timeout':min(1200,remaining)})
            run_stage(profile+'_C',[sys.executable,str(Path(__file__).resolve()),'--capacity-worker',str(d)],d/'worker.log',timeout=min(1200,remaining))
            record['capacity']=json.loads((d/'results.json').read_text());record['status']='measured'
        except Exception:record.update(status='failed',traceback=traceback.format_exc())
        current(plan);write(out/'summary.json',result)
    result.update(status='screen_complete',source_unchanged=True)
    result['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'summary.json'}
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=['r_chunk1024_d4']);p.add_argument('--count',type=int,choices=[20,60,200],default=60)
    p.add_argument('--budget-minutes',type=float,default=40);p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-worker',type=Path);p.add_argument('--capacity-worker',type=Path);a=p.parse_args()
    if a.smoke_worker:smoke_worker(a.smoke_worker);return
    if a.capacity_worker:perf.stage_worker(a.capacity_worker);return
    if not 5<=a.budget_minutes<=120:p.error('Budget5..120min')
    out=Path('/workspace/results/mellum_c_chunk_screen')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    plan=prepare(out,a.gpu,a.profiles,a.count)
    if a.prepare_only:print('RESULT prepared output',out);return
    r=execute(out,plan,a.budget_minutes*60);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
