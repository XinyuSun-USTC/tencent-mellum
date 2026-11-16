"""Acceptance and capacity follow-up for an immutable, screened source worktree.

No inference implementation is edited. C20 is screening only; C200 requires
completed acceptance. Both retain rate0.3, output256 and per-request timeout180.
"""
import argparse,json,sys,time,traceback,hashlib
from pathlib import Path
from datetime import datetime,timezone
ALLOWED=('r_reference_kv_once','r_expert_decode2','r_expert_decode4')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_screen(directory,profile,source):
    result=json.loads((directory/'summary.json').read_text());plan=json.loads((directory/'manifest.json').read_text())
    if result.get('status')!='screen_passed_full_acceptance_pending' or not result.get('source_unchanged'):
        raise ValueError('Require a completed passing screen')
    if profile not in ALLOWED:raise ValueError('Unsupported screened profile')
    if profile=='r_reference_kv_once':
        if plan['profile']!=profile or not result['generation']['developer200_gate']:raise ValueError('Candidate did not pass')
        path=Path(plan['candidate_plan'])
        if digest(path)!=plan['candidate_plan_sha256']:raise ValueError('Screen candidate plan changed')
    else:
        group=profile[-1]
        if not result['candidates'].get(group,{}).get('developer200_gate'):raise ValueError('Requested group did not pass')
        item=plan['plans'][group];path=Path(item['path'])
        if digest(path)!=item['sha256']:raise ValueError('Screen candidate plan changed')
    candidate=json.loads(path.read_text())
    if candidate['profile']!=profile:raise ValueError('Profile identity differs')
    expected=source/'ATOM/atom/models/mellum.py'
    if str(expected) not in candidate['source_sha256']:raise ValueError('Screen source worktree differs')
    for name,h in candidate['source_sha256'].items():
        if digest(name)!=h:raise ValueError('Screened source changed: '+name)
    if not result.get('artifact_sha256'):raise ValueError('Screen has no hashed artifacts')
    for name,h in result['artifact_sha256'].items():
        if digest(directory/name)!=h:raise ValueError('Screen artifact changed: '+name)
    if not result.get('paired_perf'):raise ValueError('Require reduced paired performance evidence before full follow-up')
    return candidate,result


def read_accepted_run(directory,profile,source):
    verdict=json.loads((directory/'summary.json').read_text());plan=json.loads((directory/'manifest.json').read_text())
    if verdict['status']!='local_candidate_checks_passed' or plan['profile']!=profile:
        raise ValueError('Full acceptance not passed for this profile')
    if plan.get('source_root')!=str(source) or plan.get('batch_size')!=32:
        raise ValueError('Accepted source worktree or capacity differs')
    if not verdict.get('artifact_sha256'):raise ValueError('Acceptance has no hashed artifacts')
    for name,h in verdict['artifact_sha256'].items():
        if digest(directory/name)!=h:raise ValueError('Acceptance artifact changed')
    return plan


def load_environment(source):
    sys.path.insert(0,str(source/'scripts'))
    import run_mellum_candidate_acceptance as acceptance
    import run_mellum_abc_baseline as perf
    from mellum_process_runner import run_stage
    from run_mellum_batch_campaign import write
    if Path(acceptance.__file__).resolve().parents[1]!=source:raise ValueError('Wrong source worktree imported')
    return acceptance,perf,run_stage,write


def verify_feature(out,profile):
    target=out/'candidate'
    if profile=='r_reference_kv_once':
        rows=[json.loads(l) for l in (target/'native_calls.jsonl').read_text().splitlines() if l]
        rows=[r for r in rows if r.get('kind')=='kv_write']
        if {r['phase'] for r in rows}!={'prefill','decode'} or any(r.get('backend')!='aiter_reference_kv_once' or r.get('rope_backend')!='torch' or r.get('resident_inverse') for r in rows):
            raise ValueError('Reference-only KV feature receipt missing')
    else:
        size=int(profile[-1]);rows=[json.loads(l) for l in (target/'expert_groups.jsonl').read_text().splitlines() if l]
        if not any(r['phase']=='decode' and r['requests']==32 for r in rows):raise ValueError('Actual decode32 missing')
        for r in rows:
            group=size if r['phase']=='decode' else 1
            if r['decode_group_size']!=group or r['expert_groups']!=(r['requests']+group-1)//group:raise ValueError('Actual grouping differs')
    return len(rows)


def acceptance_worker(acceptance,out,profile):
    original=acceptance.runtime
    def runtime(name,gpu,target):
        env,opts=original(name,gpu,target)
        if name.startswith('r_expert_decode'):
            import os
            cfg=json.loads(env['MELLUM_BATCH_CAMPAIGN_CONFIG']);cfg['moe_audit']=str(target/'expert_groups.jsonl')
            env['MELLUM_BATCH_CAMPAIGN_CONFIG']=json.dumps(cfg,sort_keys=True);os.environ.update(env)
        return env,opts
    acceptance.runtime=runtime
    acceptance.worker(out,suites=['humaneval','tokens','long'],batch_size=32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--profile',choices=ALLOWED,required=True);p.add_argument('--screen-run',type=Path)
    p.add_argument('--phase',choices=['acceptance','capacity'],default='acceptance');p.add_argument('--count',type=int,choices=[20,200],default=20)
    p.add_argument('--acceptance-run',type=Path);p.add_argument('--gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--budget-minutes',type=float,default=90);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--worker-kind',choices=['acceptance','capacity']);a=p.parse_args()
    source=a.source_root.resolve();acceptance,perf,run_stage,write=load_environment(source)
    if a.worker:
        if a.worker_kind=='acceptance':acceptance_worker(acceptance,a.worker,a.profile)
        else:perf.stage_worker(a.worker)
        return
    if not a.screen_run:p.error('--screen-run required')
    if not 1<=a.budget_minutes<=120:p.error('Budget1..120min')
    screened,_=read_screen(a.screen_run,a.profile,source)
    if a.phase=='capacity' and a.count==200:
        if not a.acceptance_run:p.error('C200 requires --acceptance-run')
        old=read_accepted_run(a.acceptance_run,a.profile,source)
        acceptance.current(old)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path('/workspace/results/mellum_qualified_followup')/(stamp+'_'+a.phase+'_'+a.profile);out.mkdir(parents=True)
    own=Path(__file__).resolve();own_sha=digest(own)
    if a.phase=='acceptance':
        plan=acceptance.prepare(out,a.profile,a.gpu)
        plan['source_sha256'][str(own)]=own_sha;plan['external_sha256'][str(a.screen_run/'summary.json')]=digest(a.screen_run/'summary.json')
        plan.update(screen_run=str(a.screen_run),source_root=str(source),batch_size=32);write(out/'manifest.json',plan)
        if a.prepare_only:print('RESULT qualified_acceptance_prepared output',out);return
        original=acceptance.run_stage
        def execute(name,command,log,**kwargs):
            if name=='candidate':command=[sys.executable,str(own),'--source-root',str(source),'--profile',a.profile,'--worker',str(out),'--worker-kind','acceptance']
            return original(name,command,log,**kwargs)
        acceptance.run_stage=execute;result=acceptance.evaluate(out,plan,a.budget_minutes*60)
        if result['status']=='local_candidate_checks_passed':
            try:result['actual_feature_events']=verify_feature(out,a.profile)
            except Exception:result.update(status='failed',traceback=traceback.format_exc())
        write(out/'summary.json',result)
    else:
        template=out/'template';template.mkdir();prepared=perf.prepare(template,a.gpu,False)
        case=prepared['stages']['C'][0];case['number']=a.count;case['expected']=case['expected'][:a.count]
        from mellum_perf_protocol import load_rows
        rows={r['id']:r for r in load_rows(case['source'])}
        subset=template/'inputs'/f'C{a.count}.jsonl';subset.write_text(''.join(json.dumps(rows[r['id']],ensure_ascii=False)+'\n' for r in case['expected']))
        case['source']=str(subset);prepared['input_sha256'][str(subset)]=digest(subset)
        stage=out/'C';stage.mkdir();job={'stage':'capacity','gpu':a.gpu,'profile':a.profile,'cases':[case],
            'server_port':9251+a.gpu*100,'engine_port':11826+a.gpu*100,'timeout':a.budget_minutes*60}
        write(stage/'plan.json',job)
        plan={'source_root':str(source),'profile':a.profile,'screen_run':str(a.screen_run),'phase':'capacity','count':a.count,
            'scope':'C20 capacity screening only' if a.count==20 else 'Full local C200 after acceptance; not judge median',
            'own_source_sha256':own_sha,'template':prepared};write(out/'manifest.json',plan)
        if a.prepare_only:print('RESULT capacity_prepared output',out);return
        result={'status':'started','scope':plan['scope']}
        try:
            run_stage('capacity',[sys.executable,str(own),'--source-root',str(source),'--profile',a.profile,'--worker',str(stage),'--worker-kind','capacity'],stage/'worker.log',timeout=a.budget_minutes*60)
            value=json.loads((stage/'results.json').read_text());result['results']=value
            result['status']='capacity_requests_completed' if value['status']=='complete' and all(v['metrics']['complete_without_failures'] for v in value['cases'].values()) else 'capacity_requests_failed'
        except Exception:result.update(status='failed',traceback=traceback.format_exc())
        perf.current_matches(prepared,template)
        if digest(own)!=own_sha:raise ValueError('Follow-up runner changed')
        for name,h in screened['source_sha256'].items():
            if digest(name)!=h:raise ValueError('Candidate source changed')
        result['source_unchanged']=True
        result['artifact_sha256']={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
        write(out/'summary.json',result)
    print('RESULT',result['status'],'output',out,flush=True)
if __name__=='__main__':main()
