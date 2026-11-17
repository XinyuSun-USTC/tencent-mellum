"""Full cumulative delivery matrix: frozen rows, local correctness and A512/B50/C200."""
import argparse,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_delivery_matrix import ROWS
from mellum_perf_protocol import sha
import run_mellum_full_campaign as perf
import run_mellum_candidate_acceptance as acceptance

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def check(plan):
    for f,h in plan['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Matrix source changed: '+f)

def prepare(out,gpu,rows,kind):
    profiles=[ROWS[r]['profile'] for r in rows];d=out/'performance';d.mkdir()
    performance=perf.prepare(d,gpu,profiles)
    # B first provides a quicker live server check; workloads themselves unchanged.
    performance['template']['stages']={k:performance['template']['stages'][k] for k in ('B','C','A')}
    write(d/'manifest.json',performance)
    plan={'gpu':gpu,'rows':rows,'kind':kind,'definitions':ROWS,'source_sha256':acceptance.perf.sources(),
          'scope':'Local full-scale cumulative matrix, not judge idle-machine three-repeat median',
          'tasks':{},'default_row':'step0123'}
    plan['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    if kind!='perf':
        for row in rows:
            target=out/row;target.mkdir();acceptance.prepare(target,ROWS[row]['profile'],gpu)
    write(out/'manifest.json',plan);return plan

def render(out,plan):
    f=out/'performance/summary.json';p=json.loads(f.read_text()) if f.exists() else {}
    lines=['# 最终累计消融（本地完整负载）','','Step0为最终源码关闭三项优化的重跑；单槽与32槽差异属于优化①。不是历史源码逐字节复现。失败保留分母，未完成不填性能。','',
           '| 行 | 槽数 | A tok/s/GPU | B TTFT p50/p99 ms | B TPOT p50/p99 ms | C TTFT/TPOT p99 ms | A/B/C完成数 | 正确性 |',
           '|---|---:|---:|---|---|---|---|---|']
    for row in plan['rows']:
        m=p.get('metrics',{}).get(ROWS[row]['profile'],{})
        def fmt(stage,keys,scale=1000):
            v=m.get(stage,{})
            if not v.get('complete_without_failures'):return '待测或未完成'
            return '/'.join(f'{v[k]*scale:.3f}' if v.get(k) is not None else '—' for k in keys)
        counts='/'.join(f"{m[s].get('successes')}/{m[s].get('requests')}" if s in m else '—' for s in ('A','B','C'))
        a=out/row/'summary.json';status=json.loads(a.read_text()).get('status') if a.exists() else '待测'
        lines.append(f"| {row} | {ROWS[row]['slots']} | {fmt('A',['output_tokens_per_s_gpu'],1)} | {fmt('B',['ttft_p50_s','ttft_p99_s'])} | {fmt('B',['tpot_p50_s','tpot_p99_s'])} | {fmt('C',['ttft_p99_s','tpot_p99_s'])} | {counts} | {status} |")
    (out/'matrix.md').write_text('\n'.join(lines)+'\n')

def execute(out,plan,minutes):
    check(plan)
    if 'entrypoint_smoke' not in plan['tasks']:
        import sys
        from mellum_process_runner import run_stage
        run_stage('delivery_entrypoint_smoke',[sys.executable,str(Path(__file__).with_name('smoke_delivery.py')),
            '--gpu',str(plan['gpu']),'--modes',*plan['rows'],'--output',str(out/'entrypoint_smoke')],
            out/'entrypoint_smoke.log',timeout=900,allowed_exit_codes=(0,1))
        smoke=out/'entrypoint_smoke/summary.json'
        if not smoke.exists():
            write(out/'summary.json',{'status':'entrypoint_smoke_failed','reason':'Child produced no summary; inspect entrypoint_smoke.log','source_unchanged':True})
            raise RuntimeError('Entrypoint smoke exited without summary; see '+str(out/'entrypoint_smoke.log'))
        plan['tasks']['entrypoint_smoke']=json.loads(smoke.read_text())['status']
        write(out/'manifest.json',plan)
    if plan['tasks'].get('entrypoint_smoke')!='smoke_passed':
        write(out/'summary.json',{'status':'entrypoint_smoke_failed','tasks':plan['tasks'],'source_unchanged':True})
        raise RuntimeError('Entrypoint smoke failed; full campaign not started. Inspect entrypoint_smoke/summary.json')
    if plan['kind']!='acceptance':
        p=json.loads((out/'performance/manifest.json').read_text())
        r=perf.execute(out/'performance',p,minutes*60)
        plan['tasks']['performance']=r['status'];write(out/'manifest.json',plan);render(out,plan)
    if plan['kind']!='perf':
        for row in plan['rows']:
            check(plan);target=out/row
            if (target/'summary.json').exists():
                old=json.loads((target/'summary.json').read_text())
                if old.get('status')=='local_candidate_checks_passed':
                    for f,h in old['artifact_sha256'].items():
                        if sha(target/f)!=h:raise ValueError('Completed acceptance artifact changed')
                    continue
                # Failed evidence is immutable; a retry needs a new matrix run.
                plan['tasks'][row]=old.get('status');continue
            r=acceptance.evaluate(target,json.loads((target/'manifest.json').read_text()),minutes*60)
            plan['tasks'][row]=r['status'];write(out/'manifest.json',plan);render(out,plan)
    check(plan);render(out,plan)
    good=all(x in ('complete','local_candidate_checks_passed','smoke_passed') for x in plan['tasks'].values())
    result={'status':'measured_all_passed' if good else 'measured_with_failures_or_pending','tasks':plan['tasks'],
            'source_unchanged':True,'scope':plan['scope'],
            'artifact_sha256':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}}
    write(out/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--rows',nargs='+',choices=ROWS,required=True);p.add_argument('--kind',choices=['all','perf','acceptance'],default='all')
    p.add_argument('--budget-minutes',type=float,default=180,help='Per performance campaign / per row acceptance; not total run budget')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
    if len(a.rows)!=len(set(a.rows)) or not 10<=a.budget_minutes<=360:p.error('Unique rows and budget10..360 required')
    out=a.resume or Path('/workspace/results/mellum_delivery_matrix')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if a.resume:
        plan=json.loads((out/'manifest.json').read_text())
        if (plan['gpu'],plan['rows'],plan['kind'])!=(a.gpu,a.rows,a.kind):raise ValueError('Resume options differ')
        check(plan)
    else:out.mkdir(parents=True);plan=prepare(out,a.gpu,a.rows,a.kind)
    if a.prepare_only:print('RESULT delivery_matrix_prepared output',out);return
    r=execute(out,plan,a.budget_minutes);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
