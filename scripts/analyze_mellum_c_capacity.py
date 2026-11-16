"""CPU-only completed-run capacity evidence; no model imports or response unpickling."""
import argparse,hashlib,json,sqlite3,statistics
from pathlib import Path
from datetime import datetime,timezone


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def percentile(values,p):
    values=sorted(values)
    if not values:return None
    return values[min(len(values)-1,max(0,__import__('math').ceil(len(values)*p)-1))]


def database(path):
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    try:rows=connection.execute('SELECT success,start_time,completed_time,inter_token_latencies FROM result ORDER BY start_time').fetchall()
    finally:connection.close()
    detail=[]
    for i,(success,start,end,raw) in enumerate(rows):
        intervals=json.loads(raw or '[]')
        if not isinstance(intervals,list) or any(not isinstance(x,(int,float)) or x<0 for x in intervals):raise ValueError('Unexpected interval encoding')
        detail.append({'arrival_index':i,'success':bool(success),'elapsed_s':None if start is None or end is None else end-start,
                       'interval_count':len(intervals),'interval_median_s':statistics.median(intervals) if intervals else None,
                       'interval_p99_s':percentile(intervals,.99)})
    failed=[x for x in detail if not x['success']];elapsed=[x['elapsed_s'] for x in failed if x['elapsed_s'] is not None]
    return {'requests':len(detail),'successes':sum(x['success'] for x in detail),'failures':len(failed),
            'failed_with_intervals':sum(x['interval_count']>0 for x in failed),
            'failed_without_intervals':sum(x['interval_count']==0 for x in failed),
            'failed_elapsed_min_s':min(elapsed) if elapsed else None,'failed_elapsed_max_s':max(elapsed) if elapsed else None,
            'requests_by_arrival':detail,
            'interpretation':'HTTP stream interval counts are not token counts. No intervals does not prove no engine computation. No response pickle is loaded.'}


def batches(path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    runs=[];run=0
    for r in rows:
        if r['phase']=='prefill':run+=1
        elif r['phase']=='decode':
            if run:runs.append(run)
            run=0
    if run:runs.append(run)
    prefills=[r for r in rows if r['phase']=='prefill'];decodes=[r for r in rows if r['phase']=='decode']
    return {'prefill_batches':len(prefills),'decode_batches':len(decodes),
            'max_consecutive_prefill_batches':max(runs,default=0),
            'cached_prefill_batches':sum(any(n>0 for n in r.get('cached_tokens') or []) for r in prefills),
            'max_total_prefill_tokens':max((sum(r['lengths']) for r in prefills),default=0),
            'max_decode_batch':max((r['batch_size'] for r in decodes),default=0),
            'note':'Batch audit has no timestamps; this measures ordering, not stall duration. Server warmup may be included.'}


def lifecycle(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    aborted=set();violations=[];removed=set();receipts=0;released=0
    for row in rows:
        if row.get('kind')=='abort':
            receipts+=1
            ids=set(row['removed']);aborted.update(ids);removed.update(ids)
            released+=row['kv_free_after']-row['kv_free_before']
        elif row.get('kind')=='engine_step':
            overlap=aborted.intersection(row['request_ids'])
            if overlap:violations.append({'request_ids':sorted(overlap),'start_ns':row['start_ns']})
    return {'abort_receipts':receipts,'removed_requests':len(removed),
            'net_blocks_released_at_abort':released,'steps_after_abort_with_removed_ids':violations,
            'lifecycle_receipts_consistent':not violations,
            'note':'File-order audit of removed IDs; no abort receipt is inconclusive about cancellation. Reclaimed blocks do not make timed-out requests successful.'}


def analyze(root):
    summary_path=root/'summary.json';s=json.loads(summary_path.read_text())
    if s.get('status') in ('started','running') or not s.get('source_unchanged') or not s.get('artifact_sha256'):
        raise ValueError('Require a completed run with hashed, unchanged-source evidence')
    for name,h in s['artifact_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Artifact changed: '+name)
    items=[]
    candidates=s.get('candidates') or {'candidate':s}
    for name,candidate in candidates.items():
        value=candidate.get('capacity') or candidate.get('results') or {}
        for stage,c in value.get('cases',{}).items():
            metric=c.get('metrics',{});command=c.get('command',[])
            if '--outputs-dir' not in command:continue
            client=Path(command[command.index('--outputs-dir')+1]);db=list(client.rglob('benchmark_data.db'))
            if len(db)!=1:raise ValueError('Expected one database per case')
            if not db[0].resolve().is_relative_to(root.resolve()):raise ValueError('Database outside completed run')
            info=database(db[0])
            # Worker writes one audit copy beside the case client directory.
            audit=client.parent/'batches.jsonl'
            items.append({'profile':value['server_command'][value['server_command'].index('--profile')+1] if '--profile' in value.get('server_command',[]) else name,
                          'case':stage,'metrics':{k:metric.get(k) for k in ['requests','successes','failures','ttft_p50_s','ttft_p99_s','tpot_p50_s','tpot_p99_s','actual_decode_max_batch','complete_without_failures']},
                          'tail_valid_for_complete_score':info['failures']==0,'database':str(db[0]),'database_evidence':info,
                          'batch_evidence':batches(audit) if audit.exists() else None,
                          'lifecycle_evidence':lifecycle(client.parent.parent/'server/engine_steps.jsonl') if (client.parent.parent/'server/engine_steps.jsonl').exists() else None})
    return {'run':str(root),'status':s['status'],'verified_artifacts':len(s['artifact_sha256']),'cases':items,
            'unmeasured_candidates':{name:c.get('traceback',c.get('status')) for name,c in candidates.items() if not (c.get('capacity') or c.get('results'))}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();out=a.output or Path('/workspace/results/mellum_c_capacity_analysis')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
    result={'scope':'Completed local capacity runs only; different request counts are not equivalent loads','runs':[analyze(x.resolve()) for x in a.runs]}
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# C 场景容量核对','','不同请求规模不构成等负载对照；存在失败时，成功请求分位数不能充当完整 C 成绩。','',
           '| 运行 | 配置 | 完成 | 失败且已有流间隔 | 失败且无流间隔 | 完整尾延迟可用 |','|---|---|---:|---:|---:|---|']
    for run in result['runs']:
        for c in run['cases']:
            d=c['database_evidence'];lines.append(f"| {Path(run['run']).name} | {c['profile']} | {d['successes']}/{d['requests']} | {d['failed_with_intervals']} | {d['failed_without_intervals']} | {c['tail_valid_for_complete_score']} |")
        for name,error in run['unmeasured_candidates'].items():
            lines.append(f"| {Path(run['run']).name} | {name} | 未测量／失败 | — | — | False |")
    lines+=['','无流间隔不证明引擎未计算；批次记录无时间戳，不能据此量化或证明预填充造成的停顿时长。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n');print('RESULT analyzed output',out)
if __name__=='__main__':main()
