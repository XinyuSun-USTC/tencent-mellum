"""Read-only C log analysis: first-admission prefix hits, not chunk-progress sums."""
import argparse,ast,hashlib,json,re,statistics
from pathlib import Path
from datetime import datetime,timezone
STAMP=re.compile(r'\[atom (\d\d):(\d\d):(\d\d)\]')
ARRIVAL=re.compile(r'Request (\d+) arrived, input tokens: (\d+)')
PREFILL=re.compile(r'Scheduled prefill batch: .*?\(cached: (\[[^]]*\]), new: (\[[^]]*\])\), req_ids: (\([^)]*\))')


def parse(text,min_input=4000,max_input=16384):
    arrivals={};events={};day=0;previous=None
    for line in text.splitlines():
        match=STAMP.search(line)
        if not match:continue
        h,m,s=map(int,match.groups());clock=h*3600+m*60+s
        if previous is not None and clock<previous:
            if previous-clock>43200:day+=86400
            else:raise ValueError('Out-of-order log timestamps; cannot measure admission wait')
        previous=clock;t=day+clock
        found=ARRIVAL.search(line)
        if found:
            request,n=map(int,found.groups())
            if min_input<=n<=max_input:
                if request in arrivals:raise ValueError('Duplicate arrival')
                arrivals[request]={'request_id':request,'input_tokens':n,'arrival_s':t}
        found=PREFILL.search(line)
        if not found:continue
        cached,new,ids=[ast.literal_eval(x) for x in found.groups()]
        if not len(cached)==len(new)==len(ids):raise ValueError('Ragged metadata lengths differ')
        for request,c,n in zip(ids,cached,new):
            if request not in arrivals:continue
            if c<0 or n<=0 or c+n>arrivals[request]['input_tokens']:raise ValueError('Invalid prefill progress')
            events.setdefault(request,[]).append({'time_s':t,'cached_tokens':c,'new_tokens':n})
    detail=[]
    for request,row in arrivals.items():
        steps=events.get(request,[]);first=steps[0] if steps else None
        detail.append({**row,'observed_prefill_steps':len(steps),
            'first_admission_wait_s':None if first is None else first['time_s']-row['arrival_s'],
            'first_admission_cached_tokens':None if first is None else first['cached_tokens'],
            'last_observed_prefill_end':None if first is None else steps[-1]['cached_tokens']+steps[-1]['new_tokens'],
            'last_observed_prefill_reached_prompt_end':False if first is None else steps[-1]['cached_tokens']+steps[-1]['new_tokens']==row['input_tokens'],
            'subsequent_chunk_cached_sum_not_prefix_hits':sum(x['cached_tokens'] for x in steps[1:])})
    admitted=[r for r in detail if r['observed_prefill_steps']]
    hits=sum(r['first_admission_cached_tokens'] for r in admitted)
    admitted_tokens=sum(r['input_tokens'] for r in admitted);waits=[r['first_admission_wait_s'] for r in admitted]
    return {'scope':'First observed admission in a complete log; input length filter excludes warmup. Log timestamps have one-second resolution, not kernel timing.',
        'input_filter':[min_input,max_input],'arrivals':len(detail),'observed_admitted':len(admitted),
        'not_observed_admitted':len(detail)-len(admitted),'all_arrived_input_tokens':sum(r['input_tokens'] for r in detail),
        'admitted_input_tokens':admitted_tokens,'first_admission_cached_tokens':hits,
        'first_admission_cache_fraction_of_admitted_input':hits/admitted_tokens if admitted_tokens else None,
        'requests_with_first_admission_hit':sum(r['first_admission_cached_tokens']>0 for r in admitted),
        'admission_wait_p50_s':statistics.median(waits) if waits else None,'admission_wait_max_s':max(waits) if waits else None,
        'requests':detail,'limitations':'Subsequent cached-token counts are chunk progress, not reusable prefix hits. Not observed admitted is not proof of never computed. Prompt-end scheduling is not GPU completion. Does not measure decode stall duration.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--logs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();out=a.output or Path('/workspace/results/mellum_c_prefix_log_analysis')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for path in a.logs:
        raw=path.read_bytes();rows.append({'log':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),**parse(raw.decode())})
    (out/'analysis.json').write_text(json.dumps({'logs':rows},indent=2)+'\n')
    print('RESULT analyzed output',out)
    for row in rows:print(Path(row['log']).parent.parent.name,{k:row[k] for k in ['arrivals','observed_admitted','first_admission_cached_tokens','admission_wait_p50_s','admission_wait_max_s']})
if __name__=='__main__':main()
