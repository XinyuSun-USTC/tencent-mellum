"""Bounded cumulative performance screen of delivery sources; not formal Step0 ABC."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_full_campaign as campaign
from mellum_perf_protocol import sha,load_rows
from run_mellum_batch_campaign import write

PROFILES=['sprint_clean','sprint_graph','sprint_graph_qk_router','sprint_graph_qk_router_prefix_metadata']
LABELS=['Native fast control (not historical Step0)','+ decode graph','+ fused QK and native routing','+ prefix and metadata reuse']


def shorten(plan,out):
    template=out/'template';stages=plan['template']['stages']
    stages['B'][0]['number']=6
    stages['A'][0]['number']=32;stages['A'][0]['expected']=stages['A'][0]['expected'][:32]
    c=stages['C'][0];c['number']=8;c['expected']=c['expected'][:8]
    byid={r['id']:r for r in load_rows(c['source'])}
    source=template/'inputs/C_ablation8.jsonl'
    source.write_text(''.join(json.dumps(byid[r['id']],ensure_ascii=False)+'\n' for r in c['expected']))
    c['source']=str(source);plan['template']['input_sha256'][str(source)]=sha(source)
    plan.update(scope='Reduced A32/B6/C8 cumulative screen; native fast control, NOT historical Step0 and NOT official full ABC',
        matrix_labels=dict(zip(PROFILES,LABELS)),screen_source_sha256=sha(__file__))
    write(template/'manifest.json',plan['template']);write(out/'manifest.json',plan)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--budget-minutes',type=float,default=20);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if not 1<=a.budget_minutes<=25:p.error('Budget must be 1..25 minutes')
    out=Path('/workspace/results/mellum_submission_ablation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True);plan=campaign.prepare(out,a.gpu,PROFILES);shorten(plan,out)
    print('PLAN A32/B6/C8; NOT official cumulative Step0 ablation; output',out,flush=True)
    if a.prepare_only:return
    result=campaign.execute(out,plan,a.budget_minutes*60)
    if sha(__file__)!=plan['screen_source_sha256']:raise RuntimeError('Ablation runner changed')
    result['comparisons']={};result['comparison_note']='Reduced scale; no ratios against full official ABC claimed'
    result['matrix_labels']=plan['matrix_labels'];write(out/'summary.json',result)
    lines=['# Delivery reduced cumulative screen','','A32 / B6 / C8; native fast control, not historical Step0. Not formal assessment-scale ablation.','',
        '| Configuration | A tok/s/GPU | B TPOTp50 ms | C TPOTp99 ms | C TTFTp99 ms |','|---|---:|---:|---:|---:|']
    for profile,label in zip(PROFILES,LABELS):
        metrics=result['metrics'].get(profile,{})
        def metric(stage,key,scale=1):
            row=metrics.get(stage,{})
            if not row.get('complete_without_failures') or row.get(key) is None:return 'not completed'
            return f'{row[key]*scale:.3f}'
        lines.append('| '+label+' | '+metric('A','output_tokens_per_s_gpu')+' | '+metric('B','tpot_p50_s',1000)+' | '+metric('C','tpot_p99_s',1000)+' | '+metric('C','ttft_p99_s',1000)+' |')
    (out/'matrix.md').write_text('\n'.join(lines)+'\n')
    print('RESULT',result['status'],'output',out,flush=True)
if __name__=='__main__':main()
