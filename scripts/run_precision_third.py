"""Precision-first third-point qualification and bounded performance screening; no C capacity gate."""
import argparse,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT
from mellum_execution_profiles import profile_settings
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage
import run_mellum_abc_baseline as perf
import run_mellum_candidate_acceptance as acceptance
import run_mellum_c_service_diagnosis as isolated

CONTROL='r_expert_decode4_kv_once'
CANDIDATES=('r_decode_meta','r_swa_decode4')
ORDER=[('control_before',CONTROL),('metadata',CANDIDATES[0]),('swa_group4',CANDIDATES[1]),('control_after',CONTROL)]


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def verify_result(root):
 s=json.loads((root/'summary.json').read_text())
 if not s.get('source_unchanged'):raise ValueError('Missing source evidence')
 for f,h in s['artifact_sha256'].items():
  if sha(root/f)!=h:raise ValueError('Result changed: '+f)
 return s


def check(plan):
 for group in ['source_sha256','external_sha256']:
  for f,h in plan.get(group,{}).items():
   if sha(f)!=h:raise ValueError('Source/input changed: '+f)


def qualify(root,profile):
 result=verify_result(root)
 if result['status']!='local_candidate_checks_passed' or result['profile']!=profile:
  raise ValueError('Full performance requires this candidate to pass token, long and HumanEval')
 acceptance.current(json.loads((root/'manifest.json').read_text()))
 return sha(root/'summary.json')


def prepare(out,args):
 plan={'gpu':args.gpu,'track':args.track,'settings':{p:profile_settings(p) for p in [CONTROL,*CANDIDATES]},
       'source_sha256':{**perf.sources(),str(Path(__file__).resolve()):sha(__file__),
         str(PROJECT/'scripts/run_mellum_candidate_acceptance.py'):sha(PROJECT/'scripts/run_mellum_candidate_acceptance.py'),
         str(PROJECT/'scripts/run_mellum_c_service_diagnosis.py'):sha(PROJECT/'scripts/run_mellum_c_service_diagnosis.py')},
       'external_sha256':{},'jobs':{},'scope':'Preserve reference computation; evaluate correctness and speed, not C200 completion'}
 if args.track=='accuracy':
  for profile in CANDIDATES:
   d=out/profile;d.mkdir();acceptance.prepare(d,profile,args.gpu)
 elif args.track=='full-performance':
  if args.profile not in CANDIDATES or args.acceptance_run is None:raise ValueError('Select candidate and completed --acceptance-run')
  h=qualify(args.acceptance_run,args.profile);plan.update(profile=args.profile,acceptance_run=str(args.acceptance_run))
  plan['external_sha256'][str(args.acceptance_run/'summary.json')]=h
  d=out/'template';d.mkdir();template=perf.prepare(d,args.gpu,False);plan['template']=template
  plan['order']=[('qualified_candidate',args.profile)]
 else:
  d=out/'template';d.mkdir();template=perf.prepare(d,args.gpu,False)
  template['stages']['A'][0]['number']=32;template['stages']['A'][0]['expected']=template['stages']['A'][0]['expected'][:32]
  template['stages']['B'][0]['number']=4
  plan['template']=template;plan['order']=ORDER
  # Same real C prompts, but simultaneous fixed batches and no mixed-arrival load.
  for label,profile in ORDER:
   d=out/label;d.mkdir();d=d/'decode';d.mkdir()
   isolated.prepare(d,SimpleNamespace(track='decode',gpu=args.gpu,profile=profile,batches=[8,32],tokens=64,observe_seconds=15))
 if 'template' in plan:
  plan['external_sha256'].update(plan['template']['input_sha256'])
  plan['external_sha256'][str(out/'template/inputs/adapter_manifest.json')]=plan['template']['adapter_manifest_sha256']
 write(out/'manifest.json',plan);return plan


def render(out,plan,result):
 lines=['# 精度优先第三项实验','','基线为已通过的参考①②。候选只增加decode元数据复用，或再增加SWA group4；无prefix、chunk、Graph和原生专家。','']
 if plan['track']=='accuracy':
  lines+=['| 候选 | token位置一致率 | 整序列一致 | 长组/16k匹配 | HumanEval（HF145/164） | 绝对差 pp | 完整检查 |','|---|---:|---|---|---|---:|---|']
  for p,a in result.get('accuracy',{}).items():
   if 'humaneval' not in a:lines.append(f'| {p} | — | — | — | — | — | 运行失败 |');continue
   t=a['tokens']['all'];l=a['long']['all'];k=a['long']['16k'];h=a['humaneval']
   lines.append(f"| {p} | {t['position_match_fraction']*100:.4f}% | {t['exact_sequences']}/200 | {l['matching_positions']}/{l['compared_positions']}；{k['matching_positions']}/{k['compared_positions']} | {h['passed_tasks']['candidate']}/164 | {h['absolute_difference_pp']:.4f} | {a['status']} |")
 else:
  scope='完整A512/B50' if plan['track']=='full-performance' else '短负载A32/B4，各256输出；隔离decode batch8/32、输出64'
  lines+=[scope,'','| 配置 | A token/s | A完成数 | B TPOT p50 ms | B完成数 | 隔离decode8/32 token/s |','|---|---:|---|---:|---|---|']
  for label,row in result.get('performance',{}).items():
   m=row.get('metrics',{});a=m.get('A',{});b=m.get('B',{})
   rate=row.get('decode',{}).get('cases',[])
   rates='/'.join(f"{c['delivery']['delivered_tokens_per_second']:.4f}" for c in rate) or '—'
   def val(d,k,scale=1):return '—' if d.get(k) is None else f'{d[k]*scale:.4f}'
   lines.append(f"| {label} | {val(a,'output_tokens_per_s_gpu')} | {a.get('successes','—')}/{a.get('requests','—')} | {val(b,'tpot_p50_s',1000)} | {b.get('successes','—')}/{b.get('requests','—')} | {rates} |")
  lines+=['','隔离decode为最后prefill之后的实际callback交付，不是混合C容量；带阶段审计。短测只作筛选，完整A/B仅对通过正确性的候选执行。']
 (out/'results.md').write_text('\n'.join(lines)+'\n')


def execute(out,plan,budget):
 check(plan);result={'track':plan['track'],'scope':plan['scope']}
 if plan['track']=='accuracy':
  result['accuracy']={}
  for profile in CANDIDATES:
   d=out/profile
   value=verify_result(d) if (d/'summary.json').exists() else acceptance.evaluate(d,json.loads((d/'manifest.json').read_text()),budget*60)
   result['accuracy'][profile]=value;plan['jobs'][profile]=value['status']
   current_out(out,plan,result)
 else:
  if plan['track']=='full-performance':qualify(Path(plan['acceptance_run']),plan['profile'])
  deadline=time.monotonic()+budget*60;result['performance']={}
  for label,profile in plan['order']:
   row={'profile':profile,'metrics':{}};result['performance'][label]=row;d=out/label;d.mkdir(exist_ok=True)
   for stage in ['B','A']:
    target=d/stage;target.mkdir(exist_ok=True)
    job={'stage':stage,'gpu':plan['gpu'],'profile':profile,'cases':plan['template']['stages'][stage],
         'server_port':16251+plan['gpu']*100,'engine_port':18226+plan['gpu']*100,'timeout':min(perf.LIMITS[stage],max(1,deadline-time.monotonic()))}
    write(target/'plan.json',job)
    try:
     run_stage(label+'_'+stage,[sys.executable,str(PROJECT/'scripts/run_mellum_abc_baseline.py'),'--stage',str(target)],target/'stage.log',timeout=job['timeout'])
     data=json.loads((target/'results.json').read_text());case=next(iter(data['cases'].values()))
     row['metrics'][stage]=case.get('metrics',{})
     row.setdefault('stage_status',{})[stage]=data['status']
    except Exception:row.setdefault('errors',{})[stage]=traceback.format_exc()
    current_out(out,plan,result)
   if plan['track']=='performance':
    target=d/'decode'
    try:
     run_stage(label+'_decode',[sys.executable,str(PROJECT/'scripts/run_mellum_c_service_diagnosis.py'),
        '--track','decode','--decode-worker',str(target)],target/'worker.log',timeout=min(1200,max(1,deadline-time.monotonic())))
     row['decode']=json.loads((target/'decode_results.json').read_text())
    except Exception:row.setdefault('errors',{})['decode']=traceback.format_exc()
   current_out(out,plan,result)
 check(plan);result.update(status='precision_third_measured',source_unchanged=True)
 result['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'summary.json'}
 write(out/'summary.json',result);return result


def current_out(out,plan,result):
 check(plan);render(out,plan,result);write(out/'progress.json',result);write(out/'manifest.json',plan)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],required=True)
 p.add_argument('--track',choices=['accuracy','performance','full-performance'],required=True)
 p.add_argument('--profile',choices=CANDIDATES);p.add_argument('--acceptance-run',type=Path)
 p.add_argument('--budget-minutes',type=float,default=90);p.add_argument('--prepare-only',action='store_true')
 a=p.parse_args()
 if not 15<=a.budget_minutes<=180:p.error('Budget15..180 required')
 out=Path('/workspace/results/mellum_precision_third')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+a.track);out.mkdir(parents=True)
 plan=prepare(out,a)
 if a.prepare_only:print('RESULT precision_third_prepared output',out);return
 r=execute(out,plan,a.budget_minutes);print('RESULT',r['status'],'output',out)
if __name__=='__main__':main()
