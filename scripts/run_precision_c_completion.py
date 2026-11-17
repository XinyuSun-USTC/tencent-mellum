"""Complete precision-first C200 with immutable inputs and independent repeats."""
import argparse,json,sys,time,traceback,statistics
from pathlib import Path
from datetime import datetime,timezone
import run_mellum_abc_baseline as perf
from run_precision_third import qualify,check
from mellum_runtime_config import PROJECT
from mellum_perf_protocol import sha
from mellum_process_runner import run_stage


def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def aggregate(runs):
 complete=[r['metrics'] for r in runs if r.get('metrics',{}).get('complete_without_failures')]
 return {'attempts':len(runs),'fully_successful_runs':len(complete),
  'three_run_median':{key:statistics.median(x[key] for x in complete) for key in ['tpot_p99_s','ttft_p99_s']} if len(runs)==3 and len(complete)==3 else None}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--gpu',type=int,choices=[0,1],required=True)
 p.add_argument('--repeats',type=int,choices=[1,3],default=1)
 p.add_argument('--acceptance-run',type=Path,default=Path('/workspace/results/mellum_precision_third/20261117T170629593110Z_accuracy/r_swa_decode4'))
 p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 qualify(a.acceptance_run,'r_swa_decode4')
 out=Path('/workspace/results/mellum_precision_c_completion')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
 d=out/'template';d.mkdir();template=perf.prepare(d,a.gpu,False)
 plan={'profile':'r_swa_decode4','gpu':a.gpu,'repeats':a.repeats,'acceptance_run':str(a.acceptance_run),
  'source_sha256':{**perf.sources(),str(Path(__file__).resolve()):sha(__file__)},
  'external_sha256':{**template['input_sha256'],str(a.acceptance_run/'summary.json'):sha(a.acceptance_run/'summary.json'),str(d/'inputs/adapter_manifest.json'):template['adapter_manifest_sha256']},
  'cases':template['stages']['C'],'protocol':{'number':200,'rate':0.3,'output_tokens':256,'request_timeout_seconds':180,'single_gpu':True,'stream':True,'seed':42}}
 write(out/'manifest.json',plan)
 if a.prepare_only:print('RESULT c_completion_prepared output',out);return
 result={'profile':plan['profile'],'runs':[],'scope':'Local C200; success-only tails reported alongside all failures; independent repeats, not judge measurements'}
 for i in range(a.repeats):
  check(plan);target=out/f'repeat_{i+1}';target.mkdir()
  job={'stage':'C','gpu':a.gpu,'profile':plan['profile'],'cases':plan['cases'],'server_port':16551+a.gpu*100,'engine_port':18526+a.gpu*100,'timeout':1800}
  write(target/'plan.json',job);row={'repeat':i+1};result['runs'].append(row)
  try:
   run_stage('C200_repeat_'+str(i+1),[sys.executable,str(PROJECT/'scripts/run_mellum_abc_baseline.py'),'--stage',str(target)],target/'stage.log',timeout=1800)
   data=json.loads((target/'results.json').read_text());case=next(iter(data['cases'].values()))
   row.update(stage_status=data['status'],metrics=case['metrics'])
  except Exception:row['error']=traceback.format_exc()
  check(plan);result.update(aggregate(result['runs']));write(out/'progress.json',result)
 result.update(status='c200_measured' if all('metrics' in r for r in result['runs']) else 'c200_partial',source_unchanged=True)
 result['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=out/'summary.json'}
 write(out/'summary.json',result);print('RESULT',result['status'],'output',out)
if __name__=='__main__':main()
