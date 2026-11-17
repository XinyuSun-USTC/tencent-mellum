"""Read-only CPU verification of delivery files, evidence hashes and Markdown structure."""
import hashlib,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]

def main():
 errors=[];checks=0
 def check(ok,message):
  nonlocal checks
  checks+=1
  if not ok:errors.append(message)
 for name in ['start_server.sh','scripts/serve_delivery.py','scripts/serve_mellum.py','scripts/smoke_delivery.py','scripts/mellum_submission_modes.py','scripts/smoke_dual_delivery.py','README.md','WORKLOG.md','docs/submission/01_sources_and_licenses.md',
              'TECHNICAL_REPORT.md','AI_USAGE.md','docs/submission/02_delivery_audit.md']:
  check((ROOT/name).is_file(),'Missing '+name)
 check(not subprocess.check_output(['git','ls-files','-u'],cwd=ROOT,text=True).strip(),'Unresolved Git merge conflict')
 index=json.loads((ROOT/'artifacts/submission/index.json').read_text())
 for name,item in index['files'].items():
  p=ROOT/name;check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Evidence hash differs: '+name)
 report=(ROOT/'TECHNICAL_REPORT.md').read_text()
 check(len(re.findall(r'^## [1-5]\. ',report,re.M))==5,'Technical report must contain five current sections')
 check(report.count('<!-- pagebreak -->')==3,'Technical report must retain four content divisions')
 for name in ['README.md','WORKLOG.md','TECHNICAL_REPORT.md','AI_USAGE.md']:
  p=ROOT/name
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in target or target.startswith('#'):continue
   check((p.parent/unquote(target.split('#')[0])).exists(),'Broken link in '+name+': '+target)
 from mellum_submission_modes import ROWS,MODES,DEFAULT_MODE,profile_name,settings
 from mellum_execution_profiles import profile_settings
 from serve_delivery import command
 check(DEFAULT_MODE=='performance','Default must be performance mode')
 for mode in MODES:
  for row,points in ROWS.items():
   cmd=command(row,0,8000,8026,ROOT/'results/verification',mode=mode)
   check(cmd[cmd.index('--max-num-seqs')+1]=='32','All delivery rows must use32 slots')
   check(profile_settings(profile_name(mode,points))==settings(mode,points),'Mode point switches differ from profile')
 check(settings('precision',ROWS['step0123'])==profile_settings('r_swa_decode4'),'Precision final differs from accepted configuration')
 check(settings('performance',ROWS['step0123'])==profile_settings('sprint_graph_qk_router_prefix_metadata'),'Performance final differs from measured configuration')
 for directory in ['artifacts/experiments/performance_cumulative','artifacts/experiments/precision_third','artifacts/experiments/precision_third/c_completion','artifacts/submission/dual_mode/gpu_smoke']:
  root=ROOT/directory;data=json.loads((root/'index.json').read_text())
  for name,item in data['files'].items():
   digest=item['sha256'] if isinstance(item,dict) else item
   p=root/name;check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,'Current evidence changed: '+str(p.relative_to(ROOT)))
 smoke=ROOT/'artifacts/submission/dual_mode/gpu_smoke'
 count=0
 for summary in smoke.glob('*/summary.json'):
  data=json.loads(summary.read_text());check(data['status']=='smoke_passed','GPU smoke failed')
  for label,item in data['results'].items():
   count+=1;mode,row=label.split('_',1)
   check(item['status']=='smoke_passed' and item['http_port_released'],'GPU smoke or cleanup failed')
   check(item['configuration_identity']=={'mode':mode,'points':list(ROWS[row])},'GPU smoke identity differs')
   launch=json.loads((summary.parent/label/'server/launch.json').read_text())
   for original,digest in launch['source_sha256'].items():
    old=Path(original);parts=old.parts;marker=parts.index('tencent-mellum-delivery-dual')
    current=ROOT/Path(*parts[marker+1:])
    check(current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest()==digest,'Source changed since GPU smoke: '+str(current))
 check(count==8,'Require eight cumulative GPU smoke configurations')
 check('serve_delivery.py' in (ROOT/'start_server.sh').read_text(),'Root launcher must use delivery entrypoint')
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
 for name in filter(None,tracked):
  p=ROOT/name
  check(not p.is_symlink(),'Tracked external/symlink dependency: '+name)
  check(p.suffix not in ('.pyc','.safetensors','.whl','.pem','.key'),'Unexpected generated/credential payload: '+name)
 check((ROOT/'start_server.sh').stat().st_mode & 0o111,'Launcher is not executable')
 result={'status':'passed' if not errors else 'failed','checks':checks,'errors':errors,'scope':'CPU files/hashes/links/Markdown structure; exported page count and GPU acceptance are separate'}
 print(json.dumps(result,ensure_ascii=False,indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
