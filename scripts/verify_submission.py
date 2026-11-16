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
 for name in ['start_server.sh','scripts/serve_submission.py','scripts/smoke_submission.py','README.md','WORKLOG.md','SOURCES.md',
              'report/submission/01_technical_report.md','report/submission/02_ai_usage.md','report/submission/03_delivery_checklist.md']:
  check((ROOT/name).is_file(),'Missing '+name)
 check(not subprocess.check_output(['git','ls-files','-u'],cwd=ROOT,text=True).strip(),'Unresolved Git merge conflict')
 index=json.loads((ROOT/'artifacts/submission/index.json').read_text())
 for name,item in index['files'].items():
  p=ROOT/name;check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Evidence hash differs: '+name)
 report=(ROOT/'report/submission/01_technical_report.md').read_text()
 check(len(re.findall(r'^## [1-4]\. ',report,re.M))==4,'Technical report must retain four main sections')
 check(report.count('<!-- pagebreak -->')==3,'Technical report must retain four content divisions')
 for name in ['README.md','WORKLOG.md']:
  p=ROOT/name
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in target or target.startswith('#'):continue
   check((p.parent/unquote(target.split('#')[0])).exists(),'Broken link in '+name+': '+target)
 for mode in ['conservative','aggressive']:
  d=json.loads((ROOT/'configs/submission'/f'{mode}.json').read_text())
  check(d['num_gpus']==1 and not d['mode_is_optimization_point'],'Invalid mode declaration')
 check((ROOT/'start_server.sh').stat().st_mode & 0o111,'Launcher is not executable')
 result={'status':'passed' if not errors else 'failed','checks':checks,'errors':errors,'scope':'CPU files/hashes/links/Markdown structure; exported page count and GPU acceptance are separate'}
 print(json.dumps(result,ensure_ascii=False,indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
