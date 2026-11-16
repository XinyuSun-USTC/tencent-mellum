"""Read-only CPU verification of delivery files, evidence hashes and PDF page caps."""
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
 for name,expected in [('01_technical_report.pdf',4),('02_ai_usage.pdf',1)]:
  p=ROOT/'report/submission'/name;data=p.read_bytes() if p.exists() else b''
  count=len(re.findall(rb'/Type\s*/Page\b',data));check(data.startswith(b'%PDF') and count==expected,f'Invalid PDF/page count {name}: {count}')
 for name in ['README.md','WORKLOG.md']:
  p=ROOT/name
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in target or target.startswith('#'):continue
   check((p.parent/unquote(target.split('#')[0])).exists(),'Broken link in '+name+': '+target)
 for mode in ['conservative','aggressive']:
  d=json.loads((ROOT/'configs/submission'/f'{mode}.json').read_text())
  check(d['num_gpus']==1 and not d['mode_is_optimization_point'],'Invalid mode declaration')
 check((ROOT/'start_server.sh').stat().st_mode & 0o111,'Launcher is not executable')
 result={'status':'passed' if not errors else 'failed','checks':checks,'errors':errors,'scope':'CPU files/hashes/links/PDF caps; not GPU acceptance'}
 print(json.dumps(result,ensure_ascii=False,indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
