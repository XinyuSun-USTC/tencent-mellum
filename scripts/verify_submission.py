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
 for name in ['start_server.sh','scripts/serve_delivery.py','scripts/serve_mellum.py','scripts/smoke_delivery.py','scripts/mellum_delivery_matrix.py','README.md','WORKLOG.md','docs/submission/01_sources_and_licenses.md',
              'TECHNICAL_REPORT.md','AI_USAGE.md','docs/submission/02_delivery_audit.md']:
  check((ROOT/name).is_file(),'Missing '+name)
 check(not subprocess.check_output(['git','ls-files','-u'],cwd=ROOT,text=True).strip(),'Unresolved Git merge conflict')
 index=json.loads((ROOT/'artifacts/submission/index.json').read_text())
 for name,item in index['files'].items():
  p=ROOT/name;check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Evidence hash differs: '+name)
 report=(ROOT/'TECHNICAL_REPORT.md').read_text()
 check(len(re.findall(r'^## [1-4]\. ',report,re.M))==4,'Technical report must retain four main sections')
 check(report.count('<!-- pagebreak -->')==3,'Technical report must retain four content divisions')
 for name in ['README.md','WORKLOG.md','TECHNICAL_REPORT.md','AI_USAGE.md']:
  p=ROOT/name
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in target or target.startswith('#'):continue
   check((p.parent/unquote(target.split('#')[0])).exists(),'Broken link in '+name+': '+target)
 from mellum_delivery_matrix import ROWS,DEFAULT_ROW,point_settings
 from mellum_execution_profiles import profile_settings
 check(DEFAULT_ROW=='step0123','Default must match last ablation row')
 check(all(r['slots']==32 for r in ROWS.values()),'All delivery rows must use32 slots')
 for row in ROWS.values():check(point_settings(row['points'])==profile_settings(row['profile']),'Point switches differ from cumulative profile')
 check(profile_settings('delivery_step0')==profile_settings('r'),'Step0 baseline differs from reference')
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
