"""Create offline source ZIP plus complete Git-history bundle from a clean checkout."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
 if git('status','--porcelain'):raise RuntimeError('Commit or resolve all checkout changes before packaging')
 head=git('rev-parse','HEAD');stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 out=a.output or Path('/workspace/delivery')/(stamp+'_'+head[:8]);out=out.resolve();out.mkdir(parents=True,exist_ok=False)
 archive=out/'tencent-mellum-source.zip';bundle=out/'tencent-mellum-history.bundle'
 subprocess.run(['git','archive','--format=zip','--prefix=tencent-mellum/','--output',str(archive),'HEAD'],cwd=ROOT,check=True)
 subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True)
 checked=subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,text=True,capture_output=True)
 (out/'bundle-verification.txt').write_text(checked.stdout+checked.stderr)
 d={'created_utc':stamp,'commit':head,'branch':git('branch','--show-current'),'step0_tag_commit':git('rev-list','-n','1','step0-local-validated'),
    'scope':'ZIP contains tracked delivery HEAD; bundle preserves all local Git refs/history. No model/venv/raw tensor payloads.',
    'files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [archive,bundle]}}
 (out/'delivery.json').write_text(json.dumps(d,indent=2)+'\n')
 (out/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in [archive,bundle,out/'delivery.json']))
 print('RESULT packaged output',out);print('COMMIT',head)
if __name__=='__main__':main()
