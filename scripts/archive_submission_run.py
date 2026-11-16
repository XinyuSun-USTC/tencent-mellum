"""Archive bounded, original result files with SHA256; never synthesize metrics."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
 if not a.label.replace('_','').replace('-','').isalnum():p.error('Label must be a simple directory name')
 source=a.run.resolve();base=ROOT/'artifacts/submission';dest=base/a.label
 if not (source/'summary.json').exists():raise ValueError('Require a completed or partial run with summary.json')
 index=json.loads((base/'index.json').read_text())
 index['sources'][a.label]=str(source)
 for f in sorted(source.rglob('*')):
  if not f.is_file() or f.is_symlink() or f.suffix not in ('.json','.jsonl','.txt','.md','.log') or f.stat().st_size>3_000_000:continue
  if f.name.endswith('_inputs.json'):continue
  target=dest/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);data=f.read_bytes()
  if target.exists() and target.read_bytes()!=data:raise ValueError('Refuse overwrite of different archived evidence: '+str(target))
  target.write_bytes(data);index['files'][str(target.relative_to(ROOT))]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'original':str(f)}
 (base/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
 print('Archived',a.label,'status',json.loads((source/'summary.json').read_text()).get('status'))
if __name__=='__main__':main()
