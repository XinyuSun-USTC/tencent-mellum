"""Acquire official source and create evaluation-only environment without pip installs."""
import argparse,json,hashlib,subprocess,venv
from pathlib import Path
ROOT=Path('/workspace');SOURCE=ROOT/'tools/human-eval';ENV=ROOT/'venvs/humaneval-eval'

def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',help='Exact upstream commit for reproducible installation');a=p.parse_args()
    SOURCE.parent.mkdir(parents=True,exist_ok=True)
    if not (SOURCE/'.git').exists():subprocess.run(['git','clone','https://github.com/openai/human-eval.git',str(SOURCE)],check=True)
    if a.revision:subprocess.run(['git','checkout','--detach',a.revision],cwd=SOURCE,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip():raise RuntimeError('Official source tree modified')
    venv.EnvBuilder(with_pip=False).create(ENV)
    site=next((ENV/'lib').glob('python*/site-packages'))
    (site/'evaluation_dependencies.pth').write_text('/opt/venv/lib/python3.12/site-packages\n'+str(SOURCE)+'\n')
    files={str(f.relative_to(SOURCE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((SOURCE/'human_eval').glob('*.py'))}
    report={'repository':'https://github.com/openai/human-eval','commit':commit,'source':str(SOURCE),'python':str(ENV/'bin/python'),'files_sha256':files}
    out=ROOT/'results/humaneval_environment';out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([str(ENV/'bin/python'),'-B','-c','import human_eval.evaluation,numpy,tqdm; print("HumanEval imports passed")'],check=True)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
