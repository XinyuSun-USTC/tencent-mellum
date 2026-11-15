"""Install assessment client in a separate venv, then audit with no GPU inference."""
import argparse,json,os,subprocess,sys,venv,traceback
from pathlib import Path
from datetime import datetime,timezone
from mellum_runtime_config import PROJECT

ENV=Path('/workspace/venvs/evalscope-perf')
BASE=Path('/opt/venv/bin/python')
VERSION='1.15.0'
TRANSFORMERS='5.10.1'


def frozen_state():
    code="import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata['Name'],d.version,str(d.locate_file(''))) for d in m.distributions())))"
    return json.loads(subprocess.check_output([str(BASE),'-B','-c',code],text=True))


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--wheelhouse',type=Path)
    p.add_argument('--skip-install',action='store_true',help='Audit an already installed pinned client')
    args=p.parse_args();out=Path('/workspace/results/mellum_perf_client')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    before=frozen_state();write(out/'frozen_before.json',before)
    result={'status':'prepared','python':str(ENV/'bin/python'),'evalscope':VERSION,'transformers':TRANSFORMERS,
            'source':'https://pypi.org/project/evalscope/'+VERSION+'/','gpu_inference_started':False,
            'scope':'Client installation/source and protocol audit only; no GPU benchmark'}
    write(out/'summary.json',result)
    if args.prepare_only:print('RESULT perf_client_prepared output',out);return 0
    try:
        if not (ENV/'bin/python').exists():venv.EnvBuilder(with_pip=True).create(ENV)
        site=ENV/'lib/python3.12/site-packages';site.mkdir(parents=True,exist_ok=True)
        # Read-only bridge for existing ROCm Torch/dependencies. Child site wins;
        # pip runs under child prefix and must not uninstall external packages.
        (site/'frozen_dependencies.pth').write_text('/opt/venv/lib/python3.12/site-packages\n')
        if not args.skip_install:
            command=[str(ENV/'bin/python'),'-m','pip','install','--retries','1','--timeout','30',
                     '--report',str(out/'pip_report.json'),f'evalscope[perf]=={VERSION}',f'transformers=={TRANSFORMERS}']
            if args.wheelhouse:command+=['--no-index','--find-links',str(args.wheelhouse)]
            result['install_command']=command;write(out/'summary.json',result)
            with (out/'install.log').open('w') as log:
                print('Installing evaluation-only client; log',out/'install.log',flush=True)
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:raise RuntimeError('Client installation failed; inspect install.log')
        environment=dict(os.environ)
        environment.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        environment.pop('PYTHONPATH',None)
        command=[str(ENV/'bin/python'),'-B',str(PROJECT/'scripts/audit_mellum_perf_client.py'),'--output',str(out/'audit')]
        print('Auditing actual installed release; no GPU server is started',flush=True)
        with (out/'audit.log').open('w') as log:subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        result['audit']=json.loads((out/'audit/summary.json').read_text());result['status']='client_audit_complete'
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        after=frozen_state();write(out/'frozen_after.json',after)
        result['frozen_runtime_distributions_unchanged']=before==after
        if before!=after:result.update(status='failed',error='Frozen /opt/venv distribution state changed')
        write(out/'summary.json',result)
    print('RESULT',result['status'],'output',out)
    return 0 if result['status']=='client_audit_complete' else 1
if __name__=='__main__':raise SystemExit(main())
