"""Run pinned official HumanEval only inside a fail-closed Linux sandbox."""
import argparse,json,os,shutil,subprocess,sys,tempfile,hashlib
from datetime import datetime,timezone
from pathlib import Path
from humaneval_sandbox import restrict


def worker(work):
    os.environ['OPENBLAS_NUM_THREADS']='1'
    import human_eval.evaluation as evaluation
    manifest=json.loads((work/'config.json').read_text());source=Path(manifest['source'])
    for file,h in manifest['files_sha256'].items():
        if hashlib.sha256((source/file).read_bytes()).hexdigest()!=h:raise ValueError('Official evaluator changed')
    # Imports are initialized before restricting; all generated code runs afterward.
    runtime=['/usr','/lib','/lib64','/opt/venv',str(source),sys.prefix,
             '/etc/ld.so.cache','/etc/localtime','/dev/null','/dev/urandom']
    tempfile.tempdir=str(work)
    os.environ.clear()
    os.environ.update(TMPDIR=str(work),HOME=str(work),PATH='/usr/bin:/bin',LANG='C.UTF-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    sandbox=restrict(work,runtime,work/'sandbox-root' if (work/'sandbox-root').exists() else None)
    # Prove private filesystem/network boundaries before running any samples.
    import socket
    try:socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except PermissionError:pass
    else:raise RuntimeError('Network sandbox failed')
    try:Path('/workspace/.humaneval_sandbox_probe').write_text('unexpected')
    except PermissionError:pass
    else:raise RuntimeError('Filesystem sandbox failed')
    score=evaluation.evaluate_functional_correctness(str(work/'samples.jsonl'),k=[1],n_workers=2,
        timeout=3.0,problem_file=str(work/'problems.jsonl'))
    (work/'score.json').write_text(json.dumps({'score':score,'sandbox':sandbox},indent=2)+'\n')


def build_chroot(work,source):
    """Fallback for kernels/Docker profiles without Landlock; no mounts needed."""
    root=work/'sandbox-root';root.mkdir()
    def copy(src):
        src=Path(src)
        if not src.exists():return
        dst=root/str(src).lstrip('/')
        if dst.exists():return
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','.git'))
        else:shutil.copyfile(src,dst)
    for path in ['/usr/lib/python3.12',str(source),'/etc/ld.so.cache',
                 '/opt/venv/lib/python3.12/site-packages/numpy',
                 '/opt/venv/lib/python3.12/site-packages/numpy.libs',
                 '/opt/venv/lib/python3.12/site-packages/tqdm']:
        copy(path)
    # Copy native dependencies of stdlib extensions and NumPy, preserving ldd paths.
    binaries=[Path('/usr/bin/python3.12'),*Path('/usr/lib/python3.12/lib-dynload').glob('*.so'),
              *Path('/opt/venv/lib/python3.12/site-packages/numpy').rglob('*.so'),
              *Path('/opt/venv/lib/python3.12/site-packages/numpy.libs').glob('*.so*')]
    for binary in binaries:
        for line in subprocess.check_output(['ldd',str(binary)],text=True,stderr=subprocess.DEVNULL).splitlines():
            for word in line.split():
                if word.startswith('/'):copy(word)
    inside=root/str(work).lstrip('/');inside.mkdir(parents=True)
    for f in work.glob('*.json*'):shutil.copyfile(f,inside/f.name)
    if os.geteuid()==0:
        os.chown(inside,65534,65534)
        for f in inside.iterdir():os.chown(f,65534,65534)


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path);p.add_argument('--problems',type=Path);p.add_argument('--output',type=Path);p.add_argument('--worker',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.worker:worker(a.worker);return
    if a.self_test:
        base=Path('/workspace/results/humaneval_selftest')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        base.mkdir(parents=True)
        tasks=[json.loads(x) for x in Path('/data/datasets/HumanEval.jsonl').read_text().splitlines()[:2]]
        a.problems=base/'problems.jsonl';a.samples=base/'samples.jsonl';a.output=base/'score'
        a.problems.write_text(''.join(json.dumps(x)+'\n' for x in tasks))
        a.samples.write_text(''.join(json.dumps({'task_id':x['task_id'],'completion':x['canonical_solution'] if i==0 else "\nraise RuntimeError('intentional failure')\n"})+'\n' for i,x in enumerate(tasks)))
    if not all([a.samples,a.problems,a.output]):p.error('samples, problems and output required')
    config=Path('/workspace/results/humaneval_environment/manifest.json')
    if not config.exists():raise RuntimeError('Run setup_humaneval.py first')
    manifest=json.loads(config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    # The sandbox account needs its own directory; parent retains report access.
    with tempfile.TemporaryDirectory(prefix='mellum-human-eval-') as directory:
        work=Path(directory)
        for src,name in [(a.samples,'samples.jsonl'),(a.problems,'problems.jsonl'),(config,'config.json')]:shutil.copyfile(src,work/name)
        # Landlock availability is probed without installing restrictions here.
        import ctypes
        libc=ctypes.CDLL(None,use_errno=True)
        chroot_fallback=libc.syscall(444,0,0,1)<0
        if chroot_fallback:build_chroot(work,Path(manifest['source']))
        if os.geteuid()==0:
            os.chown(work,65534,65534)
            for f in work.iterdir():
                if f.is_file():os.chown(f,65534,65534)
        log=a.output/'evaluation.log'
        with log.open('w') as stream:
            proc=subprocess.run([manifest['python'],'-B',str(Path(__file__).resolve()),'--worker',str(work)],stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        if proc.returncode:raise RuntimeError('Scoring sandbox/evaluator failed; see '+str(log))
        scored=(work/'sandbox-root'/str(work).lstrip('/')) if chroot_fallback else work
        for name in ['score.json','samples.jsonl_results.jsonl']:shutil.copyfile(scored/name,a.output/name)
        shutil.copyfile(config,a.output/'evaluator_manifest.json')
    if a.self_test and json.loads((a.output/'score.json').read_text())['score']['pass@1']!=.5:
        raise RuntimeError('Official evaluator self-test must pass exactly 1/2')
    print('RESULT humaneval_scored output',a.output)
if __name__=='__main__':main()
