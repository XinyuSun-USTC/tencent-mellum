"""Create a reference-only venv; install only Transformers, reuse frozen ROCm dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path('/workspace')
ENV = ROOT / 'venvs/hf-reference'
BASE_SITE = Path('/opt/venv/lib/python3.12/site-packages')
VERSION = '5.10.1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wheel', type=Path, help='Optional predownloaded official wheel; version verified after installation')
    args = parser.parse_args()
    if not (ENV / 'bin/python').exists():
        venv.EnvBuilder(with_pip=True).create(ENV)
    site = ENV / 'lib/python3.12/site-packages'
    # This adds the path after local site-packages, without processing the base
    # environment's editable .pth files. Local Transformers takes precedence.
    (site / 'frozen_rocm_dependencies.pth').write_text(str(BASE_SITE) + '\n')
    out = ROOT / 'results/hf_reference_environment'
    out.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        print('Prepared:', ENV, '(Transformers installation not performed)')
        return
    target = str(args.wheel.resolve()) if args.wheel else 'transformers==' + VERSION
    cmd = [str(ENV / 'bin/python'), '-m', 'pip', 'install', '--no-deps',
           '--ignore-installed', '--no-cache-dir', '--retries', '0', '--timeout', '15', target]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    (out / 'install.log').write_text(completed.stdout + completed.stderr)
    print(completed.stdout + completed.stderr, end='')
    if completed.returncode:
        raise SystemExit(completed.returncode)
    subprocess.run([str(ENV / 'bin/python'), str(ROOT / 'scripts/check_hf_reference.py')], check=True)

if __name__ == '__main__':
    main()
