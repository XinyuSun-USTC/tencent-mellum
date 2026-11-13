"""Verify versioned Step0 sources and lightweight experiment evidence, offline."""
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / 'report/validation/39_step0_manifest.json').read_text())['files']
    evidence = json.loads((root / 'artifacts/index.json').read_text())['files']
    failures = []
    for row in sources:
        path = root / row['path']
        if not path.is_file() or digest(path) != row['sha256']:
            failures.append(row['path'])
    for row in evidence:
        path = root / row['repository_path']
        if not path.is_file() or digest(path) != row['sha256']:
            failures.append(row['repository_path'])
    print(json.dumps({'source_files': len(sources), 'evidence_files': len(evidence),
                      'failures': failures, 'scope': 'No external raw tensors or GPU inference'}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
