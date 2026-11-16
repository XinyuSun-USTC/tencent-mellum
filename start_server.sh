#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ATOM_PYTHON:-/opt/venv/bin/python}" "$PROJECT_DIR/scripts/serve_submission.py" "$@"
