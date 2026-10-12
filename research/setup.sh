#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "${1:-}" == "--help" ]]; then
    echo 'Usage: bash setup.sh [--skip-smoke]'
    echo 'Linux x86_64 Python 3.10 CPU environment; installs pinned dependencies and model.'
    echo 'PYTHON selects Python; DOLPHINE_VENV selects the virtual environment path.'
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--skip-smoke' ) ]]; then
    echo 'Unknown argument; use --help' >&2
    exit 2
fi
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys,platform; assert sys.version_info[:2] == (3,10) and platform.system() == "Linux" and platform.machine() == "x86_64", "Pinned environment requires Linux x86_64 Python 3.10; no system packages are changed."'
VENV="${DOLPHINE_VENV:-$ROOT/.venv-inspect}"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.inspect.lock.txt"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
revision = Path('model_revision.txt').read_text().strip()
assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
snapshot_download('Qwen/Qwen3.5-0.8B', revision=revision,
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
print('Pinned model cached; no quantization or GPU packages requested.')
PY
if [[ "${1:-}" != '--skip-smoke' ]]; then
    "$VENV/bin/python" smoke_test.py
fi
