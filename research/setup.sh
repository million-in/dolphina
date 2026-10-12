#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/qwen-research"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install 'transformers>=5.2,<6' huggingface_hub safetensors pillow
.venv/bin/python -m pip freeze > requirements.lock.txt
.venv/bin/python smoke_test.py
