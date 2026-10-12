#!/usr/bin/env bash
# Dolphine bootstrap; invoke with bash setup.sh from any working directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/research/setup.sh" "$@"
