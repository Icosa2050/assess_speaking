#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_RUNNER="${SCRIPT_DIR}/python.sh"

if (($# > 0)); then
  exec "${PYTHON_RUNNER}" -m unittest "$@"
fi

exec "${PYTHON_RUNNER}" -m unittest discover -s tests
