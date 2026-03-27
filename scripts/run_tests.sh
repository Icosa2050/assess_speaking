#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PYTHON_RUNNER="${SCRIPT_DIR}/python.sh"

exec "${PYTHON_RUNNER}" -m pytest "$@"
