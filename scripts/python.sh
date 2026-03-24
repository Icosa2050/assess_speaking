#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  cat >&2 <<EOF
error: ${VENV_PYTHON} was not found.

Create the project virtual environment first:
  ./scripts/setup_env.sh
EOF
  exit 1
fi

exec "${VENV_PYTHON}" "$@"
