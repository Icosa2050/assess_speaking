#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PYTHON_RUNNER="${SCRIPT_DIR}/python.sh"

exec "${PYTHON_RUNNER}" -m pytest "$@"
