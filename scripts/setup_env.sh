#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_DIR="${1:-$DEFAULT_VENV_DIR}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
else
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "error: no suitable python interpreter found (looked for python3.12, python3.11, python3)." >&2
  exit 1
fi

MIN_MAJOR=3
MIN_MINOR=11

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [VENV_DIR]

Creates (or reuses) a virtual environment for assess_speaking, upgrading
packaging tools and installing requirements.

Environment variables:
  PYTHON_BIN   Python interpreter to use (default: python3)
  PIP_FLAGS    Extra flags passed to pip install commands
USAGE
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: could not find '$PYTHON_BIN'. Set PYTHON_BIN to a valid interpreter." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys
REQ = (3, 11)
if sys.version_info < REQ:
    sys.exit(1)
PY
then
  detected_version=$("$PYTHON_BIN" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)
  echo "error: Python $detected_version is too old. Please use Python 3.11 or newer." >&2
  exit 1
fi

venv_abs="$("$PYTHON_BIN" - <<PY
import os
print(os.path.abspath("${VENV_DIR}"))
PY
)"

if [[ ! -d "$venv_abs" ]]; then
  echo "Creating virtual environment at $venv_abs using $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$venv_abs"
else
  echo "Reusing existing virtual environment at $venv_abs"
fi

VENV_PYTHON="$venv_abs/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "error: expected $VENV_PYTHON to exist. Remove $venv_abs and retry." >&2
  exit 1
fi

pip_args=()
if [[ -n "${PIP_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  pip_args=(${PIP_FLAGS})
fi

"$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
if ((${#pip_args[@]})); then
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel "${pip_args[@]}"
  "$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt" "${pip_args[@]}"
else
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
  "$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

cat <<INFO

Environment ready.
Activate it with:
  source "$venv_abs/bin/activate"

Run the self-test once models are available:
  python assess_speaking.py --selftest --llm llama3.1
INFO
