#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="/Users/bernhard/Development/assess_speaking/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
PORT="${PORT:-8502}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/reports}"
DRY_RUN="${ASSESS_SPEAKING_DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/run_dashboard.sh [options] [-- <extra streamlit args>]

Options:
  --port N         Streamlit port (default: 8502)
  --log-dir PATH   Report/history directory (default: ./reports)
  --python PATH    Python interpreter to use
  --dry-run        Use stubbed assessment results for UX testing
  --help           Show this help

Examples:
  ./scripts/run_dashboard.sh
  ./scripts/run_dashboard.sh --dry-run --port 8504
  PYTHON_BIN=.venv/bin/python ./scripts/run_dashboard.sh --log-dir /tmp/reports
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ASSESS_SPEAKING_DRY_RUN="$DRY_RUN"

exec "$PYTHON_BIN" -m streamlit run \
  "$ROOT_DIR/scripts/interactive_dashboard.py" \
  --server.port "$PORT" \
  "${EXTRA_ARGS[@]}" \
  -- --log-dir "$LOG_DIR"
