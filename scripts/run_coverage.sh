#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"
PYTHON_RUNNER="${SCRIPT_DIR}/python.sh"

MODE="source"
JSON_OUT=""
HTML_DIR=""
PYTEST_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/run_coverage.sh [--source|--full] [--json-out PATH] [--html-dir PATH] [-- pytest args...]

Runs pytest under coverage using the repo-local virtualenv.

Modes:
  --source    Report source-only coverage (default). Tests stay measured but are
              omitted from the generated report, JSON, and HTML outputs.
  --full      Report full coverage including tests.

Outputs:
  source mode -> coverage.json + htmlcov/
  full mode   -> coverage.full.json + htmlcov-full/
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --source)
      MODE="source"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --json-out)
      if (( $# < 2 )); then
        echo "error: --json-out requires a path" >&2
        exit 1
      fi
      JSON_OUT="$2"
      shift 2
      ;;
    --html-dir)
      if (( $# < 2 )); then
        echo "error: --html-dir requires a path" >&2
        exit 1
      fi
      HTML_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      PYTEST_ARGS=("$@")
      break
      ;;
    *)
      PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${JSON_OUT}" ]]; then
  if [[ "${MODE}" == "full" ]]; then
    JSON_OUT="${PROJECT_ROOT}/coverage.full.json"
  else
    JSON_OUT="${PROJECT_ROOT}/coverage.json"
  fi
fi

if [[ -z "${HTML_DIR}" ]]; then
  if [[ "${MODE}" == "full" ]]; then
    HTML_DIR="${PROJECT_ROOT}/htmlcov-full"
  else
    HTML_DIR="${PROJECT_ROOT}/htmlcov"
  fi
fi

REPORT_ARGS=()
if [[ "${MODE}" == "source" ]]; then
  REPORT_ARGS+=("--omit=tests/*")
fi

cd "${PROJECT_ROOT}"

echo "==> Running pytest under coverage (${MODE})"
"${PYTHON_RUNNER}" -m coverage erase
"${PYTHON_RUNNER}" -m coverage run -m pytest "${PYTEST_ARGS[@]}"

echo "==> Terminal report"
"${PYTHON_RUNNER}" -m coverage report -m "${REPORT_ARGS[@]}"

echo "==> JSON report: ${JSON_OUT}"
"${PYTHON_RUNNER}" -m coverage json -o "${JSON_OUT}" "${REPORT_ARGS[@]}"

echo "==> HTML report: ${HTML_DIR}"
"${PYTHON_RUNNER}" -m coverage html -d "${HTML_DIR}" "${REPORT_ARGS[@]}"
