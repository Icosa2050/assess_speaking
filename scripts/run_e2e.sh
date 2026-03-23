#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_RUNNER="${SCRIPT_DIR}/python.sh"
PYTEST_CONFIG="${PROJECT_ROOT}/pytest.e2e.ini"

"${PYTHON_RUNNER}" - <<'PY'
import importlib.util
import sys

missing = [
    module
    for module in ("pytest_playwright", "pytest_base_url")
    if importlib.util.find_spec(module) is None
]
if missing:
    print(
        "error: missing pytest plugins required for E2E: "
        + ", ".join(missing)
        + ". Run ./scripts/setup_env.sh to install the project test dependencies.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

exec "${PYTHON_RUNNER}" -m pytest -c "${PYTEST_CONFIG}" tests/e2e "$@"
