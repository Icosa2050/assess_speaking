#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the CLI shipped with our locked playwright-core dependency.
PWCLI="${REPO_ROOT}/frontend/node_modules/playwright-core/lib/tools/cli-client/cli.js"
SESSION="${PLAYWRIGHT_RESEARCH_SESSION:-research}"
CONFIG="${PLAYWRIGHT_RESEARCH_CONFIG:-${REPO_ROOT}/.playwright/research-cli.config.json}"
PROFILE_DIR="${PLAYWRIGHT_RESEARCH_PROFILE_DIR:-${REPO_ROOT}/.playwright/profiles/research-chromium}"
OUTPUT_DIR="${PLAYWRIGHT_RESEARCH_OUTPUT_DIR:-${REPO_ROOT}/output/playwright/research}"

if [[ ! -f "${PWCLI}" ]]; then
  echo "Project Playwright CLI not found. Run: npm --prefix frontend ci --ignore-scripts" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${PROFILE_DIR}" "${OUTPUT_DIR}"

args=("$@")
if [[ "${1:-}" == "open" ]]; then
  has_persistent="false"
  has_profile="false"
  for arg in "$@"; do
    if [[ "${arg}" == "--persistent" ]]; then
      has_persistent="true"
    fi
    case "${arg}" in
      --profile|--profile=*) has_profile="true" ;;
    esac
  done
  if [[ "${has_persistent}" != "true" ]]; then
    args=("open" "--persistent" "${@:2}")
  fi
  if [[ "${has_profile}" != "true" ]]; then
    args+=("--profile=${PROFILE_DIR}")
  fi
fi

exec node "${PWCLI}" --session "${SESSION}" --config "${CONFIG}" "${args[@]}"
