#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
PWCLI="${CODEX_HOME}/skills/playwright/scripts/playwright_cli.sh"
SESSION="${PLAYWRIGHT_RESEARCH_SESSION:-research}"
CONFIG="${PLAYWRIGHT_RESEARCH_CONFIG:-${REPO_ROOT}/.playwright/research-cli.config.json}"
PROFILE_DIR="${PLAYWRIGHT_RESEARCH_PROFILE_DIR:-${REPO_ROOT}/.playwright/profiles/research}"
OUTPUT_DIR="${PLAYWRIGHT_RESEARCH_OUTPUT_DIR:-${REPO_ROOT}/output/playwright/research}"

mkdir -p "${PROFILE_DIR}" "${OUTPUT_DIR}"

if [[ ! -x "${PWCLI}" ]]; then
  echo "Playwright wrapper not found at ${PWCLI}" >&2
  exit 1
fi

args=("$@")
if [[ "${1:-}" == "open" ]]; then
  has_persistent="false"
  for arg in "$@"; do
    if [[ "${arg}" == "--persistent" ]]; then
      has_persistent="true"
      break
    fi
  done
  if [[ "${has_persistent}" != "true" ]]; then
    args=("open" "--persistent" "${@:2}")
  fi
fi

exec "${PWCLI}" --session "${SESSION}" --config "${CONFIG}" "${args[@]}"
