#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PLAYWRIGHT_RESEARCH_SESSION="${PLAYWRIGHT_CELI_SESSION:-celi}"
export PLAYWRIGHT_RESEARCH_CONFIG="${PLAYWRIGHT_CELI_CONFIG:-${REPO_ROOT}/.playwright/celi-cli.config.json}"
export PLAYWRIGHT_RESEARCH_PROFILE_DIR="${PLAYWRIGHT_CELI_PROFILE_DIR:-${REPO_ROOT}/.playwright/profiles/celi}"
export PLAYWRIGHT_RESEARCH_OUTPUT_DIR="${PLAYWRIGHT_CELI_OUTPUT_DIR:-${REPO_ROOT}/output/playwright/celi}"

exec "${REPO_ROOT}/scripts/playwright_research.sh" "$@"
