#!/bin/bash
set -euo pipefail

# Creates a dedicated self-hosted GitHub Actions runner for:
#   https://github.com/Icosa2050/assess_speaking
#
# Intended usage:
#   GH_TOKEN="$(gh auth token)" sudo -E bash scripts/create_assess_speaking_runner.sh
#
# Optional overrides:
#   RUNNER_USER=github-runner
#   GITHUB_OWNER=Icosa2050
#   GITHUB_REPO=assess_speaking
#   RUNNER_NAME=icosa-assess-speaking-m4-01
#   RUNNER_DIR=/Users/github-runner/actions-runner-assess-speaking-m4-01
#   TEMPLATE_DIR=/Users/github-runner/actions-runner-m4-04
#   RUNNER_LABELS=icosa-apple-ci,assess-speaking,m4-burst
#   RUNNER_TOKEN=<registration token>

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as sudo/root." >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is intended for macOS runners." >&2
  exit 1
fi

RUNNER_USER="${RUNNER_USER:-github-runner}"
RUNNER_GROUP="${RUNNER_GROUP:-staff}"
GITHUB_OWNER="${GITHUB_OWNER:-Icosa2050}"
GITHUB_REPO="${GITHUB_REPO:-assess_speaking}"
GITHUB_URL="https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}"
REPO_SLUG_HYPHEN="$(printf '%s' "${GITHUB_REPO}" | tr '_' '-')"
LOG_SLUG="${GITHUB_OWNER}-${GITHUB_REPO}"

default_template_dir="/Users/${RUNNER_USER}/actions-runner-m4-04"
TEMPLATE_DIR="${TEMPLATE_DIR:-${default_template_dir}}"

find_next_runner_name() {
  local idx name dir plist
  idx=1
  while true; do
    name="$(printf 'icosa-%s-m4-%02d' "${REPO_SLUG_HYPHEN}" "${idx}")"
    dir="/Users/${RUNNER_USER}/actions-runner-${REPO_SLUG_HYPHEN}-m4-$(printf '%02d' "${idx}")"
    plist="/Library/LaunchDaemons/com.icosa.runner.${name}.plist"
    if [[ ! -e "${dir}" && ! -e "${plist}" ]]; then
      printf '%s\n' "${name}"
      return 0
    fi
    idx=$((idx + 1))
  done
}

RUNNER_NAME="${RUNNER_NAME:-$(find_next_runner_name)}"
default_runner_dir="/Users/${RUNNER_USER}/actions-runner-${REPO_SLUG_HYPHEN}-m4-${RUNNER_NAME##*-}"
RUNNER_DIR="${RUNNER_DIR:-${default_runner_dir}}"
RUNNER_LABELS="${RUNNER_LABELS:-icosa-apple-ci,assess-speaking,m4-burst}"
PLIST_PATH="/Library/LaunchDaemons/com.icosa.runner.${RUNNER_NAME}.plist"
LOG_DIR="/Users/${RUNNER_USER}/Library/Logs/actions.runner.${LOG_SLUG}.${RUNNER_NAME}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd python3
require_cmd rsync
require_cmd launchctl
require_cmd id

if ! id -u "${RUNNER_USER}" >/dev/null 2>&1; then
  echo "Runner user '${RUNNER_USER}' does not exist." >&2
  exit 1
fi

if [[ ! -d "${TEMPLATE_DIR}" ]]; then
  echo "Template runner directory not found: ${TEMPLATE_DIR}" >&2
  exit 1
fi

if [[ -e "${RUNNER_DIR}" ]]; then
  echo "Target runner directory already exists: ${RUNNER_DIR}" >&2
  exit 1
fi

if [[ -e "${PLIST_PATH}" ]]; then
  echo "LaunchDaemon already exists: ${PLIST_PATH}" >&2
  exit 1
fi

resolve_admin_token() {
  if [[ -n "${GH_TOKEN:-}" ]]; then
    printf '%s\n' "${GH_TOKEN}"
    return 0
  fi
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    printf '%s\n' "${GITHUB_TOKEN}"
    return 0
  fi
  if command -v gh >/dev/null 2>&1 && [[ -n "${SUDO_USER:-}" ]]; then
    su - "${SUDO_USER}" -c 'gh auth token' 2>/dev/null || true
    return 0
  fi
  return 1
}

create_registration_token() {
  local admin_token response
  admin_token="$(resolve_admin_token)"
  if [[ -z "${admin_token}" ]]; then
    cat >&2 <<'EOF'
Could not resolve a GitHub admin token.
Use one of these:
  1. GH_TOKEN="$(gh auth token)" sudo -E bash scripts/create_assess_speaking_runner.sh
  2. RUNNER_TOKEN="<repo registration token>" sudo -E bash scripts/create_assess_speaking_runner.sh
EOF
    exit 1
  fi

  response="$(curl -fsSL -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${admin_token}" \
    "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/runners/registration-token")"

  python3 - "${response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
token = payload.get("token")
if not token:
    raise SystemExit("GitHub API did not return a runner registration token.")
print(token)
PY
}

RUNNER_TOKEN="${RUNNER_TOKEN:-}"
if [[ -z "${RUNNER_TOKEN}" ]]; then
  RUNNER_TOKEN="$(create_registration_token)"
fi

echo "Creating runner:"
echo "  Repo:        ${GITHUB_URL}"
echo "  Name:        ${RUNNER_NAME}"
echo "  Directory:   ${RUNNER_DIR}"
echo "  Labels:      ${RUNNER_LABELS}"
echo "  Template:    ${TEMPLATE_DIR}"
echo "  Launchd:     ${PLIST_PATH}"
echo

mkdir -p "${RUNNER_DIR}"
rsync -a \
  --exclude '.runner' \
  --exclude '.runner_migrated' \
  --exclude '.credentials' \
  --exclude '.credentials_rsaparams' \
  --exclude '.service' \
  --exclude '.path' \
  --exclude '_work' \
  --exclude '_diag' \
  "${TEMPLATE_DIR}/" "${RUNNER_DIR}/"

mkdir -p "${RUNNER_DIR}/_work" "${RUNNER_DIR}/_diag" "${LOG_DIR}"
chown -R "${RUNNER_USER}:${RUNNER_GROUP}" "${RUNNER_DIR}" "${LOG_DIR}"

# Scrub any copied registration state so the clone can be configured as a new runner.
rm -f \
  "${RUNNER_DIR}/.runner" \
  "${RUNNER_DIR}/.runner_migrated" \
  "${RUNNER_DIR}/.credentials" \
  "${RUNNER_DIR}/.credentials_rsaparams" \
  "${RUNNER_DIR}/.service" \
  "${RUNNER_DIR}/.path"

sudo -u "${RUNNER_USER}" env HOME="/Users/${RUNNER_USER}" bash -lc "
  cd '${RUNNER_DIR}'
  ./config.sh \
    --url '${GITHUB_URL}' \
    --token '${RUNNER_TOKEN}' \
    --unattended \
    --replace \
    --name '${RUNNER_NAME}' \
    --work '_work' \
    --labels '${RUNNER_LABELS}'
"

cat > "${PLIST_PATH}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.icosa.runner.${RUNNER_NAME}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${RUNNER_DIR}/runsvc.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>UserName</key>
  <string>${RUNNER_USER}</string>
  <key>WorkingDirectory</key>
  <string>${RUNNER_DIR}</string>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/stderr.log</string>
</dict>
</plist>
EOF

chmod 644 "${PLIST_PATH}"
chown root:wheel "${PLIST_PATH}"

launchctl bootstrap system "${PLIST_PATH}"
launchctl enable "system/com.icosa.runner.${RUNNER_NAME}"
launchctl kickstart -k "system/com.icosa.runner.${RUNNER_NAME}"

echo
echo "Runner installed."
echo
echo "Verification:"
echo "  launchctl print system/com.icosa.runner.${RUNNER_NAME}"
echo "  GH_TOKEN=\"\$(gh auth token)\" gh api repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/runners --jq '.runners[] | select(.name == \"${RUNNER_NAME}\") | {name,status,busy,labels:[.labels[].name]}'"
echo
echo "Expected workflow target labels:"
echo "  self-hosted, macOS, ARM64, icosa-apple-ci"
