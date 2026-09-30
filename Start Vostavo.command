#!/bin/bash
set -e
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
  source "${NVM_DIR:-$HOME/.nvm}/nvm.sh"
  nvm use 24 || { echo "Node 24 is missing. Run: nvm install 24"; read -r; exit 1; }
fi
./scripts/python.sh scripts/start_practice.py || { echo "Press Return to close."; read -r; }
