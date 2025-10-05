#!/usr/bin/env bash
set -euo pipefail
VOICE="${1:-Alice}"
TEXT="${2:-Buongiorno. Oggi parlo della riqualificazione energetica degli edifici e del canone a scaglioni.}"
say -v "$VOICE" -o sample.aiff --data-format=LEF32@16000 "$TEXT"
ffmpeg -y -i sample.aiff -ac 1 -ar 16000 sample.wav
echo "Sample saved to sample.wav"
