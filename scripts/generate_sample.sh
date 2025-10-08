#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUTPUT="${PROJECT_ROOT}/samples/italian_demo.wav"
VOICE="Alice"
TEXT="Buongiorno. Oggi parlo della riqualificazione energetica degli edifici e del canone a scaglioni."
OUTPUT="$DEFAULT_OUTPUT"

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Create a demo Italian speech sample using macOS 'say' and ffmpeg.

Options:
  -o, --output PATH   Target WAV file (default: ${DEFAULT_OUTPUT})
  -v, --voice  NAME   macOS voice to use (default: ${VOICE})
  -t, --text   TEXT   Alternative text to synthesise
  -h, --help          Show this help message
USAGE
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    -v|--voice)
      VOICE="$2"
      shift 2
      ;;
    -t|--text)
      TEXT="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "error: unknown flag '$1'" >&2
      print_usage >&2
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if ((${#ARGS[@]})); then
  echo "error: unexpected positional arguments: ${ARGS[*]}" >&2
  print_usage >&2
  exit 1
fi

if ! command -v say >/dev/null 2>&1; then
  echo "error: macOS 'say' command not found. Install Command Line Tools or run on macOS." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found. Install via 'brew install ffmpeg'." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
TMP_AIFF=$(mktemp "${PROJECT_ROOT}/samples/tmp.XXXXXX.aiff")
trap 'rm -f "$TMP_AIFF"' EXIT

say -v "$VOICE" -o "$TMP_AIFF" "$TEXT"

ffmpeg -y -loglevel error -i "$TMP_AIFF" -ac 1 -ar 16000 "$OUTPUT"

echo "Sample saved to $OUTPUT"
