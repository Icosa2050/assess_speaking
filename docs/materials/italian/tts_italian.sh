#!/usr/bin/env bash
#
# tts_italian.sh — Italian TTS via Piper (neural, offline).
#
# Reads a UTF-8 text file and renders an audio file next to it.
# Voice models are cached in $HOME/piper-voices and downloaded on first use.

set -euo pipefail
export LC_NUMERIC=C

# ---------- Defaults ----------
VOICE="paola"
RATE=120
FORMAT="m4a"
OUTPUT=""
INPUT=""

PIPER_VOICES_DIR="${PIPER_VOICES_DIR:-$HOME/piper-voices}"
PIPER_BASELINE_WPM=160   # piper's approximate natural pace for Italian
HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT"

declare -A VOICE_QUALITY=(
  [paola]="medium"
  [riccardo]="x_low"
)

# ---------- Helpers ----------
err()  { printf 'Error: %s\n' "$*" >&2; }
info() { printf '→ %s\n' "$*"; }
die()  { err "$*"; exit 1; }

usage() {
  cat <<'EOF'
Usage: tts_italian.sh <input.txt> [options]

Options:
  -v, --voice VOICE     paola (female, medium) | riccardo (male, x_low)   [default: paola]
  -r, --rate WPM        Target words/minute, 50–300 (approximate)         [default: 120]
  -f, --format FORMAT   mp3 | m4a | wav                                    [default: m4a]
  -o, --output PATH     Output path (default: <input>.<format> next to input)
  -l, --list-voices     List supported voices and exit
  -h, --help            Show this help and exit

Voice models are cached in $HOME/piper-voices (override with PIPER_VOICES_DIR env var).
EOF
}

list_voices() {
  printf 'Supported Italian voices:\n'
  for v in "${!VOICE_QUALITY[@]}"; do
    printf '  %-10s (quality: %s)\n' "$v" "${VOICE_QUALITY[$v]}"
  done | sort
}

ensure_voice() {
  local name="$1" quality voice_dir base onnx
  quality="${VOICE_QUALITY[$name]:-}"
  [[ -n "$quality" ]] || die "Unknown voice: $name (try --list-voices)"

  voice_dir="$PIPER_VOICES_DIR/it_IT-$name-$quality"
  onnx="$voice_dir/it_IT-$name-$quality.onnx"
  base="$HF_BASE/$name/$quality/it_IT-$name-$quality"

  if [[ -f "$onnx" && -f "$onnx.json" ]]; then
    printf '%s\n' "$onnx"; return 0
  fi

  command -v curl >/dev/null || die "curl is required to download voice models"
  info "Downloading voice '$name' ($quality) to $voice_dir …"
  mkdir -p "$voice_dir"
  curl -fL --progress-bar -o "$onnx.part"      "$base.onnx"      || die "Voice model download failed: $base.onnx"
  curl -fL --progress-bar -o "$onnx.json.part" "$base.onnx.json" || { rm -f "$onnx.part"; die "Voice config download failed"; }
  mv "$onnx.part"      "$onnx"
  mv "$onnx.json.part" "$onnx.json"
  printf '%s\n' "$onnx"
}

# ---------- Argument parsing ----------
LIST_VOICES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--voice)       VOICE="${2:?missing voice}";   shift 2 ;;
    -r|--rate)        RATE="${2:?missing rate}";     shift 2 ;;
    -f|--format)      FORMAT="${2:?missing format}"; shift 2 ;;
    -o|--output)      OUTPUT="${2:?missing path}";   shift 2 ;;
    -l|--list-voices) LIST_VOICES=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; break ;;
    -*)               die "Unknown option: $1 (try --help)" ;;
    *)
      [[ -z "$INPUT" ]] || die "Multiple inputs not supported: $1"
      INPUT="$1"; shift
      ;;
  esac
done

(( LIST_VOICES )) && { list_voices; exit 0; }

# ---------- Validation ----------
command -v piper >/dev/null    || die "'piper' not found in PATH (brew install piper-tts)"
[[ -n "$INPUT" ]]              || { usage; exit 1; }
[[ -f "$INPUT" ]]              || die "Input file not found: $INPUT"
[[ -s "$INPUT" ]]              || die "Input file is empty: $INPUT"
[[ "$RATE" =~ ^[0-9]+$ ]]      || die "Rate must be a positive integer: $RATE"
(( RATE >= 50 && RATE <= 300 ))|| die "Rate out of range 50–300: $RATE"
[[ -n "${VOICE_QUALITY[$VOICE]:-}" ]] || die "Unknown voice: $VOICE (try --list-voices)"

case "$FORMAT" in
  mp3|m4a|wav) ;;
  *) die "Unsupported format: $FORMAT (use mp3|m4a|wav)" ;;
esac

ENCODER=""
if [[ "$FORMAT" != "wav" ]]; then
  if [[ "$FORMAT" == "m4a" ]] && command -v afconvert >/dev/null; then
    ENCODER="afconvert"
  elif command -v ffmpeg >/dev/null; then
    ENCODER="ffmpeg"
  else
    die "Encoding to $FORMAT requires afconvert (macOS) or ffmpeg (brew install ffmpeg)"
  fi
fi

# ---------- Output path ----------
if [[ -z "$OUTPUT" ]]; then
  INPUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
  INPUT_BASE="$(basename "${INPUT%.*}")"
  OUTPUT="${INPUT_DIR}/${INPUT_BASE}.${FORMAT}"
fi
mkdir -p "$(dirname "$OUTPUT")"

# ---------- Voice + length scale ----------
MODEL="$(ensure_voice "$VOICE")"
LENGTH_SCALE=$(awk -v b="$PIPER_BASELINE_WPM" -v r="$RATE" 'BEGIN { printf "%.3f", b/r }')

# ---------- Generate ----------
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
RAW_WAV="$WORK_DIR/out.wav"

info "Input:       $INPUT"
info "Voice:       $VOICE (${VOICE_QUALITY[$VOICE]})"
info "Rate:        ${RATE} wpm (length_scale=$LENGTH_SCALE)"
info "Output:      $OUTPUT"

piper --model "$MODEL" --length-scale "$LENGTH_SCALE" \
      -i "$INPUT" -f "$RAW_WAV" 2>"$WORK_DIR/piper.log" \
  || { cat "$WORK_DIR/piper.log" >&2; die "piper synthesis failed"; }

[[ -s "$RAW_WAV" ]] || die "piper produced no audio (empty wav)"

# ---------- Encode ----------
case "$FORMAT" in
  wav) cp "$RAW_WAV" "$OUTPUT" ;;
  m4a)
    if [[ "$ENCODER" == "afconvert" ]]; then
      afconvert -f m4af -d aac "$RAW_WAV" "$OUTPUT" >/dev/null
    else
      ffmpeg -hide_banner -loglevel error -y -i "$RAW_WAV" -c:a aac -b:a 128k "$OUTPUT"
    fi
    ;;
  mp3)
    ffmpeg -hide_banner -loglevel error -y -i "$RAW_WAV" -codec:a libmp3lame -qscale:a 2 "$OUTPUT"
    ;;
esac

# ---------- Report ----------
WORDS=$(wc -w < "$INPUT" | tr -d ' ')
EST_DUR=$(awk -v w="$WORDS" -v r="$RATE" 'BEGIN { printf "%.1f", (w/r)*60 }')
info "Synthesized: $WORDS words, est. ${EST_DUR}s at ${RATE} wpm"

if command -v afinfo >/dev/null; then
  ACTUAL=$(afinfo "$OUTPUT" 2>/dev/null | awk -F': ' '/estimated duration/ {print $2}' | awk '{print $1}')
  [[ -n "$ACTUAL" ]] && info "Actual:      ${ACTUAL}s"
elif command -v ffprobe >/dev/null; then
  ACTUAL=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUTPUT" 2>/dev/null)
  [[ -n "$ACTUAL" ]] && info "Actual:      $(printf '%.1f' "$ACTUAL")s"
fi
