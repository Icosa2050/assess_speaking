#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_LANGUAGE="it"
DEFAULT_LEVEL="B1"
LANGUAGE="$DEFAULT_LANGUAGE"
LEVEL="$DEFAULT_LEVEL"
VOICE=""
TEXT=""
OUTPUT=""
GENERATE_ALL=0
LIST_PRESETS=0

supported_levels=("B1" "B2" "C1")
supported_languages=("it" "en")

preset_output() {
  local language="$1"
  local level="$2"
  case "${language}:${level}" in
    it:B1) echo "${PROJECT_ROOT}/samples/cefr/it/B1/travel_story.wav" ;;
    it:B2) echo "${PROJECT_ROOT}/samples/cefr/it/B2/remote_work.wav" ;;
    it:C1) echo "${PROJECT_ROOT}/samples/cefr/it/C1/public_debate.wav" ;;
    en:B1) echo "${PROJECT_ROOT}/samples/cefr/en/B1/travel_story.wav" ;;
    en:B2) echo "${PROJECT_ROOT}/samples/cefr/en/B2/remote_work.wav" ;;
    en:C1) echo "${PROJECT_ROOT}/samples/cefr/en/C1/public_debate.wav" ;;
    *)
      echo "error: unsupported preset ${language}/${level}" >&2
      exit 1
      ;;
  esac
}

preset_voice() {
  local language="$1"
  case "$language" in
    it) echo "Alice" ;;
    en) echo "Daniel" ;;
    *)
      echo "error: unsupported language '${language}'" >&2
      exit 1
      ;;
  esac
}

preset_text() {
  local language="$1"
  local level="$2"
  case "${language}:${level}" in
    it:B1)
      echo "Lo scorso anno ho fatto un viaggio in treno con mia sorella. Siamo andati al mare per tre giorni, abbiamo visitato il centro storico e abbiamo mangiato in un piccolo ristorante vicino alla spiaggia. Il momento piu bello e stato il tramonto dell'ultima sera, perche eravamo stanchi ma molto contenti."
      ;;
    it:B2)
      echo "Lavorare da casa puo essere comodo, perche permette di risparmiare tempo e organizzare meglio la giornata. Allo stesso tempo, pero, non e sempre facile separare il lavoro dalla vita privata, e molte persone sentono la mancanza di un vero contatto con i colleghi. Per me la soluzione migliore e un modello flessibile."
      ;;
    it:C1)
      echo "Il dibattito pubblico online offre nuove possibilita di partecipazione, ma amplifica anche polarizzazione, semplificazioni e reazioni impulsive. Se vogliamo che le piattaforme digitali sostengano una discussione democratica piu matura, dobbiamo combinare educazione mediatica, trasparenza degli algoritmi e responsabilita editoriale senza limitare inutilmente la liberta di espressione."
      ;;
    en:B1)
      echo "Last summer I took a short trip to the coast with two friends. We travelled by train, stayed in a small hotel, and spent most of the weekend walking near the beach and trying local food. The weather was warm, the town was friendly, and I still remember how relaxed I felt when we came home."
      ;;
    en:B2)
      echo "Working from home can improve concentration and save commuting time, which is why many employees prefer it for part of the week. However, remote work can also create communication problems and make teamwork slower if meetings are poorly organised. In my view, companies should offer a balanced system instead of one rigid rule for everyone."
      ;;
    en:C1)
      echo "Public debate has become faster and more visible because digital platforms allow almost anyone to react instantly to political events. That openness is valuable, yet it also rewards outrage, short-term attention, and simplified arguments. A healthier civic culture depends not only on better platform rules, but also on stronger habits of critical reading and deliberate discussion."
      ;;
    *)
      echo "error: unsupported preset ${language}/${level}" >&2
      exit 1
      ;;
  esac
}

render_sample() {
  local voice="$1"
  local text="$2"
  local output="$3"
  local tmp_aiff

  mkdir -p "$(dirname "$output")"
  tmp_aiff="$(mktemp "${PROJECT_ROOT}/samples/tmp.XXXXXX.aiff")"

  say -v "$voice" -o "$tmp_aiff" "$text"
  ffmpeg -y -loglevel error -i "$tmp_aiff" -ac 1 -ar 16000 "$output"
  rm -f "$tmp_aiff"
  echo "Sample saved to $output"
}

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Create CEFR-tagged sample speech files using macOS 'say' and ffmpeg.

Options:
  -l, --language CODE Learning language preset: it or en (default: ${DEFAULT_LANGUAGE})
  -c, --cefr LEVEL    CEFR preset: B1, B2, or C1 (default: ${DEFAULT_LEVEL})
  -o, --output PATH   Target WAV file (default: preset path for the selected language/level)
  -v, --voice  NAME   macOS voice to use (default: preset voice for the selected language)
  -t, --text   TEXT   Alternative text to synthesise
      --all           Generate all shipped EN/IT B1/B2/C1 sample files
      --list-presets  Show the built-in sample paths
  -h, --help          Show this help message
USAGE
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--language)
      LANGUAGE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    -c|--cefr|--level)
      LEVEL="$(printf '%s' "$2" | tr '[:lower:]' '[:upper:]')"
      shift 2
      ;;
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
    --all)
      GENERATE_ALL=1
      shift
      ;;
    --list-presets)
      LIST_PRESETS=1
      shift
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

case " ${supported_languages[*]} " in
  *" ${LANGUAGE} "*) ;;
  *)
    echo "error: unsupported language '${LANGUAGE}'. Use one of: ${supported_languages[*]}" >&2
    exit 1
    ;;
esac

case " ${supported_levels[*]} " in
  *" ${LEVEL} "*) ;;
  *)
    echo "error: unsupported CEFR level '${LEVEL}'. Use one of: ${supported_levels[*]}" >&2
    exit 1
    ;;
esac

if ! command -v say >/dev/null 2>&1; then
  echo "error: macOS 'say' command not found. Install Command Line Tools or run on macOS." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found. Install via 'brew install ffmpeg'." >&2
  exit 1
fi

if (( LIST_PRESETS )); then
  for language in "${supported_languages[@]}"; do
    for level in "${supported_levels[@]}"; do
      echo "${language}/${level}: $(preset_output "$language" "$level")"
    done
  done
  exit 0
fi

if (( GENERATE_ALL )); then
  if [[ -n "$TEXT" || -n "$OUTPUT" || -n "$VOICE" ]]; then
    echo "error: --all cannot be combined with --text, --output, or --voice." >&2
    exit 1
  fi
  for language in "${supported_languages[@]}"; do
    for level in "${supported_levels[@]}"; do
      render_sample \
        "$(preset_voice "$language")" \
        "$(preset_text "$language" "$level")" \
        "$(preset_output "$language" "$level")"
    done
  done
  exit 0
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$(preset_output "$LANGUAGE" "$LEVEL")"
fi
if [[ -z "$VOICE" ]]; then
  VOICE="$(preset_voice "$LANGUAGE")"
fi
if [[ -z "$TEXT" ]]; then
  TEXT="$(preset_text "$LANGUAGE" "$LEVEL")"
fi

render_sample "$VOICE" "$TEXT" "$OUTPUT"
