#!/usr/bin/env python3
import argparse, json, math, subprocess, re, sys
from pathlib import Path

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    WhisperModel = None  # type: ignore

try:
    import parselmouth  # type: ignore
    from parselmouth.praat import call  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    parselmouth = None  # type: ignore
    call = None  # type: ignore

FILLERS = {"eh","ehm","mmm","cioè","allora","dunque","tipo","insomma"}
COHESION = {
    "inoltre","per quanto riguarda","tuttavia","ciò nonostante","in definitiva",
    "da un lato","dall’altro","a mio avviso","tenuto conto di","a quanto pare",
    "presumibilmente","parrebbe che","pertanto","quindi","invece","comunque"
}

def load_audio_features(wav_path: Path):
    if parselmouth is None or call is None:
        raise RuntimeError(
            "praat-parselmouth is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )
    snd = parselmouth.Sound(str(wav_path))
    intensity = snd.to_intensity()
    thr = call(intensity, "Get mean", 0, 0) - 10
    step = 0.01; t = 0.0; pauses = []; in_pause=False; p_start=0.0
    dur_total = snd.get_total_duration()
    while t < dur_total:
        val = call(intensity, "Get value at time", t, "Cubic")
        if (math.isnan(val) or val < thr):
            if not in_pause:
                in_pause=True; p_start=t
        else:
            if in_pause:
                dur = t - p_start
                if dur >= 0.3: pauses.append((p_start, t, dur))
                in_pause=False
        t += step
    if in_pause:
        dur = dur_total-p_start
        if dur >= 0.3: pauses.append((p_start, dur_total, dur))
    return {"duration_sec": dur_total, "pauses": pauses}

def transcribe(path: Path, model_size="large-v3"):
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )
    model = WhisperModel(model_size, compute_type="int8_float32")
    segments, info = model.transcribe(str(path), vad_filter=True, word_timestamps=True, language="it")
    words = []; full_text=[]
    for seg in segments:
        full_text.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                token = w.word.strip().lower()
                if token: words.append({"t0": w.start, "t1": w.end, "text": token})
    return {"text":" ".join(full_text).strip(), "words":words}

def metrics_from(words, audio_feats):
    duration = audio_feats["duration_sec"]
    pause_total = sum(p[2] for p in audio_feats["pauses"])
    speaking_time = max(0.001, duration - pause_total)
    tokens = [re.sub(r"[^a-zà-ù’']", "", w["text"]) for w in words]
    tokens = [t for t in tokens if t]
    word_count = len(tokens)
    wpm = word_count / (speaking_time/60.0)
    fillers = sum(1 for t in tokens if t in FILLERS)
    text = " " + re.sub(r"\s+", " ", " ".join(t for t in tokens)) + " "
    cohesion_hits = 0
    for m in COHESION:
        pat = r"\b" + re.escape(m).replace(" ", r"\\s+") + r"\b"
        cohesion_hits += len(re.findall(pat, text, flags=re.IGNORECASE))
    rel_markers = len(re.findall(r"\bche\b|\bcui\b|\bnella quale\b|\bnei quali\b", text))
    cond_markers = len(re.findall(r"\bse\b|\bqualora\b", text))
    complexity = rel_markers + cond_markers
    return {
        "duration_sec": round(duration,2),
        "pause_count": len(audio_feats["pauses"]),
        "pause_total_sec": round(pause_total,2),
        "speaking_time_sec": round(speaking_time,2),
        "word_count": word_count,
        "wpm": round(wpm,1),
        "fillers": fillers,
        "cohesion_markers": cohesion_hits,
        "complexity_index": complexity
    }

def rubric_prompt_it(transcript, metr):
    return f"""
Sei un esaminatore CEFR per italiano. Valuta SOLO la competenza orale in base al trascritto (potrebbero esserci errori ASR).
Assegna punteggi 1–5 per: 1) Fluidità, 2) Coerenza/Cohesione, 3) Correttezza grammaticale, 4) Ampiezza lessicale.
Dai anche 2 esempi concreti da migliorare per ciascun criterio e un voto complessivo (media).

METRICHE OGGETTIVE:
- Durata: {metr['duration_sec']} s; Tempo di parola: {metr['speaking_time_sec']} s; Pausa totale: {metr['pause_total_sec']} s; Pausenanzahl: {metr['pause_count']}
- Parole: {metr['word_count']} → WPM: {metr['wpm']}
- Filler: {metr['fillers']} ; Marcatori di coesione rilevati: {metr['cohesion_markers']}
- Indice di complessità (relativi/periodi ipotetici, euristico): {metr['complexity_index']}

TRASCRITTO:
\"\"\"{transcript.strip()}\"\"\"

RISPONDI IN JSON con le chiavi:
fluency, cohesion, accuracy, range, overall, comments_fluency, comments_cohesion, comments_accuracy, comments_range, overall_comment.
"""

def call_ollama(model, prompt):
    # Query Ollama HTTP API; also handle /api/tags for model validation
    try:
        proc = subprocess.run(
            ["curl","-s","http://localhost:11434/api/generate",
             "-d", json.dumps({"model": model, "prompt": prompt, "stream": False})],
            capture_output=True, text=True, check=True)
        resp = proc.stdout
        try:
            return json.loads(resp)["response"]
        except Exception:
            return resp
    except subprocess.CalledProcessError as e:
        return json.dumps({"error":"ollama_not_running_or_model_missing","detail":e.stderr})

def list_ollama_models():
    try:
        p = subprocess.run(["curl","-s","http://localhost:11434/api/tags"], capture_output=True, text=True, check=True)
        return p.stdout
    except subprocess.CalledProcessError as e:
        return json.dumps({"error":"ollama_tags_failed","detail":e.stderr})

def selftest(model="llama3.1"):
    prompt = "Valuta brevemente (JSON) un testo fittizio: 'Oggi parlo dell'efficienza energetica nelle case.' Dai punteggi CEFR 1–5 per fluency, cohesion, accuracy, range e un commento."
    return call_ollama(model, prompt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", type=Path, help="Pfad zu WAV/MP3/M4A/...")
    ap.add_argument("--whisper", default="large-v3", help="faster-whisper Modell")
    ap.add_argument("--llm", default="llama3.1", help="Ollama-Modell (z. B. llama3.1, llama3.2:3b, qwen2.5:14b)")
    ap.add_argument("--list-ollama", action="store_true", help="verfügbare Ollama-Modelle anzeigen")
    ap.add_argument("--selftest", action="store_true", help="Mini-Test gegen Ollama (ohne Audio) ausführen")
    args = ap.parse_args()

    if args.list_ollama:
        print(list_ollama_models()); return

    if args.selftest:
        print(selftest(args.llm)); return

    if not args.audio:
        print("Bitte Audio-Datei angeben oder --selftest bzw. --list-ollama nutzen.", file=sys.stderr)
        sys.exit(2)

    tmp_wav = args.audio
    if args.audio.suffix.lower() not in [".wav"]:
        tmp_wav = args.audio.with_suffix(".tmp.wav")
        try:
            subprocess.run(
                ["ffmpeg","-y","-i",str(args.audio),"-ac","1","-ar","16000",str(tmp_wav)],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for non-WAV input. Please install it via Homebrew: `brew install ffmpeg`.") from exc

    audio_feats = load_audio_features(tmp_wav)
    asr = transcribe(tmp_wav, args.whisper)
    metr = metrics_from(asr["words"], audio_feats)
    prompt = rubric_prompt_it(asr["text"], metr)
    llm_json = call_ollama(args.llm, prompt)

    out = {"metrics": metr, "transcript_preview": asr["text"][:400], "llm_rubric": llm_json}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if tmp_wav.name.endswith(".tmp.wav"):
        try: tmp_wav.unlink()
        except Exception: pass

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        sys.exit(1)
