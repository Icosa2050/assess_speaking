# assess_speaking (lokal, Mac-optimiert) – v2

Lokale Pipeline: **Transkription (faster-whisper)** → **Metriken** → **CEFR-Rubrik** via **Ollama**.
Korrigierte Modelltags (ohne `:instruct`) + **Self-Test**.

## 0) Voraussetzungen
```bash
brew install ffmpeg ollama
ollama pull llama3.1              # oder: llama3.2:3b / qwen2.5:14b
ollama list
```

## 1) Virtuelle Umgebung
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 2) Testaudio ohne Mikrofon
```bash
say -v "Alice" -o sample.aiff "Buongiorno. Oggi parlo della riqualificazione energetica degli edifici."
ffmpeg -y -i sample.aiff -ac 1 -ar 16000 sample.wav
```

## 3) Modelle prüfen & Self-Test
```bash
python assess_speaking.py --list-ollama
python assess_speaking.py --selftest --llm llama3.1
```

## 4) Bewertung laufen lassen
```bash
python assess_speaking.py sample.wav --whisper large-v3 --llm llama3.1 > report.json
cat report.json
```

## Hinweise
- Default-LLM ist **llama3.1** (ohne `:instruct`).
- Alternativen: `llama3.2:3b` (schnell), `qwen2.5:14b` (stärker), je nach RAM/Speed.
- Objektive Metriken: **WPM**, Pausen (≥300ms), Marker-Zählung, Komplexitätsindex (Relativ-/Konditionalsätze).

## Lizenz
MIT
