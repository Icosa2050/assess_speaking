"""Keep the self-hosted report input compatible with production duration checks."""
import os
from pathlib import Path
import shlex
from string import Template
import subprocess
import sys

import yaml

import assess_speaking
from assessment_runtime.recording_policy import require_review_duration


ROOT = Path(__file__).resolve().parents[1]


def test_selfhosted_report_fixture_passes_decoded_duration_gate(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/real-asr-selfhosted.yml").read_text())
    step = next(step for step in workflow["jobs"]["sample-audio-integration"]["steps"]
                if step["name"] == "Generate sample assessment report")
    preparation, separator, command = step["run"].partition("python assess_speaking.py")
    assert separator, "The workflow must invoke the production assessment CLI"
    subprocess.run(["bash", "-e", "-c", preparation], cwd=ROOT, check=True, timeout=30,
                   env={**os.environ, "ARTIFACT_DIR": str(tmp_path),
                        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]})
    arguments = shlex.split(Template(command.replace("\\\n", "")).safe_substitute(
        ARTIFACT_DIR=str(tmp_path), WHISPER_MODEL="tiny"))
    audio = Path(arguments[0])
    if not audio.is_absolute():
        audio = ROOT / audio
    duration = assess_speaking.load_audio_features(audio)["duration_sec"]
    require_review_duration(duration)
