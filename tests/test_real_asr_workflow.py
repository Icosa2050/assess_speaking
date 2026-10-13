"""Guard workflow Python snippets and self-hosted report input."""
import ast
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import re
import shlex
from string import Template
import subprocess
import sys

import yaml

import assess_speaking
from assessment_runtime.recording_policy import require_review_duration


ROOT = Path(__file__).resolve().parents[1]
HEREDOC_START = re.compile(r"\s*python(?:3(?:\.\d+)?)?\s+-\s+<<\s*['\"]?(\w+)['\"]?\s*")
PYTHON_COMMAND = re.compile(r"(?<![\w./-])python(?:3(?:\.\d+)?)?\s+-c\s+")


def _run_commands(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "run" and isinstance(child, str):
                yield child
            else:
                yield from _run_commands(child)
    elif isinstance(value, list):
        for child in value:
            yield from _run_commands(child)


def _inline_python(run):
    lines = run.splitlines()
    index = 0
    while index < len(lines):
        match = HEREDOC_START.fullmatch(lines[index])
        if match:
            marker = match.group(1)
            end = index + 1
            while end < len(lines) and lines[end].strip() != marker:
                end += 1
            assert end < len(lines), f"Unterminated Python heredoc {marker}"
            yield "\n".join(lines[index + 1:end])
            index = end + 1
            continue
        for command in PYTHON_COMMAND.finditer(lines[index]):
            remainder = lines[index][command.end():] + "\n" + "\n".join(lines[index + 1:])
            lexer = shlex.shlex(remainder, posix=True)
            lexer.whitespace_split = True
            source = next(lexer, None)
            assert source, "python -c must have a code argument"
            yield source
        index += 1


def _module_resolves(name):
    parts = name.split(".")
    spec = importlib.machinery.PathFinder.find_spec(parts[0], [str(ROOT), *sys.path])
    if spec is None:
        spec = importlib.util.find_spec(parts[0])
    for part in parts[1:]:
        if spec is None or spec.submodule_search_locations is None:
            return False
        spec = importlib.machinery.PathFinder.find_spec(part, spec.submodule_search_locations)
    return spec is not None


def test_workflow_inline_python_compiles_and_imports_resolve():
    scripts = []
    for workflow_path in sorted((ROOT / ".github/workflows").rglob("*")):
        if workflow_path.suffix not in {".yml", ".yaml"}:
            continue
        workflow = yaml.safe_load(workflow_path.read_text())
        for run in _run_commands(workflow):
            scripts.extend((workflow_path, source) for source in _inline_python(run))

    assert scripts, "Expected to find inline Python in the workflows"
    for workflow_path, source in scripts:
        label = str(workflow_path.relative_to(ROOT))
        tree = ast.parse(source, filename=label)
        compile(tree, label, "exec")
        modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names]
        modules.extend(node.module for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module)
        for module in modules:
            assert _module_resolves(module), f"{label}: cannot resolve import {module}"


def test_inline_python_finds_heredocs_and_c_commands():
    run = "python - <<'PY'\nimport os\nPY\npython -c 'import pathlib'"
    assert list(_inline_python(run)) == ["import os", "import pathlib"]


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
