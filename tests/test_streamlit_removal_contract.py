from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = ROOT / root
        if path.is_file():
            files.append(path)
        elif path.exists():
            files.extend(sorted(item for item in path.rglob("*.py") if item.is_file()))
    return files


def test_no_product_python_imports_streamlit():
    offenders: list[str] = []
    for path in _python_files("app_backend", "app_core", "assess_core", "assessment_runtime", "scripts", "assess_speaking.py"):
        text = path.read_text(encoding="utf-8")
        if "import streamlit" in text or "from streamlit" in text or "streamlit_webrtc" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_product_python_does_not_import_deleted_app_shell_package():
    offenders: list[str] = []
    for path in _python_files("app_backend", "app_core", "assess_core", "assessment_runtime", "scripts", "assess_speaking.py"):
        text = path.read_text(encoding="utf-8")
        if "from app_shell" in text or "import app_shell" in text or "app_shell." in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_product_code_does_not_keep_app_shell_compatibility_names():
    offenders: list[str] = []
    forbidden_tokens = (
        "APP_SHELL_",
        "RUN_APP_SHELL_",
        "_app_shell",
        "app_shell_state",
        '"shell/',
        "'shell/",
        "App-shell",
    )
    for path in [
        *_python_files("app_backend", "app_core", "assess_core", "assessment_runtime", "scripts", "assess_speaking.py"),
        *(ROOT / "frontend" / "tests" / "e2e").glob("*.ts"),
        *(ROOT / "frontend" / "src").rglob("*.ts"),
        *(ROOT / "frontend" / "src").rglob("*.tsx"),
    ]:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden_tokens):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_legacy_streamlit_files_are_removed():
    assert not (ROOT / "streamlit_app.py").exists()
    assert not (ROOT / "pages").exists()
    assert not (ROOT / "app_shell").exists()


def test_requirements_do_not_install_streamlit():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "streamlit==" not in requirements
    assert "streamlit-webrtc" not in requirements
