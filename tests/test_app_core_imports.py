from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_no_streamlit_import_guard(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_app_core_state_and_i18n_import_without_streamlit():
    _run_no_streamlit_import_guard(
        """
import builtins
import importlib
import sys

original_import = builtins.__import__

def block_streamlit_import(name, *args, **kwargs):
    if name == "streamlit" or name.startswith("streamlit."):
        raise AssertionError("app_core modules must not import streamlit")
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_streamlit_import
state_module = importlib.import_module("app_core.state")
i18n_module = importlib.import_module("app_core.i18n")
default_state = state_module.build_default_state()
assert default_state.draft.session_id.startswith("draft-")
assert i18n_module.t("home.title", locale="en") == "Vostavo"
assert "streamlit" not in sys.modules
"""
    )


def test_backend_service_imports_do_not_require_streamlit():
    _run_no_streamlit_import_guard(
        """
import builtins
import importlib
import sys

original_import = builtins.__import__

def block_streamlit_import(name, *args, **kwargs):
    if name == "streamlit" or name.startswith("streamlit."):
        raise AssertionError("backend service imports must not require streamlit")
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_streamlit_import
importlib.import_module("app_core.services")
importlib.import_module("app_backend.app")
assert "streamlit" not in sys.modules
"""
    )


def test_app_core_modules_do_not_import_app_shell_support_modules():
    _run_no_streamlit_import_guard(
        """
import builtins
import importlib

original_import = builtins.__import__

def block_app_shell_import(name, *args, **kwargs):
    if name == "app_shell" or name.startswith("app_shell."):
        raise AssertionError(f"app_core must not import {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_app_shell_import
importlib.import_module("app_core.state")
importlib.import_module("app_core.i18n")
importlib.import_module("app_core.services")
"""
    )
