from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        ".playwright",
        "build",
        "dist",
        "node_modules",
        "output",
        "playwright-report",
        "reports",
        "test-results",
        "tmp",
        "traces",
    }
)
DEFAULT_LOCALE_MARKERS = ("locales", "locale", "i18n", "l10n")
DEFAULT_TRANSLATION_FUNCTIONS = ("t", "translate")
CODE_SUFFIXES = frozenset({".py", ".js", ".jsx", ".ts", ".tsx"})
PYTHON_SUFFIXES = frozenset({".py"})
PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")
DEFAULT_ALLOWED_ROOT_PYTHON_FILES = frozenset(
    {
        "assess_speaking.py",
        "streamlit_app.py",
    }
)
DEFAULT_ROOT_ARTIFACT_NAMES = frozenset({".coverage", "coverage.json", "-o", "--file-format=AIFF"})
DEFAULT_ROOT_PYTHON_SPRAWL_THRESHOLD = 4


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    path: str
    line: int | None
    message: str
    hint: str
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "check": self.check,
            "path": self.path,
            "line": self.line,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


@dataclass(frozen=True)
class CoverageSummary:
    ran: bool
    total_percent: float | None = None
    files_below_threshold: list[dict[str, object]] = field(default_factory=list)
    error: str | None = None
    command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ran": self.ran,
            "total_percent": self.total_percent,
            "files_below_threshold": self.files_below_threshold,
            "error": self.error,
            "command": self.command,
        }


@dataclass(frozen=True)
class AuditReport:
    root: str
    findings: list[Finding]
    coverage: CoverageSummary

    def to_dict(self) -> dict[str, object]:
        counts = {severity: 0 for severity in SEVERITY_ORDER}
        for finding in self.findings:
            counts[finding.severity] += 1
        return {
            "root": self.root,
            "summary": counts,
            "coverage": self.coverage.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def path_is_excluded(path: Path, root: Path, excluded_dirs: set[str]) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts:
        if part in excluded_dirs:
            return True
        if part.startswith(".") and part not in {".codex"}:
            return True
    return False


def is_test_path(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return "tests" in rel_parts or path.name.startswith("test_")


def walk_files(
    root: Path,
    suffixes: set[str],
    *,
    excluded_dirs: set[str],
    include_tests: bool,
) -> list[Path]:
    discovered: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in excluded_dirs and not dirname.startswith(".")
        ]
        if path_is_excluded(dir_path, root, excluded_dirs):
            dirnames[:] = []
            continue
        for filename in filenames:
            path = dir_path / filename
            if path.suffix.lower() not in suffixes:
                continue
            if path_is_excluded(path, root, excluded_dirs):
                continue
            if not include_tests and is_test_path(path, root):
                continue
            discovered.append(path)
    return sorted(discovered)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def scan_root_layout(
    root: Path,
    *,
    allowed_root_python_files: set[str] | None = None,
    root_artifact_names: set[str] | None = None,
    root_python_sprawl_threshold: int = DEFAULT_ROOT_PYTHON_SPRAWL_THRESHOLD,
) -> list[Finding]:
    allowed_root_python_files = allowed_root_python_files or set(DEFAULT_ALLOWED_ROOT_PYTHON_FILES)
    root_artifact_names = root_artifact_names or set(DEFAULT_ROOT_ARTIFACT_NAMES)

    findings: list[Finding] = []
    root_files = sorted(path for path in root.iterdir() if path.is_file())
    root_python_files = [path for path in root_files if path.suffix == ".py"]

    for path in root_files:
        if path.name in root_artifact_names or path.name.startswith("-"):
            findings.append(
                Finding(
                    severity="MEDIUM",
                    check="root-generated-artifact",
                    path=path.name,
                    line=None,
                    message=f"Generated or accidental artifact '{path.name}' is sitting in the repo root.",
                    hint="Ignore, relocate, or delete shell output and generated files instead of letting them accumulate at the root.",
                )
            )

    for path in root_python_files:
        if path.name not in allowed_root_python_files:
            findings.append(
                Finding(
                    severity="MEDIUM",
                    check="unexpected-root-python-module",
                    path=path.name,
                    line=None,
                    message=f"Root-level Python file '{path.name}' is not part of the approved root module inventory.",
                    hint="Keep stable entrypoints at the root and move new Python modules into a package unless there is a documented exception.",
                )
            )

    if len(root_python_files) > root_python_sprawl_threshold:
        findings.append(
            Finding(
                severity="LOW",
                check="root-python-module-sprawl",
                path=".",
                line=None,
                message=(
                    f"Repository root contains {len(root_python_files)} Python files, above the "
                    f"{root_python_sprawl_threshold}-file layout budget."
                ),
                hint="Treat the root as an entrypoint surface and move leaf modules into packages before adding more top-level Python files.",
                details={"count": len(root_python_files), "files": [path.name for path in root_python_files]},
            )
        )

    return findings


def flatten_locale_map(value: dict[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, child in value.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            flattened.update(flatten_locale_map(child, prefix=full_key))
        else:
            flattened[full_key] = child
    return flattened


def extract_placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(text))


def build_translation_key_regex(function_names: Sequence[str]) -> re.Pattern[str]:
    joined = "|".join(re.escape(name) for name in function_names)
    return re.compile(rf"\b(?:{joined})\(\s*(['\"])(?P<key>[A-Za-z0-9_.-]+)\1")


def discover_locale_dirs(root: Path, excluded_dirs: set[str], locale_markers: Sequence[str]) -> list[Path]:
    locale_dirs: list[Path] = []
    seen: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in excluded_dirs and not dirname.startswith(".")
        ]
        if path_is_excluded(dir_path, root, excluded_dirs):
            dirnames[:] = []
            continue
        if not filenames:
            continue
        lowered_name = dir_path.name.lower()
        if not any(marker in lowered_name for marker in locale_markers):
            continue
        json_files = sorted(path for path in dir_path.iterdir() if path.suffix.lower() == ".json" and path.is_file())
        if len(json_files) < 2 or dir_path in seen:
            continue
        seen.add(dir_path)
        locale_dirs.append(dir_path)
    return sorted(locale_dirs)


def scan_translation_issues(
    root: Path,
    *,
    excluded_dirs: set[str],
    locale_markers: Sequence[str],
    translation_functions: Sequence[str],
) -> list[Finding]:
    findings: list[Finding] = []
    locale_dirs = discover_locale_dirs(root, excluded_dirs, locale_markers)
    if not locale_dirs:
        return findings

    literal_key_regex = build_translation_key_regex(translation_functions)
    used_keys: set[str] = set()
    code_files = walk_files(root, CODE_SUFFIXES, excluded_dirs=excluded_dirs, include_tests=False)
    for path in code_files:
        used_keys.update(match.group("key") for match in literal_key_regex.finditer(read_text(path)))

    for locale_dir in locale_dirs:
        json_files = sorted(path for path in locale_dir.iterdir() if path.suffix.lower() == ".json" and path.is_file())
        locale_payloads: dict[str, dict[str, object]] = {}
        for path in json_files:
            try:
                payload = json.loads(read_text(path))
            except json.JSONDecodeError as exc:
                findings.append(
                    Finding(
                        severity="HIGH",
                        check="locale-parse-error",
                        path=str(path.relative_to(root)),
                        line=exc.lineno,
                        message=f"Locale file could not be parsed: {exc.msg}.",
                        hint="Fix the JSON syntax before relying on translation checks.",
                    )
                )
                continue
            if not isinstance(payload, dict):
                findings.append(
                    Finding(
                        severity="HIGH",
                        check="locale-root-not-object",
                        path=str(path.relative_to(root)),
                        line=None,
                        message="Locale file root must be a JSON object.",
                        hint="Rewrite the locale file to use nested translation objects.",
                    )
                )
                continue
            locale_payloads[path.stem] = flatten_locale_map(payload)

        if not locale_payloads:
            continue

        baseline_locale = "en" if "en" in locale_payloads else sorted(locale_payloads)[0]
        baseline_values = locale_payloads[baseline_locale]
        baseline_keys = set(baseline_values)

        for locale_name, locale_values in sorted(locale_payloads.items()):
            if locale_name == baseline_locale:
                continue
            missing_keys = sorted(baseline_keys - set(locale_values))
            extra_keys = sorted(set(locale_values) - baseline_keys)
            if missing_keys:
                findings.append(
                    Finding(
                        severity="MEDIUM",
                        check="locale-missing-keys",
                        path=str((locale_dir / f"{locale_name}.json").relative_to(root)),
                        line=None,
                        message=(
                            f"Locale '{locale_name}' is missing {len(missing_keys)} keys from baseline '{baseline_locale}', "
                            f"including {', '.join(missing_keys[:3])}."
                        ),
                        hint=f"Mirror the missing keys from {baseline_locale}.json and translate them.",
                        details={"missing_keys": missing_keys},
                    )
                )
            if extra_keys:
                findings.append(
                    Finding(
                        severity="LOW",
                        check="locale-extra-keys",
                        path=str((locale_dir / f"{locale_name}.json").relative_to(root)),
                        line=None,
                        message=(
                            f"Locale '{locale_name}' has {len(extra_keys)} extra keys not present in baseline '{baseline_locale}', "
                            f"including {', '.join(extra_keys[:3])}."
                        ),
                        hint=f"Remove stale keys or add them back to {baseline_locale}.json if they are still valid.",
                        details={"extra_keys": extra_keys},
                    )
                )

        for key in sorted(baseline_keys):
            baseline_value = baseline_values.get(key)
            if not isinstance(baseline_value, str):
                findings.append(
                    Finding(
                        severity="LOW",
                        check="locale-non-string-leaf",
                        path=str((locale_dir / f"{baseline_locale}.json").relative_to(root)),
                        line=None,
                        message=f"Locale key '{key}' resolves to a non-string value in baseline '{baseline_locale}'.",
                        hint="Locale leaf values should be strings so formatting and lookup stay predictable.",
                    )
                )
                continue
            baseline_placeholders = extract_placeholders(baseline_value)
            for locale_name, locale_values in sorted(locale_payloads.items()):
                candidate_value = locale_values.get(key)
                if not isinstance(candidate_value, str):
                    continue
                locale_placeholders = extract_placeholders(candidate_value)
                if locale_placeholders != baseline_placeholders:
                    findings.append(
                        Finding(
                            severity="MEDIUM",
                            check="locale-placeholder-mismatch",
                            path=str((locale_dir / f"{locale_name}.json").relative_to(root)),
                            line=None,
                            message=(
                                f"Locale key '{key}' uses placeholders {sorted(locale_placeholders)} in '{locale_name}' "
                                f"but {sorted(baseline_placeholders)} in '{baseline_locale}'."
                            ),
                            hint="Keep placeholder names aligned across locales so formatting does not fail at runtime.",
                            details={
                                "key": key,
                                "baseline_locale": baseline_locale,
                                "baseline_placeholders": sorted(baseline_placeholders),
                                "locale_placeholders": sorted(locale_placeholders),
                            },
                        )
                    )

        for key in sorted(used_keys):
            if key not in baseline_keys:
                findings.append(
                    Finding(
                        severity="HIGH",
                        check="translation-key-missing",
                        path=str(locale_dir.relative_to(root)),
                        line=None,
                        message=(
                            f"Code uses translation key '{key}', but it is missing from baseline locale '{baseline_locale}'."
                        ),
                        hint=f"Add '{key}' to {baseline_locale}.json and keep the other locales in sync.",
                    )
                )

        unused_keys = sorted(baseline_keys - used_keys)
        if unused_keys:
            findings.append(
                Finding(
                    severity="LOW",
                    check="locale-unused-keys",
                    path=str((locale_dir / f"{baseline_locale}.json").relative_to(root)),
                    line=None,
                    message=(
                        f"Baseline locale '{baseline_locale}' contains {len(unused_keys)} keys that were not found in code, "
                        f"including {', '.join(unused_keys[:3])}."
                    ),
                    hint="Delete stale locale entries or confirm whether they are only used by dynamic key lookups.",
                    details={"unused_keys": unused_keys},
                )
            )

    return findings


def is_broad_exception(exception_type: ast.expr | None) -> bool:
    if exception_type is None:
        return True
    if isinstance(exception_type, ast.Name):
        return exception_type.id in {"Exception", "BaseException"}
    if isinstance(exception_type, ast.Attribute):
        return exception_type.attr in {"Exception", "BaseException"}
    if isinstance(exception_type, ast.Tuple):
        return any(is_broad_exception(item) for item in exception_type.elts)
    return False


def is_silent_handler(body: Sequence[ast.stmt]) -> bool:
    if not body:
        return True
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, (ast.Continue, ast.Break)):
        return True
    return isinstance(statement, ast.Return) and statement.value is None


class PythonQualityVisitor(ast.NodeVisitor):
    def __init__(self, root: Path, path: Path) -> None:
        self._root = root
        self._path = path
        self.findings: list[Finding] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        relative_path = str(self._path.relative_to(self._root))
        if is_silent_handler(node.body):
            self.findings.append(
                Finding(
                    severity="HIGH",
                    check="silent-except",
                    path=relative_path,
                    line=node.lineno,
                    message="Exception handler swallows errors without preserving context or recovery behavior.",
                    hint="Handle the exception explicitly, re-raise it, or log enough detail to make failures visible.",
                )
            )
        elif is_broad_exception(node.type):
            self.findings.append(
                Finding(
                    severity="MEDIUM",
                    check="broad-except",
                    path=relative_path,
                    line=node.lineno,
                    message="Broad exception handling can hide unrelated failures and make debugging harder.",
                    hint="Catch the narrowest expected exception type instead of using a blanket handler.",
                )
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_mutable_defaults(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_mutable_defaults(node)
        self.generic_visit(node)

    def _check_mutable_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        defaults = list(node.args.defaults) + [default for default in node.args.kw_defaults if default is not None]
        for default in defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.findings.append(
                    Finding(
                        severity="MEDIUM",
                        check="mutable-default-argument",
                        path=str(self._path.relative_to(self._root)),
                        line=default.lineno,
                        message=f"Function '{node.name}' uses a mutable default argument.",
                        hint="Use None as the default and create a new list, dict, or set inside the function.",
                    )
                )


def scan_python_quality(root: Path, *, excluded_dirs: set[str], include_tests: bool) -> list[Finding]:
    findings: list[Finding] = []
    for path in walk_files(root, PYTHON_SUFFIXES, excluded_dirs=excluded_dirs, include_tests=include_tests):
        try:
            tree = ast.parse(read_text(path))
        except SyntaxError as exc:
            findings.append(
                Finding(
                    severity="HIGH",
                    check="python-parse-error",
                    path=str(path.relative_to(root)),
                    line=exc.lineno,
                    message=f"Python file could not be parsed: {exc.msg}.",
                    hint="Fix the syntax error before relying on the rest of the audit output.",
                )
            )
            continue
        visitor = PythonQualityVisitor(root, path)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


class CanonicalizeNames(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        cloned = copy.deepcopy(node)
        cloned.arg = "_"
        return cloned

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if isinstance(value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        if isinstance(value, bytes):
            return ast.copy_location(ast.Constant(value=b"BYTES"), node)
        if isinstance(value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value=0), node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        cloned = copy.deepcopy(node)
        cloned.name = "_"
        self.generic_visit(cloned)
        return cloned

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        cloned = copy.deepcopy(node)
        cloned.name = "_"
        self.generic_visit(cloned)
        return cloned

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        cloned = copy.deepcopy(node)
        cloned.name = "_"
        self.generic_visit(cloned)
        return cloned


@dataclass(frozen=True)
class FunctionSignature:
    path: Path
    qualname: str
    lineno: int
    line_count: int
    fingerprint: str


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        self.stack: list[str] = []
        self.functions: list[FunctionSignature] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        line_count = max(1, getattr(node, "end_lineno", node.lineno) - node.lineno + 1)
        if len(node.body) < 3 or line_count < 8:
            return
        module = ast.Module(body=copy.deepcopy(node.body), type_ignores=[])
        canonical = CanonicalizeNames().visit(module)
        ast.fix_missing_locations(canonical)
        fingerprint = ast.dump(canonical, annotate_fields=False, include_attributes=False)
        qualname = ".".join(self.stack + [node.name])
        self.functions.append(
            FunctionSignature(
                path=self.source_path,
                qualname=qualname,
                lineno=node.lineno,
                line_count=line_count,
                fingerprint=fingerprint,
            )
        )


def scan_duplicate_functions(root: Path, *, excluded_dirs: set[str], include_tests: bool) -> list[Finding]:
    signatures: dict[str, list[FunctionSignature]] = {}
    for path in walk_files(root, PYTHON_SUFFIXES, excluded_dirs=excluded_dirs, include_tests=include_tests):
        tree: ast.AST | None
        try:
            tree = ast.parse(read_text(path))
        except SyntaxError:
            # Parse failures are already surfaced by the quality scan; duplication
            # analysis can only continue on valid syntax trees.
            tree = None
        if tree is None:
            continue
        collector = FunctionCollector(path)
        collector.visit(tree)
        for signature in collector.functions:
            signatures.setdefault(signature.fingerprint, []).append(signature)

    findings: list[Finding] = []
    for duplicates in signatures.values():
        if len(duplicates) < 2:
            continue
        ordered = sorted(duplicates, key=lambda item: (str(item.path), item.lineno, item.qualname))
        original = ordered[0]
        for duplicate in ordered[1:]:
            severity = "MEDIUM" if duplicate.line_count >= 15 else "LOW"
            findings.append(
                Finding(
                    severity=severity,
                    check="duplicate-function-logic",
                    path=str(duplicate.path.relative_to(root)),
                    line=duplicate.lineno,
                    message=(
                        f"Function '{duplicate.qualname}' appears structurally duplicated from "
                        f"'{original.qualname}' in {original.path.relative_to(root)}:{original.lineno}."
                    ),
                    hint="Extract the shared logic into a helper or remove the stale copy if behavior must stay aligned.",
                    details={
                        "duplicate_of": {
                            "path": str(original.path.relative_to(root)),
                            "line": original.lineno,
                            "qualname": original.qualname,
                        }
                    },
                )
            )
    return findings


def count_effective_lines(path: Path) -> int:
    count = 0
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def source_kind(rel_path: Path) -> str:
    if rel_path.parts and rel_path.parts[0] == "scripts":
        return "script"
    if rel_path.parts and rel_path.parts[0] == "pages":
        return "page"
    return "module"


def has_test_reference(rel_path: Path, test_files: Sequence[Path], root: Path) -> bool:
    stem = rel_path.stem
    joined = "_".join(rel_path.with_suffix("").parts)
    dotted = ".".join(rel_path.with_suffix("").parts)
    expected_names = {f"test_{stem}.py", f"test_{joined}.py"}
    if any(path.name in expected_names for path in test_files):
        return True

    parent_module = ".".join(rel_path.with_suffix("").parts[:-1])
    import_patterns = [
        dotted,
        rel_path.as_posix(),
        f"import {dotted}",
        f"from {dotted} import",
    ]
    if parent_module:
        import_patterns.append(f"from {parent_module} import {stem}")

    for test_path in test_files:
        content = read_text(test_path)
        if any(pattern in content for pattern in import_patterns):
            return True
    return False


def scan_missing_tests(root: Path, *, excluded_dirs: set[str]) -> list[Finding]:
    source_files = walk_files(root, PYTHON_SUFFIXES, excluded_dirs=excluded_dirs, include_tests=False)
    test_files = [path for path in walk_files(root, PYTHON_SUFFIXES, excluded_dirs=excluded_dirs, include_tests=True) if is_test_path(path, root)]

    findings: list[Finding] = []
    for path in source_files:
        rel_path = path.relative_to(root)
        if path.name == "__init__.py":
            continue
        if count_effective_lines(path) < 25:
            continue
        if has_test_reference(rel_path, test_files, root):
            continue
        kind = source_kind(rel_path)
        severity = "LOW" if kind in {"script", "page"} else "MEDIUM"
        findings.append(
            Finding(
                severity=severity,
                check="missing-tests",
                path=str(rel_path),
                line=None,
                message=f"No direct test file or test reference was found for '{rel_path}'.",
                hint=(
                    f"Add tests for {rel_path.stem} or extend an existing suite that exercises this file's public behavior."
                ),
            )
        )
    return findings


def build_coverage_commands(excluded_dirs: Iterable[str]) -> tuple[list[str], list[str]]:
    omitted = sorted({f"*/{name}/*" for name in set(excluded_dirs) | {'tests'}})
    omit_flag = "--omit=" + ",".join(omitted)
    run_command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--branch",
        "--source=.",
        omit_flag,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
    ]
    json_command = [sys.executable, "-m", "coverage", "json", "-o"]
    return run_command, json_command


def run_coverage_scan(
    root: Path,
    *,
    excluded_dirs: set[str],
    source_files: Sequence[Path],
    threshold: float,
) -> CoverageSummary:
    if importlib.util.find_spec("coverage") is None:
        return CoverageSummary(ran=False, error="coverage is not installed")
    if not (root / "tests").exists():
        return CoverageSummary(ran=False, error="tests directory was not found")

    run_command, json_command = build_coverage_commands(excluded_dirs)
    with tempfile.TemporaryDirectory(prefix="repo-quality-audit-") as temp_dir:
        coverage_file = Path(temp_dir) / ".coverage"
        json_path = Path(temp_dir) / "coverage.json"
        env = os.environ.copy()
        env["COVERAGE_FILE"] = str(coverage_file)

        run_result = subprocess.run(
            run_command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if run_result.returncode != 0:
            stderr = (run_result.stderr or run_result.stdout).strip()
            return CoverageSummary(
                ran=True,
                error=f"coverage run failed: {stderr[-400:]}",
                command=run_command,
            )

        json_result = subprocess.run(
            json_command + [str(json_path)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if json_result.returncode != 0:
            stderr = (json_result.stderr or json_result.stdout).strip()
            return CoverageSummary(
                ran=True,
                error=f"coverage json export failed: {stderr[-400:]}",
                command=json_command + [str(json_path)],
            )

        payload = json.loads(json_path.read_text(encoding="utf-8"))

    coverage_files = payload.get("files", {})
    below_threshold: list[dict[str, object]] = []
    for source_path in source_files:
        rel_path = str(source_path.relative_to(root))
        entry = coverage_files.get(rel_path)
        if not entry:
            continue
        summary = entry.get("summary", {})
        num_statements = int(summary.get("num_statements", 0))
        if num_statements == 0:
            continue
        percent_covered = float(summary.get("percent_covered", 0.0))
        if percent_covered < threshold:
            below_threshold.append(
                {
                    "path": rel_path,
                    "percent_covered": round(percent_covered, 2),
                    "num_statements": num_statements,
                }
            )

    totals = payload.get("totals", {})
    total_percent = totals.get("percent_covered")
    return CoverageSummary(
        ran=True,
        total_percent=float(total_percent) if isinstance(total_percent, (int, float)) else None,
        files_below_threshold=sorted(below_threshold, key=lambda item: (item["percent_covered"], item["path"])),
        command=run_command,
    )


def add_coverage_findings(
    findings: list[Finding],
    coverage: CoverageSummary,
    *,
    threshold: float,
) -> None:
    if coverage.error:
        findings.append(
            Finding(
                severity="HIGH",
                check="coverage-run-failed",
                path="tests",
                line=None,
                message=coverage.error,
                hint="Fix the failing test or coverage invocation before trusting coverage-gap output.",
            )
        )
        return

    for item in coverage.files_below_threshold:
        path = Path(str(item["path"]))
        severity = "LOW" if source_kind(path) in {"script", "page"} else "MEDIUM"
        findings.append(
            Finding(
                severity=severity,
                check="low-coverage",
                path=str(item["path"]),
                line=None,
                message=(
                    f"Coverage is {item['percent_covered']:.2f}% for '{item['path']}', below the {threshold:.0f}% threshold."
                ),
                hint="Add focused tests for the uncovered behavior or reduce dead branches before raising the threshold.",
                details={
                    "percent_covered": item["percent_covered"],
                    "num_statements": item["num_statements"],
                },
            )
        )


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER.get(finding.severity, 99),
            finding.path,
            finding.line or 0,
            finding.check,
        ),
    )


def run_audit(
    root: Path,
    *,
    coverage_mode: str,
    coverage_threshold: float,
    include_tests: bool,
    excluded_dirs: set[str],
    locale_markers: Sequence[str],
    translation_functions: Sequence[str],
) -> AuditReport:
    source_files = walk_files(root, PYTHON_SUFFIXES, excluded_dirs=excluded_dirs, include_tests=False)
    findings: list[Finding] = []
    findings.extend(scan_root_layout(root))
    findings.extend(
        scan_translation_issues(
            root,
            excluded_dirs=excluded_dirs,
            locale_markers=locale_markers,
            translation_functions=translation_functions,
        )
    )
    findings.extend(scan_python_quality(root, excluded_dirs=excluded_dirs, include_tests=include_tests))
    findings.extend(scan_duplicate_functions(root, excluded_dirs=excluded_dirs, include_tests=include_tests))
    findings.extend(scan_missing_tests(root, excluded_dirs=excluded_dirs))

    should_run_coverage = coverage_mode == "run" or (
        coverage_mode == "auto" and importlib.util.find_spec("coverage") is not None
    )
    coverage = CoverageSummary(ran=False)
    if should_run_coverage:
        coverage = run_coverage_scan(
            root,
            excluded_dirs=excluded_dirs,
            source_files=source_files,
            threshold=coverage_threshold,
        )
        add_coverage_findings(findings, coverage, threshold=coverage_threshold)

    return AuditReport(root=str(root), findings=sort_findings(findings), coverage=coverage)


def format_finding(finding: Finding) -> str:
    location = f"{finding.path}:{finding.line}" if finding.line else finding.path
    return (
        f"{finding.severity:<6} {finding.check:<24} {location}\n"
        f"  {finding.message}\n"
        f"  hint: {finding.hint}"
    )


def render_report(report: AuditReport) -> str:
    lines = [f"Audit root: {report.root}"]
    if report.coverage.ran and report.coverage.total_percent is not None:
        lines.append(f"Coverage total: {report.coverage.total_percent:.2f}%")
    elif report.coverage.ran and report.coverage.error:
        lines.append(f"Coverage: {report.coverage.error}")
    else:
        lines.append("Coverage: skipped")

    if not report.findings:
        lines.append("No findings.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Findings:")
    for finding in report.findings:
        lines.append(format_finding(finding))
        lines.append("")

    summary = report.to_dict()["summary"]
    lines.append(
        "Summary: "
        + " | ".join(f"{severity} {summary[severity]}" for severity in ("HIGH", "MEDIUM", "LOW"))
    )
    return "\n".join(lines).rstrip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a repository for translation drift, duplication, tests, and coverage gaps.")
    parser.add_argument("--root", default=".", help="Repository root to audit.")
    parser.add_argument(
        "--coverage-mode",
        choices=("auto", "run", "skip"),
        default="auto",
        help="Whether to run coverage.py as part of the audit.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Coverage percentage threshold for low-coverage findings.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in Python-quality and duplication scans.",
    )
    parser.add_argument(
        "--locale-markers",
        default=",".join(DEFAULT_LOCALE_MARKERS),
        help="Comma-separated directory markers used to discover locale folders.",
    )
    parser.add_argument(
        "--translation-functions",
        default=",".join(DEFAULT_TRANSLATION_FUNCTIONS),
        help="Comma-separated translation helper names to scan for literal locale keys.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of text.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    report = run_audit(
        root,
        coverage_mode=args.coverage_mode,
        coverage_threshold=args.coverage_threshold,
        include_tests=args.include_tests,
        excluded_dirs=set(DEFAULT_EXCLUDED_DIRS),
        locale_markers=[item.strip() for item in args.locale_markers.split(",") if item.strip()],
        translation_functions=[item.strip() for item in args.translation_functions.split(",") if item.strip()],
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
