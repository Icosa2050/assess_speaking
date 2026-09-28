from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    "app_backend",
    "app_core",
    "scripts/run_app.py",
    "scripts/run_backend.py",
)
SUBPROCESS_BAN_ROOTS = ("app_core",)
ALLOW_PREFIX = "quality: allow["


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    line: int
    message: str


def _iter_python_files(root: Path, targets: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        candidate = (root / target).resolve()
        if candidate.is_dir():
            files.extend(
                path
                for path in sorted(candidate.rglob("*.py"))
                if "__pycache__" not in path.parts
            )
            continue
        if candidate.is_file():
            files.append(candidate)
    # Preserve order but avoid duplicates when a file sits under an included dir.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _has_allow_comment(lines: list[str], lineno: int, code: str) -> bool:
    token = f"{ALLOW_PREFIX}{code}]"
    line_indexes = [lineno - 1]
    if lineno - 2 >= 0:
        line_indexes.append(lineno - 2)
    for index in line_indexes:
        if token in lines[index]:
            return True
    return False


def _is_broad_exception(handler: ast.ExceptHandler) -> bool:
    exception_type = handler.type
    if exception_type is None:
        return False
    if isinstance(exception_type, ast.Name):
        return exception_type.id in {"Exception", "BaseException"}
    if isinstance(exception_type, ast.Tuple):
        return any(
            isinstance(item, ast.Name) and item.id in {"Exception", "BaseException"}
            for item in exception_type.elts
        )
    return False


def _call_matches_subprocess(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
        and func.attr in {"run", "Popen", "check_call", "check_output"}
    )


class QualityVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source_lines: list[str], root: Path) -> None:
        self._path = path
        self._source_lines = source_lines
        self._root = root
        self.findings: list[Finding] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        rel_path = self._path.relative_to(self._root).as_posix()
        if node.type is None and not _has_allow_comment(self._source_lines, node.lineno, "bare-except"):
            self.findings.append(
                Finding(
                    check="bare-except",
                    path=rel_path,
                    line=node.lineno,
                    message="Bare except hides unexpected failures. Catch a specific exception or document the boundary.",
                )
            )
        elif _is_broad_exception(node) and not _has_allow_comment(self._source_lines, node.lineno, "broad-except"):
            self.findings.append(
                Finding(
                    check="broad-except",
                    path=rel_path,
                    line=node.lineno,
                    message="Broad exception handling needs a narrower exception type or an explicit quality allow comment.",
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        rel_path = self._path.relative_to(self._root).as_posix()
        if (
            _call_matches_subprocess(node)
            and rel_path.startswith(SUBPROCESS_BAN_ROOTS)
            and not _has_allow_comment(self._source_lines, node.lineno, "subprocess-shellout")
        ):
            self.findings.append(
                Finding(
                    check="subprocess-shellout",
                    path=rel_path,
                    line=node.lineno,
                    message="Core runtime code should call the managed backend, not spawn subprocesses directly.",
                )
            )
        self.generic_visit(node)


def scan_quality(root: Path = ROOT, *, targets: Iterable[str] = DEFAULT_TARGETS) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in _iter_python_files(root, targets):
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        visitor = QualityVisitor(path, source_lines, root)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return sorted(findings, key=lambda item: (item.path, item.line, item.check))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check backend rollout quality rules.")
    parser.add_argument("--json", action="store_true", help="Emit findings as JSON.")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Optional repo-relative files or directories to scan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    findings = scan_quality(ROOT, targets=args.paths)
    if args.json:
        sys.stdout.write(json.dumps([asdict(finding) for finding in findings], indent=2))
        sys.stdout.write("\n")
    else:
        if findings:
            for finding in findings:
                sys.stdout.write(f"{finding.path}:{finding.line}: {finding.check}: {finding.message}\n")
        else:
            sys.stdout.write("Quality checks passed.\n")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
