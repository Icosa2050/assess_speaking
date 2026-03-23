from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ENTRYPOINT_MODULES = frozenset({"assess_speaking", "streamlit_app"})
MODULE_GROUPS: dict[str, frozenset[str]] = {}


@dataclass(frozen=True)
class ModuleAudit:
    name: str
    path: str
    group: str
    direct_imports: tuple[str, ...]
    inbound_imports: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "group": self.group,
            "direct_imports": list(self.direct_imports),
            "inbound_imports": list(self.inbound_imports),
        }


def discover_root_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".py")


def extract_local_imports(path: Path, local_module_names: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level in local_module_names:
                    imports.add(top_level)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level = node.module.split(".")[0]
            if top_level in local_module_names:
                imports.add(top_level)
    return imports


def classify_module(name: str) -> str:
    if name in ENTRYPOINT_MODULES:
        return "entrypoint"
    for group, members in MODULE_GROUPS.items():
        if name in members:
            return group
    return "unclassified"


def build_root_import_audit(root: Path) -> dict[str, object]:
    python_files = discover_root_python_files(root)
    module_names = {path.stem for path in python_files}
    direct_imports = {name: set() for name in module_names}
    inbound_imports = {name: set() for name in module_names}

    for path in python_files:
        imports = extract_local_imports(path, module_names)
        direct_imports[path.stem] = imports
        for imported in imports:
            inbound_imports[imported].add(path.stem)

    modules = [
        ModuleAudit(
            name=path.stem,
            path=str(path.relative_to(root)),
            group=classify_module(path.stem),
            direct_imports=tuple(sorted(direct_imports[path.stem])),
            inbound_imports=tuple(sorted(inbound_imports[path.stem])),
        )
        for path in python_files
    ]
    inbound_hubs = [
        {
            "name": module.name,
            "group": module.group,
            "inbound_count": len(module.inbound_imports),
            "inbound_imports": list(module.inbound_imports),
        }
        for module in sorted(
            modules,
            key=lambda item: (-len(item.inbound_imports), item.name),
        )
        if module.inbound_imports
    ]
    high_outbound = [
        {
            "name": module.name,
            "group": module.group,
            "direct_import_count": len(module.direct_imports),
            "direct_imports": list(module.direct_imports),
        }
        for module in sorted(
            modules,
            key=lambda item: (-len(item.direct_imports), item.name),
        )
        if module.direct_imports
    ]
    return {
        "root": str(root),
        "root_python_file_count": len(python_files),
        "entrypoints": sorted(f"{name}.py" for name in ENTRYPOINT_MODULES if f"{name}.py" in {path.name for path in python_files}),
        "groups": {
            group: sorted(f"{name}.py" for name in members if f"{name}.py" in {path.name for path in python_files})
            for group, members in MODULE_GROUPS.items()
        },
        "modules": [module.to_dict() for module in modules],
        "inbound_hubs": inbound_hubs,
        "high_outbound_modules": high_outbound,
    }


def render_text_report(audit: dict[str, object]) -> str:
    lines = [
        f"Root: {audit['root']}",
        f"Root Python files: {audit['root_python_file_count']}",
        "",
        "Entrypoints:",
    ]
    for entrypoint in audit["entrypoints"]:
        lines.append(f"- {entrypoint}")

    lines.append("")
    lines.append("Groups:")
    groups: dict[str, list[str]] = audit["groups"]  # type: ignore[assignment]
    if not groups:
        lines.append("- none")
    else:
        for group in sorted(groups):
            members = groups[group]
            suffix = f" ({len(members)})" if members else " (0)"
            lines.append(f"- {group}{suffix}: {', '.join(members) if members else '-'}")

    lines.append("")
    lines.append("Highest inbound dependency hubs:")
    inbound_hubs: list[dict[str, object]] = audit["inbound_hubs"]  # type: ignore[assignment]
    for item in inbound_hubs[:8]:
        imports = ", ".join(item["inbound_imports"]) or "-"
        lines.append(f"- {item['name']} [{item['group']}] <- {item['inbound_count']}: {imports}")

    lines.append("")
    lines.append("Highest outbound dependency modules:")
    high_outbound: list[dict[str, object]] = audit["high_outbound_modules"]  # type: ignore[assignment]
    for item in high_outbound[:8]:
        imports = ", ".join(item["direct_imports"]) or "-"
        lines.append(f"- {item['name']} [{item['group']}] -> {item['direct_import_count']}: {imports}")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit root-level Python imports and cleanup groupings.")
    parser.add_argument("--root", default=".", help="Repository root to audit.")
    parser.add_argument("--json", action="store_true", help="Print the audit as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    audit = build_root_import_audit(root)
    if args.json:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
    else:
        print(render_text_report(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
