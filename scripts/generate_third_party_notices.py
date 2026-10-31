#!/usr/bin/env python3
"""Generate deterministic notices from the locked macOS production inputs.

Run with the production Python interpreter after npm ci and cargo metadata are
available. The resulting file is embedded in both the webview and app Resources.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _license_files(directory: Path) -> list[tuple[str, str]]:
    if not directory.is_dir():
        return []
    result = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and re.match(r"(?i)^(license|licence|copying|notice)([._-]|$)", path.name):
            result.append((path.name, path.read_text(encoding="utf-8", errors="replace").strip()))
    return result


def _section(kind: str, name: str, version: str, license_name: str, texts: list[tuple[str, str]]) -> str:
    heading = f"{kind}: {name} {version}\nLicense: {license_name or 'UNDECLARED'}"
    return heading + "\n" + "\n".join(f"\n--- {filename} ---\n{content}\n" for filename, content in texts)


def python_notices(lock: Path) -> list[str]:
    pins = re.findall(r"(?m)^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+~-]+)\s*\\", lock.read_text())
    result = []
    for name, version in sorted(pins, key=lambda pair: pair[0].lower()):
        dist = metadata.distribution(name)
        if dist.version != version:
            raise RuntimeError(f"Python lock mismatch: {name} {version} != {dist.version}")
        license_name = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ", ".join(
            value.removeprefix("License :: ") for value in dist.metadata.get_all("Classifier", []) if value.startswith("License :: ")
        )
        files = []
        for filename in sorted(set(dist.metadata.get_all("License-File", []) or [])):
            try:
                content = dist.read_text(filename)
            except (FileNotFoundError, ValueError):
                content = None
            if content:
                files.append((filename, content.strip()))
        result.append(_section("Python", name, version, license_name, files))
    return result


def ffmpeg_notice() -> str:
    import av
    import av._core
    rows = sorted((name, value) for name, value in av._core.library_meta.items() if value["version"][0] >= 0)
    if not rows:
        raise RuntimeError("PyAV has no FFmpeg library metadata")
    lines = []
    for name, value in rows:
        license_name = value["license"]
        if not license_name or not any(word in license_name.upper() for word in ("GPL", "LGPL")):
            raise RuntimeError(f"FFmpeg license unknown for {name}: {license_name!r}")
        lines.append(f"{name} {'.'.join(map(str, value['version']))}: {license_name}\n  configuration: {value['configuration']}")
    libraries = sorted(path.name for path in (Path(av.__file__).parent / '.dylibs').glob('*.dylib'))
    if not libraries:
        raise RuntimeError('PyAV wheel has no bundled shared libraries to inventory')
    return ("FFmpeg libraries dynamically linked by PyAV " + av.__version__ + "\n" + "\n".join(lines)
            + "\nBundled PyAV shared libraries (including codec dependencies):\n" + "\n".join(libraries))


def cpython_notice() -> str:
    license_path = Path(sys.base_prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'LICENSE.txt'
    if not license_path.is_file():
        raise RuntimeError(f'Bundled CPython license missing: {license_path}')
    return _section('Runtime', 'CPython', '.'.join(map(str, sys.version_info[:3])),
                    'Python Software Foundation License', [('LICENSE.txt', license_path.read_text())])


def frozen_inventory(helper: Path) -> str:
    resources = helper / 'Contents/Resources'
    if not resources.is_dir():
        raise RuntimeError(f'Frozen helper resources missing: {resources}')
    metadata_dirs = sorted(path.relative_to(resources).as_posix() for path in resources.rglob('*.dist-info') if path.is_dir())
    native = sorted(path.relative_to(helper).as_posix() for path in helper.rglob('*')
                    if path.is_file() and (path.suffix in {'.dylib', '.so'} or path.name == 'vostavo-backend'))
    if not native or not metadata_dirs:
        raise RuntimeError('Frozen helper inventory is incomplete')
    return ('FROZEN PYINSTALLER HELPER INVENTORY\n'
            + f'Metadata directories ({len(metadata_dirs)}):\n' + '\n'.join(metadata_dirs)
            + f'\nNative files ({len(native)}):\n' + '\n'.join(native))


def npm_notices(lock: Path, node_modules: Path) -> list[str]:
    packages = json.loads(lock.read_text())["packages"]
    pending = list(packages[""]["dependencies"])
    seen = set()
    result = []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        key = "node_modules/" + name
        package = packages.get(key)
        if package is None:
            raise RuntimeError(f"npm runtime package missing from lock: {name}")
        installed = node_modules / name
        manifest = json.loads((installed / "package.json").read_text())
        if manifest.get("version") != package["version"]:
            raise RuntimeError(f"npm lock mismatch: {name}")
        pending.extend(package.get("dependencies", {}))
        result.append(_section("npm runtime", name, package["version"], str(manifest.get("license", "")), _license_files(installed)))
    return sorted(result)


def rust_notices(manifest: Path) -> list[str]:
    cargo = shutil.which('cargo') or str(Path.home() / '.cargo/bin/cargo')
    data = json.loads(subprocess.check_output([
        cargo, "metadata", "--locked", "--format-version", "1", "--manifest-path", str(manifest)
    ], text=True))
    # Cargo metadata resolves the locked cross-platform dependency graph. Include
    # every registry crate, even when target selection removes it from this DMG.
    result = []
    for package in data["packages"]:
        if not (package.get("source") or "").startswith("registry+"):
            continue
        directory = Path(package["manifest_path"]).parent
        result.append(_section("Rust crate", package["name"], package["version"], package.get("license") or "", _license_files(directory)))
    return sorted(result)


def generate(lock: Path, npm_lock: Path, node_modules: Path, cargo_manifest: Path, helper: Path) -> str:
    sections = [
        "VOSTAVO THIRD-PARTY NOTICES\nGenerated from the pinned macOS production Python lock, npm runtime graph, and Cargo.lock resolution.\n"
        "The Rust graph includes crates for other target platforms as a conservative superset.\n"
        "Whisper model weights are downloaded separately into the user's cache and are not in Vostavo.app.\n"
        "Check the model repository license before redistribution.\n"
        "The application's own code has no repository LICENSE; its license is an owner decision.\n",
        cpython_notice(),
        ffmpeg_notice(),
        frozen_inventory(helper),
        *python_notices(lock),
        *npm_notices(npm_lock, node_modules),
        *rust_notices(cargo_manifest),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    args = parser.parse_args()
    result = generate(ROOT / "packaging/requirements-macos-arm64.txt", ROOT / "frontend/package-lock.json",
                      ROOT / "frontend/node_modules", ROOT / "frontend/src-tauri/Cargo.toml", args.helper)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)


if __name__ == "__main__":
    main()
