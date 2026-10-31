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
            content = read_python_license_file(dist, filename)
            files.append((filename, content.strip()))
        result.append(_section("Python", name, version, license_name, files))
    return result


def read_python_license_file(dist: metadata.Distribution, filename: str) -> str:
    """Wheels may place declared license files at either dist-info root or licenses/."""
    for candidate in (filename, f"licenses/{filename}"):
        try:
            content = dist.read_text(candidate)
        except (FileNotFoundError, ValueError):
            content = None
        if content:
            return content
    raise RuntimeError(f"Declared Python license file missing: {dist.metadata['Name']} {filename}")


def ffmpeg_notice(helper: Path, *, source_version: str = '', source_archive: str = '',
                  require_source_reference: bool = False, allow_gpl_distribution: bool = False) -> str:
    """Record build-interpreter PyAV metadata and frozen-helper libraries."""
    version_output = subprocess.check_output([sys.executable, '-m', 'av', '--version'], text=True).strip()
    pyav = re.search(r'(?m)^PyAV v([^\s]+)$', version_output)
    configuration = re.search(r'(?m)^library configuration: (.+)$', version_output)
    licenses = re.findall(r'(?m)^library license: (.+)$', version_output)
    if not pyav or not configuration or not licenses:
        raise RuntimeError('PyAV did not report its version, FFmpeg configuration and license')
    libraries = sorted(path.relative_to(helper).as_posix() for path in helper.rglob('*.dylib') if path.is_file())
    if not libraries or not any(Path(path).name.startswith('libav') for path in libraries):
        raise RuntimeError('Frozen helper has no FFmpeg shared libraries to inventory')
    section = ('FFMPEG / PYAV IN THE FROZEN HELPER\n'
               + version_output + '\n'
               + f'Frozen helper shared libraries ({len(libraries)}; includes FFmpeg, PyAV and codec dependencies):\n'
               + '\n'.join(libraries))
    normalized = [license_name.upper() for license_name in licenses]
    if any(not name.startswith(('GPL', 'LGPL', 'NONFREE')) for name in normalized):
        raise RuntimeError(f'Unsupported FFmpeg license report: {licenses}')
    configure_flags = configuration.group(1)
    restricted = (any(not name.startswith('LGPL') for name in normalized)
                  or '--enable-nonfree' in configure_flags
                  or any(flag in configure_flags for flag in ('--enable-libx264', '--enable-libx265'))
                  or any(Path(path).name.startswith(('libx264', 'libx265')) for path in libraries))
    if require_source_reference and restricted and not allow_gpl_distribution:
        raise RuntimeError('GPL/nonfree FFmpeg or x264/x265 is not allowed in a release build; '
                           'replace it with an LGPL-only build or use --allow-gpl-ffmpeg-distribution after owner review')
    if require_source_reference and (not source_version or not source_archive):
        raise RuntimeError('FFmpeg release requires its source version and archive reference')
    section += ('\n\nFFmpeg/PyAV corresponding-source reference (complete before distribution):\n'
                + f'PyAV version: {pyav.group(1)}\n'
                + f'FFmpeg source version: {source_version or "SOURCE_VERSION_REQUIRED"}\n'
                + f'FFmpeg source archive path or URL: {source_archive or "SOURCE_ARCHIVE_REQUIRED"}\n'
                + f'FFmpeg configure flags: {configure_flags}')
    if restricted:
        section += ('\n\nPUBLIC DISTRIBUTION HOLD: This frozen build contains GPL/nonfree FFmpeg or x264/x265. '
                    'Owner review of compatibility and complete corresponding source is required.')
    license_files = ['GPL-3.0.txt'] if any(name.startswith('GPL') for name in normalized) else []
    lgpl_licenses = [name for name in licenses if name.upper().startswith('LGPL')]
    if lgpl_licenses:
        versions = [re.search(r'\b(2\.1|3(?:\.0)?)\b', license_name) for license_name in lgpl_licenses]
        if any(version is None for version in versions):
            raise RuntimeError(f'Unsupported FFmpeg LGPL license report: {licenses}')
        license_files.extend(['GPL-3.0.txt', 'LGPL-3.0.txt'] if any(version.group(1) != '2.1' for version in versions)
                             else ['LGPL-2.1.txt'])
    for filename in dict.fromkeys(license_files):
        section += f'\n\n--- {filename} ---\n' + (ROOT / 'packaging/licenses' / filename).read_text().strip()
    return section


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
    pending = [("", name) for name in packages[""]["dependencies"]]
    seen = set()
    result = []
    while pending:
        parent, name = pending.pop()
        key = resolve_npm_dependency(packages, parent, name)
        if key in seen:
            continue
        seen.add(key)
        package = packages[key]
        installed = node_modules / key.removeprefix("node_modules/")
        manifest = json.loads((installed / "package.json").read_text())
        if manifest.get("version") != package["version"]:
            raise RuntimeError(f"npm lock mismatch: {name}")
        pending.extend((key, dependency) for dependency in package.get("dependencies", {}))
        result.append(_section("npm runtime", name, package["version"], str(manifest.get("license", "")), _license_files(installed)))
    return sorted(result)


def resolve_npm_dependency(packages: dict, parent: str, name: str) -> str:
    """Follow Node's installed-package ancestor lookup recorded in package-lock."""
    current = parent
    while True:
        candidate = f"{current}/node_modules/{name}" if current else f"node_modules/{name}"
        if candidate in packages:
            return candidate
        if not current:
            break
        current = current.rsplit("/node_modules/", 1)[0] if "/node_modules/" in current else ""
    raise RuntimeError(f"npm runtime package missing from lock: {name} (required by {parent or 'root'})")


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


def generate(lock: Path, npm_lock: Path, node_modules: Path, cargo_manifest: Path, helper: Path,
             *, ffmpeg_source_version: str = '', ffmpeg_source_archive: str = '',
             require_ffmpeg_source_reference: bool = False,
             allow_gpl_ffmpeg_distribution: bool = False) -> str:
    sections = [
        "VOSTAVO THIRD-PARTY NOTICES\nGenerated from the pinned macOS production Python lock, npm runtime graph, and Cargo.lock resolution.\n"
        "The Rust graph includes crates for other target platforms as a conservative superset.\n"
        "Whisper model weights are downloaded separately into the user's cache and are not in Vostavo.app.\n"
        "Check the model repository license before redistribution.\n"
        "The application's own code has no repository LICENSE; its license is an owner decision.\n",
        cpython_notice(),
        ffmpeg_notice(helper, source_version=ffmpeg_source_version, source_archive=ffmpeg_source_archive,
                      require_source_reference=require_ffmpeg_source_reference,
                      allow_gpl_distribution=allow_gpl_ffmpeg_distribution),
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
    parser.add_argument("--ffmpeg-source-version", default='')
    parser.add_argument("--ffmpeg-source-archive", default='')
    parser.add_argument("--require-ffmpeg-source-reference", action='store_true')
    parser.add_argument("--allow-gpl-ffmpeg-distribution", action='store_true')
    args = parser.parse_args()
    result = generate(ROOT / "packaging/requirements-macos-arm64.txt", ROOT / "frontend/package-lock.json",
                      ROOT / "frontend/node_modules", ROOT / "frontend/src-tauri/Cargo.toml", args.helper,
                      ffmpeg_source_version=args.ffmpeg_source_version,
                      ffmpeg_source_archive=args.ffmpeg_source_archive,
                      require_ffmpeg_source_reference=args.require_ffmpeg_source_reference,
                      allow_gpl_ffmpeg_distribution=args.allow_gpl_ffmpeg_distribution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)


if __name__ == "__main__":
    main()
