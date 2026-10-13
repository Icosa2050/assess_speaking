#!/usr/bin/env python3
"""Build an internal DMG or a Developer ID notarized release from locked inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / '.build-macos'
PYTHON_VERSION = '3.12.11'


def run(args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def output(args):
    return run(args, capture_output=True, text=True).stdout.strip()


def macho_files(bundle):
    seen = set()
    magic = {b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'}
    for path in bundle.rglob('*'):
        if not path.is_file() or path.resolve() in seen:
            continue
        with path.open('rb') as stream:
            if stream.read(4) not in magic:
                continue
        seen.add(path.resolve())
        yield path.resolve()


def audit(bundle):
    entries = []
    minimum = (13, 4)
    for path in macho_files(bundle):
        architectures = output(['lipo', '-archs', path]).split()
        if architectures != ['arm64']:
            raise RuntimeError(f'Non-arm64 code: {path}: {architectures}')
        links = output(['otool', '-L', path])
        for line in links.splitlines()[1:]:
            library = line.strip().split(' (')[0]
            if not library.startswith(('@rpath/', '@loader_path/', '@executable_path/', '/usr/lib/', '/System/')):
                raise RuntimeError(f'External library dependency: {path}: {library}')
        targets = output(['xcrun', 'vtool', '-show-build', path])
        versions = re.findall(r'\bminos\s+(\d+\.\d+(?:\.\d+)?)', targets)
        loads = output(['otool', '-l', path])
        for block in loads.split('Load command ')[1:]:
            if 'cmd LC_RPATH\n' in block:
                match = re.search(r'\bpath (.+) \(offset', block)
                if match and not match.group(1).startswith(('@loader_path', '@executable_path', '/usr/lib', '/System/')):
                    raise RuntimeError(f'External runtime search path: {path}: {match.group(1)}')
            if 'cmd LC_VERSION_MIN_MACOSX\n' in block:
                versions.extend(re.findall(r'\bversion\s+(\d+\.\d+(?:\.\d+)?)', block))
        for version in versions:
            minimum = max(minimum, tuple(map(int, version.split('.'))))
        entries.append({'path': str(path.relative_to(bundle.resolve())), 'architectures': architectures, 'linkage': links, 'build_target': targets})
    if not entries:
        raise RuntimeError('No native executables found')
    return {'minimum_macos': '.'.join(map(str, minimum)), 'native_files': entries}


def sign(bundle, identity, release):
    options = ['--force', '--sign', identity, '--options', 'runtime', '--timestamp' if release else '--timestamp=none']
    for path in sorted(macho_files(bundle), key=lambda value: len(value.parts), reverse=True):
        run(['codesign', *options, path])
    for framework in sorted(bundle.rglob('*.framework'), key=lambda value: len(value.parts), reverse=True):
        if not framework.is_symlink():
            run(['codesign', *options, framework])
    helper = bundle / 'Contents/Helpers/VostavoBackend.app'
    # Ad-hoc identities have no Team ID, so library validation cannot match
    # their embedded Python libraries. Developer ID releases keep validation.
    helper_entitlements = [] if release else ['--entitlements', ROOT / 'packaging/internal-helper-entitlements.plist']
    run(['codesign', *options, *helper_entitlements, helper])
    run(['codesign', *options, '--entitlements', ROOT / 'packaging/entitlements.plist', bundle])
    run(['codesign', '--verify', '--deep', '--strict', '--verbose=2', bundle])


def notarize(artifact, profile):
    result = json.loads(output(['xcrun', 'notarytool', 'submit', artifact, '--keychain-profile', profile, '--wait', '--output-format', 'json']))
    if result.get('status') != 'Accepted':
        raise RuntimeError(f'Notarization was not accepted: {result.get("id")} {result.get("status")}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['internal', 'release'], required=True)
    parser.add_argument('--identity', default='')
    parser.add_argument('--notary-profile', default='')
    parser.add_argument('--python', default=PYTHON_VERSION)
    parser.add_argument('--reuse-helper', action='store_true', help='Local iteration only; forbidden for releases.')
    args = parser.parse_args(argv)
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        parser.error('This build requires an Apple Silicon Mac.')
    release = args.mode == 'release'
    if release:
        if args.reuse_helper or not args.identity.startswith('Developer ID Application:') or not args.notary_profile:
            parser.error('Release requires a Developer ID Application identity and Keychain notary profile; no reused helper.')
        if args.identity not in output(['security', 'find-identity', '-v', '-p', 'codesigning']):
            parser.error('The requested Developer ID identity is not available.')
    config = json.loads((ROOT / 'frontend/src-tauri/tauri.conf.json').read_text())
    version = config['version']
    if tomllib.loads((ROOT / 'frontend/src-tauri/Cargo.toml').read_text())['package']['version'] != version:
        raise RuntimeError('Rust and Tauri versions differ')
    BUILD.mkdir(exist_ok=True)
    python = BUILD / 'venv/bin/python'
    if not python.exists():
        run(['uv', 'venv', '--python', args.python, BUILD / 'venv'])
    run(['uv', 'pip', 'sync', '--python', python, '--require-hashes', ROOT / 'packaging/requirements-macos-arm64.txt'])
    if output([python, '-c', 'import platform; print(platform.python_version(), platform.machine())']) != PYTHON_VERSION + ' arm64':
        raise RuntimeError('Packaging requires pinned CPython 3.12.11 arm64')
    npm_env = {key: os.environ[key] for key in ('PATH', 'HOME', 'TMPDIR', 'LANG') if key in os.environ}
    run(['npm', '--prefix', ROOT / 'frontend', 'ci', '--ignore-scripts'], env=npm_env)
    run(['npm', '--prefix', ROOT / 'frontend', 'run', 'build'])
    helper = BUILD / 'dist/VostavoBackend.app'
    if not args.reuse_helper:
        run([python, '-m', 'PyInstaller', '--clean', '--noconfirm', '--distpath', BUILD / 'dist', '--workpath', BUILD / 'pyinstaller', ROOT / 'packaging/vostavo-backend.spec'], cwd=ROOT)
    if not helper.is_dir():
        raise RuntimeError('Frozen helper app was not produced')
    if output([python, '-c', 'import platform; print(platform.python_version(), platform.machine())']) != PYTHON_VERSION + ' arm64':
        raise RuntimeError('Packaging requires pinned CPython 3.12.11 arm64')
    if plistlib.loads((helper / 'Contents/Info.plist').read_bytes())['CFBundleShortVersionString'] != version:
        raise RuntimeError('Helper version differs')
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET='13.4')
    cargo = shutil.which('cargo') or str(Path.home() / '.cargo/bin/cargo')
    run([cargo, 'build', '--release', '--locked', '--features', 'custom-protocol', '--manifest-path', ROOT / 'frontend/src-tauri/Cargo.toml'], env=env)
    stage = BUILD / 'stage'
    if stage.exists():
        shutil.rmtree(stage)
    bundle = stage / 'Vostavo.app'
    macos = bundle / 'Contents/MacOS'
    resources = bundle / 'Contents/Resources'
    macos.mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(ROOT / 'frontend/src-tauri/target/release/vostavo-desktop', macos / 'vostavo-desktop')
    run(['ditto', helper, bundle / 'Contents/Helpers/VostavoBackend.app'])
    iconset = BUILD / 'Vostavo.iconset'
    iconset.mkdir(exist_ok=True)
    icon = ROOT / 'frontend/src-tauri/icons/icon.png'
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            run(['sips', '-z', str(size * scale), str(size * scale), icon, '--out', iconset / f'icon_{size}x{size}{"@2x" if scale == 2 else ""}.png'], stdout=subprocess.DEVNULL)
    run(['iconutil', '-c', 'icns', iconset, '-o', resources / 'Vostavo.icns'])
    microphone = {'en': 'Vostavo uses your microphone to record oral practice and help you review your progress.', 'it': 'Vostavo usa il microfono per registrare le esercitazioni orali e aiutarti a seguire i progressi.', 'de': 'Vostavo nutzt dein Mikrofon für mündliche Übungen und deinen Fortschritt.', 'fr': 'Vostavo utilise votre microphone pour enregistrer vos exercices oraux et suivre vos progrès.', 'es': 'Vostavo usa el micrófono para grabar prácticas orales y seguir tu progreso.'}
    for language, copy in microphone.items():
        folder = resources / f'{language}.lproj'
        folder.mkdir()
        (folder / 'InfoPlist.strings').write_text(f'"NSMicrophoneUsageDescription" = {json.dumps(copy, ensure_ascii=False)};\n')
    evidence = audit(bundle)
    info = {'CFBundleExecutable': 'vostavo-desktop', 'CFBundleIdentifier': config['identifier'], 'CFBundleName': 'Vostavo', 'CFBundleDisplayName': 'Vostavo', 'CFBundlePackageType': 'APPL', 'CFBundleShortVersionString': version, 'CFBundleVersion': version, 'CFBundleIconFile': 'Vostavo.icns', 'LSMinimumSystemVersion': evidence['minimum_macos'], 'NSMicrophoneUsageDescription': microphone['en'], 'NSAppleEventsUsageDescription': 'Vostavo can prepare a support email in Mail with your selected package attached. You review and send the draft.', 'CFBundleLocalizations': list(microphone), 'NSHighResolutionCapable': True}
    (bundle / 'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    # Keep nested bundle's declared minimum consistent with its actual native code.
    helper_info = bundle / 'Contents/Helpers/VostavoBackend.app/Contents/Info.plist'
    helper_plist = plistlib.loads(helper_info.read_bytes())
    helper_plist['LSMinimumSystemVersion'] = evidence['minimum_macos']
    helper_info.write_bytes(plistlib.dumps(helper_plist))
    sign(bundle, args.identity if release else '-', release)
    smoke_env = {key: os.environ[key] for key in ('HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG') if key in os.environ}
    smoke_env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    smoke = run([bundle / 'Contents/Helpers/VostavoBackend.app/Contents/MacOS/vostavo-backend', '--self-test'], cwd='/', env=smoke_env, capture_output=True, text=True, timeout=90)
    evidence['frozen_self_test'] = json.loads(smoke.stdout)
    if not evidence['frozen_self_test'].get('ok'):
        raise RuntimeError('Signed frozen helper self-test failed')
    if release:
        archive = BUILD / 'Vostavo.zip'
        run(['ditto', '-c', '-k', '--keepParent', bundle, archive])
        evidence['app_notarization'] = notarize(archive, args.notary_profile)
        run(['xcrun', 'stapler', 'staple', bundle])
        run(['xcrun', 'stapler', 'validate', bundle])
        run(['spctl', '-a', '-t', 'exec', '-vvv', bundle])
    (stage / 'Applications').symlink_to('/Applications')
    suffix = 'arm64' if release else 'arm64-internal-adhoc'
    artifact = BUILD / f'Vostavo-{version}-{suffix}.dmg'
    run(['hdiutil', 'create', '-ov', '-volname', 'Vostavo', '-srcfolder', stage, '-format', 'UDZO', artifact])
    run(['codesign', '--force', '--sign', args.identity if release else '-', '--timestamp' if release else '--timestamp=none', artifact])
    if release:
        evidence['dmg_notarization'] = notarize(artifact, args.notary_profile)
        run(['xcrun', 'stapler', 'staple', artifact])
        run(['xcrun', 'stapler', 'validate', artifact])
        run(['spctl', '-a', '-t', 'open', '--context', 'context:primary-signature', '-vvv', artifact])
    run(['codesign', '--verify', '--deep', '--strict', bundle])
    evidence.update(mode=args.mode, gatekeeper_accepted=release, version=version, artifact=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), lock_sha256=hashlib.sha256((ROOT / 'packaging/requirements-macos-arm64.txt').read_bytes()).hexdigest(), python=output([python, '--version']), pyinstaller=output([python, '-m', 'PyInstaller', '--version']), rust=output([cargo, '--version']))
    report = BUILD / 'build-report.json'
    report.write_text(json.dumps(evidence, indent=2))
    (BUILD / (artifact.name + '.sha256')).write_text(f'{evidence["sha256"]}  {artifact.name}\n')
    print(f'Built {artifact}\nEvidence {report}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
