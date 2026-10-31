#!/usr/bin/env python3
"""Build an internal DMG or a Developer ID notarized release from locked inputs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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
MINIMUM_MACOS = '14.0'  # The native audit of the built app requires this floor.
COPYRIGHT = 'Copyright © 2026 Vostavo'
PROFILE_ENV = 'VOSTAVO_PROVISIONING_PROFILE'


def run(args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def output(args):
    return run(args, capture_output=True, text=True).stdout.strip()


def cargo_environment() -> dict[str, str]:
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=MINIMUM_MACOS)
    for key in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS'):
        env.pop(key, None)
    env['CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS'] = 'false'
    return env


def assert_delivery_feature(shell_binary: Path, *, release: bool) -> None:
    binary = shell_binary.read_bytes()
    markers = (b'VOSTAVO_INTERNAL_DMG_ACCEPTANCE', b'VOSTAVO_INTERNAL_DMG_KEY_SERVICE')
    if any((marker in binary) == release for marker in markers):
        raise RuntimeError('Internal draft acceptance feature does not match the DMG mode')
    if (b'VOSTAVO_DEVICE_ONLY_DRAFTS_REQUIRED' in binary) != release:
        raise RuntimeError('Device-only draft policy does not match the DMG mode')


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
    minimum = tuple(map(int, MINIMUM_MACOS.split('.')))
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


def signing_certificate_sha1(identity: str, identities: str) -> str:
    matches = [match.group(1).upper() for line in identities.splitlines()
               if (match := re.fullmatch(r'\s*\d+\)\s+([0-9a-fA-F]{40})\s+"' + re.escape(identity) + r'"\s*', line))]
    if len(matches) != 1:
        raise RuntimeError('The requested Developer ID identity is missing or ambiguous in the signing keychain')
    return matches[0]


def validate_profile(profile: dict, bundle_id: str, team_id: str, signing_cert_sha1: str) -> dict:
    if not isinstance(profile, dict):
        raise RuntimeError('Developer ID provisioning profile is malformed')
    app_id = f'{team_id}.{bundle_id}'
    approved = profile.get('Entitlements', {})
    teams = profile.get('TeamIdentifier', [])
    certificates = profile.get('DeveloperCertificates', [])
    expiry = profile.get('ExpirationDate')
    if (not isinstance(approved, dict) or not isinstance(teams, list)
            or not isinstance(certificates, list) or not isinstance(expiry, datetime)):
        raise RuntimeError('Developer ID provisioning profile has missing or malformed claims')
    if (expiry.replace(tzinfo=timezone.utc) if expiry.tzinfo is None else expiry) <= datetime.now(timezone.utc):
        raise RuntimeError('Developer ID provisioning profile has expired')
    groups = approved.get('keychain-access-groups', [])
    if not isinstance(groups, list):
        raise RuntimeError('Developer ID provisioning profile has malformed Keychain groups')
    if (profile.get('ProvisionsAllDevices') is not True
            or team_id not in teams
            or approved.get('com.apple.application-identifier') != app_id
            or approved.get('com.apple.developer.team-identifier') != team_id
            or not any(group in (app_id, f'{team_id}.*') for group in groups)
            or not any(isinstance(cert, bytes) and hashlib.sha1(cert).hexdigest().upper() == signing_cert_sha1
                       for cert in certificates)
            or bool(approved.get('get-task-allow'))):
        raise RuntimeError('Developer ID provisioning profile does not authorize this signing certificate, team, app ID, and Keychain group')
    return {
        'com.apple.application-identifier': app_id,
        'com.apple.developer.team-identifier': team_id,
        'keychain-access-groups': [app_id],
    }


def load_profile(path: Path, bundle_id: str, team_id: str, signing_cert_sha1: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f'Provisioning profile is missing: {path}')
    try:
        decoded = run(['security', 'cms', '-D', '-i', path], capture_output=True).stdout
        profile = plistlib.loads(decoded)
    except (subprocess.CalledProcessError, plistlib.InvalidFileException, ValueError, TypeError) as exc:
        raise RuntimeError('Developer ID provisioning profile cannot be decoded') from exc
    return validate_profile(profile, bundle_id, team_id, signing_cert_sha1)


def embed_profile(bundle: Path, source: Path, claims: dict, entitlements: Path) -> Path:
    shutil.copy2(source, bundle / 'Contents/embedded.provisionprofile')
    entitlements.write_bytes(plistlib.dumps({
        **plistlib.loads((ROOT / 'packaging/entitlements.plist').read_bytes()), **claims,
    }))
    return entitlements


def verify_signed_profile_claims(bundle: Path, claims: dict) -> None:
    if not (bundle / 'Contents/embedded.provisionprofile').is_file():
        raise RuntimeError('Signed release app is missing its embedded provisioning profile')
    try:
        signed = plistlib.loads(run(['codesign', '-d', '--entitlements', ':-', bundle],
                                   capture_output=True).stdout)
    except (subprocess.CalledProcessError, plistlib.InvalidFileException, ValueError, TypeError) as exc:
        raise RuntimeError('Signed release app entitlements cannot be read') from exc
    if not isinstance(signed, dict) or any(signed.get(key) != value for key, value in claims.items()):
        raise RuntimeError('Signed release app is missing the authorized Keychain entitlements')


def sign(bundle, identity, release, entitlements):
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
    run(['codesign', *options, '--entitlements', entitlements, bundle])
    run(['codesign', '--verify', '--deep', '--strict', '--verbose=2', bundle])


def notarize(artifact, profile):
    result = json.loads(output(['xcrun', 'notarytool', 'submit', artifact, '--keychain-profile', profile, '--wait', '--output-format', 'json']))
    if result.get('status') != 'Accepted':
        raise RuntimeError(f'Notarization was not accepted: {result.get("id")} {result.get("status")}')
    return result


def notarize_and_staple_app(bundle: Path, profile: str, archive: Path) -> dict:
    run(['ditto', '-c', '-k', '--keepParent', bundle, archive])
    result = notarize(archive, profile)
    run(['xcrun', 'stapler', 'staple', bundle])
    run(['xcrun', 'stapler', 'validate', bundle])
    run(['spctl', '-a', '-t', 'exec', '-vvv', bundle])
    return result


def notarize_and_staple_dmg(artifact: Path, profile: str) -> dict:
    result = notarize(artifact, profile)
    run(['xcrun', 'stapler', 'staple', artifact])
    run(['xcrun', 'stapler', 'validate', artifact])
    run(['spctl', '-a', '-t', 'open', '--context', 'context:primary-signature', '-vvv', artifact])
    return result


def package_dmg(stage: Path, bundle: Path, artifact: Path, *, release: bool,
                local_sign_only: bool, profile: str, signing_identity: str) -> dict:
    """Package in the release order; mock run/output for a no-Apple dry run."""
    result = {}
    if release and not local_sign_only:
        result['app_notarization'] = notarize_and_staple_app(bundle, profile, BUILD / 'Vostavo.zip')
    (stage / 'Applications').symlink_to('/Applications')
    run(['hdiutil', 'create', '-ov', '-volname', 'Vostavo', '-srcfolder', stage, '-format', 'UDZO', artifact])
    run(['codesign', '--force', '--sign', signing_identity, '--timestamp' if release else '--timestamp=none', artifact])
    if release and not local_sign_only:
        result['dmg_notarization'] = notarize_and_staple_dmg(artifact, profile)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['internal', 'release'], required=True)
    parser.add_argument('--identity', default='')
    parser.add_argument('--notary-profile', default='')
    parser.add_argument('--local-sign-only', action='store_true',
                        help='Developer ID sign app and DMG locally without notarizing or uploading.')
    parser.add_argument('--allow-drafts-disabled', action='store_true',
                        help='Acknowledge that a local signed build without a provisioning profile cannot save drafts.')
    parser.add_argument('--python', default=PYTHON_VERSION)
    parser.add_argument('--reuse-helper', action='store_true', help='Local iteration only; forbidden for releases.')
    args = parser.parse_args(argv)
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        parser.error('This build requires an Apple Silicon Mac.')
    release = args.mode == 'release'
    if args.local_sign_only and not release:
        parser.error('--local-sign-only requires --mode release')
    if args.allow_drafts_disabled and not (release and args.local_sign_only):
        parser.error('--allow-drafts-disabled requires a local-sign-only release')
    signing_cert_sha1 = None
    if release:
        if args.reuse_helper or not args.identity.startswith('Developer ID Application:') or (not args.local_sign_only and not args.notary_profile):
            parser.error('Release requires a Developer ID Application identity and, unless local-sign-only, a Keychain notary profile; no reused helper.')
        if not args.local_sign_only and not os.environ.get(PROFILE_ENV):
            parser.error(f'Notarized release requires {PROFILE_ENV} for device-only draft storage.')
        if args.local_sign_only and not os.environ.get(PROFILE_ENV) and not args.allow_drafts_disabled:
            parser.error(f'Local signed release without {PROFILE_ENV} requires --allow-drafts-disabled.')
        try:
            signing_cert_sha1 = signing_certificate_sha1(
                args.identity, output(['security', 'find-identity', '-v', '-p', 'codesigning']))
        except RuntimeError as exc:
            parser.error(str(exc))
    config = json.loads((ROOT / 'frontend/src-tauri/tauri.conf.json').read_text())
    version = config['version']
    if config['identifier'] != 'com.vostavo.desktop' or config['bundle']['copyright'] != COPYRIGHT or config['bundle']['macOS']['minimumSystemVersion'] != MINIMUM_MACOS:
        raise RuntimeError('Tauri release identity differs from macOS build identity')
    if tomllib.loads((ROOT / 'frontend/src-tauri/Cargo.toml').read_text())['package']['version'] != version:
        raise RuntimeError('Rust and Tauri versions differ')
    profile_source = Path(os.environ[PROFILE_ENV]).expanduser() if release and os.environ.get(PROFILE_ENV) else None
    profile_claims = None
    if profile_source:
        team = re.search(r'\(([A-Z0-9]{10})\)$', args.identity)
        if not team:
            raise RuntimeError('Developer ID signing identity must include its Team ID')
        profile_claims = load_profile(profile_source, config['identifier'], team.group(1), signing_cert_sha1)
    BUILD.mkdir(exist_ok=True)
    python = BUILD / 'venv/bin/python'
    if not python.exists():
        run(['uv', 'venv', '--python', args.python, BUILD / 'venv'])
    run(['uv', 'pip', 'sync', '--python', python, '--require-hashes', ROOT / 'packaging/requirements-macos-arm64.txt'])
    if output([python, '-c', 'import platform; print(platform.python_version(), platform.machine())']) != PYTHON_VERSION + ' arm64':
        raise RuntimeError('Packaging requires pinned CPython 3.12.11 arm64')
    npm_env = {key: os.environ[key] for key in ('PATH', 'HOME', 'TMPDIR', 'LANG') if key in os.environ}
    run(['npm', '--prefix', ROOT / 'frontend', 'ci', '--ignore-scripts'], env=npm_env)
    helper = BUILD / 'dist/VostavoBackend.app'
    if not args.reuse_helper:
        run([python, '-m', 'PyInstaller', '--clean', '--noconfirm', '--distpath', BUILD / 'dist', '--workpath', BUILD / 'pyinstaller', ROOT / 'packaging/vostavo-backend.spec'], cwd=ROOT)
    if not helper.is_dir():
        raise RuntimeError('Frozen helper app was not produced')
    if output([python, '-c', 'import platform; print(platform.python_version(), platform.machine())']) != PYTHON_VERSION + ' arm64':
        raise RuntimeError('Packaging requires pinned CPython 3.12.11 arm64')
    if plistlib.loads((helper / 'Contents/Info.plist').read_bytes())['CFBundleShortVersionString'] != version:
        raise RuntimeError('Helper version differs')
    notices = BUILD / 'THIRD_PARTY_NOTICES.txt'
    run([python, ROOT / 'scripts/generate_third_party_notices.py', '--output', notices, '--helper', helper])
    public_dir = BUILD / 'frontend-public'
    shutil.copytree(ROOT / 'frontend/public', public_dir, dirs_exist_ok=True)
    public_notices = public_dir / 'THIRD_PARTY_NOTICES.txt'
    shutil.copy2(notices, public_notices)
    build_commit = output(['git', 'rev-parse', '--short=12', 'HEAD'])
    npm_env['VITE_VOSTAVO_BUILD_COMMIT'] = build_commit
    npm_env['VOSTAVO_PUBLIC_DIR'] = str(public_dir)
    run(['npm', '--prefix', ROOT / 'frontend', 'run', 'build'], env=npm_env)
    env = cargo_environment()
    cargo = shutil.which('cargo') or str(Path.home() / '.cargo/bin/cargo')
    features = 'custom-protocol,internal-dmg-acceptance' if not release else 'custom-protocol,device-only-drafts'
    run([cargo, 'build', '--release', '--locked', '--features', features, '--manifest-path', ROOT / 'frontend/src-tauri/Cargo.toml'], env=env)
    assert_delivery_feature(ROOT / 'frontend/src-tauri/target/release/vostavo-desktop', release=release)
    stage = BUILD / 'stage'
    if stage.exists():
        shutil.rmtree(stage)
    bundle = stage / 'Vostavo.app'
    macos = bundle / 'Contents/MacOS'
    resources = bundle / 'Contents/Resources'
    macos.mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(notices, resources / notices.name)
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
    if evidence['minimum_macos'] != MINIMUM_MACOS:
        raise RuntimeError(f'Native minimum macOS {evidence["minimum_macos"]} exceeds declared {MINIMUM_MACOS}')
    info = {'CFBundleExecutable': 'vostavo-desktop', 'CFBundleIdentifier': config['identifier'], 'CFBundleName': 'Vostavo', 'CFBundleDisplayName': 'Vostavo', 'CFBundlePackageType': 'APPL', 'CFBundleShortVersionString': version, 'CFBundleVersion': version, 'NSHumanReadableCopyright': COPYRIGHT, 'CFBundleIconFile': 'Vostavo.icns', 'LSMinimumSystemVersion': MINIMUM_MACOS, 'NSMicrophoneUsageDescription': microphone['en'], 'NSAppleEventsUsageDescription': 'Vostavo can prepare a support email in Mail with your selected package attached. You review and send the draft.', 'CFBundleLocalizations': list(microphone), 'NSHighResolutionCapable': True}
    (bundle / 'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    # Keep nested bundle's declared minimum consistent with its actual native code.
    helper_info = bundle / 'Contents/Helpers/VostavoBackend.app/Contents/Info.plist'
    helper_plist = plistlib.loads(helper_info.read_bytes())
    helper_plist['LSMinimumSystemVersion'] = evidence['minimum_macos']
    helper_info.write_bytes(plistlib.dumps(helper_plist))
    entitlements = ROOT / 'packaging/entitlements.plist'
    if profile_source and profile_claims:
        entitlements = embed_profile(bundle, profile_source, profile_claims, BUILD / 'release-entitlements.plist')
    sign(bundle, signing_cert_sha1 if release else '-', release, entitlements)
    if profile_claims:
        verify_signed_profile_claims(bundle, profile_claims)
    smoke_env = {key: os.environ[key] for key in ('HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG') if key in os.environ}
    smoke_env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    smoke = run([bundle / 'Contents/Helpers/VostavoBackend.app/Contents/MacOS/vostavo-backend', '--self-test'], cwd='/', env=smoke_env, capture_output=True, text=True, timeout=90)
    evidence['frozen_self_test'] = json.loads(smoke.stdout)
    if not evidence['frozen_self_test'].get('ok'):
        raise RuntimeError('Signed frozen helper self-test failed')
    suffix = 'arm64-signed-unnotarized' if args.local_sign_only else 'arm64' if release else 'arm64-internal-adhoc'
    artifact = BUILD / f'Vostavo-{version}-{suffix}.dmg'
    evidence.update(package_dmg(stage, bundle, artifact, release=release, local_sign_only=args.local_sign_only,
                                profile=args.notary_profile, signing_identity=signing_cert_sha1 if release else '-'))
    run(['codesign', '--verify', '--deep', '--strict', bundle])
    evidence.update(mode=args.mode, notarization_skipped=args.local_sign_only,
                    provisioning_profile_embedded=bool(profile_source),
                    draft_storage='device-only' if release and profile_source else 'unavailable' if release else 'development-fallback',
                    gatekeeper_accepted=release and not args.local_sign_only,
                    version=version, bundle_short_version=version, bundle_version=version, bundle_id=config['identifier'],
                    copyright=COPYRIGHT, minimum_macos=MINIMUM_MACOS, build_commit=build_commit,
                    third_party_notices_sha256=hashlib.sha256(notices.read_bytes()).hexdigest(),
                    artifact=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), lock_sha256=hashlib.sha256((ROOT / 'packaging/requirements-macos-arm64.txt').read_bytes()).hexdigest(), python=output([python, '--version']), pyinstaller=output([python, '-m', 'PyInstaller', '--version']), rust=output([cargo, '--version']))
    report = BUILD / 'build-report.json'
    report.write_text(json.dumps(evidence, indent=2))
    (BUILD / (artifact.name + '.sha256')).write_text(f'{evidence["sha256"]}  {artifact.name}\n')
    print(f'Built {artifact}\nEvidence {report}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
