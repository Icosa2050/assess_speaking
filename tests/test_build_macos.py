import ast
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import plistlib
from unittest.mock import patch
import pytest

from scripts import build_macos


def test_internal_acceptance_command_is_required_only_in_internal_binary(tmp_path: Path) -> None:
    binary = tmp_path / 'vostavo-desktop'
    binary.write_bytes(b'ordinary release shell VOSTAVO_DEVICE_ONLY_DRAFTS_REQUIRED')
    build_macos.assert_delivery_feature(binary, release=True)
    with pytest.raises(RuntimeError):
        build_macos.assert_delivery_feature(binary, release=False)
    binary.write_bytes(b'ordinary shell VOSTAVO_INTERNAL_DMG_ACCEPTANCE VOSTAVO_INTERNAL_DMG_KEY_SERVICE')
    build_macos.assert_delivery_feature(binary, release=False)
    with pytest.raises(RuntimeError):
        build_macos.assert_delivery_feature(binary, release=True)
    binary.write_bytes(b'ordinary release shell')
    with pytest.raises(RuntimeError, match='Device-only draft policy'):
        build_macos.assert_delivery_feature(binary, release=True)


def test_native_audit_preserves_mac_os_14_floor(tmp_path: Path) -> None:
    binary = tmp_path / "Vostavo.app" / "Contents" / "MacOS" / "vostavo-desktop"
    with patch.object(build_macos, "macho_files", return_value=[binary]), patch.object(
        build_macos,
        "output",
        side_effect=["arm64", "binary:\n\t/usr/lib/libSystem.B.dylib", "minos 13.4", ""],
    ):
        result = build_macos.audit(tmp_path / "Vostavo.app")

    assert build_macos.MINIMUM_MACOS == "14.0"
    assert result["minimum_macos"] == "14.0"


def test_release_cargo_environment_blocks_ambient_debug_assertions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('RUSTFLAGS', '-C debug-assertions=on')
    monkeypatch.setenv('CARGO_ENCODED_RUSTFLAGS', '-C\x1fdebug-assertions=on')
    monkeypatch.setenv('CARGO_BUILD_RUSTFLAGS', '-C debug-assertions=on')
    monkeypatch.setenv('CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS', 'true')
    env = build_macos.cargo_environment()
    assert env['CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS'] == 'false'
    assert env['MACOSX_DEPLOYMENT_TARGET'] == '14.0'
    assert all(key not in env for key in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS'))


def test_frozen_helper_declares_the_same_mac_os_floor() -> None:
    spec = ast.parse((build_macos.ROOT / "packaging/vostavo-backend.spec").read_text())
    bundle = next(
        node for node in ast.walk(spec)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "BUNDLE"
    )
    info_plist = next(keyword.value for keyword in bundle.keywords if keyword.arg == "info_plist")
    assert isinstance(info_plist, ast.Dict)
    minimum = next(
        value for key, value in zip(info_plist.keys, info_plist.values)
        if ast.literal_eval(key) == "LSMinimumSystemVersion"
    )
    assert ast.literal_eval(minimum) == build_macos.MINIMUM_MACOS


def test_developer_id_profile_must_authorize_exact_app_and_keychain_group() -> None:
    app_id = 'L98MZWWR98.com.vostavo.desktop'
    signing_certificate = b'signing certificate fixture'
    signing_cert_sha1 = hashlib.sha1(signing_certificate).hexdigest().upper()
    profile = {
        'ProvisionsAllDevices': True,
        'ExpirationDate': datetime.now(timezone.utc) + timedelta(days=30),
        'TeamIdentifier': ['L98MZWWR98'],
        'DeveloperCertificates': [signing_certificate],
        'Entitlements': {
            'com.apple.application-identifier': app_id,
            'com.apple.developer.team-identifier': 'L98MZWWR98',
            'keychain-access-groups': ['L98MZWWR98.*'],
            'get-task-allow': False,
        },
    }
    assert build_macos.validate_profile(profile, 'com.vostavo.desktop', 'L98MZWWR98', signing_cert_sha1) == {
        'com.apple.application-identifier': app_id,
        'com.apple.developer.team-identifier': 'L98MZWWR98',
        'keychain-access-groups': [app_id],
    }
    for change in (
        {'ProvisionsAllDevices': False},
        {'TeamIdentifier': ['OTHERTEAM']},
        {'DeveloperCertificates': [b'another certificate fixture']},
        {'Entitlements': {**profile['Entitlements'], 'com.apple.application-identifier': 'L98MZWWR98.other.app'}},
        {'Entitlements': {**profile['Entitlements'], 'keychain-access-groups': ['OTHERTEAM.*']}},
        {'Entitlements': {**profile['Entitlements'], 'get-task-allow': True}},
        {'Entitlements': {**profile['Entitlements'], 'get-task-allow': 1}},
    ):
        with pytest.raises(RuntimeError, match='does not authorize'):
            build_macos.validate_profile({**profile, **change}, 'com.vostavo.desktop', 'L98MZWWR98', signing_cert_sha1)
    with pytest.raises(RuntimeError, match='expired'):
        build_macos.validate_profile({**profile, 'ExpirationDate': datetime.now(timezone.utc) - timedelta(days=1)},
                                     'com.vostavo.desktop', 'L98MZWWR98', signing_cert_sha1)
    with pytest.raises(RuntimeError, match='malformed'):
        build_macos.validate_profile({**profile, 'Entitlements': []},
                                     'com.vostavo.desktop', 'L98MZWWR98', signing_cert_sha1)


def test_signing_identity_must_resolve_to_one_certificate() -> None:
    identity = 'Developer ID Application: Markus Frommherz (L98MZWWR98)'
    certificate_hash = 'A' * 40
    line = f'  1) {certificate_hash} "{identity}"'
    assert build_macos.signing_certificate_sha1(identity, line) == certificate_hash
    with pytest.raises(RuntimeError, match='missing or ambiguous'):
        build_macos.signing_certificate_sha1(identity, line + '\n' + line)
    with pytest.raises(RuntimeError, match='missing or ambiguous'):
        build_macos.signing_certificate_sha1(identity, line.replace(identity, 'Other identity'))


def test_profile_is_embedded_and_its_claims_are_signed(tmp_path: Path) -> None:
    bundle = tmp_path / 'Vostavo.app'
    (bundle / 'Contents').mkdir(parents=True)
    source = tmp_path / 'Vostavo-Developer-ID.provisionprofile'
    source.write_bytes(b'profile fixture')
    app_id = 'L98MZWWR98.com.vostavo.desktop'
    claims = {'com.apple.application-identifier': app_id,
              'com.apple.developer.team-identifier': 'L98MZWWR98',
              'keychain-access-groups': [app_id]}
    output = tmp_path / 'entitlements.plist'
    assert build_macos.embed_profile(bundle, source, claims, output) == output
    assert (bundle / 'Contents/embedded.provisionprofile').read_bytes() == source.read_bytes()
    signed = plistlib.loads(output.read_bytes())
    assert all(signed[key] == value for key, value in claims.items())
    assert signed['com.apple.security.device.audio-input'] is True
    with patch.object(build_macos, 'run', return_value=type('Result', (), {'stdout': plistlib.dumps(signed)})()):
        build_macos.verify_signed_profile_claims(bundle, claims)
    with patch.object(build_macos, 'run', return_value=type('Result', (), {'stdout': plistlib.dumps({})})()):
        with pytest.raises(RuntimeError, match='missing the authorized'):
            build_macos.verify_signed_profile_claims(bundle, claims)


def test_notarized_release_refuses_missing_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(build_macos.PROFILE_ENV, raising=False)
    monkeypatch.setattr(build_macos.sys, 'platform', 'darwin')
    monkeypatch.setattr(build_macos.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(build_macos, 'output', lambda _: 'Developer ID Application: Markus Frommherz (L98MZWWR98)')
    with pytest.raises(SystemExit) as exc_info:
        build_macos.main(['--mode', 'release', '--identity',
                          'Developer ID Application: Markus Frommherz (L98MZWWR98)',
                          '--notary-profile', 'vostavo-notary'])
    assert exc_info.value.code == 2
    assert build_macos.PROFILE_ENV in capsys.readouterr().err


def test_local_signed_release_requires_explicit_disabled_drafts_acknowledgment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    identity = 'Developer ID Application: Markus Frommherz (L98MZWWR98)'
    monkeypatch.delenv(build_macos.PROFILE_ENV, raising=False)
    monkeypatch.setattr(build_macos.sys, 'platform', 'darwin')
    monkeypatch.setattr(build_macos.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(build_macos, 'BUILD', tmp_path / 'build')
    monkeypatch.setattr(build_macos, 'output', lambda _: f'  1) {"A" * 40} "{identity}"')
    args = ['--mode', 'release', '--identity', identity, '--local-sign-only']
    with pytest.raises(SystemExit) as exc_info:
        build_macos.main(args)
    assert exc_info.value.code == 2
    assert '--allow-drafts-disabled' in capsys.readouterr().err
    with patch.object(build_macos, 'run', side_effect=RuntimeError('reached build')):
        with pytest.raises(RuntimeError, match='reached build'):
            build_macos.main(args + ['--allow-drafts-disabled'])
    with pytest.raises(SystemExit) as exc_info:
        build_macos.main(['--mode', 'internal', '--local-sign-only'])
    assert exc_info.value.code == 2
