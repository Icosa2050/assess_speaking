import json
from pathlib import Path

import pytest

from scripts import generate_third_party_notices as notices


def test_npm_notices_follow_only_locked_runtime_graph(tmp_path: Path) -> None:
    root = tmp_path / 'node_modules'
    for name, version, license_name in [('react', '19.2.5', 'MIT'), ('scheduler', '0.27.0', 'MIT')]:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / 'package.json').write_text(json.dumps({'version': version, 'license': license_name}))
        (directory / 'LICENSE').write_text(f'{name} license')
    lock = tmp_path / 'package-lock.json'
    lock.write_text(json.dumps({'packages': {
        '': {'dependencies': {'react': '19.2.5'}},
        'node_modules/react': {'version': '19.2.5', 'dependencies': {'scheduler': '^0.27.0'}},
        'node_modules/scheduler': {'version': '0.27.0'},
        'node_modules/dev-only': {'version': '1.0.0'},
    }}))
    result = notices.npm_notices(lock, root)
    assert len(result) == 2
    assert 'react license' in result[0]
    assert 'scheduler license' in result[1]
    assert 'dev-only' not in '\n'.join(result)
    (root / 'scheduler/package.json').write_text(json.dumps({'version': '0.26.0', 'license': 'MIT'}))
    with pytest.raises(RuntimeError, match='lock mismatch'):
        notices.npm_notices(lock, root)


def test_npm_notices_include_nested_installed_versions(tmp_path: Path) -> None:
    root = tmp_path / 'node_modules'
    for relative, version in [('shared', '2.0.0'), ('widget', '1.0.0'),
                              ('widget/node_modules/shared', '1.0.0')]:
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / 'package.json').write_text(json.dumps({'version': version, 'license': 'MIT'}))
    lock = tmp_path / 'package-lock.json'
    lock.write_text(json.dumps({'packages': {
        '': {'dependencies': {'shared': '2.0.0', 'widget': '1.0.0'}},
        'node_modules/shared': {'version': '2.0.0'},
        'node_modules/widget': {'version': '1.0.0', 'dependencies': {'shared': '1.0.0'}},
        'node_modules/widget/node_modules/shared': {'version': '1.0.0'},
    }}))
    result = notices.npm_notices(lock, root)
    assert len(result) == 3
    assert any('npm runtime: shared 1.0.0' in row for row in result)
    assert any('npm runtime: shared 2.0.0' in row for row in result)


def test_ffmpeg_notice_uses_frozen_helper_and_pyav_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = tmp_path / 'VostavoBackend.app'
    for relative in ('Contents/Resources/libavcodec.61.dylib',
                     'Contents/Frameworks/av/__dot__dylibs/libavcodec.61.dylib',
                     'Contents/Frameworks/av/__dot__dylibs/libnewcodec.1.dylib'):
        path = helper / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    monkeypatch.setattr(notices.subprocess, 'check_output', lambda *args, **kwargs: (
        'PyAV v14.2.0\nlibrary configuration: --enable-gpl --enable-version3\n'
        'library license: GPL version 3 or later\nlibavcodec 61.19.100\n'))
    result = notices.ffmpeg_notice(helper)
    assert 'Contents/Resources/libavcodec.61.dylib' in result
    assert 'Contents/Frameworks/av/__dot__dylibs/libnewcodec.1.dylib' in result
    assert 'library configuration: --enable-gpl --enable-version3' in result
    assert 'PUBLIC DISTRIBUTION HOLD' in result
    with pytest.raises(RuntimeError, match='not allowed in a release build'):
        notices.ffmpeg_notice(helper, require_source_reference=True)
    allowed = notices.ffmpeg_notice(helper, source_version='7.0.2', source_archive='https://example.test/ffmpeg.tar.xz',
                                    require_source_reference=True, allow_gpl_distribution=True)
    assert 'FFmpeg source version: 7.0.2' in allowed
    assert (notices.ROOT / 'packaging/licenses/GPL-3.0.txt').read_text().strip() in allowed


@pytest.mark.parametrize('extra_report', [
    'library license: GPL version 3 or later',
    'library license: nonfree',
])
def test_ffmpeg_notice_checks_every_license_line_and_nonfree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_report: str
) -> None:
    helper = tmp_path / 'VostavoBackend.app'
    dylib = helper / 'Contents/Resources/libavcodec.61.dylib'
    dylib.parent.mkdir(parents=True)
    dylib.write_bytes(b'fixture')
    monkeypatch.setattr(notices.subprocess, 'check_output', lambda *args, **kwargs: (
        'PyAV v14.2.0\nlibrary configuration: --enable-shared\n'
        f'library license: LGPL version 2.1 or later\n{extra_report}\n'))
    with pytest.raises(RuntimeError, match='not allowed in a release build'):
        notices.ffmpeg_notice(helper, source_version='7.0.2', source_archive='archive.tar.xz',
                              require_source_reference=True)


@pytest.mark.parametrize('restricted_name', ['libx264.164.dylib', 'libx265.199.dylib'])
def test_ffmpeg_notice_checks_bundled_codec_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restricted_name: str
) -> None:
    helper = tmp_path / 'VostavoBackend.app'
    for name in ('libavcodec.61.dylib', restricted_name):
        dylib = helper / 'Contents/Resources' / name
        dylib.parent.mkdir(parents=True, exist_ok=True)
        dylib.write_bytes(b'fixture')
    monkeypatch.setattr(notices.subprocess, 'check_output', lambda *args, **kwargs: (
        'PyAV v14.2.0\nlibrary configuration: --enable-shared\n'
        'library license: LGPL version 2.1 or later\n'))
    with pytest.raises(RuntimeError, match='not allowed in a release build'):
        notices.ffmpeg_notice(helper, source_version='7.0.2', source_archive='archive.tar.xz',
                              require_source_reference=True)


@pytest.mark.parametrize('license_name,expected_files', [
    ('LGPL version 2.1 or later', ('LGPL-2.1.txt',)),
    ('LGPL version 3 or later', ('GPL-3.0.txt', 'LGPL-3.0.txt')),
])
def test_lgpl_notice_includes_full_text_and_source_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, license_name: str, expected_files: tuple[str, ...]
) -> None:
    helper = tmp_path / 'VostavoBackend.app'
    dylib = helper / 'Contents/Frameworks/av/__dot__dylibs/libavcodec.61.dylib'
    dylib.parent.mkdir(parents=True)
    dylib.write_bytes(b'fixture')
    monkeypatch.setattr(notices.subprocess, 'check_output', lambda *args, **kwargs: (
        'PyAV v14.2.0\nlibrary configuration: --disable-gpl --enable-shared\n'
        f'library license: {license_name}\nlibavcodec 61.19.100\n'))
    result = notices.ffmpeg_notice(helper, source_version='7.0.2', source_archive='sources/ffmpeg-7.0.2.tar.xz')
    assert 'PyAV version: 14.2.0' in result
    assert 'FFmpeg source version: 7.0.2' in result
    assert 'FFmpeg source archive path or URL: sources/ffmpeg-7.0.2.tar.xz' in result
    assert 'FFmpeg configure flags: --disable-gpl --enable-shared' in result
    assert 'FFmpeg source version: 7.0.2' in notices.ffmpeg_notice(
        helper, source_version='7.0.2', source_archive='sources/ffmpeg-7.0.2.tar.xz',
        require_source_reference=True)
    for filename in expected_files:
        assert (notices.ROOT / 'packaging/licenses' / filename).read_text().strip() in result
    placeholder = notices.ffmpeg_notice(helper)
    assert 'SOURCE_VERSION_REQUIRED' in placeholder
    assert 'SOURCE_ARCHIVE_REQUIRED' in placeholder
    with pytest.raises(RuntimeError, match='requires its source version and archive reference'):
        notices.ffmpeg_notice(helper, require_source_reference=True)


def test_frozen_inventory_requires_native_code_and_metadata(tmp_path: Path) -> None:
    helper = tmp_path / 'VostavoBackend.app'
    resources = helper / 'Contents/Resources'
    resources.mkdir(parents=True)
    (resources / 'faster_whisper-1.2.0.dist-info').mkdir()
    (resources / 'libavcodec.61.dylib').write_bytes(b'native fixture')
    result = notices.frozen_inventory(helper)
    assert 'faster_whisper-1.2.0.dist-info' in result
    assert 'libavcodec.61.dylib' in result
    (resources / 'libavcodec.61.dylib').unlink()
    with pytest.raises(RuntimeError, match='incomplete'):
        notices.frozen_inventory(helper)


@pytest.mark.parametrize('installed_path', ['LICENSE.md', 'licenses/LICENSE.md'])
def test_declared_python_license_paths(installed_path: str) -> None:
    class FakeDistribution:
        metadata = {'Name': 'example'}

        def read_text(self, path: str) -> str | None:
            return 'license body' if path == installed_path else None

    assert notices.read_python_license_file(FakeDistribution(), 'LICENSE.md') == 'license body'
    with pytest.raises(RuntimeError, match='Declared Python license file missing'):
        notices.read_python_license_file(FakeDistribution(), 'NOTICE.txt')
