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


def test_ffmpeg_notice_records_linkage_and_license() -> None:
    result = notices.ffmpeg_notice()
    assert 'libavcodec' in result
    assert 'configuration:' in result
    assert 'GPL' in result or 'LGPL' in result


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
