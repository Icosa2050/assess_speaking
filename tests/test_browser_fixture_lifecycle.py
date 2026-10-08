from pathlib import Path
import subprocess
import sys
import tempfile
import pytest
from tests.helpers.browser_fixture import owned_root


def test_cleanup_refuses_non_owned_paths_and_symlinks(tmp_path):
    for path in [tmp_path, Path('/tmp'), tmp_path / 'vostavo-default-user']:
        with pytest.raises(ValueError): owned_root(path)
    link = tmp_path / 'vostavo-default-link'; link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError): owned_root(link)
    assert tmp_path.exists()


def test_launcher_retains_guard_evidence_and_waits_for_backend_shutdown(tmp_path):
    # Python's generated suffix can contain underscores, unlike shell mktemp.
    root = Path(tempfile.mkdtemp(prefix='vostavo-default-under_score_'))
    child = tmp_path / 'backend_fixture.py'
    child.write_text("from pathlib import Path\nimport sys\nroot=Path(sys.argv[1])\n(root/'guards.jsonl').write_text('fixture evidence')\n(root/'backend_state.json').write_text('running')\n(root/'backend_state.json').unlink()\n")
    output = tmp_path / 'evidence'
    result = subprocess.run([sys.executable, 'tests/helpers/browser_fixture.py', '--root', str(root), '--evidence', str(output), '--', str(child), str(root)], timeout=10, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not root.exists()
    assert (output / 'guards.jsonl').read_text() == 'fixture evidence'
