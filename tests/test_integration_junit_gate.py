"""Enabled CI cannot pass with missing, skipped or collection-only integration."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.check_integration_junit import mandatory_cases, verify


@pytest.mark.parametrize('lane', ['sample', 'cloud', 'recovery', 'codec'])
@pytest.mark.parametrize('outcome', ['pass', 'missing', 'skipped', 'failure', 'error', 'duplicate'])
def test_required_junit_execution(tmp_path, lane, outcome):
    root = ET.Element('testsuite')
    for i, name in enumerate(sorted(mandatory_cases(lane))):
        if i == 0 and outcome == 'missing':
            continue
        node = ET.SubElement(root, 'testcase', name=name)
        if i == 0 and outcome == 'duplicate':
            ET.SubElement(root, 'testcase', name=name)
        if i == 0 and outcome in ('skipped', 'failure', 'error'):
            ET.SubElement(node, outcome)
    ET.SubElement(root, 'testcase', name='unrelated_extra')
    path = tmp_path/'results.xml'
    ET.ElementTree(root).write(path)
    if outcome == 'pass':
        verify(lane, path)
    else:
        with pytest.raises(ValueError, match='Mandatory'):
            verify(lane, path)
