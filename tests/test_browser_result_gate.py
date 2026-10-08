import json
import pytest
from scripts.check_browser_results import verify

@pytest.mark.parametrize('outcome', ['pass', 'missing', 'skip', 'fail', 'flaky', 'retry', 'duplicate', 'collection', 'unexpected', 'wrong-project', 'wrong-file', 'suite-error'])
def test_exact_browser_execution(tmp_path, outcome):
    identity = ['e2e/practice.spec.ts', 'chromium', 'en', 'practice B1']
    manifest = tmp_path / 'expected.json'; manifest.write_text(json.dumps({'default': [identity]}))
    test = {'projectName': 'chromium', 'expectedStatus': 'passed', 'status': 'expected', 'results': [{'status': 'passed', 'retry': 0}]}
    spec = {'file': 'practice.spec.ts', 'title': 'practice B1', 'tests': [test]}
    report = {'config': {'rootDir': '/repo/frontend/tests/e2e'}, 'suites': [{'title': 'practice.spec.ts', 'suites': [{'title': 'en', 'specs': [spec]}]}]}
    if outcome == 'missing': report['suites'] = []
    if outcome == 'skip': test.update(status='skipped', results=[{'status': 'skipped'}])
    if outcome == 'fail': test['results'][0]['status'] = 'failed'
    if outcome == 'flaky': test.update(status='flaky', results=[{'status': 'failed'}, {'status': 'passed', 'retry': 1}])
    if outcome == 'retry': test['results'][0]['retry'] = 1
    if outcome == 'duplicate': spec['tests'].append(dict(test))
    if outcome == 'collection': test.update(status='skipped', results=[])
    if outcome == 'unexpected': spec['title'] = 'other'
    if outcome == 'wrong-project': test['projectName'] = 'webkit'
    if outcome == 'wrong-file': spec['file'] = 'other.spec.ts'
    if outcome == 'suite-error': report['errors'] = [{'message': 'teardown failed'}]
    result = tmp_path / 'results.json'; result.write_text(json.dumps(report))
    if outcome == 'pass': verify('default', result, manifest)
    else:
        with pytest.raises(ValueError, match='Browser execution'): verify('default', result, manifest)
