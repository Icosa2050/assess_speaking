"""Verify exact file/project/parameterized-title identities and first-pass execution."""
from collections import Counter
import json
from pathlib import Path
import sys


def cases(report):
    root = Path(report['config']['rootDir'])
    category = root.name
    def walk(suites, parents=()):
        for suite in suites:
            title = suite.get('title', '')
            next_parents = parents if title.endswith('.spec.ts') else parents + ((title,) if title else ())
            for spec in suite.get('specs', []):
                source = Path(spec['file'])
                file = source.relative_to(root).as_posix() if source.is_absolute() else source.as_posix()
                for test in spec['tests']:
                    identity = (f'{category}/{file}', test.get('projectName', ''), *next_parents, spec['title'])
                    yield identity, test
            yield from walk(suite.get('suites', []), next_parents)
    return list(walk(report.get('suites', [])))


def verify(lane, path, manifest=None):
    manifest = manifest or Path(__file__).resolve().parents[1] / 'frontend/tests/browser-identities.json'
    expected = {tuple(item) for item in json.loads(Path(manifest).read_text())[lane]}
    report = json.loads(Path(path).read_text())
    rows = cases(report)
    counts = Counter(identity for identity, _ in rows)
    failures = []
    if report.get('errors'):
        failures.append('suite errors')
    if set(counts) != expected:
        failures.append(f'missing/unexpected identities: {sorted(expected - set(counts))} / {sorted(set(counts) - expected)}')
    for identity, test in rows:
        results = test.get('results', [])
        if counts[identity] != 1 or test.get('expectedStatus') != 'passed' or test.get('status') != 'expected' or len(results) != 1 or results[0].get('status') != 'passed' or results[0].get('retry', 0) != 0:
            failures.append(f'not executed once and passed: {identity}')
    if not expected or failures:
        raise ValueError('Browser execution gate failed: ' + '; '.join(failures))
    print(f'{lane}: {len(expected)} mandatory identities executed once and passed')


if __name__ == '__main__':
    verify(sys.argv[1], sys.argv[2])
