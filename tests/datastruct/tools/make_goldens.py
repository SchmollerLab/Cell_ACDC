"""Capture the characterization goldens (not a test module).

Goldens describe main @ 7c0f7532 and are captured from a pristine git
worktree of that commit with the test helpers copied in, because
`python -m` puts the cwd first on sys.path and main's child processes
inherit PYTHONPATH::

    git -C <repo> worktree add --detach <wt> 7c0f7532
    cp -r <repo>/tests/datastruct <wt>/tests/
    rm -rf <wt>/tests/datastruct/golden
    cd <wt>
    PYTHONHASHSEED=0 PYTHONPATH=<wt> QT_QPA_PLATFORM=offscreen \\
        python -m tests.datastruct.tools.make_goldens \\
        --out <repo>/tests/datastruct/golden

Then capture a second time into a temporary folder and compare the two
trees byte by byte (`--compare <first>` does it). Module 0 scenarios run
main's real child processes (10-15 min). Never run two captures, or a
capture and a Module 0 comparison on main, at the same time: main's children
share ~/acdc-appdata/acdc_dataStruct_temp.

Output (JSON only, `manifest.dump_json`):
    module0/<id>.json      harness_module0.run_module0_scenario records
    restructure/<id>.json  harness_restructure.run_restructure_case records
    cascade.json           harness_module0.capture_cascade
    capture.json           commit, scenario ids and environment
Every record carries the informational `@env` block (ignored by the diff).
"""
from __future__ import annotations

import argparse
import filecmp
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
import time

from .. import (
    conditions, harness_module0, harness_restructure, manifest, scenarios
)

KINDS = ('restructure', 'cascade', 'module0')


def _git(*args: str) -> str:
    result = subprocess.run(
        ['git', '-C', conditions.REPO_ROOT, *args], capture_output=True,
        text=True, timeout=120
    )
    return result.stdout.strip() if result.returncode == 0 else ''


def _selected(ids, patterns) -> list[str]:
    if not patterns:
        return list(ids)
    return [i for i in ids if any(fnmatch.fnmatchcase(i, p)
                                  for p in patterns)]


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _dump(record: dict, path: str, env: dict) -> None:
    record = dict(record)
    record[manifest.ENV_KEY] = env
    manifest.dump_json(record, path)


def capture(out: str, kinds, patterns, workdir: str) -> list[str]:
    """Write the selected goldens into `out`; returns the written paths."""
    env = manifest.env_info()
    written = []
    if 'restructure' in kinds:
        for case_id in _selected(scenarios.RESTRUCTURE_IDS, patterns):
            record = harness_restructure.run_restructure_case(
                case_id, os.path.join(workdir, 'restructure', case_id)
            )
            path = os.path.join(out, 'restructure', f'{case_id}.json')
            _dump(record, path, env)
            written.append(path)
        _log(f'restructure: {len(written)} cases')
    if 'cascade' in kinds and _selected(['cascade'], patterns):
        record = harness_module0.capture_cascade(
            os.path.join(workdir, 'cascade')
        )
        path = os.path.join(out, 'cascade.json')
        _dump(record, path, env)
        written.append(path)
        _log('cascade: done')
    if 'module0' in kinds:
        for scenario_id in _selected(scenarios.MODULE0_IDS, patterns):
            start = time.time()
            record = harness_module0.run_module0_scenario(
                scenario_id, os.path.join(workdir, 'module0', scenario_id)
            )
            path = os.path.join(out, 'module0', f'{scenario_id}.json')
            _dump(record, path, env)
            written.append(path)
            _log(f'module0 {scenario_id}: {time.time() - start:.0f} s, '
                 f'task_ended={record["task_ended"]}, '
                 f'exceptions={record["exceptions"]}')
    manifest.dump_json({
        'commit': _git('rev-parse', 'HEAD'),
        'expected_commit': scenarios.MAIN_COMMIT,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'module0': list(scenarios.MODULE0_IDS),
        'restructure': list(scenarios.RESTRUCTURE_IDS),
        manifest.ENV_KEY: env,
    }, os.path.join(out, 'capture.json'))
    return written


def compare_trees(first: str, second: str) -> list[str]:
    """Relative paths whose bytes differ or that exist on one side only."""
    problems = []
    names = set()
    for root in (first, second):
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                names.add(os.path.relpath(os.path.join(dirpath, filename),
                                          root))
    for name in sorted(names):
        a, b = os.path.join(first, name), os.path.join(second, name)
        if not (os.path.isfile(a) and os.path.isfile(b)) or \
                not filecmp.cmp(a, b, shallow=False):
            problems.append(name.replace(os.sep, '/'))
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m tests.datastruct.tools.make_goldens',
        description='Capture the data-structure goldens.'
    )
    parser.add_argument('--out', required=True,
                        help='output folder (e.g. tests/datastruct/golden)')
    parser.add_argument('--kinds', default=','.join(KINDS),
                        help=f'comma-separated subset of {KINDS}')
    parser.add_argument('--only', default='',
                        help='comma-separated fnmatch patterns of scenario/'
                             "case ids ('cascade' selects the cascade)")
    parser.add_argument('--workdir', default=None,
                        help='scratch folder for inputs and outputs '
                             '(default: a new temporary folder, removed)')
    parser.add_argument('--compare', default=None, metavar='GOLDEN_DIR',
                        help='after capturing, require byte-identical files '
                             'in GOLDEN_DIR (stability check)')
    parser.add_argument('--allow-dirty', action='store_true',
                        help='capture although cellacdc/ has local changes')
    args = parser.parse_args(argv)

    harness_module0.assert_imports_this_checkout()
    dirty = _git('status', '--porcelain', '--', 'cellacdc')
    if dirty and not args.allow_dirty:
        _log(f'cellacdc/ has local changes, refusing to capture:\n{dirty}')
        return 2
    commit = _git('rev-parse', 'HEAD')
    if not commit.startswith(scenarios.MAIN_COMMIT):
        _log(f'WARNING: HEAD is {commit or "unknown"}, goldens describe '
             f'{scenarios.MAIN_COMMIT}')
    kinds = [k.strip() for k in args.kinds.split(',') if k.strip()]
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        parser.error(f'unknown kinds {unknown}')
    patterns = [p.strip() for p in args.only.split(',') if p.strip()]

    out = os.path.abspath(args.out)
    workdir = args.workdir or tempfile.mkdtemp(prefix='acdc-goldens-')
    try:
        written = capture(out, kinds, patterns, os.path.abspath(workdir))
    finally:
        if args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)
    _log(f'{len(written)} goldens written to {out}')
    if args.compare:
        problems = compare_trees(os.path.abspath(args.compare), out)
        if problems:
            _log('NOT byte-identical:\n  ' + '\n  '.join(problems))
            return 1
        _log(f'byte-identical to {args.compare}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
