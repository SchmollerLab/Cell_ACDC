"""Start-up import time of the launcher, the GUI window and the viewer window.

Usage::

    python tools/refactor/verify/importtime.py snapshot --out <dir> [--repo <tree>]
        [--runs 3]
    python tools/refactor/verify/importtime.py compare --before <dir> --after <dir>
        [--tolerance 0.2]

For each target (``cellacdc``, the launcher, the GUI window module, the
viewer window module; new module paths first, see ``TARGETS``), runs
``python -X importtime -c "import <module>"`` from the tree's root
(``QT_QPA_PLATFORM=offscreen``) once to warm up the bytecode caches, then
`runs` times, and stores the median of: the cumulative time of the target
module's own line, the total (sum of the top-level entries), the process wall
time, and the numbers of imported modules (all and ``cellacdc``).

``compare`` warns when a median grows by more than `tolerance` (relative)
plus 0.1 s; timings are noisy, so it never fails.
"""
import argparse
import os
import statistics
import sys

try:
    from . import common
except ImportError:
    import common

SNAPSHOT_FILE = 'importtime.json'

TARGETS = (
    ('cellacdc', ['cellacdc']),
    ('launcher', ['cellacdc.launcher.main_window', 'cellacdc._main']),
    ('gui_window', ['cellacdc.gui.main_window.window', 'cellacdc.gui']),
    ('viewer_window', [
        'cellacdc.measurements_viewer.main_window.window',
        'cellacdc.measurements_viewer',
    ]),
)


def parse_importtime(stderr, module):
    entries = []
    for line in stderr.splitlines():
        if not line.startswith('import time:'):
            continue
        parts = line[len('import time:'):].split('|')
        if len(parts) != 3:
            continue
        try:
            self_us = int(parts[0].strip())
            cum_us = int(parts[1].strip())
        except ValueError:
            continue  # header line
        raw = parts[2][1:] if parts[2].startswith(' ') else parts[2]
        level = (len(raw) - len(raw.lstrip(' '))) // 2
        entries.append((level, raw.strip(), self_us, cum_us))
    total = sum(cum for level, _, _, cum in entries if level == 0)
    target = next((cum for _, name, _, cum in entries if name == module), None)
    return {
        'total_s': round(total / 1e6, 4),
        'target_s': round(target / 1e6, 4) if target is not None else None,
        'modules': len(entries),
        'cellacdc_modules': sum(
            1 for _, name, _, _ in entries
            if name == 'cellacdc' or name.startswith('cellacdc.')
        ),
    }


def measure(repo, module, runs=3, timeout=120):
    env = common.child_env(repo)
    cmd = [sys.executable, '-X', 'importtime', '-c', f'import {module}']
    common.run_child(cmd, repo, env, timeout)  # warm-up (pyc caches)
    samples = []
    for _ in range(runs):
        result = common.run_child(cmd, repo, env, timeout)
        if result['returncode'] != 0 or result['timed_out']:
            return {'module': module, 'error': common.tail(result['stderr'])}
        sample = parse_importtime(result['stderr'], module)
        sample['wall_s'] = result['time']
        samples.append(sample)
    median = {
        key: statistics.median(s[key] for s in samples)
        for key in ('total_s', 'wall_s', 'modules', 'cellacdc_modules')
    }
    targets = [s['target_s'] for s in samples if s['target_s'] is not None]
    median['target_s'] = statistics.median(targets) if targets else None
    return {'module': module, 'median': median, 'samples': samples}


def snapshot(repo, out, runs=3):
    repo = os.path.abspath(repo)
    data = {'meta': common.meta(repo), 'runs': runs, 'targets': {}}
    for label, candidates in TARGETS:
        module = common.existing_candidate(candidates, repo)
        if module is None:
            data['targets'][label] = {'error': f'none of {candidates}'}
            continue
        data['targets'][label] = measure(repo, module, runs)
        print(f"[importtime] {label}: {module}: "
              f"{data['targets'][label].get('median')}")
    common.write_json(os.path.join(out, SNAPSHOT_FILE), data)
    return data


def compare(before, after, tolerance=0.2):
    warnings_, rows = [], {}
    for label in before['targets']:
        b = before['targets'][label].get('median')
        a = after['targets'].get(label, {}).get('median')
        if not b or not a:
            warnings_.append(f'{label}: missing on one side')
            continue
        row = {}
        for key in ('target_s', 'total_s', 'wall_s', 'cellacdc_modules',
                    'modules'):
            if b.get(key) is None or a.get(key) is None:
                continue
            row[key] = [b[key], a[key]]
            if key.endswith('_s') and a[key] > b[key] * (1 + tolerance) + 0.1:
                warnings_.append(
                    f'{label}: {key} {b[key]:.3f} s -> {a[key]:.3f} s'
                )
        rows[label] = row
    return {'ok': True, 'warnings': warnings_, 'targets': rows}


def load_snapshot(folder):
    return common.read_json(os.path.join(folder, SNAPSHOT_FILE))


def run(commit, repo=None, before='baseline', runs=3):
    repo = repo or common.REPO_ROOT
    out = common.out_dir(commit)
    snapshot(repo, out, runs)
    if not before or before == commit:
        return None
    before_dir = common.out_dir(before)
    if not os.path.exists(os.path.join(before_dir, SNAPSHOT_FILE)):
        return None
    report = compare(load_snapshot(before_dir), load_snapshot(out))
    common.write_json(os.path.join(out, 'importtime_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('--runs', type=int, default=3)
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--tolerance', type=float, default=0.2)
    # accepted for a uniform CLI; this comparison needs no mapping
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir')
    p.add_argument('--report')
    args = parser.parse_args(argv)
    if args.cmd == 'snapshot':
        snapshot(args.repo, args.out, args.runs)
        return 0
    report = compare(load_snapshot(args.before), load_snapshot(args.after),
                     args.tolerance)
    path = args.report or os.path.join(args.after, 'importtime_compare.json')
    common.write_json(path, report)
    for label, row in report['targets'].items():
        print(f'[importtime] {label}: {row}')
    for line in report['warnings']:
        print('WARNING:', line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
