"""Node ids of the test suite (``pytest --collect-only``), for a before/after diff.

Usage::

    python tools/refactor/verify/collect_ids.py snapshot --out <dir> [--repo <tree>]
        [--timeout 300]
    python tools/refactor/verify/collect_ids.py compare --before <dir> --after <dir>
        [--changes <changes.jsonl> ...]

``snapshot`` runs ``python -m pytest --collect-only -q -p no:cacheprovider``
from the tree's root (collection only, ``QT_QPA_PLATFORM=offscreen``; it
takes about 60-100 s, hence the long default timeout) and stores the node
ids, the collection errors and pytest's summary line in
``collect_ids.json`` (plus ``collect_ids.txt``, one id per line).

``compare`` reports node ids that disappeared (errors, unless named in a
``changes.jsonl``), new ones (info, e.g. the new tests of the refactor) and
new collection errors (errors).
"""
import argparse
import os
import sys

try:
    from . import common
except ImportError:
    import common

SNAPSHOT_FILE = 'collect_ids.json'


def snapshot(repo, out, timeout=300, extra_args=()):
    repo = os.path.abspath(repo)
    cmd = [
        sys.executable, '-m', 'pytest', '--collect-only', '-q',
        '-p', 'no:cacheprovider', *extra_args,
    ]
    result = common.run_child(cmd, repo, common.child_env(repo), timeout)
    ids, errors, summary = [], [], ''
    for line in result['stdout'].splitlines():
        text = line.strip()
        if not text:
            continue
        if '::' in text and not text.startswith(('ERROR', 'FAILED', '=')):
            ids.append(text)
        elif text.startswith('ERROR'):
            errors.append(text)
        elif 'collected' in text or 'error' in text.lower():
            summary = text
    data = {
        'meta': common.meta(repo), 'cmd': cmd[1:],
        'returncode': result['returncode'], 'timed_out': result['timed_out'],
        'time': result['time'], 'count': len(ids), 'summary': summary,
        'errors': errors, 'ids': ids,
    }
    if result['returncode'] not in (0, 5) or result['timed_out']:
        data['stderr_tail'] = common.tail(result['stderr'], 30)
        data['stdout_tail'] = common.tail(result['stdout'], 30)
    common.write_json(os.path.join(out, SNAPSHOT_FILE), data)
    with open(os.path.join(out, 'collect_ids.txt'), 'w', encoding='utf-8',
              newline='\n') as file:
        file.write('\n'.join(ids) + '\n')
    print(f"[collect_ids] {len(ids)} ids, {len(errors)} errors, "
          f"rc={result['returncode']}, {result['time']} s: {summary}")
    return data


def compare(before, after, changes=frozenset()):
    bids, aids = set(before['ids']), set(after['ids'])
    lost = sorted(bids - aids)
    new = sorted(aids - bids)
    excused = [i for i in lost if i in changes]
    lost = [i for i in lost if i not in changes]
    new_errors = sorted(set(after['errors']) - set(before['errors']))
    errors = [f'lost test: {i}' for i in lost]
    errors += [f'new collection error: {e}' for e in new_errors]
    if after.get('timed_out') or after.get('returncode') not in (0, 5):
        errors.append(f"collection failed (rc={after.get('returncode')})")
    return {
        'ok': not errors, 'errors': errors,
        'summary': {
            'before': len(bids), 'after': len(aids), 'lost': len(lost),
            'new': len(new), 'excused': len(excused),
            'new_errors': len(new_errors),
        },
        'lost': lost, 'new': new, 'excused': excused,
        'new_errors': new_errors,
    }


def load_snapshot(folder):
    return common.read_json(os.path.join(folder, SNAPSHOT_FILE))


def run(commit, repo=None, before='baseline', changes=None):
    repo = repo or common.REPO_ROOT
    out = common.out_dir(commit)
    snapshot(repo, out)
    if not before or before == commit:
        return None
    before_dir = common.out_dir(before)
    if not os.path.exists(os.path.join(before_dir, SNAPSHOT_FILE)):
        return None
    listed = common.load_changes(
        changes if changes is not None else common.changes_files_up_to(commit)
    )
    report = compare(load_snapshot(before_dir), load_snapshot(out), listed)
    common.write_json(os.path.join(out, 'collect_ids_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('--timeout', type=int, default=300)
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--changes', action='append')
    # accepted for a uniform CLI; this comparison needs no mapping
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir')
    p.add_argument('--report')
    args = parser.parse_args(argv)
    if args.cmd == 'snapshot':
        data = snapshot(args.repo, args.out, args.timeout)
        return 0 if data['returncode'] in (0, 5) else 1
    changes = args.changes or [os.path.join(args.after, 'changes.jsonl')]
    report = compare(load_snapshot(args.before), load_snapshot(args.after),
                     common.load_changes(changes))
    path = args.report or os.path.join(args.after, 'collect_ids_compare.json')
    common.write_json(path, report)
    print(f"[collect_ids] {report['summary']}")
    for line in report['errors'][:40]:
        print('ERROR:', line)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
