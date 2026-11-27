"""File lists of the sdist and the wheel built from a clean worktree.

Usage::

    python tools/refactor/verify/wheel_diff.py snapshot --out <dir> [--repo <checkout>]
        [--commit HEAD] [--isolation] [--timeout 1800]
    python tools/refactor/verify/wheel_diff.py compare --before <dir> --after <dir>
        [--mapping-table <name-table.json>] [--mapping-dir <shards>]

``snapshot`` adds a detached git worktree of `commit` in a scratch folder
(committed files only, as in a release), runs ``python -m build --outdir
<dist> <worktree>`` there (sdist, then the wheel from the sdist), lists the
archive members and removes the worktree. By default the build runs
``--no-isolation --skip-dependency-check`` with the current environment's
setuptools/setuptools_scm (offline, faster); ``--isolation`` builds in an
isolated environment (downloads the build requirements). When the ``build``
package is missing it is installed with ``pip install --target`` into a
scratch folder put on ``PYTHONPATH``. A build takes minutes: run it in the
background.

``compare`` maps the before members through the module moves
(``common.Mapping``: shards ``module_moves`` and the name table's modules,
also folder moves such as the premade configurations) and reports members
that are missing after (errors) and new members (info, grouped into new
``__init__.py`` files, compat modules and other files). Version numbers in
archive paths (``cellacdc-<version>.dist-info``) are normalized.
"""
import argparse
import os
import re
import shutil
import sys
import tarfile
import zipfile

try:
    from . import common
except ImportError:
    import common

SNAPSHOT_FILE = 'wheel.json'
VERSIONED = re.compile(r'^cellacdc-[^/]+?\.(dist-info|data)/')


def _ensure_build_module(env):
    probe = common.run_child(
        [sys.executable, '-c', 'import build, pyproject_hooks'],
        common.scratch_dir('probe'), env, 60
    )
    if probe['returncode'] == 0:
        return env
    target = common.scratch_dir('pybuild')
    if not os.path.exists(os.path.join(target, 'build')):
        result = common.run_child(
            [sys.executable, '-m', 'pip', 'install', '--target', target,
             'build'], common.scratch_dir('probe'), env, 600
        )
        if result['returncode'] != 0:
            raise RuntimeError('pip install build failed:\n'
                               + common.tail(result['stderr']))
    env = dict(env)
    env['PYTHONPATH'] = os.pathsep.join(
        [target] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
    )
    return env


def _normalize(member):
    member = member.replace('\\', '/')
    match = VERSIONED.match(member)
    if match:
        member = f'cellacdc-<version>.{match.group(1)}/' + member[
            match.end():
        ]
    return member


def list_wheel(path):
    with zipfile.ZipFile(path) as archive:
        return sorted(_normalize(n) for n in archive.namelist()
                      if not n.endswith('/'))


def list_sdist(path):
    with tarfile.open(path) as archive:
        names = [m.name for m in archive.getmembers() if m.isfile()]
    stripped = []
    for name in names:
        parts = name.replace('\\', '/').split('/', 1)
        stripped.append(parts[1] if len(parts) == 2 else parts[0])
    return sorted(stripped)


def snapshot(repo, out, commit='HEAD', isolation=False, timeout=1800,
             keep=False):
    repo = os.path.abspath(repo)
    sha = common.git(repo, 'rev-parse', commit).strip()
    base = common.scratch_dir('wheel')
    worktree = os.path.join(base, 'wt_' + sha[:12])
    dist = os.path.join(base, 'dist_' + sha[:12])
    if os.path.exists(worktree):
        common.git(repo, 'worktree', 'remove', '--force', worktree,
                   check=False)
        shutil.rmtree(worktree, ignore_errors=True)
    shutil.rmtree(dist, ignore_errors=True)
    common.git(repo, 'worktree', 'add', '--detach', worktree, sha,
               timeout=300)
    try:
        env = common.child_env(worktree, offscreen=False)
        # the worktree must not be shadowed by the checkout on PYTHONPATH
        env['PYTHONPATH'] = os.environ.get('PYTHONPATH', '')
        if not env['PYTHONPATH']:
            env.pop('PYTHONPATH')
        env = _ensure_build_module(env)
        cmd = [sys.executable, '-m', 'build', '--outdir', dist]
        if not isolation:
            cmd += ['--no-isolation', '--skip-dependency-check']
        cmd.append(worktree)
        result = common.run_child(cmd, worktree, env, timeout)
        wheels = [f for f in os.listdir(dist) if f.endswith('.whl')] if (
            os.path.isdir(dist)) else []
        sdists = [f for f in os.listdir(dist) if f.endswith('.tar.gz')] if (
            os.path.isdir(dist)) else []
        data = {
            'meta': common.meta(repo), 'commit': sha,
            'isolation': isolation, 'returncode': result['returncode'],
            'timed_out': result['timed_out'], 'build_time': result['time'],
            'wheel': wheels[0] if wheels else None,
            'sdist': sdists[0] if sdists else None,
            'wheel_members': list_wheel(os.path.join(dist, wheels[0]))
            if wheels else [],
            'sdist_members': list_sdist(os.path.join(dist, sdists[0]))
            if sdists else [],
            'log_tail': common.tail(result['stdout'] + '\n'
                                    + result['stderr'], 25),
        }
    finally:
        if not keep:
            common.git(repo, 'worktree', 'remove', '--force', worktree,
                       check=False)
            shutil.rmtree(worktree, ignore_errors=True)
    common.write_json(os.path.join(out, SNAPSHOT_FILE), data)
    print(f"[wheel_diff] rc={data['returncode']} {data['wheel']} "
          f"({len(data['wheel_members'])} files), {data['sdist']} "
          f"({len(data['sdist_members'])} files), {data['build_time']} s")
    return data


def map_member(path, mapping):
    """Path of a before member in the target layout (None: unchanged)."""
    for old, new in sorted(mapping.file_moves.items(), key=lambda x: -len(
        x[0]
    )):
        if path == old:
            return new
        if not old.endswith('.py') and path.startswith(old.rstrip('/') + '/'):
            return new.rstrip('/') + path[len(old.rstrip('/')):]
    if path.endswith('.py'):
        mod = common.module_name(path)
        if mod in mapping.modules and not path.endswith('__init__.py'):
            return mapping.modules[mod].replace('.', '/') + '.py'
    return None


def _compare_lists(before, after, mapping, prefix=''):
    after_set = set(after)
    errors, matched = [], set()
    for path in before:
        rel = path[len(prefix):] if prefix and path.startswith(prefix) else (
            path
        )
        new = map_member(rel, mapping)
        candidates = [path]
        if new is not None:
            candidates.insert(0, prefix + new if prefix else new)
            # a moved module may now be a package
            if new.endswith('.py'):
                candidates.insert(1, (prefix + new[:-3] if prefix else new[:-3])
                                  + '/__init__.py')
        hit = next((c for c in candidates if c in after_set), None)
        if hit is None:
            errors.append(f'missing: {path}' + (f' (-> {new})' if new else ''))
        else:
            matched.add(hit)
            if path in after_set:
                matched.add(path)  # compat shim left at the old path
    new_members = sorted(after_set - matched)
    groups = {'init': [], 'compat': [], 'other': []}
    for path in new_members:
        base = path.rsplit('/', 1)[-1]
        if base == '__init__.py':
            groups['init'].append(path)
        elif base.startswith('_compat'):
            groups['compat'].append(path)
        else:
            groups['other'].append(path)
    return errors, groups


def compare(before, after, mapping):
    report = {'ok': True, 'errors': [], 'archives': {}}
    for kind in ('wheel', 'sdist'):
        errors, groups = _compare_lists(
            before[f'{kind}_members'], after[f'{kind}_members'], mapping
        )
        report['archives'][kind] = {
            'before': len(before[f'{kind}_members']),
            'after': len(after[f'{kind}_members']),
            'missing': errors, 'new': groups,
        }
        report['errors'].extend(f'{kind}: {e}' for e in errors)
    report['ok'] = not report['errors']
    return report


def load_snapshot(folder):
    return common.read_json(os.path.join(folder, SNAPSHOT_FILE))


def run(commit, repo=None, before='baseline', table=None, mapping_dir=None,
        git_commit='HEAD'):
    repo = repo or common.REPO_ROOT
    out = common.out_dir(commit)
    snapshot(repo, out, git_commit)
    if not before or before == commit:
        return None
    before_dir = common.out_dir(before)
    if not os.path.exists(os.path.join(before_dir, SNAPSHOT_FILE)):
        return None
    mapping = common.Mapping.load(
        table or common.default_name_table(commit),
        mapping_dir or common.MAPPING_DIR
    )
    report = compare(load_snapshot(before_dir), load_snapshot(out), mapping)
    common.write_json(os.path.join(out, 'wheel_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('--commit', default='HEAD')
    p.add_argument('--isolation', action='store_true')
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--keep', action='store_true',
                   help='keep the worktree (debugging)')
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir', default=common.MAPPING_DIR)
    p.add_argument('--report')
    args = parser.parse_args(argv)
    if args.cmd == 'snapshot':
        data = snapshot(args.repo, args.out, args.commit, args.isolation,
                        args.timeout, args.keep)
        return 0 if data['returncode'] == 0 else 1
    mapping = common.Mapping.load(args.mapping_table, args.mapping_dir)
    report = compare(load_snapshot(args.before), load_snapshot(args.after),
                     mapping)
    path = args.report or os.path.join(args.after, 'wheel_compare.json')
    common.write_json(path, report)
    for kind, res in report['archives'].items():
        print(f"[wheel_diff] {kind}: {res['before']} -> {res['after']} files, "
              f"{len(res['missing'])} missing, new: "
              f"{ {k: len(v) for k, v in res['new'].items()} }")
    for line in report['errors'][:40]:
        print('ERROR:', line)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
