"""Headless (no Qt) entry points: what must keep working without a GUI stack.

Usage::

    python tools/refactor/verify/headless.py snapshot --out <dir> [--repo <tree>]
    python tools/refactor/verify/headless.py compare --before <dir> --after <dir>
        [--mapping-table <name-table.json>] [--mapping-dir <shards>]

Each check runs in a fresh interpreter from the tree's root with a meta path
finder that makes every Qt binding import fail (as ``NoQt`` in
``tests/test_model_server.py``), optionally with ``--server`` in
``sys.argv`` (model server mode), and records success, the exit code, the
error, the ``cellacdc`` modules loaded and whether a GUI module leaked in
(``GUI_MODULES``, old and new names). The checks (``CHECKS``) cover the plan's
headless entry points: ``import cellacdc`` (also in server mode),
``cellacdc.myutils``, the model server worker, the lineage tree integrity CLI
(``python -m cellacdc.lineage_tree_integrity <empty folder>``), the
``acdc --version`` CLI start (skipped when a non-editable ``cellacdc`` folder
sits in site-packages: ``cellacdc/__main__.py`` deletes it when run from a
checkout) and the library modules used without a GUI.

The snapshot also copies the summary of an ``import_all`` NoQt sweep found in
the same output folder (``import_all_noqt.json``: which modules import
without Qt), the headless baseline of the whole package.

``compare`` (old names mapped through ``common.Mapping``): checks that passed
before and fail after, and new GUI-module leaks, are errors; changes of the
loaded ``cellacdc`` modules are listed for review.
"""
import argparse
import json
import os
import site
import sys
import tempfile

try:
    from . import common
except ImportError:
    import common

SNAPSHOT_FILE = 'headless.json'

# Modules that must never be imported without Qt (old and new layout)
GUI_MODULES = (
    'cellacdc.widgets', 'cellacdc.apps', 'cellacdc.gui', 'cellacdc.workers',
    'cellacdc.measurements_viewer', 'cellacdc._main', 'cellacdc.dataPrep',
    'cellacdc.segm', 'cellacdc.shared.widgets', 'cellacdc.shared.dialogs',
    'cellacdc.shared.workers', 'cellacdc.launcher', 'cellacdc.data_prep',
    'cellacdc.segmentation_module',
)

# (name, server mode, kind, target, argv)
CHECKS = (
    ('import_cellacdc', False, 'import', 'cellacdc', None),
    ('import_cellacdc_server', True, 'import', 'cellacdc', None),
    ('myutils', False, 'import', 'cellacdc.myutils', None),
    ('myutils_server', True, 'import', 'cellacdc.myutils', None),
    ('service_worker', True, 'import', 'cellacdc.service.worker', None),
    ('service_client', False, 'import', 'cellacdc.service.client', None),
    ('load', False, 'import', 'cellacdc.load', None),
    ('core', False, 'import', 'cellacdc.core', None),
    ('cli', False, 'import', 'cellacdc.cli', None),
    ('measurements', False, 'import', 'cellacdc.measurements', None),
    ('timepoint_predictors', False, 'import', 'cellacdc.timepoint_predictors',
     None),
    ('background_values', False, 'import', 'cellacdc.background_values',
     None),
    ('lineage_tree_integrity_cli', False, 'run_module',
     'cellacdc.lineage_tree_integrity', ['{tmpdir}']),
    # console script entry point `acdc --version` (cellacdc/__main__.py has
    # no `if __name__ == '__main__'` block)
    ('cli_version', False, 'call', 'cellacdc.__main__:run', ['--version']),
)

CHILD_CODE = r'''
import json, os, runpy, sys, traceback
from importlib.abc import MetaPathFinder
QT_BINDINGS = %(qt)r
SPEC = json.loads(os.environ['ACDC_VERIFY_HEADLESS'])

class NoQt(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in QT_BINDINGS:
            raise ModuleNotFoundError(f'No module named {fullname!r} (NoQt)')
        return None

sys.meta_path.insert(0, NoQt())
result = {'ok': False}
argv = [SPEC['target'].split(':')[0]] + list(SPEC['argv'] or [])
if SPEC['server']:
    argv.append('--server')
sys.argv = argv
try:
    if SPEC['kind'] == 'import':
        import importlib
        importlib.import_module(SPEC['target'])
    elif SPEC['kind'] == 'call':
        import importlib
        modname, _, funcname = SPEC['target'].partition(':')
        getattr(importlib.import_module(modname), funcname)()
    else:
        runpy.run_module(SPEC['target'], run_name='__main__', alter_sys=True)
    result['ok'] = True
    result['exit_code'] = 0
except SystemExit as err:
    code = err.code
    result['exit_code'] = code if isinstance(code, int) else (
        0 if code is None else 1
    )
    result['ok'] = result['exit_code'] == 0
except BaseException as err:
    result['error'] = (type(err).__name__ + ': ' + str(err))[:500]
    result['traceback'] = traceback.format_exc()[-2500:]
cellacdc = sys.modules.get('cellacdc')
result['GUI_INSTALLED'] = getattr(cellacdc, 'GUI_INSTALLED', None)
result['RUNNING_AS_SERVER'] = getattr(cellacdc, 'RUNNING_AS_SERVER', None)
result['cellacdc_modules'] = sorted(
    m for m in sys.modules if m == 'cellacdc' or m.startswith('cellacdc.')
)
with open(os.environ['ACDC_VERIFY_OUT'], 'w', encoding='utf-8') as file:
    json.dump(result, file)
sys.stdout.flush()
os._exit(0)
'''


def _site_packages_copy():
    paths = []
    try:
        paths = site.getsitepackages()
    except Exception:
        pass
    return any(
        os.path.isdir(os.path.join(p, 'cellacdc')) for p in paths
        if p.endswith('site-packages')
    )


def _run_check(job):
    repo, check, tmp, timeout = job
    name, server, kind, target, argv = check
    out = os.path.join(tmp, name + '.json')
    workdir = tempfile.mkdtemp(prefix='acdc_headless_', dir=tmp)
    argv = [a.replace('{tmpdir}', workdir) for a in (argv or [])]
    env = common.child_env(repo, {
        'ACDC_VERIFY_HEADLESS': json.dumps({
            'server': server, 'kind': kind, 'target': target, 'argv': argv,
        }),
        'ACDC_VERIFY_OUT': out,
    })
    code = CHILD_CODE % {'qt': common.QT_BINDINGS}
    proc = common.run_child([sys.executable, '-c', code], repo, env, timeout)
    result = None
    if os.path.exists(out):
        result = common.read_json(out)
        os.remove(out)
    if result is None:
        result = {'ok': False, 'error': 'timeout' if proc['timed_out'] else (
            f"no result (rc={proc['returncode']})"
        )}
    result.update({
        'name': name, 'server': server, 'kind': kind, 'target': target,
        'time': proc['time'], 'stdout_tail': common.tail(proc['stdout'], 8),
    })
    if not result.get('ok'):
        result['stderr_tail'] = common.tail(proc['stderr'], 15)
    result['gui_leaks'] = sorted(
        m for m in result.get('cellacdc_modules', [])
        if any(m == g or m.startswith(g + '.') for g in GUI_MODULES)
    )
    return result


def snapshot(repo, out, workers=8, timeout=120):
    repo = os.path.abspath(repo)
    tmp = common.scratch_dir('headless_' + str(os.getpid()))
    checks = list(CHECKS)
    skipped = []
    if _site_packages_copy():
        skipped.append('cli_version')
        checks = [c for c in checks if c[0] != 'cli_version']
    results = common.parallel(
        _run_check, [(repo, c, tmp, timeout) for c in checks], workers
    )
    data = {
        'meta': common.meta(repo), 'skipped': skipped,
        'checks': {r['name']: r for r in results},
    }
    noqt = os.path.join(out, 'import_all_noqt.json')
    if os.path.exists(noqt):
        sweep = common.read_json(noqt)['results']
        data['noqt_sweep'] = {
            'ok': sorted(m for m, r in sweep.items() if r.get('ok')),
            'failed': sorted(m for m, r in sweep.items() if not r.get('ok')),
        }
    common.write_json(os.path.join(out, SNAPSHOT_FILE), data)
    for r in results:
        print(f"[headless] {r['name']}: ok={r.get('ok')} "
              f"exit={r.get('exit_code')} leaks={r['gui_leaks']} "
              f"{r.get('error', '')[:120]}")
    return data


def compare(before, after, mapping):
    errors, info = [], []
    for name, b in before['checks'].items():
        a = after['checks'].get(name)
        if a is None:
            if name not in after.get('skipped', []):
                errors.append(f'{name}: check missing')
            continue
        if b.get('ok') and not a.get('ok'):
            errors.append(f"{name}: now fails: {a.get('error') or a.get('exit_code')}")
        elif not b.get('ok') and a.get('ok'):
            info.append(f'{name}: now passes')
        new_leaks = sorted(set(a['gui_leaks']) - {
            mapping.map_module(m) for m in b['gui_leaks']
        } - set(b['gui_leaks']))
        if new_leaks:
            errors.append(f'{name}: GUI modules now loaded: {new_leaks}')
        bmods = {mapping.map_module(m) for m in b.get('cellacdc_modules', [])}
        amods = {mapping.map_module(m) for m in a.get('cellacdc_modules', [])}
        if bmods != amods:
            info.append({
                'check': name, 'only_before': sorted(bmods - amods),
                'only_after': sorted(amods - bmods),
            })
    sweep_b = before.get('noqt_sweep', {}).get('ok', [])
    sweep_a = set(after.get('noqt_sweep', {}).get('ok', []))
    lost = sorted(
        m for m in sweep_b
        if mapping.map_module(m) not in sweep_a and m not in sweep_a
    )
    if sweep_b and sweep_a and lost:
        errors.append(f'no longer import without Qt: {lost}')
    return {'ok': not errors, 'errors': errors, 'info': info}


def load_snapshot(folder):
    return common.read_json(os.path.join(folder, SNAPSHOT_FILE))


def run(commit, repo=None, before='baseline', table=None, mapping_dir=None):
    repo = repo or common.REPO_ROOT
    out = common.out_dir(commit)
    snapshot(repo, out)
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
    common.write_json(os.path.join(out, 'headless_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('--timeout', type=int, default=120)
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir', default=common.MAPPING_DIR)
    p.add_argument('--report')
    args = parser.parse_args(argv)
    if args.cmd == 'snapshot':
        snapshot(args.repo, args.out, timeout=args.timeout)
        return 0
    mapping = common.Mapping.load(args.mapping_table, args.mapping_dir)
    report = compare(load_snapshot(args.before), load_snapshot(args.after),
                     mapping)
    path = args.report or os.path.join(args.after, 'headless_compare.json')
    common.write_json(path, report)
    for line in report['errors']:
        print('ERROR:', line)
    print(f"[headless] {len(report['errors'])} errors, "
          f"{len(report['info'])} info")
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
