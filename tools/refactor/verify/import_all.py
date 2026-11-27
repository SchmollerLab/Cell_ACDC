"""Import every cellacdc module alone, each in a fresh interpreter.

Usage::

    python tools/refactor/verify/import_all.py snapshot --out <dir> [--repo <tree>]
        [--mode gui] [--mode noqt] [--mode server] [--workers 8] [--timeout 60]
    python tools/refactor/verify/import_all.py compare --before <dir> --after <dir>
        [--mapping-table <name-table.json>] [--mapping-dir <shards>]

Modules come from a static listing of the tree (``common.importable_modules``;
regular packages only) minus the one exclusion list of the plan
(``common.is_import_excluded``: dev scripts, docs, resources, ``_profile``,
``_deprecated``, plugin sub-packages, scripts that run at import). Each one is
imported with ``python -c`` from the tree's root (so the tree comes first on
``sys.path``), ``QT_QPA_PLATFORM=offscreen``, stdin closed, on a pool of
`workers` processes with a hard `timeout` each (``--include-plugins`` also
imports the plugin sub-packages, whose optional dependencies are mostly not
installed). Modes:

- ``gui``: plain import;
- ``noqt``: a meta path finder makes ``qtpy``/``PyQt5``/``PyQt6``/``PySide*``
  imports fail (the headless baseline: which modules import without Qt);
- ``server``: ``noqt`` plus ``--server`` in ``sys.argv`` (model server mode).

Recorded per module: success, the exception (last line and a traceback
tail), whether stderr shows a traceback even on success, the import time,
the ``cellacdc`` modules and the third-party top-level packages in
``sys.modules`` afterwards, and a global-state snapshot: pyqtgraph's
``imageAxisOrder`` and config options, the keys of pyqtgraph's
``GradientEditorItem.Gradients``, ``numpy.geterr()``, the warnings filters
and a few pandas display options (each only when the library was imported).

``compare`` maps old module names to new ones (``common.Mapping``) and
reports: modules that imported before and fail after (error), new modules
that fail (error), lost modules (error unless a shim remains at the old
path), and, for review, the ``sys.modules`` differences of the entry modules
(``common.ENTRY_MODULES``) and the global-state differences.
"""
import argparse
import json
import os
import sys

try:
    from . import common
except ImportError:
    import common

MODES = ('gui', 'noqt', 'server')
QT_DISABLED_MARK = '(Qt disabled by import_all)'
#: Errors of modules that need Qt, in the noqt/server modes: the blocker's
#: own message, or pyqtgraph's when it finds no binding (it imports the
#: bindings itself and raises before qtpy is reached)
QT_MISSING_MARKS = (QT_DISABLED_MARK, 'PyQtGraph requires one of')

CHILD_CODE = r'''
import importlib, json, os, sys, time, traceback, warnings
MOD = os.environ['ACDC_VERIFY_MODULE']
MODE = os.environ['ACDC_VERIFY_MODE']
OUT = os.environ['ACDC_VERIFY_OUT']
QT_BINDINGS = %(qt)r
if MODE in ('noqt', 'server'):
    from importlib.abc import MetaPathFinder

    class NoQt(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in QT_BINDINGS:
                raise ModuleNotFoundError(
                    f'No module named {fullname!r} (Qt disabled by import_all)'
                )
            return None

    sys.meta_path.insert(0, NoQt())
if MODE == 'server':
    sys.argv.append('--server')
result = {'module': MOD, 'mode': MODE}
start = time.perf_counter()
try:
    importlib.import_module(MOD)
    result['ok'] = True
except BaseException as err:
    result['ok'] = False
    result['error'] = (type(err).__name__ + ': ' + str(err)).strip()[:500]
    result['traceback'] = ''.join(traceback.format_exception(
        type(err), err, err.__traceback__
    ))[-3000:]
result['time'] = round(time.perf_counter() - start, 3)
mods = sys.modules
result['cellacdc_modules'] = sorted(
    m for m in mods if m == 'cellacdc' or m.startswith('cellacdc.')
)
stdlib = set(getattr(sys, 'stdlib_module_names', ()))
result['third_party'] = sorted({
    m.split('.')[0] for m in mods
    if m.split('.')[0] not in stdlib and not m.startswith('_')
    and m.split('.')[0] not in ('cellacdc', '__main__', 'builtins')
})
state = {}
try:
    if 'pyqtgraph' in mods:
        pg = mods['pyqtgraph']
        state['pg_imageAxisOrder'] = repr(pg.getConfigOption('imageAxisOrder'))
        state['pg_config'] = {
            k: repr(v) for k, v in sorted(pg.CONFIG_OPTIONS.items())
        }
        gem = mods.get('pyqtgraph.graphicsItems.GradientEditorItem')
        if gem is not None:
            state['pg_gradients'] = sorted(gem.Gradients.keys())
    if 'numpy' in mods:
        state['np_geterr'] = mods['numpy'].geterr()
    if 'pandas' in mods:
        pd = mods['pandas']
        state['pd_options'] = {
            k: repr(pd.get_option(k)) for k in (
                'display.max_columns', 'display.max_rows',
                'display.expand_frame_repr', 'mode.chained_assignment',
            )
        }
    state['warnings_filters'] = [
        '|'.join([
            str(action), getattr(msg, 'pattern', '') or '',
            getattr(cat, '__name__', str(cat)),
            getattr(mod, 'pattern', '') or '', str(lineno)
        ])
        for action, msg, cat, mod, lineno in warnings.filters
    ]
except BaseException as err:
    state['error'] = repr(err)
result['state'] = state
with open(OUT, 'w', encoding='utf-8') as file:
    json.dump(result, file)
'''


def list_modules(repo, include_plugins=False):
    modules = common.importable_modules(repo)
    selected = []
    excluded = []
    for name, (path, is_pkg) in sorted(modules.items()):
        excluded_ = common.is_import_excluded(name, is_pkg)
        if excluded_ and include_plugins and common.is_plugin_module(name):
            excluded_ = False
        if excluded_:
            excluded.append(name)
        else:
            selected.append(name)
    return selected, excluded


def _import_one(args):
    repo, module, mode, timeout, tmpdir = args
    out = os.path.join(tmpdir, f'{mode}__{module}.json')
    env = common.child_env(repo, {
        'ACDC_VERIFY_MODULE': module, 'ACDC_VERIFY_MODE': mode,
        'ACDC_VERIFY_OUT': out,
    })
    code = CHILD_CODE % {'qt': common.QT_BINDINGS}
    proc = common.run_child(
        [sys.executable, '-c', code], repo, env, timeout
    )
    result = None
    if os.path.exists(out):
        try:
            result = common.read_json(out)
        except Exception:
            result = None
        os.remove(out)
    if result is None:
        result = {
            'module': module, 'mode': mode, 'ok': False,
            'error': 'timeout' if proc['timed_out'] else (
                f"child exited with {proc['returncode']} without a result"
            ),
        }
    result['returncode'] = proc['returncode']
    result['timed_out'] = proc['timed_out']
    result['wall'] = proc['time']
    stderr = proc['stderr']
    result['stderr_traceback'] = 'Traceback (most recent call last)' in stderr
    if not result.get('ok') or result['stderr_traceback']:
        result['stderr_tail'] = common.tail(stderr, 12)
    return result


def snapshot(repo, out, modes=MODES, workers=8, timeout=60, modules=None,
             include_plugins=False):
    repo = os.path.abspath(repo)
    selected, excluded = list_modules(repo, include_plugins)
    if modules:
        selected = [m for m in selected if m in modules]
    tmpdir = common.scratch_dir('import_all_' + str(os.getpid()))
    summaries = {}
    for mode in modes:
        jobs = [(repo, m, mode, timeout, tmpdir) for m in selected]
        results = common.parallel(_import_one, jobs, workers)
        data = {
            'meta': common.meta(repo), 'mode': mode, 'timeout': timeout,
            'excluded': excluded,
            'results': {r['module']: r for r in results},
        }
        path = os.path.join(out, f'import_all_{mode}.json')
        common.write_json(path, data, indent=None)
        failed = sorted(r['module'] for r in results if not r.get('ok'))
        summaries[mode] = {
            'modules': len(results), 'ok': len(results) - len(failed),
            'failed': failed,
            'timeouts': sorted(r['module'] for r in results
                               if r.get('timed_out')),
            'stderr_tracebacks': sorted(
                r['module'] for r in results
                if r.get('ok') and r.get('stderr_traceback')
            ),
            'excluded': len(excluded),
        }
        print(f'[import_all] {mode}: {len(results)} modules, '
              f'{len(failed)} failed -> {path}')
    common.write_json(os.path.join(out, 'import_all_summary.json'), summaries)
    return summaries


# --- compare ----------------------------------------------------------------

def _map_set(names, mapping, packages=()):
    return {mapping.map_module(n, n in packages) for n in names}


def compare_mode(before, after, mapping):
    errors, warnings_, info = [], [], []
    b_results, a_results = before['results'], after['results']
    mode = after.get('mode') or before.get('mode')
    b_by_new = {}
    for name in b_results:
        b_by_new.setdefault(mapping.map_module(name), []).append(name)
    matched_after = set()
    for new, olds in sorted(b_by_new.items()):
        for old in olds:
            b = b_results[old]
            a = a_results.get(new) or a_results.get(old)
            if a is None:
                if old in a_results:
                    a = a_results[old]
                else:
                    errors.append(f'lost module: {old} (-> {new})')
                    continue
            matched_after.add(a['module'])
            if b.get('ok') and not a.get('ok'):
                errors.append(
                    f"now fails: {old} -> {a['module']}: {a.get('error')}"
                )
            elif not b.get('ok') and a.get('ok'):
                info.append(f"now imports: {old} -> {a['module']}")
            elif not b.get('ok') and not a.get('ok') and (
                b.get('error') != a.get('error')
            ):
                info.append(
                    f"still fails, other error: {old}: {b.get('error')} -> "
                    f"{a.get('error')}"
                )
            if b.get('ok') and a.get('ok') and not b.get(
                'stderr_traceback'
            ) and a.get('stderr_traceback'):
                warnings_.append(f"traceback printed now: {a['module']}")
            bstate, astate = b.get('state', {}), a.get('state', {})
            for key in sorted(set(bstate) | set(astate)):
                if bstate.get(key) != astate.get(key):
                    warnings_.append(
                        f"state {key} differs after importing {a['module']}"
                    )
    for name, a in sorted(a_results.items()):
        if name in matched_after:
            continue
        if not a.get('ok'):
            if mode in ('noqt', 'server') and any(
                    mark in str(a.get('error')) for mark in QT_MISSING_MARKS):
                # a new module holding Qt code (headless users are checked
                # by headless.py and by the old modules above)
                info.append(f"new module needs Qt: {name}")
            else:
                errors.append(f"new module fails: {name}: {a.get('error')}")
        else:
            info.append(f'new module: {name}')
    entries = {}
    for label, candidates in common.ENTRY_MODULES:
        b = next((b_results[c] for c in candidates if c in b_results), None)
        a = next((a_results[c] for c in candidates if c in a_results), None)
        if b is None or a is None:
            continue
        bset = _map_set(b.get('cellacdc_modules', []), mapping)
        aset = _map_set(a.get('cellacdc_modules', []), mapping)
        entries[label] = {
            'before': b['module'], 'after': a['module'],
            'cellacdc_only_before': sorted(bset - aset),
            'cellacdc_only_after': sorted(aset - bset),
            'third_party_only_before': sorted(
                set(b.get('third_party', [])) - set(a.get('third_party', []))
            ),
            'third_party_only_after': sorted(
                set(a.get('third_party', [])) - set(b.get('third_party', []))
            ),
            'time': [b.get('time'), a.get('time')],
        }
    return {
        'ok': not errors, 'errors': errors, 'warnings': warnings_,
        'info': info, 'entry_modules': entries,
    }


def compare(before_dir, after_dir, mapping, modes=MODES):
    report = {'ok': True, 'modes': {}}
    for mode in modes:
        bpath = os.path.join(before_dir, f'import_all_{mode}.json')
        apath = os.path.join(after_dir, f'import_all_{mode}.json')
        if not (os.path.exists(bpath) and os.path.exists(apath)):
            continue
        res = compare_mode(common.read_json(bpath), common.read_json(apath),
                           mapping)
        report['modes'][mode] = res
        report['ok'] = report['ok'] and res['ok']
    return report


def run(commit, repo=None, before='baseline', table=None, mapping_dir=None,
        modes=MODES):
    repo = repo or common.REPO_ROOT
    out = common.out_dir(commit)
    snapshot(repo, out, modes)
    if not before or before == commit:
        return None
    mapping = common.Mapping.load(
        table or common.default_name_table(commit),
        mapping_dir or common.MAPPING_DIR
    )
    report = compare(common.out_dir(before), out, mapping, modes)
    common.write_json(os.path.join(out, 'import_all_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('--mode', action='append', choices=MODES)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--module', action='append',
                   help='only these modules (default: all)')
    p.add_argument('--include-plugins', action='store_true',
                   help='also import the plugin sub-packages (models, ...)')
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir', default=common.MAPPING_DIR)
    p.add_argument('--report')
    p = sub.add_parser('list')
    p.add_argument('--repo', default=common.REPO_ROOT)
    args = parser.parse_args(argv)
    if args.cmd == 'list':
        selected, excluded = list_modules(args.repo)
        print(json.dumps({'selected': selected, 'excluded': excluded},
                         indent=1))
        return 0
    if args.cmd == 'snapshot':
        summaries = snapshot(
            args.repo, args.out, args.mode or MODES, args.workers,
            args.timeout, args.module, args.include_plugins
        )
        print(json.dumps(summaries, indent=1))
        return 0
    mapping = common.Mapping.load(args.mapping_table, args.mapping_dir)
    report = compare(args.before, args.after, mapping)
    path = args.report or os.path.join(args.after, 'import_all_compare.json')
    common.write_json(path, report)
    for mode, res in report['modes'].items():
        print(f"[import_all] {mode}: {len(res['errors'])} errors, "
              f"{len(res['warnings'])} warnings, {len(res['info'])} info")
        for line in res['errors'][:30]:
            print('  ERROR:', line)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
