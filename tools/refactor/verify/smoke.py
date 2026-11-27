"""Construct, show and close every window of Cell-ACDC (offscreen).

Usage::

    python tools/refactor/verify/smoke.py snapshot --out <dir> [--repo <tree>]
        [--target guiWin ...] [--no-utilities] [--workers 6] [--timeout 120]
    python tools/refactor/verify/smoke.py compare --before <dir> --after <dir>
        [--mapping-table <name-table.json>] [--mapping-dir <shards>]

Targets:

- the windows of ``common.WINDOW_CLASSES`` (launcher ``mainWin``, ``guiWin``
  with ``run()``, ``MeasurementsViewerWin``, ``dataPrepWin``,
  ``createDataStructWin`` and ``CreateSymLinkToPosWin`` without ``main()``
  (no Java VM, no folder dialog), ``segmWin``); new module paths first;
- every ``QWidget`` subclass defined in a ``cellacdc.utils.*`` module (the
  utility windows; base classes excluded), constructed with arguments chosen
  from the ``__init__`` signature (no experiment paths: ``{}``, the
  ``QApplication``, ``'Smoke test'`` for texts, ``None`` otherwise).

Each target runs in its own child process (``QT_QPA_PLATFORM=offscreen``,
pool of `workers`, hard `timeout`): the user's settings are redirected to a
temporary folder (``common.install_settings_isolation``: the conftest names,
every module-level path inside the settings folder, ``QSettings``, also for
modules imported later), blocking dialogs are answered by the test suite's
watchdog (``tests/conftest.py`` ``_DialogWatchdog``, a minimal one if it
cannot be imported), exceptions reaching ``sys.excepthook`` (Qt slots) are
recorded instead of aborting, and the launcher's ``checkConfigFiles`` (it
rewrites settings files) is skipped. Recorded per target: ok or the
exception, the auto-answered dialogs, the slot exceptions, top-level windows
still visible after closing, the time.

``compare`` reports targets that worked before and fail after (errors),
lost targets (errors) and changed dialogs/slot exceptions (warnings).
"""
import argparse
import inspect
import json
import os
import shutil
import sys
import time
import traceback

try:
    from . import common
except ImportError:
    import common

SNAPSHOT_FILE = 'smoke.json'
UTILITY_BASES = {'NewThreadMultipleExpBaseUtil', 'MainThreadSinglePosUtilBase'}
TEXT_PARAMS = {
    'title', 'infotext', 'progressdialoguetitle', 'module', 'info', 'text',
    'windowtitle', 'pbardesc',
}
PATH_PARAMS = {'exppaths', 'selectedexppaths', 'exp_paths', 'expPaths'}


# --- child ------------------------------------------------------------------

class _MiniWatchdog:
    """Answers (accepts, then closes) dialogs blocking a nested event loop."""

    def __init__(self):
        from qtpy.QtCore import QTimer
        self.answered = []
        self.timer = QTimer()
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.check)
        self.timer.start()

    def check(self):
        try:
            from qtpy.QtCore import QThread
            from qtpy.QtWidgets import QApplication, QDialog
            if QThread.currentThread().loopLevel() == 0:
                return
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QDialog) and widget.isVisible():
                    self.answered.append(
                        (type(widget).__name__, widget.windowTitle(), 'accept')
                    )
                    ok = getattr(widget, 'okButton', None)
                    if ok is not None and hasattr(ok, 'click'):
                        ok.click()
                    else:
                        widget.accept()
                    if widget.isVisible():
                        widget.close()
                    return
        except Exception:
            pass

    def stop(self):
        self.timer.stop()


def _utility_args(cls, app, tmpdir):
    args, used = [], []
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return args, used
    params = list(sig.parameters.values())[1:]
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not param.empty:
            continue
        name = param.name
        low = name.lower()
        if name in PATH_PARAMS or low in PATH_PARAMS:
            value, shown = {}, '{}'
        elif low == 'app':
            value, shown = app, 'app'
        elif low in TEXT_PARAMS:
            value, shown = 'Smoke test', "'Smoke test'"
        elif 'path' in low or 'folder' in low:
            value, shown = tmpdir, 'tmpdir'
        else:
            value, shown = None, 'None'
        args.append(value)
        used.append(f'{name}={shown}')
    return args, used


def _build(label, cls, app, tmpdir):
    if label == 'mainWin':
        cls.checkConfigFiles = lambda self: None
        win = cls(app)
        win.show()
        return win, []
    if label == 'guiWin':
        win = cls(app)
        win.run()
        return win, []
    if label in common.WINDOW_CLASSES:
        win = cls()
        win.show()
        return win, []
    args, used = _utility_args(cls, app, tmpdir)
    win = cls(*args)
    win.show()
    return win, used


def child_main(args):
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    result = {'target': args.label, 'ok': False}
    start = time.perf_counter()
    hook_errors = []

    def excepthook(etype, value, tb):
        hook_errors.append(
            ''.join(traceback.format_exception(etype, value, tb))[-1500:]
        )

    watchdog = None
    try:
        try:
            from tests import conftest
            Watchdog = conftest._DialogWatchdog
        except Exception as err:
            conftest = None
            Watchdog = _MiniWatchdog
            result['watchdog'] = f'minimal ({type(err).__name__})'
        import warnings
        warnings.simplefilter('ignore', DeprecationWarning)
        import cellacdc  # noqa: F401
        tmpdir = common.install_settings_isolation()
        result['settings_folder'] = tmpdir
        from qtpy.QtWidgets import QApplication, QStyleFactory
        app = QApplication.instance() or QApplication(['Cell-ACDC'])
        app.setStyle(QStyleFactory.create('Fusion'))
        app.toggle_dark_mode = False
        sys.excepthook = excepthook
        watchdog = Watchdog()
        from qtpy.QtTest import QTest
        cls = None
        errors = []
        for spec in json.loads(args.candidates):
            try:
                cls = common.resolve_attr(spec)
                result['class'] = spec
                break
            except Exception as err:
                errors.append(f'{spec}: {type(err).__name__}: {err}')
        if cls is None:
            raise ImportError('; '.join(errors))
        win, used = _build(args.label, cls, app, tmpdir)
        result['args'] = used
        QTest.qWait(300)
        result['title'] = win.windowTitle()
        win.close()
        QTest.qWait(200)
        result['visible_after_close'] = sorted(
            f'{type(w).__name__}: {w.windowTitle()}'
            for w in QApplication.topLevelWidgets()
            if w.isVisible() and w.isWindow()
        )
        result['ok'] = True
    except BaseException as err:
        result['error'] = f'{type(err).__name__}: {err}'[:500]
        result['traceback'] = traceback.format_exc()[-3000:]
    finally:
        if watchdog is not None:
            try:
                watchdog.stop()
            except Exception:
                pass
            result['dialogs'] = [list(a) for a in getattr(
                watchdog, 'answered', []
            )]
        result['slot_exceptions'] = hook_errors
        result['time'] = round(time.perf_counter() - start, 2)
        with open(args.child_out, 'w', encoding='utf-8') as file:
            json.dump(result, file)
        sys.stdout.flush()
        sys.stderr.flush()
        # Skip interpreter teardown (Qt threads/at-exit handlers can hang)
        os._exit(0)


def discover_main(args):
    """Child: list the QWidget subclasses defined in cellacdc.utils.*."""
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import importlib
    import warnings
    warnings.simplefilter('ignore')
    import cellacdc  # noqa: F401
    common.install_settings_isolation()
    from qtpy.QtWidgets import QApplication, QWidget
    app = QApplication.instance() or QApplication([])  # noqa: F841
    found, errors = [], {}
    modules = common.importable_modules(repo)
    for name in sorted(modules):
        if not name.startswith('cellacdc.utils.') or common.is_import_excluded(
            name, modules[name][1]
        ):
            continue
        try:
            module = importlib.import_module(name)
        except Exception as err:
            errors[name] = f'{type(err).__name__}: {err}'
            continue
        for attr, obj in vars(module).items():
            if (isinstance(obj, type) and issubclass(obj, QWidget)
                    and obj.__module__ == name
                    and obj.__name__ not in UTILITY_BASES):
                found.append(f'{name}:{obj.__qualname__}')
    with open(args.child_out, 'w', encoding='utf-8') as file:
        json.dump({'targets': sorted(found), 'errors': errors}, file)
    sys.stdout.flush()
    os._exit(0)


# --- parent -----------------------------------------------------------------

def _run_target(job):
    repo, label, candidates, timeout, tmp = job
    child_out = os.path.join(tmp, label.replace(':', '__') + '.json')
    cmd = [
        sys.executable, os.path.abspath(__file__), '_child', '--repo', repo,
        '--label', label, '--candidates', json.dumps(candidates),
        '--child-out', child_out,
    ]
    proc = common.run_child(cmd, repo, common.child_env(repo), timeout)
    result = None
    if os.path.exists(child_out):
        try:
            result = common.read_json(child_out)
        except Exception:
            result = None
        os.remove(child_out)
    if result is None:
        result = {
            'target': label, 'ok': False,
            'error': 'timeout' if proc['timed_out'] else (
                f"child exited with {proc['returncode']} without a result"
            ),
            'stderr_tail': common.tail(proc['stderr'], 20),
        }
    result['wall'] = proc['time']
    if 'Traceback (most recent call last)' in proc['stderr']:
        result['stderr_traceback'] = common.tail(proc['stderr'], 20)
    folder = result.pop('settings_folder', None)
    if folder and os.path.isdir(folder):
        written = []
        for root, _, names in os.walk(folder):
            for name in names:
                rel = os.path.relpath(os.path.join(root, name), folder)
                written.append(rel.replace(os.sep, '/'))
        result['settings_written'] = sorted(
            w for w in written if not w.startswith('logs/')
        )
        shutil.rmtree(folder, ignore_errors=True)
    return result


def _real_settings_folders(repo):
    """Folders that hold the real user settings (watched for changes)."""
    folders = [
        os.path.join(repo, '.acdc-settings'),
        os.path.join(os.path.expanduser('~'), 'acdc-appdata', '.acdc-settings'),
    ]
    return [f for f in folders if os.path.isdir(f)]


def _mtimes(folders):
    stamps = {}
    for folder in folders:
        for root, _, names in os.walk(folder):
            for name in names:
                path = os.path.join(root, name)
                try:
                    stamps[path] = os.stat(path).st_mtime_ns
                except OSError:
                    pass
    return stamps


def discover_utilities(repo, timeout=120):
    tmp = common.scratch_dir('smoke_' + str(os.getpid()))
    child_out = os.path.join(tmp, '_discover.json')
    cmd = [sys.executable, os.path.abspath(__file__), '_discover', '--repo',
           repo, '--child-out', child_out]
    proc = common.run_child(cmd, repo, common.child_env(repo), timeout)
    if not os.path.exists(child_out):
        raise RuntimeError('utility discovery failed:\n'
                           + common.tail(proc['stderr']))
    data = common.read_json(child_out)
    os.remove(child_out)
    return data


def snapshot(repo, out, targets=None, utilities=True, workers=6,
             timeout=120):
    repo = os.path.abspath(repo)
    jobs = []
    tmp = common.scratch_dir('smoke_' + str(os.getpid()))
    labels = targets or list(common.WINDOW_CLASSES)
    for label in labels:
        if label in common.WINDOW_CLASSES:
            jobs.append((repo, label, common.WINDOW_CLASSES[label], timeout,
                         tmp))
        elif ':' in label:
            jobs.append((repo, label, [label], timeout, tmp))
    discovery = None
    if utilities and not targets:
        discovery = discover_utilities(repo)
        for spec in discovery['targets']:
            jobs.append((repo, spec, [spec], timeout, tmp))
    watched = _real_settings_folders(repo)
    before_stamps = _mtimes(watched)
    results = common.parallel(_run_target, jobs, workers)
    after_stamps = _mtimes(watched)
    touched = sorted(
        path for path in set(before_stamps) | set(after_stamps)
        if before_stamps.get(path) != after_stamps.get(path)
    )
    data = {
        'meta': common.meta(repo), 'timeout': timeout,
        'discovery_errors': (discovery or {}).get('errors', {}),
        'real_settings_watched': watched,
        'real_settings_touched': touched,
        'results': {r['target']: r for r in results},
    }
    common.write_json(os.path.join(out, SNAPSHOT_FILE), data)
    failed = sorted(r['target'] for r in results if not r.get('ok'))
    print(f'[smoke] {len(results)} targets, {len(failed)} failed: {failed}')
    if touched:
        print(f'[smoke] WARNING: real settings files changed: {touched}')
    return data


def _canon_target(label, mapping):
    if ':' not in label:
        return label
    mod, _, qual = label.partition(':')
    newmod, newqual = mapping.map_def(mod, qual)
    return f'{newmod}:{newqual}'


def compare(before, after, mapping):
    errors, warnings_, info = [], [], []
    bres = {_canon_target(k, mapping): v for k, v in before['results'].items()}
    ares = {_canon_target(k, mapping): v for k, v in after['results'].items()}
    for key in sorted(set(bres) | set(ares)):
        b, a = bres.get(key), ares.get(key)
        if a is None:
            errors.append(f'lost target: {key}')
            continue
        if b is None:
            (info if a.get('ok') else warnings_).append(
                f"new target: {key}: {'ok' if a.get('ok') else a.get('error')}"
            )
            continue
        if b.get('ok') and not a.get('ok'):
            errors.append(f"now fails: {key}: {a.get('error')}")
        elif not b.get('ok') and a.get('ok'):
            info.append(f'now works: {key}')
        elif not b.get('ok') and b.get('error') != a.get('error'):
            warnings_.append(
                f"other error: {key}: {b.get('error')} -> {a.get('error')}"
            )
        bd = [tuple(d[:2]) for d in b.get('dialogs', [])]
        ad = [tuple(d[:2]) for d in a.get('dialogs', [])]
        if bd != ad:
            warnings_.append(f'dialogs differ: {key}: {bd} -> {ad}')
        if len(b.get('slot_exceptions', [])) != len(
            a.get('slot_exceptions', [])
        ):
            warnings_.append(
                f"slot exceptions: {key}: {len(b.get('slot_exceptions', []))}"
                f" -> {len(a.get('slot_exceptions', []))}"
            )
    if after.get('real_settings_touched'):
        errors.append(
            f"real settings files changed: {after['real_settings_touched']}"
        )
    return {'ok': not errors, 'errors': errors, 'warnings': warnings_,
            'info': info}


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
    common.write_json(os.path.join(out, 'smoke_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('--target', action='append',
                   help='a WINDOW_CLASSES label or module:Class')
    p.add_argument('--no-utilities', action='store_true')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--timeout', type=int, default=120)
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir', default=common.MAPPING_DIR)
    p.add_argument('--report')
    p = sub.add_parser('_child')
    p.add_argument('--repo', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--child-out', required=True)
    p = sub.add_parser('_discover')
    p.add_argument('--repo', required=True)
    p.add_argument('--child-out', required=True)
    args = parser.parse_args(argv)
    if args.cmd == '_child':
        return child_main(args)
    if args.cmd == '_discover':
        return discover_main(args)
    if args.cmd == 'snapshot':
        snapshot(args.repo, args.out, args.target, not args.no_utilities,
                 args.workers, args.timeout)
        return 0
    mapping = common.Mapping.load(args.mapping_table, args.mapping_dir)
    report = compare(load_snapshot(args.before), load_snapshot(args.after),
                     mapping)
    path = args.report or os.path.join(args.after, 'smoke_compare.json')
    common.write_json(path, report)
    for line in report['errors']:
        print('ERROR:', line)
    for line in report['warnings']:
        print('WARNING:', line)
    print(f"[smoke] {len(report['errors'])} errors, "
          f"{len(report['warnings'])} warnings")
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
