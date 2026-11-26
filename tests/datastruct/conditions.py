"""Shared skip conditions and environment helpers for tests/datastruct.

Every skip condition of the data-structure tests comes from here, so no test
depends on pytest markers registered elsewhere:

    requires_qt        a Qt binding and qtpy are importable
    requires_bioio     bioio and bioio-tifffile are importable
    slow               runs only with ACDC_SLOW_TESTS set
    subprocess_heavy   child-spawning tests: on CI only on Python 3.12 (or
                       with ACDC_SLOW_TESTS); locally they always run

Helpers: `hooks_env` (environment for worker children with the test hooks),
`tree_snapshot` ("nothing changed" assertions), `qapp` (offscreen
QApplication), `blocked_imports_prelude` and `run_python` (subprocess checks).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from typing import Iterable, Mapping

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
TEST_HOOKS_ENV = 'ACDC_DATASTRUCT_TEST_HOOKS'
READER_FACTORY_ENV = 'ACDC_DATASTRUCT_READER_FACTORY'
SLOW_ENV = 'ACDC_SLOW_TESTS'
QT_BINDINGS = ('PyQt5', 'PyQt6', 'PySide2', 'PySide6')
QT_MODULES = ('qtpy',) + QT_BINDINGS + ('pyqtgraph',)


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def qt_available() -> bool:
    """qtpy and at least one Qt binding can be imported (not imported here)."""
    return _importable('qtpy') and any(_importable(b) for b in QT_BINDINGS)


def bioio_available() -> bool:
    """bioio and bioio-tifffile can be imported (not imported here)."""
    return _importable('bioio') and _importable('bioio_tifffile')


def slow_enabled() -> bool:
    return bool(os.environ.get(SLOW_ENV))


def subprocess_heavy_skipped() -> bool:
    """True on CI for Python != 3.12 unless ACDC_SLOW_TESTS is set."""
    return bool(
        os.environ.get('CI') and sys.version_info[:2] != (3, 12)
        and not slow_enabled()
    )


requires_qt = pytest.mark.skipif(
    not qt_available(), reason='needs qtpy and a Qt binding'
)
requires_bioio = pytest.mark.skipif(
    not bioio_available(), reason='needs bioio and bioio-tifffile'
)
slow = pytest.mark.skipif(
    not slow_enabled(), reason=f'slow test: set {SLOW_ENV}=1'
)
subprocess_heavy = pytest.mark.skipif(
    subprocess_heavy_skipped(),
    reason='child-spawning test: on CI only on Python 3.12 '
           f'(set {SLOW_ENV}=1 to force)'
)


def hooks_env(**extra: str) -> dict[str, str]:
    """Environment for children that may use the test hooks.

    os.environ copy + ACDC_DATASTRUCT_TEST_HOOKS=1, the repo root first on
    PYTHONPATH (so `cellacdc` and `tests.datastruct` resolve to this
    checkout), QT_QPA_PLATFORM=offscreen, then `extra` (values str()-ed),
    e.g. hooks_env(ACDC_DATASTRUCT_READER_FACTORY=fake_bioio.FAKE_BIOIMAGE_PATH).
    """
    env = dict(os.environ)
    old = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]
    old = [p for p in old if os.path.normcase(os.path.abspath(p))
           != os.path.normcase(REPO_ROOT)]
    env['PYTHONPATH'] = os.pathsep.join([REPO_ROOT] + old)
    env[TEST_HOOKS_ENV] = '1'
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env.update({key: str(value) for key, value in extra.items()})
    return env


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(*roots: str) -> dict[str, tuple]:
    """Snapshot of every entry under `roots` for "nothing changed" checks.

    Keys are POSIX paths relative to the common parent of the roots (so
    'src/a.tif', 'dst/Position_1/Images/x.tif'; absolute POSIX paths if the
    roots share no parent). Values: files (size, mtime_ns, sha256); folders
    (None, None, 'dir'); symlinks (None, None, 'link:<target>'). A missing
    root contributes no keys, so creating it shows up as new keys.
    """
    abs_roots = [os.path.abspath(root) for root in roots]
    try:
        base = os.path.commonpath([os.path.dirname(r) for r in abs_roots])
    except ValueError:
        base = None

    def key_of(path: str) -> str:
        rel = path if base is None else os.path.relpath(path, base)
        return rel.replace(os.sep, '/')

    snapshot = {}
    for root in abs_roots:
        if not os.path.lexists(root):
            continue
        if os.path.islink(root) or not os.path.isdir(root):
            snapshot[key_of(root)] = _entry(root)
            continue
        snapshot[key_of(root)] = (None, None, 'dir')
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames + filenames:
                path = os.path.join(dirpath, name)
                snapshot[key_of(path)] = _entry(path)
    return dict(sorted(snapshot.items()))


def _entry(path: str) -> tuple:
    if os.path.islink(path):
        return (None, None, f'link:{os.readlink(path)}')
    if os.path.isdir(path):
        return (None, None, 'dir')
    stat = os.stat(path)
    return (stat.st_size, stat.st_mtime_ns, _sha256(path))


def qapp():
    """The QApplication (created offscreen if needed); skips without Qt."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('qtpy')
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def blocked_imports_prelude(modules: Iterable[str] = QT_MODULES) -> str:
    """Python source that makes importing `modules` (and their submodules)
    raise ModuleNotFoundError; prepend it to a `python -c` script."""
    names = tuple(modules)
    return (
        'import sys, importlib.abc\n'
        'class _AcdcBlocker(importlib.abc.MetaPathFinder):\n'
        f'    blocked = {names!r}\n'
        '    def find_spec(self, name, path=None, target=None):\n'
        "        if name.split('.')[0] in self.blocked:\n"
        "            raise ModuleNotFoundError(f'blocked by test: {name}', "
        'name=name)\n'
        '        return None\n'
        'sys.meta_path.insert(0, _AcdcBlocker())\n'
    )


def run_python(code: str, *args: str, env: Mapping[str, str] | None = None,
               cwd: str | None = None, timeout: float = 180
               ) -> subprocess.CompletedProcess:
    """Run `python -c code *args` with this interpreter; text output captured
    (UTF-8, errors replaced). `env` defaults to `hooks_env()`, `cwd` to the
    repo root."""
    return subprocess.run(
        [sys.executable, '-c', code, *args],
        env=dict(hooks_env() if env is None else env),
        cwd=REPO_ROOT if cwd is None else cwd,
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=timeout,
    )
