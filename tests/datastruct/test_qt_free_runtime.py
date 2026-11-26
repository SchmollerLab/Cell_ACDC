"""Runtime Qt-free checks in child interpreters.

(a) Qt blocked: `cellacdc.acdc_bioio_bioformats` imported first (failed on
    main with a circular import), then its submodules and every existing
    `cellacdc.acdc_datastruct` module (found at run time, so modules added
    later are covered).
(b) Qt blocked: `cellacdc.colors` imported first.
(c) Qt blocked: `cellacdc.data_cli` imported first, then `main(['--help'])`
    (when the module exists).
(d) Server mode (`--server`, the configuration of worker children):
    `cellacdc.acdc_bioio_bioformats` and the child-side modules import.
(e) Qt installed: `import cellacdc, cellacdc.apps, cellacdc.dataStruct,
    cellacdc.dataReStruct` works.

(a)-(c) also check GUI_INSTALLED is False and that no GUI module was loaded.
"""
from __future__ import annotations

import os

from .conditions import (
    REPO_ROOT, blocked_imports_prelude, requires_qt, run_python,
    subprocess_heavy
)

import pytest

pytestmark = subprocess_heavy

GUI_MODULES = (
    'cellacdc.apps', 'cellacdc.widgets', 'cellacdc.workers', 'cellacdc._main',
    'cellacdc.dataStruct', 'cellacdc.dataReStruct',
)
BIOIO_SUBMODULES = ('reader', '_utils', 'install', 'raw_metadata', 'writer')

HELPERS = f'''
import importlib, importlib.util, pkgutil, sys

def module_exists(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

def import_datastruct_modules():
    if not module_exists('cellacdc.acdc_datastruct'):
        return []
    package = importlib.import_module('cellacdc.acdc_datastruct')
    names = sorted(
        info.name for info in pkgutil.iter_modules(package.__path__)
    )
    for name in names:
        importlib.import_module(f'cellacdc.acdc_datastruct.{{name}}')
    return names

def check_headless():
    import cellacdc
    assert cellacdc.GUI_INSTALLED is False, 'GUI_INSTALLED is True'
    loaded = sorted(m for m in {GUI_MODULES!r} if m in sys.modules)
    assert not loaded, f'GUI modules loaded: {{loaded}}'
'''


def _run(code: str) -> str:
    result = run_python(code)
    assert result.returncode == 0, (
        f'child exited with {result.returncode}\n'
        f'--- stdout ---\n{result.stdout[-3000:]}\n'
        f'--- stderr ---\n{result.stderr[-6000:]}'
    )
    return result.stdout


def test_a_bioio_and_datastruct_import_without_qt():
    stdout = _run(blocked_imports_prelude() + HELPERS + f'''
import cellacdc.acdc_bioio_bioformats
for name in {BIOIO_SUBMODULES!r}:
    importlib.import_module(f'cellacdc.acdc_bioio_bioformats.{{name}}')
print('DATASTRUCT', import_datastruct_modules())
check_headless()
print('OK')
''')
    assert stdout.splitlines()[-1] == 'OK'


def test_b_colors_imports_without_qt():
    stdout = _run(blocked_imports_prelude() + HELPERS + '''
import cellacdc.colors
check_headless()
print('OK')
''')
    assert stdout.splitlines()[-1] == 'OK'


def test_c_data_cli_help_without_qt():
    if not os.path.isfile(os.path.join(REPO_ROOT, 'cellacdc', 'data_cli.py')):
        pytest.skip('cellacdc/data_cli.py does not exist yet')
    stdout = _run(blocked_imports_prelude() + HELPERS + '''
import cellacdc.data_cli
returncode = cellacdc.data_cli.main(['--help'])
assert returncode == 0, f'main(["--help"]) returned {returncode!r}'
check_headless()
print('OK')
''')
    assert stdout.splitlines()[-1] == 'OK'


def test_d_server_mode_child_imports():
    stdout = _run(HELPERS + f'''
sys.argv.append('--server')
import cellacdc
assert not cellacdc.GUI_INSTALLED, 'server mode must not load the GUI'
import cellacdc.acdc_bioio_bioformats
for name in {BIOIO_SUBMODULES!r}:
    importlib.import_module(f'cellacdc.acdc_bioio_bioformats.{{name}}')
if module_exists('cellacdc.acdc_datastruct.tasks'):
    importlib.import_module('cellacdc.acdc_datastruct.tasks')
print('OK')
''')
    assert stdout.splitlines()[-1] == 'OK'


@requires_qt
def test_e_gui_modules_import_with_qt():
    stdout = _run(
        'import cellacdc, cellacdc.apps, cellacdc.dataStruct, '
        'cellacdc.dataReStruct\n'
        'assert cellacdc.GUI_INSTALLED\n'
        "print('OK')\n"
    )
    assert stdout.splitlines()[-1] == 'OK'
