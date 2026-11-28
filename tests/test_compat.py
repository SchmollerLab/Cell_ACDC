"""Old import paths keep working (``cellacdc/_compat.py``).

The package layout refactor moved most modules (CODEMAP.md). External code
(plugins, spotMAX, users' scripts and notebooks) may still use the old
paths: they resolve to the identical objects with a ``MovedWarning`` (a
``DeprecationWarning`` whose message starts with ``Cell-ACDC import path
moved:``, which pytest turns into an error for Cell-ACDC's own code and
tests, see ``pyproject.toml``).

- Synthetic package tests: a copy of ``cellacdc/_compat.py`` in a
  temporary package with small tables (fast, in this process).
- Real tables (``cellacdc/_compat_moves.py``), in a fresh interpreter:
  every entry resolves to the object of its home module and warns at the
  caller, 1:1 moves are ``sys.modules`` aliases, the documented plugin API
  of ``cellacdc.myutils`` does not warn and ``import cellacdc`` imports no
  old module (the old modules have no files: the import hook of
  ``_compat.py`` serves them, ``OLD_MODULES``).

This is the one test file allowed to use old paths
(``tests/test_no_old_paths.py`` exempts it).
"""
import importlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import textwrap
import warnings
from unittest import mock

import pytest

import cellacdc.gui as gui
import cellacdc.measurements_viewer as measurements_viewer
import cellacdc.myutils as myutils
from cellacdc.gui.main_window import constants
from cellacdc.measurements_viewer import viewer_utils
from cellacdc.measurements_viewer.main_window import (
    configuration as viewer_configuration,
    data_io as viewer_data_io,
    image_panel as viewer_image_panel,
)
from cellacdc.myutils import filesystem
from tests import source_scan
from tests.source_scan import REPO_ROOT

pytestmark = pytest.mark.infra


@pytest.fixture(scope='module', autouse=True)
def _release_parsed_sources():
    """The parsed package takes hundreds of MB: release it after the
    module (the session keeps running other tests)."""
    yield
    source_scan.clear_caches()


PREFIX = 'Cell-ACDC import path moved: '
COMPAT_SOURCE = REPO_ROOT / 'cellacdc' / '_compat.py'

COMPAT_MOVES = '''
MODULE_ALIASES = {'{pkg}.oldmod': '{pkg}.sub.newmod'}
# 'real' has a file: the import hook must never shadow it
OLD_MODULES = {'{pkg}.oldmod': 'alias', '{pkg}.split': 'split',
               '{pkg}.real': 'split'}
NAME_MOVES = {
    '{pkg}.split': {
        'Thing': ('{pkg}.home', 'Thing'),
        'VALUE': ('{pkg}.home', 'VALUE'),
        'mod': ('{pkg}.home', None),
        'api': ('{pkg}.home', 'api'),
    },
    '{pkg}.sub': {
        'Old': ('{pkg}.home', 'Thing'),
        'api': ('{pkg}.home', 'api'),
        'VALUE': ('{pkg}.home', 'VALUE'),
    },
}
PLUGIN_API = {'{pkg}.sub': frozenset({'api'})}
PACKAGE_SUBMODULES = {'{pkg}.sub': ('newmod',)}
FORWARDERS = {'{pkg}.fwd': '{pkg}.split'}
PATCH_FANOUT = {'{pkg}.split': ('{pkg}.copy', '{pkg}.home')}
'''

# The old modules oldmod and split have no files: the import hook installed
# by the package serves them (as cellacdc/__init__.py does)
FILES = {
    '__init__.py': 'from {pkg} import _compat\n_compat.install()\n',
    'real.py': 'REAL = True\n',
    '_compat_moves.py': COMPAT_MOVES,
    'home.py': (
        'class Thing:\n    pass\n\nVALUE = 1\n\n\ndef api():\n'
        '    return 42\n'
    ),
    # Holds a copy of the old global VALUE (as the codemod copies
    # import-bound names into every module that holds old code)
    'copy.py': 'from {pkg}.home import VALUE\n\n\ndef read():\n'
               '    return VALUE\n',
    'sub/__init__.py': (
        'from {pkg} import _compat\n'
        '__getattr__, __dir__ = _compat.package_compat(__name__)\n'
    ),
    'sub/newmod.py': 'X = 5\n',
    'fwd.py': (
        'OWN = 1\nfrom {pkg} import _compat\n'
        "__getattr__, __dir__ = _compat.forward_module(__name__, '{pkg}.split')\n"
    ),
}
_PACKAGE_COUNTER = itertools.count()


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A fresh synthetic package using a copy of cellacdc/_compat.py."""
    name = f'acdc_compat_test_{next(_PACKAGE_COUNTER)}'
    root = tmp_path / name
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.replace('{pkg}', name), encoding='utf-8')
    (root / '_compat.py').write_text(
        COMPAT_SOURCE.read_text(encoding='utf-8'), encoding='utf-8'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield name
    for module in list(sys.modules):
        if module == name or module.startswith(name + '.'):
            del sys.modules[module]
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if getattr(finder, 'package', None) != name
    ]


def module(name):
    return importlib.import_module(name)


def moved_warning(pkg):
    return module(f'{pkg}._compat').MovedWarning


def recorded_moves(record, pkg=None):
    category = moved_warning(pkg) if pkg else DeprecationWarning
    return [
        w for w in record if issubclass(w.category, category)
        and str(w.message).startswith(PREFIX)
    ]


def test_alias_module_is_the_module_at_the_new_path(pkg):
    with pytest.warns(moved_warning(pkg)) as record:
        old = module(f'{pkg}.oldmod')
    new = module(f'{pkg}.sub.newmod')
    assert old is new
    assert sys.modules[f'{pkg}.oldmod'] is new
    assert [str(w.message) for w in record] == [
        f'{PREFIX}{pkg}.oldmod -> {pkg}.sub.newmod (see CODEMAP.md)'
    ]
    # Reported at the code that used the old path
    assert record[0].filename == __file__


def hooks_of(pkg):
    return [f for f in sys.meta_path if getattr(f, 'package', None) == pkg]


def test_old_modules_without_files_come_from_the_import_hook(pkg):
    module(pkg)
    hooks = hooks_of(pkg)
    assert len(hooks) == 1
    assert type(hooks[0]) is module(f'{pkg}._compat').OldPathFinder
    # Appended last: the regular finders are asked first
    assert sys.meta_path.index(hooks[0]) > 0
    for old in ('oldmod', 'split'):
        spec = importlib.util.find_spec(f'{pkg}.{old}')
        assert spec is not None and spec.origin is None
        assert type(spec.loader).__name__ == '_OldPathLoader'
    split = module(f'{pkg}.split')
    assert split.__cellacdc_compat__ is True
    assert 'moved to other modules' in split.__doc__


def test_import_hook_never_shadows_files_nor_invents_modules(pkg):
    real = module(f'{pkg}.real')
    assert real.REAL is True and real.__file__.endswith('real.py')
    with pytest.raises(ModuleNotFoundError):
        module(f'{pkg}.not_an_old_module')


def test_install_adds_the_import_hook_once(pkg):
    compat = module(f'{pkg}._compat')
    first = hooks_of(pkg)
    assert compat.install() is first[0]
    assert hooks_of(pkg) == first


def test_moved_names_resolve_lazily_through_their_home(pkg):
    split = module(f'{pkg}.split')
    with pytest.warns(moved_warning(pkg)) as record:
        thing = split.Thing
    home = module(f'{pkg}.home')
    assert thing is home.Thing
    assert str(record[0].message) == (
        f'{PREFIX}{pkg}.split.Thing -> {pkg}.home.Thing (see CODEMAP.md)'
    )
    assert record[0].filename == __file__
    with pytest.warns(moved_warning(pkg)):
        assert split.mod is home
        # Never cached: changes of the home are visible
        home.VALUE = 3
        assert split.VALUE == 3
        assert 'VALUE' not in vars(split)
        from_import = {}
        exec(f'from {pkg}.split import Thing', from_import)
    assert from_import['Thing'] is home.Thing


def test_unknown_names_and_dunders_raise_without_importing(pkg):
    split = module(f'{pkg}.split')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        for name in ('nope', '__path__', '__wrapped__', '__test__'):
            with pytest.raises(AttributeError):
                getattr(split, name)
        # e.g. pytest and inspect probe modules with hasattr
        assert not hasattr(split, '__path__')
    assert f'{pkg}.home' not in sys.modules
    assert not recorded_moves(record)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert sorted(split.__all__) == ['Thing', 'VALUE', 'api', 'mod']
        assert {'Thing', 'VALUE', 'api', 'mod'} <= set(dir(split))


def test_package_compat_imports_submodules_and_forwards_old_names(pkg):
    sub = module(f'{pkg}.sub')
    home = module(f'{pkg}.home')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert sub.newmod.X == 5
        # The documented plugin API is forwarded without a warning
        assert sub.api() == 42
    assert not recorded_moves(record)
    with pytest.warns(moved_warning(pkg), match=r'sub\.Old -> .*home\.Thing'):
        assert sub.Old is home.Thing
    with pytest.raises(AttributeError):
        sub.nope
    with pytest.raises(AttributeError):
        sub.__wrapped__
    assert {'newmod', 'Old', 'api'} <= set(dir(sub))


def test_forward_module_keeps_its_own_names(pkg):
    fwd = module(f'{pkg}.fwd')
    home = module(f'{pkg}.home')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert fwd.OWN == 1
    assert not recorded_moves(record)
    with pytest.warns(moved_warning(pkg)):
        assert fwd.Thing is home.Thing


def test_patches_of_old_names_reach_the_home_and_its_copies(pkg):
    split = module(f'{pkg}.split')
    home = module(f'{pkg}.home')
    copy = module(f'{pkg}.copy')
    with pytest.warns(moved_warning(pkg)):
        split.VALUE = 7
    assert home.VALUE == 7 and copy.VALUE == 7 and copy.read() == 7
    assert 'VALUE' not in vars(split)
    with pytest.warns(moved_warning(pkg)):
        del split.VALUE
    # Deleting a name that a pending patch set undoes that patch (the undo
    # of mock.patch and of pytest >= 9.2's monkeypatch)
    assert home.VALUE == 1 and copy.VALUE == 1
    with pytest.warns(moved_warning(pkg)):
        del split.VALUE
    # Without a pending patch the name is deleted
    assert not hasattr(home, 'VALUE') and not hasattr(copy, 'VALUE')
    with pytest.warns(moved_warning(pkg)):
        split.VALUE = 1
    # Setting the name again restores the deleted copies too
    assert home.VALUE == 1 and copy.VALUE == 1
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        split.local_only = 3
    # Unknown names stay local, without a warning
    assert vars(split)['local_only'] == 3
    assert not recorded_moves(record)


def test_deleting_patched_names_undoes_the_patches_in_order(pkg):
    """pytest >= 9.2 undoes ``monkeypatch.setattr`` of a name that is not
    in the module's ``__dict__`` (moved names never are) with ``delattr``,
    once per patch; older pytest sets the previous values again."""
    split = module(f'{pkg}.split')
    home = module(f'{pkg}.home')
    copy = module(f'{pkg}.copy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        split.VALUE = 5
        split.VALUE = 6
        assert home.VALUE == 6 and copy.VALUE == 6
        del split.VALUE
        assert home.VALUE == 5 and copy.VALUE == 5
        del split.VALUE
        assert home.VALUE == 1 and copy.read() == 1
        # the undo of older pytest
        split.VALUE = 5
        split.VALUE = 6
        split.VALUE = 5
        split.VALUE = 1
        assert home.VALUE == 1 and copy.VALUE == 1
        # no patch is pending any more: deleting deletes
        del split.VALUE
        assert not hasattr(home, 'VALUE') and not hasattr(copy, 'VALUE')
    assert 'VALUE' not in vars(split)


def test_mock_patch_round_trip_through_an_old_path(pkg):
    split = module(f'{pkg}.split')
    home = module(f'{pkg}.home')
    copy = module(f'{pkg}.copy')
    with pytest.warns(moved_warning(pkg)):
        with mock.patch(f'{pkg}.split.VALUE', 9):
            assert home.VALUE == 9 and copy.read() == 9
    assert home.VALUE == 1 and copy.VALUE == 1
    assert 'VALUE' not in vars(split)
    with pytest.warns(moved_warning(pkg)):
        with mock.patch.object(split, 'Thing', 'patched'):
            assert home.Thing == 'patched'
    assert isinstance(home.Thing, type)


def test_monkeypatch_through_old_paths(pkg):
    split = module(f'{pkg}.split')
    sub = module(f'{pkg}.sub')
    home = module(f'{pkg}.home')
    copy = module(f'{pkg}.copy')
    patcher = pytest.MonkeyPatch()
    with pytest.warns(moved_warning(pkg)):
        patcher.setattr(split, 'VALUE', 5)
        patcher.setattr(split, 'VALUE', 6, raising=False)
        # A package that replaces an old module (gui, myutils, ...)
        patcher.setattr(sub, 'Old', 'old')
    patcher.setattr(split, 'brand_new', 5, raising=False)
    assert home.VALUE == 6 and copy.VALUE == 6 and home.Thing == 'old'
    assert vars(split)['brand_new'] == 5
    with pytest.warns(moved_warning(pkg)):
        patcher.undo()
    assert home.VALUE == 1 and copy.VALUE == 1
    assert isinstance(home.Thing, type)
    assert 'brand_new' not in vars(split) and 'VALUE' not in vars(split)
    assert 'Old' not in vars(sub)


# ---------------------------------------------------- the real compat tables
CHECK_TABLES = '''
import importlib
import json
import sys
import warnings

from cellacdc import _compat
from cellacdc._compat_moves import (
    FORWARDERS, MODULE_ALIASES, NAME_MOVES, PLUGIN_API
)

errors = []
checked = 0
forwarded = 0
# Development scripts that run code at import (dialogs moved there)
DEV_SCRIPTS = {'cellacdc.test_segm_model'}


def moves(record):
    return [w for w in record if issubclass(w.category, _compat.MovedWarning)]


def check_warned(what, record, expected=True):
    warned = moves(record)
    if not expected:
        if warned:
            errors.append(f'{what}: warns {warned[0].message}')
        return
    if not warned:
        errors.append(f'{what}: no MovedWarning')
    elif any(w.filename != __file__ for w in warned):
        errors.append(f'{what}: warning reported at {warned[0].filename}')


for old, new in sorted(MODULE_ALIASES.items()):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        old_module = importlib.import_module(old)
    check_warned(old, record)
    if old_module is not importlib.import_module(new):
        errors.append(f'{old} is not {new}')
    checked += 1

for old, names in sorted(NAME_MOVES.items()):
    try:
        old_module = importlib.import_module(old)
    except Exception as err:
        errors.append(f'{old}: {err!r}')
        continue
    for name, (home, attr) in sorted(names.items()):
        what = f'{old}.{name}'
        if home in DEV_SCRIPTS:
            checked += 1
            continue
        # A name the module binds itself (e.g. an import of the package)
        bound = name in vars(old_module)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            try:
                value = getattr(old_module, name)
            except BaseException as err:
                errors.append(f'{what}: {err!r}')
                continue
        try:
            home_module = importlib.import_module(home)
            expected = home_module if attr is None else getattr(
                home_module, attr)
        except BaseException as err:
            errors.append(f'{what}: home {home}.{attr}: {err!r}')
            continue
        if value is not expected:
            errors.append(f'{what} is not {home}.{attr}')
        check_warned(
            what, record, not bound and name not in PLUGIN_API.get(old, ())
        )
        checked += 1

# Modules at new paths that shadow an old attribute path forward its names
# (cellacdc.gui.widgets.myMessageBox: cellacdc.gui used to bind widgets)
for forwarder, old in sorted(FORWARDERS.items()):
    module = importlib.import_module(forwarder)
    for name, (home, attr) in sorted(NAME_MOVES.get(old, {}).items()):
        if name in vars(module) or home in DEV_SCRIPTS:
            continue
        what = f'{forwarder}.{name}'
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            try:
                value = getattr(module, name)
            except BaseException as err:
                errors.append(f'{what}: {err!r}')
                continue
        if value is not getattr(importlib.import_module(old), name, None):
            errors.append(f'{what} is not {old}.{name}')
        check_warned(what, record)
        forwarded += 1

with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter('always')
    from cellacdc.gui import guiWin
check_warned('from cellacdc.gui import guiWin', record)
from cellacdc.gui.main_window import window
if guiWin is not window.guiWin:
    errors.append('cellacdc.gui.guiWin is not the window class')
with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter('always')
    import cellacdc.gui_navigation
import cellacdc.gui.mixins.navigation
if cellacdc.gui_navigation is not cellacdc.gui.mixins.navigation:
    errors.append('cellacdc.gui_navigation is not cellacdc.gui.mixins.navigation')

print(json.dumps(
    {'checked': checked, 'forwarded': forwarded, 'errors': errors}
))
'''

CHECK_IMPORT_CELLACDC = '''
import json
import sys
import warnings

warnings.simplefilter('error', DeprecationWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        message='(?!Cell-ACDC import path moved)')
import cellacdc

old_modules = set(json.loads(sys.argv[1]))
print(json.dumps(sorted(old_modules & set(sys.modules))))
'''


def run_python(tmp_path, script, *args, timeout=300):
    path = tmp_path / 'check_compat.py'
    path.write_text(textwrap.dedent(script), encoding='utf-8')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    env['PYTHONPATH'] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in [env.get('PYTHONPATH')] if p]
    )
    proc = subprocess.run(
        [sys.executable, '-W', 'always', str(path), *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-5000:]
    return path, json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.slow  # imports every home module in a fresh interpreter
def test_every_old_name_resolves_to_its_home_with_a_warning(tmp_path):
    _, result = run_python(tmp_path, CHECK_TABLES)
    moves = source_scan.compat_moves()
    expected = len(moves['MODULE_ALIASES']) + sum(
        len(names) for names in moves['NAME_MOVES'].values()
    )
    assert not result['errors'], '\n'.join(result['errors'])
    assert result['checked'] == expected
    assert result['forwarded'] > 0


@pytest.mark.slow  # starts a fresh interpreter
def test_import_cellacdc_imports_no_old_module(tmp_path):
    old_modules = sorted(source_scan.old_modules())
    assert 'cellacdc.apps' in old_modules
    assert 'cellacdc.gui_navigation' in old_modules
    _, loaded = run_python(
        tmp_path, CHECK_IMPORT_CELLACDC, json.dumps(old_modules)
    )
    assert loaded == []


def test_plugin_api_of_myutils_is_in_the_tables():
    """The plugin API (names that custom models, trackers and promptable
    models use as ``myutils.<name>``) is forwarded without a warning; the
    fresh-interpreter test above checks that they resolve."""
    moves = source_scan.compat_moves()
    plugin_api = moves['PLUGIN_API']['cellacdc.myutils']
    documented = {'check_install_package', 'get_model_path', 'to_uint8'}
    assert documented <= plugin_api
    assert plugin_api <= set(moves['NAME_MOVES']['cellacdc.myutils'])


def test_old_names_of_the_packages_warn_and_forward_patches():
    """The packages that replaced old modules (gui, measurements_viewer,
    myutils) forward old names and patches to the home modules."""
    patcher = pytest.MonkeyPatch()
    try:
        with pytest.warns(DeprecationWarning, match=PREFIX) as record:
            assert gui.SliderSingleStepAdd is constants.SliderSingleStepAdd
        assert record[0].filename == __file__
        with pytest.warns(DeprecationWarning, match=PREFIX):
            patcher.setattr(myutils, 'listdir', 'patched')
            assert filesystem.listdir == 'patched'
            patcher.setattr(gui, 'SliderSingleStepAdd', 'patched')
            assert constants.SliderSingleStepAdd == 'patched'
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            patcher.undo()
    assert callable(filesystem.listdir)
    assert 'listdir' not in vars(myutils)
    assert 'SliderSingleStepAdd' not in vars(gui)


def test_patches_through_the_viewer_package_reach_every_copy():
    """A patch of a global of the old single-module viewer
    (``monkeypatch.setattr(measurements_viewer, name, ...)``) reaches every
    module that holds the viewer's code now (``PATCH_FANOUT``)."""
    name = 'get_channels_and_segm_endnames'
    original = vars(viewer_utils)[name]
    holders = [
        module for module in (
            viewer_utils, viewer_configuration, viewer_data_io,
            viewer_image_panel
        )
        if vars(module).get(name) is original
    ]
    assert len(holders) >= 2
    fake = object()
    patcher = pytest.MonkeyPatch()
    try:
        with pytest.warns(DeprecationWarning, match=PREFIX):
            patcher.setattr(measurements_viewer, name, fake)
        assert all(vars(module)[name] is fake for module in holders)
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            patcher.undo()
    assert all(vars(module)[name] is original for module in holders)
    assert name not in vars(measurements_viewer)
