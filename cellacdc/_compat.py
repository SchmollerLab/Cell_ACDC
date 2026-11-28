"""Compatibility layer for moved Cell-ACDC import paths.

The package layout refactor moved most modules (see CODEMAP.md). Old
import paths keep working for external code (plugins, spotMAX, notebooks)
and emit a ``MovedWarning`` (a ``DeprecationWarning`` whose message starts
with ``Cell-ACDC import path moved:``). Cell-ACDC's own code and tests use
only the new paths (enforced by tests/test_no_old_paths.py; pytest turns
the warning into an error by its message).

Kinds of old paths (tables in the generated ``_compat_moves.py``):

- old modules without a file (``OLD_MODULES``): served by the import hook
  ``OldPathFinder``, which ``cellacdc/__init__.py`` appends to
  ``sys.meta_path`` (``install``). It is asked only for modules that no
  file provides, so it never shadows real modules. Its loader applies one
  of the next two kinds to the old name.
- 1:1 moves (``alias_module``, ``OLD_MODULES`` kind ``'alias'``):
  importing the old name puts the module at the new path into
  ``sys.modules`` under the old name, so both names give the same object
  and patches reach the code.
- split modules (``moved_module``, kind ``'split'``): the old module (or a
  module that lost some names, e.g. ``whitelist``) gets a lazy module
  ``__getattr__``/``__dir__``. Every access imports the home module and
  reads the name there; nothing is cached, so patches on the home module
  stay visible through the old path.
- packages that replace an old module of the same name (``gui``,
  ``measurements_viewer``, ``myutils``; ``package_compat``): submodules are
  imported on demand (no warning), old names are forwarded with a warning
  except the documented plugin API (``PLUGIN_API``).
- ``forward_module``: a new module that shadows an old attribute path (for
  example ``cellacdc.gui.widgets`` used to be ``cellacdc.widgets`` seen
  through ``cellacdc.gui``) forwards the old names.

Modules using these functions get the class ``CompatModule``: setting or
deleting a forwarded name on them (``monkeypatch.setattr``,
``mock.patch``) acts on the home module, with a warning. Deleting a name
that a pending patch through an old path set undoes that patch (the undo
of ``mock.patch`` and of pytest >= 9.2's ``monkeypatch``).

Unknown names and dunder names (except ``__all__``) raise
``AttributeError`` without importing anything.

This module only uses the standard library, imports nothing from
cellacdc at module level and works on Python 3.10.
"""
import importlib
import importlib.machinery
import os
import sys
import types
import warnings

PREFIX = 'Cell-ACDC import path moved: '

_PACKAGE = __name__.rpartition('.')[0]
_MOVES_MODULE = f'{_PACKAGE}._compat_moves' if _PACKAGE else '_compat_moves'
_THIS_FILE = os.path.normcase(os.path.abspath(__file__))
_RESOLVING = set()


class MovedWarning(DeprecationWarning):
    """An old Cell-ACDC import path was used."""


def _moves():
    return importlib.import_module(_MOVES_MODULE)


def _is_dunder(name):
    return name.startswith('__') and name.endswith('__')


def _is_internal(frame):
    filename = frame.f_code.co_filename
    if filename.startswith('<frozen importlib'):
        return True
    norm = os.path.normcase(os.path.abspath(filename)) \
        if not filename.startswith('<') else filename
    if norm == _THIS_FILE:
        return True
    parts = norm.replace('\\', '/').split('/')
    if 'importlib' in parts:
        return True
    if frame.f_globals.get('__cellacdc_compat__'):
        return True
    return False


def _warn(message):
    """Warn at the first frame outside importlib, this module and the
    compat shims (the code that used the old path)."""
    frame = sys._getframe(1)
    while frame is not None and _is_internal(frame):
        frame = frame.f_back
    category = MovedWarning
    if frame is None:
        warnings.warn(message, category, stacklevel=2)
        return
    module_globals = frame.f_globals
    registry = module_globals.setdefault('__warningregistry__', {})
    # module_globals is not passed: warnings.warn does not either (with it,
    # a missing loader.get_source, e.g. for `python -c`, raises)
    warnings.warn_explicit(
        message, category, frame.f_code.co_filename, frame.f_lineno,
        module=module_globals.get('__name__'), registry=registry
    )


def _entry(module_name, attr, forward=None):
    moves = _moves()
    entry = moves.NAME_MOVES.get(module_name, {}).get(attr)
    if entry is not None:
        return entry
    for old in (forward, moves.FORWARDERS.get(module_name)):
        if old:
            entry = moves.NAME_MOVES.get(old, {}).get(attr)
            if entry is not None:
                return entry
    return None


def _entries(module_name, forward=None):
    moves = _moves()
    names = set(moves.NAME_MOVES.get(module_name, {}))
    for old in (forward, moves.FORWARDERS.get(module_name)):
        if old:
            names.update(moves.NAME_MOVES.get(old, {}))
    return names


def _is_plugin_api(module_name, attr):
    return attr in _moves().PLUGIN_API.get(module_name, ())


def _resolve(module_name, attr, entry):
    home, home_attr = entry
    key = (module_name, attr)
    if key in _RESOLVING:
        raise AttributeError(
            f'module {module_name!r} has no attribute {attr!r} '
            f'(circular compat entry -> {home})'
        )
    _RESOLVING.add(key)
    try:
        module = importlib.import_module(home)
        if home_attr is None:
            return module
        try:
            return getattr(module, home_attr)
        except AttributeError:
            raise AttributeError(
                f'module {module_name!r} has no attribute {attr!r} '
                f'(it moved to {home}, which does not define it)'
            ) from None
    finally:
        _RESOLVING.discard(key)


def _message(old, new):
    return f'{PREFIX}{old} -> {new} (see CODEMAP.md)'


def _new_name(entry, attr):
    home, home_attr = entry
    return home if home_attr is None else f'{home}.{home_attr}'


_MISSING = object()
_FANOUT_DELETED = {}
#: (home module name, attribute) -> stack of the bindings that patches
#: through an old path replaced: one list of (module, previous value) per
#: patch, home first (see _patch)
_PATCH_UNDO = {}


def _fanout_modules(module_name, home):
    """Imported modules holding code of the old module `module_name` (other
    than `home`), from the generated PATCH_FANOUT table."""
    moves = _moves()
    fanout = getattr(moves, 'PATCH_FANOUT', {})
    names = list(fanout.get(module_name, ()))
    forwarded = moves.FORWARDERS.get(module_name)
    if forwarded:
        names += fanout.get(forwarded, ())
    for name in names:
        module = sys.modules.get(name)
        if module is not None and module is not home:
            yield module


def _restore(bindings, attr):
    for module, value in bindings:
        if value is _MISSING:
            if attr in module.__dict__:
                delattr(module, attr)
        else:
            setattr(module, attr, value)


def _patch(module_name, entry, value, delete=False):
    """Set (or delete) a moved name at its home and, when the old module's
    code now lives in several modules, every binding of the same object
    there: the old module had one global, the codemod copied import-bound
    names into each module that uses them, so patching the old path (e.g.
    a test's monkeypatch) still reaches every lookup.

    Every set pushes the bindings it replaces on an undo stack. Deleting a
    name that a pending patch set restores the bindings of that patch
    instead (pytest >= 9.2 undoes ``monkeypatch.setattr`` of a name that
    is not in the module's ``__dict__``, as moved names never are, with
    ``delattr``; ``mock.patch`` undoes a patch of a non-local name with
    ``delattr`` too). Setting the value a pending patch replaced (older
    pytest's undo) pops that patch. Without a pending patch, deleting
    removes the name from its home and the copies."""
    home = importlib.import_module(entry[0])
    attr = entry[1]
    old_value = home.__dict__.get(attr, _MISSING)
    stack = _PATCH_UNDO.setdefault((entry[0], attr), [])
    if delete and stack:
        _restore(stack.pop(), attr)
        return
    if delete:
        delattr(home, attr)
        removed = []
        if old_value is not _MISSING:
            for module in _fanout_modules(module_name, home):
                if module.__dict__.get(attr, _MISSING) is old_value:
                    delattr(module, attr)
                    removed.append(module)
        _FANOUT_DELETED[(module_name, attr)] = removed
        return
    copies = []
    if old_value is not _MISSING:
        copies = [module for module in _fanout_modules(module_name, home)
                  if module.__dict__.get(attr, _MISSING) is old_value]
    if stack and stack[-1][0][1] is value:
        # the value the last pending patch replaced: an undo
        stack.pop()
    else:
        stack.append([(home, old_value)]
                     + [(module, old_value) for module in copies])
    setattr(home, attr, value)
    if old_value is _MISSING:
        # a name deleted without a pending patch is set again: restore the
        # copies too
        for module in _FANOUT_DELETED.pop((module_name, attr), ()):
            setattr(module, attr, value)
        return
    for module in copies:
        setattr(module, attr, value)


class CompatModule(types.ModuleType):
    """Module class of compat modules: forwards patches of moved names."""

    def __setattr__(self, attr, value):
        entry = None
        if not _is_dunder(attr) and attr not in self.__dict__:
            entry = _entry(self.__name__, attr)
        if entry is not None and entry[1] is not None:
            _warn(_message(f'{self.__name__}.{attr}',
                           _new_name(entry, attr)))
            _patch(self.__name__, entry, value)
            return
        super().__setattr__(attr, value)

    def __delattr__(self, attr):
        entry = None
        if not _is_dunder(attr) and attr not in self.__dict__:
            entry = _entry(self.__name__, attr)
        if entry is not None and entry[1] is not None:
            _warn(_message(f'{self.__name__}.{attr}',
                           _new_name(entry, attr)))
            _patch(self.__name__, entry, None, delete=True)
            return
        super().__delattr__(attr)


def _install_class(module):
    if module is not None and type(module) is types.ModuleType:
        try:
            module.__class__ = CompatModule
        except TypeError:
            pass


def alias_module(name):
    """Make the old module `name` an alias of the module at its new path
    (called by the loader of the import hook, ``OldPathFinder``)."""
    new_name = _moves().MODULE_ALIASES.get(name)
    if new_name is None:
        raise ImportError(f'{name} has no new path in {_MOVES_MODULE}')
    module = importlib.import_module(new_name)
    sys.modules[name] = module
    _warn(_message(name, new_name))
    return module


def _public_globals(module):
    if module is None:
        return set()
    return {n for n in vars(module) if not n.startswith('_')}


def moved_module(name, forward=None):
    """(__getattr__, __dir__) for a module whose names moved elsewhere,
    completely (split shim) or in part (a module that lost names)."""
    module = sys.modules.get(name)

    def __getattr__(attr):
        if _is_dunder(attr):
            if attr == '__all__':
                return sorted(
                    _public_globals(module)
                    | {n for n in _entries(name, forward)
                       if not n.startswith('_')})
            raise AttributeError(
                f'module {name!r} has no attribute {attr!r}')
        entry = _entry(name, attr, forward)
        if entry is None:
            raise AttributeError(
                f'module {name!r} has no attribute {attr!r}')
        if not _is_plugin_api(name, attr):
            _warn(_message(f'{name}.{attr}', _new_name(entry, attr)))
        return _resolve(name, attr, entry)

    def __dir__():
        names = set(vars(module)) if module is not None else set()
        return sorted(names | _entries(name, forward))

    _install_class(module)
    return __getattr__, __dir__


def forward_module(name, old):
    """(__getattr__, __dir__) for a module that shadows the old attribute
    path of the module `old` (for example cellacdc.gui.widgets for
    cellacdc.widgets): forwards the names of `old`'s table."""
    return moved_module(name, forward=old)


def _submodules(name, module):
    names = set(_moves().PACKAGE_SUBMODULES.get(name, ()))
    for folder in getattr(module, '__path__', None) or ():
        try:
            entries = os.listdir(folder)
        except OSError:
            continue
        for entry in entries:
            if entry.endswith('.py') and entry != '__init__.py':
                names.add(entry[:-3])
            elif os.path.isfile(os.path.join(folder, entry, '__init__.py')):
                names.add(entry)
    return names


def package_compat(name, forward=None):
    """(__getattr__, __dir__) for a package that replaces an old module of
    the same name: imports submodules on demand, forwards the old names
    (no warning for the plugin API)."""
    module = sys.modules.get(name)

    def __getattr__(attr):
        if _is_dunder(attr):
            raise AttributeError(
                f'module {name!r} has no attribute {attr!r}')
        if attr in _submodules(name, module):
            return importlib.import_module(f'{name}.{attr}')
        entry = _entry(name, attr, forward)
        if entry is None:
            raise AttributeError(
                f'module {name!r} has no attribute {attr!r}')
        if not _is_plugin_api(name, attr):
            _warn(_message(f'{name}.{attr}', _new_name(entry, attr)))
        return _resolve(name, attr, entry)

    def __dir__():
        names = set(vars(module)) if module is not None else set()
        names |= _submodules(name, module)
        return sorted(names | _entries(name, forward))

    _install_class(module)
    return __getattr__, __dir__


class _OldPathLoader:
    """Loader of an old module path without a file (``OLD_MODULES``)."""

    def __init__(self, kind):
        self.kind = kind

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        module.__cellacdc_compat__ = True
        if self.kind == 'alias':
            new_name = _moves().MODULE_ALIASES[name]
            module.__doc__ = (
                f'Deprecated: {name} moved to {new_name} (see CODEMAP.md).'
            )
            # The module at the new path replaces this one in sys.modules,
            # so the import returns it (its own spec is not touched)
            alias_module(name)
            return
        module.__doc__ = (
            f'Deprecated: the contents of {name} moved to other modules '
            '(see CODEMAP.md).'
        )
        module.__getattr__, module.__dir__ = moved_module(name)


class OldPathFinder:
    """``sys.meta_path`` entry serving the old module paths of the package
    layout refactor that have no file any more (``OLD_MODULES``).

    Appended at the end of ``sys.meta_path`` (``install``), it is asked only
    for modules that the regular finders do not find.
    """

    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.package + '.'):
            return None
        kind = getattr(_moves(), 'OLD_MODULES', {}).get(fullname)
        if kind not in ('alias', 'split'):
            return None
        return importlib.machinery.ModuleSpec(fullname, _OldPathLoader(kind))

    def invalidate_caches(self):
        pass


def install():
    """Serve the old module paths (``OLD_MODULES``): append the import
    hook ``OldPathFinder`` to ``sys.meta_path`` once (cellacdc/__init__.py
    calls it). Returns the finder."""
    for finder in sys.meta_path:
        if (
            type(finder).__name__ == 'OldPathFinder'
            and getattr(finder, 'package', None) == _PACKAGE
        ):
            return finder
    finder = OldPathFinder(_PACKAGE)
    sys.meta_path.append(finder)
    return finder
