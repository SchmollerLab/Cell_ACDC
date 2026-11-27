"""Phase 2 of the package-layout refactor: rewrite references to moved names.

Usage::

    python tools/refactor/acdc_refactor/rewrite.py --table <name-table.json>
        [--report <out.json>] [--check] [--src-root <dir> | --no-src-root]
        [--external-origins] [--notebooks] [--drop-unused-imports]
        [--exclude <glob>] [--no-default-excludes] <paths...>

Folders are walked recursively (dotfiles such as tests/segm/.test_sam.py
included, ``__pycache__`` and dot folders skipped). ``DEFAULT_EXCLUDES``
(compat tables, compat tests, tests/test_related_tests.py) are skipped
unless ``--no-default-excludes``. ``--check`` writes nothing and exits
with 1 when a file would change. ``--src-root`` is the folder that holds
the ``cellacdc`` package in its new layout (default: the current folder
when it contains cellacdc/__init__.py); it is needed to retarget patches
of re-exported names (see below).

The name table is written by ``python tools/refactor/mapping.py
name-table``::

    {"modules": {"<old dotted module>": "<new dotted module>", ...},
     "names": {"<old dotted module>": {"<name>": "<new home module>"}},
     "aliases": {"<new dotted module>": "<canonical import alias>"},
     "packages": ["<new dotted package>", ...],
     optional: "entries", "module_values", "shims", "intermediate"}

``modules`` lists whole-module moves, ``names`` every top-level name of an
old module that is no longer defined in the module the old dotted name now
resolves to (also names the old module only imported). ``entries``
(``[home, attr]``, attr null for module objects, attr != name for renamed
re-exports) and ``module_values`` say which names are module objects;
without them a value ``V`` of ``names[old][X]`` is read as: ``old.X`` is a
module when ``V.X`` is a known module, or ``V`` is a known module whose
last component is ``X`` (and, with a source root, ``V`` does not define
``X``); when ``V`` is an old module, ``X`` is resolved again through
``V``; otherwise ``X`` lives in ``V``. ``shims`` kind 'real' and
``intermediate`` modules still exist: names they do not list stay there
and their imports are kept.

Every old module was a plain file, so ``alias.X`` through an import of an
old module never means a submodule ``old.X`` (``measurements_viewer.
widgets`` is the old ``cellacdc.widgets``, not the new subpackage): the
table's ``reexports`` say what such names were; a table without that key
makes the rewriter guess the old sibling module of that name and report
it (``reexport_guessed``). Literal dotted paths (``import cellacdc``
chains, strings) mean the new submodule.

What the rewriter does, file by file (in place, only when something
changes; line endings, encoding and BOM are kept, lines are never
re-wrapped):

* Module imports of old modules (``from cellacdc import apps``,
  ``from . import apps``, ``from .. import widgets as acdc_widgets``,
  ``import cellacdc.apps as m``, ``import cellacdc.widgets`` with
  ``cellacdc.widgets.X`` chains), at module level or inside functions:
  the uses of the bound name are resolved scope-aware (parameters,
  locals, class attributes and comprehension variables that shadow it are
  never touched), every ``alias.X`` (also ``alias.X.Y`` and chains through
  re-exported modules such as ``apps.widgets.myMessageBox``) gets the
  canonical alias of the home module of ``X`` (only the alias part is
  spliced), and the import is replaced in place by imports of the needed
  new modules: relative inside the same part package (``DEFAULT_PARTS``:
  the top-level packages under ``cellacdc``), absolute otherwise and in
  files outside ``cellacdc``. Other names of the statement are kept;
  new imports already visible in the scope or added by an earlier
  statement of the same block are not repeated. Names whose home is
  outside ``cellacdc`` are imported by name (``from qtpy.QtGui import
  QAction``); ``--external-origins`` does the same for third-party names
  that the table maps to a cellacdc module binding them. A name that now
  lives in the rewritten file itself becomes a bare name.
* Whole-module moves keep attribute access through the new module's
  canonical alias (``from cellacdc.gui.mixins import navigation as
  gui_navigation``).
* ``from <old> import A, B as C`` is split per home module.
* Patch targets (``monkeypatch.setattr(<old module>, 'X', v)``,
  ``monkeypatch.setattr('cellacdc.<old>.X', v)``, ``mock.patch(...)``,
  ``patch.object(<old module>, 'X')``, ``monkeypatch.delattr(...)``,
  ``setattr``/``delattr`` and ``<old module>.X = v``) are retargeted to
  the module that looks ``X`` up: the home module of ``X`` when its
  definition moved there; with a source root also every new module split
  off the old one that binds ``X`` by an import and reads it. Patches of
  re-exported names (``myutils.printl``, ``myutils.is_win``) need the
  source root and are reported otherwise. Several lookup modules: the
  patch statement is repeated per module when that is safe (simple
  arguments), else the first module is used. Every patch is reported.
  ``getattr``/``hasattr(<old module>, 'X')`` read from the home module.
* String literals that are exactly ``cellacdc.<old>`` or
  ``cellacdc.<old>.X`` are rewritten (old paths that still exist as a new
  package, e.g. ``cellacdc.measurements_viewer``, and old split modules
  are only reported); other strings, f-strings and docstrings mentioning
  old modules (dotted or ``<old>.py``) are listed under ``mentions``.
* Reported, never rewritten: old split modules used as values
  (``getattr(widgets, name)``, ``widgets.__file__``, passed as argument;
  the old import is kept), names unknown to the table, ambiguous bindings
  (the alias is also assigned, or the import shares its line), unused
  imports of old split modules, string annotations naming an old alias,
  new aliases that collide with a name anywhere in the file
  (``<alias>_mod`` is used instead), files that do not parse, and lines
  longer than 127 characters that the rewrite produced. Value uses of
  whole-moved modules are rewritten to the new module (reported as
  ``info``, or as ``unhandled`` when the old path is now a package or a
  split shim).

Added for the C12 commit of tools/refactor/run.py:

* ``reexports``, ``fanout`` and ``defined`` of the name table (plan.py):
  names an old module only imported (re-exports of modules moved whole,
  old globals that clash with a new submodule) resolve to their defining
  module without guessing; patch lookups are the modules holding the old
  module's code that bind and read the name (``fanout``), plus its home
  when the old module defined it (``defined``). Several lookup modules
  are narrowed to those the enclosing function uses (a window class
  counts as all its mixin modules), else the statement is repeated per
  module under a comment line ``# moved: x is read in each of these
  modules`` (``setattr(m, 'x', <expression>)``: the copies set
  ``<first>.x``; assignments ``old.x = v`` too). ``for name in ('a', 'b'): setattr(old,
  name, ...)`` is retargeted when all names share one lookup module.
* With a source root, modules that stay where they are but bound an old
  module that another file reads through them (``model_execution.myutils.
  X`` in a test) are followed like old modules (``find_reexports``).
* ``mod = <old split module>`` in a function, ``mod`` used only as
  ``mod.X``: the uses name the home modules and the assignment goes.
* Python code in strings of test files (subprocess scripts, dedented or
  not) is rewritten like a module, unless it checks the layout
  (``__path__``); an unused ``import cellacdc.<moved whole>`` becomes an
  import of the new path.
* ``--notebooks``: the code cells of ``*.ipynb`` files (magics masked) are
  rewritten as one module; the JSON keeps its format.
* ``--drop-unused-imports``: imports of old modules, or of names imported
  from one, without any use left are removed (also imports of old modules
  that still exist, e.g. whitelist.py, when every use now names another
  module; imports that had no use before are kept), except ``noqa`` lines,
  ``__all__`` entries, names in string annotations, module-level
  bindings another given file reads through the module (unless the table
  redirects those reads) and, in tests, imports of modules moved whole.

The JSON report: ``{"files": {path: {"changed", "references", "imports",
"patches", "strings", "unhandled", "mentions", "long_lines", "info",
"dropped"}}, "summary": {...}, "src_root": ..., "reexports": ...}``; every
entry has a ``line``, ``unhandled``/``info``/``mentions`` entries a
``kind``.

Idempotent: new module paths are never treated as old ones. Run it with
Python >= 3.12 (f-string expression positions); on older versions
references inside f-strings are reported instead of rewritten. The code
parses on Python 3.10 (CI flake8).

Python API: ``load_table(path, index=None)``, ``SourceIndex(root)``,
``rewrite_source(text, table, module_name=..., ...)`` ->
``(new_text, report)``, ``rewrite_file(path, table, ...)`` ->
``(changed, report)``, ``iter_python_files``, ``main(argv)``.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import io
import json
import os
import re
import sys
import textwrap
import tokenize
import warnings

DEFAULT_PARTS = (
    'gui', 'measurements_viewer', 'launcher', 'data_prep', 'data_structure',
    'segmentation_module', 'utils', 'shared', 'myutils',
)

# Files whose strings name old paths on purpose (compat tables and tests,
# synthetic module names of the related_tests tests)
DEFAULT_EXCLUDES = (
    'cellacdc/_compat.py',
    'cellacdc/_compat_moves.py',
    'cellacdc/_compat/*',
    'tests/test_compat*.py',
    'tests/test_related_tests.py',
)

MAX_LINE_LENGTH = 127
IMPORT_LINE_LENGTH = 79
FSTRING_POSITIONS_OK = sys.version_info >= (3, 12)

_NEWLINE_RE = re.compile(r'\r\n|\r|\n')
_DUNDER_RE = re.compile(r'^__\w+__$')
_DOTTED_RE = re.compile(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$')
_MENTION_RE = re.compile(r'\bcellacdc(?:\.[A-Za-z_]\w*)+')
_SIMPLE_STRING_RE = re.compile(r'^([rRuU]?)(\'|")(.*)\2$', re.S)

_SCOPE_NODES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
    ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp,
)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
_MATCH_NAME_NODES = tuple(
    getattr(ast, name) for name in ('MatchAs', 'MatchStar')
    if hasattr(ast, name)
)
_MATCH_MAPPING = getattr(ast, 'MatchMapping', None)


def _is_cellacdc(module):
    return module == 'cellacdc' or module.startswith('cellacdc.')


def _last(module):
    return module.rpartition('.')[2]


# ---------------------------------------------------------------------------
# Name table
# ---------------------------------------------------------------------------

class ModRef:
    """The attribute is a module object (``guessed``: not in the table)."""
    __slots__ = ('module', 'guessed')

    def __init__(self, module, guessed=False):
        self.module = module
        self.guessed = guessed

    def __repr__(self):
        return 'ModRef(%r)' % self.module


class NameRef:
    """The attribute is the name ``name`` that lives in ``home``."""
    __slots__ = ('home', 'name')

    def __init__(self, home, name):
        self.home = home
        self.name = name

    def __repr__(self):
        return 'NameRef(%r, %r)' % (self.home, self.name)


class Res:
    """Result of resolving ``root.attr1.attr2...``.

    kind:
      'unchanged'  nothing old is involved (or a literal path of a new module)
      'name'       root.attrs[:consumed] is the name ``name`` of ``home``
                   (looked up in the old module ``owner``)
      'module'     root.attrs[:consumed] is the module ``module`` (new or
                   external; ``value`` when it is the new location of the
                   whole-moved old module ``old_module`` used as a value)
      'value_old'  root.attrs[:consumed] is the old split module
                   ``old_module`` used as a value
      'unknown'    attrs[consumed] is not known in ``old_module``
    """
    __slots__ = ('kind', 'consumed', 'module', 'home', 'name', 'owner',
                 'value', 'old_module', 'attr', 'guessed')

    def __init__(self, kind, consumed=0, module=None, home=None, name=None,
                 owner=None, value=False, old_module=None, attr=None,
                 guessed=None):
        self.kind = kind
        self.consumed = consumed
        self.module = module
        self.home = home
        self.name = name
        self.owner = owner
        self.value = value
        self.old_module = old_module
        self.attr = attr
        self.guessed = guessed

    def denotes_old_module(self):
        return self.kind == 'value_old' or (
            self.kind == 'module' and self.value)


_GUESS = object()


class NameTable:
    """The name table of mapping.py (see the module docstring).

    Optional keys written by plan.py are used when present: ``entries``
    (``{old: {name: [home, attr]}}``, ``attr`` None for module objects and
    different from ``name`` for renamed re-exports), ``module_values``,
    ``shims`` (``{old: 'real'|'alias'|'split'|'package'|'none'}``: names
    not listed for a 'real' module stay there) and ``intermediate``
    (modules at an intermediate path that lost names; names not listed
    stay there). Without them, module objects are recognised by the
    heuristics described in the module docstring.
    """

    def __init__(self, data, index=None):
        self.modules = dict(data.get('modules') or {})
        self.names = {
            key: dict(value) for key, value in
            (data.get('names') or {}).items()
        }
        self.entries = {
            key: {name: tuple(entry) for name, entry in value.items()}
            for key, value in (data.get('entries') or {}).items()
        }
        self.module_values = {
            key: dict(value) for key, value in
            (data.get('module_values') or {}).items()
        }
        self.shims = dict(data.get('shims') or {})
        # names that an old module bound by an import and that the
        # rewrite view resolves to their defining module: re-exports of
        # modules moved whole and old globals that clash with a new
        # submodule of the same-name package (measurements_viewer.widgets
        # = the old cellacdc.widgets); used only for attribute access
        # through an import binding (never for literal dotted paths)
        self.reexports = {
            key: {name: tuple(entry) for name, entry in value.items()}
            for key, value in (data.get('reexports') or {}).items()
        }
        # without a reexports table a name that the table omits is
        # guessed to be the old sibling module of that name
        self.guess_siblings = 'reexports' not in data
        # old module -> new modules that hold its code (plan.py
        # code_destinations); old module -> names it defined at the base
        self.fanout = {
            key: tuple(value)
            for key, value in (data.get('fanout') or {}).items()}
        self.defined = {
            key: set(value)
            for key, value in (data.get('defined') or {}).items()}
        # host module of a class split into mixins -> the mixin modules
        # (using the class means using the code of all of them)
        self.class_modules = {}
        for spec in (data.get('classes') or {}).values():
            if spec.get('host'):
                self.class_modules.setdefault(spec['host'], set()).update(
                    mixin.rpartition('.')[0]
                    for mixin in (spec.get('methods') or {}).values())
        self.aliases = dict(data.get('aliases') or {})
        self.packages = set(data.get('packages') or ())
        self.index = index
        self.identity = {
            module for module, kind in self.shims.items()
            if kind == 'real' and module not in self.modules
        }
        for module, entries in (data.get('intermediate') or {}).items():
            if module in self.names or module in self.modules:
                continue
            self.entries[module] = {
                name: tuple(entry) for name, entry in entries.items()}
            self.names[module] = {
                name: entry[0] for name, entry in entries.items()}
            self.identity.add(module)
        self.old = set(self.modules) | set(self.names) | set(self.entries)
        new = set(self.aliases) | self.packages | set(self.modules.values())
        homes = [h for names in self.names.values() for h in names.values()]
        homes += [e[0] for entries in self.entries.values()
                  for e in entries.values()]
        for home in homes:
            if _is_cellacdc(home) and home not in self.old:
                new.add(home)
        closure = set()
        for module in new:
            parts = module.split('.')
            for i in range(1, len(parts) + 1):
                closure.add('.'.join(parts[:i]))
        self.new = closure
        prefixes = set()
        for module in self.old:
            parts = module.split('.')
            for i in range(1, len(parts)):
                prefixes.add('.'.join(parts[:i]))
        self.old_prefixes = prefixes - self.old
        self._carved = {}

    def is_known_module(self, module):
        return (
            module in self.old or module in self.new
            or module in self.old_prefixes
        )

    def alias(self, module):
        return self.aliases.get(module)

    def shim_kind(self, module):
        return self.shims.get(module)

    def _defines(self, module, name):
        if self.index is None:
            return False
        info = self.index.info(module)
        return info is not None and name in info.defines

    def _entry(self, module, name):
        entries = self.entries.get(module)
        if entries is not None and name in entries:
            return entries[name]
        names = self.names.get(module)
        if names is not None and name in names:
            values = self.module_values.get(module)
            if values is not None and name in values:
                return values[name], None
            return names[name], _GUESS
        return None

    def _follow(self, owner, home, attr, depth):
        if home in self.old and home != owner and depth < 8:
            ref, _ = self._resolve_attr(home, attr, depth + 1, False)
            if ref is not None:
                return ref
        return NameRef(home, attr)

    def _ref_from_value(self, owner, name, value, depth=0):
        candidate = value + '.' + name
        if self.is_known_module(candidate):
            return ModRef(candidate)
        if (
            self.is_known_module(value) and _last(value) == name
            and not self._defines(value, name)
        ):
            return ModRef(value)
        return self._follow(owner, value, name, depth)

    def resolve_attr(self, module, name, literal=True):
        """Resolve ``module.name`` for an old module.

        Returns ``(ref, exact)``: ``ref`` is a ModRef, a NameRef (whose
        ``name`` is the name in its home) or None (unknown); ``exact`` is
        True when the module path is literally ``module.name``.

        ``literal``: ``module.name`` is spelled as this dotted path (a
        ``cellacdc.x.y`` chain or string), so a new submodule
        ``module.name`` is meant. Otherwise the old module was reached
        through an import binding: every old module was a plain file, so
        ``name`` was one of its globals; a name that the table omits
        because it clashes with a new submodule
        (``measurements_viewer.widgets``) is guessed to be the old sibling
        module of that name.
        """
        return self._resolve_attr(module, name, 0, literal)

    def _resolve_attr(self, module, name, depth, literal):
        entry = self._entry(module, name)
        if entry is not None:
            home, attr = entry
            if attr is None:
                return ModRef(home), False
            if attr is _GUESS:
                return self._ref_from_value(module, name, home, depth), False
            return self._follow(module, home, attr, depth), False
        full = module + '.' + name
        if literal and self.is_known_module(full):
            return ModRef(full), True
        reexport = self.reexports.get(module, {}).get(name)
        if reexport is not None:
            home, attr = reexport
            if attr is None:
                return ModRef(home), False
            return self._follow(module, home, attr, depth), False
        if not literal and self.guess_siblings:
            sibling = module.rpartition('.')[0] + '.' + name
            if sibling in self.old and sibling != module and not (
                    module in self.modules
                    and self._defines(self.modules[module], name)):
                return ModRef(sibling, guessed=True), False
        if module in self.modules:
            return NameRef(self.modules[module], name), False
        if module in self.identity or self._defines(module, name):
            return NameRef(module, name), False
        return None, False

    def resolve_chain(self, root, attrs, literal=True):
        """Resolve the dotted expression ``root.attrs[0].attrs[1]...``.

        ``root`` is the module bound to the first name (an old module, a
        package that contains old modules such as ``cellacdc``, or any
        other module). ``literal``: the chain spells the dotted path from
        the top (``import cellacdc`` chains, strings), not an alias bound
        to an old module (see ``resolve_attr``).
        """
        current = root
        exact = literal
        guessed = None
        for i, attr in enumerate(attrs):
            if _DUNDER_RE.match(attr):
                return self._end(current, i, guessed)
            if current in self.old:
                ref, step_exact = self.resolve_attr(current, attr, exact)
                if ref is None:
                    return Res('unknown', i, old_module=current, attr=attr)
                if isinstance(ref, ModRef) and ref.guessed:
                    guessed = '%s.%s' % (current, attr)
            elif current in self.old_prefixes:
                full = current + '.' + attr
                if full in self.old or full in self.old_prefixes:
                    ref, step_exact = ModRef(full), True
                else:
                    return Res('unchanged')
            else:
                return Res('unchanged')
            if isinstance(ref, NameRef):
                return Res(
                    'name', i + 1, home=ref.home, name=ref.name,
                    owner=current, attr=attr, guessed=guessed
                )
            exact = exact and step_exact
            current = ref.module
            if current in self.old or current in self.old_prefixes:
                continue
            if exact:
                return Res('unchanged')
            return Res('module', i + 1, module=current, guessed=guessed)
        return self._end(current, len(attrs), guessed)

    def _end(self, current, consumed, guessed=None):
        if current in self.old:
            if current in self.modules:
                return Res(
                    'module', consumed, module=self.modules[current],
                    value=True, old_module=current, guessed=guessed
                )
            if current in self.identity:
                return Res(
                    'module', consumed, module=current, value=True,
                    old_module=current, guessed=guessed
                )
            return Res('value_old', consumed, old_module=current,
                       guessed=guessed)
        return Res('unchanged')

    def add_reexports(self, extra):
        """Old modules bound by imports in modules that stay where they
        are, read through them by other files (``model_execution.
        myutils.X`` in a test): ``{module: {local: old module}}``. Such a
        module becomes an identity module of the table (its own names
        stay) whose ``local`` is the old module."""
        for module, names in extra.items():
            self.reexports.setdefault(module, {}).update(
                {name: (old, None) for name, old in names.items()})
            if module in self.old:
                continue
            self.old.add(module)
            self.identity.add(module)
            parts = module.split('.')
            for i in range(1, len(parts)):
                prefix = '.'.join(parts[:i])
                if prefix not in self.old:
                    self.old_prefixes.add(prefix)
        self._carved = {}

    def redirects(self, module, name):
        """True when the rewrite sends reads of ``module.name`` (an old
        module) elsewhere: the name is listed for it."""
        return module in self.old and (
            name in (self.names.get(module) or ())
            or name in (self.entries.get(module) or ())
            or name in (self.reexports.get(module) or ()))

    def carved(self, module):
        """New cellacdc modules that hold code of the old ``module`` (the
        ``fanout`` table; without it: the modules that received names of
        it, homes of re-exported names included)."""
        if module in self._carved:
            return self._carved[module]
        out = set()
        if module in self.modules:
            out.add(self.modules[module])
        if module in self.identity:
            out.add(module)
        if module in self.fanout:
            out.update(self.fanout[module])
            self._carved[module] = out
            return out
        names = set(self.names.get(module) or ()) | set(
            self.entries.get(module) or ())
        for name in names:
            ref, _ = self.resolve_attr(module, name, literal=False)
            if (
                isinstance(ref, NameRef) and _is_cellacdc(ref.home)
                and ref.home != 'cellacdc' and (
                    ref.home not in self.old or ref.home in self.identity)
            ):
                out.add(ref.home)
        self._carved[module] = out
        return out


def load_table(path, index=None):
    with open(path, encoding='utf-8') as file:
        return NameTable(json.load(file), index=index)


# ---------------------------------------------------------------------------
# Source index (lookup modules of patches, external origins)
# ---------------------------------------------------------------------------

class ModuleInfo:
    __slots__ = ('defines', 'imports', 'loads')

    def __init__(self):
        self.defines = set()
        self.imports = {}
        self.loads = set()


def _resolve_relative(module_name, is_package, level, module):
    if level == 0:
        return module
    if not module_name:
        return None
    package = (
        module_name if is_package else module_name.rpartition('.')[0]
    )
    parts = package.split('.') if package else []
    up = level - 1
    if up >= len(parts):
        return None
    base = '.'.join(parts[:len(parts) - up])
    return base + '.' + module if module else base


class SourceIndex:
    """Reads the module-level bindings of modules under ``root``."""

    def __init__(self, root):
        self.root = root
        self._cache = {}

    def path(self, module):
        base = os.path.join(self.root, *module.split('.'))
        for path in (base + '.py', os.path.join(base, '__init__.py')):
            if os.path.isfile(path):
                return path
        return None

    def invalidate(self, module):
        """Forget a module that was rewritten (lookups of later files see
        its new imports)."""
        self._cache.pop(module, None)

    def info(self, module):
        if module in self._cache:
            return self._cache[module]
        info = None
        path = self.path(module)
        if path is not None:
            try:
                with open(path, 'rb') as file:
                    text = _decode(file.read())[0]
                tree = _parse(text, path)
            except (SyntaxError, ValueError, UnicodeDecodeError):
                tree = None
            if tree is not None:
                info = self._build(
                    tree, module, path.endswith('__init__.py')
                )
        self._cache[module] = info
        return info

    @staticmethod
    def _build(tree, module, is_package):
        info = ModuleInfo()

        def add_targets(target):
            for node in ast.walk(target):
                if isinstance(node, ast.Name):
                    info.defines.add(node.id)

        def visit(statements):
            for stmt in statements:
                if isinstance(
                    stmt,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    info.defines.add(stmt.name)
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        add_targets(target)
                elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                    add_targets(stmt.target)
                elif isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        if alias.asname:
                            info.imports[alias.asname] = (
                                'module', alias.name, None)
                        else:
                            first = alias.name.split('.')[0]
                            info.imports[first] = ('module', first, None)
                elif isinstance(stmt, ast.ImportFrom):
                    source = _resolve_relative(
                        module, is_package, stmt.level, stmt.module
                    )
                    if source is None:
                        continue
                    for alias in stmt.names:
                        if alias.name == '*':
                            continue
                        local = alias.asname or alias.name
                        info.imports[local] = ('name', source, alias.name)
                else:
                    for field in ('body', 'orelse', 'finalbody'):
                        visit(getattr(stmt, field, None) or ())
                    for handler in getattr(stmt, 'handlers', None) or ():
                        visit(handler.body)

        visit(tree.body)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                info.loads.add(node.id)
        return info


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

class Scope:
    __slots__ = ('kind', 'node', 'parent', 'bindings', 'globals',
                 'nonlocals', 'star_import')

    def __init__(self, kind, node, parent):
        self.kind = kind
        self.node = node
        self.parent = parent
        self.bindings = {}
        self.globals = set()
        self.nonlocals = set()
        self.star_import = False


class Binding:
    __slots__ = ('kind', 'node', 'alias', 'scope')

    def __init__(self, kind, node, scope, alias=None):
        self.kind = kind
        self.node = node
        self.alias = alias
        self.scope = scope


class ScopeBuilder:
    """Bindings and name loads per scope, with Python's lookup rules."""

    def __init__(self, tree):
        self.parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.module = Scope('module', tree, None)
        self.loads = {}
        self.bindings = {}
        self.imports = []
        self.load_scope = {}
        self._prescan(tree.body, self.module)
        for stmt in tree.body:
            self._visit(stmt, self.module)

    @staticmethod
    def _prescan(nodes, scope):
        stack = list(nodes)
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Global):
                scope.globals.update(node.names)
            elif isinstance(node, ast.Nonlocal):
                scope.nonlocals.update(node.names)
            if isinstance(node, _SCOPE_NODES):
                continue
            stack.extend(ast.iter_child_nodes(node))

    def _bind(self, scope, name, kind, node, alias=None):
        target = scope
        if scope.kind != 'module' and name in scope.globals:
            target = self.module
        elif name in scope.nonlocals:
            parent = scope.parent
            while parent is not None and parent.kind in (
                    'class', 'comprehension'):
                parent = parent.parent
            if parent is not None and parent.kind != 'module':
                target = parent
        binding = Binding(kind, node, target, alias)
        target.bindings.setdefault(name, []).append(binding)
        self.bindings.setdefault(name, []).append(binding)

    @staticmethod
    def _all_args(args):
        out = list(args.posonlyargs) + list(args.args)
        if args.vararg is not None:
            out.append(args.vararg)
        out.extend(args.kwonlyargs)
        if args.kwarg is not None:
            out.append(args.kwarg)
        return out

    def _visit_arguments_outer(self, args, scope):
        for default in args.defaults:
            self._visit(default, scope)
        for default in args.kw_defaults:
            if default is not None:
                self._visit(default, scope)
        for arg in self._all_args(args):
            if arg.annotation is not None:
                self._visit(arg.annotation, scope)

    def _visit(self, node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                self._visit(decorator, scope)
            self._visit_arguments_outer(node.args, scope)
            if node.returns is not None:
                self._visit(node.returns, scope)
            self._bind(scope, node.name, 'def', node)
            child = Scope('function', node, scope)
            for arg in self._all_args(node.args):
                self._bind(child, arg.arg, 'param', arg)
            self._prescan(node.body, child)
            for stmt in node.body:
                self._visit(stmt, child)
            return
        if isinstance(node, ast.Lambda):
            self._visit_arguments_outer(node.args, scope)
            child = Scope('function', node, scope)
            for arg in self._all_args(node.args):
                self._bind(child, arg.arg, 'param', arg)
            self._visit(node.body, child)
            return
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                self._visit(decorator, scope)
            for base in node.bases:
                self._visit(base, scope)
            for keyword in node.keywords:
                self._visit(keyword.value, scope)
            self._bind(scope, node.name, 'class', node)
            child = Scope('class', node, scope)
            self._prescan(node.body, child)
            for stmt in node.body:
                self._visit(stmt, child)
            return
        if isinstance(node, _COMPREHENSIONS):
            generators = node.generators
            self._visit(generators[0].iter, scope)
            child = Scope('comprehension', node, scope)
            for i, generator in enumerate(generators):
                if i > 0:
                    self._visit(generator.iter, child)
                self._visit(generator.target, child)
                for condition in generator.ifs:
                    self._visit(condition, child)
            if isinstance(node, ast.DictComp):
                self._visit(node.key, child)
                self._visit(node.value, child)
            else:
                self._visit(node.elt, child)
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.loads.setdefault(node.id, []).append((node, scope))
                self.load_scope[node] = scope
            else:
                kind = 'del' if isinstance(node.ctx, ast.Del) else 'store'
                self._bind(scope, node.id, kind, node)
            return
        if isinstance(node, ast.NamedExpr):
            target_scope = scope
            while target_scope.kind == 'comprehension':
                target_scope = target_scope.parent
            self._bind(target_scope, node.target.id, 'walrus', node)
            self._visit(node.value, scope)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == '*':
                    scope.star_import = True
                    continue
                local = alias.asname or alias.name.split('.')[0]
                self._bind(scope, local, 'import', node, alias)
                self.imports.append((node, alias, scope))
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return
        if isinstance(node, ast.ExceptHandler):
            if node.type is not None:
                self._visit(node.type, scope)
            if node.name:
                self._bind(scope, node.name, 'except', node)
            for stmt in node.body:
                self._visit(stmt, scope)
            return
        if _MATCH_NAME_NODES and isinstance(node, _MATCH_NAME_NODES):
            if node.name:
                self._bind(scope, node.name, 'match', node)
        elif _MATCH_MAPPING is not None and isinstance(node, _MATCH_MAPPING):
            if node.rest:
                self._bind(scope, node.rest, 'match', node)
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope)

    def resolve(self, scope, name):
        """The scope whose binding ``name`` refers to in ``scope``."""
        current = scope
        first = True
        while current is not None:
            if current.kind == 'class' and not first:
                current = current.parent
                continue
            if current.kind != 'module' and name in current.globals:
                return self.module if name in self.module.bindings else None
            if name in current.nonlocals:
                first = False
                current = current.parent
                continue
            if name in current.bindings:
                return current
            first = False
            current = current.parent
        return None


# ---------------------------------------------------------------------------
# Source text helpers
# ---------------------------------------------------------------------------

def _parse(text, path='<string>'):
    """``ast.parse`` without SyntaxWarnings (invalid escapes in old code)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ast.parse(text, path)


def _decode(data):
    """Decode source bytes; returns (text, encoding)."""
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    return data.decode(encoding), encoding


class SourceText:
    def __init__(self, text):
        self.text = text
        self.line_starts = [0]
        for match in _NEWLINE_RE.finditer(text):
            self.line_starts.append(match.end())
        first = _NEWLINE_RE.search(text)
        self.default_newline = first.group(0) if first else '\n'
        self._lines = {}

    def line_bounds(self, lineno):
        """(start, end without newline, end with newline) of a line."""
        start = self.line_starts[lineno - 1]
        if lineno < len(self.line_starts):
            end_nl = self.line_starts[lineno]
        else:
            end_nl = len(self.text)
        end = end_nl
        while end > start and self.text[end - 1] in '\r\n':
            end -= 1
        return start, end, end_nl

    def line(self, lineno):
        if lineno not in self._lines:
            start, end, _ = self.line_bounds(lineno)
            self._lines[lineno] = self.text[start:end]
        return self._lines[lineno]

    def newline(self, lineno):
        _, end, end_nl = self.line_bounds(lineno)
        return self.text[end:end_nl] or self.default_newline

    def offset(self, lineno, col):
        start = self.line_starts[lineno - 1]
        line = self.line(lineno)
        if line.isascii():
            return start + col
        prefix = line.encode('utf-8')[:col].decode('utf-8', 'ignore')
        return start + len(prefix)

    def span(self, node):
        return (
            self.offset(node.lineno, node.col_offset),
            self.offset(node.end_lineno, node.end_col_offset),
        )

    def segment(self, node):
        start, end = self.span(node)
        return self.text[start:end]


# ---------------------------------------------------------------------------
# File rewriter
# ---------------------------------------------------------------------------

class ImportEntry:
    """One name of an import statement and what it binds."""

    def __init__(self, stmt, alias, scope, local, kind, module=None,
                 old_entry=False, ref=None, source=None):
        self.stmt = stmt
        self.alias = alias
        self.scope = scope
        self.local = local
        # 'old_module': binds the old module ``module``
        # 'pkg_root': binds a package containing old modules (cellacdc)
        # 'name': from <old module ``source``> import <alias.name> (ref)
        self.kind = kind
        self.module = module
        self.old_entry = old_entry
        self.ref = ref
        self.source = source
        self.removed = False


class Group:
    """All bindings of one local name in one scope by old-module imports."""

    def __init__(self, scope, local, kind, root):
        self.scope = scope
        self.local = local
        self.kind = kind
        self.root = root
        self.entries = []
        self.ambiguous = None
        self.uses = []
        self.keep = False
        self.needs_root = False
        self.requirements = []
        self.edits = []


class UseEdit:
    def __init__(self, start, end, target, report_end, line, category,
                 suffix=''):
        self.start = start
        self.end = end
        self.target = target
        self.report_end = report_end
        self.line = line
        self.category = category
        self.suffix = suffix


class DupPatchEdit:
    """A patch statement repeated once per lookup module (``value``: span
    of the value and attribute name when the copies read the first
    module's attribute; ``name``: the patched attribute, named by the
    comment line above the copies)."""

    def __init__(self, stmt, arg_span, targets, line, value=None,
                 name=None):
        self.stmt = stmt
        self.arg_span = arg_span
        self.targets = targets
        self.line = line
        self.value = value
        self.name = name


class FileRewriter:
    def __init__(self, text, table, module_name=None, is_package=False,
                 path='<string>', index=None, parts=DEFAULT_PARTS,
                 external_origins=False, drop_unused=False,
                 external_uses=None):
        self.text = text
        # drop_unused: unused imports of old modules and of names imported
        # from old modules are removed instead of kept/rewritten.
        # external_uses: {(module, name): {files}} of all rewritten files
        # (build_use_index); module-level imports read by other files
        # through this module are kept (None: keep every module-level
        # import of a file that is a module)
        self.drop_unused = drop_unused
        self.external_uses = external_uses
        # a script in a string of a test file (_process_code_string): its
        # top level is not a module other files import
        self.in_code_string = False
        self.table = table
        self.module_name = module_name
        self.is_package = is_package
        self.path = path
        self.index = index if index is not None else table.index
        self.parts = tuple(parts)
        self.external_origins = external_origins and self.index is not None
        self.report = _empty_report()
        self.src = SourceText(text)
        self.file_part = self._part_of_file()

    # -- reporting ----------------------------------------------------------

    def _unhandled(self, line, kind, detail):
        self.report['unhandled'].append(
            {'line': line, 'kind': kind, 'detail': detail})

    def _info(self, line, kind, detail):
        self.report['info'].append(
            {'line': line, 'kind': kind, 'detail': detail})

    # -- main ---------------------------------------------------------------

    def run(self):
        try:
            tree = _parse(self.text, self.path)
        except (SyntaxError, ValueError) as error:
            self._unhandled(
                getattr(error, 'lineno', None) or 0, 'syntax_error',
                'does not parse on Python %d.%d: %s' % (
                    sys.version_info[0], sys.version_info[1], error)
            )
            return self.text
        limit = sys.getrecursionlimit()
        if limit < 20000:
            sys.setrecursionlimit(20000)
        try:
            return self._run(tree)
        finally:
            sys.setrecursionlimit(limit)

    def _run(self, tree):
        self.tree = tree
        self.sb = ScopeBuilder(tree)
        self.parents = self.sb.parents
        self.use_edits = []
        self.dup_edits = []
        self.string_edits = []
        self.plans = {}
        self.plan_order = []
        self.alias_of = {}
        self.alias_owner = {}
        self.bare_access = {}
        self.entries = {}
        self.groups = []
        self._annotation_cache = None
        self._classify_imports()
        self.tracked_locals = {group.local for group in self.groups}
        self.used_homes = self._collect_used_homes()
        for group in self.groups:
            self._process_group(group)
        self._process_name_entries()
        self._process_strings()
        self._decide_groups()
        return self._render()

    # -- file location ------------------------------------------------------

    def _part_of_file(self):
        if not self.module_name or not _is_cellacdc(self.module_name):
            return None
        parts = self.module_name.split('.')
        if len(parts) >= 3 and parts[1] in self.parts:
            return parts[1]
        if len(parts) == 2 and self.is_package and parts[1] in self.parts:
            return parts[1]
        return None

    def _file_package(self):
        if self.is_package:
            return self.module_name
        return self.module_name.rpartition('.')[0]

    def _relative(self, module):
        file_parts = self._file_package().split('.')
        target = module.split('.')
        common = 0
        while (
            common < min(len(file_parts), len(target))
            and file_parts[common] == target[common]
        ):
            common += 1
        dots = '.' * (len(file_parts) - common + 1)
        return dots + '.'.join(target[common:])

    def _from_source(self, source):
        """The text after ``from`` to import from module ``source``."""
        if self.file_part is None or not _is_cellacdc(source):
            return source
        prefix = 'cellacdc.' + self.file_part
        if source != prefix and not source.startswith(prefix + '.'):
            return source
        return self._relative(source)

    # -- imports ------------------------------------------------------------

    def _statement_on_own_lines(self, stmt):
        start, end = self.src.span(stmt)
        line_start, _, _ = self.src.line_bounds(stmt.lineno)
        _, line_end, _ = self.src.line_bounds(stmt.end_lineno)
        before = self.text[line_start:start]
        after = self.text[end:line_end]
        return not before.strip() and (
            not after.strip() or after.strip().startswith('#'))

    def _classify_imports(self):
        groups = {}
        for stmt, alias, scope in self.sb.imports:
            entry = self._classify(stmt, alias, scope)
            if entry is None:
                continue
            self.entries[id(alias)] = entry
            if entry.kind == 'name':
                continue
            key = (id(scope), entry.local)
            group = groups.get(key)
            if group is None:
                group = Group(scope, entry.local, entry.kind, entry.module)
                groups[key] = group
                self.groups.append(group)
            elif group.kind != entry.kind or group.root != entry.module:
                group.ambiguous = (
                    'bound to %s and %s' % (group.root, entry.module))
            group.entries.append(entry)
        for group in self.groups:
            self._check_group_bindings(group)

    def _classify(self, stmt, alias, scope):
        table = self.table
        if isinstance(stmt, ast.Import):
            if alias.asname:
                if alias.name in table.old:
                    return ImportEntry(
                        stmt, alias, scope, alias.asname, 'old_module',
                        module=alias.name, old_entry=True)
                if alias.name in table.old_prefixes:
                    return ImportEntry(
                        stmt, alias, scope, alias.asname, 'pkg_root',
                        module=alias.name)
                return None
            first = alias.name.split('.')[0]
            if first not in table.old_prefixes and first not in table.old:
                return None
            old_entry = False
            parts = alias.name.split('.')
            for i in range(1, len(parts) + 1):
                if '.'.join(parts[:i]) in table.old:
                    old_entry = True
                    break
            if old_entry and alias.name in table.new and (
                    alias.name not in table.old):
                old_entry = False
            if first in table.old:
                return ImportEntry(
                    stmt, alias, scope, first, 'old_module', module=first,
                    old_entry=True)
            return ImportEntry(
                stmt, alias, scope, first, 'pkg_root', module=first,
                old_entry=old_entry)
        # ImportFrom
        source = _resolve_relative(
            self.module_name, self.is_package, stmt.level, stmt.module)
        if source is None:
            if stmt.level:
                self._check_unresolved_relative(stmt, alias)
            return None
        name = alias.name
        local = alias.asname or name
        full = source + '.' + name
        canonical = table.alias(full)
        if (
            full in table.new and full not in table.old and canonical
            and local.startswith(canonical)
        ):
            # a new module under its canonical alias (also the output of
            # this tool, e.g. "from cellacdc.myutils import misc")
            return None
        if source in table.old:
            names = table.names.get(source) or {}
            if name not in names and full in table.new and (
                    full not in table.old):
                return None
            ref, _ = table.resolve_attr(source, name)
            if isinstance(ref, ModRef) and ref.module in table.old:
                return ImportEntry(
                    stmt, alias, scope, local, 'old_module',
                    module=ref.module, old_entry=True)
            return ImportEntry(
                stmt, alias, scope, local, 'name', ref=ref, source=source)
        if full in table.old:
            return ImportEntry(
                stmt, alias, scope, local, 'old_module', module=full,
                old_entry=True)
        if full in table.old_prefixes:
            return ImportEntry(
                stmt, alias, scope, local, 'pkg_root', module=full)
        return None

    def _check_unresolved_relative(self, stmt, alias):
        names = {_last(module) for module in self.table.old}
        if alias.name in names or (stmt.module or '').split('.')[0] in names:
            self._unhandled(
                stmt.lineno, 'unresolved_relative_import',
                'cannot resolve the relative import of %r without the '
                'module name of the file' % alias.name)

    def _check_group_bindings(self, group):
        if group.ambiguous:
            pass
        else:
            for binding in group.scope.bindings.get(group.local, ()):
                entry = None
                if binding.kind == 'import':
                    entry = self.entries.get(id(binding.alias))
                if entry is None or entry not in group.entries:
                    group.ambiguous = (
                        '%r is also bound by a %s on line %d' % (
                            group.local, binding.kind,
                            getattr(binding.node, 'lineno', 0)))
                    break
        if not group.ambiguous:
            for entry in group.entries:
                if not self._statement_on_own_lines(entry.stmt):
                    group.ambiguous = (
                        'the import on line %d shares its line with other '
                        'code' % entry.stmt.lineno)
                    break
        if group.ambiguous and (group.kind == 'old_module' or any(
                entry.old_entry for entry in group.entries)):
            self._unhandled(
                group.entries[0].stmt.lineno, 'ambiguous_binding',
                '%s (old module %s): imports and uses left unchanged' % (
                    group.ambiguous, group.root))

    # -- uses ---------------------------------------------------------------

    def _chain(self, name_node):
        chain = []
        current = name_node
        while True:
            parent = self.parents.get(current)
            if isinstance(parent, ast.Attribute) and parent.value is current:
                chain.append(parent)
                current = parent
            else:
                break
        return chain

    def _inside_fstring(self, node):
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, ast.JoinedStr):
                return True
            current = self.parents.get(current)
        return False

    def _span_upto(self, name_node, chain, k):
        start = self.src.span(name_node)[0]
        node = name_node if k == 0 else chain[k - 1]
        return start, self.src.span(node)[1]

    def _patch_call(self, expr):
        """(call, mode, attr) when ``expr`` is the target of a patch call."""
        call = self.parents.get(expr)
        if not isinstance(call, ast.Call) or not call.args or (
                call.args[0] is not expr):
            return None
        func = call.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in ('setattr', 'delattr'):
                mode = 'set' if name == 'setattr' else 'del'
            elif name in ('getattr', 'hasattr'):
                mode = 'read'
            else:
                return None
        elif isinstance(func, ast.Attribute):
            if func.attr in ('setattr', 'delattr'):
                mode = 'set' if func.attr == 'setattr' else 'del'
            elif func.attr == 'object' and _is_patch_func(func.value):
                mode = 'set'
            else:
                return None
        else:
            return None
        attr = None
        if len(call.args) >= 2:
            second = call.args[1]
            if isinstance(second, ast.Constant) and isinstance(
                    second.value, str):
                attr = second.value
            elif isinstance(second, ast.Name):
                attr = self._loop_constants(second)
        return call, mode, attr

    def _loop_constants(self, name_node):
        """The string constants a loop variable runs over (``for name in
        ('a', 'b'): setattr(module, name, ...)``) as a tuple, else None."""
        current = self.parents.get(name_node)
        while current is not None and not isinstance(
                current, _SCOPE_NODES):
            if isinstance(current, (ast.For, ast.AsyncFor)) and isinstance(
                    current.target, ast.Name) and (
                    current.target.id == name_node.id) and isinstance(
                    current.iter, (ast.Tuple, ast.List)) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in current.iter.elts) and current.iter.elts:
                return tuple(e.value for e in current.iter.elts)
            current = self.parents.get(current)
        return None

    def _collect_used_homes(self):
        """New modules that this file reads names from (not counting
        patches): imports of new modules and the homes of the names its
        old-module references resolve to (narrows patches, _narrow)."""
        homes = set()
        self.used_homes_by_function = {}

        def add(home, node):
            added = {home} | self.table.class_modules.get(home, set())
            homes.update(added)
            current = self.parents.get(node)
            while current is not None:
                if isinstance(current, (ast.FunctionDef,
                                        ast.AsyncFunctionDef)):
                    self.used_homes_by_function.setdefault(
                        id(current), set()).update(added)
                current = self.parents.get(current)

        for stmt, alias, scope in self.sb.imports:
            if isinstance(stmt, ast.ImportFrom):
                source = _resolve_relative(
                    self.module_name, self.is_package, stmt.level,
                    stmt.module)
                if source is not None:
                    add(source, stmt)
                    add(source + '.' + alias.name, stmt)
            else:
                add(alias.name, stmt)
        for group in self.groups:
            if group.ambiguous:
                continue
            for node, scope in self.sb.loads.get(group.local, ()):
                if self.sb.resolve(scope, group.local) is not group.scope:
                    continue
                chain = self._chain(node)
                top = chain[-1] if chain else node
                if self._patch_call(top) is not None:
                    continue
                parent = self.parents.get(top)
                if isinstance(parent, ast.Assign) and top in parent.targets:
                    continue
                attrs = [n.attr for n in chain]
                res = self.table.resolve_chain(
                    group.root, attrs, group.kind == 'pkg_root')
                if res.kind == 'name' and _is_cellacdc(res.home):
                    add(res.home, node)
                elif res.kind == 'module' and _is_cellacdc(res.module or ''):
                    add(res.module, node)
        for entry in self.entries.values():
            if entry.kind == 'name' and isinstance(entry.ref, NameRef):
                add(entry.ref.home, entry.stmt)
        return homes

    def _process_group(self, group):
        for node, scope in self.sb.loads.get(group.local, ()):
            if self.sb.resolve(scope, group.local) is group.scope:
                group.uses.append(node)
        if group.ambiguous:
            reported = group.kind == 'old_module' or any(
                entry.old_entry for entry in group.entries)
            if not reported:
                for node in group.uses:
                    attrs = [n.attr for n in self._chain(node)]
                    res = self.table.resolve_chain(
                        group.root, attrs, literal=True)
                    if res.kind != 'unchanged':
                        self._unhandled(
                            node.lineno, 'ambiguous_binding',
                            '%s: %s; left unchanged' % (
                                '.'.join([group.local] + attrs),
                                group.ambiguous))
            return
        for node in group.uses:
            self._process_use(group, node)

    def _keep(self, group, line, kind, detail):
        """A use that is not rewritten: the old import stays."""
        group.keep = True
        if group.kind == 'pkg_root':
            group.needs_root = True
        self._unhandled(line, kind, detail)

    def _process_use(self, group, name_node):
        chain = self._chain(name_node)
        attrs = [node.attr for node in chain]
        top = chain[-1] if chain else name_node
        line = name_node.lineno
        dotted = '.'.join([group.local] + attrs)
        literal = group.kind == 'pkg_root'
        if not FSTRING_POSITIONS_OK and self._inside_fstring(name_node):
            res = self.table.resolve_chain(group.root, attrs, literal)
            if res.kind != 'unchanged':
                self._keep(
                    group, line, 'fstring_reference',
                    '%s inside an f-string (positions need Python >= '
                    '3.12)' % dotted)
            elif group.kind == 'pkg_root':
                group.needs_root = True
            return
        res = self.table.resolve_chain(group.root, attrs, literal)
        if res.guessed:
            self._unhandled(
                line, 'reexport_guessed',
                '%s: %s is not in the name table; assumed to be the '
                're-exported old module %s' % (
                    dotted, res.guessed,
                    res.guessed.rpartition('.')[0].rpartition('.')[0] + '.'
                    + res.guessed.rpartition('.')[2]))
        patch = self._patch_call(top)
        if patch is not None and res.denotes_old_module():
            self._process_patch_object(group, name_node, chain, res, patch)
            return
        if res.denotes_old_module() and res.consumed == len(attrs) and (
                res.kind == 'value_old' or self.table.shim_kind(
                    res.old_module) not in (None, 'alias', 'none')) and (
                self._process_alias_assignment(group, top, res.old_module)):
            return
        kind = res.kind
        if kind == 'unchanged':
            if group.kind == 'pkg_root':
                group.needs_root = True
            else:
                self._keep(
                    group, line, 'unchanged_use',
                    '%s keeps the old import (%s)' % (dotted, group.root))
            return
        if kind == 'unknown':
            self._keep(
                group, line, 'unknown_name',
                '%s: %r is not in the name table for %s' % (
                    dotted, res.attr, res.old_module))
            return
        if kind == 'value_old':
            self._keep(
                group, line, 'value_use',
                '%s: the old split module %s is used as a value' % (
                    dotted, res.old_module))
            return
        if kind == 'module':
            start, end = self._span_upto(name_node, chain, res.consumed)
            preferred = None
            if not _is_cellacdc(res.module):
                preferred = (
                    attrs[res.consumed - 1] if res.consumed
                    else group.local)
            target = ('mod', res.module, preferred)
            self._add_use(group, start, end, target, end, line,
                          'reference')
            if res.value and res.module != res.old_module:
                detail = (
                    '%s (old module %s used as a value) now refers to %s' % (
                        dotted, res.old_module, res.module))
                if self.table.shim_kind(res.old_module) in (
                        None, 'alias', 'none'):
                    self._info(line, 'value_use_moved', detail)
                else:
                    # the old path is now a package or a split shim: the
                    # value differs from the old module (source scans)
                    self._unhandled(line, 'value_use_moved', detail)
            return
        # kind == 'name'
        consumed = res.consumed
        is_assign = (
            consumed == len(attrs) and chain
            and isinstance(top.ctx, (ast.Store, ast.Del))
        )
        if is_assign:
            self._process_assign_patch(group, name_node, chain, res)
            return
        self._rewrite_name_use(group, name_node, chain, res, line)

    def _process_alias_assignment(self, group, value, old_module):
        """``mv = <old split module>`` in a function, ``mv`` assigned once
        and used only as ``mv.X...``: every ``mv.X`` becomes the home
        module of ``X`` and the assignment goes. Returns False (nothing
        done) when a use cannot be resolved that way."""
        stmt = self.parents.get(value)
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.value is value):
            return False
        if not self._statement_on_own_lines(stmt):
            return False
        local = stmt.targets[0].id
        scope = None
        for binding in self.sb.bindings.get(local, ()):
            if binding.node is stmt.targets[0]:
                scope = binding.scope
        if scope is None or scope.kind != 'function' or len(
                scope.bindings.get(local, ())) != 1:
            return False
        uses = []
        for node, load_scope in self.sb.loads.get(local, ()):
            if self.sb.resolve(load_scope, local) is not scope:
                continue
            chain = self._chain(node)
            if not chain or isinstance(chain[0].ctx, (ast.Store, ast.Del)):
                return False
            attrs = [n.attr for n in chain]
            sub = self.table.resolve_chain(old_module, attrs, literal=False)
            if sub.kind != 'name' or not _is_cellacdc(sub.home) or (
                    sub.home == self.module_name) or (
                    sub.consumed == len(attrs) and isinstance(
                        chain[-1].ctx, (ast.Store, ast.Del))):
                return False
            uses.append((node, chain, sub))
        if not uses:
            return False
        for node, chain, sub in uses:
            start, prefix_end = self.src.span(node)
            if sub.name != sub.attr:
                x_end = self._span_upto(node, chain, sub.consumed)[1]
                self._add_use(group, start, x_end, ('mod', sub.home, None),
                              x_end, node.lineno, 'reference',
                              suffix='.' + sub.name)
            else:
                x_end = self._span_upto(node, chain, sub.consumed)[1]
                self._add_use(group, start, prefix_end,
                              ('mod', sub.home, None), x_end, node.lineno,
                              'reference')
        line_start, _, _ = self.src.line_bounds(stmt.lineno)
        _, _, end_nl = self.src.line_bounds(stmt.end_lineno)
        self.string_edits.append((line_start, end_nl, ''))
        self._info(stmt.lineno, 'alias_assignment_removed',
                   '%s = %s: its %d use(s) now name the home modules' % (
                       local, old_module, len(uses)))
        return True

    def _rewrite_name_use(self, group, name_node, chain, res, line):
        consumed = res.consumed
        home = res.home
        x_start, x_end = self._span_upto(name_node, chain, consumed)
        prefix_end = self._span_upto(name_node, chain, consumed - 1)[1]
        if self.external_origins and _is_cellacdc(home):
            origin = self._external_origin(home, res.name)
            if origin is not None:
                kind, module, attr = origin
                if kind == 'module':
                    target = ('mod', module, res.name)
                else:
                    target = ('bare', module, attr, res.name)
                self._add_use(group, x_start, x_end, target, x_end, line,
                              'reference')
                self._info(line, 'external_origin', '%s.%s -> %s' % (
                    home, res.name, module if kind == 'module'
                    else module + '.' + attr))
                return
        if home == self.module_name:
            scope = self.sb.load_scope.get(name_node, self.sb.module)
            resolved = self.sb.resolve(scope, res.name)
            if resolved is not None and resolved is not self.sb.module:
                self._keep(
                    group, line, 'self_reference_shadowed',
                    '%s lives in this module, but %r is shadowed here' % (
                        res.name, res.name))
                return
            self.use_edits.append(UseEdit(
                x_start, x_end, ('self', res.name), x_end, line,
                'reference'))
            self._info(line, 'self_reference',
                       '%s now lives in this module' % res.name)
            return
        if not _is_cellacdc(home) and home not in self.table.identity:
            target = ('bare', home, res.name, res.name)
            self._add_use(group, x_start, x_end, target, x_end, line,
                          'reference')
            return
        if res.name != res.attr:
            # renamed re-export (from x import a as b): b -> home.a
            self._add_use(group, x_start, x_end, ('mod', home, None),
                          x_end, line, 'reference', suffix='.' + res.name)
            return
        self._add_use(group, x_start, prefix_end, ('mod', home, None),
                      x_end, line, 'reference')

    def _external_origin(self, home, name):
        info = self.index.info(home)
        if info is None or name in info.defines:
            return None
        origin = info.imports.get(name)
        if origin is None:
            return None
        kind, module, attr = origin
        if _is_cellacdc(module):
            return None
        return kind, module, attr

    def _add_use(self, group, start, end, target, report_end, line,
                 category, suffix=''):
        edit = UseEdit(start, end, target, report_end, line, category,
                       suffix)
        self.use_edits.append(edit)
        group.edits.append(edit)
        if target not in group.requirements:
            group.requirements.append(target)

    # -- patches ------------------------------------------------------------

    def _lookup_modules(self, module, name):
        """New modules whose code reads the global ``name`` that tests
        patched on the old ``module``.

        The home of ``name`` when its definition moved there from
        ``module`` (other modules use attribute access, rule 6). With a
        source index also every new module holding code of ``module``
        (``carved``) that binds ``name`` by an import and reads it
        (re-exports such as ``myutils.printl``, Qt names, module
        objects); for a re-export (``module`` did not define ``name``,
        table ``defined``) the module that defines it is patched only
        when no such module exists (the old patch never reached it).
        Without the index a re-export of a name defined elsewhere cannot
        be retargeted.
        """
        ref, _ = self.table.resolve_attr(module, name, literal=False)
        if ref is None:
            return None, '%r is not in the name table for %s' % (
                name, module)
        home = ref.home if isinstance(ref, NameRef) else None
        carved = self.table.carved(module)
        moved_here = home is not None and home in carved
        defined = self.table.defined.get(module)
        definition = moved_here if defined is None else name in defined
        if self.index is None:
            if moved_here:
                return [home], None
            return None, (
                '%s.%s is a re-export of %s: pass --src-root (the new '
                'layout) to find the modules that read it' % (
                    module, name, home or 'a module object'))
        candidates = []
        for candidate in sorted(carved):
            if candidate == home and definition:
                continue
            info = self.index.info(candidate)
            if info is not None and name in info.imports and (
                    name in info.loads):
                candidates.append(candidate)
        home_defines = False
        if moved_here:
            info = self.index.info(home)
            home_defines = info is None or name in info.defines
        if home_defines and definition:
            result = [home] + candidates
        elif candidates:
            result = candidates
        elif moved_here or (home is not None and _is_cellacdc(home)
                            and not definition):
            result = [home]
        else:
            return None, (
                'no module split off %s reads %s as a global (home %s)' % (
                    module, name, home or 'a module object'))
        return result, None

    def _narrow(self, targets, line, old_target, node=None):
        """Several lookup modules: keep those that the enclosing function
        (else the file) reads names from (the code a test exercises),
        when that leaves some."""
        if len(targets) < 2:
            return targets
        used = []
        current = node
        while current is not None and not used:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                homes = self.used_homes_by_function.get(id(current), ())
                used = [t for t in targets if t in homes]
            current = self.parents.get(current)
        if not used:
            used = [t for t in targets if t in self.used_homes]
        if not used or len(used) == len(targets):
            return targets
        self._info(line, 'patch_lookup_narrowed',
                   '%s: patched in %s (used by this file); not in %s' % (
                       old_target, ', '.join(used), ', '.join(
                           t for t in targets if t not in used)))
        return used

    def _process_patch_object(self, group, name_node, chain, res, patch):
        call, mode, attr = patch
        line = name_node.lineno
        start, end = self._span_upto(name_node, chain, len(chain))
        old_module = res.old_module
        table = self.table
        names = attr if isinstance(attr, tuple) else (
            () if attr is None else (attr,))
        if old_module in table.identity and (
                (names and not any(table.redirects(old_module, n)
                                   for n in names))
                or (attr is None and not table.names.get(old_module)
                    and not table.entries.get(old_module))):
            # the module still exists and keeps the name (or only re-exports
            # old modules): the patch stays
            self._add_use(group, start, end, ('mod', old_module, None), end,
                          line, 'patch')
            return
        dotted = self.text[start:end]
        if isinstance(attr, tuple):
            # a loop over constant names: one lookup module for all of them
            found = set()
            for name in attr:
                targets, error = self._lookup_modules(old_module, name)
                found.add(tuple(targets) if targets else (error,))
            if mode != 'read' and len(found) == 1:
                targets = list(next(iter(found)))
                if len(targets) == 1 and targets[0] in self.table.new:
                    self._add_use(group, start, end, ('mod', targets[0], None),
                                  end, line, 'patch')
                    self.report['patches'].append({
                        'line': line, 'form': 'loop',
                        'target': '%s.{%s}' % (old_module, ','.join(attr)),
                        'new': [targets[0]]})
                    return
            attr = None
        if attr is None:
            self._keep(
                group, line, 'dynamic_patch' if mode != 'read'
                else 'value_use',
                '%s(%s, <not a string>) on the old module %s' % (
                    _call_name(call), dotted, old_module))
            return
        if mode == 'read':
            sub = self.table.resolve_chain(old_module, [attr], literal=False)
            if sub.kind == 'name' and _is_cellacdc(sub.home):
                if sub.home == self.module_name:
                    self._keep(
                        group, line, 'value_use',
                        '%s(%s, %r): %r now lives in this module' % (
                            _call_name(call), dotted, attr, attr))
                    return
                self._add_use(group, start, end, ('mod', sub.home, None),
                              end, line, 'reference')
                return
            if sub.kind == 'name':
                self._add_use(group, start, end, ('mod', sub.home, None),
                              end, line, 'reference')
                return
            self._keep(
                group, line, 'value_use',
                '%s(%s, %r) cannot be resolved (%s)' % (
                    _call_name(call), dotted, attr, sub.kind))
            return
        targets, error = self._lookup_modules(old_module, attr)
        old_target = '%s.%s' % (old_module, attr)
        if targets is None:
            self._keep(group, line, 'patch_unresolved',
                       '%s(%s, %r): %s' % (
                           _call_name(call), dotted, attr, error))
            return
        targets = self._narrow(targets, line, old_target, call)
        entry = {
            'line': line, 'form': 'object', 'target': old_target,
            'new': [t + '.' + attr for t in targets],
        }
        if len(targets) > 1:
            stmt = self.parents.get(call)
            mode = self._can_duplicate(stmt, call)
            if mode:
                self.dup_edits.append(DupPatchEdit(
                    stmt, (start, end), targets, line,
                    value=self._dup_value(call, attr, mode), name=attr))
                for target in targets:
                    requirement = ('mod', target, None)
                    if requirement not in group.requirements:
                        group.requirements.append(requirement)
                entry['note'] = 'repeated once per lookup module'
                self.report['patches'].append(entry)
                return
            entry['note'] = (
                'several lookup modules, only the first is patched: '
                'review')
            self._unhandled(line, 'patch_multiple_lookups',
                            '%s: also read in %s' % (
                                old_target, ', '.join(targets[1:])))
        self._add_use(group, start, end, ('mod', targets[0], None), end,
                      line, 'patch')
        self.report['patches'].append(entry)

    def _process_assign_patch(self, group, name_node, chain, res):
        line = name_node.lineno
        old_target = '%s.%s' % (res.owner, res.name)
        targets, error = self._lookup_modules(res.owner, res.name)
        if targets is None:
            self._keep(group, line, 'patch_unresolved',
                       'assignment to %s: %s' % (old_target, error))
            return
        targets = self._narrow(targets, line, old_target, name_node)
        start = self.src.span(name_node)[0]
        prefix_end = self._span_upto(name_node, chain, res.consumed - 1)[1]
        x_end = self._span_upto(name_node, chain, res.consumed)[1]
        entry = {
            'line': line, 'form': 'assign', 'target': old_target,
            'new': [t + '.' + res.name for t in targets],
        }
        stmt = self._assign_statement(chain[res.consumed - 1])
        if len(targets) > 1 and stmt is not None:
            self.dup_edits.append(DupPatchEdit(
                stmt, (start, prefix_end), targets, line, name=res.name))
            for target in targets:
                requirement = ('mod', target, None)
                if requirement not in group.requirements:
                    group.requirements.append(requirement)
            entry['note'] = 'repeated once per lookup module'
            self.report['patches'].append(entry)
            return
        if len(targets) > 1:
            entry['note'] = (
                'several lookup modules, only the first is patched: '
                'review')
            self._unhandled(line, 'patch_multiple_lookups',
                            '%s: also read in %s' % (
                                old_target, ', '.join(targets[1:])))
        self._add_use(group, start, prefix_end, ('mod', targets[0], None),
                      x_end, line, 'patch')
        self.report['patches'].append(entry)

    def _can_duplicate(self, stmt, call):
        """How a patch statement can be repeated per lookup module:
        'copy' (the values are simple: the statement is copied), 'ref'
        (``setattr(obj, 'name', <expression>)``: the copies set the
        attribute of the first patched module, so the expression is
        evaluated once and every module gets the same object) or False.
        """
        if not isinstance(stmt, ast.Expr) or stmt.value is not call:
            return False
        if not self._statement_on_own_lines(stmt):
            return False
        first = call.args[0]
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id in self.tracked_locals:
                if not _within(node, first, self.parents):
                    return False
            if isinstance(node, ast.Constant) and isinstance(
                    node.value, str) and 'cellacdc' in node.value:
                return False
        values = list(call.args[2:]) + [kw.value for kw in call.keywords]
        if all(isinstance(value, (ast.Name, ast.Attribute, ast.Constant,
                                  ast.Lambda)) for value in values):
            return 'copy'
        func = call.func
        is_setattr = (
            (isinstance(func, ast.Name) and func.id == 'setattr')
            or (isinstance(func, ast.Attribute) and func.attr == 'setattr'))
        if is_setattr and len(call.args) == 3 and not any(
                isinstance(a, ast.Starred) for a in call.args):
            return 'ref'
        return False

    def _dup_value(self, call, attr, mode):
        """(span of the value argument, attribute name) for 'ref' copies."""
        if mode != 'ref':
            return None
        return self.src.span(call.args[2]), attr

    def _assign_statement(self, target):
        """The statement ``<target> = <simple value>`` (a single target,
        on its own lines) whose target is ``target``, else None."""
        stmt = self.parents.get(target)
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or (
                stmt.targets[0] is not target):
            return None
        if not self._statement_on_own_lines(stmt):
            return None
        if not isinstance(stmt.value, (ast.Name, ast.Attribute,
                                       ast.Constant, ast.Lambda)):
            return None
        for node in ast.walk(stmt.value):
            if isinstance(node, ast.Name) and node.id in self.tracked_locals:
                return None
        return stmt

    # -- from-imports of names ----------------------------------------------

    def _process_name_entries(self):
        for entry in self.entries.values():
            if entry.kind != 'name':
                continue
            stmt = entry.stmt
            line = stmt.lineno
            name = entry.alias.name
            local = entry.local
            old = '%s.%s' % (entry.source, name)
            if not self._statement_on_own_lines(stmt):
                self._unhandled(
                    line, 'import_shares_line',
                    'from %s import %s: the statement shares its line '
                    'with other code' % (entry.source, name))
                continue
            ref = entry.ref
            if ref is None:
                self._unhandled(
                    line, 'unknown_name',
                    'from %s import %s: not in the name table' % (
                        entry.source, name))
                continue
            stays = isinstance(ref, NameRef) and (
                ref.home == entry.source and ref.name == name)
            if not stays and not self._local_used(entry.scope, local) and (
                    self._can_drop(entry.scope, local, [stmt])):
                self._plan_remove(entry)
                self._dropped(line, old, local)
                continue
            if isinstance(ref, ModRef):
                spec = ('import_mod', ref.module, local)
                self._plan_remove(entry)
                self._plan_spec(stmt, entry.scope, spec)
                self.report['imports'].append({
                    'line': line, 'old': old, 'new': ref.module})
                continue
            home = ref.home
            if home == entry.source and ref.name == name:
                continue    # the name stays in a module that still exists
            if home == self.module_name:
                if local != ref.name:
                    self._unhandled(
                        line, 'self_import',
                        'from %s import %s as %s: %s now lives in this '
                        'module' % (entry.source, name, local, name))
                    continue
                self._plan_remove(entry)
                self._plan(stmt, entry.scope)
                self._info(line, 'self_import_removed',
                           '%s now lives in this module' % name)
                continue
            if self.external_origins and _is_cellacdc(home):
                origin = self._external_origin(home, ref.name)
                if origin is not None:
                    kind, module, attr = origin
                    if kind == 'module':
                        spec = ('import_mod', module, local)
                    else:
                        spec = ('import_from', module, attr, local)
                    self._plan_remove(entry)
                    self._plan_spec(stmt, entry.scope, spec)
                    self.report['imports'].append({
                        'line': line, 'old': old,
                        'new': module if kind == 'module'
                        else module + '.' + attr})
                    continue
            self._plan_remove(entry)
            self._plan_spec(
                stmt, entry.scope, ('import_from', home, ref.name, local))
            self.report['imports'].append(
                {'line': line, 'old': old, 'new': home + '.' + ref.name})

    # -- strings ------------------------------------------------------------

    def _docstring_nodes(self):
        out = set()
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(
                        body[0].value, ast.Constant) and isinstance(
                        body[0].value.value, str):
                    out.add(id(body[0].value))
        return out

    def _patch_string_call(self, node):
        call = self.parents.get(node)
        if not isinstance(call, ast.Call) or not call.args or (
                call.args[0] is not node):
            return None
        func = call.func
        if _is_patch_func(func):
            return call
        if isinstance(func, ast.Attribute) and func.attr in (
                'setattr', 'delattr'):
            return call
        return None

    def _process_strings(self):
        docstrings = self._docstring_nodes()
        old_basenames = sorted(
            {_last(module) for module in self.table.old
             if _is_cellacdc(module)}, key=len, reverse=True)
        path_re = None
        if old_basenames:
            path_re = re.compile(
                r'(?<![\w.])(?:cellacdc[/\\])?(%s)\.py\b' % '|'.join(
                    re.escape(name) for name in old_basenames))
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Constant) and isinstance(
                    node.value, str)):
                continue
            parent = self.parents.get(node)
            in_fstring = isinstance(parent, (ast.JoinedStr,
                                             ast.FormattedValue))
            is_docstring = id(node) in docstrings
            value = node.value
            if (
                not in_fstring and not is_docstring
                and _DOTTED_RE.match(value) and self._string_is_old(value)
            ):
                self._process_dotted_string(node, value)
                continue
            if not in_fstring and not is_docstring and (
                    self._process_code_string(node, value)):
                continue
            self._report_annotation_string(node, value)
            self._report_mentions(node, value, path_re, in_fstring,
                                  is_docstring)

    def _process_code_string(self, node, value):
        """Python code in a string of a test (a script run in a
        subprocess, ``textwrap.dedent``-ed or not) is rewritten like a
        module when it imports cellacdc. Strings checking the layout
        (``__path__``) are left alone (the old branch is meant). Returns
        True when the string was handled as code."""
        if '\n' not in value or 'cellacdc' not in value or (
                '__path__' in value) or not self._is_test_file():
            return False
        if not re.search(r'^\s*(from|import)\s+cellacdc\b', value, re.M):
            return False
        code = textwrap.dedent(value)
        try:
            _parse(code)
        except (SyntaxError, ValueError):
            return False
        rewriter = FileRewriter(
            code, self.table, module_name=None, is_package=False,
            path=self.path, index=self.index, parts=self.parts,
            external_origins=self.external_origins,
            drop_unused=self.drop_unused, external_uses=None)
        rewriter.in_code_string = True
        new_code = rewriter.run()
        for key, entries in rewriter.report.items():
            for entry in entries:
                entry = dict(entry)
                if 'line' in entry:
                    entry['line'] += node.lineno - 1
                entry['code_string'] = True
                self.report[key].append(entry)
        if new_code == code:
            return True
        old_lines = value.split('\n')
        margin = ''
        for old, dedented in zip(old_lines, code.split('\n')):
            if dedented.strip():
                margin = old[:len(old) - len(dedented)]
                break
        new_lines = new_code.split('\n')
        matcher = difflib.SequenceMatcher(
            a=code.split('\n'), b=new_lines, autojunk=False)
        out = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                out.extend(old_lines[i1:i2])
            else:
                out.extend(margin + line if line.strip() else line
                           for line in new_lines[j1:j2])
        new_value = '\n'.join(out)
        segment = self.src.segment(node)
        match = re.match(r'^([rRuU]?)(\'\'\'|"""|\'|")', segment)
        quote = match.group(2) if match else None
        start, end = self.src.span(node)
        body_start = start + len(match.group(0)) if match else start
        body_end = end - len(quote) if quote else end
        body = self.text[body_start:body_end]
        newline = '\r\n' if '\r\n' in body else '\n'
        if (not match or not segment.endswith(quote)
                or body.replace('\r\n', '\n') != value
                or '\\' in value):
            self._unhandled(node.lineno, 'string_not_spliceable',
                            'code in a string should be rewritten '
                            '(escapes or implicit concatenation)')
            return True
        self.string_edits.append(
            (body_start, body_end, new_value.replace('\n', newline)))
        self._info(node.lineno, 'code_string_rewritten',
                   'Python code in a string (%d lines)' % len(old_lines))
        return True

    def _report_annotation_string(self, node, value):
        """String annotations that name an old module alias of the file."""
        current, parent = node, self.parents.get(node)
        while isinstance(parent, (ast.Subscript, ast.Tuple, ast.BinOp,
                                  ast.Attribute)):
            current, parent = parent, self.parents.get(parent)
        is_annotation = (
            (isinstance(parent, ast.arg) and parent.annotation is current)
            or (isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                and parent.returns is current)
            or (isinstance(parent, ast.AnnAssign)
                and parent.annotation is current)
        )
        if not is_annotation:
            return
        for match in re.finditer(r'\b([A-Za-z_]\w*)\.', value):
            if match.group(1) in self.tracked_locals:
                self._unhandled(
                    node.lineno, 'string_annotation',
                    'the string annotation %r names %r: not rewritten' % (
                        value, match.group(1)))
                return

    def _string_is_old(self, value):
        parts = value.split('.')
        root = parts[0]
        return root in self.table.old or root in self.table.old_prefixes

    def _report_mentions(self, node, value, path_re, in_fstring,
                         is_docstring):
        kind = 'docstring' if is_docstring else (
            'fstring' if in_fstring else 'string')
        found = []
        for match in _MENTION_RE.finditer(value):
            dotted = match.group(0)
            parts = dotted.split('.')
            res = self.table.resolve_chain(parts[0], parts[1:])
            if res.kind != 'unchanged':
                found.append(dotted)
        if path_re is not None:
            for match in path_re.finditer(value):
                found.append(match.group(0))
        for text in found:
            self.report['mentions'].append(
                {'line': node.lineno, 'kind': kind, 'text': text})

    def _process_dotted_string(self, node, value):
        line = node.lineno
        parts = value.split('.')
        res = self.table.resolve_chain(parts[0], parts[1:])
        patch_call = self._patch_string_call(node)
        if res.kind == 'unchanged':
            return
        if res.kind in ('unknown', 'value_old'):
            what = (
                '%r is not in the name table for %s' % (
                    res.attr, res.old_module) if res.kind == 'unknown'
                else 'names the old split module %s' % res.old_module)
            self._unhandled(line, 'string_old_path',
                            '%r: %s' % (value, what))
            return
        rest = parts[1 + res.consumed:]
        if res.kind == 'module':
            if res.value and value == res.old_module and (
                    value in self.table.new):
                self._unhandled(
                    line, 'string_ambiguous',
                    '%r is an old module path that still exists as a '
                    'package (new module: %s)' % (value, res.module))
                return
            new = '.'.join([res.module] + rest)
            self._add_string_edit(node, value, new, line, None)
            return
        # 'name'
        if patch_call is not None and not rest:
            targets, error = self._lookup_modules(res.owner, res.name)
            if targets is None:
                self._unhandled(line, 'patch_unresolved',
                                '%r: %s' % (value, error))
                return
            targets = self._narrow(targets, line, value, node)
            new = targets[0] + '.' + res.name
            entry = {
                'line': line, 'form': 'string', 'target': value,
                'new': [t + '.' + res.name for t in targets],
            }
            if len(targets) > 1:
                entry['note'] = (
                    'several lookup modules, only the first is patched: '
                    'review')
                self._unhandled(line, 'patch_multiple_lookups',
                                '%r: also read in %s' % (
                                    value, ', '.join(targets[1:])))
            if self._add_string_edit(node, value, new, line, 'patch'):
                self.report['patches'].append(entry)
            return
        new = '.'.join([res.home, res.name] + rest)
        if new == value:
            return
        if patch_call is not None:
            # patches an attribute of a moved class: the class's new home
            if self._add_string_edit(node, value, new, line, 'patch'):
                self.report['patches'].append({
                    'line': line, 'form': 'string', 'target': value,
                    'new': [new]})
            return
        self._add_string_edit(node, value, new, line, None)

    def _add_string_edit(self, node, value, new, line, category):
        segment = self.src.segment(node)
        match = _SIMPLE_STRING_RE.match(segment)
        if (
            not match or match.group(3) != value
            or segment.startswith(("'''", '"""'))
        ):
            self._unhandled(
                line, 'string_not_spliceable',
                '%r should become %r (escapes, implicit concatenation or '
                'triple quotes)' % (value, new))
            return False
        start, _ = self.src.span(node)
        body_start = start + len(match.group(1)) + 1
        self.string_edits.append(
            (body_start, body_start + len(value), new))
        if category != 'patch':
            self.report['strings'].append(
                {'line': line, 'old': value, 'new': new})
        return True

    # -- unused imports -----------------------------------------------------

    def _local_used(self, scope, local):
        for node, load_scope in self.sb.loads.get(local, ()):
            if self.sb.resolve(load_scope, local) is scope:
                return True
        return False

    def _module_all(self):
        names = set()
        for stmt in self.tree.body:
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
                targets = [stmt.target]
            if not any(isinstance(t, ast.Name) and t.id == '__all__'
                       for t in targets):
                continue
            for node in ast.walk(stmt.value) if stmt.value else ():
                if isinstance(node, ast.Constant) and isinstance(
                        node.value, str):
                    names.add(node.value)
        return names

    def _annotation_names(self):
        """Identifiers named by string annotations of the file."""
        if self._annotation_cache is not None:
            return self._annotation_cache
        names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(
                    node.value, str) and self._is_annotation(node):
                names.update(re.findall(r'[A-Za-z_]\w*', node.value))
        self._annotation_cache = names
        return names

    def _is_annotation(self, node):
        current, parent = node, self.parents.get(node)
        while isinstance(parent, (ast.Subscript, ast.Tuple, ast.BinOp,
                                  ast.Attribute)):
            current, parent = parent, self.parents.get(parent)
        return (
            (isinstance(parent, ast.arg) and parent.annotation is current)
            or (isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                and parent.returns is current)
            or (isinstance(parent, ast.AnnAssign)
                and parent.annotation is current)
        )

    def _can_drop(self, scope, local, stmts):
        """An unused import binding (of an old module or of a name
        imported from one) may go: no ``noqa`` on its statement, not in
        ``__all__``, not named by a string annotation, and, at module
        level, not read by another file through this module unless the
        name table redirects those reads."""
        if not self.drop_unused:
            return False
        for stmt in stmts:
            for lineno in range(stmt.lineno, stmt.end_lineno + 1):
                if 'noqa' in self.src.line(lineno):
                    return False
        if local in self._annotation_names():
            return False
        if scope is not self.sb.module or not self.module_name:
            return True
        if local in self._module_all():
            return False
        if self.external_uses is None:
            return False
        users = set(self.external_uses.get((self.module_name, local), ()))
        users.discard(os.path.abspath(self.path))
        if users and not self.table.redirects(self.module_name, local):
            return False
        return True

    def _is_test_file(self):
        """Tests import whole modules for their side effects
        (test_import_cellacdc): unused imports of modules that moved
        whole are rewritten there, never dropped."""
        if self.in_code_string:
            return True
        if self.module_name:
            return self.module_name.split('.')[0] == 'tests'
        parts = os.path.abspath(self.path).replace(os.sep, '/').split('/')
        return 'tests' in parts

    def _dropped(self, line, imported, local):
        self.report['dropped'].append(
            {'line': line, 'import': imported, 'local': local})

    # -- decisions ----------------------------------------------------------

    def _plan(self, stmt, scope):
        plan = self.plans.get(id(stmt))
        if plan is None:
            plan = {'stmt': stmt, 'scope': scope, 'remove': [],
                    'specs': []}
            self.plans[id(stmt)] = plan
            self.plan_order.append(plan)
        return plan

    def _plan_remove(self, entry):
        plan = self._plan(entry.stmt, entry.scope)
        if entry.alias not in plan['remove']:
            plan['remove'].append(entry.alias)
        entry.removed = True

    def _plan_spec(self, stmt, scope, spec):
        plan = self._plan(stmt, scope)
        if spec not in plan['specs']:
            plan['specs'].append(spec)

    def _decide_groups(self):
        for group in self.groups:
            if group.ambiguous:
                continue
            if group.kind == 'old_module':
                self._decide_module_group(group)
            else:
                self._decide_pkg_group(group)

    def _decide_module_group(self, group):
        line = group.entries[0].stmt.lineno
        if group.root in self.table.identity:
            # the module still exists: keep its import and the uses of the
            # names it still defines, add imports for names that left
            own = ('mod', group.root, None)
            dropped = {id(e) for e in group.edits if e.target == own}
            if dropped:
                self.use_edits = [
                    e for e in self.use_edits if id(e) not in dropped]
            requirements = [r for r in group.requirements if r != own]
            if group.uses and not dropped and not group.keep and (
                    self._can_drop(group.scope, group.local,
                                   [e.stmt for e in group.entries])):
                # every use names another module now (window.py's
                # `whitelist.WhitelistGUIElements`): the import of the
                # module that still exists is left unused
                for entry in group.entries:
                    self._plan_remove(entry)
                self._dropped(line, group.root, group.local)
            for entry in group.entries:
                for requirement in requirements:
                    self._plan_spec(entry.stmt, entry.scope,
                                    ('need',) + requirement)
            if requirements:
                self.report['imports'].append({
                    'line': line, 'old': group.root,
                    'new': sorted({r[1] for r in requirements}),
                    'kept': True,
                })
            return
        # unused imports of modules moved whole stay (rewritten) in tests
        keep_whole = group.root in self.table.modules and (
            self._is_test_file())
        if not group.uses and not keep_whole and self._can_drop(
                group.scope, group.local, [e.stmt for e in group.entries]):
            for entry in group.entries:
                self._plan_remove(entry)
            self._dropped(line, group.root, group.local)
            return
        if not group.uses:
            if group.root in self.table.modules:
                new = self.table.modules[group.root]
                for entry in group.entries:
                    self._plan_remove(entry)
                    self._plan_spec(entry.stmt, entry.scope,
                                    ('import_mod', new, group.local))
                self._info(line, 'unused_import_rewritten',
                           '%s is not used here; imported from its new '
                           'location %s as %s' % (
                               group.root, new, group.local))
            else:
                self._unhandled(
                    line, 'unused_import',
                    '%s (bound to %r) has no uses here: kept for its side '
                    'effects or re-export' % (group.root, group.local))
            return
        for entry in group.entries:
            if not group.keep:
                self._plan_remove(entry)
            for requirement in group.requirements:
                self._plan_spec(entry.stmt, entry.scope,
                                ('need',) + requirement)
            if group.requirements or not group.keep:
                self.report['imports'].append({
                    'line': entry.stmt.lineno, 'old': group.root,
                    'new': sorted({r[1] for r in group.requirements}),
                    'kept': group.keep,
                })

    def _decide_pkg_group(self, group):
        old_entries = [e for e in group.entries if e.old_entry]
        if not old_entries and not group.requirements:
            return
        anchor = old_entries[0] if old_entries else group.entries[0]
        moved = [e for e in old_entries
                 if not e.alias.asname and e.alias.name in self.table.modules]
        if old_entries and not group.requirements and not group.keep and (
                moved):
            # `import cellacdc.<moved whole>` (side effects, or the binding
            # of cellacdc): import the module at its new path
            for entry in moved:
                self._plan_remove(entry)
                self._plan_spec(entry.stmt, entry.scope, (
                    'import_plain', self.table.modules[entry.alias.name]))
                self.report['imports'].append({
                    'line': entry.stmt.lineno, 'old': entry.alias.name,
                    'new': self.table.modules[entry.alias.name],
                    'kept': False})
            old_entries = [e for e in old_entries if e not in moved]
            if not old_entries:
                return
        if old_entries and not group.requirements and not group.keep and (
                self._can_drop(group.scope, group.local,
                               [e.stmt for e in old_entries])):
            for entry in old_entries:
                self._plan_remove(entry)
                self._dropped(entry.stmt.lineno, entry.alias.name,
                              group.local)
            remaining = [e for e in group.entries if not e.removed]
            if group.needs_root and not remaining:
                self._plan_spec(anchor.stmt, anchor.scope,
                                ('import_mod', group.root, group.local))
            return
        if old_entries and not group.requirements and not group.keep:
            for entry in old_entries:
                self._unhandled(
                    entry.stmt.lineno, 'unused_import',
                    'import %s has no uses through %r here: kept for its '
                    'side effects' % (entry.alias.name, group.local))
            return
        if not group.keep:
            for entry in old_entries:
                self._plan_remove(entry)
            remaining = [e for e in group.entries if not e.removed]
            if group.needs_root and not remaining:
                self._plan_spec(anchor.stmt, anchor.scope,
                                ('import_mod', group.root, group.local))
        for requirement in group.requirements:
            self._plan_spec(anchor.stmt, anchor.scope,
                            ('need',) + requirement)
        if old_entries or group.requirements:
            self.report['imports'].append({
                'line': anchor.stmt.lineno,
                'old': sorted({e.alias.name for e in old_entries}),
                'new': sorted({r[1] for r in group.requirements}),
                'kept': group.keep,
            })

    # -- aliases ------------------------------------------------------------

    def _removed_alias_ids(self):
        out = set()
        for plan in self.plan_order:
            for alias in plan['remove']:
                out.add(id(alias))
        return out

    def _binding_equivalent(self, binding, key):
        if binding.kind != 'import':
            return False
        stmt, alias = binding.node, binding.alias
        if key[0] == 'mod':
            module = key[1]
            if isinstance(stmt, ast.Import):
                if alias.asname:
                    return alias.name == module
                return alias.name == module and '.' not in module
            source = _resolve_relative(
                self.module_name, self.is_package, stmt.level, stmt.module)
            return source is not None and source + '.' + alias.name == module
        module, name = key[1], key[2]
        if isinstance(stmt, ast.ImportFrom):
            source = _resolve_relative(
                self.module_name, self.is_package, stmt.level, stmt.module)
            return source == module and alias.name == name
        return False

    def _collides(self, name, key):
        removed = self._removed_ids
        for binding in self.sb.bindings.get(name, ()):
            if binding.kind == 'import' and id(binding.alias) in removed:
                continue
            if self._binding_equivalent(binding, key):
                continue
            return True
        for load in self.sb.loads.get(name, ()):
            if self.sb.resolve(load[1], name) is None:
                return True
        return False

    def _module_alias(self, module, preferred=None, line=0):
        if module in self.alias_of:
            return self.alias_of[module]
        base = preferred or self.table.alias(module) or _last(module)
        candidates = [base, base + '_mod'] + [
            '%s_mod%d' % (base, i) for i in range(2, 50)]
        for candidate in candidates:
            owner = self.alias_owner.get(candidate)
            if owner is not None and owner != ('mod', module):
                continue
            if self._collides(candidate, ('mod', module)):
                continue
            if candidate != base:
                self._unhandled(
                    line, 'alias_collision',
                    'the alias %r of %s collides with a name of the file: '
                    'using %r' % (base, module, candidate))
            self.alias_of[module] = candidate
            self.alias_owner[candidate] = ('mod', module)
            return candidate
        raise RuntimeError('no free alias for %s' % module)

    def _bare(self, module, name, local, line=0):
        key = (module, name, local)
        if key in self.bare_access:
            return self.bare_access[key]
        owner = self.alias_owner.get(local)
        if (owner is None or owner == ('from', module, name)) and (
                not self._collides(local, ('from', module, name))):
            self.alias_owner[local] = ('from', module, name)
            result = (local, ('import_from', module, name, local))
        else:
            alias = self._module_alias(module, line=line)
            self._unhandled(
                line, 'alias_collision',
                'the name %r (from %s) collides with a name of the file: '
                'using %s.%s' % (local, module, alias, name))
            result = ('%s.%s' % (alias, name),
                      ('import_mod', module, alias))
        self.bare_access[key] = result
        return result

    def _access(self, target, line=0):
        """(text, import spec or None) for a use target."""
        kind = target[0]
        if kind == 'self':
            return target[1], None
        if kind == 'mod':
            module, preferred = target[1], target[2]
            alias = self._module_alias(module, preferred, line)
            return alias, ('import_mod', module, alias)
        if kind == 'bare':
            return self._bare(target[1], target[2], target[3], line)
        raise ValueError(target)

    # -- rendering ----------------------------------------------------------

    def _render(self):
        self._removed_ids = self._removed_alias_ids()
        edits = []
        for edit in self.use_edits:
            text, _ = self._access(edit.target, edit.line)
            text += edit.suffix
            edits.append((edit.start, edit.end, text))
            old = self.text[edit.start:edit.report_end]
            new = text + self.text[edit.end:edit.report_end]
            if edit.category == 'reference':
                self.report['references'].append(
                    {'line': edit.line, 'old': old, 'new': new})
        for edit in self.dup_edits:
            edits.append(self._render_dup(edit))
        for start, end, new in self.string_edits:
            edits.append((start, end, new))
        # an import added after an earlier statement of the same block is
        # not repeated (blocks run top to bottom; other blocks may not run)
        emitted = set()
        plans = sorted(
            self.plan_order,
            key=lambda p: (p['stmt'].lineno, p['stmt'].col_offset))
        for plan in plans:
            block = self._block_key(plan['stmt'])
            specs = []
            for spec in self._final_specs(plan):
                if block is not None and (block, spec) in emitted:
                    continue
                specs.append(spec)
                if block is not None:
                    emitted.add((block, spec))
            edits.extend(self._render_plan(plan, specs))
        for entries in self.report.values():
            entries.sort(key=lambda entry: entry.get('line', 0))
        return self._apply(edits)

    def _block_key(self, stmt):
        parent = self.parents.get(stmt)
        if parent is None:
            return None
        for field in ('body', 'orelse', 'finalbody'):
            block = getattr(parent, field, None)
            if isinstance(block, list) and any(s is stmt for s in block):
                return (id(parent), field)
        return None

    def _render_dup(self, edit):
        stmt = edit.stmt
        line_start, _, _ = self.src.line_bounds(stmt.lineno)
        stmt_start, stmt_end = self.src.span(stmt)
        nl = self.src.newline(stmt.end_lineno)
        indent = self.text[line_start:stmt_start]
        arg_start, arg_end = edit.arg_span
        copies = []
        aliases = []
        for k, target in enumerate(edit.targets):
            alias, _ = self._access(('mod', target, None), edit.line)
            aliases.append(alias)
            if k and edit.value is not None:
                (value_start, value_end), attr = edit.value
                copies.append(
                    self.text[stmt_start:arg_start] + alias
                    + self.text[arg_end:value_start] + aliases[0] + '.'
                    + attr + self.text[value_end:stmt_end])
                continue
            copies.append(
                self.text[stmt_start:arg_start] + alias
                + self.text[arg_end:stmt_end])
        if indent.strip() or not edit.name:
            # not alone on its line: a trailing comment on the first copy
            copies[0] += '  # moved: also read in %s' % ', '.join(aliases[1:])
        else:
            # a comment line above the copies (a trailing comment listing
            # the modules made lines of up to 220 characters)
            copies.insert(
                0, '# moved: %s is read in each of these modules' % edit.name)
        text = (nl + indent).join(copies)
        return (stmt_start, stmt_end, text)

    def _final_specs(self, plan):
        specs = []
        for spec in plan['specs']:
            if spec[0] == 'need':
                _, spec = self._access(spec[1:], plan['stmt'].lineno)
                if spec is None:
                    continue
            if spec not in specs:
                specs.append(spec)
        scope = plan['scope']
        out = []
        for spec in specs:
            if self._equivalent_visible(scope, spec):
                continue
            out.append(spec)
        return out

    def _equivalent_visible(self, scope, spec):
        """True when ``scope`` already sees an equivalent import.

        Then no import is added: a function-level import of a name that
        the function also reads from the module level would make that
        name local to the whole function (UnboundLocalError before it).
        """
        if spec[0] == 'import_plain':
            return False
        if spec[0] == 'import_mod':
            local, key = spec[2], ('mod', spec[1])
        else:
            local, key = spec[3] or spec[2], ('from', spec[1], spec[2])
        visible = self.sb.resolve(scope, local)
        if visible is None:
            return False
        found = False
        for binding in visible.bindings.get(local, ()):
            if binding.kind == 'import' and id(binding.alias) in (
                    self._removed_ids):
                continue
            if self._binding_equivalent(binding, key):
                found = True
                continue
            return False
        return found

    def _render_spec_lines(self, specs, indent, nl):
        plain = []
        froms = {}
        for spec in specs:
            if spec[0] == 'import_plain':
                plain.append('import %s' % spec[1])
                continue
            if spec[0] == 'import_mod':
                module, alias = spec[1], spec[2]
                if '.' not in module:
                    plain.append(
                        'import %s' % module if alias == module
                        else 'import %s as %s' % (module, alias))
                    continue
                parent, _, last = module.rpartition('.')
                item = last if alias == last else '%s as %s' % (last, alias)
                froms.setdefault(self._from_source(parent), []).append(item)
            else:
                module, name, local = spec[1], spec[2], spec[3]
                item = name if local in (None, name) else '%s as %s' % (
                    name, local)
                froms.setdefault(self._from_source(module), []).append(item)
        lines = sorted(set(plain))
        for source in sorted(froms, key=lambda s: (s.startswith('.'), s)):
            items = sorted(set(froms[source]))
            if len(items) == 1:
                lines.append('from %s import %s' % (source, items[0]))
                continue
            one = 'from %s import %s' % (source, ', '.join(items))
            if len(indent) + len(one) <= IMPORT_LINE_LENGTH and not any(
                    ' as ' in item for item in items):
                lines.append(one)
                continue
            block = 'from %s import (' % source + nl
            for item in items:
                block += indent + '    ' + item + ',' + nl
            block += indent + ')'
            lines.append(block)
        return lines

    def _render_plan(self, plan, specs):
        stmt = plan['stmt']
        removed = plan['remove']
        if not removed and not specs:
            return []
        stmt_start, stmt_end = self.src.span(stmt)
        line_start, _, _ = self.src.line_bounds(stmt.lineno)
        _, _, last_end_nl = self.src.line_bounds(stmt.end_lineno)
        indent = self.text[line_start:stmt_start]
        nl = self.src.newline(stmt.end_lineno)
        lines = self._render_spec_lines(specs, indent, nl)
        if len(removed) == len(stmt.names):
            if lines:
                new_text = (nl + indent).join(lines)
                return [(stmt_start, stmt_end, new_text)]
            if self._only_statement_of_block(stmt):
                return [(stmt_start, stmt_end, 'pass')]
            return [(line_start, last_end_nl, '')]
        edits = []
        if removed:
            edits.append((stmt_start, stmt_end,
                          self._remove_aliases(stmt, removed)))
        if lines:
            insert = ''.join(indent + line + nl for line in lines)
            if last_end_nl == len(self.text) and not self.text.endswith(
                    ('\n', '\r')):
                insert = nl + insert[:-len(nl)]
            edits.append((last_end_nl, last_end_nl, insert))
        return edits

    def _only_statement_of_block(self, stmt):
        parent = self.parents.get(stmt)
        if parent is None:
            return False
        for field in ('body', 'orelse', 'finalbody'):
            block = getattr(parent, field, None)
            if isinstance(block, list) and stmt in block:
                return len(block) == 1
        return False

    def _remove_aliases(self, stmt, aliases):
        stmt_start, stmt_end = self.src.span(stmt)
        segment = self.text[stmt_start:stmt_end]
        spans = sorted(
            ((self.src.offset(a.lineno, a.col_offset) - stmt_start,
              self.src.offset(a.end_lineno, a.end_col_offset) - stmt_start)
             for a in aliases), reverse=True)
        for start, end in spans:
            j = end
            while j < len(segment) and segment[j] in ' \t':
                j += 1
            if j < len(segment) and segment[j] == ',':
                j += 1
                while j < len(segment) and segment[j] in ' \t':
                    j += 1
                segment = segment[:start] + segment[j:]
                continue
            i = start
            while i > 0 and segment[i - 1] in ' \t\r\n\\':
                i -= 1
            if i > 0 and segment[i - 1] == ',':
                segment = segment[:i - 1] + segment[end:]
            else:
                segment = segment[:start] + segment[end:]
        pieces = _NEWLINE_RE.split(segment)
        newlines = _NEWLINE_RE.findall(segment)
        out = pieces[0]
        for piece, newline in zip(pieces[1:], newlines):
            if not piece.strip():
                continue
            out += newline + piece
        return out

    def _apply(self, edits):
        edits = sorted(edits, key=lambda e: (e[0], e[1]))
        kept = []
        last_end = -1
        for start, end, text in edits:
            if start < last_end:
                line = self.text.count('\n', 0, start) + 1
                self._unhandled(
                    line, 'edit_conflict',
                    'overlapping rewrites: %r not applied' % text)
                continue
            kept.append((start, end, text))
            last_end = max(last_end, end)
        pieces = []
        position = 0
        for start, end, text in kept:
            pieces.append(self.text[position:start])
            pieces.append(text)
            position = end
        pieces.append(self.text[position:])
        new_text = ''.join(pieces)
        if new_text != self.text:
            old_lines = set(_NEWLINE_RE.split(self.text))
            for i, line in enumerate(_NEWLINE_RE.split(new_text), 1):
                if len(line) > MAX_LINE_LENGTH and line not in old_lines:
                    self.report['long_lines'].append(
                        {'line': i, 'length': len(line)})
        return new_text


def _is_patch_func(node):
    if isinstance(node, ast.Name):
        return node.id == 'patch'
    return isinstance(node, ast.Attribute) and node.attr == 'patch'


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return 'call'


def _within(node, ancestor, parents):
    current = node
    while current is not None:
        if current is ancestor:
            return True
        current = parents.get(current)
    return False


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def module_name_for_path(path):
    """(dotted module name, is_package) from the ``__init__.py`` chain.

    A file outside any package gets None (it can only use absolute
    imports).
    """
    path = os.path.abspath(path)
    directory, filename = os.path.split(path)
    stem = os.path.splitext(filename)[0]
    is_package = stem == '__init__'
    if not os.path.isfile(os.path.join(directory, '__init__.py')):
        return None, False
    parts = [] if is_package else [stem]
    while os.path.isfile(os.path.join(directory, '__init__.py')):
        parent, name = os.path.split(directory)
        parts.insert(0, name)
        if parent == directory:
            break
        directory = parent
    return '.'.join(parts), is_package


REPORT_KEYS = (
    'references', 'imports', 'patches', 'strings', 'unhandled', 'mentions',
    'long_lines', 'info', 'dropped',
)


def _empty_report():
    return {key: [] for key in REPORT_KEYS}


def rewrite_source(text, table, module_name=None, is_package=False,
                   path='<string>', index=None, parts=DEFAULT_PARTS,
                   external_origins=False, drop_unused=False,
                   external_uses=None):
    """Rewrite one source text; returns ``(new_text, report)``."""
    rewriter = FileRewriter(
        text, table, module_name=module_name, is_package=is_package,
        path=path, index=index, parts=parts,
        external_origins=external_origins, drop_unused=drop_unused,
        external_uses=external_uses)
    new_text = rewriter.run()
    return new_text, rewriter.report


def rewrite_file(path, table, write=True, index=None, parts=DEFAULT_PARTS,
                 external_origins=False, module_name=None,
                 is_package=None, drop_unused=False, external_uses=None):
    """Rewrite a file in place; returns ``(changed, report)``."""
    if path.endswith('.ipynb'):
        return rewrite_notebook(
            path, table, write=write, index=index, parts=parts,
            external_origins=external_origins, drop_unused=drop_unused,
            external_uses=external_uses)
    with open(path, 'rb') as file:
        data = file.read()
    try:
        text, encoding = _decode(data)
    except (SyntaxError, UnicodeDecodeError, LookupError) as error:
        report = _empty_report()
        report['unhandled'].append(
            {'line': 0, 'kind': 'decode_error', 'detail': str(error)})
        return False, report
    if module_name is None:
        module_name, detected_package = module_name_for_path(path)
        if is_package is None:
            is_package = detected_package
    new_text, report = rewrite_source(
        text, table, module_name=module_name, is_package=bool(is_package),
        path=path, index=index, parts=parts,
        external_origins=external_origins, drop_unused=drop_unused,
        external_uses=external_uses)
    if new_text == text:
        return False, report
    new_data = new_text.encode(encoding)
    if new_data == data:
        return False, report
    if write:
        with open(path, 'wb') as file:
            file.write(new_data)
    return True, report


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------

_CELL_MARKER = '# acdc-refactor-notebook-cell %d'
_CELL_MARKER_RE = re.compile(r'^# acdc-refactor-notebook-cell (\d+)$', re.M)
_MAGIC_RE = re.compile(r'^(\s*)([%!?].*)$')
_MAGIC_MASK = '#acdc-refactor-magic '


def _cell_source(cell):
    source = cell.get('source', '')
    return ''.join(source) if isinstance(source, list) else source


def _set_cell_source(cell, text):
    if isinstance(cell.get('source'), list):
        cell['source'] = text.splitlines(True)
    else:
        cell['source'] = text


def notebook_code(text):
    """(combined Python source of the code cells, [(cell index, ...)]);
    IPython magics and shell lines are masked as comments, each cell
    starts with a marker comment line."""
    notebook = json.loads(text)
    pieces = []
    cells = []
    for i, cell in enumerate(notebook.get('cells', ())):
        if cell.get('cell_type') != 'code':
            continue
        lines = []
        for line in _cell_source(cell).split('\n'):
            match = _MAGIC_RE.match(line)
            if match:
                line = match.group(1) + _MAGIC_MASK + match.group(2)
            lines.append(line)
        pieces.append(_CELL_MARKER % i + '\n' + '\n'.join(lines))
        cells.append(i)
    return notebook, '\n'.join(pieces) + '\n', cells


def _split_notebook_code(code):
    """{cell index: source} of a combined text of notebook_code."""
    out = {}
    matches = list(_CELL_MARKER_RE.finditer(code))
    for k, match in enumerate(matches):
        start = match.end() + 1
        end = matches[k + 1].start() - 1 if k + 1 < len(matches) else (
            len(code) - 1)
        lines = []
        for line in code[start:end].split('\n'):
            stripped = line.lstrip()
            if stripped.startswith(_MAGIC_MASK):
                line = line[:len(line) - len(stripped)] + stripped[
                    len(_MAGIC_MASK):]
            lines.append(line)
        out[int(match.group(1))] = '\n'.join(lines)
    return out


def rewrite_notebook(path, table, write=True, index=None,
                     parts=DEFAULT_PARTS, external_origins=False,
                     drop_unused=False, external_uses=None):
    """Rewrite the code cells of a Jupyter notebook as one module (cells
    joined in order, magics masked); the JSON is written back with the
    notebook format (indent 1, key order and line endings kept) only when
    it round-trips unchanged."""
    report = _empty_report()
    with open(path, 'rb') as file:
        data = file.read()
    raw = data.decode('utf-8')
    eol = '\r\n' if '\r\n' in raw else '\n'
    text = raw.replace('\r\n', '\n')
    try:
        notebook, code, cells = notebook_code(text)
    except ValueError as error:
        report['unhandled'].append(
            {'line': 0, 'kind': 'notebook_error', 'detail': str(error)})
        return False, report
    if json.dumps(notebook, indent=1, ensure_ascii=False) + '\n' != text:
        report['unhandled'].append({
            'line': 0, 'kind': 'notebook_format',
            'detail': 'the JSON does not round-trip (indent 1): not '
                      'rewritten'})
        return False, report
    new_code, report = rewrite_source(
        code, table, module_name=None, path=path, index=index,
        parts=parts, external_origins=external_origins,
        drop_unused=drop_unused, external_uses=external_uses)
    if new_code == code:
        return False, report
    sources = _split_notebook_code(new_code)
    if sorted(sources) != cells:
        report['unhandled'].append({
            'line': 0, 'kind': 'notebook_cells',
            'detail': 'the cell markers did not survive the rewrite'})
        return False, report
    for i in cells:
        cell = notebook['cells'][i]
        if sources[i] != _cell_source(cell):
            _set_cell_source(cell, sources[i])
    new_text = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    if new_text == text:
        return False, report
    if write:
        with open(path, 'wb') as file:
            file.write(new_text.replace('\n', eol).encode('utf-8'))
    return True, report


# ---------------------------------------------------------------------------
# Use index (who reads module.name from another file)
# ---------------------------------------------------------------------------

_PATCH_STRING_RE = re.compile(r'\bcellacdc(?:\.[A-Za-z_]\w*)+')


def collect_uses(text, module_name=None, is_package=False):
    """(module, name) pairs that a source reads as attributes of modules
    (over-approximated, scope-blind): from-imports, attribute chains
    through module imports, getattr/setattr/patch calls with a string
    name and dotted ``cellacdc.`` strings."""
    tree = _parse(text)
    out = set()
    modules = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            source = _resolve_relative(
                module_name, is_package, node.level, node.module)
            if source is None:
                continue
            for alias in node.names:
                out.add((source, alias.name))
                if alias.name != '*':
                    modules[alias.asname or alias.name] = (
                        source + '.' + alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    first = alias.name.split('.')[0]
                    modules[first] = first
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = []
            current = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name) and current.id in modules:
                parts = modules[current.id].split('.') + chain[::-1]
                for i in range(1, len(parts)):
                    out.add(('.'.join(parts[:i]), parts[i]))
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            first, second = node.args[0], node.args[1]
            if isinstance(first, ast.Name) and first.id in modules and (
                    isinstance(second, ast.Constant)
                    and isinstance(second.value, str)):
                out.add((modules[first.id], second.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in _PATCH_STRING_RE.finditer(node.value):
                parts = match.group(0).split('.')
                for i in range(1, len(parts)):
                    out.add(('.'.join(parts[:i]), parts[i]))
    return out


def _module_level_imports(tree):
    """Import statements at module level (also inside if/try blocks)."""
    stack = list(tree.body)
    while stack:
        stmt = stack.pop(0)
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            yield stmt
        elif not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
            for field in ('body', 'orelse', 'finalbody'):
                stack.extend(getattr(stmt, field, None) or ())
            for handler in getattr(stmt, 'handlers', None) or ():
                stack.extend(handler.body)


def find_reexports(files, table, uses):
    """{module: {local: old module}}: module-level imports of old modules
    (not identity ones) in files that are no old module themselves, when
    another file reads ``module.local`` (``uses`` of build_use_index).
    NameTable.add_reexports makes the rewriter follow them."""
    out = {}
    for path in files:
        if path.endswith('.ipynb'):
            continue
        module_name, is_package = module_name_for_path(path)
        if not module_name or module_name in table.old:
            continue
        try:
            with open(path, 'rb') as file:
                tree = _parse(_decode(file.read())[0], path)
        except (SyntaxError, ValueError, UnicodeDecodeError, LookupError):
            continue
        full = os.path.abspath(path)
        for stmt in _module_level_imports(tree):
            for alias in stmt.names:
                if isinstance(stmt, ast.Import):
                    if not alias.asname:
                        continue
                    target = alias.name
                else:
                    source = _resolve_relative(
                        module_name, is_package, stmt.level, stmt.module)
                    if source is None or alias.name == '*':
                        continue
                    target = source + '.' + alias.name
                local = alias.asname or alias.name
                if target not in table.old or target in table.identity:
                    continue
                users = uses.get((module_name, local), set()) - {full}
                if users:
                    out.setdefault(module_name, {})[local] = target
    return out


def build_use_index(files):
    """{(module, name): {absolute paths}} over Python files and the code
    cells of notebooks (see collect_uses)."""
    index = {}
    for path in files:
        try:
            with open(path, 'rb') as file:
                data = file.read()
            if path.endswith('.ipynb'):
                _, text, _ = notebook_code(
                    data.decode('utf-8').replace('\r\n', '\n'))
                module_name, is_package = None, False
            else:
                text = _decode(data)[0]
                module_name, is_package = module_name_for_path(path)
            uses = collect_uses(text, module_name, is_package)
        except (SyntaxError, ValueError, UnicodeDecodeError, LookupError):
            continue
        full = os.path.abspath(path)
        for key in uses:
            index.setdefault(key, set()).add(full)
    return index


def _matches(path, patterns):
    candidates = {path.replace('\\', '/')}
    try:
        candidates.add(os.path.relpath(path).replace('\\', '/'))
    except ValueError:
        pass
    for candidate in candidates:
        for pattern in patterns:
            if fnmatch.fnmatch(candidate, pattern) or fnmatch.fnmatch(
                    candidate, '*/' + pattern):
                return True
    return False


def iter_python_files(paths, excludes=(), notebooks=False):
    """Python files of ``paths`` (and ``*.ipynb`` with ``notebooks``);
    walks folders (dotfiles included)."""
    seen = set()
    suffixes = ('.py', '.ipynb') if notebooks else ('.py',)
    for path in paths:
        if os.path.isdir(path):
            for directory, dirnames, filenames in os.walk(path):
                dirnames[:] = sorted(
                    d for d in dirnames
                    if d != '__pycache__' and not d.startswith('.'))
                for filename in sorted(filenames):
                    if not filename.endswith(suffixes):
                        continue
                    if filename.endswith('.ipynb') and (
                            '.ipynb_checkpoints' in directory):
                        continue
                    full = os.path.join(directory, filename)
                    if _matches(full, excludes):
                        continue
                    if full not in seen:
                        seen.add(full)
                        yield full
        elif path.endswith('.py') or os.path.isfile(path):
            if _matches(path, excludes):
                continue
            if path not in seen:
                seen.add(path)
                yield path


def _summary(files):
    summary = {'files_changed': 0, 'files_with_unhandled': 0}
    keys = REPORT_KEYS
    for key in keys:
        summary[key] = 0
    for entry in files.values():
        if entry.get('changed'):
            summary['files_changed'] += 1
        if entry.get('unhandled'):
            summary['files_with_unhandled'] += 1
        for key in keys:
            summary[key] += len(entry.get(key, ()))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Rewrite references to moved Cell-ACDC modules.')
    parser.add_argument('paths', nargs='+',
                        help='Python files or folders (walked recursively)')
    parser.add_argument('--table', required=True,
                        help='name table JSON (mapping.py name-table)')
    parser.add_argument('--report', help='write the JSON report here')
    parser.add_argument('--check', action='store_true',
                        help='do not write; exit 1 when files would change')
    parser.add_argument('--src-root',
                        help='folder containing the cellacdc package in its '
                             'new layout (lookup modules of patches of '
                             're-exported names); default: the current '
                             'folder when it contains cellacdc/__init__.py')
    parser.add_argument('--no-src-root', action='store_true',
                        help='do not read the new layout (patches of '
                             're-exported names are then only reported)')
    parser.add_argument('--external-origins', action='store_true',
                        help='with --src-root: import re-exported '
                             'third-party names from their origin (qtpy) '
                             'instead of the cellacdc module binding them')
    parser.add_argument('--exclude', action='append', default=[],
                        help='glob of files to skip (repeatable)')
    parser.add_argument('--no-default-excludes', action='store_true',
                        help='do not skip ' + ', '.join(DEFAULT_EXCLUDES))
    parser.add_argument('--parts', help='comma-separated part packages')
    parser.add_argument('--notebooks', action='store_true',
                        help='also rewrite the code cells of the *.ipynb '
                             'files in the walked folders')
    parser.add_argument('--drop-unused-imports', action='store_true',
                        help='remove imports of old modules and of names '
                             'imported from old modules that have no uses '
                             '(kept: noqa, __all__, string annotations, '
                             'module attributes read by another of the '
                             'given files)')
    args = parser.parse_args(argv)

    src_root = args.src_root
    if src_root is None and not args.no_src_root and os.path.isfile(
            os.path.join('cellacdc', '__init__.py')):
        src_root = '.'
    index = None if args.no_src_root or not src_root else SourceIndex(
        src_root)
    table = load_table(args.table, index=index)
    excludes = list(args.exclude)
    if not args.no_default_excludes:
        excludes.extend(DEFAULT_EXCLUDES)
    parts = (
        tuple(p.strip() for p in args.parts.split(',') if p.strip())
        if args.parts else DEFAULT_PARTS)
    files = {}
    paths = list(iter_python_files(args.paths, excludes, args.notebooks))
    uses = None
    reexports = {}
    if args.drop_unused_imports or index is not None:
        uses = build_use_index(paths)
        reexports = find_reexports(paths, table, uses)
        table.add_reexports(reexports)
    for path in paths:
        changed, report = rewrite_file(
            path, table, write=not args.check, index=index, parts=parts,
            external_origins=args.external_origins,
            drop_unused=args.drop_unused_imports,
            external_uses=uses if args.drop_unused_imports else None)
        if changed and index is not None and path.endswith('.py'):
            index.invalidate(module_name_for_path(path)[0])
        entry = {key: value for key, value in report.items() if value}
        if changed or entry:
            entry['changed'] = changed
            files[path.replace('\\', '/')] = entry
    result = {
        'files': files, 'summary': _summary(files),
        'src_root': os.path.abspath(src_root) if index else None,
        'reexports': reexports,
    }
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.write('\n')
    summary = result['summary']
    print(
        '%s: %d file(s) %s, %d reference(s), %d patch(es), %d string(s), '
        '%d unhandled, %d mention(s), %d long line(s), %d dropped '
        'import(s)' % (
            'check' if args.check else 'rewrite',
            summary['files_changed'],
            'would change' if args.check else 'changed',
            summary['references'], summary['patches'], summary['strings'],
            summary['unhandled'], summary['mentions'],
            summary['long_lines'], summary['dropped']))
    if args.check and summary['files_changed']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
