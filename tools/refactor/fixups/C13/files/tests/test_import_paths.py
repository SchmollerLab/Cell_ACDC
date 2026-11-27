"""Every import inside ``cellacdc`` names something that exists.

A static check (nothing is imported), so it also covers imports inside
functions that the tests never run, and modules that need optional
dependencies:

- every ``from <module> import <name>`` of a ``cellacdc`` module, at any
  nesting depth, names an existing module (``.py`` file, package, compiled
  extension or Cython source, or a module generated at build time such as
  ``_version.py``), and each name is a submodule or a module-level binding
  of it, unless the module has a module ``__getattr__`` (compat modules);
- no relative import leaves the package of its program part
  (``cellacdc/<part>/``): code outside the part is imported absolutely;
- every ``<alias>.<name>`` of a name bound to a ``cellacdc`` module
  (``from cellacdc.shared.widgets import message_box as
  widgets_message_box`` ... ``widgets_message_box.myMessageBox``) exists in
  that module.

Skipped: imports in the body of a ``try`` whose handlers catch
``ImportError``/``ModuleNotFoundError`` (optional dependencies, fallbacks
of older layouts), ``cellacdc/_deprecated/`` and the generated Qt
resources.
"""
import ast
from collections import ChainMap

import pytest

from tests import source_scan
from tests.source_scan import PACKAGE

pytestmark = pytest.mark.infra


@pytest.fixture(scope='module', autouse=True)
def _release_parsed_sources():
    """The parsed package takes hundreds of MB: release it after the
    module (the session keeps running other tests)."""
    yield
    source_scan.clear_caches()


SKIPPED_FOLDERS = ('cellacdc/_deprecated/',)
IMPORT_ERRORS = {'ImportError', 'ModuleNotFoundError', 'Exception',
                 'BaseException'}
# Packages whose modules import each other relatively and everything else
# absolutely (the packages of the package layout; utils/ predates it and
# imports the top-level modules relatively: `from .. import load`)
RELATIVE_IMPORTS_INSIDE = set(source_scan.NEW_PACKAGES)
# Broken imports that already existed before the package layout refactor
# (the refactor preserves behaviour; fix them separately): (file, text of
# the problem)
KNOWN_PROBLEMS = {
    # a module of the screen recorder that imports beyond the package
    ('cellacdc/record.py', 'relative import beyond cellacdc'),
    # `path.name` of the module cellacdc.path in an error message
    # (apps.py at 4c4e498b)
    ('cellacdc/data_structure/dialogs/setup.py', 'path.name'),
}


def checked_files():
    return [
        rel for rel in source_scan.package_files()
        if not rel.startswith(SKIPPED_FOLDERS)
    ]


def is_known(problem):
    return any(
        problem.startswith(rel + ':') and text in problem
        for rel, text in KNOWN_PROBLEMS
    )


def _catches_import_error(handler):
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [
        handler.type]
    return any(
        (isinstance(t, ast.Name) and t.id in IMPORT_ERRORS)
        or (isinstance(t, ast.Attribute) and t.attr in IMPORT_ERRORS)
        for t in types
    )


def guarded_nodes(tree):
    """Nodes inside the body of a ``try`` that catches ImportError."""
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
                _catches_import_error(h) for h in node.handlers):
            for stmt in node.body:
                guarded.update(id(sub) for sub in ast.walk(stmt))
    return guarded


def check_import_froms(rel, tree):
    problems = []
    guarded = guarded_nodes(tree)
    importer = source_scan.module_name(rel)
    part = source_scan.part_of(importer)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or id(node) in guarded:
            continue
        module = source_scan.import_from_module(node, rel)
        where = f'{rel}:{node.lineno}'
        if module is None:
            problems.append(f'{where}: relative import beyond {PACKAGE}')
            continue
        if module.split('.')[0] != PACKAGE:
            continue
        leaves = source_scan.part_of(module) != part
        if node.level and part in RELATIVE_IMPORTS_INSIDE and leaves:
            problems.append(
                f'{where}: relative import from {module} leaves the '
                f'package {PACKAGE}/{part}/ (import it absolutely)'
            )
        if not source_scan.is_module(module):
            problems.append(f'{where}: no module {module}')
            continue
        for alias in node.names:
            if alias.name == '*':
                continue
            if not source_scan.module_has_name(module, alias.name):
                problems.append(f'{where}: {module} has no {alias.name}')
    return problems


class AliasUses:
    """``<alias>.<name>`` uses of names bound to cellacdc modules."""

    def __init__(self, rel, tree):
        self.rel = rel
        self.problems = []
        self.guarded = guarded_nodes(tree)
        self.scan(tree.body, ChainMap(), None)

    def bindings(self, body, args):
        """{name: modules} of a scope: the cellacdc modules an import binds
        to the name (several in different branches), None when the name is
        also bound to anything else."""
        modules = {}
        other = set()
        if args is not None:
            for arg in (args.posonlyargs + args.args + args.kwonlyargs
                        + [args.vararg, args.kwarg]):
                if arg is not None:
                    other.add(arg.arg)
        for node in source_scan.iter_module_level(body):
            guarded = id(node) in self.guarded
            if isinstance(node, ast.Import):
                for alias in node.names:
                    is_module = alias.name.split('.')[0] == PACKAGE
                    if alias.asname and is_module and not guarded:
                        modules.setdefault(alias.asname, set()).add(
                            alias.name)
                    else:
                        other.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                module = source_scan.import_from_module(node, self.rel)
                for alias in node.names:
                    name = alias.asname or alias.name
                    full = f'{module}.{alias.name}'
                    has_source = bool(source_scan.module_index().get(full))
                    if (module and module.split('.')[0] == PACKAGE
                            and has_source and not guarded):
                        modules.setdefault(name, set()).add(full)
                    else:
                        other.add(name)
            else:
                other.update(_bound_targets(node))
        local = {name: None for name in other}
        local.update({
            name: frozenset(found) for name, found in modules.items()
            if name not in other
        })
        return local

    def scan(self, body, env, args):
        env = env.new_child(self.bindings(body, args))
        for node in body:
            self.visit(node, env)

    def visit(self, node, env):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.decorator_list + node.args.defaults:
                self.visit(child, env)
            self.scan(node.body, env, node.args)
            return
        if isinstance(node, ast.ClassDef):
            for child in node.decorator_list + node.bases:
                self.visit(child, env)
            self.scan(node.body, env, None)
            return
        if isinstance(node, ast.Lambda):
            self.visit(node.body, env.new_child(
                self.bindings([], node.args)))
            return
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and env.get(node.value.id)
        ):
            modules = env[node.value.id]
            # Module attributes such as __file__ always exist
            is_dunder = node.attr.startswith('__') and node.attr.endswith(
                '__')
            if not is_dunder and not any(
                source_scan.module_has_name(module, node.attr)
                for module in modules
            ):
                self.problems.append(
                    f'{self.rel}:{node.lineno}: {node.value.id}.{node.attr}'
                    f' ({" or ".join(sorted(modules))} has no {node.attr})'
                )
            return
        for child in ast.iter_child_nodes(node):
            self.visit(child, env)


def _bound_targets(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                         ast.ClassDef)):
        yield node.name
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            yield from source_scan._target_names(target)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For,
                           ast.AsyncFor)):
        yield from source_scan._target_names(node.target)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                yield from source_scan._target_names(item.optional_vars)


@pytest.mark.slow  # parses every module of cellacdc (~3 s)
def test_every_import_from_resolves():
    problems = []
    for rel in checked_files():
        tree = source_scan.parse(rel)
        if tree is None:
            problems.append(f'{rel}: does not parse')
            continue
        problems.extend(check_import_froms(rel, tree))
    problems = [problem for problem in problems if not is_known(problem)]
    assert not problems, '\n'.join(problems)


@pytest.mark.slow  # parses every module of cellacdc (~3 s)
def test_module_alias_attributes_exist():
    problems = []
    for rel in checked_files():
        tree = source_scan.parse(rel)
        if tree is not None:
            problems.extend(AliasUses(rel, tree).problems)
    problems = [problem for problem in problems if not is_known(problem)]
    assert not problems, '\n'.join(problems)


def _check_source(rel, source):
    tree = source_scan.parse_text(source)
    return check_import_froms(rel, tree) + AliasUses(rel, tree).problems


@pytest.mark.parametrize('rel, source, expected', [
    ('cellacdc/gui/mixins/navigation.py',
     'from . import dialogs\nfrom .. import helpers\n', []),
    ('cellacdc/gui/mixins/navigation.py',
     'from ... import load\n',
     ['relative import from cellacdc leaves the package cellacdc/gui/']),
    ('cellacdc/load.py', 'from . import core\n', []),
    ('cellacdc/load.py', 'from .gui import mixins\n', []),
    ('cellacdc/load.py', 'from . import not_a_module_xyz\n',
     ['cellacdc has no not_a_module_xyz']),
    ('cellacdc/load.py', 'from .not_a_package_xyz import x\n',
     ['no module cellacdc.not_a_package_xyz']),
    ('cellacdc/load.py',
     'try:\n    from .not_a_package_xyz import x\nexcept ImportError:\n'
     '    x = None\n', []),
    ('cellacdc/load.py',
     'from cellacdc.shared.widgets import message_box as mb\n'
     'mb.myMessageBox\nmb.not_a_name_xyz\n',
     ['mb.not_a_name_xyz']),
    ('cellacdc/load.py',
     'from cellacdc.shared.widgets import message_box as mb\n'
     'def f(mb):\n    return mb.not_a_name_xyz\n', []),
    ('cellacdc/load.py', 'from cellacdc.precompiled import '
     'precompiled_functions\nfrom cellacdc._version import version\n', []),
    # Bound to one of two modules: the name must exist in one of them
    ('cellacdc/load.py',
     'if x:\n    from cellacdc import load as m\nelse:\n'
     '    from cellacdc import core as m\nm.load_image_file\nm.__file__\n',
     []),
])
def test_checks_on_examples(rel, source, expected):
    problems = _check_source(rel, source)
    assert len(problems) == len(expected), problems
    for problem, text in zip(sorted(problems), sorted(expected)):
        assert text in problem
