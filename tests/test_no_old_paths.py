"""Cell-ACDC's own code, tests and notebooks use only the new import paths.

The package layout moved most modules (CODEMAP.md). The old import paths
keep working for external code (plugins, spotMAX, users' scripts) through
compat shims that emit a ``MovedWarning`` (``cellacdc/_compat.py``, tables
in the generated ``cellacdc/_compat_moves.py``), and pytest turns that
warning into an error (``filterwarnings`` in ``pyproject.toml``). This
static scan also finds old paths that the tests never execute:

- imports of a shim (``import cellacdc.apps``, ``from cellacdc import
  widgets``, ``from . import gui_navigation``) or of a name through a module
  it moved out of (``from cellacdc.gui import guiWin``,
  ``from cellacdc.myutils import listdir``);
- attribute access through such a module (``myutils.listdir(...)``,
  ``cellacdc.gui.guiWin``), also in ``setattr``/``getattr``/
  ``mock.patch.object`` calls naming the attribute as a string;
- strings naming an old path (patch targets ``'cellacdc.apps.X'``,
  ``importlib.import_module('cellacdc.segm')``), code in strings (scripts
  run in subprocesses or model environments; the ``else`` branch of
  ``if hasattr(<module>, '__path__'):`` is the old layout of an older
  Cell-ACDC installed there and is skipped) and string annotations naming
  an old module that the file does not import (``'widgets.myMessageBox'``).

Docstrings and comments are not scanned. Every file at an old path must
be a shim: ``__cellacdc_compat__ = True`` and no definitions.

Exempt: ``cellacdc/_compat*.py``, the shims themselves and
``tests/test_compat.py``, which checks the old paths on purpose.
"""
import ast
import json
import re
import textwrap
from collections import ChainMap

import pytest

from tests import source_scan
from tests.source_scan import REPO_ROOT, module_name

pytestmark = pytest.mark.infra


@pytest.fixture(scope='module', autouse=True)
def _release_parsed_sources():
    """The parsed package takes hundreds of MB: release it after the
    module (the session keeps running other tests)."""
    yield
    source_scan.clear_caches()


EXEMPT_FILES = {
    'cellacdc/_compat.py',
    'cellacdc/_compat_moves.py',
    'tests/test_compat.py',
    # The examples of the scanner's own test
    'tests/test_no_old_paths.py',
}
DOTTED_PATH = re.compile(r'(?<![\w.])cellacdc(?:\.[A-Za-z_]\w*)+')
# Functions whose first two arguments are a module and an attribute name
ATTRIBUTE_CALLS = {'setattr', 'delattr', 'getattr', 'hasattr', 'object'}
# Calls of cellacdc/_compat.py that name old modules on purpose
COMPAT_CALLS = {'alias_module', 'moved_module', 'forward_module',
                'package_compat'}


class OldPaths:
    """The old import paths of the compat tables."""

    def __init__(self):
        moves = source_scan.compat_moves()
        self.shims = set(source_scan.shim_modules()) | set(
            moves['MODULE_ALIASES']
        )
        name_moves = moves['NAME_MOVES']
        self.moved = {}
        for module, names in name_moves.items():
            self.moved.setdefault(module, set()).update(names)
        for module, forwarded in moves['FORWARDERS'].items():
            self.moved.setdefault(module, set()).update(
                name_moves.get(forwarded, ())
            )
        for module, names in self.moved.items():
            # Names the module (still) binds itself are not old paths:
            # submodules of the packages that replace an old module
            bound, _ = source_scan.module_bindings(module)
            names.difference_update(bound)
        self.basenames = {name.rpartition('.')[2] for name in self.shims}

    def check(self, module, attr):
        """The old path ``<module>.<attr>`` if it is one, else None."""
        full = f'{module}.{attr}'
        if full in self.shims:
            return full
        if attr in self.moved.get(module, ()):
            return full
        return None

    def check_dotted(self, dotted):
        """The old path that `dotted` goes through, else None."""
        parts = dotted.split('.')
        if dotted in self.shims:
            return dotted
        current = parts[0]
        for part in parts[1:]:
            found = self.check(current, part)
            if found:
                return found
            nxt = f'{current}.{part}'
            if not source_scan.is_module(nxt):
                return None
            current = nxt
        return None


def _names_of_target(target):
    yield from source_scan._target_names(target)


def _scope_statements(body):
    """Statements of a scope, without nested function/class bodies."""
    return source_scan.iter_module_level(body)


def _is_layout_check(test):
    """``hasattr(<module>, '__path__')`` (new layout: myutils package)."""
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == 'hasattr'
        and len(test.args) == 2
        and isinstance(test.args[1], ast.Constant)
        and test.args[1].value == '__path__'
    )


class Scanner:
    """Scope-aware scan of one source file (or of code in a string)."""

    def __init__(self, old, rel, findings, line_offset=0):
        self.old = old
        self.rel = rel
        self.findings = findings
        self.line_offset = line_offset

    # ------------------------------------------------------------ helpers
    def report(self, node, what):
        line = getattr(node, 'lineno', 0) + self.line_offset
        self.findings.add((self.rel, line, what))

    def resolve_import_from(self, node):
        return source_scan.import_from_module(node, self.rel)

    def bindings(self, body, args=None):
        """{name: module or None} bound in a scope (imports of cellacdc
        modules give the module, every other binding None)."""
        local = {}
        if args is not None:
            for arg in (args.posonlyargs + args.args + args.kwonlyargs
                        + [args.vararg, args.kwarg]):
                if arg is not None:
                    local[arg.arg] = None
        imported = {}
        for node in _scope_statements(body):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        imported[alias.asname] = alias.name
                    else:
                        first = alias.name.split('.')[0]
                        imported[first] = first
            elif isinstance(node, ast.ImportFrom):
                module = self.resolve_import_from(node)
                for alias in node.names:
                    name = alias.asname or alias.name
                    full = f'{module}.{alias.name}' if module else None
                    if full and source_scan.is_module(full):
                        imported[name] = full
                    else:
                        local.setdefault(name, None)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                local[node.name] = None
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in _names_of_target(target):
                        local[name] = None
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For,
                                   ast.AsyncFor)):
                for name in _names_of_target(node.target):
                    local[name] = None
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        for name in _names_of_target(item.optional_vars):
                            local[name] = None
        for name, module in imported.items():
            # Other imports shadow outer module names
            is_cellacdc = module.split('.')[0] == 'cellacdc'
            local[name] = module if is_cellacdc else None
        return local

    # ------------------------------------------------------------- scopes
    def scan_module(self, tree):
        env = ChainMap(self.bindings(tree.body))
        self.scan_body(tree.body, env)

    def scan_body(self, body, env):
        for node in body:
            self.visit(node, env)

    def visit(self, node, env):
        if source_scan.is_string_statement(node):
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for item in node.decorator_list:
                self.visit(item, env)
            self.visit_arguments(node.args, env)
            if node.returns is not None:
                self.visit_annotation(node.returns, env)
            inner = env.new_child(self.bindings(node.body, node.args))
            self.scan_body(node.body, inner)
            return
        if isinstance(node, ast.Lambda):
            self.visit_arguments(node.args, env)
            self.visit(node.body, env.new_child(
                self.bindings([], node.args)))
            return
        if isinstance(node, ast.ClassDef):
            for item in node.decorator_list + node.bases + [
                    k.value for k in node.keywords]:
                self.visit(item, env)
            self.scan_body(node.body, env.new_child(self.bindings(node.body)))
            return
        if isinstance(node, ast.If) and (
            _is_layout_check(node.test)
            or (isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and _is_layout_check(node.test.operand))
        ):
            new_layout = node.body
            if isinstance(node.test, ast.UnaryOp):
                new_layout = node.orelse
            self.visit(node.test, env)
            self.scan_body(new_layout, env)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                found = self.old.check_dotted(alias.name)
                if found:
                    self.report(node, f'import of {found}')
            return
        if isinstance(node, ast.ImportFrom):
            module = self.resolve_import_from(node)
            if not module or module.split('.')[0] != 'cellacdc':
                return
            found = self.old.check_dotted(module)
            if found:
                self.report(node, f'import from {found}')
                return
            for alias in node.names:
                found = self.old.check(module, alias.name)
                if found:
                    self.report(node, f'import of {found}')
            return
        if isinstance(node, ast.AnnAssign):
            self.visit_annotation(node.annotation, env)
            for child in (node.target, node.value):
                if child is not None:
                    self.visit(child, env)
            return
        if isinstance(node, ast.Attribute):
            self.visit_attribute(node, env)
            return
        if isinstance(node, ast.Call) and self.visit_call(node, env):
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.scan_string(node.value, node, env)
            return
        for child in ast.iter_child_nodes(node):
            self.visit(child, env)

    def visit_arguments(self, args, env):
        for default in args.defaults + [
                d for d in args.kw_defaults if d is not None]:
            self.visit(default, env)
        for arg in (args.posonlyargs + args.args + args.kwonlyargs
                    + [args.vararg, args.kwarg]):
            if arg is not None and arg.annotation is not None:
                self.visit_annotation(arg.annotation, env)

    def visit_annotation(self, annotation, env):
        if (isinstance(annotation, ast.Constant)
                and isinstance(annotation.value, str)):
            self.scan_string(annotation.value, annotation, env)
            expression = source_scan.parse_text(annotation.value.strip())
            if expression is None:
                return
            for sub in ast.walk(expression):
                if isinstance(sub, ast.Name) and sub.id not in env and \
                        sub.id in self.old.basenames:
                    self.report(annotation, 'string annotation naming the '
                                f'old module {sub.id!r}')
            for sub in ast.walk(expression):
                if isinstance(sub, ast.Attribute):
                    sub.lineno = annotation.lineno
                    self.visit_attribute(sub, env)
            return
        self.visit(annotation, env)

    # ------------------------------------------------------ references
    def module_of(self, node, env):
        """The module an expression refers to (``name`` or a chain of
        submodules), else None."""
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.module_of(node.value, env)
            if base is None:
                return None
            full = f'{base}.{node.attr}'
            if source_scan.is_module(full):
                return full
        return None

    def visit_attribute(self, node, env):
        chain = []
        current = node
        while isinstance(current, ast.Attribute):
            chain.append(current)
            current = current.value
        if not isinstance(current, ast.Name) or env.get(current.id) is None:
            self.visit(current, env)
            return
        module = env[current.id]
        if module in self.old.shims:
            # The import is reported
            return
        for attribute in reversed(chain):
            found = self.old.check(module, attribute.attr)
            if found:
                self.report(attribute, f'use of {found}')
                return
            nxt = f'{module}.{attribute.attr}'
            if not source_scan.is_module(nxt):
                return
            module = nxt

    def visit_call(self, node, env):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(
            func, 'id', None)
        if name in COMPAT_CALLS:
            # The compat machinery names old modules on purpose
            return True
        if name in ATTRIBUTE_CALLS and len(node.args) >= 2:
            module = self.module_of(node.args[0], env)
            attr = node.args[1]
            if (module and isinstance(attr, ast.Constant)
                    and isinstance(attr.value, str)):
                found = self.old.check(module, attr.value)
                if found:
                    self.report(node, f'{name}() of {found}')

    def scan_string(self, text, node, env):
        if 'cellacdc' not in text:
            return
        if 'import' in text:
            tree = source_scan.parse_text(textwrap.dedent(text))
            if tree is not None and any(
                isinstance(sub, (ast.Import, ast.ImportFrom))
                for sub in ast.walk(tree)
            ):
                # Code run in a subprocess or a model environment
                inner = Scanner(self.old, self.rel, self.findings,
                                self.line_offset + node.lineno - 1)
                inner.scan_module(tree)
                return
        for match in DOTTED_PATH.finditer(text):
            found = self.old.check_dotted(match.group())
            if found:
                self.report(node, f'string naming {found}')


@pytest.fixture(scope='module')
def old_paths():
    return OldPaths()


def scan_file(old, rel, text=None):
    findings = set()
    tree = source_scan.parse_text(text, rel) if text is not None else \
        source_scan.parse(rel)
    if tree is None:
        return {(rel, 0, 'does not parse')}
    Scanner(old, rel, findings).scan_module(tree)
    return findings


def files_to_scan(folder):
    root = REPO_ROOT / folder
    # rglob also yields hidden files (tests/segm/.test_sam.py)
    files = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in root.rglob('*.py')
        if '__pycache__' not in path.parts
        and not source_scan.GENERATED.search(path.name)
    )
    return [
        rel for rel in files
        if rel not in EXEMPT_FILES and not source_scan.is_shim_file(rel)
    ]


def format_findings(findings):
    return '\n'.join(
        f'{rel}:{line}: {what}' for rel, line, what in sorted(findings)
    )


@pytest.mark.slow  # parses every module of cellacdc (~3 s)
def test_package_uses_no_old_import_paths(old_paths):
    findings = set()
    for rel in files_to_scan('cellacdc'):
        findings |= scan_file(old_paths, rel)
    assert not findings, (
        'Use the new import paths (CODEMAP.md; look a name up with '
        "python -c \"from cellacdc._compat_moves import NAME_MOVES as M; "
        "print(M['cellacdc.widgets']['QDialogListbox'])\"):\n"
        + format_findings(findings)
    )


@pytest.mark.slow  # parses every test file (~1 s)
def test_tests_use_no_old_import_paths(old_paths):
    findings = set()
    for rel in files_to_scan('tests'):
        findings |= scan_file(old_paths, rel)
    assert not findings, (
        'Tests must use the new import paths and patch the module that '
        'defines a name (CODEMAP.md):\n' + format_findings(findings)
    )


def notebook_code_cells(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for k, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        if isinstance(source, list):
            source = ''.join(source)
        # IPython magics and shell escapes are no Python
        lines = [
            re.sub(r'^(\s*)[%!].*$', r'\1pass', line)
            for line in source.splitlines()
        ]
        yield k, '\n'.join(lines)


def test_notebooks_use_no_old_import_paths(old_paths):
    findings = set()
    for path in sorted((REPO_ROOT / 'notebooks').glob('*.ipynb')):
        rel = path.relative_to(REPO_ROOT).as_posix()
        # The cells share one namespace: scan them as one module
        code, cells = [], []
        for k, cell_code in notebook_code_cells(path):
            if source_scan.parse_text(cell_code) is None:
                continue
            lines = cell_code.split('\n')
            code.extend(lines)
            cells.extend((k, i) for i in range(1, len(lines) + 1))
        for _, line, what in scan_file(old_paths, rel, '\n'.join(code)):
            cell, line = cells[line - 1] if 0 < line <= len(cells) else (
                '?', line)
            findings.add((f'{rel} (cell {cell})', line, what))
    assert not findings, format_findings(findings)


def test_files_at_old_paths_are_shims():
    """Every module at an old import path is a compat shim: the marker
    ``__cellacdc_compat__ = True`` and no definitions."""
    moves = source_scan.compat_moves()
    index = source_scan.module_index()
    # Every old module was a top-level module of cellacdc (tables of new
    # modules list names moved by a later step of the refactor)
    old_modules = set(moves['MODULE_ALIASES']) | {
        module for module in moves['NAME_MOVES']
        if module.count('.') == 1 and (index.get(module) or '').endswith(
            '.py') and not source_scan.is_package_file(index[module])
    }
    wrong = {}
    for module in sorted(old_modules | source_scan.shim_modules()):
        rel = index.get(module)
        if rel is None:
            wrong[module] = 'missing (old paths must keep working)'
            continue
        tree = source_scan.parse(rel)
        definitions = [
            node.name for node in source_scan.iter_module_level(tree.body)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef))
        ]
        if not source_scan.is_shim(tree):
            if module in KEPT_MODULES_WITH_MOVED_NAMES:
                continue
            wrong[module] = f'{rel} is no shim (no {source_scan.SHIM_MARKER})'
        elif definitions:
            wrong[module] = f'{rel} defines {definitions}'
    # A file next to a same-name package is never imported
    for rel in source_scan.package_files():
        path = REPO_ROOT / rel
        if path.name != '__init__.py' and \
                path.with_suffix('').joinpath('__init__.py').is_file():
            wrong[module_name(rel)] = (
                f'{rel} is shadowed by the package {path.with_suffix("").name}/'
            )
    assert not wrong, wrong


# Modules that keep their own code and lost a few names (forwarded with a
# warning by a compat block at their end)
KEPT_MODULES_WITH_MOVED_NAMES = {'cellacdc.whitelist'}


@pytest.mark.parametrize('source, expected', [
    ('from cellacdc import apps\n', ['import of cellacdc.apps']),
    ('import cellacdc.gui_navigation as nav\n',
     ['import of cellacdc.gui_navigation']),
    ('from cellacdc.gui import guiWin\n', ['import of cellacdc.gui.guiWin']),
    ('from cellacdc.gui.main_window import window\n'
     'window.guiWin\n', []),
    ('from cellacdc import myutils\nmyutils.listdir(".")\n',
     ['use of cellacdc.myutils.listdir']),
    ('import cellacdc\ncellacdc.myutils.listdir(".")\n',
     ['use of cellacdc.myutils.listdir']),
    ('from cellacdc.myutils import filesystem\nfilesystem.listdir(".")\n',
     []),
    ('from cellacdc import myutils\n'
     'def f(myutils):\n    return myutils.listdir\n', []),
    ('def test(monkeypatch):\n'
     '    monkeypatch.setattr("cellacdc.apps.QDialogListbox", None)\n',
     ['string naming cellacdc.apps']),
    ('from cellacdc import myutils\n'
     'def test(monkeypatch):\n'
     '    monkeypatch.setattr(myutils, "listdir", None)\n',
     ['setattr() of cellacdc.myutils.listdir']),
    ('SCRIPT = """\nimport cellacdc.myutils as m\n'
     'if hasattr(m, "__path__"):\n    from cellacdc.myutils import installs\n'
     'else:\n    m.check_install_package\n"""\n', []),
    ('SCRIPT = "from cellacdc.workers import X; X()"\n',
     ['import from cellacdc.workers']),
    ('def f(x: "widgets.myMessageBox"): pass\n',
     ["string annotation naming the old module 'widgets'"]),
    ('"""Docstring naming cellacdc.apps.QDialogListbox."""\n', []),
])
def test_scanner_finds_old_paths(old_paths, source, expected):
    findings = scan_file(old_paths, 'tests/test_example.py', source)
    assert sorted(what for _, _, what in findings) == sorted(expected)
