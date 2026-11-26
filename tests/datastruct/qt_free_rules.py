"""AST rule engine for the static Qt-free (QF) rules.

`check_file(path, qf=..., stdlib_only=..., new_file=..., parent_never_opens_raw=...)`
returns the list of `Violation`s of one module; `check_gui_lazy_imports(path)`
checks apps.py/widgets.py. `rules_for(relpath)` gives the flags that
apply to a file (relpath inside the `cellacdc` package, POSIX).

Rule ids (Violation.rule):
    qt-import           module-level import of a Qt module (QF)
    forbidden-import    module-level import of a forbidden cellacdc module (QF)
    not-allowlisted     module-level import of another cellacdc module (QF)
    import-style        module-level `from cellacdc.<myutils|load|io|core|
                        prompts> import Name` (absolute or relative) (QF)
    qt-name             identifier or annotation string matching QT_NAME_RE (QF)
    stdlib-only         module-level import outside the stdlib (stdlib_only)
    future-annotations  missing `from __future__ import annotations` (new_file)
    no-print            reference to `print` (new_file)
    forbidden-name      reference to _process / bioio_sample_data_folderpath /
                        check_raise_exception (new_file)
    parent-opens-raw    import (any level) of a raw-file opener
                        (parent_never_opens_raw)
    gui-eager-import    module-level import of the shared core in a GUI module
                        (check_gui_lazy_imports)
    syntax-error        the file does not parse

"Module level" includes module-level if/try/with/for/while/match bodies and
class bodies, and excludes function bodies and `if TYPE_CHECKING:` blocks
(their `else` branch is module level).
"""
from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
PACKAGE_DIR = os.path.join(REPO_ROOT, 'cellacdc')
PACKAGE = 'cellacdc'

QT_MODULES = frozenset({
    'qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'pyqtgraph', 'vispy',
})
ALLOWED_MODULES = frozenset({
    'myutils', 'load', 'io', 'acdc_regex', 'config', 'acdc_bioio_bioformats',
    'acdc_datastruct', '_entry_points',
})
"""cellacdc submodules (and their subpackages) importable at module level."""
FORBIDDEN_MODULES = frozenset({
    'apps', 'widgets', 'workers', 'gui', '_main', 'dataStruct',
    'dataReStruct', 'qutils', 'qrc_resources', 'volume_renderer',
    'html_utils', 'colors', 'prompts', 'plot', 'gui_utils',
})
FROM_IMPORT_FORBIDDEN_MODULES = frozenset({
    'cellacdc.myutils', 'cellacdc.load', 'cellacdc.io', 'cellacdc.core',
    'cellacdc.prompts',
})
"""`from <these> import Name` is forbidden at module level."""
GRANDFATHERED = frozenset({
    ('acdc_bioio_bioformats/reader.py',
     'from ..myutils import safe_get_or_call'),
    ('acdc_bioio_bioformats/_utils.py',
     'from cellacdc.config import ConfigParser'),
})
"""(relpath, statement text) pairs exempt from the import-style rule."""
QT_NAME_RE = re.compile(r'\bQ[A-Z][a-z]\w*\b')
NEW_FILE_FORBIDDEN_NAMES = frozenset({
    '_process', 'bioio_sample_data_folderpath', 'check_raise_exception',
})
RAW_OPENER_MODULES = (
    'bioio', 'bioio_base', 'cellacdc.acdc_bioio_bioformats.reader',
    'cellacdc.acdc_bioio_bioformats.raw_metadata',
    'cellacdc.acdc_bioio_bioformats.writer',
)
RAW_OPENER_NAMES = frozenset({'ImageReader'})
GUI_EAGER_MODULES = (
    'cellacdc.acdc_datastruct',
    'cellacdc.acdc_bioio_bioformats.raw_metadata',
    'cellacdc.acdc_bioio_bioformats.writer',
)

SO_FILES = (
    '_entry_points.py', 'acdc_datastruct/runner.py',
    'acdc_datastruct/_worker.py', 'data_cli.py',
)
NEW_QF_FILES = (
    '_entry_points.py', 'data_cli.py', 'acdc_bioio_bioformats/raw_metadata.py',
    'acdc_bioio_bioformats/writer.py',
)
"""Plus every acdc_datastruct/*.py."""
EXISTING_QF_FILES = (
    'acdc_bioio_bioformats/__init__.py', 'acdc_bioio_bioformats/reader.py',
    'acdc_bioio_bioformats/_utils.py', 'acdc_bioio_bioformats/install.py',
)
PARENT_NEVER_OPENS_RAW_FILES = (
    'acdc_datastruct/convert.py', 'acdc_datastruct/backends.py',
    'acdc_datastruct/planning.py', 'acdc_datastruct/sources.py',
    'acdc_datastruct/metadata.py', 'acdc_datastruct/writers.py',
    'data_cli.py',
)
GUI_LAZY_IMPORT_FILES = ('apps.py', 'widgets.py')


@dataclass(frozen=True)
class Violation:
    file: str       # path as given to check_file
    line: int
    rule: str
    text: str       # offending statement / identifier and explanation

    def __str__(self) -> str:
        return f'{self.file}:{self.line}: [{self.rule}] {self.text}'


@dataclass(frozen=True)
class ImportRef:
    """One import statement with resolved absolute module names.

    `modules` holds the effective module of each name: base.name when that
    module exists under the package dir, else base (the name is an
    attribute)."""
    line: int
    text: str                   # statement source, whitespace-normalized
    is_from: bool
    level: int                  # ImportFrom level (0 = absolute)
    base: str | None            # ImportFrom: resolved module; Import: None
    names: tuple[str, ...]      # imported names (Import: dotted modules)
    targets: tuple[str, ...]    # absolute dotted names (base.name / module)
    modules: tuple[str, ...]    # effective modules (see docstring)
    module_level: bool
    type_checking: bool


def rules_for(relpath: str) -> dict[str, bool] | None:
    """The check_file flags that apply to `relpath` (POSIX, relative to the
    cellacdc package dir), or None when the file is not a QF module."""
    relpath = relpath.replace('\\', '/')
    in_datastruct = (
        relpath.startswith('acdc_datastruct/') and relpath.count('/') == 1
        and relpath.endswith('.py')
    )
    new_file = in_datastruct or relpath in NEW_QF_FILES
    if not new_file and relpath not in EXISTING_QF_FILES:
        return None
    return {
        'qf': True,
        'stdlib_only': relpath in SO_FILES,
        'new_file': new_file,
        'parent_never_opens_raw': relpath in PARENT_NEVER_OPENS_RAW_FILES,
    }


def format_violations(violations) -> str:
    return '\n'.join(str(v) for v in violations)


# ------------------------------------------------------------------ helpers
def _statement_text(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node) or ''
    return ' '.join(segment.split())


def _derive_relpath(path: str) -> str:
    """Path relative to the nearest ancestor folder named 'cellacdc' with an
    __init__.py (POSIX); the basename when there is none."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    while True:
        if (
            os.path.basename(parent) == PACKAGE
            and os.path.isfile(os.path.join(parent, '__init__.py'))
        ):
            return os.path.relpath(path, parent).replace(os.sep, '/')
        new_parent = os.path.dirname(parent)
        if new_parent == parent:
            return os.path.basename(path)
        parent = new_parent


def _module_name(relpath: str) -> tuple[str, str]:
    """(module, package) of a file given its relpath inside cellacdc."""
    parts = relpath[:-3].split('/') if relpath.endswith('.py') else \
        relpath.split('/')
    if parts[-1] == '__init__':
        package_parts = [PACKAGE] + parts[:-1]
        return '.'.join(package_parts), '.'.join(package_parts)
    module = '.'.join([PACKAGE] + parts)
    return module, '.'.join([PACKAGE] + parts[:-1])


def _resolve_from(node: ast.ImportFrom, package: str) -> str | None:
    if node.level == 0:
        return node.module
    parts = package.split('.')
    if node.level - 1 >= len(parts):
        return None
    base = '.'.join(parts[:len(parts) - (node.level - 1)])
    return f'{base}.{node.module}' if node.module else base


def _is_package_module(dotted: str, package_dir: str) -> bool:
    parts = dotted.split('.')
    if parts[0] != PACKAGE:
        return False
    if len(parts) == 1:
        return True
    path = os.path.join(package_dir, *parts[1:])
    return os.path.isfile(path + '.py') or os.path.isdir(path)


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == 'TYPE_CHECKING'
    return isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'


def _walk_import_nodes(body, module_level: bool, type_checking: bool
                       ) -> Iterator[tuple[ast.stmt, bool, bool]]:
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, module_level, type_checking
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _walk_import_nodes(node.body, False, type_checking)
        elif isinstance(node, ast.If):
            yield from _walk_import_nodes(
                node.body, module_level,
                type_checking or _is_type_checking(node.test)
            )
            yield from _walk_import_nodes(
                node.orelse, module_level, type_checking
            )
        else:
            for name in ('body', 'orelse', 'finalbody'):
                value = getattr(node, name, None)
                if isinstance(value, list):
                    yield from _walk_import_nodes(
                        value, module_level, type_checking
                    )
            for handler in getattr(node, 'handlers', None) or ():
                yield from _walk_import_nodes(
                    handler.body, module_level, type_checking
                )
            for case in getattr(node, 'cases', None) or ():
                yield from _walk_import_nodes(
                    case.body, module_level, type_checking
                )


def collect_imports(tree: ast.Module, source: str, *, relpath: str,
                    package_dir: str = PACKAGE_DIR) -> list[ImportRef]:
    """Every import statement of `tree` (module level or not), resolved as if
    the file were cellacdc/<relpath>."""
    _, package = _module_name(relpath)
    refs = []
    for node, module_level, type_checking in _walk_import_nodes(
            tree.body, True, False):
        text = _statement_text(source, node)
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
            refs.append(ImportRef(
                node.lineno, text, False, 0, None, names, names, names,
                module_level, type_checking
            ))
            continue
        base = _resolve_from(node, package)
        names = tuple(alias.name for alias in node.names)
        if base is None:
            targets = modules = ()
        else:
            targets = tuple(
                base if name == '*' else f'{base}.{name}' for name in names
            )
            modules = tuple(
                target if _is_package_module(target, package_dir) else base
                for target in targets
            )
        refs.append(ImportRef(
            node.lineno, text, True, node.level, base, names, targets,
            modules, module_level, type_checking
        ))
    return refs


def imports_of(path: str, *, relpath: str | None = None,
               package_dir: str = PACKAGE_DIR) -> list[ImportRef]:
    """`collect_imports` of a file (e.g. to check who imports whom)."""
    path = _locate(path)
    with open(path, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source, filename=path)
    return collect_imports(
        tree, source, relpath=relpath or _derive_relpath(path),
        package_dir=package_dir
    )


def _top(name: str | None) -> str:
    return (name or '').split('.')[0]


def _under(name: str, prefixes) -> bool:
    return any(name == p or name.startswith(p + '.') for p in prefixes)


def _identifiers(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        line = getattr(node, 'lineno', 0)
        if isinstance(node, ast.Name):
            yield line, node.id
        elif isinstance(node, ast.Attribute):
            yield line, node.attr
        elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield line, node.name
        elif isinstance(node, ast.arg):
            yield line, node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield line, node.arg
        elif isinstance(node, ast.alias):
            for part in node.name.split('.'):
                yield line, part
            if node.asname:
                yield line, node.asname
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                yield line, name
        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield line, node.name


def _annotation_strings(tree: ast.AST) -> Iterator[tuple[int, str]]:
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.arg) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    for annotation in annotations:
        for sub in ast.walk(annotation):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                yield sub.lineno, sub.value


def _locate(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(REPO_ROOT, path)
    return candidate if os.path.exists(candidate) else path


# -------------------------------------------------------------- the checks
def check_source(source: str, *, filename: str = '<snippet>',
                 relpath: str = 'acdc_datastruct/_snippet.py',
                 qf: bool = True, stdlib_only: bool = False,
                 new_file: bool = False,
                 parent_never_opens_raw: bool = False,
                 package_dir: str = PACKAGE_DIR) -> list[Violation]:
    """Check module `source` as if it were cellacdc/<relpath>."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as err:
        return [Violation(filename, err.lineno or 0, 'syntax-error', str(err))]
    out: list[Violation] = []

    def add(line: int, rule: str, text: str) -> None:
        violation = Violation(filename, line, rule, text)
        if violation not in out:
            out.append(violation)

    refs = collect_imports(
        tree, source, relpath=relpath, package_dir=package_dir
    )
    runtime_module_level = [
        ref for ref in refs if ref.module_level and not ref.type_checking
    ]
    if qf:
        for ref in runtime_module_level:
            tops = {_top(t) for t in ref.targets} | {_top(ref.base)}
            if tops & QT_MODULES:
                add(ref.line, 'qt-import', ref.text)
            for module in ref.modules:
                if _top(module) != PACKAGE or module == PACKAGE:
                    continue
                first = module.split('.')[1]
                if first in FORBIDDEN_MODULES:
                    add(ref.line, 'forbidden-import',
                        f'{ref.text} (cellacdc.{first} is forbidden)')
                elif first not in ALLOWED_MODULES:
                    add(ref.line, 'not-allowlisted',
                        f'{ref.text} (cellacdc.{first} is not allowlisted)')
            if (
                ref.is_from and ref.base in FROM_IMPORT_FORBIDDEN_MODULES
                and (relpath, ref.text) not in GRANDFATHERED
            ):
                add(ref.line, 'import-style',
                    f'{ref.text} (import the module object instead)')
        for line, name in _identifiers(tree):
            if QT_NAME_RE.search(name):
                add(line, 'qt-name', name)
        for line, text in _annotation_strings(tree):
            match = QT_NAME_RE.search(text)
            if match:
                add(line, 'qt-name', f'annotation {text!r}')
    if stdlib_only:
        stdlib = sys.stdlib_module_names
        for ref in runtime_module_level:
            if ref.level > 0 or any(
                    _top(t) not in stdlib for t in ref.targets):
                add(ref.line, 'stdlib-only', ref.text)
    if new_file:
        body = tree.body
        has_code = len(body) > 1 or (body and not (
            isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ))
        has_future = any(
            isinstance(node, ast.ImportFrom) and node.module == '__future__'
            and any(alias.name == 'annotations' for alias in node.names)
            for node in body
        )
        if has_code and not has_future:
            add(1, 'future-annotations',
                'missing `from __future__ import annotations`')
        for line, name in _identifiers(tree):
            if name == 'print':
                add(line, 'no-print', 'print (use callbacks or logging)')
            if name in NEW_FILE_FORBIDDEN_NAMES:
                add(line, 'forbidden-name', name)
        for ref in refs:
            for target in ref.targets:
                hits = set(target.split('.')) & NEW_FILE_FORBIDDEN_NAMES
                for hit in sorted(hits):
                    add(ref.line, 'forbidden-name', f'{ref.text} ({hit})')
    if parent_never_opens_raw:
        for ref in refs:
            if (
                any(_under(t, RAW_OPENER_MODULES) for t in ref.targets)
                or set(ref.names) & RAW_OPENER_NAMES
            ):
                add(ref.line, 'parent-opens-raw', ref.text)
    return sorted(out, key=lambda v: (v.line, v.rule, v.text))


def check_file(path: str, *, qf: bool = True, stdlib_only: bool = False,
               new_file: bool = False, parent_never_opens_raw: bool = False,
               relpath: str | None = None,
               package_dir: str = PACKAGE_DIR) -> list[Violation]:
    """Check one module file. `path` may be relative to the cwd or to the
    repo root. `relpath` (POSIX, inside the cellacdc package; default derived
    from `path`) drives relative-import resolution and GRANDFATHERED."""
    located = _locate(path)
    with open(located, 'r', encoding='utf-8') as file:
        source = file.read()
    return check_source(
        source, filename=path,
        relpath=relpath or _derive_relpath(located), qf=qf,
        stdlib_only=stdlib_only, new_file=new_file,
        parent_never_opens_raw=parent_never_opens_raw,
        package_dir=package_dir
    )


def check_gui_lazy_source(source: str, *, filename: str = '<snippet>',
                          relpath: str = 'apps.py',
                          package_dir: str = PACKAGE_DIR) -> list[Violation]:
    """`check_gui_lazy_imports` on source text."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as err:
        return [Violation(filename, err.lineno or 0, 'syntax-error', str(err))]
    out = []
    for ref in collect_imports(
            tree, source, relpath=relpath, package_dir=package_dir):
        if not ref.module_level or ref.type_checking:
            continue
        if any(_under(m, GUI_EAGER_MODULES) for m in ref.targets + ref.modules):
            out.append(Violation(
                filename, ref.line, 'gui-eager-import',
                f'{ref.text} (import the shared core inside the method)'
            ))
    return out


def check_gui_lazy_imports(path: str) -> list[Violation]:
    """apps.py / widgets.py: no module-level import of cellacdc.acdc_datastruct
    (or of the new acdc_bioio_bioformats.raw_metadata / writer)."""
    located = _locate(path)
    with open(located, 'r', encoding='utf-8') as file:
        source = file.read()
    return check_gui_lazy_source(
        source, filename=path, relpath=_derive_relpath(located)
    )
