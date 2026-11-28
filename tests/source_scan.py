"""Static view of the ``cellacdc`` source for the tests that check its
import structure (``test_no_old_paths.py``, ``test_import_paths.py``,
``test_import_layering.py``, ``test_package_init_files.py``).

Nothing here imports ``cellacdc``: files are read and parsed with ``ast``
(each file once per test session) and the compat tables of
``cellacdc/_compat_moves.py`` are read by executing that file alone.

Package layout (CODEMAP.md): one package per program part
(``PART_PACKAGES``), ``shared/`` for code that several parts use and
library modules (``load``, ``core``, ``myutils``, ...) at the top level.
Old import paths are compat shims, files whose module level sets
``__cellacdc_compat__ = True`` (``is_shim``).
"""
import ast
import functools
import re
import warnings
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'cellacdc'
PACKAGE_DIR = REPO_ROOT / PACKAGE
# One package per program part: its window plus its own dialogs, widgets
# and workers (the batch utilities count as one part)
PART_PACKAGES = (
    'gui', 'measurements_viewer', 'launcher', 'data_prep', 'data_structure',
    'segmentation_module', 'utils',
)
# Packages created by the package layout refactor
NEW_PACKAGES = PART_PACKAGES[:-1] + ('shared', 'myutils')
SHIM_MARKER = '__cellacdc_compat__'
SHIM_LINE = re.compile(
    r'^' + SHIM_MARKER + r'[ \t]*=[ \t]*True\b', re.MULTILINE
)
# Generated Qt resource modules (~340k lines, no imports of interest)
GENERATED = re.compile(r'qrc_resources\w*\.py$')
# Modules generated at build/install time (setuptools_scm), not in git
GENERATED_MODULES = {'cellacdc._version', 'cellacdc.bioformats._version'}
EXTENSION_SUFFIXES = ('.pyd', '.so', '.pyx')


def module_name(rel):
    """Dotted name of a ``.py`` file (``cellacdc/a/b.py`` -> ``cellacdc.a.b``)."""
    parts = list(PurePosixPath(rel).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def is_package_file(rel):
    return PurePosixPath(rel).name == '__init__.py'


@functools.lru_cache(maxsize=None)
def read(rel):
    return (REPO_ROOT / rel).read_text(encoding='utf-8', errors='replace')


@functools.lru_cache(maxsize=None)
def parse(rel):
    """AST of the file `rel` (relative posix path), None if it does not parse."""
    return parse_text(read(rel), rel)


def parse_text(text, filename='<string>'):
    try:
        with warnings.catch_warnings():
            # e.g. invalid escape sequences in the scanned modules
            warnings.simplefilter('ignore')
            return ast.parse(text, filename)
    except (SyntaxError, ValueError):
        return None


@functools.lru_cache(maxsize=None)
def package_files():
    """Relative posix paths of the ``.py`` files of ``cellacdc`` (generated
    Qt resources excluded), sorted."""
    return tuple(sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in PACKAGE_DIR.rglob('*.py')
        if not GENERATED.search(path.name)
        and '__pycache__' not in path.parts
    ))


@functools.lru_cache(maxsize=None)
def module_index():
    """{dotted module name: relative path} of every importable module of
    ``cellacdc``: ``.py`` files, packages (their ``__init__.py``), compiled
    extensions (``.pyd``/``.so``, Cython ``.pyx`` sources) and the modules
    generated at build time (path None)."""
    index = {module_name(rel): rel for rel in package_files()}
    for path in PACKAGE_DIR.rglob('*'):
        if path.suffix not in EXTENSION_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        stem = path.name.split('.')[0]
        dotted = module_name(str(PurePosixPath(rel).parent / stem) + '.py')
        index.setdefault(dotted, rel)
    for dotted in GENERATED_MODULES:
        index.setdefault(dotted, None)
    # Folders without __init__.py holding modules (namespace packages)
    for dotted in list(index):
        parts = dotted.split('.')
        for end in range(1, len(parts)):
            index.setdefault('.'.join(parts[:end]), None)
    return index


def is_module(dotted):
    return dotted in module_index()


def resolve_relative(importer_rel, level, module):
    """Absolute module of ``from <level dots><module> import ...`` in the
    file `importer_rel`; None when it leaves the top-level package."""
    package = module_name(importer_rel).split('.')
    if not is_package_file(importer_rel):
        package = package[:-1]
    if level - 1 > len(package) - 1:
        return None
    base = package[:len(package) - level + 1]
    return '.'.join(base + ([module] if module else []))


def import_from_module(node, importer_rel):
    """Absolute module name of an ``ast.ImportFrom`` (None if invalid)."""
    if node.level:
        return resolve_relative(importer_rel, node.level, node.module)
    return node.module


def is_shim(tree):
    """Whether the module sets ``__cellacdc_compat__ = True`` (compat shim
    at an old import path)."""
    if tree is None:
        return False
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == SHIM_MARKER
                    for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ):
            return True
    return False


@functools.lru_cache(maxsize=None)
def shim_modules():
    """Dotted names of the compat shims (files at old import paths)."""
    return frozenset(
        module_name(rel) for rel in package_files() if is_shim_file(rel)
    )


def is_shim_file(rel):
    """``is_shim`` of a file, from its text (without parsing it)."""
    return SHIM_LINE.search(read(rel)) is not None


def clear_caches():
    """Release the cached texts and syntax trees (hundreds of MB for the
    whole package): the test modules call it when they are done."""
    for function in (read, parse, package_files, module_index,
                     shim_modules, module_bindings, compat_moves):
        function.cache_clear()


def _target_names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def iter_module_level(body):
    """Statements executed at module level: `body` and the bodies of its
    ``if``/``try``/``with``/``for``/``while`` blocks, not of functions or
    classes."""
    for node in body:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        for field in ('body', 'orelse', 'finalbody'):
            yield from iter_module_level(getattr(node, field, None) or [])
        for handler in getattr(node, 'handlers', None) or []:
            yield from iter_module_level(handler.body)


def top_level_bindings(tree):
    """``(names, dynamic)``: the names the module binds at module level
    (also in ``if``/``try`` blocks and through ``global`` in functions);
    `dynamic` when other names may exist (module ``__getattr__``, star
    import, writes to ``globals()``)."""
    names = set()
    dynamic = False
    for node in iter_module_level(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == '*':
                    dynamic = True
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.name:
                    names.add(handler.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.update(node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'globals'
        ):
            dynamic = True
        elif isinstance(node, ast.NamedExpr):
            names.update(_target_names(node.target))
    if '__getattr__' in names:
        dynamic = True
    return names, dynamic


@functools.lru_cache(maxsize=None)
def module_bindings(dotted):
    """``top_level_bindings`` of the module `dotted` (``(set(), True)`` for
    modules without source: extensions and generated ones), plus its
    submodules."""
    rel = module_index().get(dotted)
    tree = parse(rel) if rel and rel.endswith('.py') else None
    if tree is None:
        names, dynamic = set(), True
    else:
        names, dynamic = top_level_bindings(tree)
    prefix = dotted + '.'
    names = set(names) | {
        name[len(prefix):] for name in module_index()
        if name.startswith(prefix) and '.' not in name[len(prefix):]
    }
    return frozenset(names), dynamic


def module_has_name(dotted, name):
    names, dynamic = module_bindings(dotted)
    return dynamic or name in names


@functools.lru_cache(maxsize=None)
def compat_moves():
    """Namespace of ``cellacdc/_compat_moves.py`` (generated tables of the
    old import paths: MODULE_ALIASES, NAME_MOVES, PLUGIN_API, ...)."""
    path = PACKAGE_DIR / '_compat_moves.py'
    namespace = {}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'),
         namespace)
    return namespace


def part_of(dotted):
    """The subpackage of ``cellacdc`` the module belongs to (``gui``,
    ``shared``, ``models``, ...), or None for top-level modules."""
    parts = dotted.split('.')
    if len(parts) < 2 or parts[0] != PACKAGE:
        return None
    if (PACKAGE_DIR / parts[1]).is_dir():
        return parts[1]
    return None


def is_string_statement(node):
    """Whether `node` is an expression statement made of a string
    (docstrings and string "comments")."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
