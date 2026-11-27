"""The import layering of the package layout (CODEMAP.md, placement rules).

- ``shared/`` and the library modules (everything outside the program-part
  packages ``source_scan.PART_PACKAGES``) never import a part package at
  module level: library code must not load windows.
- Imports between two program parts are listed in ``CROSS_PART_IMPORTS``
  (with the reason), module-level ones marked as such: code used by
  several parts belongs in ``shared/`` (or a library module), so a new
  cross-part import needs a review.
- ``gui/mixins/`` (composed by the main GUI and the measurements viewer)
  and ``measurements_viewer/`` never import the main GUI's own code
  (``gui/main_window``, ``gui/dialogs``, ``gui/widgets``, ``gui/workers``,
  ``gui/helpers.py``), at any level.
- Foundation modules, whose classes other modules subclass at import
  time, import no member of an import cycle at module level (except the
  documented ``KNOWN_CYCLE_IMPORTS``), and the guiWin mixin modules never
  import ``window.py`` at module level.
- The modules that take part in an import cycle, and the windows, import
  first in a fresh interpreter (import order).

Imports under ``if TYPE_CHECKING:`` never run and are ignored.
"""
import ast
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests import source_scan
from tests.source_scan import PACKAGE, PART_PACKAGES, REPO_ROOT

pytestmark = [pytest.mark.infra, pytest.mark.slow]


@pytest.fixture(scope='module', autouse=True)
def _release_parsed_sources():
    """The parsed package takes hundreds of MB: release it after the
    module (the session keeps running other tests)."""
    yield
    source_scan.clear_caches()


# (importer prefix, imported prefix): (level, reason); level 'module' also
# allows module-level imports, 'lazy' only imports inside functions
CROSS_PART_IMPORTS = {
    ('cellacdc.launcher.main_window', 'cellacdc.gui.main_window.window'):
        ('module', 'the launcher opens the module windows'),
    ('cellacdc.launcher.main_window', 'cellacdc.data_prep.main_window'):
        ('module', 'the launcher opens the module windows'),
    ('cellacdc.launcher.main_window', 'cellacdc.data_structure'):
        ('module', 'the launcher opens the module windows'),
    ('cellacdc.launcher.main_window',
     'cellacdc.segmentation_module.main_window'):
        ('module', 'the launcher opens the module windows'),
    ('cellacdc.launcher.main_window', 'cellacdc.utils'):
        ('module', 'the launcher opens the utilities'),
    ('cellacdc.launcher.main_window',
     'cellacdc.measurements_viewer.main_window.window'):
        ('lazy', 'the launcher opens the measurements viewer'),
    ('cellacdc.data_structure.data_restruct', 'cellacdc.launcher.main_window'):
        ('module', 'run() annotates the launcher window it returns to'),
    ('cellacdc.measurements_viewer', 'cellacdc.gui.mixins'):
        ('module', 'the viewer window composes the shared GUI mixins'),
    ('cellacdc.measurements_viewer.main_window.configuration',
     'cellacdc.utils.computeBackgroundValues'):
        ('lazy', 'premade setup: calculate the background values'),
    ('cellacdc.gui.mixins.dialogs',
     'cellacdc.measurements_viewer.dialogs.predictor_editors'):
        ('lazy', 'customAnnotationDialog shows the predictor editor only '
                 'in the viewer (predictorOptions)'),
    ('cellacdc.gui.main_window', 'cellacdc.data_prep.main_window'):
        ('module', 'guiWin selects the z-slice/ROI with the data prep '
                   'window'),
    ('cellacdc.segmentation_module', 'cellacdc.data_prep.main_window'):
        ('module', 'the segmentation module selects the z-slice/ROI with '
                   'the data prep window'),
    ('cellacdc.segmentation_module.main_window', 'cellacdc.utils.compute'):
        ('lazy', 'compute measurements after segmentation'),
    ('cellacdc.utils.compute', 'cellacdc.gui.main_window.window'):
        ('module', 'the utility computes measurements with a hidden guiWin'),
}
# Library modules (outside the parts) that import a part at module level
LIBRARY_IMPORTS_OF_PARTS = {
    ('cellacdc.help.welcome', 'cellacdc.data_structure.data_struct'):
        'the welcome guide opens the data structure module',
    ('cellacdc.napari_utils.arboretum', 'cellacdc.utils.base'):
        'the napari arboretum plugin is a utility',
    ('cellacdc.scripts', 'cellacdc.utils.base'): 'development scripts',
}
# Lazy imports of parts by library modules
LIBRARY_LAZY_IMPORTS_OF_PARTS = {
    ('cellacdc.__main__', 'cellacdc.launcher.main_window'):
        'the entry point starts the launcher',
    ('cellacdc.help.welcome', 'cellacdc.gui.main_window.window'):
        'the welcome guide opens the main GUI',
    ('cellacdc.debugutils', 'cellacdc.gui.main_window.window'):
        'check_unused_methods (development helper) inspects guiWin',
}
GUI_ONLY = tuple(
    f'cellacdc.gui.{name}' for name in (
        'main_window', 'dialogs', 'widgets', 'workers', 'helpers'
    )
)
VIEWER_SHARED = ('cellacdc.gui.mixins', 'cellacdc.measurements_viewer')
# Modules whose names other modules use at import time (base classes,
# decorators): they must not be part of an import cycle
FOUNDATION_LEAVES = (
    'cellacdc.shared.dialogs.base',
    'cellacdc.shared.widgets.style',
    'cellacdc.shared.widgets.numeric_inputs',
    'cellacdc.shared.widgets.search_widgets',
    'cellacdc.shared.widgets.plot_items',
    'cellacdc.shared.widgets.push_buttons',
    'cellacdc.shared.widgets.toolbars',
    'cellacdc.shared.workers.base',
    'cellacdc.gui.main_window.constants',
)
# Module-level imports of cycle members by foundation modules that the
# package layout kept from the old modules: safe because the foundation
# module uses them only inside functions (the fresh-interpreter test
# below imports every cycle member first). Make them lazy when you work
# on these modules; never add new ones.
KNOWN_CYCLE_IMPORTS = {
    'cellacdc.shared.widgets.search_widgets': {'cellacdc.html_utils'},
    'cellacdc.shared.widgets.plot_items': {'cellacdc.load'},
    'cellacdc.shared.widgets.push_buttons': {
        'cellacdc.colors', 'cellacdc.html_utils',
        'cellacdc.myutils.filesystem',
        'cellacdc.shared.widgets.basic_widgets',
        'cellacdc.shared.widgets.message_box',
    },
    'cellacdc.shared.widgets.toolbars': {
        'cellacdc.html_utils', 'cellacdc.plot',
        'cellacdc.shared.widgets.basic_widgets',
        'cellacdc.shared.widgets.combo_boxes',
    },
    'cellacdc.shared.workers.base': {'cellacdc.myutils.image_utils'},
}
# Windows: imported first in a fresh interpreter too
WINDOW_MODULES = (
    'cellacdc.gui.main_window.window',
    'cellacdc.measurements_viewer.main_window.window',
    'cellacdc.launcher.main_window',
    'cellacdc.data_prep.main_window',
    'cellacdc.data_structure.data_struct',
    'cellacdc.segmentation_module.main_window',
)
SKIPPED_FOLDERS = ('cellacdc/_deprecated/',)


def _is_type_checking(node):
    test = node.test
    return (
        (isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING')
        or (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING')
    )


def _module_level(body):
    for node in body:
        if isinstance(node, ast.If) and _is_type_checking(node):
            yield from _module_level(node.orelse)
            continue
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        for field in ('body', 'orelse', 'finalbody'):
            yield from _module_level(getattr(node, field, None) or [])
        for handler in getattr(node, 'handlers', None) or []:
            yield from _module_level(handler.body)


def _type_checking_nodes(tree):
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking(node):
            for stmt in node.body:
                nodes.update(id(sub) for sub in ast.walk(stmt))
    return nodes


def imported_modules(rel, node):
    """cellacdc modules an import statement imports (the module, or the
    submodule for ``from <package> import <submodule>``)."""
    names = set()
    if isinstance(node, ast.Import):
        names.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        module = source_scan.import_from_module(node, rel)
        if module is None:
            return set()
        for alias in node.names:
            full = f'{module}.{alias.name}'
            names.add(full if source_scan.is_module(full) else module)
    return {
        name for name in names
        if name.split('.')[0] == PACKAGE and source_scan.is_module(name)
    }


class Imports:
    """Every import of cellacdc modules by cellacdc modules, with whether
    it runs at module level."""

    def __init__(self):
        self.edges = []        # (importer, imported, module_level, line)
        for rel in source_scan.package_files():
            if rel.startswith(SKIPPED_FOLDERS):
                continue
            tree = source_scan.parse(rel)
            if tree is None or source_scan.is_shim(tree):
                continue
            importer = source_scan.module_name(rel)
            top = {id(node) for node in _module_level(tree.body)}
            skipped = _type_checking_nodes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)) or \
                        id(node) in skipped:
                    continue
                for module in imported_modules(rel, node):
                    if module != importer:
                        self.edges.append(
                            (importer, module, id(node) in top, node.lineno)
                        )

    def module_level_graph(self):
        graph = {}
        for importer, module, module_level, _ in self.edges:
            graph.setdefault(importer, set())
            # The package's own __init__ (globals such as printl) is no
            # import-order hazard: it is always initialized first
            if module_level and module != PACKAGE:
                graph[importer].add(module)
        return graph


@pytest.fixture(scope='module')
def imports():
    return Imports()


def matches(name, prefix):
    return name == prefix or name.startswith(prefix + '.')


def is_part(module):
    return source_scan.part_of(module) in PART_PACKAGES


def strongly_connected(graph):
    """Strongly connected components with more than one module (Tarjan,
    iterative)."""
    index, low, on_stack, stack, components = {}, {}, set(), [], []
    counter = [0]
    for root in sorted(graph):
        if root in index:
            continue
        work = [(root, iter(sorted(graph.get(root, ()))))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            for child in children:
                if child not in graph:
                    continue
                if child not in index:
                    index[child] = low[child] = counter[0]
                    counter[0] += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph[child]))))
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1:
                        components.append(sorted(component))
    return components


def test_library_modules_do_not_import_program_parts(imports):
    wrong = []
    for importer, module, module_level, line in imports.edges:
        if is_part(importer) or not is_part(module):
            continue
        allowed = LIBRARY_IMPORTS_OF_PARTS if module_level else {
            **LIBRARY_IMPORTS_OF_PARTS, **LIBRARY_LAZY_IMPORTS_OF_PARTS}
        if any(matches(importer, a) and matches(module, b)
               for a, b in allowed):
            continue
        level = 'module level' if module_level else 'inside a function'
        wrong.append(f'{importer}:{line} imports {module} ({level})')
    assert not wrong, (
        'Library and shared code must not import the program parts '
        '(move what several parts use to shared/):\n' + '\n'.join(wrong)
    )


def test_imports_between_program_parts_are_reviewed(imports):
    wrong = []
    for importer, module, module_level, line in imports.edges:
        if not (is_part(importer) and is_part(module)):
            continue
        if source_scan.part_of(importer) == source_scan.part_of(module):
            continue
        for (a, b), (level, _) in CROSS_PART_IMPORTS.items():
            if matches(importer, a) and matches(module, b) and (
                    level == 'module' or not module_level):
                break
        else:
            level = 'module level' if module_level else 'inside a function'
            wrong.append(f'{importer}:{line} imports {module} ({level})')
    assert not wrong, (
        'New import between program parts: move the shared code to '
        'shared/, or list the import in CROSS_PART_IMPORTS with the '
        'reason:\n' + '\n'.join(wrong)
    )


def test_viewer_and_shared_mixins_never_import_main_gui_code(imports):
    wrong = [
        f'{importer}:{line} imports {module}'
        for importer, module, _, line in imports.edges
        if any(matches(importer, p) for p in VIEWER_SHARED)
        and any(matches(module, p) for p in GUI_ONLY)
    ]
    assert not wrong, '\n'.join(wrong)


def test_foundation_modules_are_not_in_import_cycles(imports):
    graph = imports.module_level_graph()
    cycles = strongly_connected(graph)
    members = set().union(*map(set, cycles)) if cycles else set()
    wrong = {}
    for leaf in FOUNDATION_LEAVES:
        assert leaf in graph, f'{leaf} does not exist any more'
        found = (graph[leaf] & members) - KNOWN_CYCLE_IMPORTS.get(leaf, set())
        if found:
            wrong[leaf] = sorted(found)
    assert not wrong, (
        'Foundation modules must import cycle members only inside '
        f'functions: {wrong}'
    )
    window_users = {
        importer for importer, module, module_level, _ in imports.edges
        if module_level and module == 'cellacdc.gui.main_window.window'
        and matches(importer, 'cellacdc.gui.main_window')
    }
    assert not window_users, (
        'guiWin mixin modules must not import window.py at module level: '
        f'{sorted(window_users)}'
    )


def _import_first(module):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    env['PYTHONPATH'] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in [env.get('PYTHONPATH')] if p]
    )
    proc = subprocess.run(
        [sys.executable, '-W', 'error:Cell-ACDC import path moved',
         '-c', f'import {module}'],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        return module, proc.stderr.strip().splitlines()[-1:] or ['failed']
    return module, None


def test_cycle_members_and_windows_import_first(imports):
    """Each module of an import cycle (and each window) imports when it is
    the first cellacdc module imported in a fresh interpreter, without
    using an old import path."""
    graph = imports.module_level_graph()
    modules = set(WINDOW_MODULES)
    for component in strongly_connected(graph):
        modules.update(
            m for m in component if source_scan.part_of(m) in
            source_scan.NEW_PACKAGES
        )
    workers = min(8, os.cpu_count() or 2)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(_import_first, sorted(modules)))
    failed = {module: error for module, error in results if error}
    assert not failed, failed
