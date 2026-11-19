"""Static Qt-free rules over cellacdc/_entry_points.py, cellacdc/data_cli.py
and every cellacdc/acdc_datastruct/*.py that exists (globbed, so modules
added later are covered without editing this file), plus layering, GUI lazy
imports and package-layout checks."""
from __future__ import annotations

import ast
import os
import sys

import pytest

from . import qt_free_rules as rules

PACKAGE_DIR = rules.PACKAGE_DIR
DATASTRUCT = 'acdc_datastruct'


def _qf_relpaths() -> list[str]:
    relpaths = ['_entry_points.py', 'data_cli.py']
    folder = os.path.join(PACKAGE_DIR, DATASTRUCT)
    if os.path.isdir(folder):
        relpaths.extend(
            f'{DATASTRUCT}/{name}' for name in sorted(os.listdir(folder))
            if name.endswith('.py')
        )
    return [
        relpath for relpath in relpaths
        if os.path.isfile(os.path.join(PACKAGE_DIR, relpath))
    ]


QF_RELPATHS = _qf_relpaths()

# Layers 1-3 of acdc_datastruct (see its docstring), by rank: a module
# imports only siblings of a lower rank
LAYER_RANK = {
    'errors': 0, 'naming': 1, 'units': 1, 'callbacks': 1, 'sources': 2,
    'metadata': 3, 'planning': 4, 'writers': 5,
}
STDLIB_CORE = ('errors', 'naming', 'units', 'callbacks')
PACKAGE_TEST_FILES = frozenset({
    'test_qt_app.py', 'test_segm_model.py', 'test_tracker.py',
})
"""test_*.py files that already exist inside the cellacdc package."""


def _path(relpath: str) -> str:
    return os.path.join(PACKAGE_DIR, *relpath.split('/'))


def _datastruct_imports(relpath: str) -> list[str]:
    """acdc_datastruct sibling modules imported (anywhere) by relpath."""
    prefix = f'cellacdc.{DATASTRUCT}.'
    siblings = []
    for ref in rules.imports_of(_path(relpath), relpath=relpath):
        for module in ref.modules + ref.targets:
            if module.startswith(prefix):
                siblings.append(module[len(prefix):].split('.')[0])
    return sorted(set(siblings))


def test_qt_free_rules_cover_the_layered_modules():
    for name in LAYER_RANK:
        assert f'{DATASTRUCT}/{name}.py' in QF_RELPATHS
    assert f'{DATASTRUCT}/__init__.py' in QF_RELPATHS
    assert '_entry_points.py' in QF_RELPATHS


@pytest.mark.parametrize('relpath', QF_RELPATHS)
def test_qt_free_rules(relpath):
    """Qt imports, forbidden/non-allowlisted cellacdc imports, import
    style, Qt names, stdlib-only (SO files), new-file rules and the
    parent-never-opens-raw rule."""
    flags = rules.rules_for(relpath)
    assert flags is not None and flags['qf'] and flags['new_file']
    violations = rules.check_file(_path(relpath), relpath=relpath, **flags)
    assert not violations, rules.format_violations(violations)


def test_rule_flags_of_layered_modules_and_entry_points():
    for name in ('sources', 'metadata', 'planning', 'writers'):
        assert rules.rules_for(f'{DATASTRUCT}/{name}.py')[
            'parent_never_opens_raw']
    assert rules.rules_for('_entry_points.py')['stdlib_only']


def test_package_init_has_no_imports():
    with open(_path(f'{DATASTRUCT}/__init__.py'), encoding='utf-8') as file:
        tree = ast.parse(file.read())
    assert ast.get_docstring(tree)
    assert len(tree.body) == 1, 'acdc_datastruct/__init__.py: docstring only'


@pytest.mark.parametrize('name', STDLIB_CORE)
def test_core_helpers_are_stdlib_only(name):
    relpath = f'{DATASTRUCT}/{name}.py'
    for ref in rules.imports_of(_path(relpath), relpath=relpath):
        for target in ref.targets:
            top = target.split('.')[0]
            if top == 'cellacdc':
                sibling = target.split('.')[2]
                assert sibling in STDLIB_CORE, ref.text
            else:
                assert top in sys.stdlib_module_names, ref.text


@pytest.mark.parametrize('name', list(LAYER_RANK))
def test_layering(name):
    """Lower layers never import higher ones; in particular writers
    imports planning and planning never imports writers."""
    for sibling in _datastruct_imports(f'{DATASTRUCT}/{name}.py'):
        assert sibling in LAYER_RANK, f'{name} imports {sibling}'
        assert LAYER_RANK[sibling] < LAYER_RANK[name], (
            f'{name} imports the higher layer {sibling}'
        )


def test_writers_imports_planning_not_the_reverse():
    assert 'planning' in _datastruct_imports(f'{DATASTRUCT}/writers.py')
    assert 'writers' not in _datastruct_imports(f'{DATASTRUCT}/planning.py')


@pytest.mark.parametrize('relpath', rules.GUI_LAZY_IMPORT_FILES)
def test_gui_modules_import_the_core_lazily(relpath):
    """apps.py / widgets.py import acdc_datastruct inside methods."""
    violations = rules.check_gui_lazy_imports(_path(relpath))
    assert not violations, rules.format_violations(violations)


def _package_dirs():
    for dirpath, dirnames, _ in os.walk(PACKAGE_DIR):
        dirnames[:] = [
            d for d in dirnames
            if d != '__pycache__' and not d.startswith('.')
        ]
        yield dirpath


def test_module_names_unique_ignoring_case():
    duplicates = []
    for dirpath in _package_dirs():
        names: dict[str, str] = {}
        for entry in sorted(os.listdir(dirpath)):
            full = os.path.join(dirpath, entry)
            if entry.endswith('.py'):
                name = entry[:-3]
            elif os.path.isdir(full) and entry != '__pycache__':
                name = entry
            else:
                continue
            key = name.casefold()
            if key in names and names[key] != name:
                duplicates.append(
                    f'{os.path.relpath(dirpath, PACKAGE_DIR)}: '
                    f'{names[key]} / {name}'
                )
            names.setdefault(key, name)
    assert not duplicates, duplicates


def test_no_shadowing_packages():
    for name in ('core', 'data'):
        assert not os.path.isdir(os.path.join(PACKAGE_DIR, name)), name


def test_no_new_test_files_inside_the_package():
    found = set()
    for dirpath in _package_dirs():
        for entry in os.listdir(dirpath):
            if entry.startswith('test_') and entry.endswith('.py'):
                relpath = os.path.relpath(
                    os.path.join(dirpath, entry), PACKAGE_DIR
                )
                found.add(relpath.replace(os.sep, '/'))
    assert found <= PACKAGE_TEST_FILES, sorted(found - PACKAGE_TEST_FILES)
