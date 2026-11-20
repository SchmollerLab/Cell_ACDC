"""Static Qt-free rules for the BioIO layer and colors.py.

The modules of `acdc_bioio_bioformats` are QF modules (raw_metadata.py and
writer.py also get the new-file rules); `colors.py` is not QF, it must only
not import `debugutils` at module level (the root of the Qt-less circular
import).
"""
from __future__ import annotations

import ast
import os
import subprocess

import pytest

from . import qt_free_rules
from .conditions import REPO_ROOT

BIOIO_DIR = os.path.join(qt_free_rules.PACKAGE_DIR, 'acdc_bioio_bioformats')
BIOIO_MODULES = (
    '__init__', 'reader', '_utils', 'install', 'raw_metadata', 'writer'
)
NEW_MODULES = ('raw_metadata', 'writer')
COLORS_PY = os.path.join(qt_free_rules.PACKAGE_DIR, 'colors.py')
MAIN_COMMIT = '7c0f7532'


@pytest.mark.parametrize('name', BIOIO_MODULES)
def test_bioio_module_follows_the_qt_free_rules(name):
    relpath = f'acdc_bioio_bioformats/{name}.py'
    flags = qt_free_rules.rules_for(relpath)
    assert flags is not None and flags['qf']
    assert flags['new_file'] is (name in NEW_MODULES)
    violations = qt_free_rules.check_file(
        os.path.join(BIOIO_DIR, f'{name}.py'), relpath=relpath, **flags
    )
    assert not violations, qt_free_rules.format_violations(violations)


@pytest.mark.parametrize('name', NEW_MODULES)
def test_new_modules_do_not_import_the_datastruct_core(name):
    refs = qt_free_rules.imports_of(os.path.join(BIOIO_DIR, f'{name}.py'))
    offending = [
        ref.text for ref in refs
        if any(t.startswith('cellacdc.acdc_datastruct') for t in ref.targets)
    ]
    assert not offending


def _module_level_debugutils_imports(source: str) -> list[str]:
    refs = qt_free_rules.collect_imports(
        ast.parse(source), source, relpath='colors.py'
    )
    return [
        ref.text for ref in refs
        if ref.module_level and not ref.type_checking
        and 'cellacdc.debugutils' in ref.targets + ref.modules
    ]


def test_colors_does_not_import_debugutils_at_module_level():
    with open(COLORS_PY, encoding='utf-8') as file:
        assert _module_level_debugutils_imports(file.read()) == []


def test_debugutils_check_flags_main_colors():
    """The check above would have caught main's colors.py:14."""
    result = subprocess.run(
        ['git', '-C', REPO_ROOT, 'show', f'{MAIN_COMMIT}:cellacdc/colors.py'],
        capture_output=True
    )
    if result.returncode != 0:
        pytest.skip(f'commit {MAIN_COMMIT} not available (shallow clone)')
    source = result.stdout.decode('utf-8')
    assert _module_level_debugutils_imports(source) == [
        'from . import GUI_INSTALLED, printl, debugutils'
    ]
