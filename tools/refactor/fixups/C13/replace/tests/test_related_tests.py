"""Tests of ``tests/related_tests.py`` (test files related to changed files)."""
import shutil
import subprocess
import sys
import textwrap

import pytest

from tests import related_tests
from tests.related_tests import ALL, Selection

pytestmark = pytest.mark.infra

FILES = {
    'cellacdc/__init__.py': 'printl = print\n',
    'cellacdc/gui.py': (
        'from cellacdc import basic\n'
        'from cellacdc.gui_mixin import Mixin\n'
    ),
    'cellacdc/gui_mixin.py': 'class Mixin: pass\n',
    'cellacdc/basic.py': '',
    # basic is shared: imported by gui and these 5 modules
    **{
        f'cellacdc/module_{i}.py': 'from cellacdc import basic\n'
        for i in range(5)
    },
    # Never importers: a compat shim at an old path and the compat tables
    # (they name the old and new paths of every moved module)
    'cellacdc/old_basic.py': (
        '"""Deprecated: moved to basic.py."""\n'
        '__cellacdc_compat__ = True\n'
        'from cellacdc import _compat\n'
        '_compat.alias_module(__name__)\n'
    ),
    'cellacdc/_compat_moves.py': (
        "MODULE_ALIASES = {'cellacdc.old_basic': 'cellacdc.basic'}\n"
        "NAME_MOVES = {'cellacdc.gui': {'X': ('cellacdc.gui_mixin', 'X')}}\n"
    ),
    # Docstrings and comment lines of package modules are not read
    'cellacdc/notes.py': (
        '"""See cellacdc.basic and cellacdc.gui_mixin."""\n'
        '# cellacdc.basic again\n'
    ),
    # A program part: the window composes six mixins, each importing the
    # part's viewer_utils (importers inside a part are followed
    # transitively, whatever their number)
    'cellacdc/measurements_viewer/__init__.py': '',
    'cellacdc/measurements_viewer/main_window/__init__.py': '',
    'cellacdc/measurements_viewer/main_window/window.py': (
        'from . import ' + ', '.join(f'mixin_{i}' for i in range(6)) + '\n'
    ),
    **{
        f'cellacdc/measurements_viewer/main_window/mixin_{i}.py':
            'from cellacdc.measurements_viewer import viewer_utils\n'
        for i in range(6)
    },
    'cellacdc/measurements_viewer/viewer_utils.py': '',
    # A shared module imported by six modules of one other part: one unit
    'cellacdc/shared/__init__.py': '',
    'cellacdc/shared/buttons.py': '',
    'cellacdc/data_prep/__init__.py': '',
    'cellacdc/data_prep/main_window.py': (
        'from . import ' + ', '.join(f'topic_{i}' for i in range(6)) + '\n'
    ),
    **{
        f'cellacdc/data_prep/topic_{i}.py':
            'from cellacdc.shared import buttons\n'
        for i in range(6)
    },
    # Skipped: generated Qt resources
    'cellacdc/qrc_resources_light.py': 'from cellacdc import gui_mixin\n',
    'cellacdc/viewer/__init__.py': 'from cellacdc.viewer import canvas\n',
    'cellacdc/viewer/canvas.py': '',
    'cellacdc/measurement_processing/__init__.py': '',
    'cellacdc/measurement_processing/Ramp/acdcProcess.py': '',
    'cellacdc/models/__init__.py': '',
    'cellacdc/models/thresholding/acdcSegment.py': '',
    'cellacdc/promptable_models/__init__.py': '',
    'cellacdc/promptable_models/sam_base.py': '',
    'cellacdc/promptable_models/sam-v1/acdcPromptSegment.py': (
        'from cellacdc.promptable_models import sam_base\n'
    ),
    'cellacdc/templates/fucci_cycle.json': '{}',
    'cellacdc/docs/source/page.rst': '',
    'tests/__init__.py': '',
    'tests/conftest.py': 'from tests.utils.helper_a import fixture_helper\n',
    'tests/utils/__init__.py': 'from .helper_b import helper\n',
    'tests/utils/helper_a.py': '',
    'tests/utils/helper_b.py': '',
    'tests/test_data/table.csv': '',
    'tests/test_gui.py': 'from cellacdc import gui, printl\n',
    'tests/test_basic.py': 'import cellacdc.basic as basic\n',
    'tests/test_plugins.py': 'from cellacdc import measurement_processing\n',
    'tests/test_patch.py': textwrap.dedent('''\
        import cellacdc

        def test_patch(monkeypatch):
            monkeypatch.setattr('cellacdc.gui.Window', None)
    '''),
    'tests/test_reuse.py': 'from tests.test_gui import helper\n',
    'tests/test_helper_b.py': 'from tests.utils.helper_b import helper\n',
    'tests/test_fucci.py': "TEMPLATE = 'fucci_cycle'\n",
    'tests/test_server.py': 'BACKEND = dict(model_name="thresholding")\n',
    'tests/test_sam.py': "MODEL = 'sam-v1'\n",
    'tests/test_words.py': '# thresholding, not quoted\n',
    'tests/test_invalid_syntax.py': 'from cellacdc import basic\ndef (\n',
    'tests/test_viewer_window.py': (
        'from cellacdc.measurements_viewer.main_window import window\n'
    ),
    'tests/test_viewer_utils.py': (
        'from cellacdc.measurements_viewer import viewer_utils\n'
    ),
    'tests/test_data_prep.py': 'from cellacdc.data_prep import main_window\n',
    'tests/.test_hidden.py': 'from cellacdc import gui\n',
}


@pytest.fixture
def repo(tmp_path):
    for name, text in FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return tmp_path


def related(repo, *changed):
    tests, _ = Selection(repo).related(list(changed))
    return tests


def test_only_collected_test_files_are_candidates(repo):
    assert '.test_hidden.py' not in ' '.join(Selection(repo).test_files)
    assert 'tests/conftest.py' not in Selection(repo).test_files


def test_tests_importing_or_naming_the_module_are_related(repo):
    assert related(repo, 'cellacdc/gui.py') == [
        'tests/test_gui.py', 'tests/test_patch.py',
    ]
    assert related(repo, 'cellacdc/basic.py') == [
        'tests/test_basic.py', 'tests/test_invalid_syntax.py',
    ]


def test_tests_of_the_importers_of_a_module_with_few_importers(repo):
    # gui_mixin is imported only by gui (qrc_resources* modules are skipped)
    assert Selection(repo).importers('cellacdc.gui_mixin') == {'cellacdc.gui'}
    assert related(repo, 'cellacdc/gui_mixin.py') == [
        'tests/test_gui.py', 'tests/test_patch.py',
    ]


def test_importing_packages_are_followed_except_the_top_level_one(repo):
    # cellacdc/__init__.py and cellacdc/viewer/__init__.py import canvas
    (repo / 'cellacdc/__init__.py').write_text(
        'from cellacdc.viewer import canvas\n', encoding='utf-8'
    )
    (repo / 'cellacdc/viewer/other.py').write_text('', encoding='utf-8')
    (repo / 'tests/test_viewer.py').write_text(
        'from cellacdc.viewer import other\n', encoding='utf-8'
    )
    assert Selection(repo).importers('cellacdc.viewer.canvas') == {
        'cellacdc.viewer'
    }
    # Importing cellacdc.viewer.other runs cellacdc/viewer/__init__.py
    assert related(repo, 'cellacdc/viewer/canvas.py') == [
        'tests/test_viewer.py'
    ]


def test_importers_of_shared_modules_are_not_followed(repo):
    importers = Selection(repo).importers('cellacdc.basic')
    # The shim and the compat tables name basic, but are no importers
    assert len(importers) == related_tests.MAX_IMPORTERS + 1
    # gui imports basic, but test_gui.py does not name basic
    assert 'tests/test_gui.py' not in related(repo, 'cellacdc/basic.py')


def test_compat_shims_docstrings_and_comments_are_no_importers(repo):
    selection = Selection(repo)
    assert selection.importers('cellacdc.gui_mixin') == {'cellacdc.gui'}
    importers = selection.importers('cellacdc.basic')
    assert not importers & {
        'cellacdc.old_basic', 'cellacdc._compat_moves', 'cellacdc.notes'
    }


def test_importers_inside_a_program_part_are_followed_transitively(repo):
    selection = Selection(repo)
    assert len(selection.importers(
        'cellacdc.measurements_viewer.viewer_utils'
    )) == related_tests.MAX_IMPORTERS + 1
    assert related(repo, 'cellacdc/measurements_viewer/viewer_utils.py') == [
        'tests/test_viewer_utils.py', 'tests/test_viewer_window.py',
    ]
    assert related(
        repo, 'cellacdc/measurements_viewer/main_window/mixin_3.py'
    ) == ['tests/test_viewer_window.py']


def test_importers_of_one_other_part_count_as_one(repo):
    assert len(Selection(repo).importers('cellacdc.shared.buttons')) == 6
    # The six modules of data_prep are one unit, followed inside the part
    assert related(repo, 'cellacdc/shared/buttons.py') == [
        'tests/test_data_prep.py',
    ]


def test_package_of_a_submodule_import_is_not_a_reference_to_all(repo):
    # test_viewer_utils.py imports viewer_utils from the package: that
    # runs the package __init__, not every module of the package
    assert 'tests/test_viewer_utils.py' not in related(
        repo, 'cellacdc/measurements_viewer/main_window/mixin_0.py'
    )
    assert 'tests/test_viewer_utils.py' in related(
        repo, 'cellacdc/measurements_viewer/__init__.py'
    )


def test_attributes_of_the_top_level_package_relate_to_nothing(repo):
    # test_gui.py imports printl from cellacdc: not related to every module
    assert 'tests/test_gui.py' not in related(
        repo, 'cellacdc/measurement_processing/__init__.py'
    )


def test_package_tests_are_related_to_its_plugins(repo):
    assert related(
        repo, 'cellacdc/measurement_processing/Ramp/acdcProcess.py'
    ) == ['tests/test_plugins.py']
    assert related(repo, 'cellacdc/measurement_processing/__init__.py') == [
        'tests/test_plugins.py',
    ]


def test_changed_test_file_and_tests_importing_it(repo):
    assert related(repo, 'tests/test_gui.py') == [
        'tests/test_gui.py', 'tests/test_reuse.py',
    ]


def test_deleted_test_file_is_not_selected(repo):
    assert related(repo, 'tests/test_removed.py') == []


def test_test_helpers(repo):
    assert related(repo, 'tests/utils/helper_b.py') == [
        'tests/test_helper_b.py',
    ]
    # The conftest imports helper_a: it can affect every test
    assert related(repo, 'tests/utils/helper_a.py') == ALL


@pytest.mark.parametrize('changed', [
    'cellacdc/__init__.py',
    'tests/conftest.py',
    'tests/__init__.py',
    'tests/utils/__init__.py',
    'tests/test_data/table.csv',
    'pyproject.toml',
    'conftest.py',
    'pytest.ini',
    'tox.ini',
    'setup.cfg',
    'requirements_gui_pyqt6.txt',
    '.github/actions/setup-cellacdc/action.yml',
])
def test_changes_that_relate_to_the_whole_suite(repo, changed):
    tests, reasons = Selection(repo).related(['cellacdc/gui.py', changed])
    assert tests == ALL
    assert reasons[changed] == 'the whole suite'


def test_data_files_relate_to_tests_mentioning_them(repo):
    assert related(repo, 'cellacdc/templates/fucci_cycle.json') == [
        'tests/test_fucci.py',
    ]
    # Generic folder names ('source', 'docs') select nothing
    assert related(repo, 'cellacdc/docs/source/page.rst') == []


def test_files_outside_the_packages_relate_to_nothing(repo):
    assert related(
        repo, 'README.rst', '.github/workflows/build.yml', 'docs/index.rst',
        'docs/conftest.py',
    ) == []


def test_mentions_matches_whole_words_only():
    assert related_tests.mentions("load('fucci_cycle')", 'fucci_cycle')
    assert not related_tests.mentions('fucci_cycle_v2', 'fucci_cycle')
    assert not related_tests.mentions('my-fucci_cycle', 'fucci_cycle')


@pytest.mark.slow  # parses every module of cellacdc (~5 s)
def test_selection_of_this_repository():
    selection = Selection()
    # The phase translator module, from the path its tests import it
    module, = [
        name for name in selection.references(
            'tests/test_phase_translator.py'
        )
        if name.split('.')[-1] == 'phase_translator'
    ]
    assert module == 'cellacdc.measurements_viewer.phase_translator'
    tests, _ = selection.related([module.replace('.', '/') + '.py'])
    assert 'tests/test_phase_translator.py' in tests
    # Imported by the viewer's mixin modules: the viewer's tests
    tests, _ = selection.related(
        ['cellacdc/measurements_viewer/viewer_utils.py']
    )
    assert 'tests/test_measurement_viewer.py' in tests
    # A mixin shared by both windows: the tests of both
    tests, _ = selection.related(['cellacdc/gui/mixins/navigation.py'])
    assert {
        'tests/test_measurement_viewer.py', 'tests/test_keep_ids.py'
    } <= set(tests)
    # Compat shims at old paths are no importers
    importers = selection.importers('cellacdc.gui.mixins.navigation')
    assert importers and not any(
        related_tests.SHIM_MARKER.search(
            selection.text(selection._module_paths[name])
        )
        for name in importers
    )
    tests, _ = selection.related(['tests/related_tests.py'])
    assert tests == [
        'tests/test_area_markers.py', 'tests/test_related_tests.py',
    ]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which('git') is None, reason='git is not installed')
def test_changed_files_include_commits_and_uncommitted_changes(repo, capsys):
    def git(*args):
        subprocess.run(
            ['git', '-c', 'user.name=test', '-c', 'user.email=test@example.com',
             '-c', 'commit.gpgsign=false', *args],
            cwd=repo, check=True, capture_output=True,
        )

    git('init', '-q', '-b', 'main')
    git('add', '.')
    git('commit', '-q', '-m', 'base')
    git('switch', '-q', '-c', 'feature')
    (repo / 'cellacdc/gui.py').write_text('# committed\n', encoding='utf-8')
    git('commit', '-q', '-am', 'change gui')
    (repo / 'cellacdc/basic.py').write_text('# unstaged\n', encoding='utf-8')
    (repo / 'tests/test new.py').write_text('', encoding='utf-8')
    # git writes UTF-8 names whatever the locale (cp1252 on Windows)
    (repo / 'cellacdc/Ábc.txt').write_text('', encoding='utf-8')

    assert related_tests.changed_files('main', repo) == [
        'cellacdc/basic.py', 'cellacdc/gui.py', 'cellacdc/Ábc.txt',
        'tests/test new.py',
    ]
    assert related_tests.changed_files('unknown-base', repo) is None
    assert 'Cannot list the changed files' in capsys.readouterr().err

    assert related_tests.main(['--base', 'main'], root=repo) == 0
    out = capsys.readouterr().out.split()
    assert out == [
        'tests/test_basic.py', 'tests/test_gui.py',
        'tests/test_invalid_syntax.py', 'tests/test_patch.py',
    ]
    # Unknown base: the whole suite
    assert related_tests.main(['--base', 'unknown-base'], root=repo) == 0
    assert capsys.readouterr().out.split() == [ALL]


def test_plugin_modules_relate_to_tests_naming_their_folder(repo):
    assert related(repo, 'cellacdc/models/thresholding/acdcSegment.py') == [
        'tests/test_server.py',
    ]


def test_shared_base_modules_relate_to_tests_naming_their_plugins(repo):
    # The plugin imports the base module, the test names only the plugin
    # folder (a dashed folder name is no importable module name)
    assert related(repo, 'cellacdc/promptable_models/sam_base.py') == [
        'tests/test_sam.py',
    ]


def test_fails_safe_on_errors(repo, monkeypatch, capsys):
    monkeypatch.setattr(
        related_tests, 'changed_files', lambda base, root: ['cellacdc/gui.py']
    )

    def fail(self, files):
        raise RuntimeError('bug')

    monkeypatch.setattr(Selection, 'related', fail)
    assert related_tests.select('main', repo) == ALL
    assert 'RuntimeError' in capsys.readouterr().err


def test_run_calls_pytest_on_the_selection_only(repo, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        related_tests.subprocess, 'call',
        lambda args, cwd: calls.append((args, cwd)) or 0,
    )
    changed = []
    monkeypatch.setattr(
        related_tests, 'changed_files', lambda base, root: changed
    )
    # Nothing related: no bare pytest (that would run the whole suite)
    assert related_tests.main(['--run', '-m', 'slow'], root=repo) == 0
    assert not calls
    assert 'No test file is related' in capsys.readouterr().err

    changed.append('cellacdc/gui.py')
    assert related_tests.main(['--run', '-m', 'slow', '-x'], root=repo) == 0
    assert calls == [([
        sys.executable, '-m', 'pytest', 'tests/test_gui.py',
        'tests/test_patch.py', '-m', 'slow', '-x',
    ], repo)]
    # pytest arguments are accepted only with --run
    with pytest.raises(SystemExit):
        related_tests.main(['-m', 'slow'], root=repo)
