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
        'from cellacdc import widgets\n'
        'from cellacdc.gui_mixin import Mixin\n'
    ),
    'cellacdc/gui_mixin.py': 'class Mixin: pass\n',
    'cellacdc/widgets.py': '',
    # widgets is shared: imported by gui and these 5 modules
    **{
        f'cellacdc/module_{i}.py': 'from cellacdc import widgets\n'
        for i in range(5)
    },
    # Skipped: generated Qt resources
    'cellacdc/qrc_resources_light.py': 'from cellacdc import gui_mixin\n',
    'cellacdc/viewer/__init__.py': 'from cellacdc.viewer import canvas\n',
    'cellacdc/viewer/canvas.py': '',
    'cellacdc/measurement_processing/__init__.py': '',
    'cellacdc/measurement_processing/Ramp/acdcProcess.py': '',
    'cellacdc/models/__init__.py': '',
    'cellacdc/models/thresholding/acdcSegment.py': '',
    'cellacdc/templates/fucci_cycle.json': '{}',
    'cellacdc/docs/source/page.rst': '',
    'tests/__init__.py': '',
    'tests/conftest.py': 'from tests.utils.helper_a import fixture_helper\n',
    'tests/utils/__init__.py': 'from .helper_b import helper\n',
    'tests/utils/helper_a.py': '',
    'tests/utils/helper_b.py': '',
    'tests/test_data/table.csv': '',
    'tests/test_gui.py': 'from cellacdc import gui, printl\n',
    'tests/test_widgets.py': 'import cellacdc.widgets as widgets\n',
    'tests/test_plugins.py': 'from cellacdc import measurement_processing\n',
    'tests/test_patch.py': textwrap.dedent('''\
        import cellacdc

        def test_patch(monkeypatch):
            monkeypatch.setattr('cellacdc.gui.guiWin', None)
    '''),
    'tests/test_reuse.py': 'from tests.test_gui import helper\n',
    'tests/test_helper_b.py': 'from tests.utils.helper_b import helper\n',
    'tests/test_fucci.py': "TEMPLATE = 'fucci_cycle'\n",
    'tests/test_server.py': 'BACKEND = dict(model_name="thresholding")\n',
    'tests/test_words.py': '# thresholding, not quoted\n',
    'tests/test_invalid_syntax.py': 'from cellacdc import widgets\ndef (\n',
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
    assert related(repo, 'cellacdc/widgets.py') == [
        'tests/test_invalid_syntax.py', 'tests/test_widgets.py',
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
    importers = Selection(repo).importers('cellacdc.widgets')
    assert len(importers) == related_tests.MAX_IMPORTERS + 1
    # gui imports widgets, but test_gui.py does not name widgets
    assert 'tests/test_gui.py' not in related(repo, 'cellacdc/widgets.py')


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
    tests, _ = selection.related(['cellacdc/phase_translator.py'])
    assert 'tests/test_phase_translator.py' in tests
    tests, _ = selection.related(['tests/related_tests.py'])
    assert tests == ['tests/test_related_tests.py']


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
    (repo / 'cellacdc/widgets.py').write_text('# unstaged\n', encoding='utf-8')
    (repo / 'tests/test new.py').write_text('', encoding='utf-8')
    # git writes UTF-8 names whatever the locale (cp1252 on Windows)
    (repo / 'cellacdc/Ábc.txt').write_text('', encoding='utf-8')

    assert related_tests.changed_files('main', repo) == [
        'cellacdc/gui.py', 'cellacdc/widgets.py', 'cellacdc/Ábc.txt',
        'tests/test new.py',
    ]
    assert related_tests.changed_files('unknown-base', repo) is None
    assert 'Cannot list the changed files' in capsys.readouterr().err

    assert related_tests.main(['--base', 'main'], root=repo) == 0
    out = capsys.readouterr().out.split()
    assert out == [
        'tests/test_gui.py', 'tests/test_invalid_syntax.py',
        'tests/test_patch.py', 'tests/test_widgets.py',
    ]
    # Unknown base: the whole suite
    assert related_tests.main(['--base', 'unknown-base'], root=repo) == 0
    assert capsys.readouterr().out.split() == [ALL]


def test_plugin_modules_relate_to_tests_naming_their_folder(repo):
    assert related(repo, 'cellacdc/models/thresholding/acdcSegment.py') == [
        'tests/test_server.py',
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
