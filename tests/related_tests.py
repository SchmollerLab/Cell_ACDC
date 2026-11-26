"""List the test files related to the files changed on a branch.

The fast tier of the tests runs on every push to a pull request (see
``.github/workflows/build-*.yml``). The CI job ``related-tests`` then runs the
slow and flaky tests of the test files this script selects for the changed
files::

    python tests/related_tests.py --base HEAD^1

It prints one test path per line (nothing when no test file is related), or
``tests`` when the whole suite is related, and explains the selection on
stderr. Without ``--base`` it compares with the merge base of
``origin/main``, including uncommitted and untracked files. With ``--run``
it runs pytest on the selection instead, passing on the other arguments, and
does nothing when no test file is related (a bare ``pytest`` would run the
whole suite)::

    python tests/related_tests.py --run -m "slow or flaky"

A test file is related to a changed file when

* it is the changed test file (``tests/**/test_*.py``);
* it imports or names the changed module (``import cellacdc.gui``,
  ``from cellacdc import gui``, ``'cellacdc.gui.guiWin'`` in a
  ``monkeypatch.setattr``), one of its submodules or a package containing it
  other than ``cellacdc`` itself (``cellacdc.measurement_processing`` for a
  change in one of its plugins); the same holds for the helper modules in
  ``tests`` (``tests.test_measurement_viewer``, ``tests.utils.segmentation``);
* it imports or names a module of ``cellacdc`` that imports the changed
  module, when at most ``MAX_IMPORTERS`` modules import it: a change in the
  mixin ``gui_annotations`` selects the tests of ``gui`` and
  ``measurements_viewer``, while the importers of shared modules such as
  ``myutils``, ``widgets`` or ``load`` (dozens) are not followed, else nearly
  every change would select every test;
* for a module of a plugin, which Cell-ACDC imports by its folder name
  (``cellacdc/models/<name>/``, ``trackers``, ``promptable_models``,
  ``measurement_processing``), it contains the folder name as a string
  literal (``LocalBackend(model_name="thresholding")``);
* for another file in ``cellacdc`` (configuration templates, docs, icons,
  compiled extensions), it contains the file name, its stem or the name of
  its folder.

Deeper dependencies are not followed: the fast tier, which runs everything
that is neither slow nor flaky, and the full suite before merging cover them.

The whole suite is related when ``cellacdc/__init__.py``, a ``conftest.py``
or an ``__init__.py`` in ``tests``, a helper the conftest imports, the test
data (``tests/test_data``), the pytest configuration (``pyproject.toml``, a
root ``conftest.py``, ``pytest.ini``, ``tox.ini``, ``setup.cfg``) or a
``requirements*.txt`` file changed, or when the changed files cannot be
listed or the selection fails (e.g. unknown base). Other files outside
``cellacdc`` and ``tests`` (docs, workflows, README) relate to no test.
"""
import argparse
import ast
import re
import subprocess
import sys
import warnings
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL = 'tests'
PACKAGES = ('cellacdc', 'tests')
DEFAULT_BASE = 'origin/main'
DOTTED_NAME = re.compile(r'\b(?:cellacdc|tests)(?:\.[A-Za-z_]\w*)+')
FROM_IMPORT = re.compile(
    r'^[ \t]*from[ \t]+((?:cellacdc|tests)(?:\.\w+)*)[ \t]+import[ \t]+'
    r'(\([^)]*\)|[^\n#]*)',
    re.MULTILINE,
)
# The tests of the importers of a changed module are related only when at
# most this many modules import it: the mixins of the GUI windows and the
# measurements viewer modules have up to 5 importers, shared modules such as
# myutils, widgets, apps, load and workers from 39 to 139
MAX_IMPORTERS = 5
# Generated Qt resource modules (~340k lines): no imports to scan
SKIPPED_MODULES = re.compile(r'qrc_resources\w*\.py$')
# Root files that configure pytest or the dependencies of every test
ROOT_FILES_OF_ALL_TESTS = {
    'pyproject.toml', 'conftest.py', 'pytest.ini', 'tox.ini', 'setup.cfg',
}
# Packages whose plugins Cell-ACDC imports by folder name
PLUGIN_PACKAGES = {
    'models', 'trackers', 'promptable_models', 'measurement_processing',
}
# Folder or file stems shorter than this match too many unrelated words
MIN_NAME_LENGTH = 5
# Folder names too generic to relate a data file to a test
GENERIC_FOLDERS = {'cellacdc', 'source', 'docs', 'images', 'resources'}


def is_test_file(path):
    """Whether `path` (relative, posix) is a test module pytest collects."""
    path = PurePosixPath(path)
    return (
        path.parts[0] == 'tests'
        and path.suffix == '.py'
        and (path.name.startswith('test_') or path.name.endswith('_test.py'))
    )


def module_name(path):
    """Dotted module name of a ``.py`` file (``cellacdc/a/b.py`` -> ``cellacdc.a.b``)."""
    parts = list(PurePosixPath(path).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def resolve_module(dotted, root):
    """Longest prefix of `dotted` that is a module or package under `root`.

    ``cellacdc.gui.guiWin`` -> ``cellacdc.gui``. Returns None when not even
    the top-level package exists.
    """
    parts = dotted.split('.')
    for end in range(len(parts), 0, -1):
        path = root.joinpath(*parts[:end])
        if path.with_suffix('.py').is_file() or path.is_dir():
            return '.'.join(parts[:end])
    return None


def from_imports(text, path):
    """Yield ``(module, names)`` of the ``from ... import`` statements.

    Relative imports are resolved with the module `path`. A file that does
    not parse (e.g. being edited) yields only its absolute imports.
    """
    try:
        with warnings.catch_warnings():
            # e.g. invalid escape sequences in the scanned modules
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        for module, names in FROM_IMPORT.findall(text):
            yield module, re.findall(r'(?:^|,)\s*(\w+)', names.strip('()'))
        return
    package = module_name(path).split('.')
    if PurePosixPath(path).name != '__init__.py':
        package = package[:-1]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ''
        if node.level:
            base = package[:len(package) - node.level + 1]
            module = '.'.join(base + ([module] if module else []))
        yield module, [alias.name for alias in node.names]


def referenced_modules(path, root):
    """Modules of ``cellacdc`` and ``tests`` that the Python file names."""
    text = Path(root, path).read_text(encoding='utf-8', errors='replace')
    names = set(DOTTED_NAME.findall(text))
    for module, imported in from_imports(text, path):
        if module.split('.')[0] not in PACKAGES:
            continue
        names.add(module)
        # Submodules (from cellacdc import gui) or attributes (resolved to
        # the module that defines them)
        names.update(f'{module}.{name}' for name in imported)
    resolved = {resolve_module(name, root) for name in names}
    resolved.discard(None)
    return resolved


def is_related_module(reference, changed):
    """Whether a file naming `reference` depends on the module `changed`."""
    if reference == changed or reference.startswith(changed + '.'):
        return True
    # A package containing the changed module, e.g. a plugin folder
    return reference not in PACKAGES and changed.startswith(reference + '.')


def mentions(text, name):
    """Whether `text` contains `name` as a whole word (``-`` counts as a letter)."""
    return re.search(rf'(?<![\w-]){re.escape(name)}(?![\w-])', text) is not None


class Selection:
    """The test files related to a list of changed files."""

    def __init__(self, root=REPO_ROOT):
        self.root = Path(root)
        self.test_files = sorted(
            path.relative_to(self.root).as_posix()
            for path in (self.root / 'tests').rglob('*.py')
            if is_test_file(path.relative_to(self.root).as_posix())
        )
        self._references = {}
        self._texts = {}
        self._package_references = None

    def references(self, path):
        if path not in self._references:
            self._references[path] = referenced_modules(path, self.root)
        return self._references[path]

    def text(self, path):
        if path not in self._texts:
            self._texts[path] = (self.root / path).read_text(
                encoding='utf-8', errors='replace'
            )
        return self._texts[path]

    def tests_referencing(self, module):
        return {
            path for path in self.test_files
            if any(is_related_module(ref, module) for ref in self.references(path))
        }

    def importers(self, module):
        """Modules of ``cellacdc`` that import or name `module` (or a submodule)."""
        if self._package_references is None:
            package = self.root / 'cellacdc'
            self._package_references = {
                module_name(path.relative_to(self.root).as_posix()): (
                    self.references(path.relative_to(self.root).as_posix())
                )
                for path in package.rglob('*.py')
                if not SKIPPED_MODULES.search(path.name)
            }
        # The top-level package imports most modules and every test
        # imports it: not an importer
        return {
            name for name, references in self._package_references.items()
            if name != module and name not in PACKAGES and any(
                ref == module or ref.startswith(module + '.')
                for ref in references
            )
        }

    def tests_mentioning(self, names):
        return {
            path for path in self.test_files
            if any(mentions(self.text(path), name) for name in names)
        }

    def related(self, changed_files):
        """Return ``(tests, reasons)``; `tests` is ``ALL`` or a sorted list.

        `reasons` maps each changed file to the tests it selected, or to the
        reason why everything is related.
        """
        selected = set()
        reasons = {}
        for changed in changed_files:
            path = PurePosixPath(changed)
            tests = self._related_to_file(path)
            if tests == ALL:
                reasons[changed] = 'the whole suite'
                return ALL, reasons
            reasons[changed] = sorted(tests)
            selected.update(tests)
        return sorted(selected), reasons

    def _related_to_file(self, path):
        top = path.parts[0]
        if len(path.parts) == 1:
            if path.name in ROOT_FILES_OF_ALL_TESTS or (
                path.name.startswith('requirements') and path.suffix == '.txt'
            ):
                return ALL
            return set()
        if top not in PACKAGES:
            return set()
        if top == 'tests' and (
            path.name in ('__init__.py', 'conftest.py')
            or path.parts[1] == 'test_data'
        ):
            return ALL
        if path.suffix != '.py':
            return self._related_to_data_file(path)
        module = module_name(path)
        if module == 'cellacdc':
            return ALL
        if top == 'tests' and any(
            is_related_module(ref, module)
            for ref in referenced_modules('tests/conftest.py', self.root)
        ):
            return ALL
        tests = self.tests_referencing(module)
        if top == 'cellacdc':
            importers = self.importers(module)
            if len(importers) <= MAX_IMPORTERS:
                for importer in importers:
                    tests |= self.tests_referencing(importer)
            tests |= self._tests_naming_plugin(path)
        if is_test_file(path) and (self.root / path).is_file():
            tests.add(path.as_posix())
        return tests

    def _tests_naming_plugin(self, path):
        """Tests that name the plugin folder of `path` in a string literal.

        Plugins (``cellacdc/models/<name>/...``, ``trackers``,
        ``promptable_models``, ``measurement_processing``) are imported by
        their folder name, e.g. ``LocalBackend(model_name="thresholding")``.
        """
        if len(path.parts) < 4 or path.parts[1] not in PLUGIN_PACKAGES:
            return set()
        name = re.escape(path.parts[2])
        literal = re.compile(rf'''["']{name}["']''')
        return {
            test for test in self.test_files
            if literal.search(self.text(test))
        }

    def _related_to_data_file(self, path):
        names = {path.name}
        if len(path.stem) >= MIN_NAME_LENGTH:
            names.add(path.stem)
        folder = path.parent.name
        if len(folder) >= MIN_NAME_LENGTH and folder not in GENERIC_FOLDERS:
            names.add(folder)
        return self.tests_mentioning(names)


def changed_files(base, root=REPO_ROOT):
    """Files changed since the merge base of `base` and ``HEAD``.

    Includes uncommitted and untracked files. Returns None when git fails
    (e.g. `base` is unknown or the history is too shallow).
    """
    def git(*args):
        # git writes paths as UTF-8 whatever the locale (cp1252 on Windows)
        return subprocess.run(
            ['git', *args], cwd=root, capture_output=True, check=True,
            encoding='utf-8', errors='surrogateescape',
        ).stdout

    try:
        merge_base = git('merge-base', base, 'HEAD').strip()
        # -z: names with spaces or non-ASCII characters are not quoted
        files = git('diff', '--name-only', '-z', '--no-renames', merge_base)
        files += git('ls-files', '-z', '--others', '--exclude-standard')
    except (OSError, subprocess.CalledProcessError) as err:
        stderr = getattr(err, 'stderr', '') or str(err)
        print(f'Cannot list the changed files: {stderr.strip()}', file=sys.stderr)
        return None
    return sorted(set(files.split('\0')) - {''})


def select(base, root=REPO_ROOT):
    """The related tests (``ALL`` or a list), explained on stderr.

    Fails safe: any error selects the whole suite.
    """
    try:
        files = changed_files(base, root)
        if files is None:
            print('Selecting the whole suite.', file=sys.stderr)
            return ALL
        tests, reasons = Selection(root).related(files)
    except Exception as err:
        print(
            f'Selecting the whole suite after an error: {err!r}',
            file=sys.stderr,
        )
        return ALL
    for changed, reason in reasons.items():
        if isinstance(reason, str):
            print(f'{changed}: {reason}', file=sys.stderr)
        else:
            print(f'{changed}: {len(reason)} test file(s)', file=sys.stderr)
    return tests


def main(argv=None, root=REPO_ROOT):
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            'Print the test files related to the files changed since the '
            'merge base of BASE and HEAD ("tests" = the whole suite), or run '
            'pytest on them with --run.'
        )
    )
    parser.add_argument(
        '--base', default=DEFAULT_BASE,
        help=f'git revision to compare with (default: {DEFAULT_BASE})',
    )
    parser.add_argument(
        '--run', action='store_true',
        help=(
            'run "python -m pytest <related files> <other arguments>" '
            '(nothing when no file is related) instead of printing the files'
        ),
    )
    args, pytest_args = parser.parse_known_args(argv)
    if pytest_args and not args.run:
        parser.error(f'unrecognized arguments: {" ".join(pytest_args)}')
    tests = select(args.base, root)
    selected = [ALL] if tests == ALL else tests
    if not args.run:
        print('\n'.join(selected))
        return 0
    if not selected:
        # A bare `pytest` would run the whole suite (testpaths)
        print('No test file is related to the changed files.', file=sys.stderr)
        return 0
    return subprocess.call(
        [sys.executable, '-m', 'pytest', *selected, *pytest_args], cwd=root
    )


if __name__ == '__main__':
    sys.exit(main())
