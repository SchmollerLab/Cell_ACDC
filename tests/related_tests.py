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
* it imports or names the changed module
  (``import cellacdc.gui.mixins.navigation``,
  ``from cellacdc.gui.mixins import navigation``,
  ``'cellacdc.shared.widgets.message_box.myMessageBox'`` in a
  ``monkeypatch.setattr``), one of its submodules or a package containing it
  other than ``cellacdc`` itself (``cellacdc.measurement_processing`` for a
  change in one of its plugins); the same holds for the helper modules in
  ``tests`` (``tests.test_measurement_viewer``, ``tests.utils.segmentation``).
  The package in ``from <package> import <submodule>`` only counts for a
  change of the package's ``__init__.py`` (importing
  ``cellacdc.shared.widgets.message_box`` does not relate a test to every
  module of ``shared/widgets/``);
* it imports or names a module of ``cellacdc`` that imports the changed
  module (see "Importers" below);
* for a module of a plugin, which Cell-ACDC imports by its folder name
  (``cellacdc/models/<name>/``, ``trackers``, ``promptable_models``,
  ``measurement_processing``), it contains the folder name as a string
  literal (``LocalBackend(model_name="thresholding")``); the same for the
  plugins among the importers followed (a shared base module such as
  ``promptable_models/micro_sam_base.py`` selects the tests naming
  ``'micro-sam_v1'``);
* for another file in ``cellacdc`` (configuration templates, docs, icons,
  compiled extensions), it contains the file name, its stem or the name of
  its folder.

Importers: the program parts are packages (``PART_PACKAGES``: ``gui``,
``measurements_viewer``, ``launcher``, ... ; CODEMAP.md), whose windows are
split into many mixin modules. The importers of a changed module inside its
own part are followed transitively (a change in
``measurements_viewer/viewer_utils.py``, imported by the viewer's mixin
modules, selects the tests of the viewer window that composes them). The
importers outside its part are followed only when they form at most
``MAX_IMPORTERS`` units, all modules of one part counting as one unit (and
followed transitively inside that part): a change in the shared mixin
``gui/mixins/navigation.py`` selects the tests of the main GUI and of the
measurements viewer, while the importers of shared modules such as
``load``, ``myutils`` or ``shared/widgets/message_box.py`` (dozens) are not
followed, else nearly every change would select every test. Compat shims at
old import paths (``__cellacdc_compat__ = True``) and the generated
``cellacdc/_compat_moves.py`` are never importers, and docstrings and
comment lines of ``cellacdc`` modules are not read (package docstrings name
files with slash paths).

Deeper dependencies are not followed: the fast tier, which runs everything
that is neither slow nor flaky, and the full suite before merging cover them.

The whole suite is related when ``cellacdc/__init__.py``, a ``conftest.py``
or an ``__init__.py`` in ``tests``, a helper the conftest imports, the test
data (``tests/test_data``), the pytest configuration (``pyproject.toml``, a
root ``conftest.py``, ``pytest.ini``, ``tox.ini``, ``setup.cfg``) or a
``requirements*.txt`` file or the CI setup of the test environment
(``.github/actions``, installs the dependencies) changed, or when the changed
files cannot be listed or the selection fails (e.g. unknown base). Other files
outside ``cellacdc`` and ``tests`` (docs, workflows, README) relate to no
test.
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
COMMENT_LINE = re.compile(r'^[ \t]*#.*$', re.MULTILINE)
# The tests of the importers of a changed module outside its program part
# are related only when the importers form at most this many units (one per
# program part, else one per module). Measured on the package layout: the
# shared GUI mixins (gui/mixins/) have 1 unit outside the gui part (the
# measurements viewer), the viewer's own modules none, timepoint_predictors
# 4; shared modules such as load, core, myutils/filesystem.py and
# shared/widgets/message_box.py from 28 to 35 (68 of 408 modules exceed 5)
MAX_IMPORTERS = 5
# Program parts: packages holding one part's window, mixins, dialogs,
# widgets and workers (importers inside a part are followed transitively)
PART_PACKAGES = (
    'gui', 'measurements_viewer', 'launcher', 'data_prep', 'data_structure',
    'segmentation_module', 'utils',
)
# Generated Qt resource modules (~340k lines): no imports to scan
SKIPPED_MODULES = re.compile(r'qrc_resources\w*\.py$')
# Generated table of the old import paths: names every moved module
COMPAT_MODULES = {'cellacdc._compat_moves', 'cellacdc._compat'}
SHIM_MARKER = re.compile(r'^__cellacdc_compat__\s*=\s*True\b', re.MULTILINE)
# Root files that configure pytest or the dependencies of every test
ROOT_FILES_OF_ALL_TESTS = {
    'pyproject.toml', 'conftest.py', 'pytest.ini', 'tox.ini', 'setup.cfg',
}
# Folder of the CI actions that set up the test environment (dependencies)
CI_SETUP_FOLDER = ('.github', 'actions')
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


def part_of(module):
    """The program part package of `module` (``cellacdc.gui``), or None."""
    parts = module.split('.')
    if len(parts) >= 2 and parts[0] == 'cellacdc' and \
            parts[1] in PART_PACKAGES:
        return '.'.join(parts[:2])
    return None


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


def _parse(text):
    try:
        with warnings.catch_warnings():
            # e.g. invalid escape sequences in the scanned modules
            warnings.simplefilter('ignore', SyntaxWarning)
            return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _resolve_from(node, path):
    module = node.module or ''
    if node.level:
        package = module_name(path).split('.')
        if PurePosixPath(path).name != '__init__.py':
            package = package[:-1]
        base = package[:len(package) - node.level + 1]
        module = '.'.join(base + ([module] if module else []))
    return module


def from_imports(text, path, tree=None):
    """Yield ``(module, names, node)`` of the ``from ... import``
    statements (`node` None when the file does not parse).

    Relative imports are resolved with the module `path`. A file that does
    not parse (e.g. being edited) yields only its absolute imports.
    """
    if tree is None:
        tree = _parse(text)
    if tree is None:
        for module, names in FROM_IMPORT.findall(text):
            yield module, re.findall(
                r'(?:^|,)\s*(\w+)', names.strip('()')), None
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield _resolve_from(node, path), [
                alias.name for alias in node.names], node


def _blank(text, spans):
    """`text` with the line spans (1-based, inclusive) replaced by empty
    lines (the line count is kept)."""
    if not spans:
        return text
    lines = text.split('\n')
    for start, end in spans:
        for k in range(start - 1, min(end, len(lines))):
            lines[k] = ''
    return '\n'.join(lines)


def _docstring_spans(tree):
    spans = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            spans.append((node.lineno, node.end_lineno))
    return spans


def references_of(path, root, code_only=False):
    """``(names, package_inits)``: the modules of ``cellacdc`` and ``tests``
    that the Python file names, and the packages whose ``__init__`` it runs
    only as the package in ``from <package> import <submodule>``.

    With `code_only` (modules of ``cellacdc``), docstrings and comment
    lines are not read.
    """
    text = Path(root, path).read_text(encoding='utf-8', errors='replace')
    tree = _parse(text)
    names = set()
    package_inits = set()
    if tree is None:
        names.update(DOTTED_NAME.findall(text))
        for module, imported, _ in from_imports(text, path, tree):
            if module.split('.')[0] not in PACKAGES:
                continue
            names.add(module)
            names.update(f'{module}.{name}' for name in imported)
    else:
        spans = []
        for module, imported, node in from_imports(text, path, tree):
            # The statement is read here, not by DOTTED_NAME (it would
            # take the package of a submodule import for a reference)
            spans.append((node.lineno, node.end_lineno))
            if module.split('.')[0] not in PACKAGES:
                continue
            submodules = 0
            for name in imported:
                full = f'{module}.{name}'
                if resolve_module(full, root) == full:
                    # Submodule (from cellacdc.gui import mixins)
                    names.add(full)
                    submodules += 1
            if submodules == len(imported):
                package_inits.add(module)
            else:
                # Attributes (resolved to the module that defines them)
                names.add(module)
        if code_only:
            spans.extend(_docstring_spans(tree))
        scanned = _blank(text, spans)
        if code_only:
            scanned = COMMENT_LINE.sub('', scanned)
        names.update(DOTTED_NAME.findall(scanned))
    resolved = {resolve_module(name, root) for name in names}
    resolved.discard(None)
    inits = {resolve_module(name, root) for name in package_inits}
    inits.discard(None)
    return resolved, inits - resolved


def referenced_modules(path, root):
    """Modules of ``cellacdc`` and ``tests`` that the Python file names
    (including the packages it imports submodules from)."""
    names, package_inits = references_of(path, root)
    return names | package_inits


def is_related_module(reference, changed):
    """Whether a file naming `reference` depends on the module `changed`."""
    if reference == changed or reference.startswith(changed + '.'):
        return True
    # A package containing the changed module, e.g. a plugin folder
    return reference not in PACKAGES and changed.startswith(reference + '.')


def runs_package_init(package, changed):
    """Whether importing a submodule of `package` runs the module
    `changed` (its ``__init__`` or the ``__init__`` of a parent package)."""
    return package == changed or package.startswith(changed + '.')


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
        self._module_paths = {}

    def reference_kinds(self, path, code_only=False):
        """``(names, package_inits)`` of the file (see ``references_of``)."""
        key = (path, code_only)
        if key not in self._references:
            self._references[key] = references_of(
                path, self.root, code_only=code_only
            )
        return self._references[key]

    def references(self, path, code_only=False):
        """Modules of ``cellacdc`` and ``tests`` that the file names,
        including the packages it imports submodules from."""
        names, package_inits = self.reference_kinds(path, code_only)
        return names | package_inits

    def text(self, path):
        if path not in self._texts:
            self._texts[path] = (self.root / path).read_text(
                encoding='utf-8', errors='replace'
            )
        return self._texts[path]

    def is_referenced(self, path, module):
        names, package_inits = self.reference_kinds(path)
        return any(is_related_module(ref, module) for ref in names) or any(
            runs_package_init(package, module) for package in package_inits
        )

    def tests_referencing(self, module):
        return {
            path for path in self.test_files
            if self.is_referenced(path, module)
        }

    def _package_modules(self):
        """{module: references} of the modules of ``cellacdc`` that can
        import others (compat shims and tables excluded)."""
        if self._package_references is None:
            package = self.root / 'cellacdc'
            self._module_paths = {
                module_name(relative): relative
                for relative in (
                    path.relative_to(self.root).as_posix()
                    for path in package.rglob('*.py')
                    if not SKIPPED_MODULES.search(path.name)
                )
            }
            self._package_references = {}
            for name, relative in self._module_paths.items():
                if name in COMPAT_MODULES or SHIM_MARKER.search(
                        self.text(relative)):
                    # The compat layer names the old and new paths of every
                    # moved module: never an importer
                    continue
                self._package_references[name] = self.references(
                    relative, code_only=True
                )
        return self._package_references

    def importers(self, module):
        """Modules of ``cellacdc`` that import or name `module` (or a submodule)."""
        # The top-level package imports most modules and every test
        # imports it: not an importer
        return {
            name for name, references in self._package_modules().items()
            if name != module and name not in PACKAGES and any(
                ref == module or ref.startswith(module + '.')
                for ref in references
            )
        }

    def part_importers(self, module, part):
        """Modules of the part package `part` that import `module`,
        directly or through other modules of the part."""
        found = set()
        todo = [module]
        while todo:
            current = todo.pop()
            for importer in self.importers(current):
                if part_of(importer) == part and importer not in found \
                        and importer != module:
                    found.add(importer)
                    todo.append(importer)
        return found

    def followed_importers(self, module):
        """The importers of `module` whose tests are related to it: inside
        its program part transitively, outside it only when they form at
        most ``MAX_IMPORTERS`` units (one per part, else one per module)."""
        part = part_of(module)
        followed = self.part_importers(module, part) if part else set()
        units = {}
        for importer in self.importers(module):
            importer_part = part_of(importer)
            if part and importer_part == part:
                continue
            units.setdefault(importer_part or importer, set()).add(importer)
        if len(units) > MAX_IMPORTERS:
            return followed
        for unit, members in units.items():
            followed |= members
            if part_of(unit) == unit:
                for member in members:
                    followed |= self.part_importers(member, unit)
        return followed

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
        if path.parts[:2] == CI_SETUP_FOLDER:
            return ALL
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
            for importer in self.followed_importers(module):
                tests |= self.tests_referencing(importer)
                # e.g. a shared base module of plugins (micro_sam_base)
                tests |= self._tests_naming_plugin(
                    PurePosixPath(self._module_paths[importer])
                )
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
