"""C17: serve the old module paths from an import hook, delete the shims.

Until C16 every old module path (cellacdc/apps.py, gui_navigation.py,
_main.py, ...) is a small compat shim file. C17 deletes those files and
serves the same paths from the import hook ``OldPathFinder`` of
cellacdc/_compat.py (installed by cellacdc/__init__.py, table
``OLD_MODULES`` in cellacdc/_compat_moves.py), so that the top level of
cellacdc shows only the new layout and git sees the modules moved whole as
renames (also when branches started on the old layout are merged).

The change is stored as a patch of the commit (fixups/C17/import_hook.patch,
everything outside tools/) and applied with ``git apply --index --3way``,
so that a rebuild on a newer main merges it with main's edits of the same
files; a conflict stops the pipeline (then regenerate the patch).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _fixuplib import FixupError, log_change, run  # noqa: E402

PATCH = os.path.join(HERE, 'import_hook.patch')
# Items of cellacdc whose code this commit adds or changes (fingerprints)
ITEMS = {
    'cellacdc/__init__.py': ['cellacdc/__init__.py::stmt:_compat.install()'],
    'cellacdc/_compat.py': [
        'cellacdc/_compat.py::__doc__',
        'cellacdc/_compat.py::alias_module',
        'cellacdc/_compat.py::_OldPathLoader',
        'cellacdc/_compat.py::OldPathFinder',
        'cellacdc/_compat.py::install',
    ],
    'cellacdc/_compat_moves.py': ['cellacdc/_compat_moves.py::OLD_MODULES'],
}


def _git(repo, *args):
    return subprocess.run(
        ['git', *args], cwd=repo, capture_output=True, text=True
    )


def main(repo):
    if not os.path.isfile(os.path.join(repo, 'cellacdc', 'apps.py')) \
            and 'OLD_MODULES' in open(
                os.path.join(repo, 'cellacdc', '_compat_moves.py'),
                encoding='utf-8').read():
        print('  import hook: already applied')
        return
    check = _git(repo, 'apply', '--index', '--3way', '--check', PATCH)
    if check.returncode != 0:
        raise FixupError(
            'fixups/C17/import_hook.patch does not apply:\n'
            + check.stderr.strip()
            + '\nregenerate it from the C17 commit (README, C17 row)')
    result = _git(repo, 'apply', '--index', '--3way', PATCH)
    if result.returncode != 0:
        raise FixupError('git apply failed:\n' + result.stderr.strip())
    if result.stderr.strip():
        print('  ' + result.stderr.strip().replace('\n', '\n  '))
    print('  import hook: applied, shim files deleted')
    for relpath, items in ITEMS.items():
        log_change(relpath, items, 'old module paths served by the '
                   'import hook of _compat.py (shim files deleted)')


if __name__ == '__main__':
    run(main)
