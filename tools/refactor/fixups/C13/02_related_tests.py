"""C13: tests/related_tests.py and tests/test_related_tests.py for the
package layout.

related_tests.py (fixups/C13/replace/tests/related_tests.py):
- compat shims (``__cellacdc_compat__ = True``) and cellacdc/_compat_moves.py
  are never importers;
- importers are counted per program part (PART_PACKAGES): importers inside
  the changed module's part are followed transitively, the importers of
  one other part count as one unit (followed transitively inside it);
- ``from <package> import <submodule>`` relates the test to the package's
  ``__init__`` only, not to every module of the package;
- docstrings and comment lines of cellacdc modules are not read;
- docstring examples and the MAX_IMPORTERS comment for the new layout.

test_related_tests.py: the synthetic repository avoids real old module
names (widgets -> basic, 'cellacdc.gui.guiWin' -> 'cellacdc.gui.Window'),
new cases (shim exclusion, a part module imported by 6 mixins of its part,
a module imported by 6 modules of one other part, package of a submodule
import), the repository selection on the new layout (keeps the
path-independent phase translator check of C1/16).

Both files are replaced whole. Edits of earlier steps of the pipeline
(C1/16, the C12 rewrite) are superseded by the replacements, which
include them. Drift check: the file at the base commit
(ACDC_REFACTOR_BASE) must be the version the replacements were made from
(LF-normalized sha256 of 4c4e498b); a rebuild on a newer main that changed
the file stops here: merge the change into fixups/C13/replace/<path> and
update PREIMAGES.
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import _fixuplib as F  # noqa: E402

REPLACE_DIR = os.path.join(HERE, 'replace')
DEFAULT_BASE = '4c4e498b'
PREIMAGES = {
    'tests/related_tests.py':
        '89068a96fe43a345d479434b0fa498e37bccb56a71966c54629bb5c915e02006',
    'tests/test_related_tests.py':
        '9071760ef12ac27cdeba4648d147be71e35f710d09629506c2375eb621ff5cc7',
}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def base_text(repo, relpath):
    base = os.environ.get('ACDC_REFACTOR_BASE') or DEFAULT_BASE
    proc = subprocess.run(
        ['git', '-C', repo, 'show', f'{base}:{relpath}'],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise F.FixupError(f'git show {base}:{relpath}: '
                           f'{proc.stderr.decode(errors="replace")}')
    return proc.stdout.decode('utf-8').replace('\r\n', '\n')


def main(repo):
    for relpath, preimage in PREIMAGES.items():
        with open(os.path.join(REPLACE_DIR, *relpath.split('/')), 'rb') as f:
            new = f.read().decode('utf-8').replace('\r\n', '\n')
        dest = os.path.join(repo, relpath)
        current, eol = F._read(dest)
        current = current.replace('\r\n', '\n')
        if current == new:
            print(f'  {relpath}: already done')
            continue
        found = sha(base_text(repo, relpath))
        if found != preimage:
            raise F.FixupError(
                f'{relpath} at the base commit differs from the version the '
                f'C13 replacement was made from (sha256 {found}): merge the '
                f'change into tools/refactor/fixups/C13/replace/{relpath} '
                'and update PREIMAGES'
            )
        F._write(dest, new.replace('\n', eol))
        print(f'  {relpath}: replaced')
        F.log_change(relpath, [relpath],
                     'replaced for the package layout (C13)')


if __name__ == '__main__':
    F.run(main)
