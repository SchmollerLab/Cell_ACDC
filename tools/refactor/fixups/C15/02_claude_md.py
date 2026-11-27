"""C15: CLAUDE.md for the new package layout.

fixups/C15/files/CLAUDE.base.md is CLAUDE.md at the base commit and
fixups/C15/files/CLAUDE.md the edited version (every reference to a moved
or split module, the pointer to CODEMAP.md, the layout, window, import and
patching rules). The edit is applied as a three-way merge (git merge-file)
of the repository's CLAUDE.md, so that a rebuild on a newer main keeps
main's own CLAUDE.md changes: an unchanged CLAUDE.md is replaced, a changed
one is merged (reported as drift), a conflict stops the pipeline (then
update both files under fixups/C15/files/ from the new main).
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _fixuplib import FixupError, log_change, run  # noqa: E402

FILES = os.path.join(HERE, 'files')
RELPATH = 'CLAUDE.md'


def _read(path):
    with open(path, 'rb') as file:
        data = file.read()
    eol = '\r\n' if b'\r\n' in data else '\n'
    return data.decode('utf-8').replace('\r\n', '\n'), eol


def _write(path, text, eol):
    with open(path, 'wb') as file:
        file.write(text.replace('\n', eol).encode('utf-8'))


def merge(repo, relpath, base_file, new_file):
    """Apply the change base_file -> new_file to repo/relpath."""
    path = os.path.join(repo, relpath)
    current, eol = _read(path)
    base, _ = _read(base_file)
    new, _ = _read(new_file)
    if current == new:
        print(f'  {relpath}: already done')
        return False
    if current == base:
        _write(path, new, eol)
        print(f'  {relpath}: replaced (unchanged since the base)')
        log_change(relpath, [relpath], 'docs for the new package layout')
        return True
    print(f'  {relpath}: DRIFT, changed since the base commit: merging')
    with tempfile.TemporaryDirectory() as tmp:
        names = {}
        for key, text in (('current', current), ('base', base),
                          ('new', new)):
            names[key] = os.path.join(tmp, key)
            with open(names[key], 'w', encoding='utf-8',
                      newline='\n') as file:
                file.write(text)
        result = subprocess.run(
            ['git', 'merge-file', '-p', '-L', 'current', '-L', 'base',
             '-L', 'C15', names['current'], names['base'], names['new']],
            capture_output=True)
    if result.returncode != 0:
        raise FixupError(
            f'{relpath}: {result.returncode} conflict(s) merging the C15 '
            f'edit into the changed {relpath}: update '
            f'fixups/C15/files/{os.path.basename(base_file)} (the new base) '
            f'and {os.path.basename(new_file)}')
    _write(path, result.stdout.decode('utf-8'), eol)
    print(f'  {relpath}: merged')
    log_change(relpath, [relpath],
               'docs for the new package layout (merged with drift)')
    return True


def main(repo):
    merge(repo, RELPATH, os.path.join(FILES, 'CLAUDE.base.md'),
          os.path.join(FILES, 'CLAUDE.md'))


if __name__ == '__main__':
    run(main)
