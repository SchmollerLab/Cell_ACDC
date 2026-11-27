"""C15: CODEMAP.md, its generator tools/codemap.py and tests/test_codemap.py.

- tools/codemap.py is copied from the main checkout (tools/codemap.py next
  to tools/refactor/; it stays after tools/refactor is removed);
- tests/test_codemap.py from fixups/C15/files/test_codemap.py;
- CODEMAP.md is written from the hand-written fixups/C15/files/CODEMAP.md
  (sections 1-2 and 4-9, empty package tree), then tools/codemap.py fills
  the package tree from the final tree: the descriptions come from
  fixups/C15/files/codemap_themes.json, else from the first sentence of the
  module docstrings. `tools/codemap.py --check` must pass afterwards.

Idempotent: CODEMAP.md is rebuilt from the template every run (the result
depends only on the template, the themes and the tree).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _fixuplib import FixupError, log_change, run, TOOLS_DIR  # noqa: E402

FILES = os.path.join(HERE, 'files')
CODEMAP_TOOL = os.path.join(os.path.dirname(TOOLS_DIR), 'codemap.py')


def _read_bytes(path):
    with open(path, 'rb') as file:
        return file.read()


def _copy(src, repo, relpath, detail):
    """Copy src to repo/relpath (LF line endings, as committed)."""
    if not os.path.isfile(src):
        raise FixupError(f'{src} not found')
    data = _read_bytes(src).replace(b'\r\n', b'\n')
    dest = os.path.join(repo, *relpath.split('/'))
    if os.path.isfile(dest) and \
            _read_bytes(dest).replace(b'\r\n', b'\n') == data:
        print(f'  {relpath}: already up to date')
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as file:
        file.write(data)
    print(f'  {relpath}: written')
    log_change(relpath, [relpath], detail, kind='fixup-new-file')
    return True


def _run_tool(repo, *args):
    cmd = [sys.executable, os.path.join(repo, 'tools', 'codemap.py'),
           '--root', repo, *args]
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise FixupError(f'{" ".join(args) or "codemap.py"} failed:\n'
                         f'{result.stdout}\n{result.stderr}')
    return result.stdout


def main(repo):
    _copy(CODEMAP_TOOL, repo, 'tools/codemap.py',
          'tools/codemap.py: generator of the CODEMAP.md package tree')
    _copy(os.path.join(FILES, 'test_codemap.py'), repo,
          'tests/test_codemap.py',
          'tests/test_codemap.py: CODEMAP.md lists every package and module')

    template = _read_bytes(os.path.join(FILES, 'CODEMAP.md')).replace(
        b'\r\n', b'\n')
    codemap = os.path.join(repo, 'CODEMAP.md')
    before = _read_bytes(codemap) if os.path.isfile(codemap) else None
    with open(codemap, 'wb') as file:
        file.write(template)
    out = _run_tool(repo, '--themes', os.path.join(FILES,
                                                   'codemap_themes.json'))
    if '  describe: ' in out:
        raise FixupError('CODEMAP.md: entries without description (add them '
                         'to fixups/C15/files/codemap_themes.json or give '
                         'the modules a docstring):\n' + out)
    _run_tool(repo, '--check')
    after = _read_bytes(codemap)
    if after != before:
        log_change('CODEMAP.md', ['CODEMAP.md'],
                   'CODEMAP.md written from fixups/C15/files/CODEMAP.md and '
                   'tools/codemap.py', kind='fixup-new-file')
    else:
        print('  CODEMAP.md: already up to date')


if __name__ == '__main__':
    run(main)
