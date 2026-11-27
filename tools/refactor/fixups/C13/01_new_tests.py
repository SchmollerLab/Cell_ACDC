"""C13: add the new tests of the package layout (and their helpers).

Copies every file of fixups/C13/files/<repo path> to <repo path>:

- tests/source_scan.py: static view of the cellacdc source (shared by the
  scans below);
- tests/test_compat.py: the compat layer (cellacdc/_compat.py) on a
  synthetic package, and the real tables in a fresh interpreter;
- tests/test_no_old_paths.py: cellacdc/, tests/ and notebooks/ use no old
  import path; files at old paths are shims;
- tests/test_import_paths.py: every from-import of cellacdc resolves,
  relative imports stay inside their package, module alias attributes
  exist;
- tests/test_import_layering.py: layering of shared/library code and the
  program parts, foundation modules, import-first in fresh interpreters;
- tests/window_layout.py, tests/test_gui_window_layout.py,
  tests/test_viewer_window_layout.py: mixin composition of the two windows;
- tests/test_package_init_files.py: folders with modules are packages;
- tests/test_read_version.py, tests/test_plot_imshow.py: read_version after
  the move into myutils/app_info.py, plot.imshow offscreen.

Idempotent: a file that already has the content is left alone. A file
that exists in HEAD with other content stops the pipeline (drift: main
added a file of that name). Line endings follow cellacdc/__init__.py.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import _fixuplib as F  # noqa: E402

FILES_DIR = os.path.join(HERE, 'files')


def tracked_in_head(repo, relpath):
    proc = subprocess.run(
        ['git', '-C', repo, 'cat-file', '-e', f'HEAD:{relpath}'],
        capture_output=True,
    )
    return proc.returncode == 0


def main(repo):
    _, eol = F._read(os.path.join(repo, 'cellacdc', '__init__.py'))
    for folder, _, names in os.walk(FILES_DIR):
        for name in sorted(names):
            if name.endswith('.pyc'):
                continue
            src = os.path.join(folder, name)
            relpath = os.path.relpath(src, FILES_DIR).replace(os.sep, '/')
            with open(src, 'rb') as file:
                text = file.read().decode('utf-8').replace('\r\n', '\n')
            new = text.replace('\n', eol)
            dest = os.path.join(repo, relpath)
            if os.path.exists(dest):
                current, _ = F._read(dest)
                if current.replace('\r\n', '\n') == text:
                    print(f'  {relpath}: already done')
                    continue
                if tracked_in_head(repo, relpath):
                    raise F.FixupError(
                        f'{relpath} exists in HEAD with other content: '
                        'merge it with fixups/C13/files/' + relpath
                    )
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            F._write(dest, new)
            print(f'  {relpath}: written')
            F.log_change(relpath, [relpath], 'new test file of C13',
                         kind='fixup-new-file')


if __name__ == '__main__':
    F.run(main)
