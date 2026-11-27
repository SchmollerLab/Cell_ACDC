"""C6: TestSegmModelInitalDialog moved from apps.py into its only user, the
dev script test_segm_model.py (completeness review). The codemod appends
items at the end of an existing module, but the script instantiates the
dialog at module level (`apps.TestSegmModelInitalDialog()`, bare after the
C12 rewrite), so the class and the imports it brought go right after the
script's import block."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import FixupError, log_change, run, _read, _write  # noqa: E402

REL = 'cellacdc/test_segm_model.py'
ANCHOR = 'import qtpy.compat'
CLASS = 'class TestSegmModelInitalDialog('


def main(repo):
    path = os.path.join(repo, REL)
    text, eol = _read(path)
    lines = text.split(eol)
    try:
        k_class = next(i for i, line in enumerate(lines)
                       if line.startswith(CLASS))
        k_anchor = lines.index(ANCHOR)
    except (StopIteration, ValueError):
        raise FixupError(f'{REL}: class or anchor {ANCHOR!r} not found')
    if k_class < k_anchor + 5:
        print(f'  {REL}: already done')
        return
    # the class block runs to the end of the file (the last item appended)
    block = lines[k_class:]
    while block and not block[-1].strip():
        block.pop()
    rest = lines[:k_class]
    while rest and not rest[-1].strip():
        rest.pop()
    # imports the codemod added for the class (after the last import line
    # of the script, which is in the middle of its code)
    moved_imports = []
    for imp in ('from qtpy.QtWidgets import QVBoxLayout',
                'from cellacdc.shared.dialogs import base as dialogs_base'):
        if imp in rest:
            k = rest.index(imp)
            if k > k_anchor:
                rest.pop(k)
                moved_imports.append(imp)
    k_anchor = rest.index(ANCHOR)
    new = (rest[:k_anchor + 1] + moved_imports + ['', ''] + block
           + ['', ''] + rest[k_anchor + 1:] + [''])
    # collapse the blank lines that followed the anchor
    out = []
    for line in new:
        if not line.strip() and len(out) >= 2 and not out[-1].strip() \
                and not out[-2].strip():
            continue
        out.append(line)
    _write(path, eol.join(out))
    print(f'  {REL}: moved TestSegmModelInitalDialog after the imports')
    log_change(REL, [REL + '::TestSegmModelInitalDialog',
                     'cellacdc/apps.py::TestSegmModelInitalDialog'],
               'class placed after the import block (used at module level)')


run(main)
