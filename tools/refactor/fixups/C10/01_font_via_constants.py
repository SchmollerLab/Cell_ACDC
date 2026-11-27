"""C10 (plan rule 7): gui.py's module-level _font moved to
gui/main_window/constants.py. resizeSlidersArea rebound it with
`global _font`; the codemod qualified every read as
gui_main_constants._font, so the writer now writes the module attribute."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, FixupError, run  # noqa: E402

MAIN = 'cellacdc/gui/main_window'
IMPORT = 'from . import constants as gui_main_constants'


def _module_of_resize(repo):
    for name in sorted(os.listdir(os.path.join(repo, MAIN))):
        if not name.endswith('.py'):
            continue
        rel = f'{MAIN}/{name}'
        with open(os.path.join(repo, rel), encoding='utf-8') as file:
            if 'def resizeSlidersArea(' in file.read():
                return rel
    raise FixupError('resizeSlidersArea not found in ' + MAIN)


def main(repo):
    rel = _module_of_resize(repo)
    replace(repo, rel,
            "    def resizeSlidersArea(self, fontSizeFactor=None, heightFactor=None):\n"
            "        global _font\n",
            "    def resizeSlidersArea(self, fontSizeFactor=None, heightFactor=None):\n",
            items=['cellacdc/gui.py::guiWin.resizeSlidersArea'])
    replace(repo, rel,
            "        _font = newFont\n",
            "        gui_main_constants._font = newFont\n",
            items=['cellacdc/gui.py::guiWin.resizeSlidersArea'])
    with open(os.path.join(repo, rel), encoding='utf-8') as file:
        text = file.read()
    if not re.search(r'^from \. import .*\bconstants as gui_main_constants\b',
                     text, re.M):
        raise FixupError(f'{rel} does not import {IMPORT!r}: add it to the '
                         'imports of the codemod output')


run(main)
