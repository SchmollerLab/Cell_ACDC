"""C3 (synthesis R6): help/welcome.py imported the gui module at module
level only to create guiWin in two methods. The gui window imports welcome
(window_setup.py from C10 on, at import time through the window's bases),
so welcome imports the window inside those methods."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

REL = 'cellacdc/help/welcome.py'


def main(repo):
    replace(repo, REL,
            "from .. import gui, dataStruct, myutils, cite_url, html_utils, urls, widgets\n",
            "from .. import dataStruct, myutils, cite_url, html_utils, urls, widgets\n",
            items=[REL + '::import:from .. import gui'])
    replace(repo, REL,
            "            self.guiWin = gui.guiWin(self.app)\n",
            "            from ..gui.main_window import window as gui_window\n"
            "            self.guiWin = gui_window.guiWin(self.app)\n",
            count=2,
            items=[REL + '::welcomeWin.openGUIsingleImage',
                   REL + '::welcomeWin.openGUIfolder'])


run(main)
