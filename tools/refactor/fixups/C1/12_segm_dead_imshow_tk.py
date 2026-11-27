"""C1 (synthesis R5): segm.py's segmWin.debugSegmWorker only called
apps.imshow_tk, which does not exist (AttributeError), and nothing connects
it (the main GUI has its own debugSegmWorker). The dead method is removed,
so no code refers to the missing apps.imshow_tk when apps.py is split.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

SEGM = 'cellacdc/segm.py'


def main(repo):
    replace(repo, SEGM,
            "        raise error\n"
            "\n"
            "    def debugSegmWorker(self, lab):\n"
            "        apps.imshow_tk(lab)\n"
            "\n"
            "    def segmWorkerProgress(self, text):\n",
            "        raise error\n"
            "\n"
            "    def segmWorkerProgress(self, text):\n",
            items=[SEGM + '::segmWin.debugSegmWorker'])


run(main)
