"""C1: utils/convert.py and utils/rename.py append the cellacdc folder to
sys.path (left over from running them as scripts). With the new packages
(gui, shared, launcher, ...) that would make them importable as top-level
packages; the modules import everything relatively, so the line is dropped.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402


def main(repo):
    for relpath in ('cellacdc/utils/convert.py', 'cellacdc/utils/rename.py'):
        replace(repo, relpath, "sys.path.append(cellacdc_path)\n", "",
                items=[relpath + '::stmt:sys.path.append(cellacdc_path)'])


run(main)
