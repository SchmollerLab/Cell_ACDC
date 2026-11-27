"""C1: dataStruct.py readMetadata imports `cellacdc.bioformat` (no such
module: ImportError with the python-bioformats backend) instead of the
`cellacdc.bioformats` package that the other methods of the worker import
in the same branch. Fixed here so that the import recomputed by the C3 move
names an existing module.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

DATASTRUCT = 'cellacdc/dataStruct.py'


def main(repo):
    replace(repo, DATASTRUCT,
            "            import javabridge\n"
            "            from cellacdc import bioformat\n",
            "            import javabridge\n"
            "            from cellacdc import bioformats\n",
            items=[DATASTRUCT + '::bioFormatsWorker.readMetadata'])


run(main)
