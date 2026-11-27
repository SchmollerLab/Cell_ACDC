"""C1: regionprops.py reads skimage.measure._regionprops at import time
(RegionProperties, _cached) but never imports that private submodule:
skimage.measure loads it lazily, so it worked only when another module had
imported it before (at the base: the eager myutils import chain). With the
lazy myutils facade (C8) `import cellacdc.regionprops` fails without Qt;
import the submodule explicitly (it is also why cellacdc.measure fails to
import first without Qt at the base)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import ensure_line_after, run  # noqa: E402


def main(repo):
    ensure_line_after(repo, 'cellacdc/regionprops.py',
                      'import skimage.measure',
                      'import skimage.measure._regionprops')


run(main)
