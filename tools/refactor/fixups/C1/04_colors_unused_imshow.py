"""C1: colors.py imports plot.imshow at module level but never uses it.
The import closes the cycle colors -> plot -> core -> load -> prompts ->
html_utils -> colors (html_utils needs colors.rgb_uint_to_html_hex at
import time), which breaks `import cellacdc` once myutils is a lazy facade
(C8) and already breaks importing cellacdc.colors first without Qt."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402


def main(repo):
    replace(repo, 'cellacdc/colors.py',
            "    import seaborn as sns\n"
            "    from cellacdc.plot import imshow\n",
            "    import seaborn as sns\n",
            items=['cellacdc/colors.py::stmt:if GUI_INSTALLED:',
                   'cellacdc/colors.py::import:from cellacdc.plot import imshow'])


run(main)
