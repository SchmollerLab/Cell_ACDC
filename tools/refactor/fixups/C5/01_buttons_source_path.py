"""C5: load.get_all_buttons_names reads the source of the push buttons
(regex `class X(PushButton)`); all 61 of them moved to
shared/widgets/push_buttons.py (the PushButton family stays together)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402


def main(repo):
    replace(repo, 'cellacdc/load.py',
            "    widgets_filepath = os.path.join(cellacdc_path, 'widgets.py')\n",
            "    widgets_filepath = os.path.join(\n"
            "        cellacdc_path, 'shared', 'widgets', 'push_buttons.py'\n"
            "    )\n",
            items=['cellacdc/load.py::get_all_buttons_names'])


run(main)
