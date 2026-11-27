"""C12: the dev script _view_all_buttons.py builds every button class that
load.get_all_buttons_names lists (the classes of
shared/widgets/push_buttons.py since C5) with getattr(widgets, name), a
value use of the old split module that the rewrite keeps and reports.
Read them from push_buttons (imported at the same place, before
_run._setup_app as before)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

VIEW = 'cellacdc/_view_all_buttons.py'


def main(repo):
    replace(repo, VIEW,
            "from cellacdc import widgets, _run\n",
            "from cellacdc import _run\n"
            "from cellacdc.shared.widgets import push_buttons as "
            "widgets_push_buttons\n",
            items=[VIEW + '::<imports>'])
    replace(repo, VIEW,
            "        button_class = getattr(widgets, button_name)\n",
            "        button_class = getattr(widgets_push_buttons, button_name)\n",
            items=[VIEW + '::stmt:for i in range(nrows):'])


run(main)
