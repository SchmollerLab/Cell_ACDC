"""C1: plot.py and whitelist.py call myutils.print_call_stack(), which only
exists in debugutils.py (AttributeError today, in debug branches). Call it
there, with a function-level import as the other debugutils calls of
whitelist.py (no new module-level import edge)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402


def main(repo):
    replace(repo, 'cellacdc/plot.py',
            "    if print_call_stack:\n"
            "        myutils.print_call_stack()\n",
            "    if print_call_stack:\n"
            "        from . import debugutils\n"
            "        debugutils.print_call_stack()\n",
            items=['cellacdc/plot.py::imshow'])
    replace(repo, 'cellacdc/whitelist.py',
            "            printl('Propagating IDs...')\n"
            "            myutils.print_call_stack()\n",
            "            printl('Propagating IDs...')\n"
            "            from . import debugutils\n"
            "            debugutils.print_call_stack()\n",
            items=['cellacdc/whitelist.py::Whitelist.propagateIDs'])


run(main)
