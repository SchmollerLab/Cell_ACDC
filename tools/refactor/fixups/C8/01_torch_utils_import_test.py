"""C8: tests/test_torch_version.py::test_torch_utils_import_does_not_load_gui_or_torch
blocks the exact module name cellacdc.myutils. From C8 the torch helpers
live in myutils/torch_utils.py (cellacdc/_torch_utils.py is an alias shim),
so importing them imports the lazy myutils package itself; the test keeps
blocking every other (heavier) myutils submodule instead (moves shard
notes: retarget in C8, not C13)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

REL = 'tests/test_torch_version.py'


def main(repo):
    replace(repo, REL,
            "        if fullname.split('.')[0] in {'torch', 'torchvision', 'qtpy'} or fullname in {\n"
            "            'cellacdc.apps', 'cellacdc.widgets', 'cellacdc.myutils',\n"
            "            'cellacdc.service.environment',\n"
            "        }:\n",
            "        if fullname.split('.')[0] in {'torch', 'torchvision', 'qtpy'} or fullname in {\n"
            "            'cellacdc.apps', 'cellacdc.widgets',\n"
            "            'cellacdc.service.environment',\n"
            "        } or (\n"
            "            fullname.startswith('cellacdc.myutils.')\n"
            "            and fullname != 'cellacdc.myutils.torch_utils'\n"
            "        ):\n",
            items=[REL + '::test_torch_utils_import_does_not_load_gui_or_torch'])


run(main)
