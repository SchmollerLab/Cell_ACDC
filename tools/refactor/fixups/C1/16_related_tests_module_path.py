"""C1 (path-independent tests, synthesis 6E): test_related_tests.py's
test_selection_of_this_repository hard-codes 'cellacdc/phase_translator.py',
which becomes an alias shim in C3 (module at
measurements_viewer/phase_translator.py): once the tests import the new
path (C12), nothing relates the shim to tests/test_phase_translator.py and
the slow test fails. The test now takes the module from the imports of
tests/test_phase_translator.py (Selection.references, static), so it
checks the file that test file depends on at every commit.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

RELATED = 'tests/test_related_tests.py'


def main(repo):
    replace(repo, RELATED,
            "    selection = Selection()\n"
            "    tests, _ = selection.related(['cellacdc/phase_translator.py'])\n"
            "    assert 'tests/test_phase_translator.py' in tests\n",
            "    selection = Selection()\n"
            "    # The phase translator module, from the path its tests import it\n"
            "    module, = [\n"
            "        name for name in selection.references(\n"
            "            'tests/test_phase_translator.py'\n"
            "        )\n"
            "        if name.split('.')[-1] == 'phase_translator'\n"
            "    ]\n"
            "    tests, _ = selection.related([module.replace('.', '/') + '.py'])\n"
            "    assert 'tests/test_phase_translator.py' in tests\n",
            items=[RELATED + '::test_selection_of_this_repository'])


run(main)
