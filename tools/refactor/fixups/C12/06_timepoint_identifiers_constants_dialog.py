"""C12: test_constants_are_disabled asserted that the old apps module has
no CellConstantsDialog (the rewrite keeps `apps` for that hasattr and
reports it). Check the modules that hold the equation dialogs now: the
shared combine-measurements editors, the dialogs of the viewer-shared GUI
mixins and the viewer's predictor editors."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

TP = 'tests/test_timepoint_identifiers.py'


def main(repo):
    replace(repo, TP,
            "from cellacdc import apps, timepoint_predictors\n",
            "from cellacdc import timepoint_predictors\n",
            items=[TP + '::<imports>'])
    replace(repo, TP,
            "    assert not hasattr(apps, 'CellConstantsDialog')\n",
            "    for module in (dialogs_combine_measurements, gui_mixin_dialogs,\n"
            "                   viewer_dialogs_predictor_editors):\n"
            "        assert not hasattr(module, 'CellConstantsDialog')\n",
            items=[TP + '::test_constants_are_disabled'])


run(main)
