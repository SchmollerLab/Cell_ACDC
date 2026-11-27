"""C12: the rewrite qualified both dialog classes of the loop in
test_viewer_deletes_identifier_and_condition_dialogs with their new module
aliases, a line of 142 characters (the rewriter never rewraps lines and
reports them as long_lines). Wrap the tuple."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

TP = 'tests/test_timepoint_identifiers.py'


def main(repo):
    replace(repo, TP,
            "    for cls in (viewer_dialogs_predictor_editors."
            "TimepointIdentifierDialog, viewer_dialogs_conditional_formatting."
            "_FormattingConditionDialog):\n",
            "    for cls in (\n"
            "        viewer_dialogs_predictor_editors.TimepointIdentifierDialog,\n"
            "        viewer_dialogs_conditional_formatting"
            "._FormattingConditionDialog,\n"
            "    ):\n",
            items=[TP + '::test_viewer_deletes_identifier_and_condition_dialogs'])


run(main)
