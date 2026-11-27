"""C12: five parametrized tests of test_keep_ids.py build their widget with
getattr(widgets, cls) from a class name (the rewrite keeps such value uses
of the old split module and reports them). Look the classes up in a table
of the new homes instead; the parameter ids (class names) stay."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

KEEP = 'tests/test_keep_ids.py'


def main(repo):
    replace(repo, KEEP,
            "from cellacdc import widgets\n"
            "from cellacdc.gui.main_window import window as gui_window\n",
            "from cellacdc.gui.main_window import window as gui_window\n",
            items=[KEEP + '::<imports>'])
    replace(repo, KEEP,
            "INSTRUCTIONS = ' (Separate IDs by comma)'\n",
            "INSTRUCTIONS = ' (Separate IDs by comma)'\n"
            "# The widget classes of the tests parametrized by class name\n"
            "WIDGET_CLASSES = {\n"
            "    'KeepIDsLineEdit': gui_widgets_edit_toolbars.KeepIDsLineEdit,\n"
            "    'IDsLineEdit': gui_widgets_edit_toolbars.IDsLineEdit,\n"
            "    'SpinBox': widgets_numeric_inputs.SpinBox,\n"
            "    'DoubleSpinBox': widgets_numeric_inputs.DoubleSpinBox,\n"
            "}\n",
            items=[KEEP + '::WIDGET_CLASSES'])
    replace(repo, KEEP,
            "getattr(widgets, cls)(", "WIDGET_CLASSES[cls](", count=5,
            items=[KEEP + '::test_ids_line_edit_accepts_enter',
                   KEEP + '::test_spin_box_with_disable_key_press_enter',
                   KEEP + '::test_spin_box_keys_passed_on',
                   KEEP + '::test_spin_box_typed_value_when_focus_returns'])


run(main)
