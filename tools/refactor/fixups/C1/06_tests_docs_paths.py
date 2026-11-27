"""C1 (path-independent tests): three tests build the path of the Sphinx
docs from the folder of a module that moves into a subpackage in C3
(_main.py -> launcher/, measurements_viewer.py -> measurements_viewer/);
they use the package folder (cellacdc_path) instead."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import ensure_line_after, replace, run  # noqa: E402


def main(repo):
    rel = 'tests/test_launcher_layout.py'
    replace(repo, rel,
            "        os.path.dirname(_main.__file__),\n",
            "        cellacdc_path,\n",
            items=[rel + '::test_getting_started_lists_the_module_buttons_in_launcher_order'])
    ensure_line_after(repo, rel,
                      'from cellacdc import _main, dataStruct, measurements_viewer, myutils',
                      'from cellacdc import cellacdc_path')
    rel = 'tests/test_measurement_viewer.py'
    replace(repo, rel,
            "        os.path.dirname(measurements_viewer.__file__),\n",
            "        cellacdc_path,\n",
            items=[rel + '::test_user_guide_limits_alt_menu_navigation_to_windows_and_linux'])
    ensure_line_after(repo, rel,
                      'from cellacdc import measurements_viewer_utils',
                      'from cellacdc import cellacdc_path')
    rel = 'tests/test_measurement_viewer_review_feedback.py'
    replace(repo, rel,
            "    __import__('pathlib').Path(measurements_viewer.__file__).parent\n",
            "    __import__('pathlib').Path(cellacdc_path)\n",
            items=[rel + '::DOCS_RST_PATH'])
    ensure_line_after(repo, rel,
                      'from cellacdc import measurements_viewer_utils',
                      'from cellacdc import cellacdc_path')


run(main)
