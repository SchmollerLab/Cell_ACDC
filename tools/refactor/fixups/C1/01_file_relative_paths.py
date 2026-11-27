"""C1: paths built from __file__ in modules that move later -> cellacdc_path.

Behaviour-preserving on the old layout (cellacdc_path is the folder of
cellacdc/__init__.py, i.e. the folder of these modules today); after the
moves (C3 dataStruct.py, C6 apps.py, C8 myutils.py) __file__ would point
one folder deeper.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, ensure_line_after, run  # noqa: E402


def main(repo):
    # dataStruct.py: user manual / Fiji macros folders of the repository
    replace(repo, 'cellacdc/dataStruct.py',
            "main_path = pathlib.Path(__file__).resolve().parents[1]",
            "main_path = pathlib.Path(cellacdc_path).resolve().parent",
            count=2,
            items=['cellacdc/dataStruct.py::createDataStructWin.on_linkActivated'])
    ensure_line_after(repo, 'cellacdc/dataStruct.py',
                      'from . import load, settings_csv_path',
                      'from . import cellacdc_path')
    # apps.py: default tracker model folder / model path of two dialogs
    replace(repo, 'cellacdc/apps.py',
            "this_path = os.path.dirname(os.path.abspath(__file__))",
            "this_path = cellacdc_path",
            count=2,
            items=['cellacdc/apps.py::BayesianTrackerParamsWin.__init__',
                   'cellacdc/apps.py::DeltaTrackerParamsWin.__init__'])
    ensure_line_after(repo, 'cellacdc/apps.py',
                      'from . import user_profile_path, promptable_models_path, models_path',
                      'from . import cellacdc_path')
    # myutils.py: setuptools_scm root (the repository folder)
    replace(repo, 'cellacdc/myutils.py',
            "get_version(root='..', relative_to=__file__)",
            "get_version(\n"
            "                root='..',\n"
            "                relative_to=os.path.join(cellacdc_path, '__init__.py')\n"
            "            )",
            items=['cellacdc/myutils.py::read_version'])
    # measurements_viewer.py: folder of the premade configurations
    replace(repo, 'cellacdc/measurements_viewer.py',
            "PREMADE_CONFIGURATIONS_PATH = os.path.join(\n"
            "    os.path.dirname(os.path.abspath(__file__)),\n"
            "    'measurements_viewer_premade_configurations'\n"
            ")",
            "PREMADE_CONFIGURATIONS_PATH = os.path.join(\n"
            "    cellacdc_path, 'measurements_viewer_premade_configurations'\n"
            ")",
            items=['cellacdc/measurements_viewer.py::PREMADE_CONFIGURATIONS_PATH'])
    ensure_line_after(repo, 'cellacdc/measurements_viewer.py',
                      'from . import urls', 'from . import cellacdc_path')


run(main)
