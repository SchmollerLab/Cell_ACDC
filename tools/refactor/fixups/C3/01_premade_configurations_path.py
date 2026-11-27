"""C3: the premade configurations folder moved to
measurements_viewer/premade_configurations (module_moves of the moves
shard); C1 made the path relative to cellacdc_path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402


def main(repo):
    replace(repo, 'cellacdc/measurements_viewer/main_window/window.py',
            "    cellacdc_path, 'measurements_viewer_premade_configurations'\n",
            "    cellacdc_path, 'measurements_viewer', 'premade_configurations'\n",
            items=['cellacdc/measurements_viewer/main_window/window.py'
                   '::PREMADE_CONFIGURATIONS_PATH'])


run(main)
