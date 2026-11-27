"""C1: unused module-level imports of cellacdc modules (synthesis, C1 row)
that would otherwise become import edges to the split modules:

- `gui` in utils/base.py and utils/acdcToSymDiv.py (decision in the plan:
  utilities run from the launcher, which imports the GUI anyway, so its
  import-time settings such as np.seterr are still in place);
- `apps` in trackers/trackpy/trackpy_tracker.py;
- `myutils` in models/DeepSea/acdcSegment.py, models/BABY/acdcSegment.py,
  trackers/DeepSea/DeepSea_tracker.py and html_utils.py (html_utils is
  imported early by most modules; it never used myutils).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402


def main(repo):
    rel = 'cellacdc/utils/base.py'
    replace(repo, rel,
            "from .. import (\n"
            "    widgets, apps, workers, html_utils, myutils,\n"
            "    gui, load, printl, exception_handler\n"
            ")\n",
            "from .. import (\n"
            "    widgets, apps, workers, html_utils, myutils,\n"
            "    load, printl, exception_handler\n"
            ")\n",
            items=[rel + '::import:from .. import ('])
    rel = 'cellacdc/utils/acdcToSymDiv.py'
    replace(repo, rel,
            "from .. import (\n"
            "    widgets, apps, workers, html_utils, myutils,\n"
            "    gui, load, printl\n"
            ")\n",
            "from .. import (\n"
            "    widgets, apps, workers, html_utils, myutils,\n"
            "    load, printl\n"
            ")\n",
            items=[rel + '::import:from .. import ('])
    rel = 'cellacdc/trackers/trackpy/trackpy_tracker.py'
    replace(repo, rel,
            "from cellacdc import apps, printl\n",
            "from cellacdc import printl\n",
            items=[rel + '::import:from cellacdc import apps, printl'])
    for rel in ('cellacdc/models/DeepSea/acdcSegment.py',
                'cellacdc/trackers/DeepSea/DeepSea_tracker.py'):
        replace(repo, rel,
                "from cellacdc import myutils, printl\n",
                "from cellacdc import printl\n",
                items=[rel + '::import:from cellacdc import myutils, printl'])
    rel = 'cellacdc/models/BABY/acdcSegment.py'
    replace(repo, rel,
            "from cellacdc import myutils\n"
            "from cellacdc.trackers import BABY\n",
            "from cellacdc.trackers import BABY\n",
            items=[rel + '::import:from cellacdc import myutils'])
    rel = 'cellacdc/html_utils.py'
    replace(repo, rel,
            "from . import GUI_INSTALLED, myutils\n",
            "from . import GUI_INSTALLED\n",
            items=[rel + '::import:from . import GUI_INSTALLED, myutils'])


run(main)
