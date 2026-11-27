"""C1: dataReStruct.py imported the launcher module (_main) at module level
only for the string annotation of run(mainWin: '_main.mainWin'). The
import moves under `if TYPE_CHECKING:` (no runtime import edge from the
data structure part to the launcher; the launcher imports dataReStruct).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

DATARESTRUCT = 'cellacdc/dataReStruct.py'


def main(repo):
    replace(repo, DATARESTRUCT,
            "from pathlib import Path\n"
            "\n"
            "import numpy as np\n",
            "from pathlib import Path\n"
            "from typing import TYPE_CHECKING\n"
            "\n"
            "import numpy as np\n",
            items=[DATARESTRUCT + '::import:from typing import TYPE_CHECKING'])
    replace(repo, DATARESTRUCT,
            "from . import _main\n"
            "from . import apps, html_utils, myutils, printl, widgets, workers\n"
            "from . import exception_handler\n",
            "from . import apps, html_utils, myutils, printl, widgets, workers\n"
            "from . import exception_handler\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from . import _main\n",
            items=[DATARESTRUCT + '::stmt:if TYPE_CHECKING:'])


run(main)
