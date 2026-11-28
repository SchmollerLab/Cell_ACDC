"""The main GUI: the window for annotating and correcting segmentation,
tracking and cell cycle data (class guiWin).

- main_window/: guiWin and the mixins that only guiWin composes.
  window.py holds the class body, every other module one or more
  <Topic>GuiElements mixins of one topic.
- mixins/: mixins that the measurements viewer window ALSO composes:
  changes there affect both windows.
- dialogs/, widgets/, workers/: dialogs, widgets and workers used only
  by guiWin.
- helpers.py: non-UI helpers used only by guiWin.

mixins/ and the measurements viewer never import main_window/, dialogs/,
widgets/, workers/ or helpers.py.

This __init__ imports qtpy, pyqtgraph and matplotlib (the GUI
requirements), so importing the package still fails without them
(spotMAX relies on it). It imports nothing else eagerly: the names that
the former single-module main GUI defined at this import path (guiWin,
its constants, ...) are served lazily with a deprecation warning, for
external code only (see _compat.py and CODEMAP.md).
"""

from qtpy import QtCore  # noqa: F401
import pyqtgraph  # noqa: F401
import matplotlib  # noqa: F401

from cellacdc import _compat as _acdc_compat

__getattr__, __dir__ = _acdc_compat.package_compat(__name__)
