"""Module-level constants of the main GUI window, shared by the
gui/main_window modules: colors (GREEN_HEX, ORANGE_HEX), cursor sizes,
regionprops update thresholds (RP_OPT_*), fonts (_font, font_13px),
scroll bar actions (Slider*) and CCA_CHECKER_ORPHAN_THREADS.
LayoutGuiElements.resizeSlidersArea rebinds _font: always read and write
it through this module (gui_main_constants._font), never with a
from-import. The import-time side effects of the main window (pyqtgraph
row-major, warning filters, np.seterr, Windows AppUserModelID) are in
gui/main_window/window.py, not here.
"""
from qtpy.QtGui import QFont
from cellacdc import QtScoped
from cellacdc import _palettes


# important functions
# def _get_setting_value(
# def get2DRP
# def get_2Dlab(force_z=True
# posData = self.data[self.pos_i]

GREEN_HEX = _palettes.green()
ORANGE_HEX = _palettes.orange()

MIRRORED_CURSOR_PG_SYMBOL = 'd'
TOOL_SIZE_CURSOR = 52
TOTAL_SIZE_CURSOR = 128
CENTER_SIZE_CURSOR = 64
FINAL_SIZE_CURSOR = 32

RP_OPT_NUM_CELLS_MIN = 30 # th for trying to do local updates to regionprops, rp becomes slow for high num of cells
RP_OPT_PERC_CUTOUT_MAX = 0.3 # th for trying to do local updates to regionprops, 
                             # if region which we have to update is too large too 
                             # many cells are probably inside and its not worth
                             # local updating (since we actually need to call RP twice!)

# Hard-coded shortcuts that are reserved for non-customizable actions.

_font = QFont()
_font.setPixelSize(11)

font_13px = QFont()
font_13px.setPixelSize(13)

SliderSingleStepAdd = QtScoped.SliderSingleStepAdd()
SliderSingleStepSub = QtScoped.SliderSingleStepSub()
SliderPageStepAdd = QtScoped.SliderPageStepAdd()
SliderPageStepSub = QtScoped.SliderPageStepSub()
SliderMove = QtScoped.SliderMove()

# Cell cycle annotations integrity checker threads that did not stop in time
# (see `guiWin.stopCcaIntegrityCheckerWorker`). Stored at module level and not
# on the window, because the window can be garbage collected before these
# threads finish and destroying a running QThread aborts the process
CCA_CHECKER_ORPHAN_THREADS = []
