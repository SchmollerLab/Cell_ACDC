"""Workers used only by the main GUI window (guiWin and its gui/main_window
mixins), by theme:

- loading_saving.py: loading Positions and frames, undo states,
  autosave.
- segmentation_tracking.py: repeat segmentation, promptable models,
  tracking, post-processing and segmentation edits.
- annotation_checks.py: the cell cycle annotation integrity checker.
- image_processing.py: preprocessing and combining channels, saving
  their images.

The measurements viewer never uses these modules (the viewer-shared
gui/mixins use only SimpleWorker from shared/workers/base.py). The save
worker of the main GUI, saveDataWorker, is in shared/workers/saving.py.

This package re-exports nothing: import its modules. Names of the former
workers.py module reached through the import path of this package are
forwarded by the compat layer with a deprecation warning.
"""


# Old names of cellacdc.workers that code reached through this module
# path before it existed (cellacdc/_compat.py).
from cellacdc import _compat as _acdc_compat  # noqa: E402

__getattr__, __dir__ = _acdc_compat.forward_module(
    __name__, 'cellacdc.workers')
