"""The data prep module: prepares Positions before segmentation (z-slice
and frame selection, cropping, alignment, background ROIs).

- main_window.py: the window dataPrepWin, and the entry points that open
  it from other modules for z-plane or ROI selection.
- dialogs.py, workers.py: dialogs and workers used only by the data prep
  module.

This __init__ imports nothing.
"""
