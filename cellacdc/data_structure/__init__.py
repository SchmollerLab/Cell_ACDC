"""The data structure module: creates the Cell-ACDC folder structure
(experiment folder, Position_<n>/Images) from raw microscopy files or
from loose TIFF files.

- data_struct.py: the data structure window createDataStructWin
  (Bio-Formats conversion), the symlinked-Positions window
  CreateSymLinkToPosWin and the Fiji macro setup InitFijiMacro.
- data_restruct.py: re-structuring TIFF files into Positions (run,
  started from the launcher).
- helpers.py: the Java, Bio-Formats and javabridge helpers (locating,
  downloading and installing them) and the git check.
- workers.py, dialogs/: workers and dialogs used only by this module.

This __init__ imports nothing.
"""
