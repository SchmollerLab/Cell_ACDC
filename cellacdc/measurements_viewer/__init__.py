"""The measurements viewer: a standalone window that plots per-cell time
courses of acdc_output columns, with time point identifiers, custom
annotations, phases and measurement processing.

- main_window/: the window class MeasurementsViewerWin (window.py) and
  its Viewer<Topic>GuiElements mixins.
- viewer_utils.py, position_data.py, annotation_data.py: the widget-free
  logic (MeasurementsViewerUtils and its bases) and the module functions
  (loading positions and tables, frame masks, logger).
- dialogs/, widgets.py, workers.py: dialogs, widgets and workers used
  only by the viewer.
- custom_annotation_predictors.py, phase_translator.py: the custom
  annotation predictors and the phase translation. The time point
  predictors stay in the top-level timepoint_predictors.py, which shared
  and library code also use.
- premade_configurations/: the premade configuration templates (JSON
  files).
- __main__.py: runs the viewer with python -m on this package,
  optionally loading a folder.

The window also composes the shared mixins of gui/mixins/. Nothing here
imports gui/main_window/, gui/dialogs/, gui/widgets/, gui/workers/ or
gui/helpers.py. This __init__ imports nothing eagerly: the names that
the former single-module viewer defined at this import path are served
lazily with a deprecation warning, for external code only (see
_compat.py).
"""

from cellacdc import _compat as _acdc_compat

__getattr__, __dir__ = _acdc_compat.package_compat(__name__)
