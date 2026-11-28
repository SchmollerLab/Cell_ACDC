"""Mixins composed by BOTH the main GUI window (gui/main_window/window.py)
and the measurements viewer window
(measurements_viewer/main_window/window.py): changes affect both
windows.

Host-specific behaviour goes through hooks with main-GUI defaults that
the viewer overrides (e.g. showDeleteObjectShortcut,
canDeleteSavedCustomAnnot). dialogs.py and widgets.py hold the dialogs
and widgets reached only from these mixins.

Nothing here imports gui/main_window/, gui/dialogs/, gui/widgets/,
gui/workers/ or gui/helpers.py. The only import of the viewer is the
lazy one of its predictor editors in dialogs.py, made when the viewer
passes predictor options.
"""
