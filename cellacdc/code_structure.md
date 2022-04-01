# Custom annotations

Currently we are implementing the following types of annotations:

1. Single time-point
2. Multiple time-points
3. Multiple values class

As of April 2022 we only implemented 1. Single time-point.

## Single time-point annotation

If the user loads time-lapse data then the annotation mode needs to be activated from the `modeComboBox` control. Otherwise it is always active.

A custom annotation can be added by clicking on the `addCustomAnnotationAction`
tool button on the left side toolbar. This action is connected (with `triggered` slot) to the `addCustomAnnotation` function.

This function will open a dialog from the `apps` module called `customAnnotationDialog` and the window reference is stored in `addAnnotWin` attribute of the gui.

The `addAnnotWin` window has the following attributes:

- `symbol`: any of the `pyqtgraph` valid symbols (only the string inside the quotes, e.g., 'o' for circle, see `widgets.pgScatterSymbolsCombobox`)
- `keySequence`: a `QKeySequence` built with valid PyQt shortcut text, see [here](https://doc.qt.io/qt-5/qkeysequence.html#QKeySequence-1). Note that macOS shortcut strings are converted to valid PyQt string using the `widgets.macShortcutToQKeySequence` function.
