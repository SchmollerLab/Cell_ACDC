"""Widgets used by several program parts, by library modules or by plugins,
grouped by theme (formerly widgets.py).

The package imports style.py, so the pyqtgraph setup (row-major images,
colormap registry) runs before the body of any widget module, as
importing widgets.py did.

- style.py: stylesheets, colors, font, colormap registries (cmaps,
  Gradients, custom colormaps).
- basic_widgets.py: separators, spacers, ScrollArea, labels, progress
  bars, log console.
- push_buttons.py: PushButton and the icon buttons, getPushButton,
  Toggle, myColorButton.
- message_box.py: myMessageBox and its content helpers.
- line_edits.py: validated and numeric line edits, path controls,
  formWidget, shortcut text conversion.
- numeric_inputs.py: SpinBox, DoubleSpinBox, sliders, scroll bars,
  RangeSelector.
- combo_boxes.py: ComboBox and other combo boxes, PreProcessingSelector.
- item_views.py: QDialogListbox, listWidget, TreeWidget.
- search_widgets.py: ButtonSearchWidget and the fuzzy search helpers.
- lut_items.py: histogram LUT items and colormap menus.
- plot_items.py: pyqtgraph plot, image and graphics items.
- toolbars.py: ToolBar with overflow shortcuts, OverlayToolbar, tool
  buttons.
- imshow_window.py: the image preview window of imshow in plot.py.

Dialog modules import widget modules at module level. Three widget
modules import dialog modules at module level too: combo_boxes.py
(shared/dialogs/generic.py and preprocessing.py), line_edits.py and
lut_items.py (shared/dialogs/generic.py). These import cycles are safe
because the dialogs are used only at call time through the module alias
(dialogs_generic.QInput), never through a from-import of a name. The
other widget modules import no dialog module at module level.
"""

from . import style  # noqa: F401
