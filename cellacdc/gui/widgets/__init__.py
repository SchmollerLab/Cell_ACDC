"""Widgets used only by the main GUI window (gui/main_window/), never by
the measurements viewer. Widgets that the viewer-shared mixins reach are
in gui/mixins/widgets.py.

- edit_toolbars.py: editing toolbars (manual tracking, merge and
  whitelist IDs, separate bud, edit ID, ...).
- main_window_items.py: ghost items, brush circle, delete ROI, log
  widget and other window parts.
- points_layer_widgets.py: points layers and magic prompts toolbars and
  items.
"""


# Old names of cellacdc.widgets that code reached through this module
# path before it existed (cellacdc/_compat.py).
from cellacdc import _compat as _acdc_compat  # noqa: E402

__getattr__, __dir__ = _acdc_compat.forward_module(
    __name__, 'cellacdc.widgets')
