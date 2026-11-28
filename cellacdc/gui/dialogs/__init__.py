"""Dialogs used only by the main GUI window (guiWin), one module per theme:
loading, saving, restoring and exporting data; editing objects and
annotations (edit IDs, manual separation, the cell cycle table editor);
points layers; segmentation and tracking (promptable models, tracker
selection, start and stop frames).

Dialogs opened from the mixins shared with the measurements viewer live
in gui/mixins/dialogs.py, dialogs of several program parts in
shared/dialogs/.

This package re-exports nothing: import its modules.
"""
