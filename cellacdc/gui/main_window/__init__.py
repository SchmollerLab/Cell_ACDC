"""The main GUI window guiWin and the code that only guiWin composes. The
measurements viewer never imports these modules.

- window.py: class guiWin with its base classes (QMainWindow first, then
  the window-only mixins of this package, then the shared mixins of
  gui/mixins/), Signals, overriding class attributes, __init__, the
  Qt-named overrides and the methods that call super(). It also runs the
  import-time setup of the main GUI.
- <topic>.py: the <Topic>GuiElements mixins of one topic each (window
  setup, layout, menus and actions, toolbars, cursors and hover, mouse
  and keyboard, modes, frame navigation and data, image display, drawing
  tools, object editing, ROIs, segmentation, magic prompts, tracking,
  cell cycle and lineage tree annotation, points layers, loading,
  saving, exports, image processing, whitelist IDs). Add a new guiWin
  method to the mixin of its topic, never to the class body unless it is
  Qt-named or calls super().
- constants.py: module-level constants of the main GUI. Read and write
  the font as constants._font: a from-import would not see the font size
  changes.

This __init__ imports nothing, so importing one module does not load
window.py.
"""
