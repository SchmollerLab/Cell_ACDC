"""Keep Cell-ACDC windows above their parents instead of above all apps.

``Qt.WindowStaysOnTopHint`` makes a window float above every application on 
the desktop. Use :func:`set_window_flags_on_top` instead:

* Windows with a parent stay above their whole parent chain and are 
  minimized/restored together with their parent.
* Windows without a parent stay above all other Cell-ACDC windows (but not 
  above other applications).
"""
import sys
from functools import partial

from qtpy.QtCore import Qt, QObject
from qtpy.QtWidgets import QApplication

KEEP_ON_TOP_PROPERTY = 'acdcKeepOnTopOfParents'
_MINIMIZED_WITH_PARENT_PROPERTY = 'acdcMinimizedWithParent'
_STATE_HOOKED_PROPERTY = 'acdcWindowStateHooked'

# On Windows, Qt makes windows with a parent "owned" windows: the OS keeps 
# them above their owner and hides/restores them together with it.
_NATIVE_OWNERSHIP = sys.platform.startswith('win')

_REGULAR_WINDOW_TYPES = (Qt.Window, Qt.Dialog, Qt.Tool)

_manager = None

def set_window_flags_on_top(widget, flags=Qt.Window):
    """Replacement for ``widget.setWindowFlags(flags | Qt.WindowStaysOnTopHint)``
    """
    widget.setWindowFlags(flags)
    keep_on_top_of_parents(widget)

def keep_on_top_of_parents(widget, keep=True):
    widget.setProperty(KEEP_ON_TOP_PROPERTY, bool(keep))
    if keep:
        manager = _get_manager()
        if manager is not None:
            manager.hookWindowStates()

def _get_manager():
    global _manager
    if _manager is not None:
        return _manager
    
    app = QApplication.instance()
    if app is None:
        return
    
    _manager = _WindowStackingManager(app)
    return _manager

def _is_kept_on_top(widget):
    return bool(widget.property(KEEP_ON_TOP_PROPERTY))

def _is_regular_window(widget):
    return widget.windowType() in _REGULAR_WINDOW_TYPES

def _parent_windows(widget):
    parent = widget.parentWidget()
    while parent is not None:
        window = parent.window()
        yield window
        parent = window.parentWidget()

def _descendant_windows(window):
    """Visible windows owned (directly or not) by `window`, parents first"""
    descendants = []
    for widget in QApplication.topLevelWidgets():
        if widget is window:
            continue
        if not widget.isVisible() or not _is_regular_window(widget):
            continue
        for depth, parent_window in enumerate(_parent_windows(widget)):
            if parent_window is window:
                descendants.append((depth, widget))
                break
    descendants.sort(key=lambda depth_widget: depth_widget[0])
    return [widget for _, widget in descendants]

def _widget_for_window_handle(window_handle):
    for widget in QApplication.topLevelWidgets():
        if widget.windowHandle() is window_handle:
            return widget

def _raise_windows(windows):
    for window in windows:
        if not window.isMinimized():
            window.raise_()

class _WindowStackingManager(QObject):
    def __init__(self, app):
        super().__init__(app)
        app.focusWindowChanged.connect(self.onFocusWindowChanged)
    
    def hookWindowStates(self):
        if _NATIVE_OWNERSHIP:
            return
        
        for widget in QApplication.topLevelWidgets():
            window_handle = widget.windowHandle()
            if window_handle is None:
                continue
            if window_handle.property(_STATE_HOOKED_PROPERTY):
                continue
            window_handle.setProperty(_STATE_HOOKED_PROPERTY, True)
            window_handle.windowStateChanged.connect(
                partial(self.onWindowStateChanged, window_handle)
            )
    
    def onWindowStateChanged(self, window_handle, state):
        window = _widget_for_window_handle(window_handle)
        if window is None:
            return
        
        descendants = _descendant_windows(window)
        if state & Qt.WindowMinimized:
            for descendant in descendants:
                if descendant.isMinimized():
                    continue
                descendant.setProperty(_MINIMIZED_WITH_PARENT_PROPERTY, True)
                descendant.showMinimized()
            return
        
        for descendant in descendants:
            if not descendant.property(_MINIMIZED_WITH_PARENT_PROPERTY):
                continue
            descendant.setProperty(_MINIMIZED_WITH_PARENT_PROPERTY, False)
            descendant.setWindowState(
                descendant.windowState() & ~Qt.WindowMinimized
            )
            descendant.show()
        _raise_windows(descendants)
    
    def onFocusWindowChanged(self, window_handle):
        self.hookWindowStates()
        if window_handle is None:
            return
        
        focused = _widget_for_window_handle(window_handle)
        if focused is None or not _is_regular_window(focused):
            return
        
        if not _NATIVE_OWNERSHIP:
            _raise_windows(_descendant_windows(focused))
        
        if QApplication.activeModalWidget() is not None:
            return
        
        if _is_kept_on_top(focused):
            return
        
        if any(_is_kept_on_top(win) for win in _parent_windows(focused)):
            return
        
        # Parentless kept-on-top windows stay above all other ACDC windows
        for widget in QApplication.topLevelWidgets():
            if widget is focused or widget.parentWidget() is not None:
                continue
            if not widget.isVisible() or widget.isMinimized():
                continue
            if not _is_kept_on_top(widget):
                continue
            widget.raise_()
            if not _NATIVE_OWNERSHIP:
                _raise_windows(_descendant_windows(widget))
