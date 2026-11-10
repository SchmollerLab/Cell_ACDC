"""Keep Cell-ACDC windows above their parents instead of above all apps.

``Qt.WindowStaysOnTopHint`` makes a window float above every application on 
the desktop. Use :func:`set_window_flags_on_top` instead:

* Windows with a parent stay above their whole parent chain and are 
  minimized/restored together with their parent.
* Windows without a parent stay above all other Cell-ACDC windows (but not
  above other applications). When several of them are open, the most
  recently active one is on top.

Windows created without a parent by third-party code (e.g., pyqtgraph colour
dialogs) can be given one with the `parent` argument of
:func:`set_window_flags_on_top`.

Windows with ``keep_on_top_of_parents(widget, False)`` are not raised above
their parent by this module, but they are still minimized/restored together
with it. On Windows, the OS always keeps a window above its parent (owner).
"""
import sys
from functools import partial

from qtpy.QtCore import Qt, QObject
from qtpy.QtWidgets import QApplication, QWidget

KEEP_ON_TOP_PROPERTY = 'acdcKeepOnTopOfParents'
# Token of the ancestor window whose minimization also minimized this window
_MINIMIZED_BY_PROPERTY = 'acdcMinimizedByWindow'
_WINDOW_TOKEN_PROPERTY = 'acdcWindowStackingToken'
_LAST_ACTIVE_PROPERTY = 'acdcWindowStackingLastActive'
_STATE_HOOKED_PROPERTY = 'acdcWindowStateHooked'

# On Windows, Qt makes windows with a parent "owned" windows: the OS keeps 
# them above their owner and hides/restores them together with it.
_NATIVE_OWNERSHIP = sys.platform.startswith('win')

_REGULAR_WINDOW_TYPES = (Qt.Window, Qt.Dialog, Qt.Tool)

_manager = None
_last_window_token = 0
_last_activation = 0

def set_window_flags_on_top(widget, flags=Qt.Window, parent=None):
    """Replacement for ``widget.setWindowFlags(flags | Qt.WindowStaysOnTopHint)``

    Parameters
    ----------
    widget : QWidget
        Window to keep on top of its parents.
    flags : Qt.WindowFlags, optional
        Window flags (type) of `widget`. Default is ``Qt.Window``.
    parent : QWidget, optional
        If not None, `widget` becomes a window owned by the window of
        `parent`, so that it stays above it. Use this for windows created
        without a parent (e.g., pyqtgraph colour dialogs). Default is None.
    """
    if isinstance(parent, QWidget):
        # Note that `setParent(parent)` without flags would turn `widget`
        # into a child widget embedded in `parent`
        widget.setParent(parent.window(), flags)
    else:
        widget.setWindowFlags(flags)
    keep_on_top_of_parents(widget)

def keep_on_top_of_parents(widget, keep=True):
    widget.setProperty(KEEP_ON_TOP_PROPERTY, bool(keep))
    if keep:
        if widget.parentWidget() is None:
            # Newly shown parentless windows go on top of the other ones
            _mark_activated(widget)
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

def _window_token(window):
    """Unique (per session) string identifying `window`"""
    global _last_window_token
    token = window.property(_WINDOW_TOKEN_PROPERTY)
    if token:
        return token
    _last_window_token += 1
    token = str(_last_window_token)
    window.setProperty(_WINDOW_TOKEN_PROPERTY, token)
    return token

def _mark_activated(window):
    global _last_activation
    _last_activation += 1
    window.setProperty(_LAST_ACTIVE_PROPERTY, _last_activation)

def _last_active(window):
    return window.property(_LAST_ACTIVE_PROPERTY) or 0

def _root_window(window):
    root = window
    for root in _parent_windows(window):
        pass
    return root

def _raise_windows(windows):
    for window in windows:
        if window.isMinimized() or not _is_kept_on_top(window):
            continue
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
        
        token = _window_token(window)
        descendants = _descendant_windows(window)
        if state & Qt.WindowMinimized:
            # Remember which ancestor minimized each descendant, so that
            # restoring another ancestor does not restore branches whose
            # closer ancestor is still minimized.
            for descendant in descendants:
                if descendant.isMinimized():
                    continue
                descendant.setProperty(_MINIMIZED_BY_PROPERTY, token)
                descendant.showMinimized()
            return

        # `window` is no longer minimized: it is not waiting to be restored
        # together with its ancestor anymore
        window.setProperty(_MINIMIZED_BY_PROPERTY, None)

        for descendant in descendants:
            if descendant.property(_MINIMIZED_BY_PROPERTY) != token:
                continue
            descendant.setProperty(_MINIMIZED_BY_PROPERTY, None)
            descendant.setWindowState(
                descendant.windowState() & ~Qt.WindowMinimized
            )
            # Do not call `show()`: several Cell-ACDC windows override it
            # with required arguments or side effects (e.g.,
            # `QDialogWorkerProgress.show(app)`)
            descendant.setVisible(True)
        _raise_windows(descendants)
    
    def onFocusWindowChanged(self, window_handle):
        self.hookWindowStates()
        if window_handle is None:
            return
        
        focused = _widget_for_window_handle(window_handle)
        if focused is None or not _is_regular_window(focused):
            return

        root = _root_window(focused)
        if _is_kept_on_top(root) and root.parentWidget() is None:
            _mark_activated(root)

        if not _NATIVE_OWNERSHIP:
            _raise_windows(_descendant_windows(focused))

        if QApplication.activeModalWidget() is not None:
            return

        if _is_kept_on_top(focused):
            return

        if any(_is_kept_on_top(win) for win in _parent_windows(focused)):
            return

        # Parentless kept-on-top windows stay above all other ACDC windows,
        # with the most recently active one on top
        kept_on_top = [
            widget for widget in QApplication.topLevelWidgets()
            if widget is not focused
            and widget.parentWidget() is None
            and widget.isVisible()
            and not widget.isMinimized()
            and _is_kept_on_top(widget)
        ]
        kept_on_top.sort(key=_last_active)
        for widget in kept_on_top:
            widget.raise_()
            if not _NATIVE_OWNERSHIP:
                _raise_windows(_descendant_windows(widget))
