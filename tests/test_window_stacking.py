import gc

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from cellacdc import _window_stacking

pytestmark = pytest.mark.gui


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _window_deleted_by_gc():
    """Window deleted together with its parent by the garbage collector.

    The parent is reachable only through a reference cycle, so
    `gc.collect()` frees it and its C++ object deletes the windows it owns,
    while the Python wrapper of the child window is still referenced.
    """
    parent = QWidget()
    parent.selfReference = parent
    window = QWidget(parent, Qt.Window)
    del parent
    gc.collect()
    with pytest.raises(RuntimeError, match='deleted'):
        window.windowHandle()
    return window


def test_deleted_top_level_widgets_are_skipped(app, monkeypatch):
    # The deleted windows only reach the code that hooks the window states
    # (the code path of macOS and Linux)
    monkeypatch.setattr(_window_stacking, '_NATIVE_OWNERSHIP', False)
    deleted = _window_deleted_by_gc()
    alive = QWidget()
    # Create the native window (and its window handle) without showing it
    alive.winId()
    window_handle = alive.windowHandle()
    assert window_handle is not None
    # `topLevelWidgets()` can still list windows that are deleted before
    # they are accessed
    monkeypatch.setattr(
        QApplication, 'topLevelWidgets',
        staticmethod(lambda: [deleted, alive])
    )

    manager = _window_stacking._get_manager()
    manager.hookWindowStates()
    assert window_handle.property(_window_stacking._STATE_HOOKED_PROPERTY)
    assert _window_stacking._widget_for_window_handle(window_handle) is alive
    assert _window_stacking._descendant_windows(alive) == []
    manager.onFocusWindowChanged(window_handle)
    _window_stacking._raise_windows([deleted, alive])

    alive.deleteLater()
