"""Tests for the napari-style script API."""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest


def _reload_viewer_module():
    import cellacdc.viewer

    return importlib.reload(cellacdc.viewer)


def test_viewer_sets_segmentation_and_tracking_mode():
    mock_win = MagicMock()
    viewer_mod = _reload_viewer_module()

    with (
        patch("cellacdc._event_loop.get_qapp", return_value=MagicMock()),
        patch.object(viewer_mod, "_read_version", return_value="test"),
        patch.object(viewer_mod, "_create_gui_window", return_value=mock_win),
        patch.object(viewer_mod, "_check_gui_installed"),
    ):
        viewer = viewer_mod.Viewer()

    mock_win.modeComboBox.setCurrentText.assert_called_once_with(
        "Segmentation and Tracking"
    )
    mock_win.raise_.assert_called_once()
    mock_win.activateWindow.assert_called_once()
    assert viewer.window is mock_win


def test_viewer_open_dispatches_folder_and_file(tmp_path):
    mock_win = MagicMock()
    folder = tmp_path / "experiment"
    folder.mkdir()
    file_path = tmp_path / "image.tif"
    file_path.write_text("")
    viewer_mod = _reload_viewer_module()

    with (
        patch("cellacdc._event_loop.get_qapp", return_value=MagicMock()),
        patch.object(viewer_mod, "_read_version", return_value="test"),
        patch.object(viewer_mod, "_create_gui_window", return_value=mock_win),
        patch.object(viewer_mod, "_check_gui_installed"),
    ):
        viewer = viewer_mod.Viewer(show=False)
        viewer.open(folder)
        viewer.open(file_path)

    mock_win.openFolder.assert_called_once_with(exp_path=os.fspath(folder))
    mock_win.openFile.assert_called_once_with(file_path=os.fspath(file_path))


def test_current_viewer_returns_latest_instance():
    mock_win = MagicMock()
    viewer_mod = _reload_viewer_module()

    with (
        patch("cellacdc._event_loop.get_qapp", return_value=MagicMock()),
        patch.object(viewer_mod, "_read_version", return_value="test"),
        patch.object(viewer_mod, "_create_gui_window", return_value=mock_win),
        patch.object(viewer_mod, "_check_gui_installed"),
    ):
        assert viewer_mod.current_viewer() is None
        viewer_mod.Viewer(show=False)
        second = viewer_mod.Viewer(show=False)
        assert viewer_mod.current_viewer() is second


def test_run_warns_without_top_level_widgets():
    mock_app = MagicMock()
    mock_app.topLevelWidgets.return_value = []
    mock_app.thread.return_value.loopLevel.return_value = 0

    with (
        patch("cellacdc._event_loop._ipython_has_eventloop", return_value=False),
        patch("qtpy.QtWidgets.QApplication") as mock_qapp_cls,
        pytest.warns(UserWarning, match="Refusing to run a QApplication"),
    ):
        mock_qapp_cls.instance.return_value = mock_app
        from cellacdc._event_loop import run

        run()


def test_run_starts_event_loop_when_widgets_exist():
    mock_app = MagicMock()
    mock_app.topLevelWidgets.return_value = [MagicMock()]
    mock_app.thread.return_value.loopLevel.return_value = 0

    with (
        patch("cellacdc._event_loop._ipython_has_eventloop", return_value=False),
        patch("qtpy.QtWidgets.QApplication") as mock_qapp_cls,
    ):
        mock_qapp_cls.instance.return_value = mock_app
        from cellacdc._event_loop import run

        run()

    mock_app.exec_.assert_called_once()


def test_lazy_exports_from_package():
    import cellacdc

    assert cellacdc.Viewer.__name__ == "Viewer"
    assert cellacdc.current_viewer.__name__ == "current_viewer"
    assert cellacdc.run.__name__ == "run"
    assert cellacdc.get_qapp.__name__ == "get_qapp"
    assert cellacdc.quit_app.__name__ == "quit_app"
