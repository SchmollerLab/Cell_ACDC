"""Napari-style script API for launching the Cell-ACDC GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakSet

from cellacdc.data_source import ExperimentData

if TYPE_CHECKING:
    from cellacdc.gui import guiWin

_DEFAULT_MODE = "Segmentation and Tracking"


def _check_gui_installed() -> None:
    from cellacdc import GUI_INSTALLED

    if not GUI_INSTALLED:
        raise RuntimeError(
            "Cell-ACDC GUI dependencies are not installed. "
            'Install them with `pip install "cellacdc[gui]"`.'
        )


def _read_version() -> str:
    from cellacdc import utils

    return utils.read_version()


def _create_gui_window(app, version: str):
    from cellacdc import gui

    win = gui.guiWin(app, mainWin=None, version=version)
    win.run()
    return win


class Viewer:
    """Launch the Cell-ACDC annotation GUI from a script or notebook."""

    _instances: WeakSet[Viewer] = WeakSet()

    def __init__(
        self,
        data: ExperimentData | None = None,
        *,
        show: bool = True,
        mode: str = _DEFAULT_MODE,
    ):
        _check_gui_installed()

        from cellacdc._event_loop import get_qapp

        app = get_qapp()
        version = _read_version()
        win = _create_gui_window(app, version)
        win.modeComboBox.setCurrentText(mode)

        self._data = data
        if data is not None:
            data.load_into(win)

        if show:
            win.raise_()
            win.activateWindow()

        self._window = win
        self._instances.add(self)

    @property
    def data(self) -> ExperimentData | None:
        return self._data

    @property
    def window(self) -> guiWin:
        return self._window

    def close(self) -> None:
        self._window.close()


def current_viewer() -> Viewer | None:
    """Return the most recently created viewer, if any."""
    instances = list(Viewer._instances)
    if not instances:
        return None
    return instances[-1]


def imshow(
    data: ExperimentData,
    *,
    show: bool = True,
    mode: str = _DEFAULT_MODE,
) -> tuple[Viewer, ExperimentData]:
    """Open the GUI with an :class:`ExperimentData` instance."""
    viewer = Viewer(data, show=show, mode=mode)
    return viewer, data
