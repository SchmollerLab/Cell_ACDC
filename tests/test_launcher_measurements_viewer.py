"""The launcher (`_main.mainWin`) opens the measurements viewer from
Utilities > Measurements and tracks it until it is closed."""
import logging

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QMenu, QPushButton

from cellacdc.launcher import main_window as launcher_window
from cellacdc.measurements_viewer.main_window import window as viewer_window
from cellacdc.info import utilsInfo

pytestmark = pytest.mark.gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class _FakeViewerWin(QObject):
    sigClosed = Signal(object)
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.shown = False
        _FakeViewerWin.instances.append(self)

    def show(self):
        self.shown = True

    def close(self):
        self.sigClosed.emit(self)


class _LauncherHost:
    """Minimal stand-in for `mainWin`: real action creation/connection and
    launch slots, no-op for every other slot."""
    launchMeasurementsViewer = launcher_window.mainWin.launchMeasurementsViewer
    measurementsViewerClosed = launcher_window.mainWin.measurementsViewerClosed
    getOpenModules = launcher_window.mainWin.getOpenModules

    def __init__(self):
        self.measurementsViewerWins = []
        self.guiWins = []
        self.spotmaxWins = []
        self.dataPrepWins = []
        # Segmentation module not launched
        self.segmButton = QPushButton()
        self.moduleLaunchedColor = '#123456'
        self.logger = logging.getLogger('test_launcher')

    def _gc_collect(self):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_measurements_viewer_action_opens_and_tracks_window(
        app, monkeypatch
    ):
    _FakeViewerWin.instances = []
    monkeypatch.setattr(
        viewer_window, 'MeasurementsViewerWin', _FakeViewerWin
    )
    host = _LauncherHost()
    host.recentPathsMenu = QMenu()
    launcher_window.mainWin.createActions(host)
    launcher_window.mainWin.connectActions(host)

    action = host.launchMeasurementsViewerAction
    # Listed in the Utilities help
    assert action.text() in utilsInfo

    action.trigger()

    assert len(_FakeViewerWin.instances) == 1
    win = _FakeViewerWin.instances[0]
    assert win.shown
    assert host.measurementsViewerWins == [win]
    # Closing the launcher asks about (and closes) open viewers too
    assert host.getOpenModules() == [win]

    win.close()
    assert host.measurementsViewerWins == []


def test_python_m_measurements_viewer_runs_the_viewer(monkeypatch):
    """``python -m cellacdc.measurements_viewer [folder]`` (the package's
    ``__main__.py``) runs ``run`` of the viewer window module."""
    import runpy
    import sys
    calls = []
    monkeypatch.setattr(
        viewer_window, 'run', lambda folder=None: calls.append(folder)
    )
    monkeypatch.setattr(sys, 'argv', ['measurements_viewer', 'exp_folder'])
    runpy.run_module('cellacdc.measurements_viewer', run_name='__main__')
    monkeypatch.setattr(sys, 'argv', ['measurements_viewer'])
    runpy.run_module('cellacdc.measurements_viewer', run_name='__main__')
    assert calls == ['exp_folder', None]
