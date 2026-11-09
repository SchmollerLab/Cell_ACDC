"""The launcher (`_main.mainWin`) opens the measurements viewer from
Utilities > Measurements and tracks it until it is closed."""
import logging

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QMenu, QPushButton

from cellacdc import _main, measurements_viewer
from cellacdc.info import utilsInfo


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
    launchMeasurementsViewer = _main.mainWin.launchMeasurementsViewer
    measurementsViewerClosed = _main.mainWin.measurementsViewerClosed
    getOpenModules = _main.mainWin.getOpenModules

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
        measurements_viewer, 'MeasurementsViewerWin', _FakeViewerWin
    )
    host = _LauncherHost()
    host.recentPathsMenu = QMenu()
    _main.mainWin.createActions(host)
    _main.mainWin.connectActions(host)

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
