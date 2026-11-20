"""The launcher (`_main.mainWin`) groups the module buttons into vertically
stacked group boxes (preprocessing, segmentation and tracking, measurements
visualisation) above the "Controls" box, numbered across the groups in the
order of a typical analysis."""
import logging
import os

import pytest
from qtpy.QtCore import QObject, QSize, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QGroupBox, QPushButton

from cellacdc import _main, measurements_viewer, myutils


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


@pytest.fixture
def buildLauncher(app, monkeypatch, tmp_path):
    """Build the real `mainWin` without touching the user's files: no log
    file (and no `sys.stdout` replacement) and no clean-up of the (not
    redirected) favourite measurements file."""
    def setupLogger(module='base', **kwargs):
        logger = logging.getLogger(f'test_launcher_layout-{module}')
        log_filename = f'{module}.log'
        return (
            logger, str(tmp_path), os.path.join(tmp_path, log_filename),
            log_filename
        )

    monkeypatch.setattr(myutils, 'setupLogger', setupLogger)
    monkeypatch.setattr(_main.mainWin, 'checkConfigFiles', lambda self: None)
    windows = []

    def build(spotmaxInstalled=False):
        if spotmaxInstalled:
            monkeypatch.setattr(_main, 'SPOTMAX_INSTALLED', True)
            monkeypatch.setattr(
                _main, 'spotmax_logo_path', ':logo.svg', raising=False
            )
        win = _main.mainWin(app)
        windows.append(win)
        return win

    yield build

    for win in windows:
        win.hide()
        win.deleteLater()


def _moduleGroupBoxes(win):
    groupBoxes = win.centralWidget().findChildren(QGroupBox)
    # Top-to-bottom order as laid out in the window
    layout = win.centralWidget().layout()
    order = [layout.indexOf(groupBox) for groupBox in groupBoxes]
    return [groupBox for _, groupBox in sorted(zip(order, groupBoxes))]


def _buttonTexts(groupBox):
    layout = groupBox.layout()
    buttons = [
        layout.itemAt(i).widget() for i in range(layout.count())
        if isinstance(layout.itemAt(i).widget(), QPushButton)
    ]
    return [button.text().strip() for button in buttons], buttons


def test_module_buttons_are_grouped_in_stacked_group_boxes(buildLauncher):
    win = buildLauncher()

    groupBoxes = _moduleGroupBoxes(win)
    assert [groupBox.title() for groupBox in groupBoxes] == [
        'Image preprocessing',
        'Segmentation and tracking',
        'Measurements visualisation',
        'Controls',
    ]
    assert groupBoxes[:3] == [
        win.preprocessingGroupBox, win.segmTrackingGroupBox,
        win.measurementsGroupBox
    ]
    assert groupBoxes[3] is win.controlsGroupBox

    texts, buttons = _buttonTexts(win.preprocessingGroupBox)
    assert buttons == [win.dataStructButton, win.dataPrepButton]
    assert texts == [
        'Create data structure from microscopy/image file(s)...',
        'Launch data prep module...',
    ]

    texts, buttons = _buttonTexts(win.segmTrackingGroupBox)
    assert buttons == [win.segmButton, win.guiButton]
    assert texts == [
        'Automatic: launch segmentation and tracking module...',
        'Correction: launch GUI...',
    ]

    texts, buttons = _buttonTexts(win.measurementsGroupBox)
    assert buttons == [win.measurementsViewerButton]
    assert texts == ['Launch measurements viewer...']
    assert not hasattr(win, 'spotmaxButton')

    assert win.moduleButtons == [
        win.dataStructButton, win.dataPrepButton, win.segmButton,
        win.guiButton, win.measurementsViewerButton
    ]
    for button in win.moduleButtons:
        assert not button.icon().isNull()
    size = QSize(32, 32)
    expected = QIcon(':graphs.svg').pixmap(size).toImage()
    assert not expected.isNull()
    assert win.measurementsViewerButton.icon().pixmap(size).toImage() == (
        expected
    )


def test_module_buttons_get_the_same_height_when_shown(
        app, buildLauncher, monkeypatch
    ):
    # Set by _run._setup_app
    monkeypatch.setattr(app, 'toggle_dark_mode', False, raising=False)
    win = buildLauncher()
    # No migration prompt of an old user profile folder
    win.checkUserDataFolderPath = False

    win.show()

    heights = {button.minimumHeight() for button in win.moduleButtons}
    assert len(heights) == 1
    assert heights.pop() > 0
    assert (
        win.measurementsViewerButton.minimumHeight()
        == win.closeButton.minimumHeight()
    )


def test_spotmax_button_is_in_measurements_group(buildLauncher):
    win = buildLauncher(spotmaxInstalled=True)

    texts, buttons = _buttonTexts(win.measurementsGroupBox)
    assert buttons == [win.measurementsViewerButton, win.spotmaxButton]
    assert texts == [
        'Launch measurements viewer...', 'Launch SpotMAX...'
    ]
    assert win.moduleButtons[-1] is win.spotmaxButton


def test_measurements_viewer_button_launches_viewer(
        buildLauncher, monkeypatch
    ):
    calls = []
    monkeypatch.setattr(
        _main.mainWin, 'launchMeasurementsViewer',
        lambda self, checked=False: calls.append(self)
    )
    win = buildLauncher()

    win.measurementsViewerButton.click()

    assert calls == [win]


def test_measurements_viewer_button_opens_and_tracks_window(
        buildLauncher, monkeypatch
    ):
    _FakeViewerWin.instances = []
    monkeypatch.setattr(
        measurements_viewer, 'MeasurementsViewerWin', _FakeViewerWin
    )
    win = buildLauncher()
    win.setColorsAndText()

    win.measurementsViewerButton.click()

    assert len(_FakeViewerWin.instances) == 1
    viewer = _FakeViewerWin.instances[0]
    assert viewer.shown
    assert win.measurementsViewerWins == [viewer]
    assert win.getOpenModules() == [viewer]

    viewer.close()
    assert win.measurementsViewerWins == []
    assert win.getOpenModules() == []
