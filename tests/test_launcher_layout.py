"""The launcher (`_main.mainWin`) groups the module buttons into vertically
stacked group boxes (preprocessing, segmentation and tracking, measurements
visualisation) above the "Controls" box, in the order of a typical analysis.
The buttons show no numbers, and docs and messages name the modules (e.g.
"the GUI"), never by number: getting-started.rst lists the buttons in launcher
order."""
import logging
import os
import re

import pytest
from qtpy.QtCore import QObject, QSize, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QGroupBox, QPushButton, QWidget

from cellacdc.data_structure import data_struct
from cellacdc.launcher import main_window as launcher_window
from cellacdc.measurements_viewer.main_window import window as viewer_window
from cellacdc.myutils import app_info
from cellacdc.shared.widgets import message_box as widgets_message_box
from cellacdc import cellacdc_path

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

    monkeypatch.setattr(app_info, 'setupLogger', setupLogger)
    monkeypatch.setattr(launcher_window.mainWin, 'checkConfigFiles', lambda self: None)
    windows = []

    def build(spotmaxInstalled=False):
        if spotmaxInstalled:
            monkeypatch.setattr(launcher_window, 'SPOTMAX_INSTALLED', True)
            monkeypatch.setattr(
                launcher_window, 'spotmax_logo_path', ':logo.svg', raising=False
            )
        win = launcher_window.mainWin(app)
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


@pytest.mark.smoke
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
        launcher_window.mainWin, 'launchMeasurementsViewer',
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
        viewer_window, 'MeasurementsViewerWin', _FakeViewerWin
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


def test_data_structure_button_texts_have_no_module_number(
        buildLauncher, monkeypatch
    ):
    # The other module buttons show no numbers either
    monkeypatch.setattr(
        launcher_window.mainWin, '_showDataStructWin', lambda self: None
    )
    win = buildLauncher()
    win.setColorsAndText()
    initial = win.dataStructButton.text()

    win.launchDataStruct()
    assert win.dataStructButton.text() == 'Creating data structure running...'

    # Restored exactly (the initial label has trailing spaces)
    win.restoreDefaultButtons()
    assert win.dataStructButton.text() == initial


def test_python_bioformats_data_structure_button_has_no_module_number(
        buildLauncher, monkeypatch
    ):
    def question(self, parent, title, text, buttonsTexts=(), **kwargs):
        # buttonsTexts = ('Cancel', "Use BioIO" button, ...)
        self.clickedButton = buttonsTexts[1]
        self.cancel = False

    class FakeDataStructWin(QWidget):
        bioformats_backend = 'python-bioformats'

        def __init__(self, *args, **kwargs):
            super().__init__()

        def show(self):
            pass

        def main(self):
            pass

    monkeypatch.setattr(widgets_message_box.myMessageBox, 'question', question)
    monkeypatch.setattr(data_struct, 'createDataStructWin', FakeDataStructWin)
    win = buildLauncher()
    win.setColorsAndText()

    win._showDataStructWin()

    button = win.dataStructButton
    assert not button.isEnabled()
    assert button.text() == (
        'Restart Cell-ACDC to create a data structure again'
    )
    toolTip = button.toolTip()
    assert 'the data structure module can be launched only once' in (
        ' '.join(toolTip.split())
    )
    assert 'module 0' not in toolTip
    assert 'moduel' not in toolTip and 'interal' not in toolTip
    assert all(len(line) <= 80 for line in toolTip.splitlines())


def test_getting_started_lists_the_module_buttons_in_launcher_order(
        buildLauncher
    ):
    # The docs list the buttons by their texts, in launcher order (no
    # numbers: docs and messages name the modules)
    win = buildLauncher(spotmaxInstalled=True)
    rst_path = os.path.join(
        cellacdc_path,
        'docs', 'source', 'getting-started.rst'
    )
    with open(rst_path, encoding='utf-8') as file:
        text = file.read()
    start = text.index('\nThe Main Menu\n')
    end = text.index('\nTop ribbon options include:', start)
    items = re.findall(r'^\* (.+)$', text[start:end], flags=re.MULTILINE)

    buttonTexts = [button.text().strip() for button in win.moduleButtons]
    assert buttonTexts[-1] == 'Launch SpotMAX...'
    assert [itemText.strip() for itemText in items] == buttonTexts
