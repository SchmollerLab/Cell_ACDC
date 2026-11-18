"""Tests of the GUI logic of brush strokes prompts (e.g., ConvPaint): the
"Train only, do not segment" right-click action of "Segment image", and
the selection of cells of the segmentation masks to add them to the brush
strokes (training data).

The `guiWin` methods are tested on fake `self` objects (no main window).
"""
import os
from collections import deque
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from cellacdc import gui, widgets


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def toolbar(app):
    toolbar = widgets.MagicPromptsToolbar()
    yield toolbar
    toolbar.deleteLater()


@pytest.fixture
def controls(app):
    controls = widgets.BrushStrokesSelectCellsToolbar('Select cells')
    yield controls
    controls.deleteLater()


def _lab():
    lab = np.zeros((6, 8), dtype=np.uint32)
    lab[0:2, 0:2] = 1
    lab[0:2, 4:6] = 2
    lab[4:6, 5:8] = 3
    return lab


# ---------------------------------------------------------------------------
# MagicPromptsToolbar
# ---------------------------------------------------------------------------

def test_train_is_right_click_action_of_segment_image(toolbar):
    # Not a toolbar button anymore
    assert toolbar.trainAction not in toolbar.actions()
    assert toolbar.trainAction not in toolbar._brushStrokesOnlyActions

    button = toolbar.widgetForAction(toolbar.segmentImageAction)
    assert button is toolbar.segmentImageButton
    assert (
        button.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    )
    menuActions = toolbar.segmentImageContextMenu.actions()
    assert menuActions == [toolbar.trainAction]
    assert toolbar.trainAction.text() == 'Train only, do not segment'
    assert 'Right-click' in toolbar.segmentImageAction.toolTip()

    emitted = []
    toolbar.sigTrain.connect(emitted.append)
    toolbar.trainAction.trigger()
    assert emitted == [toolbar]


def test_segment_image_context_menu_not_shown_when_disabled(toolbar):
    toolbar.segmentImageContextMenu.exec = Mock()
    # Hidden actions (e.g., "Points" prompt type) are disabled
    toolbar.setPromptType('point')
    toolbar.showSegmentImageContextMenu(toolbar.segmentImageButton.pos())
    toolbar.segmentImageContextMenu.exec.assert_not_called()

    toolbar.setPromptType('brush_stroke')
    toolbar.segmentImageAction.setDisabled(True)
    toolbar.showSegmentImageContextMenu(toolbar.segmentImageButton.pos())
    toolbar.segmentImageContextMenu.exec.assert_not_called()

    toolbar.segmentImageAction.setDisabled(False)
    toolbar.showSegmentImageContextMenu(toolbar.segmentImageButton.pos())
    toolbar.segmentImageContextMenu.exec.assert_called_once()


def test_select_cells_actions_only_for_brush_strokes(toolbar):
    brushOnly = toolbar._brushStrokesOnlyActions
    assert toolbar.selectCellsAction in brushOnly
    assert toolbar.selectAllCellsAction in brushOnly
    assert toolbar.selectCellsAction.isCheckable()

    toolbar.setPromptType('brush_stroke')
    assert toolbar.selectCellsAction.isVisible()
    assert toolbar.selectAllCellsAction.isVisible()

    toolbar.setPromptType('point')
    assert not toolbar.selectCellsAction.isVisible()
    assert not toolbar.selectAllCellsAction.isVisible()


def test_select_all_cells_shortcut_is_customizable(toolbar):
    action = toolbar.selectAllCellsAction
    assert action.shortcut().toString() == 'Ctrl+A'
    assert toolbar.widgetsWithShortcut[
        'Select all cells (magic prompts)'
    ] is action
    # Pattern updated by `guiWin.setShortcuts`
    assert 'Shortcut: "Ctrl+A"' in action.toolTip()


def test_switching_to_points_stops_selecting_cells(toolbar):
    toggles = []
    toolbar.sigSelectCellsToggled.connect(
        lambda tb, checked: toggles.append(checked)
    )
    selectAll = []
    toolbar.sigSelectAllCells.connect(selectAll.append)

    toolbar.setPromptType('brush_stroke')
    toolbar.selectCellsAction.setChecked(True)
    toolbar.selectAllCellsAction.trigger()
    toolbar.setPromptType('point')

    assert toggles == [True, False]
    assert not toolbar.selectCellsAction.isChecked()
    assert selectAll == [toolbar]


def test_help_explains_example_pixels_and_train_only(toolbar, monkeypatch):
    texts = []

    class FakeMessageBox:
        def __init__(self, *args, **kwargs):
            pass

        def information(self, parent, title, txt, *args, **kwargs):
            texts.append(txt)

    monkeypatch.setattr(widgets, 'myMessageBox', FakeMessageBox)
    toolbar.showHelp()
    txt = texts[0]
    assert 'example pixels' in txt
    assert 'not meant to cover all' in txt
    assert 'Train only, do not segment' in txt
    assert 'Ctrl+A' in txt
    assert '"<b>Train</b>"' not in txt


# ---------------------------------------------------------------------------
# BrushStrokesSelectCellsToolbar
# ---------------------------------------------------------------------------

def test_select_cells_controls_toggle_and_reset(controls):
    edited = []
    controls.sigSelectedIDsEdited.connect(
        lambda tb, IDs: edited.append(IDs)
    )
    action = controls.addToTrainingDataAction
    assert not action.isEnabled()

    assert controls.toggleSelectedID(np.uint32(3)) is True
    assert controls.toggleSelectedID(5) is True
    assert list(controls.selectedIDs) == [3, 5]
    assert controls.selectedIDsLineEdit.text() == '3,5'
    assert action.isEnabled()

    assert controls.toggleSelectedID(3) is False
    assert list(controls.selectedIDs) == [5]

    controls.setSelectedIDs([1, 2, 3, 7, 7])
    assert list(controls.selectedIDs) == [1, 2, 3, 7]
    assert controls.selectedIDsLineEdit.text() == '1-3,7'
    assert isinstance(controls.selectedIDs, widgets.KeptObjectIDsList)

    controls.resetSelection()
    assert list(controls.selectedIDs) == []
    assert controls.selectedIDsLineEdit.text() == ''
    assert not action.isEnabled()

    # Programmatic changes do not emit the "edited" signal
    assert edited == []


def test_select_cells_controls_typed_ids(controls):
    edited = []
    controls.sigSelectedIDsEdited.connect(
        lambda tb, IDs: edited.append(IDs)
    )
    controls.selectedIDsLineEdit.setText('1-3')
    assert edited == [[1, 2, 3]]

    controls.syncSelectedIDs([1, 2])
    assert list(controls.selectedIDs) == [1, 2]
    # Text typed by the user is not modified
    assert controls.selectedIDsLineEdit.text() == '1-3'
    assert controls.addToTrainingDataAction.isEnabled()

    controls.syncSelectedIDs([])
    assert not controls.addToTrainingDataAction.isEnabled()


def test_add_to_training_data_requires_selection(controls):
    emitted = []
    controls.sigAddToTrainingData.connect(emitted.append)

    controls.emitSigAddToTrainingData()
    controls.selectedIDsLineEdit.sigEnterPressed.emit()
    assert emitted == []

    controls.toggleSelectedID(4)
    controls.addToTrainingDataAction.trigger()
    controls.selectedIDsLineEdit.sigEnterPressed.emit()
    assert emitted == [controls, controls]


# ---------------------------------------------------------------------------
# guiWin methods with a fake `self`
# ---------------------------------------------------------------------------

def _bind(window, *names):
    for name in names:
        func = getattr(gui.guiWin, name)
        func = getattr(func, '__wrapped__', func)
        setattr(window, name, MethodType(func, window))


def _make_window(
        lab, selectedIDs=(), only_zoom=False, z=None,
        zProj='single z-slice', view_range=None
    ):
    IDs = [int(ID) for ID in np.unique(lab) if ID > 0]
    posData = SimpleNamespace(lab=lab, segm_data=None, frame_i=0, IDs=IDs)
    controls = SimpleNamespace(
        selectedIDs=list(selectedIDs),
        onlyZoomCheckbox=Mock(isChecked=Mock(return_value=only_zoom)),
        resetSelection=Mock(),
    )
    if view_range is None:
        view_range = ((0, lab.shape[-1]), (0, lab.shape[-2]))
    window = SimpleNamespace(
        data=[posData],
        pos_i=0,
        brushStrokesAnnotData={},
        brushStrokesUndo={},
        brushStrokesChangedSinceTraining=False,
        logger=Mock(),
        undoAction=Mock(),
        titleLabel=Mock(),
        titleColor='w',
        setBrushStrokesLayerImage=Mock(),
        warnBrushStrokesNotPaintable=Mock(),
        isBrushStrokesSelectCellsActive=Mock(return_value=True),
        getBrushStrokesDepthAxes=Mock(return_value='z'),
        zProjComboBox=Mock(currentText=Mock(return_value=zProj)),
        zSliceScrollBar=Mock(
            sliderPosition=Mock(return_value=0 if z is None else z)
        ),
        ax1=Mock(viewRange=Mock(return_value=view_range)),
    )
    _bind(
        window,
        'getBrushStrokesAnnotShape', 'getBrushStrokesAnnot',
        'isBrushStrokesViewPaintable', 'getBrushStrokesAnnot2D',
        'getBrushStrokesZoomRegion', 'storeBrushStrokesUndo',
        'undoBrushStrokes', 'magicPromptsAddSelectedCellsToBrushStrokes',
    )
    return window, posData, controls


def test_add_selected_cells_to_brush_strokes_2D():
    lab = _lab()
    window, posData, controls = _make_window(lab, selectedIDs=[1, 3])
    # Existing strokes: foreground on a pixel without masks, background on
    # a selected cell --> both are kept
    annot = window.getBrushStrokesAnnot(create=True)
    annot[3, 0] = 2
    annot[0, 0] = 1
    before = annot.copy()

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    annot = window.brushStrokesAnnotData[0][0]
    assert annot[3, 0] == 2 and annot[0, 0] == 1
    selected = np.isin(lab, [1, 3])
    selected[0, 0] = False
    assert (annot[selected] == 2).all()
    background = lab == 0
    background[3, 0] = False
    assert (annot[background] == 1).all()
    assert (annot[lab == 2] == 0).all()

    assert window.brushStrokesChangedSinceTraining is True
    controls.resetSelection.assert_called_once()
    window.setBrushStrokesLayerImage.assert_called()

    # Ctrl+Z undoes the whole conversion
    undoQueue = window.brushStrokesUndo[(0, 0)]
    assert len(undoQueue) == 1 and undoQueue[0][0] is None
    assert window.undoBrushStrokes() is True
    assert np.array_equal(window.brushStrokesAnnotData[0][0], before)


def test_add_selected_cells_to_brush_strokes_3D_only_displayed_zslice():
    lab = np.zeros((3, 6, 8), dtype=np.uint32)
    lab[:] = _lab()
    window, posData, controls = _make_window(lab, selectedIDs=[2], z=1)

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    annot = window.brushStrokesAnnotData[0][0]
    assert annot.shape == lab.shape
    assert not annot[[0, 2]].any()
    assert (annot[1][lab[1] == 2] == 2).all()
    assert (annot[1][lab[1] == 0] == 1).all()
    assert window.brushStrokesUndo[(0, 0)][0][0] == 1


def test_add_selected_cells_to_brush_strokes_projection_not_paintable():
    lab = np.zeros((3, 6, 8), dtype=np.uint32)
    lab[:] = _lab()
    window, posData, controls = _make_window(
        lab, selectedIDs=[2], z=1, zProj='max z-projection'
    )

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    window.warnBrushStrokesNotPaintable.assert_called_once()
    assert window.brushStrokesChangedSinceTraining is False
    controls.resetSelection.assert_not_called()
    annot = window.brushStrokesAnnotData.get(0, {}).get(0)
    assert annot is None or not annot.any()


def test_add_selected_cells_to_brush_strokes_only_zoomed_area():
    lab = _lab()
    # Displayed area: columns 0-3 and rows 0-3
    window, posData, controls = _make_window(
        lab, selectedIDs=[1, 2], only_zoom=True,
        view_range=((0, 3), (0, 3))
    )

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    annot = window.brushStrokesAnnotData[0][0]
    assert not annot[3:].any()
    assert not annot[:, 3:].any()
    # ID 2 is selected but outside of the zoomed area
    assert (annot[lab == 2] == 0).all()
    assert (annot[0:2, 0:2] == 2).all()
    assert (annot[2, 0:3] == 1).all()


def test_add_selected_cells_nothing_to_add_discards_undo():
    lab = _lab()
    window, posData, controls = _make_window(lab, selectedIDs=[1])
    annot = window.getBrushStrokesAnnot(create=True)
    annot[:] = 1

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    assert not window.brushStrokesUndo.get((0, 0))
    assert window.brushStrokesChangedSinceTraining is False
    controls.resetSelection.assert_not_called()
    assert (window.brushStrokesAnnotData[0][0] == 1).all()


def test_add_selected_cells_ignores_ids_not_in_frame():
    lab = _lab()
    window, posData, controls = _make_window(lab, selectedIDs=[99])

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    assert window.brushStrokesAnnotData == {}
    assert window.brushStrokesChangedSinceTraining is False


def test_add_selected_cells_not_selecting_does_nothing():
    lab = _lab()
    window, posData, controls = _make_window(lab, selectedIDs=[1])
    window.isBrushStrokesSelectCellsActive.return_value = False

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    assert window.brushStrokesAnnotData == {}


def _make_select_window(lab, controls, toolbar=None):
    IDs = [int(ID) for ID in np.unique(lab) if ID > 0]
    posData = SimpleNamespace(lab=lab, IDs=IDs)
    window = SimpleNamespace(
        data=[posData],
        pos_i=0,
        currentLab2D=lab,
        brushStrokesSelectCellsToolbar=controls,
        magicPromptsToolbar=toolbar,
        titleLabel=Mock(),
        titleColor='w',
        setBrushStrokesSelectedCellsLayerImage=Mock(),
        isBrushStrokesPromptActive=Mock(return_value=True),
        isBrushStrokesSelectCellsActive=Mock(return_value=True),
    )
    _bind(
        window,
        'brushStrokesSelectCellClick', 'brushStrokesSelectAllCells',
        'brushStrokesSelectedIDsEdited', 'brushStrokesSelectCellsEscape',
    )
    return window


def _click_event(x, y):
    pos = Mock(x=Mock(return_value=x), y=Mock(return_value=y))
    return Mock(pos=Mock(return_value=pos))


def test_select_cell_click_toggles_selection(controls):
    lab = _lab()
    window = _make_select_window(lab, controls)

    window.brushStrokesSelectCellClick(_click_event(0.5, 0.5))
    window.brushStrokesSelectCellClick(_click_event(6.2, 4.7))
    assert list(controls.selectedIDs) == [1, 3]
    assert window.setBrushStrokesSelectedCellsLayerImage.call_count == 2

    # Click on background --> selection unchanged
    window.brushStrokesSelectCellClick(_click_event(3, 3))
    assert list(controls.selectedIDs) == [1, 3]
    assert window.titleLabel.setText.call_args.kwargs['color'] == 'r'

    # Click outside of the image --> ignored
    window.brushStrokesSelectCellClick(_click_event(100, 3))
    assert list(controls.selectedIDs) == [1, 3]

    # Click again on a selected cell --> unselected
    window.brushStrokesSelectCellClick(_click_event(1, 1))
    assert list(controls.selectedIDs) == [3]


def test_select_all_cells_activates_selection(controls, toolbar):
    lab = _lab()
    window = _make_select_window(lab, controls, toolbar=toolbar)
    toolbar.setPromptType('brush_stroke')

    window.brushStrokesSelectAllCells(toolbar)

    assert toolbar.selectCellsAction.isChecked()
    assert list(controls.selectedIDs) == [1, 2, 3]
    assert controls.addToTrainingDataAction.isEnabled()
    window.setBrushStrokesSelectedCellsLayerImage.assert_called()


def test_select_all_cells_requires_brush_strokes_active(controls, toolbar):
    lab = _lab()
    window = _make_select_window(lab, controls, toolbar=toolbar)
    window.isBrushStrokesPromptActive.return_value = False

    window.brushStrokesSelectAllCells(toolbar)

    assert not toolbar.selectCellsAction.isChecked()
    assert list(controls.selectedIDs) == []


def test_typed_selected_ids_are_validated(controls):
    lab = _lab()
    window = _make_select_window(lab, controls)
    # PyQt cannot connect methods of objects without weak references
    controls.sigSelectedIDsEdited.connect(
        lambda tb, IDs: window.brushStrokesSelectedIDsEdited(tb, IDs)
    )

    controls.selectedIDsLineEdit.setText('1,3,99')
    assert sorted(controls.selectedIDs) == [1, 3]
    assert 'IGNORED' in controls.selectedIDsLineEdit._label.text()

    controls.selectedIDsLineEdit.setText('2')
    assert list(controls.selectedIDs) == [2]
    assert 'IGNORED' not in controls.selectedIDsLineEdit._label.text()


def test_escape_clears_selection_then_stops_selecting(controls, toolbar):
    lab = _lab()
    window = _make_select_window(lab, controls, toolbar=toolbar)
    toolbar.setPromptType('brush_stroke')
    toolbar.selectCellsAction.setChecked(True)
    controls.setSelectedIDs([1, 2])

    assert window.brushStrokesSelectCellsEscape() is True
    assert list(controls.selectedIDs) == []
    assert toolbar.selectCellsAction.isChecked()

    assert window.brushStrokesSelectCellsEscape() is True
    assert not toolbar.selectCellsAction.isChecked()

    window.isBrushStrokesSelectCellsActive.return_value = False
    assert window.brushStrokesSelectCellsEscape() is False


def test_get_objects_lab2D_shared_with_keep_ids():
    import skimage.measure

    lab = _lab()
    posData = SimpleNamespace(rp=skimage.measure.regionprops(lab))
    window = SimpleNamespace(
        data=[posData], pos_i=0, currentLab2D=lab, isSegm3D=False,
        z_lab=Mock(return_value=None),
    )
    _bind(
        window, 'getObjectsLab2D', 'isObjInCurrSlice', 'getObjSlice',
        'getObjImage'
    )

    objsLab = window.getObjectsLab2D([1, 3])
    expected = np.where(np.isin(lab, [1, 3]), lab, 0)
    assert np.array_equal(objsLab, expected)

    assert not window.getObjectsLab2D([]).any()
    assert not window.getObjectsLab2D([99]).any()


# ---------------------------------------------------------------------------
# Regression tests from code review
# ---------------------------------------------------------------------------

def test_add_selected_cells_no_op_keeps_full_undo_queue():
    lab = _lab()
    window, posData, controls = _make_window(lab, selectedIDs=[1])
    annot = window.getBrushStrokesAnnot(create=True)
    annot[:] = 1
    states = [np.full(lab.shape, i, dtype=np.uint8) for i in range(20)]
    window.brushStrokesUndo[(0, 0)] = deque(
        [(None, state) for state in states], maxlen=20
    )

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    undoQueue = window.brushStrokesUndo[(0, 0)]
    assert len(undoQueue) == 20
    # The oldest undo state was not dropped
    assert undoQueue[0][1] is states[0]


def test_add_selected_cells_outside_added_area_is_refused():
    lab = _lab()
    # Only ID 2 selected, but the zoomed area contains only ID 1
    window, posData, controls = _make_window(
        lab, selectedIDs=[2], only_zoom=True, view_range=((0, 3), (0, 3))
    )

    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    annot = window.brushStrokesAnnotData[0][0]
    assert not annot.any()
    assert not window.brushStrokesUndo.get((0, 0))
    assert window.brushStrokesChangedSinceTraining is False
    controls.resetSelection.assert_not_called()
    assert window.titleLabel.setText.call_args.kwargs['color'] == 'r'


def test_points_only_model_init_hides_select_cells_controls(
        toolbar, controls
    ):
    # `_importInitMagicPromptModel` blocks the toolbar signals while
    # setting the supported prompt types of the new model
    lab = _lab()
    window = _make_select_window(lab, controls, toolbar=toolbar)
    window.magicPromptsToolButton = Mock(isChecked=Mock(return_value=True))
    window.ax1_BrushCircle = Mock()
    window.ax2_BrushCircle = Mock()
    window.setHoverToolSymbolData = Mock()
    _bind(window, 'applyBrushStrokesSelectCellsMode')

    toolbar.setPromptType('brush_stroke')
    toolbar.selectCellsAction.setChecked(True)
    controls.setSelectedIDs([1, 2])
    controls.setVisible(True)

    toolbar.blockSignals(True)
    toolbar.setSupportedPromptTypes(('point',))
    toolbar.blockSignals(False)
    window.applyBrushStrokesSelectCellsMode()

    assert not toolbar.selectCellsAction.isChecked()
    assert not controls.isVisible()
    assert list(controls.selectedIDs) == []
    window.setBrushStrokesSelectedCellsLayerImage.assert_called()


def test_apply_prompt_type_applies_select_cells_mode():
    for isMagicPromptsOn in (False, True):
        window = Mock()
        window.magicPromptsToolButton.isChecked.return_value = (
            isMagicPromptsOn
        )
        window.magicPromptsToolbar.promptType.return_value = 'point'
        gui.guiWin.applyMagicPromptsPromptType(window)
        window.applyBrushStrokesSelectCellsMode.assert_called_once()


def test_shortcut_right_click_menu_only_on_action_button(app):
    # On PyQt5, `QAction.associatedWidgets` also returns the toolbar --> the
    # "Set shortcut" menu must be installed only on the action's button
    toolbarWidget, button = Mock(name='toolbar'), Mock(name='button')

    class PyQt5LikeAction(gui.QAction):
        def associatedWidgets(self):
            return [toolbarWidget, button]

    name = 'Select all cells (magic prompts)'
    action = PyQt5LikeAction('Select all')
    window = SimpleNamespace(
        widgetsForActions={name: button},
        _installRightClickFilter=Mock(),
    )
    _bind(window, '_setupRightClickMenuOnButton')
    # QMenu(self) needs a QWidget parent or None
    original_qmenu = gui.QMenu
    try:
        gui.QMenu = lambda parent=None: original_qmenu()
        isSetup, numWidgets = window._setupRightClickMenuOnButton(
            action, name
        )
    finally:
        gui.QMenu = original_qmenu

    assert isSetup and numWidgets == 1
    installed = [
        call.args[0] for call in window._installRightClickFilter.call_args_list
    ]
    assert installed == [button]
