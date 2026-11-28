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
import pandas as pd
import pytest

pytestmark = pytest.mark.gui

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from cellacdc.gui.main_window import (
    keyboard as gui_main_keyboard,
    layout as gui_main_layout,
    menus_actions as gui_main_menus_actions,
    rois as gui_main_rois,
    toolbars as gui_main_toolbars,
    window as gui_window,
)
from cellacdc.gui.widgets import (
    edit_toolbars as gui_widgets_edit_toolbars,
    points_layer_widgets as gui_widgets_points_layer_widgets,
)
from cellacdc.shared.widgets import message_box as widgets_message_box
from qtpy.QtWidgets import QAction, QMenu


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def toolbar(app):
    toolbar = gui_widgets_points_layer_widgets.MagicPromptsToolbar()
    yield toolbar
    toolbar.deleteLater()


@pytest.fixture
def controls(app):
    controls = gui_widgets_edit_toolbars.BrushStrokesSelectCellsToolbar('Select cells')
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

    monkeypatch.setattr(widgets_message_box, 'myMessageBox', FakeMessageBox)
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
    assert isinstance(controls.selectedIDs, gui_widgets_edit_toolbars.KeptObjectIDsList)

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
        func = getattr(gui_window.guiWin, name)
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
        brushStrokesImageSources={},
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
        'getBrushStrokesImageSource', 'getBrushStrokesImageSourceMismatch',
        'recordBrushStrokesImageSource',
        'warnBrushStrokesImageSourceMismatch',
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
        [(None, state, None) for state in states], maxlen=20
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
        gui_window.guiWin.applyMagicPromptsPromptType(window)
        window.applyBrushStrokesSelectCellsMode.assert_called_once()


def test_shortcut_right_click_menu_only_on_action_button(app):
    # On PyQt5, `QAction.associatedWidgets` also returns the toolbar --> the
    # "Set shortcut" menu must be installed only on the action's button
    toolbarWidget, button = Mock(name='toolbar'), Mock(name='button')

    class PyQt5LikeAction(QAction):
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
    original_qmenu = QMenu
    try:
        # moved: QMenu is read in each of these modules
        gui_main_keyboard.QMenu = lambda parent=None: original_qmenu()
        gui_main_layout.QMenu = lambda parent=None: original_qmenu()
        gui_main_menus_actions.QMenu = lambda parent=None: original_qmenu()
        gui_main_rois.QMenu = lambda parent=None: original_qmenu()
        gui_main_toolbars.QMenu = lambda parent=None: original_qmenu()
        isSetup, numWidgets = window._setupRightClickMenuOnButton(
            action, name
        )
    finally:
        # moved: QMenu is read in each of these modules
        gui_main_keyboard.QMenu = original_qmenu
        gui_main_layout.QMenu = original_qmenu
        gui_main_menus_actions.QMenu = original_qmenu
        gui_main_rois.QMenu = original_qmenu
        gui_main_toolbars.QMenu = original_qmenu

    assert isSetup and numWidgets == 1
    installed = [
        call.args[0] for call in window._installRightClickFilter.call_args_list
    ]
    assert installed == [button]


# ---------------------------------------------------------------------------
# 2D brush strokes of z-stack data are paired with the z-slice (or
# z-projection) displayed when they were painted
# ---------------------------------------------------------------------------

def _make_zstack_window(numPos=1, SizeT=3, SizeZ=4, selectedIDs=()):
    """Fake window with z-stack data and 2D segmentation masks. The
    displayed z-slice or z-projection of each frame is stored in
    `segmInfo_df` (like `update_z_slice` and `updateZproj` do)."""
    lab = _lab()
    window, _, controls = _make_window(lab, selectedIDs=selectedIDs)
    rng = np.random.default_rng(0)
    window.data = []
    for pos_i in range(numPos):
        filename = f'pos{pos_i}'
        index = pd.MultiIndex.from_product(
            [[filename], range(SizeT)], names=['filename', 'frame_i']
        )
        segmInfo_df = pd.DataFrame(
            {'z_slice_used_gui': 1, 'which_z_proj_gui': 'single z-slice'},
            index=index
        )
        window.data.append(SimpleNamespace(
            img_data=rng.random((SizeT, SizeZ, *lab.shape)),
            SizeZ=SizeZ, SizeT=SizeT, filename=filename,
            segmInfo_df=segmInfo_df, lab=lab, segm_data=None, frame_i=0,
            IDs=[1, 2, 3],
        ))
    window.brushStrokesDiskMasks = {}
    window.brushStrokesLayerImg1 = Mock()
    window.magicPromptsToolbar = SimpleNamespace(
        brushSizeSpinbox=Mock(value=Mock(return_value=1))
    )
    window.magicPromptsRunModelTask = Mock(return_value={})
    _bind(
        window,
        'brushStrokesMousePress', 'getBrushStrokesPaintTarget',
        'brushStrokesPaintAt', 'brushStrokesEndDrag',
        'getBrushStrokesDiskMask', 'getBrushStrokesAnnotDisplay',
        'getBrushStrokesTrainingItems', 'magicPromptsTrainBrushStrokes',
        'magicPromptsClearBrushStrokes',
    )
    return window, controls


def _display(window, source):
    """Display the z-slice index or z-projection name `source` from the
    current frame on (like `update_z_slice` and `updateZproj`)"""
    posData = window.data[window.pos_i]
    idx = [
        (posData.filename, frame_i)
        for frame_i in range(posData.frame_i, posData.SizeT)
    ]
    if isinstance(source, str):
        posData.segmInfo_df.loc[idx, 'which_z_proj_gui'] = source
    else:
        posData.segmInfo_df.loc[idx, 'which_z_proj_gui'] = 'single z-slice'
        posData.segmInfo_df.loc[idx, 'z_slice_used_gui'] = source


def _paint(window, x, y, left_click=True, modifiers=Qt.NoModifier):
    window.brushStrokesMousePress(_click_event(x, y), left_click, modifiers)
    window.brushStrokesEndDrag()


def _train(window):
    """Train and return {(pos_i, frame_i): image paired with the strokes}"""
    toolbar = SimpleNamespace(
        viewModelParamsAction=Mock(isEnabled=Mock(return_value=True)),
        model=Mock(), model_name='ConvPaint'
    )
    window.magicPromptsRunModelTask.reset_mock()
    assert window.magicPromptsTrainBrushStrokes(toolbar) is True
    prompts_data = window.magicPromptsRunModelTask.call_args.args[3]
    return {
        (pos_i, frame_i): image
        for pos_i, frame_i, annot, image in prompts_data
    }


def test_brush_strokes_2D_zstack_trained_with_painted_zslice():
    window, _ = _make_zstack_window()
    posData = window.data[0]
    _paint(window, 1, 1)
    _paint(window, 3, 3, left_click=False)
    assert window.brushStrokesImageSources == {(0, 0): 1}

    # Scrolling to another z-slice does not change the training image
    _display(window, 3)
    images = _train(window)
    assert np.array_equal(images[(0, 0)], posData.img_data[0, 1])

    # The displayed z-slice is the one segmented
    window.setBrushID = Mock(return_value=1)
    window.startMagicPromptsWorkerAndWait = Mock()
    toolbar = SimpleNamespace(
        model=Mock(), model_name='ConvPaint', model_segment_kwargs={}
    )
    _bind(window, 'magicPromptsSegmentCurrentFrame')
    window.magicPromptsSegmentCurrentFrame(toolbar)
    image = window.startMagicPromptsWorkerAndWait.call_args.args[0]
    assert np.array_equal(image, posData.img_data[0, 3])


def test_brush_strokes_2D_zstack_other_frames_and_positions_keep_image():
    window, _ = _make_zstack_window(numPos=2)
    # Position n. 2, frame n. 3 painted on z-slice index 0 (valid source)
    window.pos_i = 1
    window.data[1].frame_i = 2
    _display(window, 0)
    _paint(window, 1, 1)
    _paint(window, 3, 3, left_click=False)
    # Position n. 1, frame n. 1 painted on the max z-projection
    window.pos_i = 0
    _display(window, 'max z-projection')
    _paint(window, 1, 1)
    _paint(window, 3, 3, left_click=False)
    assert window.brushStrokesImageSources == {
        (1, 2): 0, (0, 0): 'max z-projection'
    }

    # Changed from other frames or positions (e.g., z-slice scrolled with
    # "lock z-slice" on, or `updateZproj` changing the next positions)
    for posData in window.data:
        posData.segmInfo_df['which_z_proj_gui'] = 'single z-slice'
        posData.segmInfo_df['z_slice_used_gui'] = 3

    images = _train(window)
    assert np.array_equal(images[(1, 2)], window.data[1].img_data[2, 0])
    assert np.array_equal(
        images[(0, 0)], window.data[0].img_data[0].max(axis=0)
    )

    # Painted on the max z-projection, z-slice displayed --> refused
    window.titleLabel.reset_mock()
    _paint(window, 4, 4)
    txt = window.titleLabel.setText.call_args.args[0]
    assert 'painted on the max z-projection' in txt
    assert window.brushStrokesAnnotData[0][0][4, 4] == 0


def test_brush_strokes_2D_zstack_painting_other_source_refused():
    window, _ = _make_zstack_window()
    _paint(window, 1, 1)
    annot = window.brushStrokesAnnotData[0][0]
    before = annot.copy()
    assert window.getBrushStrokesAnnotDisplay() is annot

    for source in (2, 'max z-projection'):
        _display(window, source)
        # Brush strokes of another z-slice are hidden
        assert window.getBrushStrokesAnnotDisplay() is None
        window.titleLabel.reset_mock()
        _paint(window, 4, 4)
        _paint(window, 1, 1, modifiers=Qt.ShiftModifier)

        assert np.array_equal(annot, before)
        assert len(window.brushStrokesUndo[(0, 0)]) == 1
        txt = window.titleLabel.setText.call_args.args[0]
        assert 'painted on z-slice n. 2' in txt
        assert window.titleLabel.setText.call_args.kwargs['color'] == 'r'
        assert window.brushStrokesImageSources == {(0, 0): 1}

    # Back to the painted z-slice --> shown and paintable again
    _display(window, 1)
    assert window.getBrushStrokesAnnotDisplay() is annot
    _paint(window, 4, 4)
    assert annot[4, 4] == 2


def test_brush_strokes_2D_zstack_clear_then_paint_other_source_and_undo():
    window, _ = _make_zstack_window()
    posData = window.data[0]
    _paint(window, 1, 1)
    _paint(window, 3, 3, left_click=False)
    painted_z1 = window.brushStrokesAnnotData[0][0].copy()

    # Cleared --> the frame can be painted on another z-slice
    window.magicPromptsClearBrushStrokes(None)
    _display(window, 2)
    _paint(window, 4, 4)
    _paint(window, 5, 0, left_click=False)
    assert window.brushStrokesImageSources == {(0, 0): 2}
    assert np.array_equal(_train(window)[(0, 0)], posData.img_data[0, 2])

    # Undo the strokes painted on z-slice 2 and the clear --> the strokes
    # painted on z-slice 1 are paired with z-slice 1 again
    for _ in range(3):
        assert window.undoBrushStrokes() is True
    assert np.array_equal(window.brushStrokesAnnotData[0][0], painted_z1)
    assert window.brushStrokesImageSources == {(0, 0): 1}
    assert window.getBrushStrokesAnnotDisplay() is None
    assert np.array_equal(_train(window)[(0, 0)], posData.img_data[0, 1])

    # Undo everything --> no image source left
    while window.undoBrushStrokes():
        pass
    assert not window.brushStrokesAnnotData[0][0].any()
    assert window.brushStrokesImageSources == {}


def test_add_selected_cells_2D_zstack_other_source_refused():
    window, controls = _make_zstack_window(selectedIDs=[1])
    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)
    annot = window.brushStrokesAnnotData[0][0]
    assert annot.any()
    assert window.brushStrokesImageSources == {(0, 0): 1}
    before = annot.copy()
    controls.resetSelection.reset_mock()

    _display(window, 'mean z-projection')
    controls.selectedIDs = [2]
    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)

    assert np.array_equal(annot, before)
    assert len(window.brushStrokesUndo[(0, 0)]) == 1
    controls.resetSelection.assert_not_called()
    assert 'z-slice n. 2' in window.titleLabel.setText.call_args.args[0]
    assert window.brushStrokesImageSources == {(0, 0): 1}


def test_brush_stroke_ends_when_zslice_changes_during_drag():
    window, _ = _make_zstack_window()
    window.brushStrokesMousePress(_click_event(1, 1), True, Qt.NoModifier)
    assert window.isBrushStrokeDragImg1

    # z-slice changed with the keyboard while dragging
    _display(window, 2)
    window.brushStrokesPaintAt(5, 4)

    assert not window.isBrushStrokeDragImg1
    assert window.brushStrokesAnnotData[0][0][4, 5] == 0


def test_no_image_source_with_3D_masks_or_2D_data():
    # 3D masks: the brush strokes of each z-slice are painted on it
    lab = np.zeros((4, 6, 8), dtype=np.uint32)
    lab[:] = _lab()
    window, posData, controls = _make_window(lab, selectedIDs=[1], z=2)
    posData.SizeZ = 4
    posData.img_data = np.zeros((1, *lab.shape))
    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)
    assert window.brushStrokesAnnotData[0][0][2].any()
    assert window.brushStrokesImageSources == {}

    # 2D data
    window, posData, controls = _make_window(_lab(), selectedIDs=[1])
    posData.SizeZ = 1
    posData.img_data = np.zeros((1, *_lab().shape))
    window.magicPromptsAddSelectedCellsToBrushStrokes(controls)
    assert window.brushStrokesAnnotData[0][0].any()
    assert window.brushStrokesImageSources == {}


def test_init_global_attr_resets_image_sources():
    window = Mock()
    window.brushStrokesImageSources = {(0, 0): 3}
    gui_window.guiWin.initGlobalAttr(window)
    assert window.brushStrokesImageSources == {}
