import configparser

import pytest
from qtpy.QtCore import QItemSelectionModel, QModelIndex
from qtpy.QtWidgets import QApplication

from cellacdc import (
    measurement_processing
)
from cellacdc.measurements_viewer.dialogs import processing_recipe as viewer_dialogs_processing_recipe
from cellacdc.myutils import segm_models
from cellacdc.shared.dialogs import preprocessing as dialogs_preprocessing
from cellacdc.shared.widgets import (
    combo_boxes as widgets_combo_boxes,
    item_views as widgets_item_views,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
    search_widgets as widgets_search_widgets,
)
from qtpy.QtGui import QStandardItem, QStandardItemModel

pytestmark = pytest.mark.viewer


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def leaf_indexes(model, parent=QModelIndex()):
    """Measurement (leaf) indexes of the hierarchical measurement tree."""
    for row in range(model.rowCount(parent)):
        index = model.index(row, 0, parent)
        if model.hasChildren(index):
            yield from leaf_indexes(model, index)
        else:
            yield index


def flat_search_reference(names):
    """Viewer search proxy over a flat model of the measurement names."""
    flat = QStandardItemModel()
    for name in names:
        item = QStandardItem(name)
        item.setData(name, widgets_search_widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE)
        flat.appendRow(item)
    reference = widgets_search_widgets.ButtonSearchCompleter(
        searchRoles=(widgets_search_widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,),
        tooltipRole=None,
    )
    reference.setSourceModel(flat)
    reference.setDynamicSortFilter(True)
    reference.sort(0)
    return flat, reference


def test_processor_signature_provides_typed_parameter_help(dummy_processor):
    processor = measurement_processing.MEASUREMENT_PROCESSING_MAPPER[
        'Smooth measurement'
    ]
    window, = segm_models.get_function_argspec(processor['function'])
    assert window.name == 'window'
    assert window.type is int
    assert window.default == 3
    assert 'Positive number of observations' in window.desc
    assert 'Missing input' in window.desc
    assert window.docstring == processor['function'].__doc__
    dummy = measurement_processing.MEASUREMENT_PROCESSING_MAPPER['Dummy']
    assert segm_models.get_function_argspec(dummy['function']) == []


def test_processing_window_uses_integer_control_and_docstring_help(app, monkeypatch):
    selector = viewer_dialogs_processing_recipe.SignalProcessingSelector()
    defaults = iter([3, 7])

    def accept(dialog):
        control, = dialog.argsWidgets
        assert control.name == 'window'
        assert isinstance(control.widget, widgets_numeric_inputs.SpinBox)
        assert control.widget.value() == next(defaults)
        help_button, = [
            button for button in dialog.findChildren(widgets_push_buttons.infoPushButton)
            if getattr(button, 'param_name', None) == 'window'
        ]
        assert 'Positive number of observations' in help_button.infoText
        control.widget.setValue(7)
        dialog.ok_cb()

    monkeypatch.setattr(dialogs_preprocessing.FunctionParamsDialog, 'exec_', accept)
    assert selector.askSetParams() == {'window': 7}
    assert selector.askSetParams() == {'window': 7}
    assert 'Centered moving average' in selector.htmlInfo()
    selector.deleteLater()


@pytest.mark.parametrize('close_with', ['cancel_button', 'window_close'])
def test_cancelled_params_dialog_restores_recipe_kwargs(
        app, monkeypatch, close_with
    ):
    """Edits are applied live: Cancel/closing must restore the old values."""
    widget = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    selector = widget.stepsWidgets[1]['selector']
    assert selector.currentText() == 'Smooth measurement'
    live = {}

    def edit_and_cancel(dialog):
        control, = dialog.argsWidgets
        control.widget.setValue(11)
        live['window'] = widget.recipe()[0]['kwargs']['window']
        if close_with == 'cancel_button':
            dialog.buttonsLayout.cancelButton.click()
        else:
            dialog.close()

    monkeypatch.setattr(dialogs_preprocessing.FunctionParamsDialog, 'exec_', edit_and_cancel)
    assert selector.askSetParams() is None
    assert live['window'] == 11
    assert widget.stepsWidgets[1]['step_kwargs'] == {'window': 3}
    assert widget.recipe()[0]['kwargs'] == {'window': 3}
    widget.deleteLater()


def test_processing_recipe_roundtrips_typed_parameters(app, dummy_processor):
    original = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    original.stepsWidgets[1]['step_kwargs'] = {'window': 7}
    original.addStep()
    original.stepsWidgets[2]['selector'].setCurrentText('Dummy')
    saved = original.recipeConfigPars('acdc')
    loaded = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    loaded.loadRecipe({section: saved[section] for section in saved.sections()})
    assert loaded.recipe() == original.recipe()
    assert loaded.recipe()[0]['kwargs'] == {'window': 7}
    original.deleteLater()
    loaded.deleteLater()


def test_processing_recipe_roundtrips_bool_parameters(app, tmp_path, monkeypatch):
    """INI recipes store strings: 'False' must not be loaded as True."""
    original = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    selector = original.stepsWidgets[1]['selector']
    selector.setCurrentText('Ramp detection')
    selector.setParams('Ramp detection', {
        'threshold': 0.5, 'gap_frames': 3, 'min_length': 2, 'relative': False
    })
    expected = [{'method': 'Ramp detection', 'kwargs': {
        'threshold': 0.5, 'gap_frames': 3, 'min_length': 2, 'relative': False
    }}]
    assert original.recipe() == expected

    ini_path = tmp_path / 'recipe.ini'
    with open(ini_path, 'w') as ini_file:
        original.recipeConfigPars('acdc').write(ini_file)
    cp = configparser.ConfigParser()
    cp.read(ini_path)
    loaded = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    loaded.loadRecipe({section: cp[section] for section in cp.sections()})
    loaded_recipe = loaded.recipe()
    assert loaded_recipe == expected
    kwargs = loaded_recipe[0]['kwargs']
    assert kwargs['relative'] is False
    assert type(kwargs['threshold']) is float
    assert type(kwargs['gap_frames']) is int

    shown = {}

    def accept(dialog):
        for control in dialog.argsWidgets:
            shown[control.name] = control.valueGetter(control.widget)
        dialog.ok_cb()

    monkeypatch.setattr(dialogs_preprocessing.FunctionParamsDialog, 'exec_', accept)
    loaded.stepsWidgets[1]['selector'].askSetParams()
    assert shown['relative'] is False
    assert shown['threshold'] == 0.5
    original.deleteLater()
    loaded.deleteLater()


def test_processing_recipe_rejects_invalid_bool_value(app, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        widgets_message_box.myMessageBox, 'warning',
        lambda self, parent, title, text, *args, **kwargs: warnings.append(title)
    )
    widget = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    widget.loadRecipe({'acdc.preprocess.step1': {
        'method': 'Ramp detection', 'relative': 'maybe'
    }})
    assert warnings == ['Invalid measurement recipe']
    assert widget.recipe()[0]['method'] != 'Ramp detection'
    widget.deleteLater()


def test_image_processing_keeps_its_own_registry(app, dummy_processor):
    image_selector = widgets_combo_boxes.PreProcessingSelector()
    signal_selector = viewer_dialogs_processing_recipe.SignalProcessingSelector()
    assert image_selector.preprocessMapper is not signal_selector.preprocessMapper
    assert 'Dummy' in signal_selector.preprocessMapper
    assert 'Dummy' not in image_selector.preprocessMapper
    image_selector.deleteLater()
    signal_selector.deleteLater()


def test_measurement_selection_search_matches_viewer_search(app):
    names = ['area', 'nuclear_area', 'volume', 'mean_intensity']
    dialog = viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog(names)
    # The dialog's model is a tree (group headings with measurement leaves):
    # compare its leaves with the viewer search over a flat list of names.
    flat, reference = flat_search_reference(names)  # keep flat model alive
    for query in ('AREA', 'vol', 'intensitty', 'zzzzzzzzzz', ''):
        dialog.measurementsSearchLineEdit.setText(query)
        reference.setFilterText(query)
        actual = [
            index.data()
            for index in leaf_indexes(dialog.measurementsSearchProxy)
        ]
        expected = [
            reference.index(row, 0).data()
            for row in range(reference.rowCount())
        ]
        assert actual == expected
        assert dialog.measurementsHighlightDelegate._filterText == query.strip()
    assert actual == names
    dialog.deleteLater()


def test_measurement_search_clear_and_multiple_selection(app):
    names = ['area', 'volume', 'intensity']
    dialog = viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog(names)
    proxy = dialog.measurementsSearchProxy
    dialog.measurementsSearchLineEdit.setText('zzzzzzzzzz')
    assert proxy.rowCount() == 0
    assert list(leaf_indexes(proxy)) == []
    dialog.measurementsSearchLineEdit.clear()
    assert sorted(index.data() for index in leaf_indexes(proxy)) == sorted(names)
    dialog.listSelector.clearSelection()
    selection = dialog.listSelector.selectionModel()
    for index in leaf_indexes(proxy):
        if index.data() in ('area', 'intensity'):
            selection.select(index, QItemSelectionModel.Select)
    # Processed measurements are appended as new columns, so a suffix that
    # does not overwrite an existing measurement is required.
    dialog.append_new_colname.setText('smooth')
    dialog.ok_cb()
    assert not dialog.cancel
    assert sorted(dialog.selectedChannels) == ['area', 'intensity']
    assert sorted(dialog.saveNames) == ['area_smooth', 'intensity_smooth']
    dialog.deleteLater()


def test_channel_selector_remains_unchanged(app):
    dialog = dialogs_preprocessing.PreProcessRecipeDialogUtil(['phase', 'GFP'])
    assert isinstance(dialog.listSelector, widgets_item_views.listWidget)
    assert dialog.listSelector.selectedItemsText() == ['phase']
    dialog.deleteLater()
