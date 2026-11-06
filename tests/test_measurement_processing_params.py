import pytest
from qtpy.QtCore import QItemSelectionModel, QModelIndex
from qtpy.QtWidgets import QApplication

from cellacdc import (
    apps, apps_measurements_viewer, measurement_processing, myutils, widgets
)


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
    flat = widgets.QStandardItemModel()
    for name in names:
        item = widgets.QStandardItem(name)
        item.setData(name, widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE)
        flat.appendRow(item)
    reference = widgets.ButtonSearchCompleter(
        searchRoles=(widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,),
        tooltipRole=None,
    )
    reference.setSourceModel(flat)
    reference.setDynamicSortFilter(True)
    reference.sort(0)
    return flat, reference


def test_processor_signature_provides_typed_parameter_help():
    processor = measurement_processing.MEASUREMENT_PROCESSING_MAPPER[
        'Smooth measurement'
    ]
    window, = myutils.get_function_argspec(processor['function'])
    assert window.name == 'window'
    assert window.type is int
    assert window.default == 3
    assert 'Positive number of observations' in window.desc
    assert 'Missing input' in window.desc
    assert window.docstring == processor['function'].__doc__
    dummy = measurement_processing.MEASUREMENT_PROCESSING_MAPPER['Dummy']
    assert myutils.get_function_argspec(dummy['function']) == []


def test_processing_window_uses_integer_control_and_docstring_help(app, monkeypatch):
    selector = widgets.SignalProcessingSelector()
    defaults = iter([3, 7])

    def accept(dialog):
        control, = dialog.argsWidgets
        assert control.name == 'window'
        assert isinstance(control.widget, widgets.SpinBox)
        assert control.widget.value() == next(defaults)
        help_button, = [
            button for button in dialog.findChildren(widgets.infoPushButton)
            if getattr(button, 'param_name', None) == 'window'
        ]
        assert 'Positive number of observations' in help_button.infoText
        control.widget.setValue(7)
        dialog.ok_cb()

    monkeypatch.setattr(apps.FunctionParamsDialog, 'exec_', accept)
    assert selector.askSetParams() == {'window': 7}
    assert selector.askSetParams() == {'window': 7}
    assert 'Centered moving average' in selector.htmlInfo()
    selector.deleteLater()


def test_processing_recipe_roundtrips_typed_parameters(app):
    original = apps_measurements_viewer.MeasurementProcessingParamsWidget()
    original.stepsWidgets[1]['step_kwargs'] = {'window': 7}
    original.addStep()
    original.stepsWidgets[2]['selector'].setCurrentText('Dummy')
    saved = original.recipeConfigPars('acdc')
    loaded = apps_measurements_viewer.MeasurementProcessingParamsWidget()
    loaded.loadRecipe({section: saved[section] for section in saved.sections()})
    assert loaded.recipe() == original.recipe()
    assert loaded.recipe()[0]['kwargs'] == {'window': 7}
    original.deleteLater()
    loaded.deleteLater()


def test_image_processing_keeps_its_own_registry(app):
    image_selector = widgets.PreProcessingSelector()
    signal_selector = widgets.SignalProcessingSelector()
    assert image_selector.preprocessMapper is not signal_selector.preprocessMapper
    assert 'Dummy' not in image_selector.preprocessMapper
    image_selector.deleteLater()
    signal_selector.deleteLater()


def test_measurement_selection_search_matches_viewer_search(app):
    names = ['area', 'nuclear_area', 'volume', 'mean_intensity']
    dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(names)
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
    dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(names)
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
    dialog = apps.PreProcessRecipeDialogUtil(['phase', 'GFP'])
    assert isinstance(dialog.listSelector, widgets.listWidget)
    assert dialog.listSelector.selectedItemsText() == ['phase']
    dialog.deleteLater()
