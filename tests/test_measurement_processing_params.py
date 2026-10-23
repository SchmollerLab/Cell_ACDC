import pytest
from qtpy.QtCore import QItemSelectionModel
from qtpy.QtWidgets import QApplication

from cellacdc import apps, measurement_processing, myutils, widgets


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


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
    assert myutils.get_function_argspec(
        measurement_processing.Dummy.run
    ) == []


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
    original = apps.SignalProcessingParamsWidget()
    original.stepsWidgets[1]['step_kwargs'] = {'window': 7}
    original.addStep()
    original.stepsWidgets[2]['selector'].setCurrentText('Dummy')
    saved = original.recipeConfigPars('acdc')
    loaded = apps.SignalProcessingParamsWidget()
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
    dialog = apps.SignalProcessingRecipeDialog(names)
    reference = widgets.ButtonSearchCompleter(
        searchRoles=(widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,),
        tooltipRole=None,
    )
    reference.setSourceModel(dialog.measurementsModel)
    reference.setDynamicSortFilter(True)
    reference.sort(0)
    for query in ('AREA', 'vol', 'intensitty', 'zzzzzzzzzz', ''):
        dialog.measurementsSearchLineEdit.setText(query)
        reference.setFilterText(query)
        actual = [
            dialog.measurementsSearchProxy.index(row, 0).data()
            for row in range(dialog.measurementsSearchProxy.rowCount())
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
    dialog = apps.SignalProcessingRecipeDialog(['area', 'volume', 'intensity'])
    dialog.measurementsSearchLineEdit.setText('zzzzzzzzzz')
    assert dialog.measurementsSearchProxy.rowCount() == 0
    dialog.measurementsSearchLineEdit.clear()
    assert dialog.measurementsSearchProxy.rowCount() == 3
    dialog.listSelector.clearSelection()
    selection = dialog.listSelector.selectionModel()
    for row in (0, 2):
        selection.select(
            dialog.measurementsSearchProxy.index(row, 0),
            QItemSelectionModel.Select
        )
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.selectedChannels == ['area', 'intensity']
    dialog.deleteLater()


def test_channel_selector_remains_unchanged(app):
    dialog = apps.PreProcessRecipeDialogUtil(['phase', 'GFP'])
    assert isinstance(dialog.listSelector, widgets.listWidget)
    assert dialog.listSelector.selectedItemsText() == ['phase']
    dialog.deleteLater()
