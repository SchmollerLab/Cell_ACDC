from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from qtpy.QtCore import QItemSelectionModel, QModelIndex, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QTreeView

from cellacdc import apps, measurements, measurements_viewer, widgets


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        'GFP_mean': [2., 4.],
        'GFP_autoBkgr_bkgrVal_median': [1., 1.],
        'GFP_custom_metric': [3., 5.],
        'RFP_mean': [6., 8.],
        'cell_area_pxl': [10., 20.],
        'eccentricity': [0.1, 0.2],
        'unclassified_value': [8., 9.],
        'stage': pd.Categorical(['G1', 'S']),
        'known': [True, False],
    }, index=pd.MultiIndex.from_tuples(
        [(0, 1), (1, 1)], names=['frame_i', 'Cell_ID']
    ))


def leaf_indexes(model, parent=QModelIndex()):
    for row in range(model.rowCount(parent)):
        index = model.index(row, 0, parent)
        if model.hasChildren(index):
            yield from leaf_indexes(model, index)
        else:
            yield index


def item_path(item):
    path = []
    while item is not None:
        path.append(item.text())
        item = item.parent()
    return tuple(reversed(path))


def test_model_uses_same_groups_as_combined_equations(app, dataframe):
    model = widgets.MeasurementTreeModel()
    model.populate(dataframe, ['GFP', 'RFP'], checkable=True)
    actual = {item_path(item) for item in model.measurementItems()}
    expected = {
        (*path, col)
        for path, cols in measurements.measurement_column_groups(
            dataframe, ['GFP', 'RFP']
        ) for col in cols
    }
    assert actual == expected
    assert ('Numerical inputs', 'GFP measurements', 'Cell measurements', 'GFP_mean') in actual
    assert ('Numerical inputs', 'GFP measurements', 'Background values',
            'GFP_autoBkgr_bkgrVal_median') in actual
    assert ('Numerical inputs', 'GFP measurements', 'Custom measurements',
            'GFP_custom_metric') in actual
    assert ('Numerical inputs', 'Size measurements', 'cell_area_pxl') in actual
    assert ('Numerical inputs', 'Region properties', 'eccentricity') in actual
    assert ('Categorical inputs', 'known') in actual
    assert len(actual) == len(dataframe.columns)


def test_recursive_search_matches_flat_fuzzy_search_across_branches(app, dataframe):
    model = widgets.MeasurementTreeModel()
    model.populate(dataframe, ['GFP', 'RFP'])
    proxy = widgets.MeasurementTreeSearchProxy()
    proxy.setSourceModel(model)
    flat = widgets.QStandardItemModel()
    for name in dataframe:
        item = widgets.QStandardItem(name)
        item.setData(name, widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE)
        flat.appendRow(item)
    reference = widgets.ButtonSearchCompleter(
        searchRoles=(widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,),
        tooltipRole=None
    )
    reference.setSourceModel(flat)
    for query in ('GFP', 'RFP', 'eccentricitty', 'stage', 'known', 'zzzzzzzzzz', ''):
        proxy.setFilterText(query)
        reference.setFilterText(query)
        actual = {index.data() for index in leaf_indexes(proxy)}
        expected = {
            reference.index(row, 0).data()
            for row in range(reference.rowCount())
        }
        assert actual == expected


def test_recipe_selects_only_measurement_leaves_across_groups(app, dataframe):
    dialog = apps.MeasurementProcessingRecipeDialog(
        dataframe.columns, acdc_df=dataframe, allChNames=['GFP', 'RFP']
    )
    assert isinstance(dialog.listSelector, QTreeView)
    first, = dialog.listSelector.selectedIndexes()
    assert first.data() == 'GFP_mean'
    group = dialog.measurementsModel.item(0)
    assert not group.isSelectable()
    assert not group.isCheckable()
    dialog.listSelector.clearSelection()
    selected = {'GFP_mean', 'cell_area_pxl', 'known'}
    selection = dialog.listSelector.selectionModel()
    for item in dialog.measurementsModel.measurementItems():
        if item.text() in selected:
            selection.select(
                dialog.measurementsSearchProxy.mapFromSource(item.index()),
                QItemSelectionModel.Select
            )
    dialog.ok_cb()
    assert not dialog.cancel
    assert set(dialog.selectedChannels) == selected
    dialog.deleteLater()


def test_recipe_search_expands_ancestors_and_clears(app, dataframe):
    dialog = apps.MeasurementProcessingRecipeDialog(
        dataframe.columns, acdc_df=dataframe, allChNames=['GFP', 'RFP']
    )
    dialog.listSelector.collapseAll()
    dialog.measurementsSearchLineEdit.setText('RFP_mean')
    proxy = dialog.measurementsSearchProxy
    index, = [idx for idx in leaf_indexes(proxy) if idx.data() == 'RFP_mean']
    parent = index.parent()
    while parent.isValid():
        assert dialog.listSelector.isExpanded(parent)
        parent = parent.parent()
    dialog.measurementsSearchLineEdit.setText('zzzzzzzzzz')
    assert proxy.rowCount() == 0
    dialog.measurementsSearchLineEdit.clear()
    assert len(list(leaf_indexes(proxy))) == len(dataframe.columns)
    dialog.deleteLater()


def test_recipe_group_heading_cannot_be_processed(app, dataframe, monkeypatch):
    dialog = apps.MeasurementProcessingRecipeDialog(
        dataframe.columns, acdc_df=dataframe, allChNames=['GFP', 'RFP']
    )
    dialog.listSelector.clearSelection()
    group_index = dialog.measurementsSearchProxy.index(0, 0)
    dialog.listSelector.selectionModel().select(
        group_index, QItemSelectionModel.Select
    )
    warning = Mock()
    monkeypatch.setattr(widgets.myMessageBox, 'warning', warning)
    dialog.ok_cb()
    assert dialog.cancel
    warning.assert_called_once()
    dialog.deleteLater()


def test_recipe_name_only_api_and_image_channel_selector_remain_supported(app):
    dialog = apps.MeasurementProcessingRecipeDialog(['area', 'volume', 'intensity'])
    assert {item.text() for item in dialog.measurementsModel.measurementItems()} == {
        'area', 'volume', 'intensity'
    }
    selected, = dialog.listSelector.selectedIndexes()
    assert selected.data() in {'area', 'volume', 'intensity'}
    dialog.ok_cb()
    assert dialog.selectedChannels == [selected.data()]
    dialog.deleteLater()
    image = apps.PreProcessRecipeDialogUtil(['phase', 'GFP'])
    assert isinstance(image.listSelector, widgets.listWidget)
    assert image.listSelector.selectedItemsText() == ['phase']
    image.deleteLater()


@pytest.fixture
def viewer(app, dataframe, monkeypatch):
    monkeypatch.setattr(measurements_viewer.MeasurementsViewerWin, 'loadConfiguration', Mock())
    window = measurements_viewer.MeasurementsViewerWin()
    window.acdc_df = dataframe
    window.data = [SimpleNamespace(acdc_df=dataframe, chNames=['GFP', 'RFP'], SizeT=2)]
    window.centralWidget().setEnabled(True)
    window.selectedID = 1
    window.frame_i = 0
    window.updateColumnsList()
    yield window
    window.deleteLater()
    app.processEvents()


def test_viewer_tree_retains_checks_axes_and_colors_on_refresh(viewer):
    assert isinstance(viewer.columnsListWidget, QTreeView)
    assert viewer.columnsListWidget.indentation() == 12
    items = {item.text(): item for item in viewer.columnsModel.measurementItems()}
    assert 'stage' not in items
    assert 'known' in items
    for item in items.values():
        item.setCheckState(Qt.Unchecked)
    items['GFP_mean'].setCheckState(Qt.Checked)
    assert not items['GFP_mean'].icon().isNull()
    viewer.filterColumns('zzzzzzzzzz')
    assert viewer.checkedColumns() == ['GFP_mean']
    viewer.filterColumns('')
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    items['GFP_mean'].setCheckState(Qt.Checked)
    items['RFP_mean'].setCheckState(Qt.Checked)
    assert viewer.measurementsAxes == {'X': 'GFP_mean', 'Y': 'RFP_mean'}
    viewer.updateColumnsList()
    items = {item.text(): item for item in viewer.columnsModel.measurementItems()}
    assert set(viewer.checkedColumns()) == {'GFP_mean', 'RFP_mean'}
    assert items['GFP_mean'].data(Qt.AccessibleDescriptionRole) == 'X'
    assert items['RFP_mean'].data(Qt.AccessibleDescriptionRole) == 'Y'
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    assert viewer.checkedColumns() == ['GFP_mean']


def test_viewer_passes_table_and_channels_to_processing_tree(viewer, monkeypatch):
    def cancel(dialog):
        assert dialog._measurementDataframe is viewer.acdc_df
        assert dialog._measurementChannels == ['GFP', 'RFP']
        paths = {
            item_path(item) for item in dialog.measurementsModel.measurementItems()
        }
        assert ('Numerical inputs', 'GFP measurements', 'Cell measurements',
                'GFP_mean') in paths
        dialog.cancel = True

    monkeypatch.setattr(apps.MeasurementProcessingRecipeDialog, 'exec_', cancel)
    viewer.processMeasurements()


def test_tree_leaf_click_and_keyboard_toggle_but_group_does_not(viewer, app):
    viewer._startupConfigurationPrompted = True
    viewer.show()
    viewer.columnsListWidget.expandAll()
    app.processEvents()
    item = next(item for item in viewer.columnsModel.measurementItems()
                if item.text() == 'GFP_mean')
    index = viewer.columnsSearchProxy.mapFromSource(item.index())
    tree = viewer.columnsListWidget
    tree.scrollTo(index)
    app.processEvents()
    rect = tree.visualRect(index)
    original = item.checkState()
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, pos=rect.center())
    assert item.checkState() != original
    tree.setCurrentIndex(index)
    QTest.keyClick(tree, Qt.Key_Space)
    assert item.checkState() == original
    group = viewer.columnsModel.item(0)
    assert not group.isCheckable()
    before = viewer.checkedColumns()
    group_index = viewer.columnsSearchProxy.mapFromSource(group.index())
    QTest.mouseClick(tree.viewport(), Qt.LeftButton,
                     pos=tree.visualRect(group_index).center())
    assert viewer.checkedColumns() == before
    viewer.hide()
