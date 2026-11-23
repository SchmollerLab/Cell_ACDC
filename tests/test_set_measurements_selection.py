"""Measurements selected in `apps.SetMeasurementsDialog` are the only ones
saved (main GUI and "Compute measurements" utility).

User report: selecting only the mean of the mean projections of two channels
and the cell area saved background values and many other columns.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import skimage.io
import skimage.measure
from qtpy.QtWidgets import QApplication

from cellacdc import apps, cli, cca_functions, measurements
from cellacdc.utils import compute as utils_compute

BASENAME = 'exp_s01_'
CHANNELS = ['phase', 'GFP', 'mCherry']
FLUO = ['GFP', 'mCherry']
WANTED_CH = {ch: [f'{ch}_mean_meanProj'] for ch in FLUO}
WANTED_SIZE = ['cell_area_pxl']
WANTED = {*WANTED_SIZE, *(c for cols in WANTED_CH.values() for c in cols)}


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def make_experiment(root, SizeT=2, SizeZ=3, size=32):
    images_path = os.path.join(root, 'exp', 'Position_1', 'Images')
    os.makedirs(images_path)
    rng = np.random.default_rng(0)
    segm = np.zeros((SizeT, size, size), dtype=np.uint32)
    segm[:, 4:12, 4:12] = 1
    segm[:, 16:26, 6:14] = 2
    for ch in CHANNELS:
        img = rng.integers(100, 200, size=(SizeT, SizeZ, size, size))
        img = img + 300*(segm[:, None] > 0)
        skimage.io.imsave(
            os.path.join(images_path, f'{BASENAME}{ch}.tif'),
            img.astype(np.uint16), check_contrast=False
        )
    np.savez_compressed(
        os.path.join(images_path, f'{BASENAME}segm.npz'), segm
    )
    pd.DataFrame({
        'Description': [
            'SizeT', 'SizeZ', 'basename', 'PhysicalSizeX', 'PhysicalSizeY',
            'PhysicalSizeZ', 'TimeIncrement'
        ],
        'values': [SizeT, SizeZ, BASENAME, 0.1, 0.1, 0.3, 60],
    }).to_csv(
        os.path.join(images_path, f'{BASENAME}metadata.csv'), index=False
    )
    return images_path


def img_path_of(images_path):
    return os.path.join(images_path, f'{BASENAME}phase.tif')


class _ComputeWorkerStub:
    """What `kernel.run` uses of `workers.ComputeMetricsWorker`
    (`do_init_metrics=False`: the test builds the dialog)."""
    def __init__(self, kernel):
        self.kernel = kernel
        self.standardMetricsErrors = {}
        self.abort = False
        self.signals = SimpleNamespace(
            initProgressBar=cli.HeadlessSignal(),
            progressBar=cli.HeadlessSignal(),
        )
        saveDataWorker = SimpleNamespace(
            customMetricsCritical=cli.HeadlessSignal(),
            regionPropsCritical=cli.HeadlessSignal()
        )
        self.mainWin = SimpleNamespace(
            gui=SimpleNamespace(saveDataWorker=saveDataWorker)
        )

    def emitSigComputeVolume(self, posData, stop_frame_n):
        # As `computeMeasurmentsUtilWin.computeVolumeRegionprop`
        posData.allData_li = []
        for lab in posData.segm_data[:stop_frame_n]:
            posData.allData_li.append({
                'labels': lab,
                'regionprops': skimage.measure.regionprops(lab)
            })
        if 'cell_vol_vox' not in self.kernel.sizeMetricsToSave:
            return
        for data_dict in posData.allData_li:
            for obj in data_dict['regionprops']:
                obj.vol_vox, obj.vol_fl = cca_functions._calc_rot_vol(
                    obj, posData.PhysicalSizeY, posData.PhysicalSizeX
                )


def load_posData(images_path):
    kernel = cli.ComputeMeasurementsKernel(None, '', False)
    return kernel._load_posData(img_path_of(images_path), 'segm')


def build_dialog(posData, existing_cols=None, **kwargs):
    # As `utils.compute.computeMeasurmentsUtilWin.initAddMetricsWorker`
    return apps.SetMeasurementsDialog(
        list(posData.chNames), [], posData.SizeZ > 1, posData.isSegm3D,
        posData=posData, allPosData=[posData],
        addDoNotAskAgainCheckbox=True,
        allPos_acdc_df_cols=existing_cols, acdc_df_path=posData.images_path,
        **kwargs
    )


def kernel_from_dialog(win):
    kernel = cli.ComputeMeasurementsKernel(None, '', False)
    kernel.init_args([gb.chName for gb in win.chNameGroupboxes], 'segm')
    kernel.set_metrics_from_set_measurements_dialog(win)
    return kernel


def run_kernel(images_path, win):
    win.ok_cb()
    kernel = kernel_from_dialog(win)
    posData = kernel._load_posData(img_path_of(images_path), 'segm')
    kernel.run(
        img_path=img_path_of(images_path), stop_frame_n=posData.SizeT,
        end_filename_segm='segm',
        computeMetricsWorker=_ComputeWorkerStub(kernel),
        do_init_metrics=False, posData=posData
    )
    acdc_csv = [
        f for f in os.listdir(images_path) if f.endswith('acdc_output.csv')
    ][0]
    return kernel, pd.read_csv(os.path.join(images_path, acdc_csv))


def check_only(checkboxes, wanted):
    for checkbox in checkboxes:
        checkbox.setChecked(checkbox.text() in wanted)


def select_individually(win):
    """Untick every unwanted checkbox and the unwanted groups."""
    for gb in win.chNameGroupboxes:
        if gb.chName not in FLUO:
            gb.setChecked(False)
            continue
        check_only(gb.checkBoxes, WANTED_CH[gb.chName])
    check_only(win.sizeMetricsQGBox.checkBoxes, WANTED_SIZE)
    win.regionPropsQGBox.setChecked(False)


def select_background_group_first(win):
    """Untick the "Background values" groups while the amounts are still
    ticked (their medians are required and stay ticked), then deselect the
    standard measurements."""
    for gb in win.chNameGroupboxes:
        if gb.chName not in FLUO:
            gb.setChecked(False)
            continue
        # The required medians stay ticked ("Background value required")
        gb.doNotWarn = True
        gb.bkgrValsQGBox.setChecked(False)
        gb.doNotWarn = False
        gb.metricsQGBox.checkAll(None, False)
        check_only(gb.metricsQGBox.checkBoxes, WANTED_CH[gb.chName])
        customMetricsQGBox = getattr(gb, 'customMetricsQGBox', None)
        if customMetricsQGBox is not None:
            customMetricsQGBox.setChecked(False)
    check_only(win.sizeMetricsQGBox.checkBoxes, WANTED_SIZE)
    win.regionPropsQGBox.setChecked(False)


def select_with_deselect_all(win):
    win.selectAll(None, False)
    for gb in win.chNameGroupboxes:
        if gb.chName not in FLUO:
            gb.setChecked(False)
            continue
        for checkbox in gb.checkBoxes:
            if checkbox.text() in WANTED_CH[gb.chName]:
                checkbox.setChecked(True)
    for checkbox in win.sizeMetricsQGBox.checkBoxes:
        if checkbox.text() in WANTED_SIZE:
            checkbox.setChecked(True)


SCENARIOS = {
    'individually': select_individually,
    'background_group_first': select_background_group_first,
    'deselect_all': select_with_deselect_all,
}


def selected_by_kernel(kernel):
    selected = {c for cols in kernel.metricsToSave.values() for c in cols}
    selected.update(kernel.sizeMetricsToSave)
    selected.update(kernel.regionPropsToSave)
    selected.update(getattr(kernel, 'chIndipendCustomMetricsToSave', ()))
    return selected


@pytest.fixture
def experiment(tmp_path):
    images_path = make_experiment(str(tmp_path))
    return images_path, load_posData(images_path)


# ------------------------------- classifier -------------------------------
def test_unselected_measurement_columns_recognises_measurements_only():
    colnames = [
        'Cell_ID', 'cell_cycle_stage', 'is_cell_dead', 'x_centroid',
        'my_annotation', 'GFP_mean_meanProj', 'GFP_mean_maxProj',
        'GFP_autoBkgr_bkgrVal_median_meanProj', 'GFP_amount_autoBkgr_zSlice',
        'GFP_concentration_autoBkgr_from_vol_fl_meanProj',
        'GFP_mean_zSlice2', 'mCherry_mean_zSlice1',
        'GFP_mean_meanProj_smoothed', 'phase_sum_meanProj',
        'cell_area_pxl', 'cell_area_um2', 'cell_area_pxl_zslice1',
        'velocity_pixel', 'area', 'centroid', 'centroid-0', 'moments-1-2',
        'GFP_my_combined', 'not_a_prop-0',
    ]
    unselected = measurements.unselected_measurement_columns(
        colnames, ['phase', 'GFP', 'mCherry'],
        ['GFP_mean_meanProj', 'GFP_mean_zSlice', 'mCherry_mean_zSlice',
         'cell_area_pxl', 'centroid'],
        other_measurement_cols=['GFP_my_combined'],
        zslice_channels=['mCherry'],
    )
    assert unselected == [
        'GFP_mean_maxProj', 'GFP_autoBkgr_bkgrVal_median_meanProj',
        'GFP_amount_autoBkgr_zSlice',
        'GFP_concentration_autoBkgr_from_vol_fl_meanProj',
        # z-slice columns of a channel without "Calculate for each z-slice"
        'GFP_mean_zSlice2',
        'phase_sum_meanProj', 'cell_area_um2',
        # size z-slice columns without "Calculate for each z-slice"
        'cell_area_pxl_zslice1',
        'velocity_pixel', 'area', 'moments-1-2', 'GFP_my_combined',
    ]


# --------------------------- dialog -> kernel ------------------------------
@pytest.mark.parametrize('scenario', list(SCENARIOS))
def test_kernel_computes_only_the_selection(app, experiment, scenario):
    images_path, posData = experiment
    win = build_dialog(posData)
    SCENARIOS[scenario](win)
    assert set(win.selectedMeasurements()) == WANTED
    win.ok_cb()
    kernel = kernel_from_dialog(win)
    assert selected_by_kernel(kernel) == WANTED
    assert kernel.chNamesToSkip == ['phase']
    win.deleteLater()


def test_required_background_medians_are_released(app, experiment):
    _, posData = experiment
    win = build_dialog(posData)
    gb = next(g for g in win.chNameGroupboxes if g.chName == 'GFP')
    median = next(
        c for c in gb.bkgrValsQGBox.checkBoxes
        if c.text() == 'GFP_autoBkgr_bkgrVal_median_meanProj'
    )
    amount = next(
        c for c in gb.metricsQGBox.checkBoxes
        if c.text() == 'GFP_amount_autoBkgr_meanProj'
    )
    # Unticking a required median is refused, but remembered
    gb.doNotWarn = True
    median.setChecked(False)
    assert median.isChecked()
    assert 'GFP_autoBkgr_bkgrVal_median_meanProj' in gb.selectedMetrics()
    amount.setChecked(False)
    assert not median.isChecked()
    # Re-ticking the amount ticks the median again (required)
    amount.setChecked(True)
    assert median.isChecked()
    win.deleteLater()


# ------------------------- saved acdc_output -------------------------------
def test_fresh_table_has_only_the_selection(app, experiment):
    images_path, posData = experiment
    win = build_dialog(posData, existing_cols=[])
    select_individually(win)
    kernel, df = run_kernel(images_path, win)
    channels = list(posData.chNames)
    assert WANTED.issubset(df.columns)
    extra = measurements.unselected_measurement_columns(
        df.columns, channels, WANTED
    )
    # Before: cell_area_um2, cell_vol_vox and cell_vol_fl were always saved
    assert extra == []
    win.deleteLater()


@pytest.mark.parametrize('delete', [True, False])
def test_existing_unselected_measurements(
        app, experiment, monkeypatch, delete
    ):
    images_path, posData = experiment
    # First save with the default selection (everything)
    win0 = build_dialog(posData, existing_cols=[])
    _, df_full = run_kernel(images_path, win0)
    win0.deleteLater()
    assert any('_bkgrVal_' in col for col in df_full.columns)

    asked = []
    def answer(parent, colnames, acdc_df_path=None):
        asked.append(list(colnames))
        return False, delete
    monkeypatch.setattr(apps, 'askDeleteUncheckedExistingMeasurements', answer)

    posData = load_posData(images_path)
    existing = list(posData.acdc_df.columns)
    win = build_dialog(posData, existing_cols=existing)
    select_background_group_first(win)
    kernel, df = run_kernel(images_path, win)
    win.deleteLater()

    unselected_before = measurements.unselected_measurement_columns(
        existing, list(posData.chNames), WANTED
    )
    assert asked == [unselected_before]
    assert kernel.delete_unselected_existing_measurements is delete
    unselected_after = measurements.unselected_measurement_columns(
        df.columns, list(posData.chNames), WANTED
    )
    if delete:
        assert unselected_after == []
    else:
        assert set(unselected_after) == set(unselected_before)
    assert WANTED.issubset(df.columns)
    # Non-measurement columns are never dropped
    for col in ('is_cell_dead', 'is_cell_excluded', 'x_centroid'):
        assert col in df.columns


def test_utility_passes_existing_columns(app, experiment, monkeypatch):
    _, posData = experiment
    created = {}
    original = apps.SetMeasurementsDialog

    def capture(*args, **kwargs):
        created.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(apps, 'SetMeasurementsDialog', capture)
    UtilWin = utils_compute.computeMeasurmentsUtilWin
    host = SimpleNamespace(
        doNotAskAgain=False,
        allPosData=[posData],
        _getAllChNames=UtilWin._getAllChNames,
        _getAllAcdcDfColumns=UtilWin._getAllAcdcDfColumns,
        addCombineMetric=lambda *args: None,
        askSaveObjectsCount=lambda *args: None,
        cancelWorkerMeasurementsWin=lambda *args: None,
    )
    inputs = [
        {'chNames': list(posData.chNames), 'acdcDfColumns': ['a', 'b']},
        {'chNames': list(posData.chNames), 'acdcDfColumns': ['b', 'c']},
    ]
    UtilWin.initAddMetricsWorker(host, posData, inputs)
    assert created['allPos_acdc_df_cols'] == ['a', 'b', 'c']
    host.measurementsWin.hide()
    host.measurementsWin.deleteLater()


# ------------------------- dialog state helpers ----------------------------
def test_restore_state_keeps_unchecked_groups_unchecked(app, experiment):
    _, posData = experiment
    win = build_dialog(posData)
    select_individually(win)
    state = win.state()
    win.deleteLater()

    # As after "Add combined measurement..." (dialog rebuilt + restoreState)
    win2 = build_dialog(posData)
    win2.restoreState(state)
    assert set(win2.selectedMeasurements()) == WANTED
    assert not win2.regionPropsQGBox.isChecked()
    win2.deleteLater()


def test_set_state_with_new_measurement(app, experiment):
    _, posData = experiment
    win = build_dialog(posData)
    select_individually(win)
    state = win.state()
    win.deleteLater()

    title = 'GFP metrics'
    state[title].pop('GFP_mean_maxProj')  # e.g. added after saving the state
    win2 = build_dialog(posData, state=state)
    assert WANTED.issubset(win2.selectedMeasurements())
    win2.deleteLater()


def test_selection_mapper_round_trip(app, experiment):
    _, posData = experiment
    win = build_dialog(posData)
    select_background_group_first(win)
    mapper = win.currentSelectionMapper()
    assert 'GFP_autoBkgr_bkgrVal_median_meanProj' not in mapper['GFP']
    win.deleteLater()

    win2 = build_dialog(posData)
    win2.setCurrentSelectionFromMapper(mapper)
    assert set(win2.selectedMeasurements()) == WANTED
    win2.deleteLater()


def test_combined_measurement_inputs_are_released(app, experiment):
    _, posData = experiment
    posData.addEquationCombineMetrics(
        'GFP_mean_meanProj/mCherry_sum_meanProj', 'GFP_over_mCherry', True
    )
    win = build_dialog(posData)
    mixed = win.mixedChannelsCombineMetricsQGBox
    assert mixed is not None
    gb = next(g for g in win.chNameGroupboxes if g.chName == 'mCherry')
    mCherry_sum = next(
        c for c in gb.checkBoxes if c.text() == 'mCherry_sum_meanProj'
    )
    assert mCherry_sum.isChecked() and not mCherry_sum.isEnabled()
    # Deselect all also releases the inputs of the combined measurement
    win.selectAll(None, False)
    assert not mCherry_sum.isChecked() and mCherry_sum.isEnabled()
    assert win.selectedMeasurements() == []
    win.deleteLater()


# ------------------------- review follow-ups -------------------------------
def test_vs_suffix_is_stripped_only_for_custom_metrics():
    colnames = [
        'GFP_mean_meanProj', 'GFP_mean_vs_mCherry', 'GFP_mean_vs_RFP',
        'cell_area_pxl',
    ]
    unselected = measurements.unselected_measurement_columns(
        colnames, ['GFP', 'RFP', 'mCherry'],
        ['GFP_mean_meanProj', 'cell_area_pxl', 'GFP_mean_vs_RFP'],
        other_measurement_cols=['GFP_mean_vs_RFP'],
    )
    # 'GFP_mean_vs_mCherry' is not a Cell-ACDC measurement (e.g. added by 
    # another tool) and 'GFP_mean_vs_RFP' is a selected combined measurement
    assert unselected == []


def test_linked_acdc_df_columns(tmp_path):
    from cellacdc import load
    pd.DataFrame({'frame_i': [0], 'Cell_ID': [1], 'GFP_mean': [2.0]}).to_csv(
        tmp_path / 'b_acdc_output_run1.csv', index=False
    )
    columns = load.get_linked_acdc_df_columns(
        str(tmp_path), 'b_', 'segm_run1.npz'
    )
    assert columns == ['frame_i', 'Cell_ID', 'GFP_mean']
    assert load.get_linked_acdc_df_columns(str(tmp_path), 'b_', 'segm') == []


def test_inputs_of_selected_concentration_are_computed(
        app, experiment, monkeypatch
    ):
    images_path, posData = experiment
    win0 = build_dialog(posData, existing_cols=[])
    run_kernel(images_path, win0)
    win0.deleteLater()

    monkeypatch.setattr(
        apps, 'askDeleteUncheckedExistingMeasurements', 
        lambda *args, **kwargs: (False, True)
    )
    posData = load_posData(images_path)
    win = build_dialog(posData, existing_cols=list(posData.acdc_df.columns))
    select_individually(win)
    conc = 'GFP_concentration_autoBkgr_from_vol_fl_meanProj'
    gb = next(g for g in win.chNameGroupboxes if g.chName == 'GFP')
    next(c for c in gb.checkBoxes if c.text() == conc).setChecked(True)
    kernel, df = run_kernel(images_path, win)
    win.deleteLater()
    
    # The amount (input) was deleted, recomputed and dropped again
    assert 'GFP_amount_autoBkgr_meanProj' not in df.columns
    assert df[conc].notna().all()


def test_restore_keeps_combined_inputs_and_new_measurements(app, experiment):
    _, posData = experiment
    posData.addEquationCombineMetrics(
        'GFP_mean_meanProj/mCherry_sum_meanProj', 'GFP_over_mCherry', True
    )
    win = build_dialog(posData)
    win.selectAll(None, False)
    select_individually(win)
    state = win.state()
    win.deleteLater()

    # Combined measurement unchecked --> its inputs stay unchecked
    win2 = build_dialog(posData, state=state)
    assert set(win2.selectedMeasurements()) == WANTED
    win2.deleteLater()

    # A channel combined measurement added after the state was saved keeps 
    # its default (checked) state, as before
    posData.addEquationCombineMetrics(
        'GFP_mean_meanProj*2', 'GFP_double', False
    )
    win3 = build_dialog(posData)
    win3.restoreState(state)
    assert set(win3.selectedMeasurements()) == {*WANTED, 'GFP_double'}
    win3.deleteLater()


def test_combined_input_not_computed_keeps_existing_values(
        app, experiment, monkeypatch
    ):
    images_path, posData = experiment
    win0 = build_dialog(posData, existing_cols=[])
    run_kernel(images_path, win0)
    win0.deleteLater()

    # Annotation column (not a measurement) used by a combined measurement
    acdc_csv = [
        f for f in os.listdir(images_path) if f.endswith('acdc_output.csv')
    ][0]
    acdc_path = os.path.join(images_path, acdc_csv)
    df0 = pd.read_csv(acdc_path)
    df0['GFP_positive'] = np.arange(len(df0)) % 2
    df0.to_csv(acdc_path, index=False)
    posData = load_posData(images_path)
    posData.addEquationCombineMetrics(
        'GFP_mean_meanProj*GFP_positive', 'GFP_pos_mean', False
    )
    posData.saveCombineMetrics()

    monkeypatch.setattr(
        apps, 'askDeleteUncheckedExistingMeasurements', 
        lambda *args, **kwargs: (False, True)
    )
    win = build_dialog(posData, existing_cols=list(posData.acdc_df.columns))
    select_individually(win)
    gb = next(g for g in win.chNameGroupboxes if g.chName == 'GFP')
    next(c for c in gb.checkBoxes if c.text() == 'GFP_pos_mean').setChecked(True)
    kernel, df = run_kernel(images_path, win)
    win.deleteLater()

    assert 'GFP_positive' not in kernel._helper_metrics
    assert (df['GFP_positive'].values == df0['GFP_positive'].values).all()
    expected = df['GFP_mean_meanProj']*df['GFP_positive']
    assert np.allclose(df['GFP_pos_mean'], expected)


def test_forced_input_kept_when_its_group_is_unchecked(app, experiment):
    _, posData = experiment
    posData.addEquationCombineMetrics(
        'GFP_mean_meanProj*area', 'GFP_times_area', True
    )
    win = build_dialog(posData)
    area = next(
        c for c in win.regionPropsQGBox.checkBoxes if c.text() == 'area'
    )
    assert area.forcedByCombined
    win.regionPropsQGBox.setChecked(False)
    assert area.isChecked()
    assert 'area' in win.selectedMeasurements()
    win.deleteLater()


def test_deleted_combined_measurement_is_not_selected(app, experiment):
    _, posData = experiment
    posData.addEquationCombineMetrics('GFP_mean_meanProj*2', 'GFP_double', False)
    win = build_dialog(posData)
    gb = next(g for g in win.chNameGroupboxes if g.chName == 'GFP')
    double = next(c for c in gb.checkBoxes if c.text() == 'GFP_double')
    # As the delete button (`onDelClicked` + `delMixedChannelCombineMetric`)
    double.setChecked(False)
    double.hide()
    win.selectAll(None, True)
    assert not double.isChecked()
    assert 'GFP_double' not in win.selectedMeasurements()
    win.deleteLater()
