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

pytestmark = pytest.mark.core

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
        if not self.kernel.needs_rotational_volume(posData):
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
        posData=posData, allPosData=kwargs.pop('allPosData', [posData]),
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
@pytest.mark.smoke
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


@pytest.mark.slow
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
    def answer(parent, colnames, **kwargs):
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
@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


# ------------------------- PR #1318 review follow-ups -----------------------
def add_equations(images_path, channel=(), mixed=()):
    """Save combined measurements `[(name, equation)]` of the Position"""
    posData = load_posData(images_path)
    for name, equation in channel:
        posData.addEquationCombineMetrics(equation, name, False)
    for name, equation in mixed:
        posData.addEquationCombineMetrics(equation, name, True)
    posData.saveCombineMetrics()
    return load_posData(images_path)


def check_names(win, *names):
    checkboxes = [c for gb in win.chNameGroupboxes for c in gb.checkBoxes]
    mixed = win.mixedChannelsCombineMetricsQGBox
    if mixed is not None:
        checkboxes.extend(mixed.checkBoxes)
    for checkbox in checkboxes:
        if checkbox.text() in names:
            checkbox.setChecked(True)


def select_only(win, *names):
    """`select_individually` plus the measurements `names`, without the
    channel independent custom metrics and the unrequested mixed channels
    combined measurements."""
    select_individually(win)
    if win.chIndipendCustomeMetricsQGBox is not None:
        win.chIndipendCustomeMetricsQGBox.setChecked(False)
    mixed = win.mixedChannelsCombineMetricsQGBox
    if mixed is not None:
        for checkbox in mixed.checkBoxes:
            checkbox.setChecked(False)
    check_names(win, *names)


def save_everything_then_answer(images_path, monkeypatch, delete):
    """First save with the default selection, then answer the Delete/Keep
    question with `delete`. Returns the list of the asked columns."""
    win0 = build_dialog(load_posData(images_path), existing_cols=[])
    run_kernel(images_path, win0)
    win0.deleteLater()
    asked = []
    def answer(parent, colnames, **kwargs):
        asked.append(list(colnames))
        return False, delete
    monkeypatch.setattr(apps, 'askDeleteUncheckedExistingMeasurements', answer)
    return asked


def objects_of_first_frame(images_path):
    lab = load_posData(images_path).segm_data[0]
    return {obj.label: obj for obj in skimage.measure.regionprops(lab)}


@pytest.mark.slow
@pytest.mark.parametrize('inp, existing', [
    ('area', 'delete'),
    ('centroid-0', 'fresh'),
    ('cell_vol_fl', 'fresh'),
    ('cell_vol_fl', 'delete'),
])
def test_non_channel_inputs_of_selected_combined_are_computed(
        app, experiment, monkeypatch, inp, existing
    ):
    images_path, _ = experiment
    asked = []
    if existing == 'delete':
        asked = save_everything_then_answer(images_path, monkeypatch, True)
    posData = add_equations(
        images_path, channel=[('GFP_x', f'GFP_mean_meanProj*`{inp}`')]
    )
    existing_cols = (
        list(posData.acdc_df.columns) if existing == 'delete' else []
    )
    win = build_dialog(posData, existing_cols=existing_cols)
    select_only(win, 'GFP_x')
    kernel, df = run_kernel(images_path, win)
    win.deleteLater()

    if existing == 'delete':
        # Listed and deleted, then computed again only for the equation
        assert inp in asked[0]
    assert inp not in df.columns
    objs = objects_of_first_frame(images_path)
    expected = {
        'area': lambda obj: obj.area,
        'centroid-0': lambda obj: obj.centroid[0],
        'cell_vol_fl': lambda obj: cca_functions._calc_rot_vol(
            obj, posData.PhysicalSizeY, posData.PhysicalSizeX
        )[1],
    }[inp]
    frame0 = df[df['frame_i'] == 0].set_index('Cell_ID')
    for ID, row in frame0.iterrows():
        assert np.isclose(
            row['GFP_x'], row['GFP_mean_meanProj']*expected(objs[ID])
        )


@pytest.mark.slow
@pytest.mark.parametrize('existing', ['fresh', 'delete'])
def test_combined_input_of_combined_is_computed(
        app, experiment, monkeypatch, existing
    ):
    images_path, _ = experiment
    # GFP_quad is defined before its input
    equations = [
        ('GFP_quad', 'GFP_double*2'), ('GFP_double', 'GFP_mean_meanProj*2')
    ]
    asked = []
    if existing == 'delete':
        add_equations(images_path, channel=equations)
        asked = save_everything_then_answer(images_path, monkeypatch, True)
    posData = add_equations(images_path, channel=equations)
    existing_cols = (
        list(posData.acdc_df.columns) if existing == 'delete' else []
    )
    win = build_dialog(posData, existing_cols=existing_cols)
    select_only(win, 'GFP_quad')
    kernel, df = run_kernel(images_path, win)
    win.deleteLater()

    if existing == 'delete':
        assert 'GFP_double' in asked[0]
    assert 'GFP_double' in kernel._helper_metrics
    assert 'GFP_double' not in df.columns
    assert np.allclose(df['GFP_quad'], 4*df['GFP_mean_meanProj'])


def test_combined_measurements_are_evaluated_after_their_inputs(
        app, experiment
    ):
    images_path, _ = experiment
    posData = add_equations(
        images_path,
        channel=[
            ('GFP_ratio', 'mCherry_double/GFP_mean_meanProj'),
            ('GFP_r2', 'ratio_gm*2'),
            ('mCherry_double', 'mCherry_mean_meanProj*2'),
        ],
        mixed=[('ratio_gm', 'GFP_mean_meanProj/mCherry_mean_meanProj')]
    )
    win = build_dialog(posData, existing_cols=[])
    select_only(win, 'GFP_ratio', 'GFP_r2', 'mCherry_double', 'ratio_gm')
    _, df = run_kernel(images_path, win)
    win.deleteLater()

    gfp, mcherry = df['GFP_mean_meanProj'], df['mCherry_mean_meanProj']
    assert np.allclose(df['GFP_ratio'], 2*mcherry/gfp)
    assert np.allclose(df['GFP_r2'], 2*gfp/mcherry)


def test_failed_combined_measurement_is_not_saved_as_values():
    kernel = object.__new__(cli.ComputeMeasurementsKernel)
    kernel.metricsToSkip = {}
    kernel.mixedChCombineMetricsToSkip = []
    position = SimpleNamespace(
        loadedChNames=['GFP'],
        combineMetricsConfig={
            'equations': {'GFP_x': 'GFP_missing*2'},
            'user_path_equations': {}, 'mixed_channels_equations': {},
        },
    )
    # Pre-existing (stale) values of the combined measurement
    df = pd.DataFrame({'GFP_mean': [1., 2.], 'GFP_x': [5., 5.]})
    kernel._add_combined_metrics(position, df)
    assert df['GFP_x'].isna().all()


@pytest.mark.slow
def test_volume_input_of_selected_combined_is_computed(app, experiment):
    images_path, _ = experiment
    posData = add_equations(
        images_path, channel=[
            ('GFP_result', 'GFP_concentration_autoBkgr_from_vol_fl_meanProj*2')
        ]
    )
    win = build_dialog(posData, existing_cols=[])
    select_only(win)
    win.ok_cb()
    kernel = kernel_from_dialog(win)
    # Plain selection --> no rotational volume
    assert not kernel.needs_rotational_volume(load_posData(images_path))
    win.deleteLater()

    win = build_dialog(posData, existing_cols=[])
    select_only(win, 'GFP_result')
    kernel, df = run_kernel(images_path, win)
    win.deleteLater()

    assert kernel.needs_rotational_volume(load_posData(images_path))
    assert df['GFP_result'].notna().all()
    for helper in ('cell_vol_fl', 'cell_vol_vox',
            'GFP_concentration_autoBkgr_from_vol_fl_meanProj',
            'GFP_amount_autoBkgr_meanProj'):
        assert helper not in df.columns


def test_size_zslice_columns_only_for_selected_units():
    lab = np.zeros((3, 10, 10), dtype=np.uint32)
    lab[0:2, 2:5, 2:5] = 1
    lab[1:3, 6:9, 6:9] = 2
    rp = skimage.measure.regionprops(lab)

    def zslice_cols(size_metrics):
        df = pd.DataFrame(index=pd.Index([1, 2], name='Cell_ID'))
        df = measurements.add_size_metrics(
            df, rp, size_metrics, True, 0.01, 0.003,
            calc_size_for_each_zslice=True
        )
        return sorted(col for col in df.columns if '_zslice' in col)

    assert zslice_cols(['cell_area_pxl']) == [
        f'cell_area_pxl_zslice{z}' for z in range(3)
    ]
    assert zslice_cols(['cell_vol_vox_3D']) == []


def test_helper_columns_are_dropped():
    kernel = cli.ComputeMeasurementsKernel(None, '', False)
    kernel.sizeMetricsToSave = ['cell_area_pxl']
    kernel.calc_size_for_each_zslice = True
    kernel.metricsToSave = {'GFP': [], 'mCherry': []}
    kernel._helper_metrics = {'GFP_vsmet_zSlice'}
    kernel._helper_custom = {'GFP_vsmet_zSlice'}
    kernel._helper_props = {'centroid'}
    df = pd.DataFrame(columns=[
        'cell_area_pxl_zslice0', 'cell_area_um2_zslice0', 'velocity_um',
        'GFP_vsmet_zSlice_vs_mCherry0', 'GFP_vsmet_zSlice_vs_mCherry',
        'centroid-0', 'centroid-1', 'area', 'cell_vol_fl', 'is_cell_dead'
    ])
    kept = kernel._drop_helper_columns(df, existing_cols={'cell_vol_fl'})
    assert list(kept.columns) == [
        'cell_area_pxl_zslice0', 'area', 'cell_vol_fl', 'is_cell_dead'
    ]


@pytest.mark.parametrize('selected, zslice_channels, expected', [
    (set(), set(), [0, 1, 2, 3]),
    ({'GFP_vsmet_zSlice'}, {'GFP'}, [3]),
    ({'GFP_vsmet_zSlice'}, set(), [1, 2, 3]),
])
def test_per_zslice_vs_custom_metrics_are_measurements(
        monkeypatch, selected, zslice_channels, expected
    ):
    monkeypatch.setattr(
        measurements, '_get_custom_metrics_names', lambda: {'vsmet': 'vsmet'}
    )
    colnames = [
        'GFP_vsmet_zSlice_vs_mCherry', 'GFP_vsmet_zSlice_vs_mCherry0',
        'GFP_vsmet_zSlice_vs_mCherry12', 'GFP_vsmet_meanProj_vs_mCherry',
        # Unknown channel --> not a measurement
        'GFP_vsmet_zSlice_vs_RFP0',
    ]
    unselected = measurements.unselected_measurement_columns(
        colnames, ['GFP', 'mCherry'], selected,
        zslice_channels=zslice_channels
    )
    assert unselected == [colnames[i] for i in expected]


def test_delete_removes_only_listed_columns(app, experiment):
    images_path, posData = experiment
    win = build_dialog(posData)
    select_only(win)
    win.ok_cb()
    kernel = kernel_from_dialog(win)
    win.deleteLater()
    kernel.record_existing_measurements_answer(['GFP_sum_meanProj'], True)
    df = pd.DataFrame(columns=[
        'GFP_sum_meanProj', 'GFP_median_meanProj', 'solidity', 'is_cell_dead'
    ])
    kept = kernel._drop_unselected_existing_measurements(df, posData)
    assert list(kept.columns) == [
        'GFP_median_meanProj', 'solidity', 'is_cell_dead'
    ]

    # Workflow file saved without the list --> every unselected measurement
    kernel.existing_measurements_to_delete = None
    kernel._unselected_existing_cache = None
    kept = kernel._drop_unselected_existing_measurements(df, posData)
    assert list(kept.columns) == ['is_cell_dead']


def test_workflow_file_keeps_the_columns_to_delete(app, experiment, tmp_path):
    from cellacdc import config, load
    _, posData = experiment
    win = build_dialog(posData)
    select_only(win)
    win.ok_cb()
    kernel = kernel_from_dialog(win)
    win.deleteLater()
    kernel.record_existing_measurements_answer(
        ['GFP_sum_meanProj', 'solidity'], True
    )
    configPars = config.ConfigParser()
    configPars['measurements'] = {
        key: str(value) for key, value
        in kernel.to_workflow_config_params().items()
    }
    ini_path = tmp_path / 'workflow.ini'
    with open(ini_path, 'w') as ini:
        configPars.write(ini)
    params = load.read_measurements_workflow_from_config(str(ini_path))
    loaded = cli.ComputeMeasurementsKernel(None, '', True)
    loaded.set_metrics_from_workflow_config_params(params['measurements'])
    assert loaded.existing_measurements_to_delete == {
        'GFP_sum_meanProj', 'solidity'
    }


def test_reused_selection_asks_again_for_columns_not_listed(
        app, experiment, monkeypatch, tmp_path
    ):
    """With "Use the same selection for all following experiment folders",
    a following folder with unselected measurements that were not listed is
    asked again (only about those) and the answer applies only to them."""
    _, posData = experiment
    win = build_dialog(posData)
    select_only(win)
    win.ok_cb()
    kernel = kernel_from_dialog(win)
    win.deleteLater()
    # Answer for the first experiment folder
    kernel.record_existing_measurements_answer(['GFP_sum_meanProj'], True)

    asked = []
    def answer(parent, colnames, **kwargs):
        asked.append((list(colnames), kwargs))
        return False, False
    monkeypatch.setattr(apps, 'askDeleteUncheckedExistingMeasurements', answer)
    exp2 = tmp_path / 'exp2'
    inputs = [
        {
            'file_path': str(exp2 / pos / 'Images' / 'img.tif'),
            'chNames': list(posData.chNames),
            'combineMetricsConfig': None,
            'acdcDfColumns': columns,
        }
        for pos, columns in (
            ('Position_1', ['GFP_sum_meanProj', 'solidity', 'is_cell_dead']),
            ('Position_2', ['GFP_sum_meanProj', 'area']),
        )
    ]
    host = SimpleNamespace(worker=SimpleNamespace(kernel=kernel, abort=False))
    pos = SimpleNamespace(images_path=str(exp2), exp_path=str(exp2))
    UtilWin = utils_compute.computeMeasurmentsUtilWin
    UtilWin.askDeleteUnselectedExistingMeasurements(host, pos, inputs)
    assert len(asked) == 1
    colnames, kwargs = asked[0]
    assert colnames == ['solidity', 'area']
    assert kwargs['locations']['solidity'] == ['Position_1']
    assert kwargs['locations']['area'] == ['Position_2']
    assert kwargs['exp_foldername'] == 'exp2'
    # Keep --> only the column listed for the first folder is deleted
    assert kernel.existing_measurements_to_delete == {'GFP_sum_meanProj'}

    # Same columns in the next folder --> not asked again
    UtilWin.askDeleteUnselectedExistingMeasurements(host, pos, inputs)
    assert len(asked) == 1


def test_dialog_checks_each_position_with_its_combined_measurements(
        app, experiment, monkeypatch
    ):
    import shutil
    images_path, posData = experiment
    pos2_path = os.path.join(
        os.path.dirname(os.path.dirname(images_path)), 'Position_2'
    )
    shutil.copytree(os.path.dirname(images_path), pos2_path)
    pos2 = load_posData(os.path.join(pos2_path, 'Images'))
    # Combined measurements defined only in Position_2
    pos2.addEquationCombineMetrics('GFP_mean_maxProj*2', 'GFP_p2only', False)
    pos2.addEquationCombineMetrics(
        'phase_mean_maxProj*2', 'phase_p2only', False
    )

    asked = []
    def answer(parent, colnames, **kwargs):
        asked.append((list(colnames), kwargs))
        return False, True
    monkeypatch.setattr(apps, 'askDeleteUncheckedExistingMeasurements', answer)
    columns = [
        ['GFP_sum_meanProj', 'is_cell_dead'],
        ['GFP_sum_meanProj', 'GFP_p2only', 'phase_p2only'],
    ]
    win = build_dialog(
        posData, existing_cols=sorted({*columns[0], *columns[1]}),
        allPosData=[posData, pos2], allPos_acdc_df_cols_per_pos=columns
    )
    select_only(win)
    win.ok_cb()
    win.deleteLater()

    colnames, kwargs = asked[0]
    # GFP_p2only is computed again in Position_2 (GFP is selected), the
    # phase channel is not selected
    assert colnames == ['GFP_sum_meanProj', 'phase_p2only']
    assert kwargs['locations'] == {
        'GFP_sum_meanProj': ['Position_1', 'Position_2'],
        'phase_p2only': ['Position_2'],
    }
    assert kwargs['num_positions'] == 2


def test_mixed_measurement_without_checkbox_is_not_saved(app, experiment):
    images_path, _ = experiment
    # 'mix2' reads another mixed measurement --> no checkbox in the dialog
    posData = add_equations(
        images_path, 
        mixed=[('mix1', 'GFP_mean_meanProj*2'), ('mix2', 'mix1*2')]
    )
    win = build_dialog(posData, existing_cols=[])
    offered = [c.text() for c in win.mixedChannelsCombineMetricsQGBox.checkBoxes]
    assert 'mix2' not in offered
    select_only(win)
    _, df = run_kernel(images_path, win)
    win.deleteLater()
    assert 'mix1' not in df.columns
    assert 'mix2' not in df.columns


def test_selected_combined_named_like_a_vs_column_is_saved(app, experiment):
    images_path, _ = experiment
    posData = add_equations(
        images_path, channel=[(
            'GFP_sum_meanProj_vs_mCherry', 
            'GFP_sum_meanProj/mCherry_mean_meanProj'
        )]
    )
    win = build_dialog(posData, existing_cols=[])
    select_only(win, 'GFP_sum_meanProj_vs_mCherry')
    kernel, df = run_kernel(images_path, win)
    win.deleteLater()
    # The input is a helper, the selected combined measurement is saved
    assert 'GFP_sum_meanProj' in kernel._helper_metrics
    assert 'GFP_sum_meanProj' not in df.columns
    assert df['GFP_sum_meanProj_vs_mCherry'].notna().all()
