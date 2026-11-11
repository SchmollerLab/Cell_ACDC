"""Background intensity utility (Utilities > Misc): `cellacdc.background_values`,
`workers.ComputeBackgroundValuesWorker` and the launcher wiring.

The reference formulas are the ones of the FUCCI analysis scripts the utility
replicates: per-frame `np.mean(img, axis=(1, 2), where=segm == 0)` and pooled
`np.percentile(values, [5, 95])` of the cell means (minus the constant
background).
"""
import logging
from types import SimpleNamespace
import os

import numpy as np
import pandas as pd
import pytest
import skimage.io
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QMainWindow, QMenu

from cellacdc import _main, apps, background_values as bv, workers
from cellacdc.info import utilsInfo
from cellacdc.utils import computeBackgroundValues as utilsComputeBackgroundValues


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _synthetic_data(num_frames=5, size=24, seed=0):
    rng = np.random.default_rng(seed)
    segm = np.zeros((num_frames, size, size), dtype=np.uint32)
    segm[:, 3:8, 3:8] = 1
    segm[:, 12:18, 10:16] = 2
    gfp = rng.normal(100, 5, segm.shape) + 3*np.arange(num_frames)[:, None, None]
    gfp[segm > 0] += 80
    cherry = rng.normal(200, 10, segm.shape)
    cherry[segm == 2] += 150
    return segm, {'GFP': gfp, 'mCherry': cherry}


def test_background_matches_fucci_reference():
    segm, images = _synthetic_data()
    df = bv.compute_background_values(images, segm, exclude_zero_pixels=False)
    assert df['frame_i'].tolist() == list(range(len(segm)))
    assert (df['num_objects'] == 2).all()
    for channel, img in images.items():
        expected_mean = np.mean(img, axis=(1, 2), where=segm == 0)
        expected_median = [np.median(img[t][segm[t] == 0]) for t in range(len(img))]
        np.testing.assert_allclose(df[f'{channel}_bkgr_mean'], expected_mean)
        np.testing.assert_allclose(df[f'{channel}_bkgr_median'], expected_median)
        assert (df[f'{channel}_bkgr_num_pixels'] == (segm[0] == 0).sum()).all()


def test_zero_pixels_empty_frames_and_frame_mismatch():
    segm, images = _synthetic_data(num_frames=4)
    img = images['GFP'][:3].copy()  # fewer image frames than masks
    img[:, 0, :] = 0  # alignment padding
    segm[1] = 0  # frame without segmented objects
    df = bv.compute_background_values({'GFP': img}, segm)
    assert len(df) == 3
    assert np.isnan(df.loc[1, 'GFP_bkgr_mean'])
    assert np.isnan(df.loc[1, 'GFP_bkgr_median'])
    mask = (segm[0] == 0) & (img[0] != 0)
    assert df.loc[0, 'GFP_bkgr_mean'] == pytest.approx(img[0][mask].mean())
    assert df.loc[0, 'GFP_bkgr_num_pixels'] == mask.sum()

    df_all = bv.compute_background_values(
        {'GFP': img}, segm, exclude_zero_pixels=False
    )
    assert df_all.loc[0, 'GFP_bkgr_mean'] < df.loc[0, 'GFP_bkgr_mean']


def test_2D_masks_applied_to_every_z_slice_and_shape_errors():
    segm, images = _synthetic_data(num_frames=2)
    zstack = np.stack([images['GFP'], images['GFP'] + 10], axis=1)  # T,Z,Y,X
    df = bv.compute_background_values({'GFP': zstack}, segm)
    expected = np.mean(
        zstack, axis=(1, 2, 3), where=np.broadcast_to(
            (segm == 0)[:, None], zstack.shape
        )
    )
    np.testing.assert_allclose(df['GFP_bkgr_mean'], expected)

    segm3D = np.repeat(segm[:, None], 2, axis=1)
    df3D = bv.compute_background_values({'GFP': zstack}, segm3D)
    np.testing.assert_allclose(df3D['GFP_bkgr_mean'], expected)

    with pytest.raises(ValueError, match='Shape mismatch'):
        bv.compute_background_values({'GFP': images['GFP'][:, :10]}, segm)
    with pytest.raises(ValueError, match='3D segmentation'):
        bv.compute_background_values({'GFP': images['GFP']}, segm3D)


def test_as_frames_adds_time_axis_only_for_single_frame_data():
    assert bv.as_frames(np.zeros((5, 4)), 1).shape == (1, 5, 4)
    assert bv.as_frames(np.zeros((3, 5, 4)), 1).shape == (1, 3, 5, 4)
    assert bv.as_frames(np.zeros((3, 5, 4)), 3).shape == (3, 5, 4)
    # Time-lapse segmentation saved with a single (squeezed) frame
    assert bv.as_frames(np.zeros((5, 4)), 3).shape == (1, 5, 4)
    assert bv.as_frames(np.zeros((1, 5, 4)), 3).shape == (1, 5, 4)
    # Singleton z-axis of a time-lapse
    assert bv.as_frames(np.zeros((3, 1, 5, 4)), 3).shape == (3, 5, 4)


def test_cell_mean_values_and_normalization_constants():
    bkgr_df = pd.DataFrame({
        'frame_i': [0, 1, 2],
        'GFP_bkgr_mean': [10.0, 20.0, np.nan],
    })
    acdc_df = pd.DataFrame({
        'frame_i': [0, 0, 1, 1, 2, 3],
        'Cell_ID': [1, 2, 1, 2, 1, 1],
        'GFP_mean': [50.0, 70.0, 60.0, 90.0, 55.0, 65.0],
        'is_cell_excluded': [0, 0, 0, 1, 0, 0],
    })
    original = acdc_df.copy()
    source_col, raw = bv.cell_mean_values(acdc_df, 'GFP')
    assert source_col == 'GFP_mean'
    # Excluded cell (90) ignored, table not modified
    assert raw.tolist() == [50.0, 70.0, 60.0, 55.0, 65.0]
    pd.testing.assert_frame_equal(acdc_df, original)
    assert bv.cell_mean_values(acdc_df, 'mCherry')[0] is None

    included = bv.included_cells_mask(acdc_df)
    assert included.tolist() == [True, True, True, False, True, True]
    as_strings = acdc_df.assign(
        is_cell_excluded=['False', '0', 0, 'True', 0, 0]
    )
    assert bv.included_cells_mask(as_strings).tolist() == included.tolist()

    summary = bv.normalization_constants(
        {'GFP': [raw]},
        {'GFP': [bkgr_df['GFP_bkgr_mean'].to_numpy()], 'BF': [[1.0, 3.0]]},
        low_percentile=5, high_percentile=95
    )
    assert 'per_frame_min' not in summary.columns
    gfp = summary.set_index('channel').loc['GFP']
    assert gfp['constant_background'] == pytest.approx(15.0)
    assert gfp['constant_min'] == pytest.approx(np.percentile(raw - 15.0, 5))
    assert gfp['constant_max'] == pytest.approx(np.percentile(raw - 15.0, 95))
    assert gfp['num_cell_values'] == raw.size
    # Channel with only background values
    bf = summary.set_index('channel').loc['BF']
    assert bf['constant_background'] == pytest.approx(2.0)
    assert np.isnan(bf['constant_min']) and np.isnan(bf['constant_max'])

    with pytest.raises(ValueError):
        bv.normalization_constants({}, {}, 95, 5)


def test_zstack_mean_columns_are_used():
    """Cell-ACDC saves no `<ch>_mean` for z-stacks: 2D masks use the mean
    projection, 3D masks `<ch>_mean_3D`."""
    acdc_df = pd.DataFrame({
        'frame_i': [0], 'Cell_ID': [1], 'GFP_mean_maxProj': [90.0],
        'GFP_mean_meanProj': [50.0], 'GFP_mean_zSlice': [60.0],
        'GFP_mean_3D': [55.0],
    })
    col, values = bv.cell_mean_values(acdc_df, 'GFP')
    assert col == 'GFP_mean_meanProj' and values.tolist() == [50.0]
    col3D, values3D = bv.cell_mean_values(acdc_df, 'GFP', is_segm_3D=True)
    assert col3D == 'GFP_mean_3D' and values3D.tolist() == [55.0]
    assert bv.cell_mean_values(
        acdc_df.drop(columns='GFP_mean_3D'), 'GFP', is_segm_3D=True
    )[0] is None


def _make_position(exp_path, pos, segm, images, acdc_df):
    images_path = os.path.join(exp_path, pos, 'Images')
    os.makedirs(images_path)
    basename = 'exp_s01_'
    for channel, img in images.items():
        skimage.io.imsave(
            os.path.join(images_path, f'{basename}{channel}.tif'),
            img.astype(np.uint16), check_contrast=False
        )
    np.savez_compressed(
        os.path.join(images_path, f'{basename}segm_complete.npz'), segm
    )
    tracked = segm.copy()
    tracked[tracked == 2] = 0
    np.savez_compressed(os.path.join(images_path, f'{basename}segm.npz'), tracked)
    pd.DataFrame({
        'Description': ['SizeT', 'SizeZ', 'basename'],
        'values': [len(segm), 1, basename],
    }).to_csv(os.path.join(images_path, f'{basename}metadata.csv'), index=False)
    acdc_df.to_csv(
        os.path.join(images_path, f'{basename}acdc_output.csv'), index=False
    )
    return images_path, basename


DEFAULT_PARAMS = {
    'channels': ['GFP', 'mCherry'], 'statistic': 'mean',
    'exclude_zero_pixels': True, 'output_name': 'background_values',
    'use_acdc_output': True, 'low_percentile': 5.0,
    'high_percentile': 95.0,
}


def _run_worker(expPaths, params, monkeypatch):
    """Run the worker synchronously, answering its questions directly."""
    worker = workers.ComputeBackgroundValuesWorker(_UtilStub(expPaths))

    def askParams(expPaths):
        worker.params = params
        return False

    monkeypatch.setattr(worker, 'emitAskParams', askParams)
    monkeypatch.setattr(worker, 'emitSelectSegmFiles', lambda *args: False)
    monkeypatch.setattr(
        worker, 'emitSelectAcdcOutputFiles', lambda *args, **kwargs: False
    )
    summaries, finished, criticals = [], [], []
    worker.sigSummary.connect(lambda df, info: summaries.append((df, info)))
    worker.signals.finished.connect(finished.append)
    worker.signals.critical.connect(criticals.append)
    worker.run()
    return worker, summaries, finished, criticals


class _UtilStub:
    """Stands in for the utility window: answers the worker's questions."""
    def __init__(self, expPaths):
        self.expPaths = expPaths
        self.endFilenameSegm = 'segm_complete'
        self.selectedAcdcOutputEndnames = ['acdc_output']


def test_worker_saves_tables_and_corrects_acdc_output(tmp_path, monkeypatch):
    segm, images = _synthetic_data(num_frames=4)
    images = {ch: np.round(img) for ch, img in images.items()}
    rows = []
    for frame_i in range(len(segm)):
        rows.append({
            'frame_i': frame_i, 'Cell_ID': 1, 'cell_cycle_stage': 'G1',
            'GFP_mean': images['GFP'][frame_i][segm[frame_i] == 1].mean(),
            'mCherry_mean': images['mCherry'][frame_i][segm[frame_i] == 1].mean(),
            'is_cell_excluded': 0,
        })
    acdc_df = pd.DataFrame(rows)

    exp_path = str(tmp_path / 'exp')
    images_path, basename = _make_position(
        exp_path, 'Position_1', segm, images, acdc_df
    )
    # 'mGFP' ends with 'GFP': its file must not be loaded for 'GFP'
    skimage.io.imsave(
        os.path.join(images_path, f'{basename}mGFP.tif'),
        np.full(segm.shape, 7, dtype=np.uint16), check_contrast=False
    )
    params = dict(
        DEFAULT_PARAMS, channels=['GFP', 'mCherry', 'missing_channel']
    )
    worker, summaries, finished, criticals = _run_worker(
        {exp_path: ['Position_1']}, params, monkeypatch
    )

    assert not criticals, criticals
    assert finished == [worker]
    bkgr_df = pd.read_csv(
        os.path.join(images_path, f'{basename}background_values.csv')
    )
    expected = np.mean(images['GFP'], axis=(1, 2), where=segm == 0)
    np.testing.assert_allclose(bkgr_df['GFP_bkgr_mean'], expected, rtol=1e-6)

    # The acdc_output table is only read
    saved = pd.read_csv(os.path.join(images_path, f'{basename}acdc_output.csv'))
    assert list(saved.columns) == list(acdc_df.columns)

    summary_df, info = summaries[0]
    assert info == {'num_positions': 1}
    assert summary_df['channel'].tolist() == ['GFP', 'mCherry']
    gfp = summary_df.set_index('channel').loc['GFP']
    constant = np.median(bkgr_df['GFP_bkgr_mean'])
    assert gfp['constant_background'] == pytest.approx(constant)
    assert gfp['constant_max'] == pytest.approx(
        np.percentile(acdc_df['GFP_mean'], 95) - constant
    )
    saved_summary = pd.read_csv(
        os.path.join(exp_path, 'background_values_summary.csv')
    )
    assert saved_summary['channel'].tolist() == ['GFP', 'mCherry']
    assert (saved_summary['background_statistic'] == 'mean').all()


def test_worker_finishes_cleanly_when_no_position_is_processed(
        tmp_path, monkeypatch
    ):
    """Experiments without a segmentation file are skipped: the run ends
    with an empty summary instead of an error."""
    images_path = tmp_path / 'exp' / 'Position_1' / 'Images'
    images_path.mkdir(parents=True)
    skimage.io.imsave(
        str(images_path / 'exp_s01_GFP.tif'),
        np.zeros((3, 8, 8), dtype=np.uint16), check_contrast=False
    )
    worker, summaries, finished, criticals = _run_worker(
        {str(tmp_path / 'exp'): ['Position_1']}, DEFAULT_PARAMS, monkeypatch
    )
    assert not criticals and finished == [worker]
    assert summaries[0][1] == {'num_positions': 0}
    assert summaries[0][0].empty
    assert not os.path.exists(
        tmp_path / 'exp' / 'background_values_summary.csv'
    )


def test_output_name_cannot_be_the_name_of_another_file(tmp_path):
    segm, images = _synthetic_data(num_frames=2)
    acdc_df = pd.DataFrame({'frame_i': [0], 'Cell_ID': [1], 'GFP_mean': [1.0]})
    exp_path = str(tmp_path / 'exp')
    images_path, basename = _make_position(
        exp_path, 'Position_1', segm, images, acdc_df
    )
    bv.compute_background_values(images, segm).to_csv(
        os.path.join(images_path, f'{basename}background_values.csv'),
        index=False
    )
    stub = _UtilStub({exp_path: ['Position_1']})
    find = utilsComputeBackgroundValues.ComputeBackgroundValuesUtil.findExistingFile
    assert find(stub, 'acdc_output') == f'{basename}acdc_output.csv'
    assert find(stub, 'metadata') == f'{basename}metadata.csv'
    assert find(stub, 'GFP') == f'{basename}GFP.tif'
    assert find(stub, 'segm') == f'{basename}segm.npz'
    # Background table of a previous run can be overwritten
    assert find(stub, 'background_values') is None
    assert find(stub, 'new_name') is None


def test_params_dialog_validates_and_returns_params(app, monkeypatch):
    warnings = []
    taken = {'acdc_output': 'exp_s01_acdc_output.csv'}
    win = apps.BackgroundValuesParamsDialog(
        ['GFP', 'mCherry'], findExistingFile=taken.get
    )
    monkeypatch.setattr(win, '_warn', warnings.append)
    win.highPercentileSpinbox.setValue(2)
    win.ok_cb()
    assert win.cancel and len(warnings) == 1
    # Percentiles are validated also when the correction is unchecked
    win.normalizationGroupbox.setChecked(False)
    win.ok_cb()
    assert win.cancel and len(warnings) == 2
    win.normalizationGroupbox.setChecked(True)

    win.highPercentileSpinbox.setValue(99)
    win.outputNameLineEdit.setText('acdc_output')
    win.ok_cb()
    assert win.cancel and 'exp_s01_acdc_output.csv' in warnings[-1]
    win.outputNameLineEdit.setText('background_values')

    win.highPercentileSpinbox.setValue(99)
    win.channelsListWidget.item(1).setCheckState(Qt.Unchecked)
    win.statisticCombobox.setCurrentText('median')
    win.ok_cb()
    assert not win.cancel
    assert win.params == {
        'channels': ['GFP'], 'statistic': 'median',
        'exclude_zero_pixels': True, 'output_name': 'background_values',
        'use_acdc_output': True, 'low_percentile': 5.0,
        'high_percentile': 99.0,
    }


def test_read_summary_table(tmp_path):
    summary = bv.normalization_constants(
        {'GFP': [np.array([1.0, 2.0, 3.0])]},
        {'GFP': [np.array([0.5])], '561': [np.array([4.0])],
         'NA': [np.array([5.0])]},
    )
    summary.insert(1, 'background_statistic', 'median')
    path = tmp_path / 'background_values_summary.csv'
    summary.to_csv(path, index=False)
    table = bv.read_summary_table(path)
    # Channel names stay text, empty values are NaN
    assert table['channel'].tolist() == ['GFP', '561', 'NA']
    row = table.set_index('channel').loc['561']
    assert row['constant_background'] == 4.0
    assert np.isnan(row['constant_min'])
    assert table['low_percentile'].dtype == float

    summary.iloc[[0, 0]].to_csv(path, index=False)
    with pytest.raises(ValueError, match='more than one row'):
        bv.read_summary_table(path)
    summary.iloc[:0].to_csv(path, index=False)
    with pytest.raises(ValueError, match='no channels'):
        bv.read_summary_table(path)
    pd.DataFrame({'channel': ['GFP'], 'x': [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Missing columns'):
        bv.read_summary_table(path)
    pd.DataFrame({'frame_i': [0], 'num_objects': [1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match='single Position'):
        bv.read_summary_table(path)


def test_params_dialog_presets(app, monkeypatch):
    preset = dict(
        DEFAULT_PARAMS, channels=['mCherry', 'not_in_experiments'],
        statistic='median', exclude_zero_pixels=False, low_percentile=10.0,
        high_percentile=90.0, output_name='bkgr'
    )
    win = apps.BackgroundValuesParamsDialog(
        ['BF', 'GFP', 'mCherry'], presetParams=preset, presetNote='Preset'
    )
    assert win.selectedChannels() == ['mCherry']
    win.ok_cb()
    assert not win.cancel
    assert win.params == dict(preset, channels=['mCherry'])


def test_select_positions_asks_only_positions(app, tmp_path, monkeypatch):
    multi, single = tmp_path / 'multi', tmp_path / 'single'
    for exp, positions in ((multi, 3), (single, 1)):
        for i in range(positions):
            images = exp / f'Position_{i + 1}' / 'Images'
            images.mkdir(parents=True)
            (images / 'exp_GFP.tif').write_bytes(b'')   # not empty
    prompts = []
    answers = [['Position_1', 'Position_3'], None]

    def QtPrompt(self, parent, values, selectedValues=None, **kwargs):
        prompts.append((self.exp_path, len(values), len(selectedValues)))
        answer = answers.pop(0)
        self.cancel = answer is None
        self.selected_pos = answer

    monkeypatch.setattr(
        utilsComputeBackgroundValues.load.select_exp_folder, 'QtPrompt',
        QtPrompt
    )
    result = utilsComputeBackgroundValues.select_positions(
        None, {str(multi): ['Position_2'], str(single): []}
    )
    # Only the experiment with several Positions is asked, with the
    # Positions of the selected folder preselected
    assert prompts == [(str(multi), 3, 1)]
    assert result == {
        str(multi): ['Position_1', 'Position_3'], str(single): ['Position_1']
    }
    # Cancelled
    assert utilsComputeBackgroundValues.select_positions(
        None, {str(multi): []}
    ) is None
    assert prompts[-1] == (str(multi), 3, 3)


def test_worker_cancelled_during_last_position_saves_no_summary(
        tmp_path, monkeypatch
    ):
    segm, images = _synthetic_data(num_frames=2)
    exp_path = str(tmp_path / 'exp')
    _make_position(
        exp_path, 'Position_1', segm, images,
        pd.DataFrame({'frame_i': [0], 'Cell_ID': [1], 'GFP_mean': [1.0]})
    )
    process = workers.ComputeBackgroundValuesWorker._processPosition

    def processThenCancel(self, *args, **kwargs):
        result = process(self, *args, **kwargs)
        self.abort = True   # progress window closed meanwhile
        return result

    monkeypatch.setattr(
        workers.ComputeBackgroundValuesWorker, '_processPosition',
        processThenCancel
    )
    aborted = []
    worker = workers.ComputeBackgroundValuesWorker(
        _UtilStub({exp_path: ['Position_1']})
    )
    worker.sigAborted.connect(lambda: aborted.append(True))

    def askParams(expPaths):
        worker.params = DEFAULT_PARAMS
        return False

    monkeypatch.setattr(worker, 'emitAskParams', askParams)
    monkeypatch.setattr(worker, 'emitSelectSegmFiles', lambda *args: False)
    monkeypatch.setattr(
        worker, 'emitSelectAcdcOutputFiles', lambda *args, **kwargs: False
    )
    summaries = []
    worker.sigSummary.connect(lambda *args: summaries.append(args))
    worker.run()
    assert aborted == [True] and not summaries
    assert not os.path.exists(
        os.path.join(exp_path, 'background_values_summary.csv')
    )


def _patch_util_dialogs(monkeypatch, shown, messages):
    """Answer the questions of the real utility without blocking."""
    Util = utilsComputeBackgroundValues.ComputeBackgroundValuesUtil

    def acceptPresets(dialog):
        shown.append(dialog.selectedChannels())
        dialog.ok_cb()

    def selectSegm(self, exp_path, pos_foldernames):
        self.endFilenameSegm = 'segm_complete'
        self.worker.abort = False
        self.worker.waitCond.wakeAll()

    def selectAcdc(self, exp_path, pos_foldernames, *args, **kwargs):
        self.selectedAcdcOutputEndnames = ['acdc_output']
        self.worker.abort = False
        self.worker.waitCond.wakeAll()

    monkeypatch.setattr(apps.BackgroundValuesParamsDialog, 'exec_', acceptPresets)
    monkeypatch.setattr(Util, 'selectSegmFileLoadData', selectSegm)
    monkeypatch.setattr(Util, 'selectAcdcOutputTables', selectAcdc)
    box = SimpleNamespace(
        information=lambda *a: messages.append(a),
        warning=lambda *a: messages.append(a),
        critical=lambda *a: messages.append(a),
    )
    monkeypatch.setattr(
        utilsComputeBackgroundValues.widgets, 'myMessageBox',
        lambda **kwargs: box
    )
    return Util


def _exec_util(win):
    """exec_ with a safety net; returns the worker thread (or None)."""
    from qtpy.QtCore import QTimer

    def forceClose():
        win._workerDone = True
        win.close()

    # Never block the test session: force the window closed after 30 s
    QTimer.singleShot(30000, forceClose)
    win.exec_()
    thread = win._workerThread()
    if thread is not None:
        try:
            thread.wait(5000)
        except RuntimeError:
            thread = None   # already deleted
    return thread


def test_util_error_closes_modal_window_and_stops_thread(
        app, tmp_path, monkeypatch, _auto_answer_blocking_dialogs
    ):
    # The real worker thread runs: keep the watchdog off the progress window
    _auto_answer_blocking_dialogs.stop()
    segm, images = _synthetic_data(num_frames=2)
    exp_path = str(tmp_path / 'exp')
    _make_position(
        exp_path, 'Position_1', segm, images,
        pd.DataFrame({'frame_i': [0], 'Cell_ID': [1], 'GFP_mean': [1.0]})
    )
    shown, messages = [], []
    Util = _patch_util_dialogs(monkeypatch, shown, messages)

    def fail(*args, **kwargs):
        raise RuntimeError('test failure')

    monkeypatch.setattr(
        workers.ComputeBackgroundValuesWorker, '_processPosition', fail
    )
    criticals = []
    monkeypatch.setattr(
        utilsComputeBackgroundValues.NewThreadMultipleExpBaseUtil,
        'workerCritical', lambda self, error: criticals.append(error)
    )
    win = Util(
        {exp_path: ['Position_1']}, app, 'Compute background intensity',
        'info', 'Computing', presetParams=DEFAULT_PARAMS
    )
    thread = _exec_util(win)
    # Reported once although the worker emits the error twice
    assert len(criticals) == 1 and win.summaryDf is None
    assert thread is None or not thread.isRunning()


def test_util_with_presets_runs_modally_and_returns_summary(
        app, tmp_path, monkeypatch, _auto_answer_blocking_dialogs
    ):
    """Run the real utility (worker thread, preset parameters dialog) as
    the measurements viewer setup does: `exec_` returns when done and
    `summaryDf` holds the summary."""
    # The real worker thread runs: keep the watchdog off the progress window
    _auto_answer_blocking_dialogs.stop()
    segm, images = _synthetic_data(num_frames=3)
    images = {ch: np.round(img) for ch, img in images.items()}
    rows = [{
        'frame_i': frame_i, 'Cell_ID': 1,
        'GFP_mean': images['GFP'][frame_i][segm[frame_i] == 1].mean(),
        'mCherry_mean': images['mCherry'][frame_i][segm[frame_i] == 1].mean(),
    } for frame_i in range(len(segm))]
    exp_path = str(tmp_path / 'exp')
    _make_position(exp_path, 'Position_1', segm, images, pd.DataFrame(rows))

    shown, messages = [], []
    Util = _patch_util_dialogs(monkeypatch, shown, messages)
    win = Util(
        {exp_path: ['Position_1']}, app, 'Compute background intensity',
        'info', 'Computing', presetParams=dict(
            DEFAULT_PARAMS, channels=['GFP'], statistic='median'
        ), presetNote='Preset'
    )
    thread = _exec_util(win)
    assert thread is None or not thread.isRunning()
    assert shown == [['GFP']]
    assert win.summaryDf is not None
    assert win.summaryDf['channel'].tolist() == ['GFP']
    assert win.summaryDf['background_statistic'].tolist() == ['median']
    assert np.isfinite(win.summaryDf['constant_min']).all()
    assert messages and 'completed' in messages[-1][2]


class _MenuHost(QMainWindow):
    """Real launcher actions, menus and launch slot; no-op otherwise."""
    createActions = _main.mainWin.createActions
    connectActions = _main.mainWin.connectActions
    createMenuBar = _main.mainWin.createMenuBar
    launchComputeBackgroundValuesUtil = (
        _main.mainWin.launchComputeBackgroundValuesUtil
    )

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('test_launcher')

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_launcher_misc_menu_launches_utility(app, monkeypatch):
    launched = []

    class _FakeUtil:
        def __init__(self, expPaths, *args, **kwargs):
            launched.append(expPaths)

        def show(self):
            launched.append('shown')

    monkeypatch.setattr(
        _main.utilsComputeBackgroundValues, 'ComputeBackgroundValuesUtil',
        _FakeUtil
    )
    host = _MenuHost()
    host.recentPathsMenu = QMenu()
    host.getSelectedExpPaths = lambda *args, **kwargs: {'exp': ['Position_1']}
    host.createActions()
    host.connectActions()
    host.createMenuBar()

    action = host.computeBackgroundValuesAction
    assert action.text() in utilsInfo
    miscMenus = [
        a.menu() for a in host.utilsMenu.actions()
        if a.menu() is not None and a.menu().title() == 'Misc'
    ]
    assert len(miscMenus) == 1
    assert action in miscMenus[0].actions()

    action.trigger()
    assert launched == [{'exp': ['Position_1']}, 'shown']
    host.deleteLater()
