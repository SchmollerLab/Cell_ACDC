"""Measurements viewer side of "Ignore frames after the last analyzed frame":
detection of the last analyzed frame of each position, the loading prompt,
the option (menu action, configuration) and its effect on the calculations
(time point and custom annotation predictors, phases, combined measurements,
measurement processing and conditional formatting).

The library parts are tested in ``test_phase_translator_last_frame.py`` and
``test_apply_recipe_last_frame.py``.
"""
import json
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from cellacdc import apps, measurements_viewer
from cellacdc.measurements_viewer import (
    LAST_ANALYZED_CCA, LAST_ANALYZED_TRACKED, MeasurementsViewerWin,
    analysis_last_frame, last_analyzed_frame, update_last_analyzed_frame,
)
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX, TimepointIdentifier

from tests.test_measurement_viewer import make_timelapse_position


SIZE_T = 6
LAST = 2  # cca annotated frames 0-2 of 6


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    """Real viewer that fails on exceptions raised in Qt slots, with the
    blocking prompts answered (the unanalyzed frames prompt by a Mock
    returning True: "Ignore later frames")."""
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    window.slotErrors = errors
    monkeypatch.setattr(
        window, 'askSaveBeforeLoading', Mock(return_value=True)
    )
    monkeypatch.setattr(window, 'measurementProcessingWarning', Mock())
    monkeypatch.setattr(window, 'configurationError', Mock())
    monkeypatch.setattr(
        window, 'selectConfigurationInput',
        lambda config: window.configurationInputOptions(config)[0]
    )
    monkeypatch.setattr(
        window, 'askMeasurementReplay', Mock(return_value='recalculate')
    )
    monkeypatch.setattr(
        window, 'askMaskUnanalyzedFrames', Mock(return_value=True)
    )
    yield window
    window.deleteLater()
    app.processEvents()


def make_position(root, pos=1, SizeT=SIZE_T, cca_frames=range(LAST + 1),
                  extra_cells=()):
    """Time-lapse position (see `make_timelapse_position`) with a ``signal``
    column = 10 * frame_i + Cell_ID - 1, cell cycle annotated on
    `cca_frames` (None: no cell cycle columns)."""
    if cca_frames is not None:
        cca_frames = list(cca_frames)
    images = make_timelapse_position(
        root, pos=pos, SizeT=SizeT, extra_cells=extra_cells,
        cca_frames=cca_frames
    )
    path = os.path.join(images, 'test_acdc_output.csv')
    table = pd.read_csv(path)
    table['signal'] = table['frame_i'] * 10. + table['Cell_ID'] - 1
    table.to_csv(path, index=False)
    return images


def load(viewer, root, mask=True, **kwargs):
    path = make_position(root, **kwargs)
    viewer.askMaskUnanalyzedFrames.return_value = mask
    viewer.loadImagesPaths([path], channel='phase')
    assert viewer.isDataLoaded
    return path


def same_path(a, b):
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(
        os.path.normpath(b)
    )


def cell_values(df, column, ID=1):
    return df.xs(ID, level='Cell_ID')[column].tolist()


def assert_nan_after(values, last=LAST):
    assert not any(pd.isna(value) for value in values[:last + 1]), values
    assert all(pd.isna(value) for value in values[last + 1:]), values


def wait_for_worker(window):
    deadline = time.monotonic() + 10
    while (window.measurementProcessingThread is not None
           and time.monotonic() < deadline):
        QTest.qWait(10)
    assert window.measurementProcessingThread is None


# ---------------------------------------------------------------------------
# 1. last_analyzed_frame and the module helpers
# ---------------------------------------------------------------------------

def cca_table(SizeT=SIZE_T, cca_frames=range(LAST + 1), index=False):
    frames = np.arange(SizeT)
    annotated = np.isin(frames, list(cca_frames))
    table = pd.DataFrame({
        'frame_i': frames, 'Cell_ID': 1, 'area': 1.,
        'cell_cycle_stage': np.where(annotated, 'G1', None),
        'generation_num': np.where(annotated, 1, np.nan),
        'relationship': np.where(annotated, 'mother', None),
    })
    if index:
        table = table.set_index(['frame_i', 'Cell_ID'])
    return table


@pytest.mark.parametrize('index', [False, True])
def test_last_cell_cycle_annotated_frame(index):
    assert last_analyzed_frame(cca_table(index=index)) == (
        LAST, LAST_ANALYZED_CCA
    )


def test_cca_rule_needs_all_cca_columns_filled():
    table = cca_table()
    # One cca column missing on the last annotated frame
    table.loc[table['frame_i'] == LAST, 'generation_num'] = np.nan
    assert last_analyzed_frame(table) == (LAST - 1, LAST_ANALYZED_CCA)


def test_tracked_fallback_without_or_with_empty_cca_columns():
    table = cca_table().drop(
        columns=['cell_cycle_stage', 'generation_num', 'relationship']
    )
    assert last_analyzed_frame(table) == (SIZE_T - 1, LAST_ANALYZED_TRACKED)
    # cca columns present but no annotated row
    table = cca_table(cca_frames=[])
    assert last_analyzed_frame(table) == (SIZE_T - 1, LAST_ANALYZED_TRACKED)
    # Only other cca columns: cell_cycle_stage decides whether the rule applies
    table = cca_table().drop(columns=['cell_cycle_stage'])
    assert last_analyzed_frame(table) == (SIZE_T - 1, LAST_ANALYZED_TRACKED)


def test_missing_or_empty_table_and_clipping():
    assert last_analyzed_frame(None) == (None, None)
    assert last_analyzed_frame(pd.DataFrame()) == (None, None)
    assert last_analyzed_frame(
        pd.DataFrame(columns=['frame_i', 'Cell_ID'])
    ) == (None, None)
    # Table with more frames than the video (e.g. SizeT from metadata)
    assert last_analyzed_frame(cca_table(cca_frames=[]), SizeT=4) == (
        3, LAST_ANALYZED_TRACKED
    )
    assert last_analyzed_frame(cca_table(cca_frames=range(6)), SizeT=3) == (
        2, LAST_ANALYZED_CCA
    )
    # No clipping needed
    assert last_analyzed_frame(cca_table(), SizeT=100) == (
        LAST, LAST_ANALYZED_CCA
    )


def test_update_and_analysis_last_frame():
    posData = SimpleNamespace(acdc_df=cca_table(index=True), SizeT=SIZE_T)
    update_last_analyzed_frame(posData)
    assert posData.lastAnalyzedFrame == LAST
    assert posData.lastAnalyzedKind == LAST_ANALYZED_CCA
    assert analysis_last_frame(posData) == LAST
    assert analysis_last_frame(posData, mask=False) is None
    # Analyzed until the last frame: nothing to ignore
    full = SimpleNamespace(acdc_df=cca_table(cca_frames=range(6)), SizeT=6)
    update_last_analyzed_frame(full)
    assert full.lastAnalyzedFrame == SIZE_T - 1
    assert analysis_last_frame(full) is None
    empty = SimpleNamespace(acdc_df=None, SizeT=SIZE_T)
    update_last_analyzed_frame(empty)
    assert (empty.lastAnalyzedFrame, empty.lastAnalyzedKind) == (None, None)
    assert analysis_last_frame(empty) is None
    assert MeasurementsViewerWin.partiallyAnalyzedPositions(
        [posData, full, empty]
    ) == [posData]


def test_mask_frames_and_blank_unanalyzed_outputs():
    table = cca_table(index=True)
    table['out'] = 1.
    assert measurements_viewer.mask_frames(table, None) is table
    masked = measurements_viewer.mask_frames(table, LAST)
    assert masked.index.get_level_values('frame_i').tolist() == [0, 1, 2]
    flat = measurements_viewer.mask_frames(table.reset_index(), 1)
    assert flat['frame_i'].tolist() == [0, 1]
    changed = measurements_viewer.blank_unanalyzed_outputs(
        table, ['out', 'missing', 'area'], LAST
    )
    assert changed == ['out', 'area']
    assert_nan_after(table['out'].tolist())
    # Already blank: nothing changes
    assert measurements_viewer.blank_unanalyzed_outputs(
        table, ['out'], LAST
    ) == []
    assert measurements_viewer.blank_unanalyzed_outputs(
        table, ['out'], None
    ) == []


def test_evaluate_masked_cell_equation_uses_analyzed_frames():
    table = cca_table(index=True)
    table['signal'] = np.arange(SIZE_T) * 10.
    evaluate = measurements_viewer.evaluate_masked_cell_equation
    unmasked = evaluate(table, 'MAX(signal)', SizeT=SIZE_T)
    assert list(unmasked) == [50.] * SIZE_T
    masked = evaluate(table, 'MAX(signal)', SizeT=SIZE_T, last_frame=LAST)
    assert list(masked) == [20.] * (LAST + 1)
    sizeT = evaluate(
        table, 'signal * 0 + SizeT', SizeT=SIZE_T, last_frame=LAST
    )
    assert list(sizeT) == [LAST + 1.] * (LAST + 1)
    # Aligned on the index when assigned to the full table
    table['peak'] = masked
    assert_nan_after(table['peak'].tolist())
    # No analyzed row at all
    late = table[table.index.get_level_values('frame_i') > LAST]
    empty = evaluate(late, 'signal', SizeT=SIZE_T, last_frame=LAST)
    assert len(empty) == len(late) and empty.isna().all()


# ---------------------------------------------------------------------------
# 2. Loading prompt
# ---------------------------------------------------------------------------

def test_loading_partial_position_prompts_and_masks(viewer, tmp_path):
    path = load(viewer, tmp_path, mask=True)
    viewer.askMaskUnanalyzedFrames.assert_called_once()
    partial, data = viewer.askMaskUnanalyzedFrames.call_args.args
    assert len(partial) == len(data) == 1
    assert same_path(partial[0].images_path, path)
    posData = viewer.posData()
    assert posData is partial[0]
    assert posData.lastAnalyzedFrame == LAST
    assert posData.lastAnalyzedKind == LAST_ANALYZED_CCA
    assert viewer.maskUnanalyzedFrames
    assert viewer.maskUnanalyzedFramesAction.isChecked()
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 3 (later frames ignored)'
    )
    assert viewer.analysisLastFrame() == LAST
    assert viewer.analysisSizeT() == LAST + 1
    analysisDf = viewer.analysisDf(viewer.acdc_df)
    assert analysisDf.index.get_level_values('frame_i').max() == LAST
    assert viewer.dialogAnalysisSizeT(posData) == LAST + 1
    assert viewer.dialogAnalysisSizeT(None) is None
    assert viewer.dialogAnalysisDf(viewer.acdc_df, None) is viewer.acdc_df
    # Images and plots still show every frame
    assert viewer.navigateScrollBar.maximum() == SIZE_T
    assert not viewer.slotErrors


def test_loading_with_use_all_frames(viewer, tmp_path):
    load(viewer, tmp_path, mask=False)
    viewer.askMaskUnanalyzedFrames.assert_called_once()
    assert not viewer.maskUnanalyzedFrames
    assert not viewer.maskUnanalyzedFramesAction.isChecked()
    assert viewer.posData().lastAnalyzedFrame == LAST
    assert viewer.analysisLastFrame() is None
    assert viewer.analysisSizeT() == SIZE_T
    assert viewer.analysisDf(viewer.acdc_df) is viewer.acdc_df
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 3 (all frames used)'
    )
    assert not viewer.slotErrors


def test_cancelling_the_prompt_keeps_loaded_data(viewer, tmp_path):
    full = load(viewer, tmp_path / 'full', cca_frames=range(SIZE_T))
    viewer.askMaskUnanalyzedFrames.assert_not_called()
    before = viewer.data
    viewer.askMaskUnanalyzedFrames.return_value = None
    viewer.loadImagesPaths(
        [make_position(tmp_path / 'partial')], channel='phase'
    )
    viewer.askMaskUnanalyzedFrames.assert_called_once()
    assert viewer.data is before
    assert same_path(viewer.posData().images_path, full)
    assert viewer.maskUnanalyzedFrames
    assert not viewer._isLoadingData
    assert not viewer.slotErrors


def test_cancelling_the_prompt_without_loaded_data(viewer, tmp_path):
    viewer.askMaskUnanalyzedFrames.return_value = None
    viewer.loadImagesPaths([make_position(tmp_path)], channel='phase')
    viewer.askMaskUnanalyzedFrames.assert_called_once()
    assert not viewer.isDataLoaded
    assert not viewer.data


@pytest.mark.parametrize('cca_frames', [range(SIZE_T), None])
def test_fully_analyzed_data_does_not_prompt(viewer, tmp_path, cca_frames):
    load(viewer, tmp_path, cca_frames=cca_frames)
    viewer.askMaskUnanalyzedFrames.assert_not_called()
    posData = viewer.posData()
    assert posData.lastAnalyzedFrame == SIZE_T - 1
    assert posData.lastAnalyzedKind == (
        LAST_ANALYZED_TRACKED if cca_frames is None else LAST_ANALYZED_CCA
    )
    assert viewer.maskUnanalyzedFrames
    assert viewer.analysisLastFrame() is None
    assert viewer.analysisSizeT() == SIZE_T
    assert viewer.lastAnalyzedFrameLabel.text() == ''
    assert not viewer.unanalyzedRegion.isVisible()


def test_partially_tracked_position_without_cca(viewer, tmp_path):
    path = make_position(tmp_path, cca_frames=None)
    tablepath = os.path.join(path, 'test_acdc_output.csv')
    table = pd.read_csv(tablepath)
    table[table['frame_i'] <= 3].to_csv(tablepath, index=False)
    viewer.loadImagesPaths([path], channel='phase')
    viewer.askMaskUnanalyzedFrames.assert_called_once()
    posData = viewer.posData()
    assert posData.lastAnalyzedFrame == 3
    assert posData.lastAnalyzedKind == LAST_ANALYZED_TRACKED
    assert viewer.analysisLastFrame() == 3
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last tracked frame n. = 4 (later frames ignored)'
    )
    assert not viewer.slotErrors


def test_only_partial_positions_are_listed(viewer, tmp_path):
    paths = [
        make_position(tmp_path, pos=1, cca_frames=range(SIZE_T)),
        make_position(tmp_path, pos=2),
    ]
    viewer.loadImagesPaths(paths, channel='phase')
    partial, data = viewer.askMaskUnanalyzedFrames.call_args.args
    assert len(data) == 2
    assert [same_path(pos.images_path, paths[1]) for pos in partial] == [True]
    # The displayed (first) position is analyzed until its last frame
    assert viewer.analysisLastFrame() is None
    assert viewer.lastAnalyzedFrameLabel.text() == ''
    assert viewer.analysisLastFrame(viewer.data[1]) == LAST
    assert viewer.analysisSizeT(viewer.data[1]) == LAST + 1
    assert not viewer.unanalyzedRegion.isVisible()
    # Label and shading follow the displayed position
    viewer.loadPosition(1)
    assert viewer.analysisLastFrame() == LAST
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 3 (later frames ignored)'
    )
    assert viewer.unanalyzedRegion.isVisible()
    viewer.loadPosition(0)
    assert viewer.lastAnalyzedFrameLabel.text() == ''
    assert not viewer.unanalyzedRegion.isVisible()
    assert not viewer.slotErrors


# ---------------------------------------------------------------------------
# 3. Time point and custom annotation predictors
# ---------------------------------------------------------------------------

def add_identifiers(viewer, **equations):
    identifiers = [
        TimepointIdentifier(
            name, (0, 0, 255), shortcut,
            {'type': 'equation', 'equation': equation, 'mode': 'first_true'},
            autoApply=False
        )
        for (name, equation), shortcut in zip(equations.items(), 'QWERT')
    ]
    viewer.setTimepointIdentifiers(identifiers)
    return [identifier.uid for identifier in viewer.timepointIdentifiers]


def predict_timepoints(viewer, uids):
    for uid in uids:
        assert viewer.runTimepointPredictor(uid, force=True)
    return {
        identifier.name: {
            ID: viewer.cellTimepoints(identifier.uid, ID) for ID in (1, 2)
        }
        for identifier in viewer.timepointIdentifiers
    }


@pytest.mark.parametrize('mask', [True, False])
def test_timepoint_predictor_uses_analyzed_frames(viewer, tmp_path, mask):
    # Cell 2 appears only after the last analyzed frame
    load(viewer, tmp_path, mask=mask, extra_cells=[(4, 2), (5, 2)])
    uids = add_identifiers(
        viewer, End='frame_i == SizeT - 1', Peak='signal == MAX(signal)',
        Any='signal >= 0'
    )
    timepoints = predict_timepoints(viewer, uids)
    if mask:
        assert timepoints == {
            'End': {1: [LAST], 2: []},
            'Peak': {1: [LAST], 2: []},
            'Any': {1: [0], 2: []},
        }
    else:
        assert timepoints == {
            'End': {1: [SIZE_T - 1], 2: [SIZE_T - 1]},
            'Peak': {1: [SIZE_T - 1], 2: [SIZE_T - 1]},
            'Any': {1: [0], 2: [4]},
        }
    assert viewer.measurementProcessingWarning.call_count == 0
    assert not viewer.slotErrors


def test_toggling_mask_changes_new_predictions(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    uid, = add_identifiers(viewer, End='frame_i == SizeT - 1')
    assert predict_timepoints(viewer, [uid])['End'][1] == [LAST]
    viewer.maskUnanalyzedFramesAction.trigger()
    assert not viewer.maskUnanalyzedFrames
    assert predict_timepoints(viewer, [uid])['End'][1] == [SIZE_T - 1]


def custom_annotation_state(name, equation, consolidation='all'):
    return {
        'name': name, 'symbolColor': [0, 0, 255, 255], 'autoApply': False,
        'predictor': {
            'type': 'equation', 'equation': equation,
            'consolidation': consolidation
        }
    }


@pytest.mark.parametrize('mask, expected', [(True, 1), (False, 0)])
def test_custom_annotation_consolidated_from_analyzed_frames(
        viewer, tmp_path, mask, expected
    ):
    load(viewer, tmp_path, mask=mask)
    # True on frames 0-2 only: All is true only on the analyzed frames
    viewer.setCustomAnnotations([custom_annotation_state('Early', 'signal < 25')])
    button, = viewer.customAnnotDict
    assert viewer.runCustomAnnotPredictor(button, force=True)
    # The annotation is cell-wide: also on the ignored frames
    assert cell_values(viewer.acdc_df, 'Early') == [expected] * SIZE_T
    viewer.measurementProcessingWarning.assert_not_called()
    assert not viewer.slotErrors


def test_custom_annotation_most_counts_analyzed_frames(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    # True on frames 0-1 (2 of 3 analyzed frames, 2 of 6 frames)
    viewer.setCustomAnnotations([
        custom_annotation_state('Most', 'signal < 15', 'most')
    ])
    button, = viewer.customAnnotDict
    assert viewer.runCustomAnnotPredictor(button, force=True)
    assert cell_values(viewer.acdc_df, 'Most') == [1] * SIZE_T
    viewer.setMaskUnanalyzedFrames(False)
    assert viewer.runCustomAnnotPredictor(button, force=True)
    assert cell_values(viewer.acdc_df, 'Most') == [0] * SIZE_T


# ---------------------------------------------------------------------------
# 4. Phases
# ---------------------------------------------------------------------------

def fallback_rule():
    return {
        'name': 'Fallback', 'predictor': {'type': 'none'},
        'timepoints': [], 'annotations': [], 'phases': ['Other'],
    }


def boundary_rule():
    return {
        'name': 'Cycle', 'predictor': {'type': 'none'},
        'timepoints': [f'{TIMEPOINT_PREFIX}Start'], 'annotations': [],
        'phases': ['Before', 'After'],
    }


@pytest.mark.parametrize('mask', [True, False])
def test_phases_stop_at_last_analyzed_frame(viewer, tmp_path, mask):
    load(viewer, tmp_path, mask=mask)
    assert viewer.applyPhaseTranslator([fallback_rule()])
    phases = cell_values(viewer.acdc_df, 'phase')
    if mask:
        assert phases == ['Other'] * (LAST + 1) + [''] * (SIZE_T - LAST - 1)
    else:
        assert phases == ['Other'] * SIZE_T
    assert viewer.posData().acdc_df is viewer.acdc_df
    assert not viewer.slotErrors


def test_phase_boundary_after_last_frame_is_ignored(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    uid, = add_identifiers(viewer, Start='frame_i == 4')
    # Predicted on all frames: the time point is after the last frame
    viewer.setMaskUnanalyzedFrames(False, update=False)
    assert predict_timepoints(viewer, [uid])['Start'][1] == [4]
    viewer.setMaskUnanalyzedFrames(True, update=False)
    assert viewer.applyPhaseTranslator([boundary_rule()])
    phases = cell_values(viewer.acdc_df, 'phase')
    # The boundary after the last analyzed frame is not seen
    assert phases[LAST + 1:] == [''] * (SIZE_T - LAST - 1)
    assert 'After' not in phases
    viewer.setMaskUnanalyzedFrames(False, update=False)
    assert viewer.applyPhaseTranslator([boundary_rule()])
    assert cell_values(viewer.acdc_df, 'phase') == (
        ['Before'] * 4 + ['After'] * 2
    )


def test_toggling_mask_reapplies_automatic_phases(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    viewer.autoApplyPhases = True
    assert viewer.applyPhaseTranslator([fallback_rule()])
    assert cell_values(viewer.acdc_df, 'phase')[-1] == ''
    viewer.maskUnanalyzedFramesAction.trigger()
    QTest.qWait(20)
    assert cell_values(viewer.acdc_df, 'phase') == ['Other'] * SIZE_T
    viewer.maskUnanalyzedFramesAction.trigger()
    QTest.qWait(20)
    assert cell_values(viewer.acdc_df, 'phase')[LAST + 1:] == (
        [''] * (SIZE_T - LAST - 1)
    )
    assert not viewer.slotErrors


def test_phase_translator_dialog_options_use_analyzed_frames(
        viewer, tmp_path, monkeypatch
    ):
    load(viewer, tmp_path, mask=True)
    options = []

    class FakeDialog:
        def __init__(self, rules, timepoints, annotations, dialogOptions,
                     **kwargs):
            options.append(dialogOptions)
            self.cancel = True
            self.autoApplyCheckbox = Mock()
            self.sigApply = Mock()

        def exec_(self):
            pass

        def deleteLater(self):
            pass

    monkeypatch.setattr(
        measurements_viewer.apps_measurements_viewer,
        'PhaseTranslatorDialog', FakeDialog
    )
    viewer.configurePhaseTranslator()
    viewer.setMaskUnanalyzedFrames(False)
    viewer.configurePhaseTranslator()
    masked, unmasked = options
    assert masked['SizeT'] == LAST + 1
    assert masked['acdc_df'].index.get_level_values('frame_i').max() == LAST
    assert unmasked['SizeT'] == SIZE_T
    assert unmasked['acdc_df'] is viewer.acdc_df


# ---------------------------------------------------------------------------
# 5. Combined measurements and measurement processing
# ---------------------------------------------------------------------------

def accept_formula(monkeypatch, formula, name):
    dialogs = []

    def accept(dialog):
        dialogs.append(dialog)
        dialog.newColNameLineEdit.setText(name)
        dialog.formulaEditWidget.setText(formula)
        dialog.test_cb()
        dialog.ok_cb()

    monkeypatch.setattr(apps.CombineMetricsEquationDialog, 'exec_', accept)
    return dialogs


@pytest.mark.parametrize('mask', [True, False])
def test_combined_measurement_is_nan_after_last_frame(
        viewer, tmp_path, monkeypatch, mask
    ):
    load(viewer, tmp_path, mask=mask)
    dialogs = accept_formula(monkeypatch, 'MAX(signal)', 'peak')
    viewer.combineMeasurements()
    assert dialogs[0].SizeT == (LAST + 1 if mask else SIZE_T)
    viewer.measurementProcessingWarning.assert_not_called()
    peak = cell_values(viewer.acdc_df, 'peak')
    if mask:
        assert peak[:LAST + 1] == [20.] * (LAST + 1)
        assert_nan_after(peak)
    else:
        assert peak == [50.] * SIZE_T
    assert len(viewer.acdc_df) == SIZE_T
    assert viewer.posData().acdc_df is viewer.acdc_df
    assert viewer.combinedMeasurementSteps[-1]['equations'] == {
        'peak': 'MAX(signal)'
    }
    assert not viewer.slotErrors


def steps_configuration():
    return {
        'measurement_processing': [{
            'order': 0, 'columns': ['signal'],
            'recipe': [{'method': 'Dummy', 'kwargs': {}}],
            'output_columns': ['copied']
        }],
        'combined_measurements': [
            {'order': 1, 'equations': {'peak': 'MAX(copied)'}},
            {'order': 2, 'equations': {'sizeT': 'copied * 0 + SizeT'}},
        ],
    }


def frames_table():
    rows = [(frame_i, ID) for frame_i in range(SIZE_T) for ID in (1, 2)]
    table = pd.DataFrame(rows, columns=['frame_i', 'Cell_ID'])
    table['signal'] = table['frame_i'] * 10. + table['Cell_ID'] - 1
    return table.set_index(['frame_i', 'Cell_ID'])


@pytest.mark.parametrize('last_frame', [LAST, None])
def test_replay_measurement_steps_with_last_frame(
        dummy_processor, last_frame
    ):
    posData = SimpleNamespace(acdc_df=frames_table(), SizeT=SIZE_T)
    MeasurementsViewerWin.replayMeasurementSteps(
        posData, steps_configuration(), last_frame=last_frame
    )
    result = posData.acdc_df
    assert len(result) == 2 * SIZE_T
    assert result.index.equals(frames_table().index)
    for ID in (1, 2):
        copied = cell_values(result, 'copied', ID)
        peak = cell_values(result, 'peak', ID)
        sizeT = cell_values(result, 'sizeT', ID)
        signal = cell_values(result, 'signal', ID)
        if last_frame is None:
            assert copied == signal
            assert peak == [signal[-1]] * SIZE_T
            assert sizeT == [float(SIZE_T)] * SIZE_T
        else:
            assert copied[:LAST + 1] == signal[:LAST + 1]
            assert_nan_after(copied)
            assert peak[:LAST + 1] == [signal[LAST]] * (LAST + 1)
            assert_nan_after(peak)
            assert sizeT[:LAST + 1] == [LAST + 1.] * (LAST + 1)
            assert_nan_after(sizeT)


def test_processing_in_viewer_is_nan_after_last_frame(
        viewer, tmp_path, dummy_processor
    ):
    load(viewer, tmp_path, mask=True)
    viewer.startMeasurementsProcessing(
        ['signal'], [{'method': 'Dummy', 'kwargs': {}}], ['signal_copy']
    )
    assert viewer._measurementProcessingTarget[-1] == LAST
    wait_for_worker(viewer)
    viewer.measurementProcessingWarning.assert_not_called()
    copied = cell_values(viewer.acdc_df, 'signal_copy')
    assert copied[:LAST + 1] == [0., 10., 20.]
    assert_nan_after(copied)
    assert len(viewer.acdc_df) == SIZE_T
    assert not viewer.slotErrors


def test_processing_results_stale_when_last_frame_changes(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    posData = viewer.posData()
    viewer._measurementProcessingTarget = (
        posData, getattr(posData, 'acdcDfReloadCount', 0), None, [], LAST
    )
    assert viewer.measurementProcessingTargetTable(posData) is viewer.acdc_df
    viewer.setMaskUnanalyzedFrames(False)
    assert viewer.measurementProcessingTargetTable(posData) is None


# ---------------------------------------------------------------------------
# 6. Conditional formatting
# ---------------------------------------------------------------------------

def test_conditional_formatting_with_last_frame():
    df_ID = frames_table().xs(1, level='Cell_ID')
    rules = [
        {'condition': 'frame_i == SizeT - 1', 'color': '#000000'},
        {'condition': 'signal >= 0', 'color': '#ffffff'},
    ]
    evaluate = MeasurementsViewerWin.evaluateConditionalFormatting
    unmasked = evaluate(df_ID, rules, ID=1, SizeT=SIZE_T)
    assert unmasked.tolist() == [1] * (SIZE_T - 1) + [0]
    masked = evaluate(df_ID, rules, ID=1, SizeT=SIZE_T, last_frame=LAST)
    assert masked.tolist() == [1, 1, 0] + [-1] * (SIZE_T - LAST - 1)
    # Per-cell functions see only the analyzed frames
    peak = [{'condition': 'signal == MAX(signal)', 'color': '#000000'}]
    assert evaluate(
        df_ID, peak, ID=1, SizeT=SIZE_T, last_frame=LAST
    ).tolist() == [-1, -1, 0, -1, -1, -1]
    # No row up to the last frame
    late = df_ID[df_ID.index > LAST]
    assert evaluate(
        late, rules, ID=1, SizeT=SIZE_T, last_frame=LAST
    ).tolist() == [-1] * len(late)


def test_viewer_conditional_formatting_uses_analysis_frames(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    df_ID = viewer.acdc_df.xs(1, level='Cell_ID')
    rules = [{'condition': 'frame_i == SizeT - 1', 'color': '#000000'}]
    assert viewer.cellConditionalFormatting(df_ID, 'signal', rules).tolist() == (
        [-1, -1, 0, -1, -1, -1]
    )
    viewer.setMaskUnanalyzedFrames(False)
    assert viewer.cellConditionalFormatting(df_ID, 'signal', rules).tolist() == (
        [-1] * (SIZE_T - 1) + [0]
    )


# ---------------------------------------------------------------------------
# 7. Configuration
# ---------------------------------------------------------------------------

def timelapse_configuration(path, **extra):
    option = {
        'images_paths': [path], 'channel': 'phase', 'segm_endname': 'segm',
        'overlay_channels': [], 'visible_channels': ['phase']
    }
    configuration = {
        'version': 2, **option, 'input_options': [option],
        'measurements_mode': 0, 'time_course_columns': ['signal'],
        'measurements_axes': {'X': '', 'Y': ''},
        'normalize_measurements': False,
        **steps_configuration(),
    }
    configuration.update(extra)
    return configuration


def write_configuration(tmp_path, configuration):
    filepath = tmp_path / 'workflow.json'
    filepath.write_text(json.dumps(configuration))
    return str(filepath)


@pytest.mark.parametrize('stored', [True, False])
def test_configuration_with_stored_choice_does_not_prompt(
        viewer, tmp_path, dummy_processor, stored
    ):
    path = make_position(tmp_path)
    filepath = write_configuration(tmp_path, timelapse_configuration(
        path, mask_unanalyzed_frames=stored
    ))
    viewer.maskUnanalyzedFrames = not stored
    assert viewer.loadConfiguration(filepath=filepath)
    viewer.configurationError.assert_not_called()
    viewer.askMaskUnanalyzedFrames.assert_not_called()
    assert viewer.maskUnanalyzedFrames is stored
    assert viewer.maskUnanalyzedFramesAction.isChecked() is stored
    assert viewer.configuration()['mask_unanalyzed_frames'] is stored
    assert not viewer.hasUnsavedConfiguration()
    for posData in viewer.data:
        copied = cell_values(posData.acdc_df, 'copied')
        peak = cell_values(posData.acdc_df, 'peak')
        sizeT = cell_values(posData.acdc_df, 'sizeT')
        if stored:
            assert_nan_after(copied)
            assert peak[:LAST + 1] == [20.] * (LAST + 1)
            assert_nan_after(peak)
            assert sizeT[:LAST + 1] == [LAST + 1.] * (LAST + 1)
        else:
            assert copied == [0., 10., 20., 30., 40., 50.]
            assert peak == [50.] * SIZE_T
            assert sizeT == [float(SIZE_T)] * SIZE_T
    assert not viewer.slotErrors


@pytest.mark.parametrize('answer', [True, False])
def test_configuration_without_stored_choice_prompts(
        viewer, tmp_path, dummy_processor, answer
    ):
    path = make_position(tmp_path)
    filepath = write_configuration(tmp_path, timelapse_configuration(path))
    viewer.askMaskUnanalyzedFrames.return_value = answer
    assert viewer.loadConfiguration(filepath=filepath)
    viewer.askMaskUnanalyzedFrames.assert_called_once()
    assert viewer.maskUnanalyzedFrames is answer
    assert viewer.configuration()['mask_unanalyzed_frames'] is answer
    # The answer is not in the file yet
    assert viewer.hasUnsavedConfiguration()
    assert viewer.saveConfiguration()
    assert json.loads(
        open(filepath).read()
    )['mask_unanalyzed_frames'] is answer
    assert not viewer.hasUnsavedConfiguration()


def test_configuration_without_stored_choice_on_full_data(
        viewer, tmp_path, dummy_processor
    ):
    path = make_position(tmp_path, cca_frames=range(SIZE_T))
    filepath = write_configuration(tmp_path, timelapse_configuration(path))
    assert viewer.loadConfiguration(filepath=filepath)
    viewer.askMaskUnanalyzedFrames.assert_not_called()
    assert viewer.maskUnanalyzedFrames
    assert not viewer.hasUnsavedConfiguration()


def test_cancelling_prompt_cancels_configuration_load(
        viewer, tmp_path, dummy_processor
    ):
    path = make_position(tmp_path)
    filepath = write_configuration(tmp_path, timelapse_configuration(path))
    before = viewer.data
    viewer.askMaskUnanalyzedFrames.return_value = None
    assert not viewer.loadConfiguration(filepath=filepath)
    viewer.askMaskUnanalyzedFrames.assert_called_once()
    viewer.askMeasurementReplay.assert_not_called()
    assert viewer.data is before
    viewer.configurationError.assert_not_called()


def test_existing_outputs_are_blanked_after_last_frame(
        viewer, tmp_path, dummy_processor
    ):
    path = make_position(tmp_path)
    tablepath = os.path.join(path, 'test_acdc_output.csv')
    table = pd.read_csv(tablepath)
    for column in ('copied', 'peak', 'sizeT'):
        table[column] = -1.
    table.to_csv(tablepath, index=False)
    filepath = write_configuration(tmp_path, timelapse_configuration(
        path, mask_unanalyzed_frames=True
    ))
    viewer.askMeasurementReplay.return_value = 'existing'
    assert viewer.loadConfiguration(filepath=filepath)
    for column in ('copied', 'peak', 'sizeT'):
        values = cell_values(viewer.acdc_df, column)
        assert values[:LAST + 1] == [-1.] * (LAST + 1)
        assert_nan_after(values)
    # Input columns are not touched
    assert cell_values(viewer.acdc_df, 'signal') == [
        0., 10., 20., 30., 40., 50.
    ]


@pytest.mark.parametrize('value', ['yes', 1, None])
def test_validate_configuration_rejects_non_bool(viewer, tmp_path, value):
    load(viewer, tmp_path)
    configuration = viewer.configuration()
    assert configuration['mask_unanalyzed_frames'] is True
    MeasurementsViewerWin.validateConfiguration(configuration)
    configuration['mask_unanalyzed_frames'] = value
    with pytest.raises(ValueError, match='mask_unanalyzed_frames'):
        MeasurementsViewerWin.validateConfiguration(configuration)
    # Optional (configurations saved before the option existed)
    del configuration['mask_unanalyzed_frames']
    MeasurementsViewerWin.validateConfiguration(configuration)


def test_toggling_the_option_marks_configuration_unsaved(viewer, tmp_path):
    load(viewer, tmp_path)
    assert not viewer.hasUnsavedConfiguration()
    viewer.maskUnanalyzedFramesAction.trigger()
    assert viewer.configuration()['mask_unanalyzed_frames'] is False
    assert viewer.hasUnsavedConfiguration()


# ---------------------------------------------------------------------------
# 8. Toggling the action: label and shading
# ---------------------------------------------------------------------------

def test_toggling_action_updates_label_and_shading(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    regions = (viewer.unanalyzedRegion, viewer.phaseBandUnanalyzedRegion)
    assert all(region.isVisible() for region in regions)
    for region in regions:
        assert region.getRegion() == pytest.approx(
            (LAST + 1.5, SIZE_T + 0.5)
        )
    viewer.maskUnanalyzedFramesAction.trigger()
    assert not viewer.maskUnanalyzedFrames
    assert not viewer.maskUnanalyzedFramesAction.isChecked()
    assert viewer.analysisLastFrame() is None
    assert viewer.analysisSizeT() == SIZE_T
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 3 (all frames used)'
    )
    assert not any(region.isVisible() for region in regions)
    viewer.maskUnanalyzedFramesAction.trigger()
    assert viewer.maskUnanalyzedFrames
    assert viewer.analysisLastFrame() == LAST
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 3 (later frames ignored)'
    )
    assert all(region.isVisible() for region in regions)
    # Hidden in comparison mode
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.plotMeasurements()
    assert not any(region.isVisible() for region in regions)
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    viewer.plotMeasurements()
    assert all(region.isVisible() for region in regions)
    assert not viewer.slotErrors


def test_toggling_is_refused_while_loading(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    viewer._isLoadingData = True
    try:
        viewer.maskUnanalyzedFramesAction.trigger()
    finally:
        viewer._isLoadingData = False
    assert viewer.maskUnanalyzedFrames
    assert viewer.maskUnanalyzedFramesAction.isChecked()


def test_set_option_without_update_keeps_label(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    viewer.setMaskUnanalyzedFrames(False, update=False)
    assert not viewer.maskUnanalyzedFramesAction.isChecked()
    # Not refreshed without `update`
    assert viewer.lastAnalyzedFrameLabel.text().endswith(
        '(later frames ignored)'
    )
    viewer.updateLastAnalyzedFrameLabel()
    assert viewer.lastAnalyzedFrameLabel.text().endswith('(all frames used)')


# ---------------------------------------------------------------------------
# 9. Reloading the table from file
# ---------------------------------------------------------------------------

def rewrite_cca_frames(path, cca_frames):
    tablepath = os.path.join(path, 'test_acdc_output.csv')
    table = pd.read_csv(tablepath)
    annotated = table['frame_i'].isin(list(cca_frames))
    table['cell_cycle_stage'] = np.where(annotated, 'G1', None)
    for col in ('generation_num', 'relative_ID', 'emerg_frame_i',
                'division_frame_i', 'corrected_on_frame_i', 'will_divide'):
        table[col] = np.where(annotated, 1, np.nan)
    table['relationship'] = np.where(annotated, 'mother', None)
    table['is_history_known'] = np.where(annotated, True, None)
    table.to_csv(tablepath, index=False)


def test_reload_updates_last_analyzed_frame(viewer, tmp_path):
    path = load(viewer, tmp_path, mask=True)
    posData = viewer.posData()
    rewrite_cca_frames(path, range(5))
    viewer.reloadAcdcDfFile(posData)
    assert posData.lastAnalyzedFrame == 4
    assert posData.lastAnalyzedKind == LAST_ANALYZED_CCA
    assert viewer.analysisLastFrame(posData) == 4
    assert viewer.analysisSizeT(posData) == 5
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 5 (later frames ignored)'
    )
    # Annotated until the end: all frames are used
    rewrite_cca_frames(path, range(SIZE_T))
    viewer.reloadAcdcDfFile(posData)
    assert posData.lastAnalyzedFrame == SIZE_T - 1
    assert viewer.analysisLastFrame(posData) is None
    assert viewer.lastAnalyzedFrameLabel.text() == ''
    assert not viewer.slotErrors


def test_reload_blanks_recorded_outputs_after_last_frame(viewer, tmp_path):
    path = load(viewer, tmp_path, mask=True)
    posData = viewer.posData()
    viewer.combinedMeasurementSteps.append(
        {'order': 0, 'equations': {'double': 'signal * 2'}}
    )
    tablepath = os.path.join(path, 'test_acdc_output.csv')
    table = pd.read_csv(tablepath)
    table['double'] = table['signal'] * 2
    table.to_csv(tablepath, index=False)
    viewer.reloadAcdcDfFile(posData)
    double = cell_values(posData.acdc_df, 'double')
    assert double[:LAST + 1] == [0., 20., 40.]
    assert_nan_after(double)


# ---------------------------------------------------------------------------
# 10. The prompt itself
# ---------------------------------------------------------------------------

def partial_position(pos, last=4, SizeT=10):
    return SimpleNamespace(
        images_path=os.path.join('exp', f'Position_{pos}', 'Images'),
        lastAnalyzedFrame=last, lastAnalyzedKind=LAST_ANALYZED_CCA,
        SizeT=SizeT
    )


@pytest.mark.parametrize('clicked, cancel, expected', [
    ('Cancel', False, None),
    ('Use all frames', False, False),
    ('Ignore later frames (recommended)', False, True),
    (None, True, None),  # dialog closed
])
def test_ask_mask_unanalyzed_frames_choices(
        app, monkeypatch, clicked, cancel, expected
    ):
    class FakeMessageBox:
        def __init__(self, *args, **kwargs):
            self.cancel = False

        def question(self, parent, title, txt, detailsText='',
                     buttonsTexts=()):
            FakeMessageBox.text = txt
            FakeMessageBox.details = detailsText
            FakeMessageBox.buttons = buttonsTexts
            self.cancel = cancel
            self.clickedButton = clicked
            return buttonsTexts

    monkeypatch.setattr(
        measurements_viewer.widgets, 'myMessageBox', FakeMessageBox
    )
    partial = [partial_position(pos) for pos in range(1, 8)]
    data = [*partial, partial_position(8, last=9)]
    window = SimpleNamespace(
        lastAnalyzedFrameText=MeasurementsViewerWin.lastAnalyzedFrameText
    )
    assert MeasurementsViewerWin.askMaskUnanalyzedFrames(
        window, partial, data
    ) is expected
    assert FakeMessageBox.buttons == (
        'Cancel', 'Use all frames', 'Ignore later frames (recommended)'
    )
    assert '7 of 8 position(s)' in FakeMessageBox.text
    assert '(2 more)' in FakeMessageBox.text
    details = FakeMessageBox.details.splitlines()
    assert len(details) == 7
    assert details[0] == (
        'Position_1: last cell cycle annotated frame n. 5 of 10'
    )


def test_resolve_mask_unanalyzed_frames(viewer, monkeypatch):
    partial = [partial_position(1)]
    full = [partial_position(2, last=9)]
    asked = viewer.askMaskUnanalyzedFrames
    assert viewer.resolveMaskUnanalyzedFrames(full, default=False) == (
        False, False
    )
    assert viewer.resolveMaskUnanalyzedFrames(partial, stored=False) == (
        False, False
    )
    asked.assert_not_called()
    asked.return_value = False
    assert viewer.resolveMaskUnanalyzedFrames(partial) == (False, True)
    asked.return_value = None
    assert viewer.resolveMaskUnanalyzedFrames(partial + full) == (None, True)
    assert asked.call_args.args == (partial, partial + full)
    # A non-bool stored value is not a choice
    asked.return_value = True
    assert viewer.resolveMaskUnanalyzedFrames(partial, stored='yes') == (
        True, True
    )


# ---------------------------------------------------------------------------
# 11. Review fixes: step outputs on newly analyzed frames, unsaved state,
#     dtypes, tool tips, refused toggles
# ---------------------------------------------------------------------------

def set_column(path, column, values_by_frame):
    tablepath = os.path.join(path, 'test_acdc_output.csv')
    table = pd.read_csv(tablepath)
    table[column] = table['frame_i'].map(values_by_frame)
    table.to_csv(tablepath, index=False)


def assert_snapshot_matches(posData, columns):
    """Blanked step outputs match the "as on disk" snapshot (restoring
    NaN read back as 0 is not an edit of the table)."""
    current = posData.acdc_df
    for column in columns:
        pd.testing.assert_series_equal(
            current[column],
            posData.savedAcdcDf[column].reindex(current.index),
            check_dtype=False
        )


def test_reload_keeps_step_outputs_empty_on_newly_analyzed_frames(
        viewer, tmp_path, monkeypatch
    ):
    path = load(viewer, tmp_path, mask=True)
    posData = viewer.posData()
    accept_formula(monkeypatch, 'signal * 2', 'double')
    viewer.combineMeasurements()
    key = measurements_viewer.position_key(posData)
    assert viewer.measurementStepsLastFrames == {key: LAST}
    assert viewer.configuration()['measurement_steps_last_frames'] == {
        key: LAST
    }
    # Saved with NaN after frame 2, read back as 0 (cell cycle columns),
    # then 2 more frames annotated in the main GUI
    set_column(path, 'double', {0: 0., 1: 20., 2: 40., 3: 0., 4: 0., 5: 0.})
    rewrite_cca_frames(path, range(5))
    viewer.reloadAcdcDfFile(posData)
    viewer.refreshAcdcDf()
    assert viewer.analysisLastFrame(posData) == 4
    double = cell_values(posData.acdc_df, 'double')
    assert double[:LAST + 1] == [0., 20., 40.]
    assert_nan_after(double)
    # Still calculated only up to frame 2
    assert viewer.measurementStepsLastFrames == {key: LAST}
    assert_snapshot_matches(posData, ['double'])
    assert not viewer.slotErrors


def test_existing_values_blanked_after_recorded_steps_frame(
        viewer, tmp_path, dummy_processor
    ):
    path = make_position(tmp_path)
    for column in ('copied', 'peak', 'sizeT'):
        set_column(path, column, {frame: -1. for frame in range(SIZE_T)})
    key = os.path.normcase(os.path.abspath(path))
    filepath = write_configuration(tmp_path, timelapse_configuration(
        path, mask_unanalyzed_frames=True,
        measurement_steps_last_frames={key: 1}
    ))
    viewer.askMeasurementReplay.return_value = 'existing'
    assert viewer.loadConfiguration(filepath=filepath)
    for column in ('copied', 'peak', 'sizeT'):
        values = cell_values(viewer.acdc_df, column)
        assert values[:2] == [-1., -1.]
        assert_nan_after(values, last=1)
    assert viewer.measurementStepsLastFrames == {key: 1}
    assert_snapshot_matches(viewer.posData(), ['copied', 'peak', 'sizeT'])
    assert not viewer.hasUnsavedConfiguration()


def test_recalculated_steps_record_their_last_frame(
        viewer, tmp_path, dummy_processor
    ):
    path = make_position(tmp_path)
    filepath = write_configuration(tmp_path, timelapse_configuration(
        path, mask_unanalyzed_frames=True,
        measurement_steps_last_frames={'C:/other/Images': 3}
    ))
    assert viewer.loadConfiguration(filepath=filepath)
    key = os.path.normcase(os.path.abspath(path))
    # Entries of other experiments are kept
    assert viewer.measurementStepsLastFrames == {
        'C:/other/Images': 3, key: LAST
    }
    with pytest.raises(ValueError, match='measurement_steps_last_frames'):
        MeasurementsViewerWin.validateConfiguration({
            **viewer.configuration(),
            'measurement_steps_last_frames': {key: -1}
        })


def test_masked_bool_combined_measurement_stays_numeric(
        viewer, tmp_path, monkeypatch
    ):
    load(viewer, tmp_path, mask=True)
    accept_formula(monkeypatch, 'signal > 5', 'bright')
    viewer.combineMeasurements()
    viewer.measurementProcessingWarning.assert_not_called()
    bright = cell_values(viewer.acdc_df, 'bright')
    assert bright[:LAST + 1] == [0., 1., 1.]
    assert_nan_after(bright)
    assert 'bright' in viewer.acdc_df.select_dtypes(
        include=['number', 'bool']
    ).columns
    table = frames_table()
    table['flag'] = table['signal'] > 5
    measurements_viewer.blank_unanalyzed_outputs(table, ['flag'], LAST)
    assert pd.api.types.is_float_dtype(table['flag'])


def test_combine_refused_when_option_toggled_meanwhile(
        viewer, tmp_path, monkeypatch
    ):
    load(viewer, tmp_path, mask=True)

    def accept(dialog):
        dialog.newColNameLineEdit.setText('peak')
        dialog.formulaEditWidget.setText('MAX(signal)')
        dialog.test_cb()
        viewer.setMaskUnanalyzedFrames(False)
        dialog.ok_cb()

    monkeypatch.setattr(apps.CombineMetricsEquationDialog, 'exec_', accept)
    viewer.combineMeasurements()
    viewer.measurementProcessingWarning.assert_called_once()
    assert 'peak' not in viewer.acdc_df.columns
    assert not viewer.combinedMeasurementSteps


def test_shaded_regions_have_no_tool_tip(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    for region in (viewer.unanalyzedRegion, viewer.phaseBandUnanalyzedRegion):
        assert region.isVisible()
        assert region.toolTip() == ''


def test_toggling_is_refused_while_processing(viewer, tmp_path):
    load(viewer, tmp_path, mask=True)
    viewer.measurementProcessingThread = object()
    try:
        viewer.maskUnanalyzedFramesAction.trigger()
        assert viewer.maskUnanalyzedFrames
        assert viewer.maskUnanalyzedFramesAction.isChecked()
    finally:
        viewer.measurementProcessingThread = None


def test_reload_refreshes_label_with_option_off(viewer, tmp_path):
    path = load(viewer, tmp_path, mask=False)
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 3 (all frames used)'
    )
    rewrite_cca_frames(path, range(5))
    viewer.reloadAcdcDfFile()
    assert viewer.lastAnalyzedFrameLabel.text() == (
        'Last cc annot. frame n. = 5 (all frames used)'
    )
    rewrite_cca_frames(path, range(SIZE_T))
    viewer.reloadAcdcDfFile()
    assert viewer.lastAnalyzedFrameLabel.text() == ''
