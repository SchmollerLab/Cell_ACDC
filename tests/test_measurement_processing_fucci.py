"""FUCCI premade configuration: processing plugins and the full pipeline."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cellacdc import measurement_processing, phase_translator
from cellacdc.custom_annotation_predictors import CustomAnnotationPredictor
from cellacdc.measurement_processing.MinMaxNormalization.acdcProcess import (
    Model as MinMaxNormalization
)
from cellacdc.measurement_processing.RampDetection.acdcProcess import (
    Model as RampDetection
)
from cellacdc.measurements_viewer import (
    MeasurementsViewerWin, PREMADE_CONFIGURATIONS_PATH
)
from cellacdc.timepoint_predictors import (
    PREDICTOR_NONE, PredictorDependencies, TimepointPredictor,
    TIMEPOINT_PREFIX
)

FUCCI_TEMPLATE = os.path.join(
    PREMADE_CONFIGURATIONS_PATH, 'fucci_cell_cycle.json'
)
SUBSET_PATH = os.path.join(
    os.path.dirname(__file__), 'test_data', 'fucci_cycling_subset.csv'
)


def fucci_configuration(mapping, overrides=None):
    """Return the FUCCI template with placeholders replaced.

    `overrides` maps a recipe method name to kwargs updated in every step.
    """
    with open(FUCCI_TEMPLATE, encoding='utf-8') as file:
        template = json.load(file)
    configuration = MeasurementsViewerWin.resolvePremadeTemplate(
        template, mapping
    )
    for step in configuration['measurement_processing']:
        for item in step['recipe']:
            item['kwargs'].update((overrides or {}).get(item['method'], {}))
    return configuration


def run_fucci_pipeline(acdc_df, mapping, SizeT=None, overrides=None):
    """Replay steps, predict time points/annotations and translate phases.

    Mirrors what the measurements viewer does when the FUCCI configuration
    is loaded and every cell is predicted. Returns (table, configuration).
    """
    configuration = fucci_configuration(mapping, overrides)
    df = acdc_df
    if list(df.index.names) != ['frame_i', 'Cell_ID']:
        df = df.set_index(['frame_i', 'Cell_ID'])
    posData = SimpleNamespace(acdc_df=df.sort_index(), SizeT=SizeT)
    MeasurementsViewerWin.replayMeasurementSteps(posData, configuration)
    table = posData.acdc_df
    timepoints = {
        f'{TIMEPOINT_PREFIX}{identifier["name"]}': identifier['predictor']
        for identifier in configuration['timepoint_identifiers']
    }
    # Manual annotations (e.g. "Excluded") have no predictor, like in
    # `MeasurementsViewerWin.validateConfiguration`
    annotations = {
        state['name']: state.get('predictor') or {'type': PREDICTOR_NONE}
        for state in configuration['custom_annotations']
    }
    predictors = {**timepoints, **annotations}
    for column in PredictorDependencies.order(predictors):
        if column in timepoints:
            predicted = TimepointPredictor.predict(
                table, predictors[column], SizeT=SizeT
            )
            table[column] = False
            value = True
        else:
            predicted = CustomAnnotationPredictor.predict(
                table, predictors[column], SizeT=SizeT
            )
            table[column] = 0
            value = 1
        for ID, frames in predicted.items():
            for frame_i in frames:
                table.loc[(frame_i, ID), column] = value
    evaluation = phase_translator.evaluate(
        table, configuration['phase_translator_rules'], SizeT=SizeT
    )
    return evaluation.table, configuration


def timepoint_frame(table, ID, name):
    cell = table.xs(ID, level='Cell_ID')
    frames = cell.index[cell[f'{TIMEPOINT_PREFIX}{name}'].astype(bool)]
    return int(frames[0]) if len(frames) else None


# --- Processing plugins ---------------------------------------------------

def test_new_processors_are_registered():
    registry = measurement_processing.MEASUREMENT_PROCESSING_MAPPER
    assert registry['Min-max normalization']['class'] is MinMaxNormalization
    assert registry['Ramp detection']['class'] is RampDetection
    assert MinMaxNormalization.is_bulk_processing
    assert RampDetection.is_timelapse


def test_min_max_normalization():
    df = pd.DataFrame(
        {'a': [10., 20., np.nan], 'b': [15., 30., 40.]},
        index=pd.MultiIndex.from_tuples(
            [(0, 1), (1, 1), (0, 2)], names=['frame_i', 'Cell_ID']
        )
    )
    result = MinMaxNormalization.run(
        df, background_value=10, min_value=0, max_value=20
    )
    np.testing.assert_allclose(result['a'], [0, 0.5, np.nan])
    np.testing.assert_allclose(result['b'], [0.25, 1, 1.5])
    pd.testing.assert_frame_equal(MinMaxNormalization.run(df), df)
    out = measurement_processing.apply_recipe(
        df, ['a'], [{'method': 'Min-max normalization',
                     'kwargs': {'min_value': 10.0, 'max_value': 20.0}}],
        ['a_norm']
    )
    np.testing.assert_allclose(out['a_norm'], [0, 1, np.nan])


@pytest.mark.parametrize('kwargs', [
    {'min_value': 1.0, 'max_value': 1.0}, {'max_value': -1.0},
    {'background_value': 'x'}, {'min_value': True}
])
def test_min_max_normalization_rejects_invalid_constants(kwargs):
    with pytest.raises(ValueError):
        MinMaxNormalization.run(pd.DataFrame({'a': [1.0]}), **kwargs)


def test_ramp_detection_segments_gaps_and_min_length():
    up = np.arange(1, 21, dtype=float)
    signal = np.concatenate([up, up[::-1]])
    result = RampDetection.run(signal, gap_frames=0, min_length=5)
    assert list(result[:19]) == [1] * 19
    assert list(result[20:39]) == [-1] * 19
    # 20 -> 20 is no change; the last observation has no next value
    assert result[19] == 0 and result[-1] == 0

    # A short falling interruption is bridged, unless it exceeds gap_frames
    bumpy = np.array([1, 2, 3, 4, 3, 5, 6, 7, 8, 9], dtype=float)
    assert list(RampDetection.run(bumpy, gap_frames=1, min_length=9)) == (
        [1] * 9 + [0]
    )
    # Without bridging, the 3-step rise and the 1-step fall are too short
    assert list(RampDetection.run(bumpy, gap_frames=0, min_length=4)) == (
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    )
    assert list(RampDetection.run(bumpy, gap_frames=0, min_length=1)) == (
        [1, 1, 1, -1, 1, 1, 1, 1, 1, 0]
    )
    assert not RampDetection.run(bumpy, gap_frames=0, min_length=6).any()


def test_ramp_detection_does_not_bridge_long_opposite_ramp():
    # Regression: with min_length <= gap_frames, a falling run long enough
    # by itself was bridged by the rise and also kept as a fall, giving 0
    rise = np.arange(21, dtype=float)
    fall = rise[-1] - np.arange(1, 7, dtype=float)
    rise2 = fall[-1] + np.arange(1, 21, dtype=float)
    signal = np.concatenate([rise, fall, rise2])
    result = RampDetection.run(
        signal, gap_frames=10, min_length=5, relative=False
    )
    assert list(result[:20]) == [1] * 20
    assert list(result[20:26]) == [-1] * 6
    assert list(result[26:46]) == [1] * 20
    assert result[-1] == 0
    assert list(result) == list(RampDetection.run(
        signal, gap_frames=3, min_length=5, relative=False
    ))

    # The pieces of a bridged ramp left after removing the opposite ramp
    # must still be long enough
    signal = np.array(
        [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0], dtype=float
    )
    assert list(RampDetection.run(
        signal, gap_frames=10, min_length=5, relative=False
    )) == [0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0]

    # Short runs that bridge in both directions still cancel out where they
    # overlap (only the first rise and the last fall are not overlapped)
    oscillation = np.tile(np.r_[np.arange(8.0), np.arange(8.0, 0, -1)], 6)
    result = RampDetection.run(
        oscillation, gap_frames=10, min_length=40, relative=False
    )
    assert list(result[:8]) == [1] * 8 and list(result[-8:-1]) == [-1] * 7
    assert not result[8:-8].any()


def test_ramp_detection_threshold_relative_and_missing_values():
    signal = np.array([100, 101, 102, 103, np.nan, 104], dtype=float)
    result = RampDetection.run(signal, gap_frames=0, min_length=1)
    assert list(result[:3]) == [1, 1, 1] and np.isnan(result[4])
    # Relative changes of ~1% do not pass a 5% threshold
    assert not np.nan_to_num(
        RampDetection.run(signal, threshold=0.05, min_length=1)
    ).any()
    # Absolute changes of 1 pass an absolute threshold of 0.5
    assert RampDetection.run(
        signal, threshold=0.5, min_length=1, relative=False
    )[0] == 1


@pytest.mark.parametrize('signal, expected', [
    # Changes towards and away from zero keep their sign (regression: the
    # step to 0 was replaced with no change and the ramp was discarded)
    ([3, 2, 1, 0], [-1, -1, -1, 0]),
    ([0, 1, 2, 3], [1, 1, 1, 0]),
    ([-2, -1, 0, 1], [1, 1, 1, 0]),
    ([1, 0, -1, -2], [-1, -1, -1, 0]),
])
def test_ramp_detection_relative_change_through_zero(signal, expected):
    signal = np.array(signal, dtype=float)
    result = RampDetection.run(signal, gap_frames=0, min_length=3)
    assert list(result) == expected
    # Same direction as the absolute change
    assert list(RampDetection.run(
        signal, gap_frames=0, min_length=3, relative=False
    )) == expected


def test_ramp_detection_relative_change_zero_and_missing_values():
    # 0 -> 0 is no change and interrupts the ramps
    signal = np.array([2, 1, 0, 0, 1, 2], dtype=float)
    assert list(RampDetection.run(signal, gap_frames=0, min_length=1)) == (
        [-1, -1, 0, 1, 1, 0]
    )
    # The step to 0 is a full (100%) relative decrease
    assert RampDetection.run(
        np.array([1, 0], dtype=float), threshold=0.99, min_length=1
    )[0] == -1
    # Ordinary nonzero data: relative to the next value (1 -> 2 is +50%)
    rising = np.array([1, 2], dtype=float)
    assert RampDetection.run(rising, threshold=0.49, min_length=1)[0] == 1
    assert RampDetection.run(rising, threshold=0.51, min_length=1)[0] == 0
    # Changes from or to a missing value are no change
    signal = np.array([0, np.nan, 0, 1], dtype=float)
    result = RampDetection.run(signal, gap_frames=0, min_length=1)
    assert result[0] == 0 and np.isnan(result[1]) and result[2] == 1


@pytest.mark.parametrize('kwargs', [
    {'threshold': -1.0}, {'gap_frames': -1}, {'min_length': 0},
    {'min_length': 2.5}, {'gap_frames': True}, {'relative': 'False'}
])
def test_ramp_detection_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        RampDetection.run(np.arange(5.0), **kwargs)


# --- Full pipeline ----------------------------------------------------------

def synthetic_fucci_table():
    """One mother (frames 0-149) dividing into two daughters at frame 150.

    G1 until ~40 (Geminin low), Cdt1 peaks at ~60, the channels cross at
    ~90, Cdt1 is low from ~115 (G2) and the volume jumps in the last frames.
    """
    t = np.arange(150, dtype=float)
    geminin = np.clip((t - 40) / 80, 0, 1) * 0.9 + 0.05
    cdt1 = np.where(
        t < 60, 0.3 + 0.6 * t / 60, np.clip(0.9 - (t - 60) / 55, 0.02, 1)
    )
    volume = 100 + t * 0.5
    volume[-6:] *= 2.5
    rows = [
        {'frame_i': int(f), 'Cell_ID': 1, 'GFP_mean': g, 'mCherry_mean': c,
         'cell_vol_fl': v, 'parent_ID_tree': -1}
        for f, g, c, v in zip(t, geminin, cdt1, volume)
    ]
    for ID in (2, 3):
        rows += [
            {'frame_i': f, 'Cell_ID': ID, 'GFP_mean': 0.05,
             'mCherry_mean': 0.3, 'cell_vol_fl': 120.0, 'parent_ID_tree': 1}
            for f in range(150, 170)
        ]
    return pd.DataFrame(rows)


def test_fucci_pipeline_on_synthetic_cell():
    table, _ = run_fucci_pipeline(
        synthetic_fucci_table(), {'GEMININ': 'GFP', 'CDT1': 'mCherry'},
        SizeT=170
    )
    landmarks = {
        name: timepoint_frame(table, 1, name)
        for name in ('G1_S', 'Cdt1_peak', 'Crossing', 'G2_start', 'M_start')
    }
    assert None not in landmarks.values(), landmarks
    frames = list(landmarks.values())
    assert frames == sorted(frames) and len(set(frames)) == 5, landmarks
    assert 30 <= landmarks['G1_S'] <= 50
    assert 55 <= landmarks['Cdt1_peak'] <= 65
    assert landmarks['M_start'] >= 144
    phases = table.xs(1, level='Cell_ID')['phase']
    assert list(dict.fromkeys(phases)) == [
        'G1', 'S Early', 'S', 'S Late', 'G2', 'M'
    ]
    # Daughters have no full cycle and no daughters themselves
    assert timepoint_frame(table, 2, 'M_start') is None
    assert table.xs(2, level='Cell_ID')['Arrested'].eq(0).all()


def test_fucci_pipeline_with_dashed_channel_names():
    """Channel names that are not identifiers stay quoted in the equations
    of the resolved template and give the same result."""
    expected, _ = run_fucci_pipeline(
        synthetic_fucci_table(), {'GEMININ': 'GFP', 'CDT1': 'mCherry'},
        SizeT=170
    )
    df = synthetic_fucci_table().rename(columns={
        'GFP_mean': 'mAG-hGem_mean', 'mCherry_mean': 'mKO2-hCdt1_mean'
    })
    table, configuration = run_fucci_pipeline(
        df, {'GEMININ': 'mAG-hGem', 'CDT1': 'mKO2-hCdt1'}, SizeT=170
    )
    assert configuration['combined_measurements'][0]['equations'][
        'fucci_ramp_pattern'
    ] == '`mAG-hGem_ramp` - `mKO2-hCdt1_ramp`'
    assert 'mKO2-hCdt1_fucci' in table
    pd.testing.assert_series_equal(table['phase'], expected['phase'])
    for column in ('fucci_crossing', 'Arrested', *(
            f'{TIMEPOINT_PREFIX}{name}' for name in
            ('G1_S', 'Cdt1_peak', 'Crossing', 'G2_start', 'M_start'))):
        pd.testing.assert_series_equal(table[column], expected[column])
    rules = configuration['conditional_formatting']['mKO2-hCdt1_fucci']
    cell = table.xs(1, level='Cell_ID')
    matched = MeasurementsViewerWin.evaluateConditionalFormatting(
        cell, rules, ID=1, SizeT=170
    )
    assert matched.tolist() == np.select(
        [cell['mKO2-hCdt1_ramp'] == 1, cell['mKO2-hCdt1_ramp'] == -1],
        [0, 1], -1
    ).tolist()


def test_fucci_pipeline_matches_original_classifier():
    """Real FUCCI tracks (cycling, Position_1) classified by the original
    FUCCI-cell-cycle-phase-and-plots repo (`repo_classification`, without
    manual corrections). Intensities are the repo's normalized means, so
    only the classification logic is compared. The comparison over all
    positions gave ~84% per-frame agreement (cycling)."""
    df = pd.read_csv(SUBSET_PATH)
    repo = df.pop('repo_classification')
    repo.index = pd.MultiIndex.from_frame(df[['frame_i', 'Cell_ID']])
    table, _ = run_fucci_pipeline(
        df, {'GEMININ': 'GFP', 'CDT1': 'mCherry'}, SizeT=300
    )
    phases = table['phase'].fillna('').reindex(repo.index)
    assert (phases == repo).mean() >= 0.88
    for ID in (20, 55, 56, 58):  # complete cycles from birth to division
        cell = phases.xs(ID, level='Cell_ID')
        assert (cell == repo.xs(ID, level='Cell_ID')).mean() >= 0.95
        assert list(dict.fromkeys(cell)) == [
            'G1', 'S Early', 'S', 'S Late', 'G2', 'M'
        ]


def test_arrested_cell_is_annotated():
    frames = range(200)
    df = pd.DataFrame({
        'frame_i': list(frames), 'Cell_ID': 1,
        'GFP_mean': 0.5, 'mCherry_mean': 0.6, 'cell_vol_fl': 100.0,
        'parent_ID_tree': -1,
    })
    table, _ = run_fucci_pipeline(
        df, {'GEMININ': 'GFP', 'CDT1': 'mCherry'}, SizeT=200
    )
    assert table['Arrested'].eq(1).all()
    assert table['phase'].eq('arrested').all()
