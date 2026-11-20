"""Re-structure core `cellacdc.acdc_datastruct.restructure`: the move out
of `dataReStruct` (re-exports, Qt-free import), the headless signal
adapters of `callbacks` driving the core, option normalization,
plan-before-write, squeeze, segmentation warning, the GUI side (thread
quits on error, dialog backstop) and the registry predicates. The golden
cases themselves run in `test_golden_restructure.py`.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest
import tifffile

from cellacdc.acdc_datastruct import callbacks, errors, restructure

from . import conditions, expected_restructure_changes, harness_restructure
from . import manifest, scenarios, synthetic

_APP = None

MOVED_NAMES = (
    'fail', 'mode_for', 'tiff_stem', 'natural_key', 'parse_index',
    'parse_channel', 'axis_index', 'get_layout', 'normalize_array',
    'create_datastruct_from_restruct_tiff_dialogue_options',
)
GUI_MODULES = ('cellacdc.apps', 'cellacdc.widgets', 'cellacdc.workers',
               'cellacdc._main', 'cellacdc.dataStruct',
               'cellacdc.dataReStruct')

run_core = restructure.create_datastruct_from_restruct_tiff_dialogue_options


def prepare_case(case_id: str, tmp_path) -> tuple[dict, dict]:
    """Inputs and `selectedOptions` of a golden case (as the harness builds
    them) -> (options, path placeholders)."""
    case = scenarios.RESTRUCTURE[case_id]
    work = tmp_path / 'work'
    src, dst = work / 'src', work / 'dst'
    src.mkdir(parents=True)
    dst.mkdir()
    placeholders = {'<SRC>': str(src), '<DST>': str(dst), '<WORK>': str(work)}
    case.make_inputs(str(src), str(dst), str(work))
    options = harness_restructure.build_options(
        case, str(src), str(dst), placeholders
    )
    return options, placeholders


# ------------------------------------------------------------------- move
@conditions.requires_qt
def test_datarestruct_reexports_the_moved_core():
    from cellacdc import dataReStruct

    for name in MOVED_NAMES:
        assert getattr(dataReStruct, name) is getattr(restructure, name), name


@conditions.requires_qt
def test_datarestruct_keeps_its_gui_names():
    from cellacdc import dataReStruct

    for name in ('run', '_log', 'saveTiff'):
        assert callable(getattr(dataReStruct, name)), name


def test_harness_uses_the_qt_free_core():
    assert harness_restructure.core_module_name() == \
        'cellacdc.acdc_datastruct.restructure'


_HEADLESS_RUN = '''
import json, os, sys
import numpy as np, tifffile
from cellacdc.acdc_datastruct import callbacks, restructure
src, dst = sys.argv[1], sys.argv[2]
for i in (1, 2):
    tifffile.imwrite(os.path.join(src, f'img{i}.tif'),
                     np.full((4, 5), i, dtype='uint16'))
lines = []
cb = callbacks.ProgressCallbacks(
    log=lambda text, level='INFO': lines.append([text, level]))
out = restructure.create_datastruct_from_restruct_tiff_dialogue_options(
    {'source_folder': src, 'destination_folder': dst,
     'overwrite_destination': False,
     'create_new_position_destination': False,
     'dimensions': {k: {'mode': 'not_present'}
                    for k in restructure.DIMENSION_KEYS}},
    signals=callbacks.SignalsAdapter(cb))
import cellacdc
blocked = _AcdcBlocker.blocked
loaded = sorted(m for m in sys.modules
                if m.split('.')[0] in blocked or m in GUI_MODULES)
print(json.dumps({'gui': cellacdc.GUI_INSTALLED, 'loaded': loaded,
                  'positions': sorted(out['position_folders']),
                  'log': lines}))
'''


@conditions.subprocess_heavy
def test_restructure_runs_without_qt(tmp_path):
    """Import and run the core in a child where every Qt binding is blocked:
    no GUI module is loaded."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir()
    code = (conditions.blocked_imports_prelude()
            + f'GUI_MODULES = {GUI_MODULES!r}\n' + _HEADLESS_RUN)
    proc = conditions.run_python(code, str(src), str(dst))
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {
        'gui': False, 'loaded': [], 'positions': ['img1', 'img2'],
        'log': [['Processing TIFF files...', 'INFO'],
                ['Creating Position folder Position_1...', 'INFO'],
                ['Creating Position folder Position_2...', 'INFO']],
    }
    assert (dst / 'Position_2' / 'Images' / 'Channel_1.tif').is_file()


# ---------------------------------------------- callbacks <-> signals adapters
@pytest.mark.parametrize('case_id', ['RA', 'RK', 'combo_if-pf-if'])
def test_signals_adapter_drives_progress_callbacks(case_id, tmp_path):
    """`SignalsAdapter(ProgressCallbacks(...))` (what the command line
    passes) receives exactly the GUI's signal sequence, in order."""
    golden = scenarios.load_golden('restructure', case_id)
    options, placeholders = prepare_case(case_id, tmp_path)
    calls = []

    def log(text, level='INFO'):
        calls.append(['progress',
                      manifest.normalize_path_text(text, placeholders),
                      level])

    progress = callbacks.ProgressCallbacks(
        log=log,
        progress_init=lambda total: calls.append(['initProgressBar', total]),
        progress_step=lambda step: calls.append(['progressBar', step]),
    )
    run_core(options, signals=callbacks.SignalsAdapter(progress))
    assert calls == golden['signals']


def test_progress_callbacks_and_signals_adapter_defaults():
    progress = callbacks.ProgressCallbacks()
    progress.log('text', 'WARNING')
    progress.progress_init(3)
    progress.progress_step(1)
    assert progress.is_cancelled() is False

    calls = []
    adapter = callbacks.SignalsAdapter(callbacks.ProgressCallbacks(
        log=lambda text, level='INFO': calls.append((text, level)),
    ))
    adapter.progress.emit('default level')
    adapter.progress.emit('warn', 'WARNING')
    adapter.critical.emit((None, ValueError('x')))
    adapter.finished.emit({'position_folders': {}})
    adapter.initProgressBar.emit(2)
    adapter.progressBar.emit(1)
    assert calls == [('default level', 'INFO'), ('warn', 'WARNING')]


# ------------------------------------------------------ normalize (GV-R2)
def _dims(**changes):
    dims = {key: {'mode': 'not_present'}
            for key in restructure.DIMENSION_KEYS}
    dims.update(changes)
    return dims


def _options(**changes):
    options = {
        'source_folder': 'S', 'destination_folder': 'D',
        'overwrite_destination': False,
        'create_new_position_destination': False,
        'dimensions': _dims(),
    }
    options.update(changes)
    return options


def test_normalize_returns_a_new_validated_dict():
    opts = _options(dimensions={
        'channels': {'mode': 'per_file', 'count': 2, 'names': [
            {'name': ' GFP ', 'filename_identifier': '_gfp'},
            {'name': 'mCh', 'filename_identifier': '_mch'}]},
        'timepoints': {'mode': 'in_file', 'axis': '1'},
    }, basename='exp-1', files=['a.tif'])
    before = json.dumps(opts, sort_keys=True)
    options = restructure.normalize_restructure_options(opts)
    assert json.dumps(opts, sort_keys=True) == before
    assert options is not opts
    assert options['dimensions'] == {
        'channels': {'mode': 'per_file', 'count': 2, 'names': [
            {'name': 'GFP', 'filename_identifier': '_gfp'},
            {'name': 'mCh', 'filename_identifier': '_mch'}]},
        'timepoints': {'mode': 'in_file', 'axis': '1'},
        'z_slices': {'mode': 'not_present'},
    }
    assert options['basename'] == 'exp-1' and options['files'] == ['a.tif']


def test_normalize_defaults():
    opts = _options()
    del opts['overwrite_destination']
    del opts['create_new_position_destination']
    options = restructure.normalize_restructure_options(opts)
    assert options['overwrite_destination'] is False
    assert options['create_new_position_destination'] is False
    assert options['dimensions']['channels']['names'] == [
        {'name': 'Channel_1'}
    ]
    in_file = restructure.normalize_restructure_options(_options(
        dimensions=_dims(channels={'mode': 'in_file', 'axis_index': -1,
                                   'names': ['a', 'b']}),
        basename='  exp_  ',
    ))
    assert in_file['dimensions']['channels'] == {
        'mode': 'in_file', 'axis_index': -1,
        'names': [{'name': 'a'}, {'name': 'b'}],
    }
    assert in_file['basename'] == '  exp_  '


_INVALID_OPTIONS = [
    ('not_a_mapping', [], 'must be a mapping'),
    ('no_source', _options(source_folder=''),
     'The source folder is required.'),
    ('no_destination', {'source_folder': 'S'},
     'The destination folder is required.'),
    ('both_flags', _options(overwrite_destination=True,
                            create_new_position_destination=True),
     'Choose either to overwrite'),
    ('no_dimensions', _options(dimensions=None),
     'The dimension options are required.'),
    ('dimension_not_mapping', _options(dimensions=_dims(z_slices='in_file')),
     'The options of z_slices must be a mapping'),
    ('unknown_mode', _options(dimensions=_dims(channels={'mode': 'separate'})),
     "Unsupported mode 'separate' for channels."),
    ('no_mode', _options(dimensions=_dims(timepoints={})),
     'Unsupported mode None for timepoints.'),
    ('empty_name', _options(dimensions=_dims(channels={
        'mode': 'not_present', 'names': [{'name': ' '}]})),
     'Provide a name for every channel.'),
    ('name_with_space', _options(dimensions=_dims(channels={
        'mode': 'not_present', 'names': ['my GFP']})),
     "Invalid channel name 'my GFP'"),
    ('name_with_dot', _options(dimensions=_dims(channels={
        'mode': 'not_present', 'names': ['GFP.tif']})),
     "Invalid channel name 'GFP.tif'"),
    ('duplicate_names', _options(dimensions=_dims(channels={
        'mode': 'in_file', 'axis': 0, 'names': ['GFP', 'gfp']})),
     'Channel names must be unique.'),
    ('not_present_two_names', _options(dimensions=_dims(channels={
        'mode': 'not_present', 'names': ['a', 'b']})),
     'provide exactly one channel name'),
    ('missing_identifier', _options(dimensions=_dims(channels={
        'mode': 'per_file', 'names': [
            {'name': 'a', 'filename_identifier': 'a'}, {'name': 'b'}]})),
     "No filename identifier was provided for channel 'b'."),
    ('missing_t_marker', _options(dimensions=_dims(
        timepoints={'mode': 'per_file'})),
     'No filename marker was provided for timepoint.'),
    ('empty_z_marker', _options(dimensions=_dims(
        z_slices={'mode': 'per_file', 'filename_marker': ''})),
     'No filename marker was provided for z-slice.'),
    ('missing_axis', _options(dimensions=_dims(
        timepoints={'mode': 'in_file'})),
     'Enter an axis index for timepoints.'),
    ('axis_not_int', _options(dimensions=_dims(
        z_slices={'mode': 'in_file', 'axis': 'Z'})),
     "The axis index for z_slices must be an integer; got 'Z'."),
    ('basename_with_space', _options(basename='my exp'),
     "Invalid basename 'my exp'"),
    ('basename_not_text', _options(basename=3), 'The basename must be text'),
]


@pytest.mark.parametrize(
    'opts, message', [case[1:] for case in _INVALID_OPTIONS],
    ids=[case[0] for case in _INVALID_OPTIONS],
)
def test_normalize_rejects_invalid_options(opts, message):
    with pytest.raises(errors.RestructureOptionsError) as info:
        restructure.normalize_restructure_options(opts)
    assert message in str(info.value)
    assert isinstance(info.value, errors.UsageError)
    assert isinstance(info.value, ValueError)


# ------------------------------------------------ plan before write (GV-R1)
def _add_existing(dst, relpath, text='old\n'):
    path = dst.joinpath(*relpath.split('/'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


@pytest.mark.parametrize('case_id, error', [
    ('RE', errors.ExistingPositionsError),
    ('RH', errors.RestructureOptionsError),
    ('RI', errors.RestructureOptionsError),
    ('RN', errors.SourceError),
    ('extra_yx_mismatch', errors.SourceError),
    ('extra_overlap', errors.SourceError),
    ('extra_existing_later_position', errors.ExistingPositionsError),
    ('extra_create_new_collides_invalid_position',
     errors.ExistingPositionsError),
    ('extra_in_file_channel_count', errors.RestructureOptionsError),
    ('extra_axis_out_of_range', errors.RestructureOptionsError),
    ('extra_both_overwrite_and_create_new', errors.RestructureOptionsError),
    ('extra_unknown_channel_mode', errors.RestructureOptionsError),
    ('extra_channel_name_with_space', errors.RestructureOptionsError),
])
def test_errors_are_raised_before_writing(case_id, error, tmp_path):
    options, placeholders = prepare_case(case_id, tmp_path)
    before = conditions.tree_snapshot(str(tmp_path / 'work'))
    with pytest.raises(error) as info:
        run_core(options)
    assert isinstance(info.value, errors.UsageError)
    assert conditions.tree_snapshot(str(tmp_path / 'work')) == before
    golden = scenarios.load_golden('restructure', case_id)
    if golden['exception'] is not None:
        assert manifest.normalize_path_text(
            str(info.value), placeholders) == golden['exception'][1]


def test_overwrite_deletes_nothing_when_a_later_position_fails(tmp_path):
    """Main removed Position_1/Images and wrote Position_1 before failing
    at position b."""
    options, _ = prepare_case('RN', tmp_path)
    dst = tmp_path / 'work' / 'dst'
    _add_existing(dst, 'Position_1/Images/segm.npz.txt')
    _add_existing(dst, 'Position_2/Images/acdc_output.csv')
    options['overwrite_destination'] = True
    before = conditions.tree_snapshot(str(dst))
    with pytest.raises(errors.SourceError, match="channel 'mCh'"):
        run_core(options)
    assert conditions.tree_snapshot(str(dst)) == before


@pytest.mark.parametrize('p2_dtypes, common', [
    (('int64', 'uint16'), 'int64'),
    (('uint64', 'uint16'), 'uint64'),
    (('float16', 'float16'), 'float16'),
    (('uint32', 'int16'), 'int64'),
])
def test_unsupported_pixel_type_fails_before_writing(p2_dtypes, common,
                                                     tmp_path):
    """Main removed Position_2/Images, then failed writing its first
    channel (OmeXmlError) and left a partial file there."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir()
    for position, dtypes in (('p1', ('uint16', 'uint16')), ('p2', p2_dtypes)):
        for channel, dtype in zip(('GFP', 'mCh'), dtypes):
            tifffile.imwrite(src / f'{position}_{channel}.tif',
                             np.zeros((4, 5), dtype=dtype))
    for number in (1, 2):
        _add_existing(dst, f'Position_{number}/Images/acdc_output.csv')
    channels = {'mode': 'per_file', 'names': [
        {'name': name, 'filename_identifier': name} for name in ('GFP', 'mCh')
    ]}
    options = _options(
        source_folder=str(src), destination_folder=str(dst),
        overwrite_destination=True, dimensions=_dims(channels=channels),
    )
    before = conditions.tree_snapshot(str(dst))
    with pytest.raises(errors.SourceError) as info:
        run_core(options)
    message = str(info.value)
    assert message.startswith(
        f'OME-TIFF cannot store the pixel type {common!r} of the files '
        "grouped as position 'p2' ("
    )
    assert f'{p2_dtypes[0]}: p2_GFP.tif' in message
    assert conditions.tree_snapshot(str(dst)) == before


def test_ome_pixel_types_match_tifffile(tmp_path):
    """The plan-time check accepts exactly the types the writer stores."""
    writable = []
    for dtype in ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8',
                  'uint16', 'uint32', 'uint64', 'float16', 'float32',
                  'float64', 'complex64', 'complex128'):
        try:
            tifffile.imwrite(tmp_path / f'{dtype}.tif',
                             np.zeros((2, 2), dtype=dtype), ome=True)
        except tifffile.OmeXmlError:
            continue
        writable.append(dtype)
    assert sorted(writable) == sorted(restructure._OME_PIXEL_TYPES)


def test_existing_positions_error_lists_every_conflict(tmp_path):
    options, _ = prepare_case('RB', tmp_path)
    dst = tmp_path / 'work' / 'dst'
    _add_existing(dst, 'Position_1/Images/a.txt')
    _add_existing(dst, 'Position_2/Images/a.txt')
    with pytest.raises(errors.ExistingPositionsError) as info:
        restructure.plan_restructure(options)
    err = info.value
    assert isinstance(err, FileExistsError)
    assert err.existing == [1, 2]
    assert err.choices == ('overwrite', 'create-new')
    assert str(err) == ('Output Position folder already exist: '
                        f'{dst / "Position_1"}. ')


@pytest.mark.parametrize('case_id', [
    'extra_source_missing', 'extra_channel_name_with_space', 'RL',
    'extra_missing_identifier', 'extra_both_overwrite_and_create_new',
])
def test_early_errors_emit_nothing(case_id, tmp_path):
    """Option checks, the source folder and the TIFF listing fail before
    the first signal, as on main."""
    options, placeholders = prepare_case(case_id, tmp_path)
    recorder = harness_restructure.RecordingSignals(placeholders)
    with pytest.raises(errors.UsageError):
        run_core(options, signals=recorder)
    assert recorder.events == []


def test_execute_errors_are_runtime_errors(tmp_path):
    """A file that changes between plan and execution is a runtime error
    (ValueError, not a UsageError: the command line exits 1)."""
    options, _ = prepare_case('RB', tmp_path)
    plan = restructure.plan_restructure(options)
    path = plan.positions[1].channels[0].records[0]['path']
    tifffile.imwrite(path, np.zeros((6, 5), dtype='uint16'))
    with pytest.raises(ValueError, match='TIFF shape changed') as info:
        restructure.execute_restructure_plan(plan)
    assert not isinstance(info.value, errors.UsageError)


# ----------------------------------------- create new, plan dict, squeeze
def test_create_new_without_existing_positions_starts_at_1(tmp_path):
    options, _ = prepare_case('RB', tmp_path)
    exp = tmp_path / 'new' / 'exp'
    options['create_new_position_destination'] = True
    options['destination_folder'] = str(exp)
    plan = restructure.plan_restructure(options)
    assert not (tmp_path / 'new').exists()
    assert [position.number for position in plan.positions] == [1, 2]
    result = restructure.execute_restructure_plan(plan)
    assert result['position_folders'] == {
        'img1': str(exp / 'Position_1'), 'img2': str(exp / 'Position_2'),
    }


def test_create_new_counts_valid_positions_only(tmp_path):
    options, _ = prepare_case('RB', tmp_path)
    dst = tmp_path / 'work' / 'dst'
    _add_existing(dst, 'Position_3/Images/a.txt')
    (dst / 'Position_7' / 'Images').mkdir(parents=True)
    options['create_new_position_destination'] = True
    plan = restructure.plan_restructure(options)
    assert [position.number for position in plan.positions] == [4, 5]


def _read_metadata_csv(path):
    with open(path, encoding='utf-8') as file:
        return dict(line.rstrip('\n').split(',', 1) for line in file)


@pytest.mark.parametrize('case_id', [
    'RA', 'RF', 'RG', 'RK', 'RN_subset', 'combo_pf-pf-pf', 'combo_if-if-if',
])
def test_plan_to_dict_matches_the_execution(case_id, tmp_path):
    options, _ = prepare_case(case_id.replace('_subset', ''), tmp_path)
    if case_id == 'RN_subset':
        options['files'] = [
            name for name in os.listdir(options['source_folder'])
            if name.startswith('a_')
        ]
    existing_images = sorted(
        str(path) for path in (tmp_path / 'work' / 'dst').glob('*/Images')
    )
    plan = restructure.plan_restructure(options)
    summary = plan.to_dict()
    assert json.loads(json.dumps(summary)) == summary
    result = restructure.execute_restructure_plan(plan)
    assert {p['identifier']: p['folder'] for p in summary['positions']} \
        == result['position_folders']
    assert [channel['image_file'] for p in summary['positions']
            for channel in p['channels']] == result['image_files']
    written = set(result['position_folders'].values())
    assert summary['to_delete'] == [
        path for path in existing_images
        if plan.overwrite and os.path.dirname(path) in written
    ]
    assert bool(summary['to_delete']) == (case_id == 'RF')
    for position in summary['positions']:
        rows = _read_metadata_csv(os.path.join(
            position['folder'], 'Images',
            f'{summary["prefix"]}metadata.csv'
        ))
        assert rows['SizeT'] == str(position['size_t'])
        assert rows['SizeZ'] == str(position['size_z'])
        assert all(channel['files'] for channel in position['channels'])


@pytest.mark.parametrize('file_axes, sizes, written', [
    ('TYX', {'T': 3, 'Y': 1, 'X': 5}, (3, 1, 5)),
    ('TYX', {'T': 3, 'Y': 4, 'X': 1}, (3, 4, 1)),
    ('ZYX', {'Z': 2, 'Y': 1, 'X': 1}, (2, 1, 1)),
    ('TYX', {'T': 1, 'Y': 4, 'X': 5}, (4, 5)),
    ('TZYX', {'T': 2, 'Z': 3, 'Y': 4, 'X': 5}, (2, 3, 4, 5)),
    ('YX', {'Y': 1, 'X': 5}, (1, 5)),
])
def test_only_singleton_t_and_z_are_squeezed(file_axes, sizes, written,
                                             tmp_path):
    """GV-R3."""
    src = tmp_path / 'src'
    src.mkdir()
    synthetic.make_loose_tiff_set(str(src), pattern='a.tif',
                                  file_axes=file_axes, **sizes)
    dims = _dims()
    for letter, key in (('T', 'timepoints'), ('Z', 'z_slices')):
        if letter in file_axes:
            dims[key] = {'mode': 'in_file', 'axis': file_axes.index(letter)}
    result = run_core(_options(
        source_folder=str(src), destination_folder=str(tmp_path / 'dst'),
        dimensions=dims,
    ))
    data = tifffile.imread(result['image_files'][0])
    assert data.shape == written
    source = tifffile.imread(src / 'a.tif')
    assert np.array_equal(data.reshape(source.shape), source)


def test_segmentation_folder_warning(tmp_path, caplog):
    """GV-R4: one WARNING after the parity signals; logged without
    signals; nothing for an empty folder path."""
    options, placeholders = prepare_case('extra_segmentation_folder',
                                         tmp_path / 'one')
    recorder = harness_restructure.RecordingSignals(placeholders)
    run_core(options, signals=recorder)
    warnings = [event for event in recorder.events
                if event[0] == 'progress' and event[2] == 'WARNING']
    assert warnings == [[
        'progress',
        'The segmentation mask folder "<WORK>/segm" is ignored: '
        '"Re-structure image files" only re-structures the TIFF files of '
        'the source folder.', 'WARNING',
    ]]
    assert recorder.events.index(warnings[0]) == 2

    options, _ = prepare_case('extra_segmentation_folder', tmp_path / 'two')
    with caplog.at_level('WARNING', logger=restructure.__name__):
        run_core(options)
    assert 'segmentation mask folder' in caplog.text

    options, placeholders = prepare_case('RB', tmp_path / 'three')
    options['segmentation_mask_folder'] = ''
    recorder = harness_restructure.RecordingSignals(placeholders)
    run_core(options, signals=recorder)
    assert all(event[-1] != 'WARNING' for event in recorder.events)


# ------------------------------------------------------- registry predicates
def _check_golden(case_id, work):
    golden = scenarios.load_golden('restructure', case_id)
    actual = harness_restructure.run_restructure_case(case_id, str(work))
    return scenarios.assert_golden(case_id, golden, actual,
                                   expected_restructure_changes.CHANGES)


def test_registry_claims_create_new_only_from_position_1(tmp_path,
                                                         monkeypatch):
    """GV-R2: RD must give the plain run RB, not any successful run."""
    assert _check_golden('RD', tmp_path / 'ok')
    monkeypatch.setattr(restructure.planning, 'existing_position_numbers',
                        lambda folder: [2])
    with pytest.raises(AssertionError) as info:
        _check_golden('RD', tmp_path / 'shifted')
    for key in ('dst/Position_3/Images/Channel_1.tif', 'result', 'signals'):
        assert f'UNCLAIMED {key}' in str(info.value)


def _swap_y_and_x(array):
    return array.reshape(array.shape[:-2] + array.shape[:-3:-1])


def _add_one(array):
    return array + 1


@pytest.mark.parametrize('mutation', [_swap_y_and_x, _add_one])
@pytest.mark.parametrize('case_id',
                         expected_restructure_changes.GV_R3_CASES)
def test_registry_claims_only_the_kept_size1_axis(case_id, mutation,
                                                  tmp_path, monkeypatch):
    """GV-R3: the source (T, Y, X) shape and the OME sizes that follow
    from it; swapped Y/X or other pixel values stay unclaimed."""
    assert _check_golden(case_id, tmp_path / 'ok')
    squeeze = restructure._squeeze_time_and_z
    monkeypatch.setattr(restructure, '_squeeze_time_and_z',
                        lambda array: mutation(squeeze(array)))
    with pytest.raises(AssertionError, match=re.escape(
            f'UNCLAIMED {expected_restructure_changes.GV_R3_FILE}')):
        _check_golden(case_id, tmp_path / 'mutated')


# -------------------------------------------------------- GUI (Qt, GV-R5)
@conditions.requires_qt
def test_worker_thread_quits_after_an_error(tmp_path, monkeypatch):
    """GV-R5: `critical` also quits the restructure QThread (main left it
    running); `finished` is still not emitted on error."""
    import time
    from types import SimpleNamespace

    global _APP
    _APP = app = conditions.qapp()
    from qtpy.QtCore import QThread

    from cellacdc import dataReStruct

    threads, threadEvents = [], []

    class RecordingThread(QThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.finished.connect(lambda: threadEvents.append('finished'))
            threads.append(self)

    class FakeSetupDialog:
        def __init__(self, parent=None):
            self.cancel = False
            self.selectedOptions = _options(
                source_folder=str(tmp_path / 'missing'),
                destination_folder=str(tmp_path / 'dst'),
            )

        def exec_(self):
            pass

    monkeypatch.setattr(dataReStruct, 'QThread', RecordingThread)
    monkeypatch.setattr(dataReStruct.apps,
                        'ReStructTiffDataStructureSetupDialogue',
                        FakeSetupDialog)
    criticals, finished = [], []
    mainWin = SimpleNamespace(
        workerCritical=criticals.append,
        workerInitProgressbar=lambda total: None,
        workerUpdateProgressbar=lambda step: None,
        workerProgress=lambda *args: None,
        workerFinished=lambda *args: finished.append(args),
    )
    assert dataReStruct.run(mainWin) is True
    try:
        deadline = time.monotonic() + 10
        while 'finished' not in threadEvents \
                and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert threadEvents == ['finished']
        assert len(criticals) == 1
        _, error = criticals[0]
        assert isinstance(error, errors.SourceError)
        assert finished == []
    finally:
        if 'finished' not in threadEvents:
            threads[0].quit()
            threads[0].wait(5000)


@conditions.requires_qt
def test_dialog_ok_rejects_invalid_options(tmp_path, monkeypatch):
    """The dialog shows the RestructureOptionsError text and stays open (a
    backstop: its widgets already prevent such options)."""
    global _APP
    _APP = conditions.qapp()
    from cellacdc import apps

    monkeypatch.setattr(apps, 'restruct_files_recipes_path',
                        str(tmp_path / 'recipes'))
    dlg = apps.ReStructTiffDataStructureSetupDialogue()
    try:
        warnings = []
        monkeypatch.setattr(dlg, '_warn', warnings.append)
        dlg.sourceFolderPathControl.le.setText(str(tmp_path / 'src'))
        dlg.dstFolderPathControl.le.setText(str(tmp_path / 'dst'))
        dlg.selectedOptions.update(
            files=['a.tif'], overwrite_destination=True,
            create_new_position_destination=True,
        )
        dlg.ok_cb()
        assert dlg.cancel is True
        assert warnings == [
            'Choose either to overwrite the existing Position folders or '
            'to create new ones, not both.'
        ]
        dlg.selectedOptions['create_new_position_destination'] = False
        warnings.clear()
        dlg.ok_cb()
        assert warnings == [] and dlg.cancel is False
        assert dlg.selectedOptions['overwrite_destination'] is True
    finally:
        dlg.close()
        dlg.deleteLater()
