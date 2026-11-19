"""`convert()` with `InProcessBackend(reader_factory=FakeBioImage)`.

Covers every layout x {tif, h5, symlink} with decoded pixel origins, frame
order with time ranges, the interactive schedule driven by a scripted
"dialog" (ask -> trust -> incompatible re-ask, reuse, cancel, eye re-read
through `ConfirmContext.read_sample`, dialog-edited SizeT keeps an explicit
range), the upfront schedule (every `UsageError` leaves source and
destination unchanged, every raw file and scene checked against the plan
before any change, dry run, move-raw + symlinks into
`dst/raw_microscopy_files`), progress totals, `result.written` built from
the events only, files that do not fit the plan failing before their Images
folder is touched, error mapping and `partial_result`, and the backends'
construction (`InProcessBackend()`, `default_backend`).
"""
from __future__ import annotations

import configparser
import dataclasses
import io
import json
import logging
import os
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tifffile

from cellacdc.acdc_datastruct import (
    backends, convert, errors, metadata, planning, runner, sources, tasks,
    units, writers
)

from . import conditions, fake_bioio, synthetic

FACTORY = fake_bioio.FakeBioImage
RAW = sources.RAW_MICROSCOPY_FILES_FOLDERNAME
LAYOUT0, LAYOUT1, LAYOUT2 = (sources.RawLayout(n) for n in range(3))


@pytest.fixture(autouse=True)
def fresh_fake_log():
    FACTORY.reset_log()
    yield
    FACTORY.reset_log()


def make_backend() -> backends.InProcessBackend:
    return backends.InProcessBackend(reader_factory=FACTORY)


# ------------------------------------------------------------------ inputs
def fake(folder, name: str, **spec) -> str:
    os.makedirs(folder, exist_ok=True)
    spec.setdefault('channel_names', ['GFP', 'mCherry', 'Cy5'][:spec.get(
        'C', 2)])
    fake_bioio.write_fake_raw(
        os.path.join(folder, name), fake_bioio.make_spec(**spec)
    )
    return name


def layout0_inputs(src, **spec) -> list[str]:
    return [fake(src, 'exp.czi', **{'scenes': 3, **spec})]


def layout1_inputs(src, count: int = 3, overrides=None, **spec) -> list[str]:
    overrides = overrides or {}
    return [
        fake(src, f'pos{i + 1}.czi', **{
            's_offset': i, 'metadata_text': f'<file {i + 1}/>', **spec,
            **overrides.get(i + 1, {}),
        })
        for i in range(count)
    ]


def layout2_inputs(src, positions=(1, 2), channels=('GFP', 'mNeon-Green'),
                   skip=(), overrides=None, **spec) -> list[str]:
    """One fake file per (position, channel): s = position, c = channel
    index (decoded from the pixels)."""
    overrides = overrides or {}
    files = []
    for position in positions:
        for c, channel in enumerate(channels):
            if (position, channel) in skip:
                continue
            files.append(fake(src, f'ASY015_{position}_{channel}.czi', **{
                'C': 1, 's_offset': position, 'c_offset': c,
                'channel_names': [channel], **spec,
                **overrides.get((position, channel), {}),
            }))
    return sorted(files)


def make_request(src, dst, layout, names, **kwargs) -> convert.ConvertRequest:
    fields = dict(
        src_dir=str(src), dst_dir=str(dst), layout=layout, files=tuple(names)
    )
    fields.update(kwargs)
    return convert.ConvertRequest(**fields)


class Recorder:
    """ConvertCallbacks that record everything."""
    def __init__(self, confirm=None, **kwargs) -> None:
        self.logs: list[tuple[str, str]] = []
        self.inits: list[int] = []
        self.steps: list[int] = []
        self.callbacks = convert.ConvertCallbacks(
            log=lambda text, level='INFO': self.logs.append((level, text)),
            progress_init=self.inits.append,
            progress_step=self.steps.append,
            confirm=confirm or convert.confirm_from_metadata(), **kwargs
        )

    def messages(self, level: str = 'INFO') -> list[str]:
        return [text for lvl, text in self.logs if lvl == level]


def run(req, confirm=None, backend=None, **kwargs):
    recorder = Recorder(confirm, **kwargs)
    result = convert.convert(
        req, callbacks=recorder.callbacks, backend=backend or make_backend()
    )
    return result, recorder


def overrides(**kwargs):
    """Command line confirm hook with these `MetadataOverrides`."""
    return convert.confirm_from_metadata(metadata.MetadataOverrides(**kwargs))


# ----------------------------------------------------------------- outputs
def plane_origins(array: np.ndarray) -> list[tuple[int, int, int, int]]:
    planes = array.reshape(-1, *array.shape[-2:])
    return [synthetic.decode_pixel(plane[0, 0]) for plane in planes]


def expected_origins(s: int, c: int, frames, size_z: int = 2) -> list:
    return [(s, t, c, z) for t in frames for z in range(size_z)]


def read_ini(path: str) -> configparser.ConfigParser:
    ini = configparser.ConfigParser()
    ini.read(path)
    return ini


def images_files(dst, number: int) -> list[str]:
    return sorted(os.listdir(os.path.join(str(dst), f'Position_{number}',
                                          'Images')))


# ------------------------------------------------- layouts x image formats
LAYOUT_CASES = {
    # layout: (inputs, {position: (basename, [(channel, raw file, s, c)])})
    LAYOUT0: (layout0_inputs, {
        n: (f'exp_s{n}_', [('GFP', 'exp.czi', n - 1, 0),
                           ('mCherry', 'exp.czi', n - 1, 1)])
        for n in (1, 2, 3)
    }),
    LAYOUT1: (layout1_inputs, {
        n: (f'pos{n}_s{n}_', [('GFP', f'pos{n}.czi', n - 1, 0),
                              ('mCherry', f'pos{n}.czi', n - 1, 1)])
        for n in (1, 2, 3)
    }),
    LAYOUT2: (layout2_inputs, {
        n: (f'ASY015__s{n}_', [('GFP', f'ASY015_{n}_GFP.czi', n, 0),
                               ('mNeon_Green', f'ASY015_{n}_mNeon-Green.czi',
                                n, 1)])
        for n in (1, 2)
    }),
}


@pytest.mark.parametrize('output', ['tif', 'h5', 'symlink'])
@pytest.mark.parametrize('layout', list(LAYOUT_CASES))
def test_layouts_and_formats(tmp_path, layout, output):
    make_inputs, expected = LAYOUT_CASES[layout]
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = make_inputs(src)
    req = make_request(src, dst, layout, files,
                       use_symlinks=output == 'symlink')
    result, recorder = run(req, overrides(to_h5=output == 'h5'))

    assert result.ok and not result.cancelled
    assert sorted(os.listdir(dst)) == [f'Position_{n}' for n in expected]
    for (number, (basename, channels)), written in zip(
            expected.items(), result.written):
        assert written.output_number == number
        names = images_files(dst, number)
        if output == 'symlink':
            image_files = [f'{basename}symlink.ini']
        else:
            image_files = [f'{basename}{ch}.{output}' for ch, *_ in channels]
        assert names == sorted(
            [f'{basename}metadata.csv', f'{basename}metadataXML.txt']
            + image_files
        )
        folder = os.path.join(str(dst), f'Position_{number}', 'Images')
        for channel, raw_file, s, c in channels:
            if output == 'symlink':
                section = read_ini(os.path.join(folder, image_files[0]))[
                    f'channel_name.{channel}']
                assert section['source_filepath'] == str(src / raw_file)
                assert os.path.isabs(section['source_filepath'])
                assert section['series_index'] == str(
                    s if layout == LAYOUT0 else 0
                )
                continue
            path = os.path.join(folder, f'{basename}{channel}.{output}')
            if output == 'h5':
                with h5py.File(path, 'r') as h5:
                    data = h5['data'][()]
            else:
                data = tifffile.imread(path)
            assert plane_origins(data) == expected_origins(s, c, range(3))
        csv = writers.read_metadata_csv(
            os.path.join(folder, f'{basename}metadata.csv')
        )
        assert csv['basename'] == basename
        assert [csv['channel_0_name'], csv['channel_1_name']] == [
            channel for channel, *_ in channels
        ]
    assert len(recorder.inits) == 1
    assert sum(recorder.steps) == recorder.inits[0] == 2 * len(expected)


def test_time_range_frame_order(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=1, T=5)
    result, _ = run(
        make_request(src, dst, LAYOUT1, files), overrides(time_range=(2, 4))
    )
    folder = result.written[0].images_path
    data = tifffile.imread(os.path.join(folder, 'pos1_s1_mCherry.tif'))
    assert plane_origins(data) == expected_origins(0, 1, (1, 2, 3))
    csv = writers.read_metadata_csv(
        os.path.join(folder, 'pos1_s1_metadata.csv')
    )
    assert csv['SizeT'] == '3'
    assert result.jobs[0].time_range == (2, 4)


def test_layout0_position_selection_and_progress(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout0_inputs(src)
    result, recorder = run(
        make_request(src, dst, LAYOUT0, files),
        overrides(positions=frozenset({1, 3}))
    )
    assert sorted(os.listdir(dst)) == ['Position_1', 'Position_3']
    assert [s.input_id for s in result.skipped] == [2]
    assert [job.series for job in result.jobs] == [0, 2]
    # numPos * SizeC, one SizeC step per position (saved or skipped)
    assert recorder.inits == [6] and recorder.steps == [2, 2, 2]
    assert result.to_dict()['skipped'] == [2]


# ------------------------------------------------- interactive (GUI) schedule
def dialog_values(ctx: convert.ConfirmContext, button: str = 'ok',
                  **edits) -> SimpleNamespace:
    """What `QDialogMetadataXML` holds after `ok_cb` when the user changes
    `edits` (TimeIncrement in s, PhysicalSize in um, as `convertUnits`)."""
    r = ctx.resolved
    win = SimpleNamespace(
        to_h5=False, LensNA=r.lens_na, SizeT=r.size_t, SizeZ=r.size_z,
        SizeC=r.size_c, SizeS=r.size_s,
        TimeIncrement=units.time_increment_to_seconds(
            r.time_increment, r.time_increment_unit
        ),
        PhysicalSizeX=units.physical_size_to_um(
            r.physical_size_x, r.physical_size_unit
        ),
        PhysicalSizeY=units.physical_size_to_um(
            r.physical_size_y, r.physical_size_unit
        ),
        PhysicalSizeZ=units.physical_size_to_um(
            r.physical_size_z, r.physical_size_unit
        ),
        chNames=list(r.channel_names), saveChannels=[True] * r.size_c,
        emWavelens=list(r.em_wavelens), selectedPos=['All Positions'],
        addImageName=False, overWrite=button == 'reuse',
        trust=button == 'trust', timeRangeToSave=None,
    )
    for key, value in edits.items():
        setattr(win, key, value)
    if win.timeRangeToSave is None:
        # RangeSelector follows the SizeT spin box (setRange(1, SizeT))
        win.timeRangeToSave = (1, win.SizeT)
    return win


class ScriptedDialog:
    """The `confirm` hook as the GUI adapter implements it: optional eye
    re-read through `ctx.read_sample`, then the dialog values mapped by
    `metadata.settings_from_dialog_values`. `answers` are consumed in
    order: None = Cancel, else (button, edits)."""
    def __init__(self, *answers, read_sample: bool = False) -> None:
        self.answers = list(answers)
        self.read_sample = read_sample
        self.contexts: list[convert.ConfirmContext] = []
        self.samples: list[list[np.ndarray]] = []

    def __call__(self, ctx: convert.ConfirmContext):
        self.contexts.append(ctx)
        if self.read_sample:
            r = ctx.resolved
            self.samples.append(
                ctx.read_sample(r.size_c, r.size_t, r.size_z)
            )
        answer = self.answers.pop(0)
        if answer is None:
            return None
        button, edits = answer
        settings, apply_to_next = metadata.settings_from_dialog_values(
            dialog_values(ctx, button, **edits), file_size_t=ctx.raw.size_t
        )
        return convert.ConfirmResult(settings, apply_to_next)

    @property
    def filenames(self) -> list[str]:
        return [ctx.filename for ctx in self.contexts]


def csv_of(dst, number: int) -> dict[str, str]:
    folder = os.path.join(str(dst), f'Position_{number}', 'Images')
    name = next(n for n in os.listdir(folder) if n.endswith('metadata.csv'))
    return writers.read_metadata_csv(os.path.join(folder, name))


def interactive(src, dst, layout, files, **kwargs):
    return make_request(src, dst, layout, files, schedule='interactive',
                        on_existing=planning.OnExisting.ADD, **kwargs)


def test_interactive_trust_reopens_the_dialog_for_an_incompatible_file(
        tmp_path):
    """After "Trust metadata reader" a compatible file is converted without
    asking (reader values, units converted), a file with another channel
    count shows the dialog again."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, overrides={
        1: {'time_interval_s': 2.0}, 2: {'time_interval_s': 0.5},
        3: {'C': 3},
    })
    dialog = ScriptedDialog(('trust', {'LensNA': 0.9}), ('ok', {}))
    result, recorder = run(
        interactive(src, dst, LAYOUT1, files), confirm=dialog
    )
    assert dialog.filenames == ['pos1.czi', 'pos3.czi']
    assert dialog.contexts[0].previous is None
    trusted = dialog.contexts[1].previous
    assert (trusted.lens_na, trusted.time_increment_s) == (1.4, 0.5)
    assert csv_of(dst, 1)['LensNA'] == '0.9'
    assert csv_of(dst, 1)['TimeIncrement'] == '2.0'
    assert csv_of(dst, 2)['TimeIncrement'] == '0.5'
    assert csv_of(dst, 3)['channel_2_name'] == 'Cy5'
    assert [w.output_number for w in result.written] == [1, 2, 3]
    # GUI total of the first file (numPos * SizeC), one step per position
    assert recorder.inits == [6] and recorder.steps == [2, 2, 3]


def test_interactive_trust_reopens_for_a_time_range_that_does_not_fit(
        tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2, overrides={1: {'T': 5}, 2: {'T': 3}})
    dialog = ScriptedDialog(
        ('trust', {'timeRangeToSave': (2, 4)}), ('ok', {})
    )
    result, _ = run(interactive(src, dst, LAYOUT1, files), confirm=dialog)
    assert dialog.filenames == ['pos1.czi', 'pos2.czi']
    assert [job.time_range for job in result.jobs] == [(2, 4), (1, 3)]


def test_interactive_reuse(tmp_path):
    """'Use the above metadata for all the next positions': one dialog;
    each position keeps its own file's metadataXML (GV-07)."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src)
    dialog = ScriptedDialog(
        ('reuse', {'LensNA': 0.8, 'chNames': ['A', 'B']})
    )
    result, recorder = run(
        interactive(src, dst, LAYOUT1, files), confirm=dialog
    )
    assert dialog.filenames == ['pos1.czi']
    for number in (1, 2, 3):
        csv = csv_of(dst, number)
        assert (csv['LensNA'], csv['channel_0_name']) == ('0.8', 'A')
        folder = os.path.join(str(dst), f'Position_{number}', 'Images')
        with open(os.path.join(folder, f'pos{number}_s{number}_'
                               'metadataXML.txt'), encoding='utf-8') as file:
            assert file.read() == f'<file {number}/>'
    assert recorder.messages().count(convert.READING_METADATA_TEXT) == 3


def test_interactive_cancel_at_the_first_dialog(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src)
    before = conditions.tree_snapshot(str(src), str(dst))
    result, recorder = run(
        interactive(src, dst, LAYOUT1, files), confirm=ScriptedDialog(None)
    )
    assert result.cancelled and not result.ok
    assert result.written == [] and result.jobs == []
    assert recorder.inits == [] and recorder.steps == []
    assert conditions.tree_snapshot(str(src), str(dst)) == before


@pytest.mark.parametrize('file_t, edits, frames', [
    (10, {'SizeT': 5}, 5),
    (5, {'SizeT': 1, 'timeRangeToSave': (1, 5)}, 5),
    (5, {}, 5),
])
def test_interactive_dialog_edited_size_t_keeps_the_range(
        tmp_path, file_t, edits, frames):
    """S18: the frames the range widget holds are written, also after
    the user edited SizeT (incl. SizeT 1 with the widget hidden)."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=1, T=file_t)
    result, _ = run(interactive(src, dst, LAYOUT1, files),
                    confirm=ScriptedDialog(('ok', edits)))
    job = result.jobs[0]
    assert job.time_range == (1, frames)
    assert csv_of(dst, 1)['SizeT'] == str(frames)
    data = tifffile.imread(os.path.join(job.images_path, 'pos1_s1_GFP.tif'))
    assert plane_origins(data) == expected_origins(0, 0, range(frames))


def test_interactive_log_order_and_sample_data(tmp_path):
    """GUI log parity: 'Reading OME metadata...', the warning lines, then
    the Position lines from the writer; the dialog reads sample data
    through `ctx.read_sample` (port of the legacy child)."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2, T=5)
    dialog = ScriptedDialog(('ok', {}), ('ok', {}), read_sample=True)
    warnings = []
    result, recorder = run(
        interactive(src, dst, LAYOUT1, files), confirm=dialog,
        warn=warnings.append
    )
    images = [w.images_path for w in result.written]
    assert recorder.logs == [
        ('INFO', convert.READING_METADATA_TEXT),
        ('INFO', f'Position 1/2: saving data to {images[0]}...'),
        ('INFO', convert.READING_METADATA_TEXT),
        ('INFO', f'Position 2/2: saving data to {images[1]}...'),
    ]
    assert [w.field for w in warnings] == [
        'LensNA', 'TimeIncrement', 'TimeIncrementUnit', 'PhysicalSizeUnit',
        'Image Name'
    ] * 2
    assert result.warnings == warnings
    sample = dialog.samples[1]
    assert len(sample) == 2 and sample[1].shape == (4, 2, 16, 20)
    assert plane_origins(sample[1]) == expected_origins(1, 1, range(4))


def test_default_warn_logs_the_gui_lines(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=1)
    result, recorder = run(make_request(src, dst, LAYOUT1, files))
    expected = [
        line for warning in result.warnings for line in warning.gui_lines()
    ]
    assert recorder.messages('WARNING') == expected
    assert expected[1] == str(src / 'pos1.czi')


def test_interactive_moves_first(tmp_path):
    """GUI order (07e572d5): raw files are moved before the first dialog;
    the metadata is read from the moved file."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    seen = []

    def confirm(ctx):
        seen.append((ctx.raw_path, sorted(os.listdir(src))))
        return dialog(ctx)

    dialog = ScriptedDialog(('ok', {}), ('ok', {}))
    result, _ = run(
        interactive(src, dst, LAYOUT1, files, move_raw=True,
                    use_symlinks=True),
        confirm=confirm
    )
    raw_dir = dst / RAW
    assert seen == [(str(raw_dir / 'pos1.czi'), []),
                    (str(raw_dir / 'pos2.czi'), [])]
    assert result.moved == [
        (str(src / name), str(raw_dir / name)) for name in files
    ]


def test_interactive_layout2_one_dialog(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout2_inputs(src, positions=(1, 2, 3))
    dialog = ScriptedDialog(('ok', {}))
    result, recorder = run(
        interactive(src, dst, LAYOUT2, files), confirm=dialog
    )
    ctx = dialog.contexts[0]
    assert (ctx.filename, ctx.unit.label) == ('ASY015_1_GFP.czi', 'ASY015_')
    assert ctx.resolved.channel_names == ('GFP', 'mNeon-Green')
    assert [w.output_number for w in result.written] == [1, 2, 3]
    # GV-13: one SizeC step per position, exactly 100 %
    assert recorder.inits == [6] and recorder.steps == [2, 2, 2]


def test_interactive_settings_are_validated(tmp_path):
    """The core re-validates what the dialog returned (backstop)."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=1)
    dialog = ScriptedDialog(('ok', {'saveChannels': [False, False]}))
    with pytest.raises(errors.SettingsError, match='No channel selected'):
        run(interactive(src, dst, LAYOUT1, files), confirm=dialog)
    assert not dst.exists()


def test_interactive_metadata_read_error(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, overrides={2: {'fail': {'stage': 'open'}}})
    with pytest.raises(errors.MetadataReadError) as info:
        run(interactive(src, dst, LAYOUT1, files),
            confirm=ScriptedDialog(('trust', {})))
    err = info.value
    assert err.filename == 'pos2.czi'
    assert 'Traceback' in err.details
    assert 'injected open failure' in err.details
    partial = convert.partial_result(err)
    assert [w.output_number for w in partial.written] == [1]


def test_interactive_move_permission_error(tmp_path, monkeypatch):
    from cellacdc import io as acdc_io
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    move = acdc_io.move_raw_microscopy_file

    def move_or_deny(path, folder):
        if path.endswith('pos2.czi'):
            raise PermissionError('in use')
        return move(path, folder)

    monkeypatch.setattr(acdc_io, 'move_raw_microscopy_file', move_or_deny)
    with pytest.raises(errors.MoveRawFilesError) as info:
        run(interactive(src, dst, LAYOUT1, files, move_raw=True),
            confirm=ScriptedDialog())
    assert info.value.permission
    assert convert.partial_result(info.value).moved == [
        (str(src / 'pos1.czi'), str(dst / RAW / 'pos1.czi'))
    ]


# ------------------------------------------------ upfront (CLI) schedule
def _existing(dst, *numbers) -> None:
    for number in numbers:
        folder = dst / f'Position_{number}' / 'Images'
        folder.mkdir(parents=True)
        (folder / 'old.txt').write_text('old')


def _add_channel(ctx):
    settings, _ = metadata.settings_from_metadata(
        ctx.resolved, None, layout=ctx.layout, context=ctx.unit.label
    )
    return convert.ConfirmResult(dataclasses.replace(
        settings, size_c=3, channel_names=settings.channel_names + ('X',),
        save_channels=(True,) * 3, em_wavelens=(500.0,) * 3
    ))


def _select_the_other_file(ctx):
    """Two files: each selects the other file's input id, so no file has a
    Position to convert (backstop after the conflict check)."""
    settings, _ = metadata.settings_from_metadata(
        ctx.resolved, None, layout=ctx.layout, context=ctx.unit.label
    )
    other = 3 - ctx.unit.file_ordinal
    return convert.ConfirmResult(dataclasses.replace(
        settings, selected_positions=frozenset({other})
    ))


def _usage_case(name, src, dst):
    """(request kwargs, confirm, expected error type, match)."""
    if name == 'existing positions':
        _existing(dst, 1)
        return {}, None, errors.ExistingPositionsError, 'Position_1'
    if name == 'overwrite, time range beyond the file':
        _existing(dst, 1, 2)
        return ({'on_existing': planning.OnExisting.OVERWRITE},
                overrides(time_range=(1, 9)), errors.SettingsError,
                'exceeds')
    if name == 'nothing to convert':
        return ({}, _select_the_other_file, errors.SettingsError,
                'No positions selected')
    if name == 'create-new explicit start':
        _existing(dst, 1, 2)
        return ({'on_existing': planning.OnExisting.CREATE_NEW,
                 'start_pos_n': 2}, None, errors.ExistingPositionsError,
                'Position_2')
    if name == 'unknown positions':
        return {}, overrides(positions=frozenset({7})), errors.PlanError, '7'
    if name == 'no position selected':
        return ({}, overrides(positions=frozenset()), errors.SettingsError,
                'No position selected')
    if name == 'time range beyond the file':
        return ({}, overrides(time_range=(1, 9)), errors.SettingsError,
                'exceeds')
    if name == 'unknown channel to save':
        return ({}, overrides(save_channels=('YFP',)), errors.SettingsError,
                'YFP')
    if name == 'image name missing':
        return ({}, overrides(add_image_name=True), errors.SettingsError,
                'image name')
    if name == 'missing file':
        return ({'files': ('pos1.czi', 'ghost.czi')}, None,
                errors.SourceError, 'ghost.czi')
    if name == 'relative source':
        return ({'src_dir': 'raw'}, None, errors.UsageError, 'absolute')
    if name == 'destination is a file':
        dst.write_text('x')
        return {}, None, errors.PlanError, 'not a folder'
    if name == 'move target exists':
        (dst / RAW).mkdir(parents=True)
        (dst / RAW / 'pos2.czi').write_text('x')
        return {'move_raw': True}, None, errors.PlanError, 'pos2.czi'
    if name == 'dry run interactive':
        return ({'dry_run': True, 'schedule': 'interactive'}, None,
                errors.UsageError, 'upfront')
    raise AssertionError(name)


USAGE_CASES = (
    'existing positions', 'overwrite, time range beyond the file',
    'nothing to convert', 'create-new explicit start', 'unknown positions',
    'no position selected', 'time range beyond the file',
    'unknown channel to save', 'image name missing', 'missing file',
    'relative source', 'destination is a file', 'move target exists',
    'dry run interactive',
)


@pytest.mark.parametrize('name', USAGE_CASES)
def test_usage_errors_change_nothing(tmp_path, name):
    """Every UsageError is raised before the source or the destination is
    changed. Every case also renames and moves the raw files, so a change
    made before the error alters the snapshot."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    kwargs, confirm, error_type, match = _usage_case(name, src, dst)
    kwargs = {
        'move_raw': True, 'renames': {'pos1.czi': 'pos_1.czi'}, **kwargs
    }
    req = make_request(src, dst, LAYOUT1, files, **kwargs)
    before = conditions.tree_snapshot(str(src), str(dst))
    with pytest.raises(error_type, match=match) as info:
        run(req, confirm)
    assert conditions.tree_snapshot(str(src), str(dst)) == before
    partial = convert.partial_result(info.value)
    assert partial.written == [] and partial.moved == []
    assert partial.renamed == [] and partial.deletions == []


def test_usage_errors_cover_the_conflict_check(tmp_path):
    """The renames and moves of the usage cases are real: the same request
    without the conflicting Position converts and changes the source."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    result, _ = run(make_request(
        src, dst, LAYOUT1, files, move_raw=True,
        renames={'pos1.czi': 'pos_1.czi'}
    ))
    assert os.listdir(src) == []
    assert sorted(os.listdir(dst / RAW)) == ['pos2.czi', 'pos_1.czi']
    assert result.renamed == [(str(src / 'pos1.czi'), str(src / 'pos_1.czi'))]
    assert len(result.moved) == 2


LAYOUT2_USAGE_CASES = {
    'size_c above the tokens': (
        {}, _add_channel, errors.SettingsError, 'cannot exceed the 2'
    ),
    'missing channel file': (
        {'skip': ((2, 'mNeon-Green'),)}, None,
        errors.MissingChannelFilesError, 'position 2, channel mNeon-Green'
    ),
    'no pattern': ({'bad_names': True}, None, errors.FilenamePatternError,
                   'position number'),
}


@pytest.mark.parametrize('name', list(LAYOUT2_USAGE_CASES))
def test_layout2_usage_errors_change_nothing(tmp_path, name):
    inputs, confirm, error_type, match = LAYOUT2_USAGE_CASES[name]
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    if inputs.get('bad_names'):
        files = [fake(src, 'a.czi', C=1), fake(src, 'b.czi', C=1)]
    else:
        files = layout2_inputs(src, skip=inputs.get('skip', ()))
    before = conditions.tree_snapshot(str(src), str(dst))
    with pytest.raises(error_type, match=match):
        run(make_request(src, dst, LAYOUT2, files, move_raw=True), confirm)
    assert conditions.tree_snapshot(str(src), str(dst)) == before


def test_layout0_needs_one_file(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    with pytest.raises(errors.SourceError, match='exactly one file'):
        convert.prepare(make_request(src, dst, LAYOUT0, files))


def test_dry_run_writes_nothing_and_matches_the_run(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src)
    _existing(dst, 1)
    kwargs = dict(on_existing=planning.OnExisting.OVERWRITE, move_raw=True)
    before = conditions.tree_snapshot(str(src), str(dst))
    plan, recorder = run(make_request(src, dst, LAYOUT1, files, dry_run=True,
                                      **kwargs))
    assert conditions.tree_snapshot(str(src), str(dst)) == before
    assert plan.dry_run and plan.ok and plan.written == []
    assert recorder.inits == [] and recorder.messages('INFO') == [
        'Reading the metadata of 3 file(s)...',
        'Checking the raw files of 3 position(s)...',
    ]
    images_1 = str(dst / 'Position_1' / 'Images')
    assert plan.deletions == [images_1]
    assert plan.moved == [
        (str(src / name), str(dst / RAW / name)) for name in files
    ]
    planned = plan.to_dict()
    assert json.loads(json.dumps(planned)) == planned
    assert planned['dry_run'] and planned['deleted_images_folders'] == [
        images_1
    ]

    done, _ = run(make_request(src, dst, LAYOUT1, files, **kwargs))
    executed = done.to_dict()
    assert [dict(p, files=[]) for p in executed['positions']] == (
        planned['positions']
    )
    assert executed['moved'] == planned['moved']
    assert not os.path.exists(os.path.join(images_1, 'old.txt'))


def test_dry_run_lists_the_other_files(tmp_path):
    """`--other-files move` changes the source: the plan lists each other
    file (source, target, mode) and the result the ones moved."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout2_inputs(src)
    (src / 'ASY015_1_notes.txt').write_text('notes')
    expected = [{
        'src': str(src / 'ASY015_1_notes.txt'),
        'dst': str(dst / 'Position_1' / 'Images' / 'ASY015__s1_notes.txt'),
        'mode': 'move',
    }]
    req = make_request(src, dst, LAYOUT2, files, other_files='move')
    plan, _ = run(dataclasses.replace(req, dry_run=True))
    planned = plan.to_dict()['positions']
    assert [p['other_files'] for p in planned] == [expected, []]

    done, _ = run(req)
    executed = done.to_dict()['positions']
    assert [p['other_files'] for p in executed] == [expected, []]
    assert not (src / 'ASY015_1_notes.txt').exists()
    assert [dict(p, files=[]) for p in executed] == planned


def test_result_omits_other_files_that_failed(tmp_path, monkeypatch):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout2_inputs(src)
    (src / 'ASY015_1_notes.txt').write_text('notes')

    def copy_fails(src_path, dst_path):
        raise OSError('disk full')

    monkeypatch.setattr(writers.shutil, 'copy', copy_fails)
    result, recorder = run(make_request(src, dst, LAYOUT2, files,
                                        other_files='copy'))
    assert [p['other_files'] for p in result.to_dict()['positions']] == [
        [], []
    ]
    assert any('disk full' in text for text in recorder.messages('WARNING'))


def test_move_raw_and_symlinks_point_into_the_destination(tmp_path):
    """symlink.ini holds the absolute path of the moved raw file."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout2_inputs(src)
    (src / 'ASY015_1_notes.txt').write_text('notes')
    result, _ = run(make_request(src, dst, LAYOUT2, files, move_raw=True,
                                 use_symlinks=True, other_files='copy'))
    raw_dir = dst / RAW
    assert sorted(os.listdir(raw_dir)) == files
    assert os.listdir(src) == ['ASY015_1_notes.txt']
    ini = read_ini(os.path.join(result.written[0].images_path,
                                'ASY015__s1_symlink.ini'))
    source = ini['channel_name.mNeon_Green']['source_filepath']
    assert os.path.isabs(source)
    assert source == str(raw_dir / 'ASY015_1_mNeon-Green.czi')
    assert result.written[0].files[-1] == os.path.join(
        result.written[0].images_path, 'ASY015__s1_notes.txt'
    )
    assert result.written[0].source_file == 'ASY015_1_GFP.czi'


def test_on_existing_overwrite_and_create_new(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    _existing(dst, 1, 2)
    (dst / 'Position_1' / 'notes.txt').write_text('keep')
    result, _ = run(make_request(src, dst, LAYOUT1, files,
                                 on_existing=planning.OnExisting.OVERWRITE))
    assert result.deletions == [
        str(dst / f'Position_{n}' / 'Images') for n in (1, 2)
    ]
    assert 'old.txt' not in images_files(dst, 1)
    assert (dst / 'Position_1' / 'notes.txt').read_text() == 'keep'

    result, _ = run(make_request(src, dst, LAYOUT1, files,
                                 on_existing=planning.OnExisting.CREATE_NEW))
    assert [w.output_number for w in result.written] == [3, 4]
    assert images_files(dst, 3)[0] == 'pos1_s3_GFP.tif'
    assert result.deletions == []


def test_files_that_do_not_fit_fail_before_their_images_folder(tmp_path):
    """GV-05: the frames of position 2 do not fit (the plan used the
    first file's SizeT); nothing is cleared or written."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout2_inputs(src, overrides={(2, 'GFP'): {'T': 2}})
    _existing(dst, 1, 2)
    before = conditions.tree_snapshot(str(dst))
    with pytest.raises(ValueError, match='does not fit the 2 frame') as info:
        run(make_request(src, dst, LAYOUT2, files,
                         on_existing=planning.OnExisting.OVERWRITE))
    assert conditions.tree_snapshot(str(dst)) == before
    partial = convert.partial_result(info.value)
    assert partial.written == [] and partial.deletions == []
    assert partial.to_dict()['deleted_images_folders'] == []


def _short_second_scene(src, name: str = 'exp (1).czi') -> str:
    """Single-multi-pos file whose scene 1 has fewer frames (2) than scene
    0 (4), which the plan is read from."""
    os.makedirs(src, exist_ok=True)
    scene = {'C': 1, 'Z': 1, 'Y': 8, 'X': 8}
    fake_bioio.write_fake_raw(os.path.join(src, name), {
        'scenes': [{'T': 4, **scene}, {'T': 2, **scene}],
        'channel_names': ['GFP'],
    })
    return name


class CheckSpy(backends.InProcessBackend):
    """Fake-reader backend that records the raw paths of the jobs passed to
    `check_positions` and `save_positions`."""
    def __init__(self) -> None:
        super().__init__(reader_factory=FACTORY)
        self.calls: list[tuple[str, list[str]]] = []

    def _record(self, task, jobs) -> None:
        self.calls.append((task, [
            channel.raw_path for job in jobs for channel in job.channels
        ]))

    def check_positions(self, jobs):
        self._record('check_positions', jobs)
        return super().check_positions(jobs)

    def save_positions(self, jobs, texts, *, on_event):
        self._record('save_positions', jobs)
        return super().save_positions(jobs, texts, on_event=on_event)


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('case', [
    'one-per-channel, move', 'single-multi-pos, rename and move',
    'single-multi-pos, rename, symlinks', 'single-multi-pos, h5',
])
def test_upfront_checks_every_raw_file_before_any_change(
        tmp_path, case, dry_run):
    """The plan of a unit comes from one file (one-per-channel) or scene
    (single-multi-pos); every raw file and scene is checked against it
    before anything is renamed, moved or created, also for a dry run:
    SettingsError naming the file at its current path, SRC unchanged, no
    DST."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    kwargs, confirm = {'dry_run': dry_run}, None
    if case.startswith('one-per-channel'):
        files = layout2_inputs(src, overrides={(2, 'GFP'): {'T': 2}})
        layout, short, frames = LAYOUT2, 'ASY015_2_GFP.czi', '1-3'
        series = 0
        kwargs['move_raw'] = True
    else:
        files = [_short_second_scene(src)]
        layout, short, frames, series = LAYOUT0, files[0], '1-4', 1
        if 'rename' in case:
            kwargs['renames'] = {files[0]: 'exp_1_.czi'}
        kwargs['move_raw'] = 'move' in case
        kwargs['use_symlinks'] = 'symlinks' in case
        confirm = overrides(to_h5='h5' in case)
    before = conditions.tree_snapshot(str(src))
    with pytest.raises(errors.SettingsError) as info:
        run(make_request(src, dst, layout, files, **kwargs), confirm)
    assert info.value.message == (
        f'Time range {frames} (1-based, inclusive) does not fit the 2 '
        f'frame(s) of "{src / short}" (series {series}).'
    )
    assert '--time-range' in info.value.hint
    assert conditions.tree_snapshot(str(src)) == before
    assert not dst.exists()
    partial = convert.partial_result(info.value)
    assert (partial.moved, partial.renamed, partial.written) == ([], [], [])


def test_upfront_checks_the_raw_files_at_their_current_paths(tmp_path):
    """The check task gets the source paths (nothing renamed or moved yet),
    the save task the final ones; the interactive (GUI) schedule has no
    separate check."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    renames = {'pos1.czi': 'pos_1.czi', 'pos2.czi': 'pos_2.czi'}
    spy = CheckSpy()
    result, recorder = run(make_request(
        src, dst, LAYOUT1, files, renames=renames, move_raw=True
    ), backend=spy)
    assert len(result.written) == 2
    assert spy.calls == [
        ('check_positions', [str(src / name) for name in files
                             for _ in range(2)]),
        ('save_positions', [str(dst / RAW / renames[name]) for name in files
                            for _ in range(2)]),
    ]
    assert recorder.messages('INFO')[1] == (
        'Checking the raw files of 2 position(s)...'
    )
    gui_src = tmp_path / 'gui_src'
    files = layout1_inputs(gui_src, count=2)
    spy = CheckSpy()
    run(interactive(gui_src, tmp_path / 'gui_dst', LAYOUT1, files),
        confirm=ScriptedDialog(('ok', {}), ('ok', {})), backend=spy)
    assert [task for task, _ in spy.calls] == ['save_positions'] * 2


def test_partial_result_lists_only_the_deleted_images_folders(tmp_path):
    """OVERWRITE: the result lists the Images folders actually deleted (the
    worker's 'images_deleted' events), not the planned ones."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2, overrides={
        1: {'fail': {'stage': 'read'}}
    })
    _existing(dst, 1, 2)
    with pytest.raises(RuntimeError, match='injected read failure') as info:
        run(make_request(src, dst, LAYOUT1, files,
                         on_existing=planning.OnExisting.OVERWRITE))
    partial = convert.partial_result(info.value)
    images_1 = str(dst / 'Position_1' / 'Images')
    assert partial.written == [] and partial.deletions == [images_1]
    assert partial.to_dict()['deleted_images_folders'] == [images_1]
    assert 'old.txt' not in os.listdir(images_1)
    assert images_files(dst, 2) == ['old.txt']


@pytest.mark.parametrize('failure', [PermissionError, KeyboardInterrupt])
def test_partial_result_after_a_failed_move(tmp_path, monkeypatch, failure):
    """Renames and moves are recorded one by one: a failed (or interrupted)
    move of the second file reports both renames and the first move."""
    from cellacdc import io as acdc_io
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    renames = {'pos1.czi': 'pos_1.czi', 'pos2.czi': 'pos_2.czi'}
    move = acdc_io.move_raw_microscopy_file

    def move_or_fail(path, folder):
        if path.endswith('pos_2.czi'):
            raise failure('in use')
        return move(path, folder)

    monkeypatch.setattr(acdc_io, 'move_raw_microscopy_file', move_or_fail)
    expected = (
        errors.MoveRawFilesError if failure is PermissionError else failure
    )
    with pytest.raises(expected) as info:
        run(make_request(src, dst, LAYOUT1, files, renames=renames,
                         move_raw=True))
    assert os.listdir(src) == ['pos_2.czi']
    first_move = (str(src / 'pos_1.czi'), str(dst / RAW / 'pos_1.czi'))
    partial = convert.partial_result(info.value)
    assert partial.renamed == [
        (str(src / old), str(src / new)) for old, new in renames.items()
    ]
    assert partial.moved == [first_move]
    assert partial.to_dict()['renamed'] == [
        list(pair) for pair in partial.renamed
    ]
    if failure is PermissionError:
        assert info.value.moved == [first_move]
        assert info.value.failed == str(src / 'pos_2.czi')


def test_partial_result_after_a_failed_rename(tmp_path, monkeypatch):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)
    renames = {'pos1.czi': 'pos_1.czi', 'pos2.czi': 'pos_2.czi'}
    rename = os.rename

    def rename_or_fail(old, new):
        if old.endswith('pos2.czi'):
            raise PermissionError('in use')
        return rename(old, new)

    monkeypatch.setattr(writers.os, 'rename', rename_or_fail)
    with pytest.raises(errors.MoveRawFilesError) as info:
        run(make_request(src, dst, LAYOUT1, files, renames=renames,
                         move_raw=True))
    assert sorted(os.listdir(src)) == ['pos2.czi', 'pos_1.czi']
    partial = convert.partial_result(info.value)
    assert partial.renamed == [(str(src / 'pos1.czi'), str(src / 'pos_1.czi'))]
    assert partial.moved == [] and info.value.moved == []


def test_written_comes_from_the_events_only(tmp_path):
    """result.written == the 'position_done' events; a different
    return value of the backend is only reported."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=2)

    class Spy(backends.InProcessBackend):
        events: list = []

        def save_positions(self, jobs, texts, *, on_event):
            def record(event):
                self.events.append(event)
                on_event(event)
            written = super().save_positions(jobs, texts, on_event=record)
            return written[:1]

    spy = Spy(reader_factory=FACTORY)
    result, recorder = run(make_request(src, dst, LAYOUT1, files),
                           backend=spy)
    done = [e for e in spy.events if e['kind'] == 'position_done']
    assert [
        (w.input_id, w.output_number, w.images_path, list(w.files))
        for w in result.written
    ] == [
        (e['input_id'], e['output_number'], e['images_path'], e['files'])
        for e in done
    ]
    assert len(result.written) == 2
    assert any('do not match' in text for text in recorder.messages(
        'WARNING'))


def test_cancellation(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout1_inputs(src, count=1)
    for schedule in ('upfront', 'interactive'):
        with pytest.raises(errors.ConversionCancelled):
            run(make_request(src, dst, LAYOUT1, files, schedule=schedule),
                is_cancelled=lambda: True)
    assert FACTORY.log == [] and not dst.exists()


def test_confirm_is_required(tmp_path):
    req = make_request(tmp_path, tmp_path / 'dst', LAYOUT1, ())
    with pytest.raises(ValueError, match='confirm'):
        convert.convert(req, callbacks=convert.ConvertCallbacks(),
                        backend=make_backend())


def test_prepare_is_read_only_and_idempotent(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout2_inputs(src)
    _existing(dst, 4)
    req = make_request(src, dst, LAYOUT2, files, move_raw=True,
                       on_existing=planning.OnExisting.CREATE_NEW)
    before = conditions.tree_snapshot(str(src), str(dst))
    first = convert.prepare(req)
    assert convert.prepare(req) == first
    assert conditions.tree_snapshot(str(src), str(dst)) == before
    assert (first.existing, first.start_pos_n) == ((4,), 5)
    assert first.one_per_channel.position_ids == (1, 2)
    assert len(first.mutations.moves) == 4


def test_read_file_metadata(tmp_path):
    path = str(tmp_path / fake(tmp_path, 'a.czi', T=4, time_interval_s=0.5))
    raw, resolved, warnings = convert.read_file_metadata(
        path, backend=make_backend()
    )
    assert (raw.path, resolved.size_t, resolved.time_increment) == (
        path, 4, 0.5
    )
    assert [w.field for w in warnings] == [
        'LensNA', 'PhysicalSizeUnit', 'Image Name'
    ]
    with pytest.raises(errors.UsageError, match='absolute'):
        convert.read_file_metadata('a.czi', backend=make_backend())
    bad = str(tmp_path / fake(tmp_path, 'b.czi', fail={'stage': 'open'}))
    with pytest.raises(errors.MetadataReadError) as info:
        convert.read_file_metadata(bad, backend=make_backend())
    assert info.value.filename == 'b.czi'


def test_confirm_from_metadata_reports_warnings(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = [fake(src, 'a.czi', channel_names=['GFP', '?'])]
    texts = []
    confirm = convert.confirm_from_metadata(on_warning=texts.append)
    result, _ = run(make_request(src, dst, LAYOUT1, files), confirm)
    assert result.jobs[0].metadata_csv.channel_names == ('GFP', 'channel_1')
    assert len(texts) == 1 and 'channel_1' in texts[0]


def test_confirm_from_metadata_ignores_unselected_files(tmp_path):
    """One file per position: a file the positions do not select is
    skipped without warnings, and neither overrides that do not fit it
    (frames, number of channel names) nor its own metadata (duplicate
    channel names, which --channels would fix) reject the run."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = [
        fake(src, 'pos1.czi', T=3, C=1, channel_names=['']),
        fake(src, 'pos2.czi', T=5, s_offset=1),
    ]
    texts = []
    confirm = convert.confirm_from_metadata(metadata.MetadataOverrides(
        positions=frozenset({2}), time_range=(4, 5),
        channel_names=('A', 'B')
    ), on_warning=texts.append)
    result, _ = run(make_request(src, dst, LAYOUT1, files), confirm)
    assert [(job.output_number, job.time_range, job.metadata_csv.channel_names)
            for job in result.jobs] == [(2, (4, 5), ('A', 'B'))]
    assert [(p.input_id, p.size_c) for p in result.skipped] == [(1, 1)]
    assert texts == []
    assert not (dst / 'Position_1').exists()
    # Duplicate channel names in the unselected file, with --channels and a
    # time range that does not fit it, or without any other override
    dup = tmp_path / 'dup'
    dup_files = [
        fake(dup, 'pos1.czi', T=3, channel_names=['GFP', 'GFP']),
        fake(dup, 'pos2.czi', T=5, s_offset=1),
    ]
    for i, kwargs in enumerate([
        dict(time_range=(4, 5), channel_names=('A', 'B')), {},
    ]):
        result, _ = run(
            make_request(dup, tmp_path / f'dup_dst{i}', LAYOUT1, dup_files),
            overrides(positions=frozenset({2}), **kwargs)
        )
        assert [job.output_number for job in result.jobs] == [2]
        assert [(p.input_id, p.size_c) for p in result.skipped] == [(1, 2)]
    # A selected file that does not fit is still an error, naming the file
    with pytest.raises(errors.SettingsError,
                       match='^pos1.czi: Invalid --time-range 4:5'):
        run(make_request(src, tmp_path / 'dst2', LAYOUT1, files), overrides(
            positions=frozenset({1}), time_range=(4, 5)
        ))


# ---------------------------------------------------------------- backends
def test_in_process_backend_construction():
    assert backends.InProcessBackend().reader_factory is None
    backend = backends.InProcessBackend(
        reader_factory=fake_bioio.FAKE_BIOIMAGE_PATH
    )
    assert backend.reader_factory is FACTORY


def test_in_process_backend_is_strict_json(tmp_path, monkeypatch):
    """Like the worker child: a numpy scalar in an event is an error."""
    def bad_task(params, ctx):
        ctx.emit({'kind': 'log', 'value': np.int64(1)})
        return {}

    monkeypatch.setitem(tasks.TASKS, 'init_reader', bad_task)
    with pytest.raises(TypeError):
        make_backend().init_reader(str(tmp_path / 'a.czi'))


def test_default_backend_for_the_gui(monkeypatch):
    monkeypatch.delenv(runner.TEST_HOOKS_ENV, raising=False)
    backend = backends.default_backend(for_gui=True)
    assert isinstance(backend, backends.SubprocessBackend)
    assert backend.reader_factory is None and backend.options is None
    console = io.StringIO()
    monkeypatch.setattr(sys, '__stderr__', console)
    backend.on_output('child line')
    assert console.getvalue() == 'child line\n'
    monkeypatch.setattr(sys, '__stderr__', None)
    backend.on_output('no console')

    class Broken:
        def write(self, text):
            raise OSError('closed')

    monkeypatch.setattr(sys, '__stderr__', Broken())
    backend.on_output('broken console')


def test_default_backend_for_the_command_line(monkeypatch, caplog):
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')
    monkeypatch.setenv(
        backends.READER_FACTORY_ENV, fake_bioio.FAKE_BIOIMAGE_PATH
    )
    options = runner.RunnerOptions(keep_run_dir=True)
    backend = backends.default_backend(for_gui=False, options=options)
    assert backend.reader_factory == fake_bioio.FAKE_BIOIMAGE_PATH
    assert backend.options is options
    with caplog.at_level(logging.DEBUG, logger=backends.__name__):
        backend.on_output('child line')
    assert caplog.messages == ['child line']
    lines = []
    custom = backends.default_backend(for_gui=False, on_output=lines.append)
    assert custom.on_output == lines.append
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '0')
    assert backends.default_backend(for_gui=False).reader_factory is None
