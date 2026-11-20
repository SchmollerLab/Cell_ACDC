"""Worker tasks in-process with `FakeBioImage`.

`read_metadata` (batch, error context), `read_sample` (parity with the
legacy `_read_sample_data.py` loop), `check_positions` (every file and
series checked, nothing written, readers released), `save_positions`
(legacy order, events, checks of every job before the first write, a partly
deleted Images folder reported, output-existence check, reader cache),
`ping`/`init_reader`, the task table and the contract with `_worker.main`
(production ops, `TaskContext` keywords, `TaskError.context`).
"""
from __future__ import annotations

import configparser
import dataclasses
import io
import json
import os
import re

import h5py
import numpy as np
import pytest
import tifffile

import cellacdc
from cellacdc.acdc_bioio_bioformats import reader, writer
from cellacdc.acdc_datastruct import (
    _worker, metadata, planning, runner, tasks, writers
)

from . import fake_bioio, synthetic

FACTORY = fake_bioio.FakeBioImage


@pytest.fixture(autouse=True)
def fresh_fake_log():
    FACTORY.reset_log()
    yield
    FACTORY.reset_log()


def make_ctx(tmp_path, events: list | None = None) -> tasks.TaskContext:
    out_dir = tmp_path / 'run'
    out_dir.mkdir(exist_ok=True)
    sink = [] if events is None else events
    return tasks.TaskContext(
        out_dir=str(out_dir), emit=sink.append, reader_factory=FACTORY
    )


def fake_raw(folder, name: str, **spec) -> str:
    path = os.path.join(str(folder), name)
    fake_bioio.write_fake_raw(path, fake_bioio.make_spec(**spec))
    return path


def closes_match_opens() -> bool:
    opened = [event[1] for event in FACTORY.log if event[0] == 'open']
    closed = [event[1] for event in FACTORY.log if event[0] == 'close']
    return sorted(opened) == sorted(closed)


def make_job(images_path: str, raw_path: str, *, input_id: int = 1,
             output_number: int = 1, num_positions: int = 1, series: int = 0,
             channels=(('GFP', 0), ('RFP', 1)), time_range=(1, 3),
             size_z: int | None = 2, output: str = 'tif',
             lazy_load: bool = True, clear_images: bool = False,
             other_files=(), stem: str = 'exp') -> planning.PositionJob:
    pos = str(output_number).zfill(len(str(num_positions)))
    basename = f'{stem}_s{pos}_'
    names = tuple(name for name, _ in channels)
    return planning.PositionJob(
        unit_index=0, input_id=input_id, output_number=output_number,
        num_positions=num_positions, pos_str=pos, images_path=images_path,
        clear_images=clear_images, basename=basename, filename_no_ext=stem,
        series=series, source_file=os.path.basename(raw_path),
        channels=tuple(
            planning.ChannelSource(name, c, raw_path, index)
            for c, (name, index) in enumerate(channels)
        ),
        metadata_csv=planning.MetadataCsvSpec(
            lens_na=1.4, saved_size_t=time_range[1] - time_range[0] + 1,
            size_z=size_z or 1, time_increment_s=2.0,
            physical_size_z_um=0.5, physical_size_y_um=0.1,
            physical_size_x_um=0.1, basename=basename, channel_names=names,
            save_channels=(True,) * len(names),
            em_wavelens=(0.0,) * len(names),
        ),
        metadata_xml_ref='xml:0', time_range=tuple(time_range),
        size_z=size_z, output=output, lazy_load=lazy_load,
        time_increment_s=2.0, physical_size_zyx_um=(0.5, 0.1, 0.1),
        other_files=tuple(other_files),
    )


def save(jobs, ctx, texts=None) -> dict:
    params = {
        'jobs': [job.to_dict() for job in jobs],
        'texts': {'xml:0': '<xml/>'} if texts is None else texts,
    }
    return tasks.save_positions(json.loads(json.dumps(params)), ctx)


def plane_origins(array: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Decoded (s, t, c, z) of every plane (planes = all but the last two
    axes, in C order)."""
    planes = array.reshape(-1, *array.shape[-2:])
    return [synthetic.decode_pixel(plane[0, 0]) for plane in planes]


# ------------------------------------------------------------- task table
def test_task_table():
    assert set(tasks.TASKS) == {
        'ping', 'init_reader', 'read_metadata', 'read_sample',
        'check_positions', 'save_positions',
    }
    assert tasks.TASKS['check_positions'] is tasks.check_positions
    assert tasks.TASKS['save_positions'] is tasks.save_positions


def test_task_context_is_built_with_keywords(tmp_path):
    """`_worker` builds `ops.TaskContext(out_dir=, emit=, reader_factory=)`."""
    ctx = tasks.TaskContext(
        out_dir=str(tmp_path), emit=print, reader_factory=FACTORY
    )
    assert (ctx.out_dir, ctx.emit, ctx.reader_factory) == (
        str(tmp_path), print, FACTORY
    )
    assert tasks.TaskContext(out_dir='x', emit=print).reader_factory is None


def test_task_error_context_is_a_mapping_copy():
    context = {'path': 'a.czi'}
    err = tasks.TaskError('boom', context=context)
    assert err.context == context and err.context is not context
    assert str(err) == 'boom'
    assert tasks.TaskError('boom').context == {}


def test_ping(tmp_path):
    result = tasks.ping({}, make_ctx(tmp_path))
    assert result['protocol'] == runner.PROTOCOL
    assert runner.same_path(result['cellacdc_file'], cellacdc.__file__)
    assert result['pid'] == os.getpid()
    assert json.loads(json.dumps(result)) == result


def test_init_reader_opens_and_closes(tmp_path):
    path = fake_raw(tmp_path, 'a.czi')
    assert tasks.init_reader({'path': path}, make_ctx(tmp_path)) == {
        'path': path
    }
    assert [event[:2] for event in FACTORY.log] == [
        ('open', path), ('close', path)
    ]


def test_init_reader_failure_propagates(tmp_path):
    path = fake_raw(tmp_path, 'a.czi', fail={'stage': 'open'})
    with pytest.raises(RuntimeError, match='injected open failure'):
        tasks.init_reader({'path': path}, make_ctx(tmp_path))


# ---------------------------------------------------------- read_metadata
def test_read_metadata_batch(tmp_path):
    first = fake_raw(
        tmp_path, 'a.czi', T=4, C=2, channel_names=['GFP', 'RFP'],
        time_interval_s=3.0, metadata_text='<a/>'
    )
    second = fake_raw(tmp_path, 'b.nd2', scenes=2, T=2, C=1, Z=3)
    result = tasks.read_metadata(
        {'paths': [first, second]}, make_ctx(tmp_path)
    )
    assert json.loads(json.dumps(result)) == result
    raws = [
        metadata.RawFileMetadata.from_dict(raw) for raw in result['metadata']
    ]
    assert [raw.path for raw in raws] == [first, second]
    assert (raws[0].size_t, raws[0].size_c, raws[0].time_increment) == (
        4, 2, 3.0
    )
    assert raws[0].channel_names == ('GFP', 'RFP')
    assert raws[0].metadata_xml == '<a/>'
    assert (raws[1].size_s, raws[1].size_t, raws[1].size_z) == (2, 2, 3)
    assert FACTORY.opened_paths() == [first, second]
    assert closes_match_opens()


def test_read_metadata_error_names_the_file(tmp_path):
    good = fake_raw(tmp_path, 'a.czi')
    bad = fake_raw(tmp_path, 'b, µ.czi', fail={'stage': 'open'})
    with pytest.raises(tasks.TaskError) as info:
        tasks.read_metadata({'paths': [good, bad]}, make_ctx(tmp_path))
    assert info.value.context == {'path': bad}
    assert bad in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


# ------------------------------------------------------------ read_sample
def legacy_sample(path: str, size_c: int, size_t: int, size_z: int,
                  lazy_load: bool) -> list[np.ndarray]:
    """Inline copy of `_read_sample_data.py` (main) on the same reader."""
    if size_t >= 4:
        sample_size_t = 4
    else:
        sample_size_t = size_t
    if size_z > 20:
        sample_size_z = 20
    else:
        sample_size_z = size_z
    all_channels = []
    with reader.ImageReader(
            path, lazy_load=lazy_load, bioimage_factory=FACTORY) as rdr:
        for c in range(size_c):
            img_tz = []
            for t in range(sample_size_t):
                img_z = []
                for z in range(sample_size_z):
                    img = rdr.read(c=c, z=z, t=t, rescale=False)
                    img_z.append(img)
                img_z = np.array(img_z, dtype=img.dtype)
                img_z = np.squeeze(img_z)
                img_tz.append(img_z)
            img_tz = np.array(img_tz, dtype=img.dtype)
            img_tz = np.squeeze(img_tz)
            all_channels.append(img_tz)
    return all_channels


@pytest.mark.parametrize('lazy_load', [True, False])
@pytest.mark.parametrize('sizes', [(2, 5, 2), (1, 3, 1), (3, 4, 9)])
def test_read_sample_matches_the_legacy_child(tmp_path, sizes, lazy_load):
    size_c, size_t, size_z = sizes
    path = fake_raw(tmp_path, 'a.czi', T=size_t, C=size_c, Z=size_z, Y=4,
                    X=5)
    ctx = make_ctx(tmp_path)
    result = tasks.read_sample({
        'path': path, 'size_c': size_c, 'size_t': size_t, 'size_z': size_z,
        'lazy_load': lazy_load,
    }, ctx)
    assert result == {
        'arrays': [f'sample_{c}.npy' for c in range(size_c)]
    }
    arrays = [
        np.load(os.path.join(ctx.out_dir, name)) for name in result['arrays']
    ]
    expected = legacy_sample(path, size_c, size_t, size_z, lazy_load)
    assert len(arrays) == len(expected)
    for array, oracle in zip(arrays, expected):
        assert array.dtype == oracle.dtype
        np.testing.assert_array_equal(array, oracle)
    assert closes_match_opens()


def test_read_sample_options(tmp_path):
    path = fake_raw(tmp_path, 'a.czi', scenes=2, T=5, C=1, Z=3, Y=4, X=5)
    ctx = make_ctx(tmp_path)
    result = tasks.read_sample({
        'path': path, 'size_c': 1, 'size_t': 5, 'size_z': 3,
        'lazy_load': True, 'max_t': 2, 'max_z': 1, 'series': 1,
    }, ctx)
    array = np.load(os.path.join(ctx.out_dir, result['arrays'][0]))
    assert array.shape == (2, 4, 5)
    assert plane_origins(array) == [(1, 0, 0, 0), (1, 1, 0, 0)]


def test_read_sample_rejects_empty_sizes(tmp_path):
    path = fake_raw(tmp_path, 'a.czi')
    with pytest.raises(ValueError, match='SizeT=0'):
        tasks.read_sample({
            'path': path, 'size_c': 1, 'size_t': 0, 'size_z': 1,
            'lazy_load': True,
        }, make_ctx(tmp_path))


# -------------------------------------------------------- check_positions
def check(jobs, ctx) -> dict:
    params = {'jobs': [job.to_dict() for job in jobs]}
    return tasks.check_positions(json.loads(json.dumps(params)), ctx)


def two_scene_raw(folder, name: str = 'exp.czi', *, t1: int = 2) -> str:
    """Scene 0 has 4 frames, scene 1 has `t1`."""
    path = os.path.join(str(folder), name)
    fake_bioio.write_fake_raw(path, {'scenes': [
        {'T': 4, 'C': 2, 'Z': 2, 'Y': 8, 'X': 8},
        {'T': t1, 'C': 2, 'Z': 2, 'Y': 8, 'X': 8},
    ]})
    return path


def test_check_positions_checks_every_series_and_writes_nothing(tmp_path):
    raw_path = two_scene_raw(tmp_path)
    dst = tmp_path / 'dst'
    jobs = [
        make_job(str(dst / f'Position_{n}' / 'Images'), raw_path,
                 input_id=n, output_number=n, num_positions=2, series=n - 1,
                 time_range=(1, 4), clear_images=True)
        for n in (1, 2)
    ]
    events = []
    with pytest.raises(ValueError) as info:
        check(jobs, make_ctx(tmp_path, events))
    assert str(info.value) == (
        'Time range 1-4 (1-based, inclusive) does not fit the 2 frame(s) '
        f'of "{raw_path}" (series 1).'
    )
    jobs = [dataclasses.replace(job, time_range=(1, 2)) for job in jobs]
    assert check(jobs, make_ctx(tmp_path, events)) == {'checked': 2}
    assert events == [] and not dst.exists()
    assert FACTORY.opened_paths() == [raw_path, raw_path]
    assert closes_match_opens()


@pytest.mark.parametrize('problem, match', [
    ({'size_z': 3}, 'Cannot save 3 z-slice'),
    ({'channels': (('GFP', 0), ('RFP', 2))}, 'Channel index 2 ("RFP")'),
])
def test_check_positions_z_and_channels(tmp_path, problem, match):
    raw_path = fake_raw(tmp_path, 'exp.czi', T=3, C=2, Z=2)
    job = make_job(str(tmp_path / 'dst' / 'Position_1' / 'Images'),
                   raw_path, **problem)
    with pytest.raises(ValueError, match=re.escape(match)):
        check([job], make_ctx(tmp_path))
    assert closes_match_opens()


def test_check_positions_every_file_of_a_job(tmp_path):
    """One file per channel: the second file of a job is checked too, and
    each file is closed after the last job that reads it."""
    first = fake_raw(tmp_path, 'x_1_GFP.czi', T=3, C=1)
    second = fake_raw(tmp_path, 'x_1_mCh.czi', T=2, C=1)
    third = fake_raw(tmp_path, 'x_2_GFP.czi', T=3, C=1)
    images = str(tmp_path / 'dst' / 'Position_1' / 'Images')
    job = make_job(images, first, channels=(('GFP', 0),))
    job = dataclasses.replace(job, channels=(
        *job.channels, planning.ChannelSource('mCh', 1, second, 0)
    ))
    with pytest.raises(ValueError, match='frame.*x_1_mCh.czi'):
        check([job], make_ctx(tmp_path))
    assert closes_match_opens()
    FACTORY.reset_log()
    jobs = [
        dataclasses.replace(job, time_range=(1, 2)),
        make_job(str(tmp_path / 'dst' / 'Position_2' / 'Images'), third,
                 input_id=2, output_number=2, num_positions=2,
                 channels=(('GFP', 0),), time_range=(1, 2)),
    ]
    assert check(jobs, make_ctx(tmp_path)) == {'checked': 2}
    log = [event[:2] for event in FACTORY.log]
    assert log.index(('close', first)) < log.index(('open', third))
    assert log.index(('close', second)) < log.index(('open', third))
    assert closes_match_opens()


# --------------------------------------------------------- save_positions
def test_save_positions_legacy_order_and_events(tmp_path):
    raw_path = fake_raw(tmp_path, 'exp.czi', scenes=2, T=3, C=2, Z=2)
    dst = tmp_path / 'dst'
    jobs = [
        make_job(str(dst / f'Position_{n}' / 'Images'), raw_path,
                 input_id=n, output_number=n, num_positions=2, series=n - 1)
        for n in (1, 2)
    ]
    events = []
    result = save(jobs, make_ctx(tmp_path, events))
    assert json.loads(json.dumps(result)) == result
    assert json.loads(json.dumps(events)) == events

    kinds = [
        (event['kind'], event.get('name') or event.get('message'))
        for event in events
        if event['kind'] != 'position_done'
    ]
    images = [str(dst / f'Position_{n}' / 'Images') for n in (1, 2)]
    assert kinds == [
        ('log', f'Position 1/2: saving data to {images[0]}...'),
        ('channel_done', 'GFP'), ('channel_done', 'RFP'),
        ('log', f'Position 2/2: saving data to {images[1]}...'),
        ('channel_done', 'GFP'), ('channel_done', 'RFP'),
    ]
    done = [event for event in events if event['kind'] == 'position_done']
    assert [event['input_id'] for event in done] == [1, 2]
    assert result['written'] == [
        {key: event[key] for key in
         ('input_id', 'output_number', 'images_path', 'files')}
        for event in done
    ]
    for n, entry in zip((1, 2), result['written']):
        folder = images[n - 1]
        assert entry['files'] == [
            os.path.join(folder, f'exp_s{n}_{suffix}')
            for suffix in ('metadataXML.txt', 'metadata.csv', 'GFP.tif',
                           'RFP.tif')
        ]
        assert all(os.path.isfile(path) for path in entry['files'])
        with open(entry['files'][0], encoding='utf-8') as file:
            assert file.read() == '<xml/>'
        csv = writers.read_metadata_csv(entry['files'][1])
        assert (csv['basename'], csv['SizeT'], csv['channel_1_name']) == (
            f'exp_s{n}_', '3', 'RFP'
        )
        rfp = tifffile.imread(entry['files'][3])
        assert plane_origins(rfp) == [
            (n - 1, t, 1, z) for t in range(3) for z in range(2)
        ]
    assert closes_match_opens()


@pytest.mark.parametrize('output', ['h5', 'symlink'])
def test_save_positions_other_formats(tmp_path, output):
    raw_path = fake_raw(tmp_path, 'exp.czi', T=4, C=2, Z=2)
    images = str(tmp_path / 'dst' / 'Position_1' / 'Images')
    job = make_job(images, raw_path, time_range=(2, 3), output=output)
    entry = save([job], make_ctx(tmp_path))['written'][0]
    if output == 'h5':
        assert [os.path.basename(p) for p in entry['files'][2:]] == [
            'exp_s1_GFP.h5', 'exp_s1_RFP.h5'
        ]
        with h5py.File(entry['files'][3], 'r') as h5:
            data = h5['data'][()]
        assert plane_origins(data) == [
            (0, t, 1, z) for t in (1, 2) for z in range(2)
        ]
    else:
        assert [os.path.basename(p) for p in entry['files'][2:]] == [
            'exp_s1_symlink.ini'
        ]
        ini = configparser.ConfigParser()
        ini.read(entry['files'][2])
        section = ini['channel_name.RFP']
        assert section['source_filepath'] == raw_path
        assert (section['frames_range'], section['channel_index']) == (
            '1,3', '1'
        )


def test_save_positions_checks_every_job_before_writing(tmp_path):
    """A job that does not fit its file fails before any folder is cleared
    or created, also for the jobs before it."""
    good = fake_raw(tmp_path, 'a.czi', T=3)
    short = fake_raw(tmp_path, 'b.czi', T=2)
    dst = tmp_path / 'dst'
    existing = dst / 'Position_1' / 'Images'
    existing.mkdir(parents=True)
    (existing / 'keep.txt').write_text('old')
    jobs = [
        make_job(str(existing), good, clear_images=True),
        make_job(str(dst / 'Position_2' / 'Images'), short, input_id=2,
                 output_number=2, num_positions=2),
    ]
    events = []
    with pytest.raises(ValueError, match='does not fit the 2 frame'):
        save(jobs, make_ctx(tmp_path, events))
    assert events == []
    assert (existing / 'keep.txt').read_text() == 'old'
    assert sorted(os.listdir(existing)) == ['keep.txt']
    assert not (dst / 'Position_2').exists()
    assert closes_match_opens()


def test_save_positions_requires_every_text(tmp_path):
    raw_path = fake_raw(tmp_path, 'a.czi')
    job = make_job(str(tmp_path / 'dst' / 'Position_1' / 'Images'), raw_path)
    with pytest.raises(ValueError, match='xml:0'):
        save([job], make_ctx(tmp_path), texts={})
    assert not (tmp_path / 'dst').exists()


def test_save_positions_overwrite_clears_only_images(tmp_path):
    raw_path = fake_raw(tmp_path, 'exp.czi')
    position = tmp_path / 'dst' / 'Position_1'
    (position / 'Images').mkdir(parents=True)
    (position / 'Images' / 'old_segm.npz').write_text('old')
    (position / 'notes.txt').write_text('keep')
    new = tmp_path / 'dst' / 'Position_2' / 'Images'
    jobs = [
        make_job(str(position / 'Images'), raw_path, clear_images=True),
        make_job(str(new), raw_path, clear_images=True, input_id=2,
                 output_number=2, num_positions=2),
    ]
    events = []
    save(jobs, make_ctx(tmp_path, events))
    assert sorted(os.listdir(position / 'Images')) == [
        'exp_s1_GFP.tif', 'exp_s1_RFP.tif', 'exp_s1_metadata.csv',
        'exp_s1_metadataXML.txt',
    ]
    assert (position / 'notes.txt').read_text() == 'keep'
    # only the folder that existed is reported, before its first log line
    assert [event['kind'] for event in events[:2]] == ['images_deleted', 'log']
    assert [
        event['images_path'] for event in events
        if event['kind'] == 'images_deleted'
    ] == [str(position / 'Images')]


@pytest.mark.parametrize('failure, reported', [
    ('rmtree part way', True),
    ('rmtree at once', False),
    ('makedirs', True),
])
def test_save_positions_reports_a_partly_deleted_images_folder(
        tmp_path, monkeypatch, failure, reported):
    """OVERWRITE: when deleting the old Images folder fails part way (e.g.
    a file open in another program on Windows) or creating the new one
    fails, 'images_deleted' is still emitted before the error; not when
    nothing was deleted."""
    raw_path = fake_raw(tmp_path, 'exp.czi')
    images = tmp_path / 'dst' / 'Position_1' / 'Images'
    images.mkdir(parents=True)
    for name in ('a_segm.npz', 'b_acdc_output.csv', 'c_open.tif'):
        (images / name).write_text('old')
    rmtree, makedirs = writers.shutil.rmtree, writers.os.makedirs

    def failing_rmtree(path, *args, **kwargs):
        if os.path.normpath(path) != os.path.normpath(images):
            return rmtree(path, *args, **kwargs)
        if failure == 'rmtree part way':
            os.remove(os.path.join(path, 'a_segm.npz'))
        raise PermissionError('c_open.tif is open in another program')

    def failing_makedirs(path, *args, **kwargs):
        if os.path.normpath(path) == os.path.normpath(images):
            raise PermissionError('access denied')
        return makedirs(path, *args, **kwargs)

    if failure == 'makedirs':
        monkeypatch.setattr(writers.os, 'makedirs', failing_makedirs)
    else:
        monkeypatch.setattr(writers.shutil, 'rmtree', failing_rmtree)
    events = []
    job = make_job(str(images), raw_path, clear_images=True)
    with pytest.raises(PermissionError):
        save([job], make_ctx(tmp_path, events))
    expected = [{'kind': 'images_deleted', 'images_path': str(images)}]
    assert events == (expected if reported else [])
    assert closes_match_opens()


def test_save_positions_opens_each_lazy_file_once(tmp_path):
    raw_path = fake_raw(tmp_path, 'exp.czi', scenes=3)
    jobs = [
        make_job(str(tmp_path / 'dst' / f'Position_{n}' / 'Images'),
                 raw_path, input_id=n, output_number=n, num_positions=3,
                 series=n - 1)
        for n in (1, 2, 3)
    ]
    save(jobs, make_ctx(tmp_path))
    assert FACTORY.opened_paths() == [raw_path]
    assert closes_match_opens()


def test_save_positions_releases_files_after_their_last_job(tmp_path):
    """Eager readers (whole file in RAM) are closed once no later job of the
    call needs them; the checks open the files lazily."""
    first = fake_raw(tmp_path, 'a.czi')
    second = fake_raw(tmp_path, 'b.czi')
    jobs = [
        make_job(str(tmp_path / 'dst' / f'Position_{n}' / 'Images'), path,
                 input_id=n, output_number=n, num_positions=2,
                 lazy_load=False)
        for n, path in ((1, first), (2, second))
    ]
    save(jobs, make_ctx(tmp_path))
    # lazy checks of both files, then one eager reader per job
    assert FACTORY.opened_paths() == [first, second, first, second]
    log = [event[:2] for event in FACTORY.log]
    eager_open_of_second = max(
        index for index, event in enumerate(log) if event == ('open', second)
    )
    closes_of_first = [
        index for index, event in enumerate(log) if event == ('close', first)
    ]
    assert len(closes_of_first) == 2
    assert max(closes_of_first) < eager_open_of_second
    assert closes_match_opens()


def test_save_positions_reports_a_missing_output(tmp_path, monkeypatch):
    raw_path = fake_raw(tmp_path, 'exp.czi')
    images = tmp_path / 'dst' / 'Position_1' / 'Images'
    ghost = str(images / 'ghost.tif')
    save_images = writer.save_position_images

    def save_and_report_a_ghost(*args, **kwargs):
        return save_images(*args, **kwargs) + [ghost]

    monkeypatch.setattr(
        writer, 'save_position_images', save_and_report_a_ghost
    )
    events = []
    with pytest.raises(tasks.TaskError) as info:
        save([make_job(str(images), raw_path)], make_ctx(tmp_path, events))
    assert info.value.context == {'path': ghost}
    assert 'expected output missing' in str(info.value)
    assert not any(event['kind'] == 'position_done' for event in events)
    assert closes_match_opens()


def test_save_positions_other_files(tmp_path):
    raw_path = fake_raw(tmp_path, 'ASY015_1_GFP.czi', C=1)
    src = tmp_path
    (src / 'ASY015_1_notes.txt').write_text('notes')
    images = tmp_path / 'dst' / 'Position_1' / 'Images'
    actions = [
        planning.OtherFileAction(
            str(src / 'ASY015_1_notes.txt'),
            str(images / 'ASY015_s1_notes.txt'), 'copy'
        ),
        planning.OtherFileAction(
            str(src / 'missing.txt'), str(images / 'missing.txt'), 'move'
        ),
    ]
    events = []
    job = make_job(str(images), raw_path, channels=(('GFP', 0),),
                   other_files=actions, stem='ASY015_')
    entry = save([job], make_ctx(tmp_path, events))['written'][0]
    assert entry['files'][-1] == str(images / 'ASY015_s1_notes.txt')
    assert (images / 'ASY015_s1_notes.txt').read_text() == 'notes'
    assert (src / 'ASY015_1_notes.txt').exists()
    warnings = [
        event['message'] for event in events
        if event['kind'] == 'log' and event['level'] == 'WARNING'
    ]
    assert len(warnings) == 1 and 'missing.txt' in warnings[0]


# ------------------------------------------------- worker (production ops)
def run_worker(tmp_path, task: str, params: dict) -> tuple[int, list, dict]:
    run_dir = tmp_path / 'worker'
    run_dir.mkdir()
    job = {
        'protocol': runner.PROTOCOL, 'task': task, 'params': params,
        'out_dir': str(run_dir),
        'expected_package_dir': os.path.dirname(cellacdc.__file__),
        'reader_factory': fake_bioio.FAKE_BIOIMAGE_PATH, 'ops_module': None,
    }
    job_path = run_dir / runner.JOB_FILENAME
    job_path.write_text(json.dumps(job), encoding='utf-8')
    stream = io.StringIO()
    code = _worker.main([str(job_path)], event_stream=stream)
    events = [
        json.loads(line[len(runner.EVENT_PREFIX):])
        for line in stream.getvalue().splitlines()
    ]
    report_name = (
        runner.RESULT_FILENAME if code == runner.EXIT_OK
        else runner.ERROR_FILENAME
    )
    with open(run_dir / report_name, encoding='utf-8') as file:
        report = json.load(file)
    return code, events, report


def test_worker_runs_the_production_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')
    raw_path = fake_raw(tmp_path, 'exp.czi')
    code, _, report = run_worker(
        tmp_path, 'read_metadata', {'paths': [raw_path]}
    )
    assert code == runner.EXIT_OK
    raw = metadata.RawFileMetadata.from_dict(report['result']['metadata'][0])
    assert raw.reader == fake_bioio.FAKE_BIOIMAGE_PATH.replace(':', '.')


def test_worker_events_of_save_positions(tmp_path, monkeypatch):
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')
    raw_path = fake_raw(tmp_path, 'exp.czi')
    job = make_job(str(tmp_path / 'dst' / 'Position_1' / 'Images'), raw_path)
    code, events, report = run_worker(tmp_path, 'save_positions', {
        'jobs': [job.to_dict()], 'texts': {'xml:0': 'x'},
    })
    assert code == runner.EXIT_OK
    assert [event['kind'] for event in events] == [
        'log', 'channel_done', 'channel_done', 'position_done'
    ]
    assert report['result']['written'][0]['files'] == events[-1]['files']


def test_worker_runs_check_positions(tmp_path, monkeypatch):
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')
    raw_path = two_scene_raw(tmp_path)
    jobs = [
        make_job(str(tmp_path / 'dst' / f'Position_{n}' / 'Images'),
                 raw_path, input_id=n, output_number=n, num_positions=2,
                 series=n - 1, time_range=(1, 4))
        for n in (1, 2)
    ]
    code, events, report = run_worker(tmp_path, 'check_positions', {
        'jobs': [job.to_dict() for job in jobs],
    })
    assert code == runner.EXIT_TASK_FAILED and events == []
    assert report['error_type'] == 'ValueError'
    assert 'does not fit the 2 frame(s)' in report['message']
    assert not (tmp_path / 'dst').exists()


def test_worker_reports_the_failing_path(tmp_path, monkeypatch):
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')
    bad = fake_raw(tmp_path, 'bad.czi', fail={'stage': 'open'})
    code, _, report = run_worker(tmp_path, 'read_metadata', {'paths': [bad]})
    assert code == runner.EXIT_TASK_FAILED
    assert report['error_type'] == 'TaskError'
    assert report['context'] == {'path': bad}
    assert 'injected open failure' in report['traceback']
