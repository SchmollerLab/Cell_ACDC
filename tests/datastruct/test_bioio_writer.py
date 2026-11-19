"""acdc_bioio_bioformats.writer.save_position_images and the _utils
helpers it relies on.

The writer must produce exactly what a direct legacy
`_utils.saveImgDataChannel` call produces (tif decoded + tags, h5, ini
text), validate the request against the real file before writing, and
open each raw file once per job. Runs on FakeBioImage (no bioio needed);
the legacy child-script smoke test needs bioio and ACDC_SLOW_TESTS.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid

import numpy as np
import pytest
import tifffile

from . import fake_bioio, manifest, synthetic
from .conditions import hooks_env, requires_bioio, slow
from .fake_bioio import FakeBioImage

from cellacdc import _process
from cellacdc import acdc_bioio_bioformats as bioformats
from cellacdc.acdc_bioio_bioformats import _utils, reader, writer


@pytest.fixture(autouse=True)
def _reset_fake_log():
    FakeBioImage.reset_log()
    yield
    FakeBioImage.reset_log()


def _fake_raw(folder, filename='raw.nd2', **spec_kwargs):
    spec = fake_bioio.make_spec(**spec_kwargs)
    return fake_bioio.write_fake_raw(os.path.join(folder, filename), spec)


def _images_dir(tmp_path, name):
    path = tmp_path / name / 'Images'
    path.mkdir(parents=True)
    return str(path)


def _request(images_path, channels, **changes):
    fields = dict(
        series=0, images_path=images_path, filename_no_ext='exp.1_pos',
        s0p='02', channels=tuple(channels), time_range=(1, 3), size_z=2,
        time_increment_s=7.5, physical_size_zyx_um=(0.5, 0.2, 0.1),
        fmt='tif', lazy_load=True
    )
    fields.update(changes)
    return writer.PositionImagesRequest(**fields)


def _legacy_save(req: writer.PositionImagesRequest, size_t: int):
    """What the legacy child scripts do: one saveImgDataChannel per channel
    (same arguments as _save_data.py / _save_data_single_channel.py)."""
    pz, py, px = req.physical_size_zyx_um
    for channel in req.channels:
        with reader.ImageReader(
                channel.raw_path, lazy_load=req.lazy_load,
                bioimage_factory=FakeBioImage) as image_reader:
            size_z = req.size_z
            if size_z is None:
                size_z = image_reader.dims(0)[2]
            _utils.saveImgDataChannel(
                image_reader, req.series, req.images_path,
                req.filename_no_ext, req.s0p, channel.name,
                channel.file_channel_index, {}, size_t, size_z,
                req.time_increment_s, pz, py, px, req.fmt == 'h5',
                list(req.time_range), req.fmt == 'symlink',
                channel.raw_path, req.lazy_load
            )


def _manifest(images_path, raw_dir):
    return manifest.build_manifest(
        images_path, placeholders={'<RAW>': raw_dir}, include_env=False
    )


# ------------------------------------------------------- legacy parity
@pytest.mark.parametrize('lazy_load', [True, False])
@pytest.mark.parametrize('fmt', ['tif', 'h5', 'symlink'])
def test_output_equals_direct_legacy_call(tmp_path, fmt, lazy_load):
    raw_dir = str(tmp_path / 'raw')
    raw = _fake_raw(raw_dir, scenes=2, T=4, C=3, Z=2, Y=6, X=5)
    channels = (
        writer.ChannelImageSpec(raw, 2, 'GFP'),
        writer.ChannelImageSpec(raw, 0, 'phase'),
    )
    legacy = _request(
        _images_dir(tmp_path, 'legacy'), channels, series=1,
        time_range=(2, 3), fmt=fmt, lazy_load=lazy_load
    )
    _legacy_save(legacy, size_t=4)
    req = _request(
        _images_dir(tmp_path, 'new'), channels, series=1, time_range=(2, 3),
        fmt=fmt, lazy_load=lazy_load
    )
    written = writer.save_position_images(req, reader_factory=FakeBioImage)

    expected = _manifest(legacy.images_path, raw_dir)
    assert _manifest(req.images_path, raw_dir) == expected
    assert sorted(written) == sorted(
        os.path.join(req.images_path, name)
        for name in os.listdir(req.images_path)
    )
    if fmt == 'symlink':
        assert [os.path.basename(p) for p in written] == [
            'exp_1_pos_s02_symlink.ini'
        ]
    else:
        assert [os.path.basename(p) for p in written] == [
            f'exp_1_pos_s02_GFP.{fmt}', f'exp_1_pos_s02_phase.{fmt}'
        ]
    if fmt == 'tif':
        data = tifffile.imread(written[0])
        assert data.shape == (2, 2, 6, 5)        # frames 2..3, two z-slices
        assert synthetic.decode_pixel(data[0, 0, 0, 0]) == (1, 1, 2, 0)
        assert synthetic.decode_pixel(data[1, 1, 0, 0]) == (1, 2, 2, 1)


def test_one_file_per_channel_uses_each_file_size_z(tmp_path):
    raw_dir = str(tmp_path / 'raw')
    raw_a = _fake_raw(raw_dir, 'ASY015_1_GFP.nd2', T=3, C=1, Z=3)
    raw_b = _fake_raw(raw_dir, 'ASY015_1_mCh.nd2', T=3, C=1, Z=3)
    channels = (
        writer.ChannelImageSpec(raw_a, 0, 'GFP'),
        writer.ChannelImageSpec(raw_b, 0, 'mCh'),
    )
    legacy = _request(
        _images_dir(tmp_path, 'legacy'), channels, size_z=None,
        filename_no_ext='ASY015_'
    )
    _legacy_save(legacy, size_t=3)
    req = _request(
        _images_dir(tmp_path, 'new'), channels, size_z=None,
        filename_no_ext='ASY015_'
    )
    writer.save_position_images(req, reader_factory=FakeBioImage)
    assert _manifest(req.images_path, raw_dir) == _manifest(
        legacy.images_path, raw_dir
    )
    data = tifffile.imread(
        os.path.join(req.images_path, 'ASY015__s02_mCh.tif')
    )
    assert data.shape == (3, 3, 16, 20)


# ---------------------------------------------------------- image_name
@pytest.mark.parametrize('fmt, names', [
    ('tif', ['exp_1_pos_Scene_1_s02_GFP.tif']),
    ('h5', ['exp_1_pos_Scene_1_s02_GFP.h5']),
    ('symlink', ['exp_1_pos_Scene_1_s02_symlink.ini']),
])
def test_image_name_in_file_names(tmp_path, fmt, names):
    raw = _fake_raw(str(tmp_path / 'raw'))
    req = _request(
        _images_dir(tmp_path, 'new'), [writer.ChannelImageSpec(raw, 1, 'GFP')],
        fmt=fmt, image_name='Scene_1'
    )
    written = writer.save_position_images(req, reader_factory=FakeBioImage)
    assert [os.path.basename(p) for p in written] == names
    assert sorted(os.listdir(req.images_path)) == names


# ------------------------------------------------------------ validation
@pytest.mark.parametrize('changes, message', [
    ({'time_range': (0, 2)}, 'Time range 0-2'),
    ({'time_range': (2, 5)}, 'Time range 2-5'),
    ({'time_range': (3, 2)}, 'Time range 3-2'),
    ({'size_z': 0}, '0 z-slice'),
    ({'size_z': 3}, '3 z-slice'),
    ({'channel_index': 2}, 'Channel index 2'),
    ({'channel_index': -1}, 'Channel index -1'),
])
def test_invalid_request_raises_before_writing(tmp_path, changes, message):
    raw_dir = str(tmp_path / 'raw')
    raw = _fake_raw(raw_dir, scenes=2, T=4, C=2, Z=2)
    other = _fake_raw(raw_dir, 'other.nd2', scenes=2, T=4, C=2, Z=2)
    index = changes.pop('channel_index', 1)
    channels = (
        writer.ChannelImageSpec(raw, 0, 'GFP'),     # valid, written first
        writer.ChannelImageSpec(other, index, 'mCh'),
    )
    req = _request(
        _images_dir(tmp_path, 'new'), channels, series=1, **changes
    )
    with pytest.raises(ValueError, match=message) as info:
        writer.save_position_images(req, reader_factory=FakeBioImage)
    assert 'series 1' in str(info.value)
    assert 'raw' in str(info.value)
    assert os.listdir(req.images_path) == []
    assert [e for e in FakeBioImage.log if e[0] == 'close']   # cleaned up


@pytest.mark.parametrize('changes, message', [
    ({'time_range': (2, 5)}, 'Time range 2-5'),
    ({'size_z': 3}, '3 z-slice'),
    ({'fmt': 'png'}, 'png'),
])
def test_check_position_images_validates_without_writing(
        tmp_path, changes, message):
    raw = _fake_raw(str(tmp_path / 'raw'), T=4, C=2, Z=2)
    channels = (writer.ChannelImageSpec(raw, 1, 'GFP'),)
    images_path = str(tmp_path / 'Position_1' / 'Images')
    cache = {}
    valid = _request(images_path, channels)
    writer.check_position_images(
        valid, reader_factory=FakeBioImage, reader_cache=cache
    )
    with pytest.raises(ValueError, match=message):
        writer.check_position_images(
            _request(images_path, channels, **changes),
            reader_factory=FakeBioImage, reader_cache=cache
        )
    assert not os.path.exists(images_path)
    os.makedirs(images_path)
    writer.save_position_images(
        valid, reader_factory=FakeBioImage, reader_cache=cache
    )
    assert FakeBioImage.opened_paths() == [raw]     # the cache is shared
    for image_reader in cache.values():
        image_reader.close()
    writer.check_position_images(valid, reader_factory=FakeBioImage)
    assert FakeBioImage.log[-1] == ('close', raw)   # no cache: closed


def test_unknown_format_is_rejected(tmp_path):
    raw = _fake_raw(str(tmp_path / 'raw'))
    req = _request(
        _images_dir(tmp_path, 'new'), [writer.ChannelImageSpec(raw, 0, 'a')],
        fmt='png'
    )
    with pytest.raises(ValueError, match='png'):
        writer.save_position_images(req, reader_factory=FakeBioImage)


# ------------------------------------------------------- reader caching
def test_each_raw_file_is_opened_once_per_job(tmp_path):
    raw = _fake_raw(str(tmp_path / 'raw'), scenes=2, C=2)
    channels = (
        writer.ChannelImageSpec(raw, 0, 'GFP'),
        writer.ChannelImageSpec(raw, 1, 'mCh'),
    )
    cache = {}
    done = []
    for series in (0, 1):
        req = _request(
            _images_dir(tmp_path, f'Position_{series + 1}'), channels,
            series=series, s0p=str(series + 1)
        )
        writer.save_position_images(
            req, reader_factory=FakeBioImage, reader_cache=cache,
            on_channel_done=done.append
        )
    assert FakeBioImage.opened_paths() == [raw]
    assert list(cache) == [(raw, True)]
    assert not [e for e in FakeBioImage.log if e[0] == 'close']  # caller's
    assert done == ['GFP', 'mCh', 'GFP', 'mCh']
    for image_reader in cache.values():
        image_reader.close()


def test_readers_are_closed_without_a_cache(tmp_path):
    raw = _fake_raw(str(tmp_path / 'raw'))
    req = _request(
        _images_dir(tmp_path, 'new'), [writer.ChannelImageSpec(raw, 0, 'a')]
    )
    writer.save_position_images(req, reader_factory=FakeBioImage)
    assert FakeBioImage.log[-1] == ('close', raw)


# ------------------------------------------------------- _utils helpers
def test_get_filename_with_and_without_image_name():
    assert _utils.getFilename('x.ome', '01', 'GFP', 0, '.tif') == (
        'x_ome_s01_GFP.tif'
    )
    assert _utils.getFilename(
        'x.ome', '01', 'metadata', 0, '.csv', return_basename=True,
        image_name='Scene_1'
    ) == ('x_ome_Scene_1_s01_metadata.csv', 'x_ome_Scene_1_s01_')


@pytest.mark.parametrize('name, expected', [
    ('GFP', 'GFP'), ('GFP!', 'GFP'), ('a b', 'a_b'), ('mNeon-Green_', 'mNeon_Green'),
    ('', ''),
])
def test_remove_invalid_characters_returns_the_name(name, expected):
    assert _utils.removeInvalidCharacters(name) == expected


def test_dump_exception_creates_its_folder(tmp_path, monkeypatch):
    folder = str(tmp_path / 'acdc-appdata' / 'acdc_dataStruct_temp')
    monkeypatch.setattr(_utils, 'bioio_sample_data_folderpath', folder)
    error_id = uuid.uuid4()
    try:
        raise KeyError('boom')
    except KeyError as err:
        _utils.dump_exception(err, error_id)
    with pytest.raises(RuntimeError, match='boom'):
        _utils.check_raise_exception(error_id)
    assert os.listdir(folder) == []


# ------------------------------------------------- legacy child scripts
@slow
@requires_bioio
def test_legacy_save_data_child_still_runs(tmp_path):
    """_save_data.py through _process.py, exactly as dataStruct.saveData
    calls it on the BioIO path (comma-joined command)."""
    raw = synthetic.make_imagej_tif(str(tmp_path / 'raw' / 'img.tif'), T=3)
    images_path = _images_dir(tmp_path, 'Position_1')
    save_data_py = os.path.join(
        os.path.dirname(bioformats.__file__), '_save_data.py'
    )
    error_id = uuid.uuid4()
    command = (
        f'{sys.executable}, {save_data_py}, -f, {raw}, -d, True True, '
        f'-c, phase GFP, -s, 0, -i, {images_path}, -p, img, -pos, 1, '
        f'-t, 3, -z, 2, -time_increment, 1.0, -zyx, 1.0 1.0 1.0, -r, 2 3, '
        f'-uuid, {error_id}'
    )
    subprocess.run(
        [sys.executable, _process.__file__, '-c', command],
        env=hooks_env(), stdin=subprocess.DEVNULL, capture_output=True,
        timeout=600
    )
    _utils.check_raise_exception(error_id)
    assert sorted(os.listdir(images_path)) == ['img_s1_GFP.tif', 'img_s1_phase.tif']
    data = tifffile.imread(os.path.join(images_path, 'img_s1_GFP.tif'))
    assert data.shape == (2, 2, 16, 20)
    assert synthetic.decode_pixel(np.asarray(data)[0, 1, 0, 0]) == (0, 1, 1, 1)
