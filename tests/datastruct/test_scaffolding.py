"""Self-tests of the shared data-structure test scaffolding.

Runs without bioio and without Qt (those cases skip); never imports cellacdc.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from . import conditions, fake_bioio, fake_omexml, manifest, qt_free_rules
from . import synthetic

MAIN_COMMIT = '7c0f7532'


def _sha(path) -> str:
    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


def _git_show(relpath: str) -> str:
    try:
        result = subprocess.run(
            ['git', '-C', conditions.REPO_ROOT, 'show',
             f'{MAIN_COMMIT}:{relpath}'],
            capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip('git is not available')
    if result.returncode != 0:
        pytest.skip(f'commit {MAIN_COMMIT} is not available (shallow clone?)')
    return result.stdout.decode('utf-8')


# ================================================================ conditions
def test_skip_conditions_are_pytest_marks():
    for mark in (conditions.requires_qt, conditions.requires_bioio,
                 conditions.slow, conditions.subprocess_heavy):
        assert isinstance(mark, pytest.MarkDecorator)
        assert mark.name == 'skipif'


@pytest.mark.parametrize('ci, slow_env, version, skipped', [
    ('', '', (3, 11, 0), False),
    ('1', '', (3, 11, 0), True),
    ('1', '', (3, 12, 4), False),
    ('1', '1', (3, 13, 0), False),
])
def test_subprocess_heavy_rule(monkeypatch, ci, slow_env, version, skipped):
    monkeypatch.setenv('CI', ci)
    monkeypatch.setenv('ACDC_SLOW_TESTS', slow_env)
    monkeypatch.setattr(
        conditions, 'sys', SimpleNamespace(version_info=version)
    )
    assert conditions.subprocess_heavy_skipped() is skipped


def test_slow_rule(monkeypatch):
    monkeypatch.delenv('ACDC_SLOW_TESTS', raising=False)
    assert not conditions.slow_enabled()
    monkeypatch.setenv('ACDC_SLOW_TESTS', '1')
    assert conditions.slow_enabled()


def test_hooks_env(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(['x_dir', 'y_dir']))
    env = conditions.hooks_env(
        ACDC_DATASTRUCT_READER_FACTORY=fake_bioio.FAKE_BIOIMAGE_PATH, N=3
    )
    assert env['ACDC_DATASTRUCT_TEST_HOOKS'] == '1'
    assert env['QT_QPA_PLATFORM'] == 'offscreen'
    assert env['PYTHONPATH'].split(os.pathsep) == [
        conditions.REPO_ROOT, 'x_dir', 'y_dir'
    ]
    assert env['ACDC_DATASTRUCT_READER_FACTORY'] == (
        'tests.datastruct.fake_bioio:FakeBioImage'
    )
    assert env['N'] == '3'
    assert os.path.isdir(os.path.join(conditions.REPO_ROOT, 'cellacdc'))


def test_tree_snapshot_detects_every_change(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    (src / 'sub').mkdir(parents=True)
    (src / 'a.txt').write_bytes(b'abc')
    before = conditions.tree_snapshot(str(src), str(dst))
    assert set(before) == {'src', 'src/a.txt', 'src/sub'}
    assert before['src/sub'] == (None, None, 'dir')
    assert before['src/a.txt'][0] == 3
    assert conditions.tree_snapshot(str(src), str(dst)) == before

    (src / 'a.txt').write_bytes(b'abd')                 # same size
    os.utime(src / 'a.txt', ns=(1, before['src/a.txt'][1]))
    changed = conditions.tree_snapshot(str(src), str(dst))
    assert changed['src/a.txt'][2] != before['src/a.txt'][2]

    (src / 'a.txt').write_bytes(b'abc')
    os.utime(src / 'a.txt', ns=(1, before['src/a.txt'][1] + 10**9))
    assert conditions.tree_snapshot(str(src), str(dst)) != before

    dst.mkdir()                                         # created root
    assert 'dst' in conditions.tree_snapshot(str(src), str(dst))
    (src / 'sub' / 'new').mkdir()                       # new empty dir
    assert 'src/sub/new' in conditions.tree_snapshot(str(src))


def test_blocked_imports_prelude_and_run_python():
    code = conditions.blocked_imports_prelude() + (
        'import json\n'
        'for name in ("qtpy", "PyQt6.QtCore", "pyqtgraph"):\n'
        '    try:\n'
        '        __import__(name)\n'
        '    except ModuleNotFoundError as err:\n'
        '        print("blocked", err.name)\n'
        'print(json.dumps({"ok": True}))\n'
    )
    result = conditions.run_python(code, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        'blocked qtpy', 'blocked PyQt6', 'blocked pyqtgraph',
        '{"ok": true}',
    ]


@conditions.requires_qt
def test_qapp_is_offscreen_singleton():
    app = conditions.qapp()
    assert app is conditions.qapp()


# ================================================================= synthetic
def test_encode_decode_roundtrip():
    for s in (0, 1, 2, 10, 64):
        for t in range(10):
            for c in (0, 5, 9):
                for z in (0, 3, 9):
                    value = synthetic.encode_pixel(s, t, c, z)
                    assert synthetic.decode_pixel(value) == (s, t, c, z)
    assert synthetic.decode_pixel(np.uint16(2121)) == (2, 1, 2, 1)
    assert synthetic.decode_pixel(np.float32(1100.0)) == (1, 1, 0, 0)
    for bad in (-1, 10.5):
        with pytest.raises(ValueError):
            synthetic.decode_pixel(bad)
    with pytest.raises(ValueError):
        synthetic.encode_pixel(0, 10, 0, 0)


def test_encoded_array_orders_offsets_and_limits():
    data = synthetic.encoded_array('CTYX', T=3, C=2, Y=4, X=5, s=3, z0=7)
    assert data.shape == (2, 3, 4, 5) and data.dtype == np.uint16
    for c in range(2):
        for t in range(3):
            plane = data[c, t]
            assert (plane == plane[0, 0]).all()
            assert synthetic.decode_pixel(plane[0, 0]) == (3, t, c, 7)
    assert synthetic.encoded_array('YX', Y=2, X=2, t0=4).max() == 400
    with pytest.raises(ValueError):
        synthetic.encoded_array('TYX', T=11)
    with pytest.raises(ValueError):
        synthetic.encoded_array('TYX', T=3, t0=8)
    with pytest.raises(ValueError):
        synthetic.encoded_array('TZ')
    with pytest.raises(ValueError):
        synthetic.encoded_array('YX', s=1, dtype='uint8')


def test_imagej_tif_axes_decode_and_determinism(tmp_path):
    path = synthetic.make_imagej_tif(
        str(tmp_path / 'a.tif'), spacing=0.5, finterval=7.5, unit='um',
        physical_size_xy=0.1
    )
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        assert series.axes == 'TZCYX'
        assert series.shape == (3, 2, 2, 16, 20)
        data = series.asarray()
        meta = tif.imagej_metadata
        assert tif.pages[0].tags['XResolution'].value == (10, 1)
    assert (meta['frames'], meta['slices'], meta['channels']) == (3, 2, 2)
    assert (meta['spacing'], meta['finterval'], meta['unit']) == (
        0.5, 7.5, 'um'
    )
    for t in range(3):
        for z in range(2):
            for c in range(2):
                assert synthetic.decode_pixel(data[t, z, c, 5, 7]) == (
                    0, t, c, z
                )
    again = synthetic.make_imagej_tif(
        str(tmp_path / 'b.tif'), spacing=0.5, finterval=7.5, unit='um',
        physical_size_xy=0.1
    )
    assert _sha(path) == _sha(again)

    one_channel = synthetic.make_imagej_tif(
        str(tmp_path / 'c.tif'), T=10, C=1, Z=1, s=2
    )
    with tifffile.TiffFile(one_channel) as tif:
        assert tif.series[0].axes == 'TYX'
        assert synthetic.decode_pixel(tif.series[0].asarray()[9, 0, 0]) == (
            2, 9, 0, 0
        )


def _ome_ns(xml: str) -> dict:
    return {'ome': xml.split('xmlns="', 1)[1].split('"', 1)[0]}


def test_ome_tif_series_and_metadata(tmp_path):
    path = synthetic.make_ome_tif(str(tmp_path / 'exp.ome.tif'))
    with tifffile.TiffFile(path) as tif:
        assert [s.axes for s in tif.series] == ['TCZYX']*3
        assert [s.shape for s in tif.series] == [(3, 2, 2, 16, 20)]*3
        for k, series in enumerate(tif.series):
            data = series.asarray()
            assert synthetic.decode_pixel(data[2, 1, 1, 0, 0]) == (k, 2, 1, 1)
        xml = tif.ome_metadata
    root = ET.fromstring(xml)
    ns = _ome_ns(xml)
    images = root.findall('ome:Image', ns)
    assert [image.get('Name') for image in images] == [
        'Scene #1', 'Scene #2', 'Scene #3'
    ]
    pixels = images[0].find('ome:Pixels', ns)
    assert pixels.get('PhysicalSizeX') == '0.1'
    assert pixels.get('PhysicalSizeZ') == '0.5'
    assert pixels.get('PhysicalSizeXUnit') == '\u00b5m'
    assert pixels.get('TimeIncrement') == '7.5'
    assert pixels.get('TimeIncrementUnit') == 's'
    channels = pixels.findall('ome:Channel', ns)
    assert [c.get('Name') for c in channels] == ['GFP', 'mCherry']
    assert [c.get('EmissionWavelength') for c in channels] == [
        '510.0', '610.0'
    ]
    assert root.get('UUID') == synthetic.OME_UUID
    assert synthetic.read_ome_xml(path) == xml
    again = synthetic.make_ome_tif(str(tmp_path / 'again.ome.tif'))
    assert _sha(path) == _sha(again)


def test_ome_tif_variants(tmp_path):
    path = synthetic.make_ome_tif(
        str(tmp_path / 'v.ome.tif'), series=2, T=1, C=1, image_names=None,
        names=None, em_wavelens=None, physical_sizes=None,
        time_increment=None, lens_na=0.95, s_offset=4,
        xml_transform=lambda xml: xml.replace('Type="uint16"', 'Type="uint16" ')
    )
    with tifffile.TiffFile(path) as tif:
        assert len(tif.series) == 2
        data = tif.series[1].asarray()
        xml = tif.ome_metadata
    assert synthetic.decode_pixel(data.flat[0]) == (5, 0, 0, 0)
    root = ET.fromstring(xml)
    ns = _ome_ns(xml)
    assert all(i.get('Name') is None for i in root.findall('ome:Image', ns))
    objective = root.find('ome:Instrument/ome:Objective', ns)
    assert objective.get('LensNA') == '0.95'
    pixels = root.find('ome:Image/ome:Pixels', ns)
    for key in ('PhysicalSizeX', 'TimeIncrement'):
        assert pixels.get(key) is None
    assert pixels.find('ome:Channel', ns).get('Name') is None
    with pytest.raises(ValueError):
        synthetic.make_ome_tif(str(tmp_path / 'bad.ome.tif'), names=('a',))


def test_one_per_channel_sets(tmp_path):
    folder = str(tmp_path / 'l2')
    files = synthetic.make_one_per_channel_set(
        folder, positions=(1, 2, 10), skip={(2, 'GFP')}
    )
    assert files == [
        'ASY015_1_GFP.tif', 'ASY015_1_mNeon-Green.tif',
        'ASY015_2_mNeon-Green.tif', 'ASY015_10_GFP.tif',
        'ASY015_10_mNeon-Green.tif',
    ]
    assert sorted(os.listdir(folder)) == sorted(files)
    with tifffile.TiffFile(os.path.join(folder, files[-1])) as tif:
        assert tif.series[0].axes == 'TZYX'
        assert tif.series[0].shape == (3, 2, 16, 20)
        data = tif.series[0].asarray()
    assert synthetic.decode_pixel(data[2, 1, 0, 0]) == (10, 2, 1, 1)

    padded = synthetic.make_one_per_channel_set(
        str(tmp_path / 'pad'), basename='exp_', pad=2, ext='.ome.tif',
        channels=('GFP',)
    )
    assert padded == ['exp_01_GFP.ome.tif', 'exp_02_GFP.ome.tif']
    path = os.path.join(str(tmp_path / 'pad'), padded[1])
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome
        assert synthetic.decode_pixel(tif.series[0].asarray()[1, 1, 0, 0]) \
            == (2, 1, 0, 1)
        assert 'Name="GFP"' in tif.ome_metadata


def test_one_per_pos_set_and_truncated(tmp_path):
    files = synthetic.make_one_per_pos_set(
        str(tmp_path), filenames=('p1.tif', 'p2.tif', 'p3.ome.tif'),
        overrides={'p2.tif': {'T': 10}}
    )
    assert files == ['p1.tif', 'p2.tif', 'p3.ome.tif']
    with tifffile.TiffFile(str(tmp_path / 'p2.tif')) as tif:
        assert tif.series[0].shape == (10, 2, 2, 16, 20)
        assert synthetic.decode_pixel(tif.series[0].asarray()[9, 0, 0, 0, 0]) \
            == (1, 9, 0, 0)
    with tifffile.TiffFile(str(tmp_path / 'p3.ome.tif')) as tif:
        assert len(tif.series) == 1
        assert synthetic.decode_pixel(tif.series[0].asarray().flat[0]) == (
            2, 0, 0, 0
        )

    full = synthetic.make_imagej_tif(str(tmp_path / 'full.tif'))
    cut = synthetic.make_truncated_tif(str(tmp_path / 'cut.tif'))
    assert os.path.getsize(cut) == os.path.getsize(full) // 2
    try:
        with tifffile.TiffFile(cut) as tif:
            shape = tif.series[0].shape
            tif.series[0].asarray()
    except Exception:
        shape = None
    assert shape != (3, 2, 2, 16, 20)
    header = synthetic.make_truncated_tif(
        str(tmp_path / 'hdr.tif'), keep_bytes=4
    )
    with pytest.raises(Exception):
        tifffile.TiffFile(header).close()


def test_loose_tiff_set_for_restructure(tmp_path):
    files = synthetic.make_loose_tiff_set(
        str(tmp_path / 'A'), pattern='{pos}_{ch}_{t}.tif',
        positions=('pos1', 'pos2'), channels=('GFP', 'mCh'),
        timepoints=(1, 3, 5), skip={('pos2', 'mCh', 5, None)}
    )
    assert len(files) == 11 and files[0] == 'pos1_GFP_1.tif'
    assert 'pos2_mCh_5.tif' not in files
    data = tifffile.imread(str(tmp_path / 'A' / 'pos2_mCh_3.tif'))
    assert data.shape == (4, 5)
    assert synthetic.decode_pixel(data[0, 0]) == (1, 3, 1, 0)

    synthetic.make_loose_tiff_set(
        str(tmp_path / 'C'), pattern='well_A1.tif', file_axes='CTYX',
        C=2, T=3
    )
    with tifffile.TiffFile(str(tmp_path / 'C' / 'well_A1.tif')) as tif:
        data = tif.series[0].asarray()
    assert data.shape == (2, 3, 4, 5)
    assert synthetic.decode_pixel(data[1, 2, 0, 0]) == (0, 2, 1, 0)


# ================================================================ fake_bioio
def _fake(tmp_path, name='raw.nd2', **spec_kwargs) -> str:
    return fake_bioio.write_fake_raw(
        str(tmp_path / name), fake_bioio.make_spec(**spec_kwargs)
    )


def test_fake_bioimage_api(tmp_path):
    fake_bioio.FakeBioImage.reset_log()
    path = _fake(
        tmp_path, 'm.czi', scenes=3, physical_pixel_sizes=[0.5, 0.1, 0.2],
        time_interval_s=7.5, channel_names=['GFP', 'mNeon Green'],
        np_str_channel_names=True, metadata_text='<x/>'
    )
    image = fake_bioio.FakeBioImage(path, use_aicspylibczi=True)
    assert image.kwargs == {'use_aicspylibczi': True}
    assert image.scenes == ('Image:0', 'Image:1', 'Image:2')
    assert image.current_scene_index == 0 and image.current_scene == 'Image:0'
    assert image.dims.order == 'TCZYX'
    assert image.dims.shape == image.shape == (3, 2, 2, 16, 20)
    assert (image.dims.T, image.dims['Z', 'X']) == (3, (2, 20))
    assert image.dtype == np.uint16
    image.set_scene(2)
    assert image.current_scene_index == 2
    plane = image.get_image_dask_data('YX', T=1, C=1, Z=0)
    assert plane.shape == (16, 20)
    assert synthetic.decode_pixel(plane.compute()[3, 4]) == (2, 1, 1, 0)
    stack = image.get_image_dask_data('ZYX', T=2, C=0).compute()
    assert [synthetic.decode_pixel(v) for v in stack[:, 0, 0]] == [
        (2, 2, 0, 0), (2, 2, 0, 1)
    ]
    frames = image.get_image_data('TYX', T=[0, 2], C=1, Z=1)
    assert [synthetic.decode_pixel(v) for v in frames[:, 0, 0]] == [
        (2, 0, 1, 1), (2, 2, 1, 1)
    ]
    assert image.get_image_dask_data('TYX', T=1, C=0, Z=0).shape == (1, 16, 20)
    assert synthetic.decode_pixel(image.data[1, 0, 1, 0, 0]) == (2, 1, 0, 1)
    with pytest.raises(ValueError):
        image.get_image_dask_data('YX', T=0, C=0)        # Z not selected
    image.set_scene('Image:1')
    assert image.current_scene_index == 1
    with pytest.raises(IndexError):
        image.set_scene(3)
    names = image.channel_names
    assert names == ['GFP', 'mNeon Green']
    assert all(type(name) is np.str_ for name in names)
    sizes = image.physical_pixel_sizes
    assert (sizes.Z, sizes.Y, sizes.X) == (0.5, 0.1, 0.2)
    assert image.time_interval == datetime.timedelta(seconds=7.5)
    assert image.standard_metadata.timelapse_interval.total_seconds() == 7.5
    assert str(image.metadata) == '<x/>'
    with image:
        pass
    assert image.closed
    assert fake_bioio.FakeBioImage.opened_paths() == [path]
    assert fake_bioio.FakeBioImage.log[-1] == ('close', path)


def test_fake_bioimage_defaults_rgb_offsets_and_dims(tmp_path):
    image = fake_bioio.FakeBioImage(_fake(tmp_path, C=3, scenes=2))
    image.set_scene(1)
    assert image.channel_names == [
        'Channel:1:0', 'Channel:1:1', 'Channel:1:2'
    ]
    assert type(image.channel_names[0]) is str
    assert tuple(image.physical_pixel_sizes) == (None, None, None)
    assert image.time_interval is None
    assert image.standard_metadata.timelapse_interval is None
    assert image.metadata == '<FakeBioImage/>'

    rgb = fake_bioio.FakeBioImage(_fake(
        tmp_path, 'rgb.png', T=1, C=1, Z=1, rgb=True, dtype='uint8'
    ))
    assert rgb.dims.order == 'TCZYXS' and rgb.dims.S == 3
    assert rgb.get_image_dask_data('YXS', T=0, C=0, Z=0).compute().shape == (
        16, 20, 3
    )

    offsets = fake_bioio.FakeBioImage(_fake(
        tmp_path, 'o.nd2', T=2, C=1, Z=1, s_offset=5, c_offset=3,
        dims='ZTCYX', timelapse_interval_s=60.0
    ))
    assert offsets.dims.order == 'ZTCYX' and offsets.shape == (1, 2, 1, 16, 20)
    assert synthetic.decode_pixel(offsets.data[0, 1, 0, 0, 0]) == (5, 1, 3, 0)
    assert offsets.standard_metadata.timelapse_interval.total_seconds() == 60
    assert offsets.time_interval is None


def test_fake_spec_validation(tmp_path):
    with pytest.raises(ValueError):
        fake_bioio.make_spec(colour=True)
    with pytest.raises(ValueError):
        fake_bioio.write_fake_raw(str(tmp_path / 'a.nd2'), {'scenes': []})
    with pytest.raises(ValueError):
        fake_bioio.make_spec(dims='TCZY')
    with pytest.raises(ValueError):
        fake_bioio.make_spec(fail={'stage': 'later'})
    not_fake = tmp_path / 'real.tif'
    not_fake.write_bytes(b'II*\x00')
    with pytest.raises(ValueError):
        fake_bioio.FakeBioImage(str(not_fake))


def test_fake_bioimage_fail_modes(tmp_path):
    with pytest.raises(RuntimeError, match='injected open failure'):
        fake_bioio.FakeBioImage(_fake(tmp_path, 'a.nd2', fail={'stage': 'open'}))
    with pytest.raises(OSError):
        fake_bioio.FakeBioImage(_fake(
            tmp_path, 'b.nd2', fail={'stage': 'open', 'exception': 'OSError'}
        ))

    reader = fake_bioio.FakeBioImage(_fake(
        tmp_path, 'c.nd2', fail={'stage': 'read', 'after_planes': 1}
    ))
    lazy = reader.get_image_dask_data('YX', T=0, C=0, Z=0)
    lazy.compute()
    assert reader.channel_names                        # metadata still fine
    with pytest.raises(RuntimeError, match='injected read failure'):
        lazy.compute()

    meta = fake_bioio.FakeBioImage(_fake(
        tmp_path, 'd.nd2', fail={'stage': 'metadata'}
    ))
    assert meta.dims.shape == (3, 2, 2, 16, 20)
    meta.get_image_dask_data('YX', T=0, C=0, Z=0).compute()
    for attribute in ('channel_names', 'physical_pixel_sizes',
                      'time_interval', 'standard_metadata', 'metadata'):
        with pytest.raises(RuntimeError, match='injected metadata failure'):
            getattr(meta, attribute)

    with pytest.raises(KeyboardInterrupt):
        fake_bioio.FakeBioImage(_fake(
            tmp_path, 'e.nd2', fail={'stage': 'open', 'how': 'keyboard'}
        ))


def test_fake_bioimage_exit3_and_noise_in_child(tmp_path):
    path = _fake(
        tmp_path, 'f.nd2', noise='NOISE',
        fail={'stage': 'read', 'how': 'exit3'}
    )
    code = (
        'import sys, importlib\n'
        "module, attr = sys.argv[1].split(':')\n"
        'factory = getattr(importlib.import_module(module), attr)\n'
        'image = factory(sys.argv[2])\n'
        "image.get_image_dask_data('YX', T=0, C=0, Z=0).compute()\n"
        "print('not reached')\n"
    )
    result = conditions.run_python(
        code, fake_bioio.FAKE_BIOIMAGE_PATH, path, timeout=60
    )
    assert result.returncode == 3, result.stderr
    assert result.stdout.splitlines() == ['NOISE', 'NOISE']


def test_fake_bioimage_dotted_path_resolves():
    module, attr = fake_bioio.FAKE_BIOIMAGE_PATH.split(':')
    assert getattr(importlib.import_module(module), attr) is (
        fake_bioio.FakeBioImage
    )


@conditions.requires_bioio
def test_fake_bioimage_matches_bioio_on_same_content(tmp_path):
    from bioio import BioImage

    real = BioImage(synthetic.make_imagej_tif(str(tmp_path / 'r.tif')))
    fake = fake_bioio.FakeBioImage(_fake(tmp_path))
    assert fake.dims.order == real.dims.order
    assert fake.dims.shape == real.dims.shape
    assert fake.scenes == real.scenes
    assert fake.current_scene_index == real.current_scene_index
    assert fake.dtype == real.dtype
    requests = [
        ('YX', {'T': 2, 'C': 1, 'Z': 1}), ('ZYX', {'T': 1, 'C': 0}),
        ('TYX', {'T': [0, 2], 'C': 1, 'Z': 0}), ('CZYX', {'T': 0}),
        ('TYX', {'T': 1, 'C': 1, 'Z': 0}),
    ]
    for order, kwargs in requests:
        expected = real.get_image_dask_data(order, **kwargs).compute()
        actual = fake.get_image_dask_data(order, **kwargs).compute()
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fake.data, real.data)
    assert [type(n) for n in real.channel_names] == [np.str_, np.str_]
    assert fake.channel_names == [str(n) for n in real.channel_names]


# =============================================================== fake_omexml
def test_fake_omexml_full_spec_answers_legacy_expressions():
    md = fake_omexml.FakeOMEXML(fake_omexml.FULL_SPEC)
    assert float(md.instrument().Objective.LensNA) == 1.4
    assert int(md.get_image_count()) == 1
    assert md.image().Name == 'Scene #1'
    pixels = md.image().Pixels
    assert (pixels.SizeZ, pixels.SizeT, pixels.SizeC) == (2, 3, 2)
    assert (pixels.PhysicalSizeX, pixels.PhysicalSizeZ) == (0.1, 0.5)
    assert pixels.node.get('TimeIncrement') == 7.5
    assert pixels.node.get('TimeIncrementUnit') == 's'
    assert pixels.node.get('PhysicalSizeXUnit') == '\u00b5m'
    assert pixels.node.get('Unknown') is None
    assert pixels.Channel(1).Name == 'mCherry'
    assert pixels.Channel(1).node.get('EmissionWavelength') == 610.0
    with pytest.raises(IndexError):
        pixels.Channel(2)
    assert md.accessed[:3] == [
        'instrument()', 'instrument().Objective.LensNA', 'get_image_count()'
    ]
    assert 'image().Pixels.Channel(1).node.get(EmissionWavelength)' in (
        md.accessed
    )


def test_fake_omexml_raises_per_spec():
    spec = fake_omexml.spec_with(
        LensNA=fake_omexml.RAISE, SizeT=None,
        SizeZ=fake_omexml.raises('ValueError', 'boom'),
        node={'TimeIncrement': fake_omexml.raises('KeyError')},
        channels=[{'Name': fake_omexml.np_str('GFP')}, fake_omexml.RAISE],
        Name=fake_omexml.DELETE, omexml_string='<OME/>',
    )
    md = fake_omexml.FakeOMEXML(spec)
    with pytest.raises(AttributeError):
        md.instrument().Objective.LensNA
    with pytest.raises(ValueError, match='boom'):
        md.image().Pixels.SizeZ
    assert md.image().Pixels.SizeT is None
    with pytest.raises(KeyError):
        md.image().Pixels.node.get('TimeIncrement')
    assert md.image().Pixels.node.get('TimeIncrementUnit') is None
    with pytest.raises(AttributeError):
        md.image().Name
    name = md.image().Pixels.Channel(0).Name
    assert type(name) is np.str_ and name == 'GFP'
    assert md.image().Pixels.Channel(0).node.get('EmissionWavelength') is None
    with pytest.raises(AttributeError):
        md.image().Pixels.Channel(1)
    assert md.omexml_string == '<OME/>'
    for level in fake_omexml.LEVELS:
        md = fake_omexml.FakeOMEXML(
            fake_omexml.spec_with(**{level: fake_omexml.RAISE})
        )
        with pytest.raises(AttributeError):
            if level in ('instrument', 'Objective'):
                md.instrument().Objective.LensNA
            elif level == 'Channel':
                md.image().Pixels.Channel(0)
            else:
                md.image().Pixels.node.get('TimeIncrement')
    empty = fake_omexml.FakeOMEXML({})
    with pytest.raises(AttributeError):
        empty.get_image_count()


def test_cascade_grid_covers_every_field():
    grid = fake_omexml.cascade_grid()
    assert json.loads(json.dumps(grid)) == grid
    assert grid == fake_omexml.cascade_grid()
    for field in fake_omexml.ATTRIBUTE_FIELDS:
        for prefix in ('raise', 'none', 'absent'):
            assert f'{prefix}:{field}' in grid
    for key in fake_omexml.NODE_FIELDS:
        assert f'raise:node.{key}' in grid and f'none:node.{key}' in grid
    for level in fake_omexml.LEVELS:
        assert f'raise:{level}' in grid
    expressions = {
        'LensNA': lambda m: m.instrument().Objective.LensNA,
        'image_count': lambda m: m.get_image_count(),
        'Name': lambda m: m.image().Name,
        'SizeZ': lambda m: m.image().Pixels.SizeZ,
        'SizeT': lambda m: m.image().Pixels.SizeT,
        'SizeC': lambda m: m.image().Pixels.SizeC,
        'PhysicalSizeX': lambda m: m.image().Pixels.PhysicalSizeX,
        'PhysicalSizeY': lambda m: m.image().Pixels.PhysicalSizeY,
        'PhysicalSizeZ': lambda m: m.image().Pixels.PhysicalSizeZ,
    }
    for field, expression in expressions.items():
        assert expression(fake_omexml.FakeOMEXML(grid['full'])) is not None
        with pytest.raises(AttributeError):
            expression(fake_omexml.FakeOMEXML(grid[f'raise:{field}']))
        assert expression(fake_omexml.FakeOMEXML(grid[f'none:{field}'])) \
            is None
    md = fake_omexml.FakeOMEXML(grid['strings'])
    assert int(md.image().Pixels.SizeT) == 3
    assert float(md.image().Pixels.PhysicalSizeX) == 0.1


# ================================================================== manifest
def _make_tree(tmp_path):
    src, dst = tmp_path / 'src dir, x', tmp_path / 'dst'
    images = dst / 'Position_1' / 'Images'
    images.mkdir(parents=True)
    (dst / 'Position_2' / 'Images').mkdir(parents=True)
    src.mkdir()
    raw = synthetic.make_imagej_tif(str(src / 'raw.tif'))
    data = synthetic.encoded_array('TZYX', T=3, Z=2, c0=1, Y=8, X=9)
    tifffile.imwrite(
        str(images / 'x_s1_GFP.tif'), data, imagej=True,
        metadata={'axes': 'TZYX', 'spacing': 0.5, 'unit': 'um',
                  'finterval': 7.5},
        resolution=(10.0, 10.0)
    )
    (images / 'x_s1_metadata.csv').write_text(
        'Description,values\r\nLensNA,1.4\r\nSizeT,3\r\n', encoding='utf-8'
    )
    (images / 'x_s1_metadataXML.txt').write_text('None', encoding='utf-8')
    (images / 'x_s1_notes.txt').write_bytes(b'notes')
    ini = (
        '[channel_name.GFP]\n'
        f'source_filepath = {os.path.join(str(src), "raw.tif")}\n'
        'frames_range = 0,3\n'
    )
    (images / 'x_s1_symlink.ini').write_text(ini, encoding='utf-8')
    np.savez(str(images / 'x_segm.npz'), np.zeros((2, 3), np.uint32))
    np.save(str(images / 'x_arr.npy'), np.arange(4))
    h5py = pytest.importorskip('h5py')
    with h5py.File(str(images / 'x_s1_RFP.h5'), 'w') as h5:
        h5.create_dataset('data', data=data, chunks=(1, 1, 8, 9))
    return src, dst, raw


def test_build_manifest_entries(tmp_path):
    src, dst, _ = _make_tree(tmp_path)
    placeholders = {'<SRC>': str(src), '<DST>': str(dst)}
    m = manifest.build_manifest(str(dst), placeholders=placeholders)
    prefix = 'Position_1/Images/x_s1_'
    assert m['Position_2/Images'] == {'kind': 'empty_dir'}
    tif = m[prefix + 'GFP.tif']
    assert tif['kind'] == 'tif' and tif['axes'] == 'TZYX'
    assert tif['shape'] == [3, 2, 8, 9] and tif['dtype'] == 'uint16'
    assert tif['origins'] == [[0, 0, 1, 0], [0, 1, 1, 0], [0, 2, 1, 0]]
    assert tif['imagej']['spacing'] == 0.5
    assert tif['imagej']['finterval'] == 7.5
    assert tif['resolution'][0] == [10, 1]
    assert tif['ome'] is None and tif['other_series'] == []
    assert m[prefix + 'metadata.csv']['rows'] == [
        ['Description', 'values'], ['LensNA', '1.4'], ['SizeT', '3']
    ]
    xml = m[prefix + 'metadataXML.txt']
    assert xml['is_literal_None'] is True and xml['nonempty'] is True
    ini = m[prefix + 'symlink.ini']
    assert ini['sections']['channel_name.GFP']['source_filepath'] == (
        '<SRC>/raw.tif'
    )
    h5 = m[prefix + 'RFP.h5']['datasets']['data']
    assert h5['shape'] == [3, 2, 8, 9] and h5['chunks'] == [1, 1, 8, 9]
    assert h5['sha256'] == tif['sha256']
    assert h5['origins'] == tif['origins']
    assert m['Position_1/Images/x_segm.npz']['arrays']['arr_0']['shape'] == [
        2, 3
    ]
    assert m['Position_1/Images/x_arr.npy']['dtype'] == str(
        np.arange(4).dtype
    )
    assert m[prefix + 'notes.txt'] == {
        'kind': 'file', 'sha256': hashlib.sha256(b'notes').hexdigest()
    }
    assert manifest.ENV_KEY in m

    src_m = manifest.build_manifest(str(src), placeholders=placeholders)
    assert src_m['raw.tif']['axes'] == 'TZCYX'
    assert len(src_m['raw.tif']['origins']) == 3
    assert manifest.build_manifest(
        str(tmp_path / 'missing'), include_env=False
    ) == {'.': {'kind': 'missing'}}
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert manifest.build_manifest(str(empty), include_env=False) == {
        '.': {'kind': 'empty_dir'}
    }
    excluded = manifest.build_manifest(
        str(dst), exclude=('*.h5', 'Position_2'), include_env=False
    )
    assert prefix + 'RFP.h5' not in excluded
    assert 'Position_2/Images' not in excluded


def test_dump_json_puts_short_scalar_lists_on_one_line(tmp_path):
    path = str(tmp_path / 'x.json')
    long_text = 'y' * 80
    manifest.dump_json(
        {'b': (1, 2.0, None), 'a': [{'k': ['x']}], 'c': [long_text]}, path
    )
    with open(path, encoding='utf-8') as file:
        assert file.read() == (
            '{\n'
            ' "a": [\n'
            '  {\n'
            '   "k": ["x"]\n'
            '  }\n'
            ' ],\n'
            ' "b": [1, 2.0, null],\n'
            ' "c": [\n'
            f'  "{long_text}"\n'
            ' ]\n'
            '}\n'
        )


def test_manifest_json_round_trip_and_diffs(tmp_path):
    src, dst, _ = _make_tree(tmp_path)
    placeholders = {'<SRC>': str(src), '<DST>': str(dst)}
    golden = manifest.build_manifest(str(dst), placeholders=placeholders)
    path = str(tmp_path / 'golden.json')
    manifest.dump_json(golden, path)
    with open(path, 'rb') as file:
        first = file.read()
    manifest.dump_json(manifest.load_json(path), path)
    with open(path, 'rb') as file:
        assert file.read() == first
    loaded = manifest.load_json(path)
    assert loaded == golden
    assert manifest.diff_manifests(golden, loaded) == []

    loaded[manifest.ENV_KEY] = {'python': 'other'}
    loaded['Position_1/Images/x_s1_metadataXML.txt'][manifest.INFO_KEY] = {}
    assert manifest.diff_manifests(golden, loaded) == []

    images = dst / 'Position_1' / 'Images'
    (images / 'x_s1_metadata.csv').write_text(
        'Description,values\nLensNA,1.2\nSizeT,3\n', encoding='utf-8'
    )
    (images / 'x_s1_notes.txt').unlink()
    (images / 'x_s1_new.txt').write_bytes(b'')
    actual = manifest.build_manifest(str(dst), placeholders=placeholders)
    diffs = manifest.diff_manifests(golden, actual)
    prefix = 'Position_1/Images/x_s1_'
    assert [d.key for d in diffs] == [
        prefix + 'metadata.csv/rows/1/1', prefix + 'new.txt',
        prefix + 'notes.txt',
    ]
    assert (diffs[0].old, diffs[0].new) == ('1.4', '1.2')
    assert diffs[1].old is manifest.ABSENT
    assert diffs[2].new is manifest.ABSENT
    assert 'rows/1/1' in manifest.format_diffs(diffs)

    shorter = json.loads(json.dumps(golden))
    shorter[prefix + 'metadata.csv']['rows'].pop()
    keys = [d.key for d in manifest.diff_manifests(golden, shorter)]
    assert keys == [prefix + 'metadata.csv/rows']
    flipped = json.loads(json.dumps(golden))
    flipped[prefix + 'metadataXML.txt']['nonempty'] = 1
    assert [d.key for d in manifest.diff_manifests(golden, flipped)] == [
        prefix + 'metadataXML.txt/nonempty'
    ]


def test_manifest_unreadable_and_multiseries(tmp_path):
    synthetic.make_truncated_tif(str(tmp_path / 'bad.tif'), keep_bytes=4)
    synthetic.make_ome_tif(str(tmp_path / 'm.ome.tif'), T=2)
    m = manifest.build_manifest(str(tmp_path), include_env=False)
    assert m['bad.tif']['kind'] == 'unreadable'
    multi = m['m.ome.tif']
    assert [s['origins'] for s in multi['other_series']] == [
        [[1, 0, 0, 0], [1, 1, 0, 0]], [[2, 0, 0, 0], [2, 1, 0, 0]]
    ]
    images = multi['ome']['images']
    assert [i['Name'] for i in images] == ['Scene #1', 'Scene #2', 'Scene #3']
    assert images[0]['Pixels']['TimeIncrement'] == '7.5'
    assert images[0]['Channels'][1] == {
        'Name': 'mCherry', 'EmissionWavelength': '610.0'
    }


def test_normalize_path_text(tmp_path):
    root = str(tmp_path / 'dst')
    placeholders = {'<DST>': root, '<SRC>': str(tmp_path / 'src')}
    backslash = root.replace('/', '\\')
    assert manifest.normalize_path_text(
        backslash + '\\Position_1\\a.tif', placeholders
    ) == '<DST>/Position_1/a.tif'
    assert manifest.normalize_path_text(
        root.replace('\\', '/') + '/x', placeholders
    ) == '<DST>/x'
    text = f'saving data to {backslash}\\Position_1\\Images...'
    assert manifest.normalize_path_text(text, placeholders) == (
        'saving data to <DST>/Position_1/Images...'
    )
    other = root + '2' + os.sep + 'a'
    assert manifest.normalize_path_text(other, placeholders) == other
    if os.name == 'nt':
        assert manifest.normalize_path_text(
            root.upper() + '\\a', placeholders
        ) == '<DST>/a'
    assert manifest.normalize_path_text('plain', None) == 'plain'


# ============================================================ qt_free_rules
GOOD_NEW_MODULE = '''"""Docstring mentioning QDialogMetadataXML and QThread is fine."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .. import myutils, load, io, acdc_regex, config, printl
from cellacdc import myutils as mu
import cellacdc.acdc_bioio_bioformats._utils
from . import errors
from .errors import UsageError
from ..acdc_bioio_bioformats import _utils

if TYPE_CHECKING:
    import qtpy
    from ..myutils import listdir
    from .. import apps

logger = logging.getLogger(__name__)
QUEUE_POLL_S = 0.2
QC = 1


def f(path: str, flag: 'bool | None' = None) -> 'list[str]':
    from .. import colors
    from ..myutils import listdir
    return listdir(path) + [colors, np, mu, myutils, load, io, acdc_regex,
                            config, printl, errors, UsageError, _utils]
'''


def test_good_module_passes_every_rule():
    violations = qt_free_rules.check_source(
        GOOD_NEW_MODULE, relpath='acdc_datastruct/good.py', qf=True,
        new_file=True, parent_never_opens_raw=True
    )
    assert violations == [], qt_free_rules.format_violations(violations)


@pytest.mark.parametrize('snippet, flags, rule', [
    ('import qtpy', {}, 'qt-import'),
    ('from PyQt6.QtCore import pyqtSignal', {}, 'qt-import'),
    ('try:\n    import pyqtgraph\nexcept ImportError:\n    pass', {},
     'qt-import'),
    ('class A:\n    import vispy', {}, 'qt-import'),
    ('if TYPE_CHECKING:\n    pass\nelse:\n    import PySide6', {},
     'qt-import'),
    ('with open("x"):\n    import PySide2', {}, 'qt-import'),
    ('from .. import apps', {}, 'forbidden-import'),
    ('import cellacdc.widgets', {}, 'forbidden-import'),
    ('from cellacdc import colors', {}, 'forbidden-import'),
    ('from ..dataStruct import bioFormatsWorker', {}, 'forbidden-import'),
    ('from .. import prompts', {}, 'forbidden-import'),
    ('from .. import core', {}, 'not-allowlisted'),
    ('import cellacdc.core', {}, 'not-allowlisted'),
    ('from ..myutils import listdir', {}, 'import-style'),
    ('from cellacdc.load import loadData', {}, 'import-style'),
    ('from cellacdc.io import rename', {}, 'import-style'),
    ('from ..core import x', {}, 'import-style'),
    ('x: "QWidget" = None', {}, 'qt-name'),
    ('def f(w: "list[QThread]"):\n    pass', {}, 'qt-name'),
    ('def f() -> "QDialog":\n    pass', {}, 'qt-name'),
    ('class QDialogFoo:\n    pass', {}, 'qt-name'),
    ('def f(self):\n    return self.QMainWindow', {}, 'qt-name'),
    ('import numpy', {'stdlib_only': True}, 'stdlib-only'),
    ('from . import errors', {'stdlib_only': True}, 'stdlib-only'),
    ('from cellacdc import myutils', {'stdlib_only': True}, 'stdlib-only'),
    ('import json', {'new_file': True}, 'future-annotations'),
    ('print(1)', {'new_file': True}, 'no-print'),
    ('log = print', {'new_file': True}, 'no-print'),
    ('def f():\n    from .. import _process', {'new_file': True},
     'forbidden-name'),
    ('def f(u):\n    u.check_raise_exception()', {'new_file': True},
     'forbidden-name'),
    ('x = bioio_sample_data_folderpath', {'new_file': True},
     'forbidden-name'),
    ('def f():\n    from bioio import BioImage',
     {'parent_never_opens_raw': True}, 'parent-opens-raw'),
    ('def f():\n    import bioio_base.exceptions',
     {'parent_never_opens_raw': True}, 'parent-opens-raw'),
    ('def f():\n    from ..acdc_bioio_bioformats import ImageReader',
     {'parent_never_opens_raw': True}, 'parent-opens-raw'),
    ('def f():\n    from ..acdc_bioio_bioformats.reader import x',
     {'parent_never_opens_raw': True}, 'parent-opens-raw'),
    ('from cellacdc.acdc_bioio_bioformats import raw_metadata',
     {'parent_never_opens_raw': True}, 'parent-opens-raw'),
    ('def f():\n    from ..acdc_bioio_bioformats import writer',
     {'parent_never_opens_raw': True}, 'parent-opens-raw'),
    ('if TYPE_CHECKING:\n    import cellacdc.acdc_bioio_bioformats.writer',
     {'parent_never_opens_raw': True}, 'parent-opens-raw'),
    ('def f(:', {}, 'syntax-error'),
])
def test_bad_snippets_are_flagged(snippet, flags, rule):
    violations = qt_free_rules.check_source(
        snippet, relpath='acdc_datastruct/bad.py', **flags
    )
    assert rule in {v.rule for v in violations}, violations


@pytest.mark.parametrize('snippet, flags', [
    ('from __future__ import annotations\nimport json', {'stdlib_only': True}),
    ('def f():\n    import numpy', {'stdlib_only': True}),
    ('if TYPE_CHECKING:\n    import numpy', {'stdlib_only': True}),
    ('from . import writers', {'parent_never_opens_raw': True}),
    ('from ..acdc_bioio_bioformats import _utils',
     {'parent_never_opens_raw': True}),
    ('import bioio_tifffile', {'parent_never_opens_raw': True}),
    ('"""Docstring only."""', {'new_file': True}),
    ('', {'new_file': True}),
    ('from .. import printl, logs_path', {}),
    ('import cellacdc', {}),
    ('import cellacdc._entry_points', {}),
    ('from ..acdc_bioio_bioformats import EXTENSION_PACKAGE_MAPPER', {}),
    ('QUEUE_POLL_S = 0.2\nQC = 1\nx = "QWidget"', {}),
    ('def f():\n    from qtpy.QtCore import Signal', {}),
])
def test_good_snippets_pass(snippet, flags):
    violations = qt_free_rules.check_source(
        snippet, relpath='acdc_datastruct/ok.py', qf=True, **flags
    )
    assert violations == [], qt_free_rules.format_violations(violations)


def test_violation_fields():
    [violation] = qt_free_rules.check_source(
        'import json\n\nimport qtpy\n', filename='x.py'
    )
    assert (violation.file, violation.line, violation.rule, violation.text) \
        == ('x.py', 3, 'qt-import', 'import qtpy')
    assert str(violation) == 'x.py:3: [qt-import] import qtpy'


def test_grandfathered_statements():
    statement = 'from ..myutils import safe_get_or_call'
    assert qt_free_rules.check_source(
        statement, relpath='acdc_bioio_bioformats/reader.py'
    ) == []
    assert [v.rule for v in qt_free_rules.check_source(
        statement, relpath='acdc_bioio_bioformats/other.py'
    )] == ['import-style']
    assert qt_free_rules.check_source(
        'from cellacdc.config import ConfigParser',
        relpath='acdc_bioio_bioformats/_utils.py'
    ) == []
    assert ('acdc_bioio_bioformats/reader.py', statement) in (
        qt_free_rules.GRANDFATHERED
    )


MAIN_READER_HEADER = '''import os
import re

import numpy as np

from .. import printl
from .. import colors
from ..myutils import safe_get_or_call
from . import _utils
from . import install, EXTENSION_PACKAGE_MAPPER
from . import EXTENSION_BIOIMAGE_KWARGS_MAPPER
from . import EXTENSION_METADATA_ATTR_MAPPER

def set_reader(image_filepath, **kwargs):
    try:
        import bioio_bioformats
    except ImportError:
        from bioio_base.exceptions import UnsupportedFileFormatError
    print(key)
'''


def _assert_only_colors_violation(violations):
    assert [(v.line, v.rule, v.text.split(' (')[0]) for v in violations] == [
        (7, 'forbidden-import', 'from .. import colors')
    ], qt_free_rules.format_violations(violations)


def test_engine_on_main_reader_header():
    _assert_only_colors_violation(qt_free_rules.check_source(
        MAIN_READER_HEADER, relpath='acdc_bioio_bioformats/reader.py',
        qf=True, stdlib_only=False, new_file=False,
        parent_never_opens_raw=False
    ))


def test_engine_on_main_reader_file(tmp_path):
    source = _git_show('cellacdc/acdc_bioio_bioformats/reader.py')
    path = tmp_path / 'reader.py'
    path.write_text(source, encoding='utf-8')
    _assert_only_colors_violation(qt_free_rules.check_file(
        str(path), qf=True, stdlib_only=False, new_file=False,
        parent_never_opens_raw=False,
        relpath='acdc_bioio_bioformats/reader.py'
    ))


def test_check_file_resolves_repo_relative_paths():
    violations = qt_free_rules.check_file(
        'tests/datastruct/qt_free_rules.py', qf=True, relpath='x.py'
    )
    assert violations == [], qt_free_rules.format_violations(violations)
    assert qt_free_rules._derive_relpath(os.path.join(
        conditions.REPO_ROOT, 'cellacdc', 'acdc_bioio_bioformats', 'reader.py'
    )) == 'acdc_bioio_bioformats/reader.py'


@pytest.mark.parametrize('snippet, flagged', [
    ('from .acdc_datastruct import convert', True),
    ('from . import acdc_datastruct', True),
    ('import cellacdc.acdc_datastruct.metadata', True),
    ('from cellacdc.acdc_datastruct.sources import RawLayout', True),
    ('from cellacdc.acdc_bioio_bioformats import writer', True),
    ('from .acdc_bioio_bioformats import raw_metadata', True),
    ('try:\n    from .acdc_datastruct import convert\nexcept Exception:\n'
     '    pass', True),
    ('from . import acdc_bioio_bioformats', False),
    ('class D:\n    def m(self):\n        from .acdc_datastruct import x', False),
    ('if TYPE_CHECKING:\n    from .acdc_datastruct import convert', False),
])
def test_check_gui_lazy_imports_snippets(snippet, flagged):
    violations = qt_free_rules.check_gui_lazy_source(snippet)
    assert bool(violations) is flagged, violations
    assert all(v.rule == 'gui-eager-import' for v in violations)


def test_check_gui_lazy_imports_on_main(tmp_path):
    for name in qt_free_rules.GUI_LAZY_IMPORT_FILES:
        path = tmp_path / name
        path.write_text(_git_show(f'cellacdc/{name}'), encoding='utf-8')
        assert qt_free_rules.check_gui_lazy_imports(str(path)) == []


def test_rules_for_gives_the_flags_of_each_file():
    rules = qt_free_rules.rules_for
    assert rules('_entry_points.py') == {
        'qf': True, 'stdlib_only': True, 'new_file': True,
        'parent_never_opens_raw': False,
    }
    assert rules('data_cli.py')['stdlib_only']
    assert rules('data_cli.py')['parent_never_opens_raw']
    assert rules('acdc_datastruct/runner.py')['stdlib_only']
    assert rules('acdc_datastruct/_worker.py')['stdlib_only']
    assert rules('acdc_datastruct/tasks.py') == {
        'qf': True, 'stdlib_only': False, 'new_file': True,
        'parent_never_opens_raw': False,
    }
    for name in ('convert', 'backends', 'planning', 'sources', 'metadata',
                 'writers'):
        assert rules(f'acdc_datastruct/{name}.py')['parent_never_opens_raw']
    assert rules('acdc_datastruct/__init__.py')['new_file']
    assert rules('acdc_bioio_bioformats/reader.py') == {
        'qf': True, 'stdlib_only': False, 'new_file': False,
        'parent_never_opens_raw': False,
    }
    assert rules('acdc_bioio_bioformats/raw_metadata.py')['new_file']
    assert rules('acdc_bioio_bioformats\\writer.py')['new_file']
    assert rules('apps.py') is None and rules('colors.py') is None
    assert rules('acdc_datastruct/sub/x.py') is None


def test_imports_of_reports_module_level_and_resolution(tmp_path):
    path = tmp_path / 'writers.py'
    path.write_text(
        'from . import planning\n'
        'if TYPE_CHECKING:\n    from .errors import X\n'
        'def f():\n    from .. import colors\n',
        encoding='utf-8'
    )
    refs = qt_free_rules.imports_of(
        str(path), relpath='acdc_datastruct/writers.py'
    )
    assert [(r.line, r.base, r.module_level, r.type_checking) for r in refs] \
        == [
            (1, 'cellacdc.acdc_datastruct', True, False),
            (3, 'cellacdc.acdc_datastruct.errors', True, True),
            (5, 'cellacdc', False, False),
        ]
    assert refs[0].targets == ('cellacdc.acdc_datastruct.planning',)
    assert refs[2].modules == ('cellacdc.colors',)
