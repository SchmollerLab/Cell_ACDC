"""acdc_bioio_bioformats.raw_metadata: legacy accessor expressions
evaluated into the JSON-safe RawFileMetadata dict.

`extract_from_metadata_object` runs on the shared FakeOMEXML cascade grid
(each raising accessor -> None); `read_raw_metadata` on FakeBioImage files
(no bioio needed), synthetic '.ome.tif' files (tifffile only) and, with bioio
installed, a real ImageJ TIFF.
"""
from __future__ import annotations

import json
import math
import re

import pytest
import tifffile

from . import fake_bioio, fake_omexml, synthetic
from .conditions import requires_bioio
from .fake_bioio import FakeBioImage

from cellacdc.acdc_bioio_bioformats import raw_metadata, reader

KEYS = (
    'schema', 'path', 'source', 'reader', 'lens_na', 'size_s', 'size_z',
    'size_t', 'time_increment', 'time_increment_unit', 'size_c',
    'physical_size_x', 'physical_size_y', 'physical_size_z',
    'physical_size_unit', 'image_name', 'channel_names', 'em_wavelens',
    'scene_names', 'size_y', 'size_x', 'dtype', 'metadata_xml',
)
FAKE_READER = 'tests.datastruct.fake_bioio.FakeBioImage'


@pytest.fixture(autouse=True)
def _reset_fake_log():
    FakeBioImage.reset_log()
    yield
    FakeBioImage.reset_log()


def _assert_json_safe(raw):
    text = json.dumps(raw, allow_nan=True)
    assert json.loads(text) == raw


# ------------------------------------------- extract_from_metadata_object
FULL = {
    'lens_na': 1.4, 'size_s': 1, 'size_z': 2, 'size_t': 3,
    'time_increment': 7.5, 'time_increment_unit': 's', 'size_c': 2,
    'physical_size_x': 0.1, 'physical_size_y': 0.1, 'physical_size_z': 0.5,
    'physical_size_unit': fake_omexml.MICRO_SIGN_UM,
    'image_name': 'Scene #1',
    'channel_names': ['GFP', 'mCherry'], 'em_wavelens': [510.0, 610.0],
}
FIELD_KEYS = {
    'LensNA': 'lens_na', 'image_count': 'size_s', 'Name': 'image_name',
    'SizeZ': 'size_z', 'SizeT': 'size_t', 'SizeC': 'size_c',
    'PhysicalSizeX': 'physical_size_x', 'PhysicalSizeY': 'physical_size_y',
    'PhysicalSizeZ': 'physical_size_z',
    'node.TimeIncrement': 'time_increment',
    'node.TimeIncrementUnit': 'time_increment_unit',
    'node.PhysicalSizeXUnit': 'physical_size_unit',
}
PIXELS_KEYS = (
    'size_z', 'size_t', 'time_increment', 'time_increment_unit', 'size_c',
    'physical_size_x', 'physical_size_y', 'physical_size_z',
    'physical_size_unit',
)
NODE_KEYS = ('time_increment', 'time_increment_unit', 'physical_size_unit')
NO_CHANNEL = {'channel_names': [None], 'em_wavelens': [None]}


def _expected(name: str) -> dict:
    """Expected extraction of cascade_grid()[name], derived from the spec."""
    expected = dict(FULL)
    kind, _, field = name.partition(':')
    if name in ('full', 'strings'):
        pass
    elif name == 'empty':
        expected = {key: None for key in FULL}
        expected.update(NO_CHANNEL)
    elif name == 'scenes:3':
        expected['size_s'] = 3
    elif kind in ('raise', 'none', 'absent', 'bad') and field in FIELD_KEYS:
        expected[FIELD_KEYS[field]] = None
        if field == 'SizeC':        # range(1) when SizeC is unknown
            expected.update(channel_names=['GFP'], em_wavelens=[510.0])
    elif name in ('raise:instrument', 'raise:Objective'):
        expected['lens_na'] = None
    elif name == 'raise:image':
        expected.update({key: None for key in PIXELS_KEYS + ('image_name',)})
        expected.update(NO_CHANNEL)
    elif name == 'raise:Pixels':
        expected.update({key: None for key in PIXELS_KEYS})
        expected.update(NO_CHANNEL)
    elif name == 'raise:node':
        expected.update({key: None for key in NODE_KEYS})
    elif name in ('raise:Channel', 'channels:none'):
        expected.update(channel_names=[None, None], em_wavelens=[None, None])
    elif name == 'channels:short':
        expected.update(channel_names=['GFP', None], em_wavelens=[510.0, None])
    elif name == 'channels:np_str':
        pass
    elif name in ('channels:name_none', 'channels:name_raise'):
        expected['channel_names'] = [None, 'mCherry']
    elif name in ('channels:no_emission', 'channels:bad_emission'):
        expected['em_wavelens'] = [None, 610.0]
    elif name == 'channels:entry_raise':
        expected.update(channel_names=[None, 'mCherry'], em_wavelens=[None, 610.0])
    elif name == 'channels:size_c_3':
        expected.update(
            size_c=3, channel_names=['GFP', 'mCherry', None],
            em_wavelens=[510.0, 610.0, None]
        )
    elif name.startswith('units:time_'):
        spec = fake_omexml.cascade_grid()[name]
        expected['time_increment_unit'] = spec['node']['TimeIncrementUnit']
    elif name.startswith('units:length_'):
        spec = fake_omexml.cascade_grid()[name]
        expected['physical_size_unit'] = spec['node']['PhysicalSizeXUnit']
    else:
        raise AssertionError(f'no expectation for grid entry {name!r}')
    return expected


def _extract(metadata, **kwargs):
    kwargs = {
        'path': 'C:/raw/x.nd2', 'source': 'bioio', 'reader': 'fake',
        'metadata_xml': '<xml/>', **kwargs
    }
    return raw_metadata.extract_from_metadata_object(metadata, **kwargs)


@pytest.mark.parametrize('name', sorted(fake_omexml.cascade_grid()))
def test_extract_on_the_cascade_grid(name):
    spec = fake_omexml.cascade_grid()[name]
    raw = _extract(fake_omexml.FakeOMEXML(spec))
    assert tuple(raw) == KEYS
    assert {key: raw[key] for key in FULL} == _expected(name)
    assert raw['schema'] == raw_metadata.SCHEMA == 1
    assert (raw['path'], raw['source'], raw['reader']) == (
        'C:/raw/x.nd2', 'bioio', 'fake'
    )
    assert raw['metadata_xml'] == '<xml/>'
    assert (raw['scene_names'], raw['size_y'], raw['size_x'], raw['dtype']) == (
        [], None, None, None
    )
    for value in raw['channel_names'] + [
            raw['image_name'], raw['time_increment_unit'],
            raw['physical_size_unit']]:
        assert value is None or type(value) is str
    _assert_json_safe(raw)


def test_extract_evaluates_the_legacy_expressions_in_order():
    fake = fake_omexml.FakeOMEXML(fake_omexml.FULL_SPEC)
    _extract(fake)
    # leaf accessors only (not the image()/instrument()/Channel(c) calls)
    leaves = [
        accessor for accessor in fake.accessed
        if accessor not in ('image()', 'instrument()')
        and not re.fullmatch(r'image\(\)\.Pixels\.Channel\(\d+\)', accessor)
    ]
    pixels = 'image().Pixels'
    assert leaves == [
        'instrument().Objective.LensNA', 'get_image_count()',
        f'{pixels}.SizeZ', f'{pixels}.SizeT',
        f'{pixels}.node.get(TimeIncrement)',
        f'{pixels}.node.get(TimeIncrementUnit)', f'{pixels}.SizeC',
        f'{pixels}.PhysicalSizeX', f'{pixels}.PhysicalSizeY',
        f'{pixels}.PhysicalSizeZ', f'{pixels}.node.get(PhysicalSizeXUnit)',
        'image().Name',
        f'{pixels}.Channel(0).Name', f'{pixels}.Channel(1).Name',
        f'{pixels}.Channel(0).node.get(EmissionWavelength)',
        f'{pixels}.Channel(1).node.get(EmissionWavelength)',
    ]


def test_extract_keeps_legacy_casts():
    spec = fake_omexml.spec_with(SizeZ=2.9, SizeT='4', LensNA='0.95')
    raw = _extract(fake_omexml.FakeOMEXML(spec))
    assert (raw['size_z'], raw['size_t'], raw['lens_na']) == (2, 4, 0.95)
    spec = fake_omexml.spec_with(SizeT='4.0')       # int('4.0') raises
    assert _extract(fake_omexml.FakeOMEXML(spec))['size_t'] is None


def test_extract_nan_is_json_safe():
    spec = fake_omexml.spec_with(PhysicalSizeX='nan')
    raw = _extract(fake_omexml.FakeOMEXML(spec))
    assert math.isnan(raw['physical_size_x'])
    json.dumps(raw, allow_nan=True)


# ------------------------------------------- read_raw_metadata, FakeBioImage
def _fake_raw(tmp_path, filename='raw.nd2', **spec_kwargs):
    spec = fake_bioio.make_spec(**spec_kwargs)
    return fake_bioio.write_fake_raw(str(tmp_path / filename), spec)


def test_read_fake_bioio_file(tmp_path):
    path = _fake_raw(
        tmp_path, scenes=2, T=4, C=2, Z=3, Y=8, X=9, time_interval_s=2.5,
        physical_pixel_sizes=[0.5, 0.2, 0.1],
        channel_names=['GFP', 'mCh'], np_str_channel_names=True,
        metadata_text='<FakeMetadata a="1"/>'
    )
    raw = raw_metadata.read_raw_metadata(path, reader_factory=FakeBioImage)
    assert tuple(raw) == KEYS
    assert raw == {
        'schema': 1, 'path': path, 'source': 'bioio', 'reader': FAKE_READER,
        'lens_na': None, 'size_s': 2, 'size_z': 3, 'size_t': 4,
        'time_increment': 2.5, 'time_increment_unit': 's', 'size_c': 2,
        'physical_size_x': 0.1, 'physical_size_y': 0.2,
        'physical_size_z': 0.5, 'physical_size_unit': None,
        'image_name': None, 'channel_names': ['GFP', 'mCh'],
        'em_wavelens': [None, None], 'scene_names': [], 'size_y': 8,
        'size_x': 9, 'dtype': 'uint16',
        'metadata_xml': '<FakeMetadata a="1"/>',
    }
    assert all(type(name) is str for name in raw['channel_names'])
    _assert_json_safe(raw)
    # one reader, opened once and closed
    assert FakeBioImage.opened_paths() == [path]
    assert FakeBioImage.log[-1] == ('close', path)


def test_czi_mapper_provides_time_increment(tmp_path):
    path = _fake_raw(
        tmp_path, filename='raw.czi', time_interval_s=None,
        timelapse_interval_s=3.0
    )
    raw = raw_metadata.read_raw_metadata(path, reader_factory=FakeBioImage)
    assert raw['time_increment'] == 3.0
    assert raw['time_increment_unit'] is None


def test_czi_mapper_failure_keeps_time_interval(tmp_path):
    path = _fake_raw(
        tmp_path, filename='raw.CZI', time_interval_s=2.0,
        timelapse_interval_s=None
    )
    raw = raw_metadata.read_raw_metadata(path, reader_factory=FakeBioImage)
    assert (raw['time_increment'], raw['time_increment_unit']) == (2.0, 's')


def test_missing_physical_sizes_are_one_as_on_main(tmp_path):
    path = _fake_raw(tmp_path, physical_pixel_sizes=[None, 0.25, None])
    raw = raw_metadata.read_raw_metadata(path, reader_factory=FakeBioImage)
    assert (
        raw['physical_size_x'], raw['physical_size_y'],
        raw['physical_size_z']
    ) == (1.0, 0.25, 1.0)


@pytest.mark.parametrize('spec_kwargs', [
    {},
    {'scenes': 3, 'C': 3, 'physical_pixel_sizes': [None, 0.25, None]},
    {'time_interval_s': 0.5, 'np_str_channel_names': True,
     'channel_names': ['a b', "it's"]},
    {'physical_pixel_sizes': [2.0, 0.1, 0.1], 'time_interval_s': 60.0},
])
def test_bioio_values_match_the_legacy_text_round_trip(tmp_path, spec_kwargs):
    """Main's GUI evaluated the cascade on OMEXML.init_from_file(text written
    by the child): the direct extraction must give the same values (the
    TimeIncrement lines are the port of faba2925 from #1109, GV-02)."""
    path = _fake_raw(tmp_path, **spec_kwargs)
    raw = raw_metadata.read_raw_metadata(path, reader_factory=FakeBioImage)
    text_path = str(tmp_path / 'metadata.txt')
    with reader.ImageReader(path, bioimage_factory=FakeBioImage) as image_reader:
        reader.OMEXML().init_from_bioimage(path, image_reader).to_file(
            text_path
        )
    legacy = _extract(
        reader.OMEXML().init_from_file(text_path, path), path=path,
        reader=raw['reader'], metadata_xml=raw['metadata_xml']
    )
    for key in ('scene_names', 'size_y', 'size_x', 'dtype'):
        legacy[key] = raw[key]
    assert raw == legacy


@pytest.mark.parametrize('lazy_load', [True, False])
def test_lazy_load_is_passed_to_the_reader(tmp_path, lazy_load):
    path = _fake_raw(tmp_path, T=2, C=1, Z=1)
    raw = raw_metadata.read_raw_metadata(
        path, lazy_load=lazy_load, reader_factory=FakeBioImage
    )
    assert raw['size_t'] == 2
    reads = [event for event in FakeBioImage.log if event[0] == 'read']
    assert bool(reads) is (not lazy_load)     # eager mode loads scene 0


@pytest.mark.parametrize('stage', ['open', 'metadata'])
def test_reader_errors_propagate(tmp_path, stage):
    path = _fake_raw(tmp_path, fail={'stage': stage, 'how': 'raise'})
    with pytest.raises(RuntimeError, match='injected'):
        raw_metadata.read_raw_metadata(path, reader_factory=FakeBioImage)
    if stage == 'metadata':
        assert FakeBioImage.log[-1] == ('close', path)


# ---------------------------------------------------- read_raw_metadata, OME
def test_read_ome_tif(tmp_path):
    path = synthetic.make_ome_tif(str(tmp_path / 'multi.ome.tif'), lens_na=0.95)
    raw = raw_metadata.read_raw_metadata(path)
    assert tuple(raw) == KEYS
    assert raw == {
        'schema': 1, 'path': path, 'source': 'ome_tif',
        'reader': 'cellacdc.load.OMEXML',
        'lens_na': 0.95, 'size_s': 3, 'size_z': 2, 'size_t': 3,
        'time_increment': 7.5, 'time_increment_unit': 's', 'size_c': 2,
        'physical_size_x': 0.1, 'physical_size_y': 0.1,
        'physical_size_z': 0.5,
        'physical_size_unit': fake_omexml.MICRO_SIGN_UM,
        'image_name': 'Scene #1', 'channel_names': ['GFP', 'mCherry'],
        'em_wavelens': [510.0, 610.0],
        'scene_names': ['Scene #1', 'Scene #2', 'Scene #3'],
        'size_y': None, 'size_x': None, 'dtype': None,
        'metadata_xml': synthetic.read_ome_xml(path),
    }
    _assert_json_safe(raw)


def test_read_ome_tif_defaults_of_load_omexml(tmp_path):
    path = synthetic.make_ome_tif(
        str(tmp_path / 'UPPER.OME.TIF'), series=2, image_names=None,
        time_increment=None, em_wavelens=None
    )
    raw = raw_metadata.read_raw_metadata(path)
    assert raw['source'] == 'ome_tif'
    assert raw['lens_na'] == 1.4             # load.OMEXML default objective
    assert raw['scene_names'] == ['', '']
    assert raw['image_name'] == ''
    assert raw['time_increment'] is None
    assert raw['time_increment_unit'] is None
    assert raw['em_wavelens'] == [None, None]


def test_ome_tif_without_ome_xml_writes_none_text(tmp_path):
    path = str(tmp_path / 'plain.ome.tif')
    tifffile.imwrite(
        path, synthetic.encoded_array('YX', Y=4, X=5), ome=False,
        metadata=None
    )
    with tifffile.TiffFile(path) as tif:
        assert tif.ome_metadata is None
    raw = raw_metadata.read_raw_metadata(path)
    assert raw['metadata_xml'] == 'None'       # str(omexml_string), as main
    assert raw['scene_names'] == []
    assert raw['size_s'] == 1
    _assert_json_safe(raw)


# ------------------------------------------------------------- real bioio
@requires_bioio
def test_read_real_imagej_tif(tmp_path):
    path = synthetic.make_imagej_tif(str(tmp_path / 'img.tif'), T=3, C=2, Z=2)
    raw = raw_metadata.read_raw_metadata(path)
    assert raw['source'] == 'bioio'
    assert raw['reader'].startswith('bioio_tifffile.')
    assert (raw['size_s'], raw['size_t'], raw['size_c'], raw['size_z']) == (
        1, 3, 2, 2
    )
    assert (raw['size_y'], raw['size_x'], raw['dtype']) == (16, 20, 'uint16')
    assert (
        raw['physical_size_x'], raw['physical_size_y'],
        raw['physical_size_z']
    ) == (1.0, 1.0, 1.0)
    assert raw['time_increment'] is None
    assert len(raw['channel_names']) == 2
    assert 'ImageJ' in raw['metadata_xml']
    _assert_json_safe(raw)
