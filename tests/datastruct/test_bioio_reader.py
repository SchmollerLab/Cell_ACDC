"""acdc_bioio_bioformats.reader and _utils.get_supported_image_extensions:
lazy colors, the bioimage_factory seam, dims/close, the faba2925
TimeIncrement round trip of #1109 (credit keejkrej), Node no-clobber,
case-insensitive set_reader without mapper mutation.

Everything runs on FakeBioImage (no bioio needed) except the cases marked
requires_bioio.
"""
from __future__ import annotations

import datetime
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import fake_bioio, qt_free_rules, synthetic
from .conditions import requires_bioio
from .fake_bioio import FakeBioImage

from cellacdc import acdc_bioio_bioformats as bioformats
from cellacdc.acdc_bioio_bioformats import _utils, reader

READER_PY = os.path.join(
    qt_free_rules.PACKAGE_DIR, 'acdc_bioio_bioformats', 'reader.py'
)


@pytest.fixture(autouse=True)
def _reset_fake_log():
    FakeBioImage.reset_log()
    yield
    FakeBioImage.reset_log()


def _fake_raw(tmp_path, filename='raw.nd2', **spec_kwargs):
    spec = fake_bioio.make_spec(**spec_kwargs)
    return fake_bioio.write_fake_raw(str(tmp_path / filename), spec)


def _open(path, **kwargs):
    return reader.ImageReader(path, bioimage_factory=FakeBioImage, **kwargs)


def _block_bioio(monkeypatch):
    for name in ('bioio', 'bioio_base', 'bioio_base.exceptions'):
        monkeypatch.setitem(sys.modules, name, None)


# --------------------------------------------------------------- lazy colors
def test_colors_is_imported_only_inside_functions():
    colors_refs = [
        ref for ref in qt_free_rules.imports_of(READER_PY)
        if 'cellacdc.colors' in ref.targets
    ]
    assert colors_refs, 'reader.py should still use cellacdc.colors'
    assert not [ref for ref in colors_refs if ref.module_level]


@pytest.mark.parametrize('lazy_load', [True, False])
def test_rgb_planes_are_converted_to_grayscale(tmp_path, lazy_load):
    path = _fake_raw(
        tmp_path, T=2, C=1, Z=1, Y=6, X=5, rgb=True, dtype='uint8'
    )
    with _open(path, lazy_load=lazy_load) as image_reader:
        img = image_reader.read(c=0, z=0, t=1)
    assert img.shape == (6, 5)
    assert img.dtype == np.uint8


# -------------------------------------------------------- bioimage_factory
def test_factory_replaces_bioio(tmp_path, monkeypatch):
    _block_bioio(monkeypatch)
    path = _fake_raw(tmp_path, filename='raw.CZI')
    image_reader = _open(path)
    assert isinstance(image_reader._bioioimage, FakeBioImage)
    # set_reader extras are passed to the factory (case-insensitive ext)
    assert FakeBioImage.log[0] == ('open', path, {'use_aicspylibczi': True})


def test_factory_is_retried_without_extension_kwargs(tmp_path):
    path = _fake_raw(tmp_path, filename='raw.czi')
    calls = []

    def factory(image, **kwargs):
        calls.append(kwargs)
        if kwargs:
            raise TypeError('unexpected keyword argument')
        return FakeBioImage(image)

    reader.ImageReader(path, bioimage_factory=factory)
    assert calls == [{'use_aicspylibczi': True}, {}]


@pytest.mark.parametrize('lazy_load', [True, False])
def test_read_returns_the_requested_plane(tmp_path, lazy_load):
    path = _fake_raw(tmp_path, scenes=2, T=3, C=2, Z=2)
    with _open(path, lazy_load=lazy_load) as image_reader:
        for series, t, c, z in ((1, 2, 1, 0), (0, 1, 0, 1), (1, 0, 1, 1)):
            img = image_reader.read(c=c, z=z, t=t, series=series)
            assert img.shape == (16, 20)
            assert synthetic.decode_pixel(img[0, 0]) == (series, t, c, z)


# ---------------------------------------------------------- dims and close
TWO_SCENES = [
    {'T': 3, 'C': 2, 'Z': 4, 'Y': 6, 'X': 7},
    {'T': 5, 'C': 2, 'Z': 1, 'Y': 6, 'X': 7},
]


def test_dims_are_returned_in_tczyx_order(tmp_path):
    path = fake_bioio.write_fake_raw(
        str(tmp_path / 'raw.nd2'), {'scenes': TWO_SCENES, 'dims': 'ZTCYX'}
    )
    with _open(path) as image_reader:
        assert image_reader.dims() == (3, 2, 4, 6, 7)
        assert image_reader.dims(1) == (5, 2, 1, 6, 7)


@pytest.mark.parametrize('lazy_load', [True, False])
def test_dims_per_scene_keeps_reads_consistent(tmp_path, lazy_load):
    path = fake_bioio.write_fake_raw(
        str(tmp_path / 'raw.nd2'), {'scenes': TWO_SCENES}
    )
    with _open(path, lazy_load=lazy_load) as image_reader:
        assert image_reader.dims() == (3, 2, 4, 6, 7)
        assert image_reader.dims(1) == (5, 2, 1, 6, 7)
        # dims() switching the scene must not break the next reads
        img = image_reader.read(c=1, z=0, t=4, series=1)
        assert synthetic.decode_pixel(img[0, 0]) == (1, 4, 1, 0)
        assert image_reader.dims(0) == (3, 2, 4, 6, 7)
        img = image_reader.read(c=0, z=3, t=2, series=0)
        assert synthetic.decode_pixel(img[0, 0]) == (0, 2, 0, 3)


def test_exit_closes_the_bioimage(tmp_path):
    path = _fake_raw(tmp_path)
    with _open(path) as image_reader:
        assert ('close', path) not in FakeBioImage.log
    assert ('close', path) in FakeBioImage.log
    assert image_reader._bioioimage.closed
    image_reader.close()     # idempotent


def test_close_without_close_method(tmp_path):
    # bioio 3 BioImage has no close(): nothing to do, no error
    bioimage = SimpleNamespace(current_scene_index=0)
    image_reader = reader.ImageReader(
        str(tmp_path / 'raw.nd2'), bioimage_factory=lambda path: bioimage
    )
    image_reader.close()
    with image_reader:
        pass


# ------------------------------------------------------------------- Metadata
def test_metadata_init_from_file_returns_self(tmp_path):
    path = tmp_path / 'metadataXML.txt'
    path.write_text('<OME/>')
    metadata = bioformats.Metadata()
    assert metadata.init_from_file(str(path)) is metadata
    assert str(metadata) == '<OME/>'


# -------------------------------- TimeIncrement round trip (faba2925 of #1109)
def _round_trip(tmp_path, path):
    with _open(path) as image_reader:
        omexml = bioformats.OMEXML().init_from_bioimage(path, image_reader)
        text_path = str(tmp_path / 'metadata.txt')
        omexml.to_file(text_path)
    return bioformats.OMEXML().init_from_file(text_path, path)


def test_round_trip_keeps_time_increment(tmp_path, capsys):
    path = _fake_raw(
        tmp_path, T=4, C=2, Z=3, Y=8, X=9, time_interval_s=2.5,
        physical_pixel_sizes=[0.5, 0.1, 0.2],
        channel_names=['GFP', 'mCh'], np_str_channel_names=True
    )
    capsys.readouterr()
    omexml = _round_trip(tmp_path, path)
    assert capsys.readouterr().out == ''       # no debug print
    pixels = omexml.image().Pixels
    assert float(pixels.node.get('TimeIncrement')) == 2.5
    assert pixels.node.get('TimeIncrementUnit') == 's'
    assert omexml.bioimage.time_increment == 2.5
    assert omexml.bioimage.time_increment_unit == 's'
    assert (pixels.SizeT, pixels.SizeC, pixels.SizeZ) == (4, 2, 3)
    assert (pixels.SizeY, pixels.SizeX) == (8, 9)
    assert (
        pixels.PhysicalSizeZ, pixels.PhysicalSizeY, pixels.PhysicalSizeX
    ) == (0.5, 0.1, 0.2)
    assert [pixels.Channel(c).Name for c in range(2)] == ['GFP', 'mCh']
    assert all(type(name) is str for name in pixels.channel_names)
    assert omexml.get_image_count() == 1


def test_round_trip_without_time_increment(tmp_path):
    path = _fake_raw(tmp_path, time_interval_s=None)
    text_path = str(tmp_path / 'metadata.txt')
    with _open(path) as image_reader:
        omexml = bioformats.OMEXML().init_from_bioimage(path, image_reader)
        assert 'TimeIncrement' not in str(omexml)
        omexml.to_file(text_path)
    node = bioformats.OMEXML().init_from_file(text_path, path).Pixels.node
    with pytest.raises(ValueError):
        node.get('TimeIncrement')


def test_init_from_file_uses_defaults_for_missing_keys(tmp_path, capsys):
    text_path = tmp_path / 'metadata.txt'
    text_path.write_text('Image: x.nd2\nSizeC: 2\nChannels: not a list\n')
    omexml = bioformats.OMEXML().init_from_file(str(text_path), 'x.nd2')
    assert capsys.readouterr().out == ''
    pixels = omexml.image().Pixels
    assert (pixels.SizeC, pixels.SizeT, pixels.SizeZ) == (2, 1, 1)
    assert pixels.PhysicalSizeX == 1.0
    assert pixels.channel_names == ['ch0']
    assert omexml.get_image_count() == 1
    assert omexml.bioimage.time_increment is None
    assert omexml.bioimage.time_increment_unit is None


def test_init_from_bioimage_does_not_reopen(tmp_path):
    path = _fake_raw(tmp_path)
    with _open(path) as image_reader:
        omexml = bioformats.OMEXML().init_from_bioimage(path, image_reader)
        assert omexml.bioimage is image_reader._bioioimage
    assert FakeBioImage.opened_paths() == [path]


# ---------------------------------------------------------------------- Node
def _czi_like(time_interval_s, timelapse_s=None, standard_raises=False):
    def seconds(value):
        return None if value is None else datetime.timedelta(seconds=value)

    class BioImageLike:
        time_interval = seconds(time_interval_s)

        @property
        def standard_metadata(self):
            if standard_raises:
                raise RuntimeError('mapper failure')
            return SimpleNamespace(timelapse_interval=seconds(timelapse_s))

    return BioImageLike()


@pytest.mark.parametrize('standard_raises', [False, True])
def test_node_mapper_failure_keeps_time_interval(standard_raises):
    node = reader.Node('x.czi', _czi_like(2.0, standard_raises=standard_raises))
    assert node.get('TimeIncrement') == 2.0
    assert node.get('TimeIncrementUnit') == 's'


def test_node_mapper_value_used_for_upper_case_czi():
    node = reader.Node('x.CZI', _czi_like(None, timelapse_s=3.0))
    assert node.get('TimeIncrement') == 3.0


def test_node_mapper_none_value_is_not_stored(monkeypatch):
    monkeypatch.setitem(
        reader.EXTENSION_METADATA_ATTR_MAPPER, '.czi',
        {'TimeIncrement': 'missing_value'}
    )
    bioimage = SimpleNamespace(
        time_interval=datetime.timedelta(seconds=4.0), missing_value=None
    )
    assert reader.Node('x.czi', bioimage).get('TimeIncrement') == 4.0


def test_node_falls_back_to_time_increment_attribute():
    bioimage = reader.BioImageMetadata(
        1, 1, 1, 4, 4, 1.0, 1.0, 1.0, ['a'], 1,
        time_increment=4.0, time_increment_unit='ms'
    )
    node = reader.Node('x.nd2', bioimage)
    assert (node.get('TimeIncrement'), node.get('TimeIncrementUnit')) == (
        4.0, 'ms'
    )


# ---------------------------------------------------------------- set_reader
@pytest.mark.parametrize('filename', ['a.TIF', 'a.Tiff', 'a.tif', 'b.ND2'])
def test_set_reader_upper_case_uses_the_plugin_route(filename):
    assert reader.set_reader(filename) == ({}, {})


def test_set_reader_returns_fresh_dicts():
    before = {
        ext: dict(kwargs)
        for ext, kwargs in reader.EXTENSION_BIOIMAGE_KWARGS_MAPPER.items()
    }
    kwargs, other_kwargs = reader.set_reader('a.CZI')
    assert (kwargs, other_kwargs) == ({}, {'use_aicspylibczi': True})
    assert other_kwargs is not reader.EXTENSION_BIOIMAGE_KWARGS_MAPPER['.czi']
    other_kwargs['extra'] = 1
    assert reader.EXTENSION_BIOIMAGE_KWARGS_MAPPER == before


def test_set_reader_keeps_an_explicit_reader():
    marker = object()
    assert reader.set_reader('a.vsi', reader=marker) == (
        {'reader': marker}, {}
    )


@requires_bioio
def test_set_reader_imports_ims_reader_without_mutating_the_mapper():
    pytest.importorskip('imaris_ims_file_reader')
    _, other_kwargs = reader.set_reader('a.IMS')
    assert isinstance(other_kwargs['reader'], type)
    assert isinstance(
        reader.EXTENSION_BIOIMAGE_KWARGS_MAPPER['.ims']['reader'], str
    )


# --------------------------------------------- get_supported_image_extensions
def test_supported_extensions_without_bioio(monkeypatch):
    _block_bioio(monkeypatch)
    extensions = _utils.get_supported_image_extensions()
    assert set(bioformats.EXTENSION_PACKAGE_MAPPER) <= extensions
    assert '.vsi' in extensions       # Bio-Formats list
    assert all(ext == ext.strip() for ext in extensions)


@requires_bioio
def test_supported_extensions_are_stripped():
    extensions = _utils.get_supported_image_extensions()
    assert '.tif' in extensions
    assert all(ext == ext.strip() for ext in extensions)


# --------------------------------------------------------------- real bioio
@requires_bioio
def test_upper_case_tif_opens_with_the_tifffile_plugin(tmp_path):
    path = synthetic.make_imagej_tif(str(tmp_path / 'IMG.TIF'))
    with reader.ImageReader(path) as image_reader:
        bioimage = image_reader._bioioimage
        assert type(bioimage.reader).__module__.startswith('bioio_tifffile')
        assert image_reader.dims() == (3, 2, 2, 16, 20)
        img = image_reader.read(c=1, z=1, t=2)
    assert synthetic.decode_pixel(img[0, 0]) == (0, 2, 1, 1)


@requires_bioio
def test_legacy_metadata_child_objects_on_a_real_file(tmp_path):
    # what _read_metadata.py does in the legacy child
    path = synthetic.make_imagej_tif(str(tmp_path / 'img.tif'))
    metadata_xml = bioformats.get_omexml_metadata(path)
    omexml = bioformats.OMEXML().init_from_metadata(metadata_xml)
    assert 'ImageJ' in str(metadata_xml)
    text = str(omexml)
    assert 'SizeT: 3' in text and 'SizeC: 2' in text
    assert "Channels: ['" in text
