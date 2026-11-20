"""Child-side extraction of raw-file metadata into a JSON-safe dict.

The values are what the legacy accessor expressions of
`dataStruct.bioFormatsWorker.readMetadata` return on the legacy metadata
objects (`load.OMEXML` for '.ome.tif', `reader.OMEXML` otherwise), with None
wherever the legacy code took its except branch. The defaults and warnings of
the legacy cascade are applied later, by
`cellacdc.acdc_datastruct.metadata.resolve_metadata`.

This module runs where raw files may be opened (worker children, tests); the
parent process never imports it.
"""
from __future__ import annotations

import logging
from typing import Callable, Literal

from .. import load
from .reader import ImageReader, OMEXML

logger = logging.getLogger(__name__)

SCHEMA = 1

PHYSICAL_SIZE_KEYS = ('physical_size_x', 'physical_size_y', 'physical_size_z')


def _evaluate(expression: Callable[[], object]) -> object:
    """Value of `expression()`, or None when it raises (legacy except branch).
    A None value is returned as None too, which covers the legacy
    `if value is None: raise` checks."""
    try:
        return expression()
    except Exception:
        return None


def _as_str(value: object) -> str | None:
    """JSON-safe text (np.str_ -> str); None stays None."""
    return None if value is None else str(value)


def _qualified_name(obj: object) -> str:
    cls = type(obj)
    return f'{cls.__module__}.{cls.__qualname__}'


def extract_from_metadata_object(
        metadata: object, *, path: str, source: Literal['ome_tif', 'bioio'],
        reader: str, metadata_xml: str
    ) -> dict:
    """Evaluate the expressions of `dataStruct.readMetadata` on `metadata`.

    Each expression is evaluated on its own, in the legacy order, with the
    legacy `float()`/`int()` casts. A field is None when the legacy code
    took its except branch (including the `is None` checks of
    TimeIncrementUnit, PhysicalSizeUnit and ImageName). Channels: for c in
    range(size_c, or 1 when unknown), the name of `Channel(c)` (None if it
    raised or is None) and float of its 'EmissionWavelength' node entry
    (None if it raised). Returns the JSON-safe dict of `RawFileMetadata`
    fields; `scene_names`, `size_y`, `size_x` and `dtype` are left empty
    (set by `read_raw_metadata`).
    """
    def pixels():
        return metadata.image().Pixels

    lens_na = _evaluate(
        lambda: float(metadata.instrument().Objective.LensNA)
    )
    size_s = _evaluate(lambda: int(metadata.get_image_count()))
    size_z = _evaluate(lambda: int(pixels().SizeZ))
    size_t = _evaluate(lambda: int(pixels().SizeT))
    time_increment = _evaluate(
        lambda: float(pixels().node.get('TimeIncrement'))
    )
    time_increment_unit = _evaluate(
        lambda: _as_str(pixels().node.get('TimeIncrementUnit'))
    )
    size_c = _evaluate(lambda: int(pixels().SizeC))
    physical_size_x = _evaluate(lambda: float(pixels().PhysicalSizeX))
    physical_size_y = _evaluate(lambda: float(pixels().PhysicalSizeY))
    physical_size_z = _evaluate(lambda: float(pixels().PhysicalSizeZ))
    physical_size_unit = _evaluate(
        lambda: _as_str(pixels().node.get('PhysicalSizeXUnit'))
    )
    image_name = _evaluate(lambda: _as_str(metadata.image().Name))

    num_channels = size_c if size_c is not None else 1
    channel_names = [
        _evaluate(lambda c=c: _as_str(pixels().Channel(c).Name))
        for c in range(num_channels)
    ]
    em_wavelens = [
        _evaluate(
            lambda c=c: float(
                pixels().Channel(c).node.get('EmissionWavelength')
            )
        )
        for c in range(num_channels)
    ]

    return {
        'schema': SCHEMA,
        'path': path,
        'source': source,
        'reader': reader,
        'lens_na': lens_na,
        'size_s': size_s,
        'size_z': size_z,
        'size_t': size_t,
        'time_increment': time_increment,
        'time_increment_unit': time_increment_unit,
        'size_c': size_c,
        'physical_size_x': physical_size_x,
        'physical_size_y': physical_size_y,
        'physical_size_z': physical_size_z,
        'physical_size_unit': physical_size_unit,
        'image_name': image_name,
        'channel_names': channel_names,
        'em_wavelens': em_wavelens,
        'scene_names': [],
        'size_y': None,
        'size_x': None,
        'dtype': None,
        'metadata_xml': str(metadata_xml),
    }


def _ome_scene_names(omexml: object) -> list[str]:
    """Name attribute of every OME <Image> ('' if absent); [] without
    OME-XML."""
    if omexml.root is None:
        return []
    images = omexml.root.findall(f'{omexml.ome_schema}Image')
    return [image.attrib.get('Name', '') for image in images]


def read_raw_metadata(
        path: str, *, lazy_load: bool = True,
        reader_factory: Callable | None = None
    ) -> dict:
    """Read the metadata of the raw file `path` (see the module docstring).

    '.ome.tif' (case-insensitive): `load.OMEXML(path)`, as the GUI;
    metadata_xml is `str(omexml_string)` (the text 'None' for a file without
    OME-XML, as main writes it) and scene_names the OME <Image> names.

    Otherwise ONE `ImageReader(path, lazy_load=lazy_load,
    bioimage_factory=reader_factory)` is opened and closed again;
    the expressions are evaluated on `reader.OMEXML().init_from_bioimage`,
    metadata_xml is `str(bioimage.metadata)` and size_y, size_x and dtype
    come from scene 0. A physical size the reader does not report is stored
    as 1.0: on main the GUI received these values through the text round
    trip of `OMEXML.init_from_file`, which turned them into 1.0 silently,
    so storing None would add 'not found' warnings that main never showed.

    Exceptions propagate (the calling task adds the path).
    """
    if path.lower().endswith('.ome.tif'):
        omexml = load.OMEXML(path)
        raw = extract_from_metadata_object(
            omexml, path=path, source='ome_tif',
            reader=_qualified_name(omexml),
            metadata_xml=str(omexml.omexml_string)
        )
        raw['scene_names'] = _ome_scene_names(omexml)
        return raw

    image_reader = ImageReader(
        path, qparent=None, lazy_load=lazy_load,
        bioimage_factory=reader_factory
    )
    try:
        bioimage = image_reader._bioioimage
        omexml = OMEXML().init_from_bioimage(path, image_reader)
        raw = extract_from_metadata_object(
            omexml, path=path, source='bioio',
            reader=_qualified_name(getattr(bioimage, 'reader', bioimage)),
            metadata_xml=str(bioimage.metadata)
        )
        for key in PHYSICAL_SIZE_KEYS:
            if raw[key] is None:
                logger.debug('%s: %s not reported, using 1.0', path, key)
                raw[key] = 1.0
        _, _, _, size_y, size_x = image_reader.dims(0)
        raw['size_y'] = size_y
        raw['size_x'] = size_x
        raw['dtype'] = str(bioimage.dtype)
    finally:
        image_reader.close()
    return raw
