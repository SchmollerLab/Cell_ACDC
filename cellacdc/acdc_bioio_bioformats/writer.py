"""Write the image files of one Position from raw microscopy files.

`save_position_images` validates the request against the real files, then
calls the legacy pixel writer `_utils.saveImgDataChannel` once per channel,
so the files written are the ones the GUI has always written.
`check_position_images` runs the same validation without writing.

This module runs where raw files may be opened (worker children, tests); the
parent process never imports it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal, Mapping

from . import _utils
from .reader import ImageReader

logger = logging.getLogger(__name__)

FORMATS = ('tif', 'h5', 'symlink')


@dataclass(frozen=True)
class ChannelImageSpec:
    raw_path: str
    file_channel_index: int
    name: str


@dataclass(frozen=True)
class PositionImagesRequest:
    series: int
    images_path: str
    filename_no_ext: str                 # legacy -p (dots replaced by _utils)
    s0p: str
    channels: tuple[ChannelImageSpec, ...]
    time_range: tuple[int, int]          # 1-based inclusive
    size_z: int | None                   # None -> scene-0 Z of each raw file
    time_increment_s: float
    physical_size_zyx_um: tuple[float, float, float]
    fmt: Literal['tif', 'h5', 'symlink']
    lazy_load: bool
    image_name: str = ''


def _get_reader(raw_path: str, lazy_load: bool, reader_cache: dict,
                reader_factory: Callable | None) -> ImageReader:
    key = (raw_path, lazy_load)
    image_reader = reader_cache.get(key)
    if image_reader is None:
        image_reader = ImageReader(
            raw_path, qparent=None, lazy_load=lazy_load,
            bioimage_factory=reader_factory
        )
        reader_cache[key] = image_reader
    return image_reader


def _check_request(req: PositionImagesRequest,
                   readers: Mapping[str, ImageReader]
                   ) -> dict[str, tuple[int, int]]:
    """Check `req` against every raw file; return raw_path -> (SizeT of the
    series, number of z-slices to save). ValueError on a violation."""
    t0, t1 = req.time_range
    sizes = {}
    for raw_path, image_reader in readers.items():
        where = f'"{raw_path}" (series {req.series})'
        size_t, size_c, size_z, _, _ = image_reader.dims(req.series)
        if not 1 <= t0 <= t1 <= size_t:
            raise ValueError(
                f'Time range {t0}-{t1} (1-based, inclusive) does not fit the '
                f'{size_t} frame(s) of {where}.'
            )
        if req.size_z is not None:
            z_to_save = req.size_z
        elif req.series == 0:
            z_to_save = size_z
        else:
            z_to_save = image_reader.dims(0)[2]
        if not 1 <= z_to_save <= size_z:
            raise ValueError(
                f'Cannot save {z_to_save} z-slice(s): {where} has {size_z}.'
            )
        for channel in req.channels:
            index = channel.file_channel_index
            if channel.raw_path == raw_path and not 0 <= index < size_c:
                raise ValueError(
                    f'Channel index {index} ("{channel.name}") is outside '
                    f'the {size_c} channel(s) of {where}.'
                )
        sizes[raw_path] = (size_t, z_to_save)
    return sizes


def _check_format(req: PositionImagesRequest) -> None:
    if req.fmt not in FORMATS:
        raise ValueError(
            f'Unknown image format {req.fmt!r} (expected one of {FORMATS}).'
        )


def _open_readers(req: PositionImagesRequest, reader_cache: dict,
                  reader_factory: Callable | None) -> dict[str, ImageReader]:
    return {
        channel.raw_path: _get_reader(
            channel.raw_path, req.lazy_load, reader_cache, reader_factory
        )
        for channel in req.channels
    }


def _close_readers(reader_cache: dict) -> None:
    for image_reader in reader_cache.values():
        image_reader.close()


def check_position_images(
        req: PositionImagesRequest, *, reader_factory: Callable | None = None,
        reader_cache: dict | None = None
    ) -> None:
    """Check `req` against the raw files without writing anything.

    Same checks and ValueError messages as `save_position_images`, so a
    caller can validate every Position of a job before it clears or creates
    any folder. Readers are opened as in `save_position_images` (kept in
    `reader_cache` when given, otherwise closed before returning).
    """
    _check_format(req)
    owns_cache = reader_cache is None
    if owns_cache:
        reader_cache = {}
    try:
        _check_request(req, _open_readers(req, reader_cache, reader_factory))
    finally:
        if owns_cache:
            _close_readers(reader_cache)


def _output_path(req: PositionImagesRequest, channel_name: str) -> str:
    if req.fmt == 'symlink':
        append_txt, ext = 'symlink', '.ini'
    else:
        append_txt, ext = channel_name, f'.{req.fmt}'
    filename = _utils.getFilename(
        req.filename_no_ext, req.s0p, append_txt, req.series, ext,
        image_name=req.image_name
    )
    return os.path.join(req.images_path, filename)


def save_position_images(
        req: PositionImagesRequest, *, reader_factory: Callable | None = None,
        reader_cache: dict | None = None,
        on_channel_done: Callable[[str], None] | None = None
    ) -> list[str]:
    """Write the image files of one Position and return their paths.

    Each distinct (raw_path, lazy_load) is opened once; with `reader_cache`
    the open readers are shared across calls (one job) and the caller closes
    them, otherwise they are closed before returning. Before the first
    channel is written, the request is checked against
    `ImageReader.dims(series)` (T, C, Z, Y, X) of every raw file:
    1 <= t0 <= t1 <= T, 1 <= size_z <= Z (size_z None: the file's scene-0 Z)
    and 0 <= file_channel_index < C; a violation raises ValueError naming
    the file and the series. Then `_utils.saveImgDataChannel` writes each
    channel and `on_channel_done(name)` is called. Returned paths: one
    .tif/.h5 per channel, or the single symlink .ini.
    """
    _check_format(req)
    owns_cache = reader_cache is None
    if owns_cache:
        reader_cache = {}
    try:
        readers = _open_readers(req, reader_cache, reader_factory)
        sizes = _check_request(req, readers)
        physical_size_z, physical_size_y, physical_size_x = (
            req.physical_size_zyx_um
        )
        written = []
        for channel in req.channels:
            size_t, size_z = sizes[channel.raw_path]
            logger.debug(
                'Saving channel "%s" (index %d of "%s", series %d)',
                channel.name, channel.file_channel_index, channel.raw_path,
                req.series
            )
            _utils.saveImgDataChannel(
                readers[channel.raw_path], req.series, req.images_path,
                req.filename_no_ext, req.s0p, channel.name,
                channel.file_channel_index, {}, size_t, size_z,
                req.time_increment_s, physical_size_z, physical_size_y,
                physical_size_x, req.fmt == 'h5', req.time_range,
                req.fmt == 'symlink', channel.raw_path, req.lazy_load,
                image_name=req.image_name
            )
            path = _output_path(req, channel.name)
            if path not in written:
                written.append(path)
            if on_channel_done is not None:
                on_channel_done(channel.name)
        return written
    finally:
        if owns_cache:
            _close_readers(reader_cache)
