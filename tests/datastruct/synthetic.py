"""Deterministic synthetic raw microscopy files for the data-structure tests.

Every pixel encodes where it came from::

    value = 1000*s + 100*t + 10*c + z          (decode_pixel(value) -> (s, t, c, z))

`s` is a "source" id chosen by the generator: the scene (series) index for
multi-series files, the 0-based file index (= input id - 1) for
one-file-per-position sets and the position id for one-file-per-channel sets.
`t`, `c` and `z` are 0-based indices (plus offsets), so every index must stay
below 10 (`MAX_INDEX_SIZE`). Planes are constant, so any pixel of a plane
identifies the plane.

All generators are parametric and only need numpy + tifffile. Files are
byte-identical between runs in the same environment (fixed OME UUID and
creator, no timestamps).
"""
from __future__ import annotations

import os
import re
from typing import Callable, Collection, Sequence

import numpy as np
import tifffile

MAX_INDEX_SIZE = 10
OME_UUID = 'urn:uuid:00000000-0000-4000-8000-00000000acdc'
OME_CREATOR = 'cellacdc-tests'
IMAGEJ_AXES = 'TZCYX'


class _Default:
    def __repr__(self) -> str:
        return 'DEFAULT'


DEFAULT = _Default()
"""Sentinel for "use the documented default" (None means "omit")."""


def encode_pixel(s: int, t: int, c: int, z: int) -> int:
    """Pixel value of plane (s, t, c, z)."""
    for name, value in (('t', t), ('c', c), ('z', z)):
        if not 0 <= value < MAX_INDEX_SIZE:
            raise ValueError(
                f'{name}={value} cannot be encoded (0 <= {name} < '
                f'{MAX_INDEX_SIZE})'
            )
    if s < 0:
        raise ValueError(f's={s} cannot be encoded (s >= 0)')
    return 1000*s + 100*t + 10*c + z


def decode_pixel(value) -> tuple[int, int, int, int]:
    """Inverse of `encode_pixel`: returns (s, t, c, z)."""
    v = int(value)
    if v < 0 or v != value:
        raise ValueError(f'{value!r} is not an encoded pixel value')
    return v // 1000, (v // 100) % 10, (v // 10) % 10, v % 10


def encoded_array(
        axes: str = 'TCZYX', *, T: int = 1, C: int = 1, Z: int = 1,
        Y: int = 16, X: int = 20, s: int = 0, t0: int = 0, c0: int = 0,
        z0: int = 0, dtype: str = 'uint16'
    ) -> np.ndarray:
    """Encoded array with the given axis order.

    `axes` is any order of a subset of 'TCZYX' that contains Y and X.
    Index i along T encodes t = t0 + i (same for C/c0 and Z/z0); an axis that
    is absent encodes its offset. Raises ValueError when an index cannot be
    encoded or the values do not fit `dtype`.
    """
    axes = axes.upper()
    if (
        len(set(axes)) != len(axes) or not set(axes) <= set('TCZYX')
        or 'Y' not in axes or 'X' not in axes
    ):
        raise ValueError(f'invalid axes {axes!r} (subset of TCZYX with Y, X)')
    sizes = {'T': T, 'C': C, 'Z': Z, 'Y': Y, 'X': X}
    offsets = {'T': t0, 'C': c0, 'Z': z0}
    weights = {'T': 100, 'C': 10, 'Z': 1}
    for key in 'TCZ':
        last = offsets[key] + (sizes[key] - 1 if key in axes else 0)
        if sizes[key] < 1 or offsets[key] < 0 or last >= MAX_INDEX_SIZE:
            raise ValueError(
                f'{key}={sizes[key]} with offset {offsets[key]} cannot be '
                f'encoded (indices must stay below {MAX_INDEX_SIZE})'
            )
    shape = tuple(sizes[key] for key in axes)
    values = np.full(shape, encode_pixel(s, t0, c0, z0), dtype=np.int64)
    for i, key in enumerate(axes):
        if key not in weights:
            continue
        ramp_shape = [1]*len(axes)
        ramp_shape[i] = sizes[key]
        ramp = np.arange(sizes[key], dtype=np.int64).reshape(ramp_shape)
        values = values + weights[key]*ramp
    out_dtype = np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        if values.max() > np.iinfo(out_dtype).max:
            raise ValueError(
                f'max encoded value {values.max()} does not fit {out_dtype}'
            )
    return values.astype(out_dtype)


def _check_dir(path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)


def make_imagej_tif(
        path: str, *, T: int = 3, C: int = 2, Z: int = 2, Y: int = 16,
        X: int = 20, dtype: str = 'uint16', s: int = 0,
        spacing: float | None = None, finterval: float | None = None,
        unit: str | None = None, physical_size_xy: float | None = None
    ) -> str:
    """ImageJ hyperstack with logical TCZYX content (stored as ImageJ TZCYX).

    tifffile reports `series[0].axes == 'TZCYX'` with singleton axes dropped
    (e.g. C=1 -> 'TZYX'). Optional ImageJ calibration: `spacing` (Z step),
    `finterval` (s), `unit`, and `physical_size_xy` written as
    resolution 1/physical_size_xy pixels per unit. Returns `path`.
    """
    _check_dir(path)
    data = encoded_array(
        IMAGEJ_AXES, T=T, C=C, Z=Z, Y=Y, X=X, s=s, dtype=dtype
    )
    metadata = {'axes': IMAGEJ_AXES}
    for key, value in (
            ('spacing', spacing), ('finterval', finterval), ('unit', unit)):
        if value is not None:
            metadata[key] = value
    kwargs = {}
    if physical_size_xy is not None:
        kwargs['resolution'] = (1/physical_size_xy, 1/physical_size_xy)
    tifffile.imwrite(path, data, imagej=True, metadata=metadata, **kwargs)
    return path


def _resolve(value, default):
    return default if value is DEFAULT else value


def _xml_remove_image_names(xml: str) -> str:
    return re.sub(r'(<Image ID="[^"]*") Name="[^"]*"', r'\1', xml)


def _xml_add_lens_na(xml: str, lens_na: float) -> str:
    instrument = (
        '<Instrument ID="Instrument:0">'
        f'<Objective ID="Objective:0" LensNA="{lens_na}"/>'
        '</Instrument>'
    )
    return xml.replace('<Image ID=', instrument + '<Image ID=', 1)


def read_ome_xml(path: str) -> str | None:
    """OME-XML text of a TIFF (None if the file has none), as tifffile reads it."""
    with tifffile.TiffFile(path) as tif:
        return tif.ome_metadata


def rewrite_ome_xml(path: str, transform: Callable[[str], str]) -> str:
    """Replace the OME-XML of `path` by `transform(old_xml)` (in place)."""
    xml = read_ome_xml(path)
    if xml is None:
        raise ValueError(f'{path} has no OME-XML')
    tifffile.tiffcomment(path, transform(xml).encode('utf-8'))
    return path


def make_ome_tif(
        path: str, *, series: int = 3, T: int = 3, C: int = 2, Z: int = 2,
        Y: int = 16, X: int = 20, dtype: str = 'uint16',
        names=DEFAULT, em_wavelens=DEFAULT, physical_sizes=DEFAULT,
        physical_size_unit: str = '\u00b5m', time_increment=DEFAULT,
        time_increment_unit: str = 's', image_names=DEFAULT,
        lens_na: float | None = None, s_offset: int = 0,
        xml_transform: Callable[[str], str] | None = None
    ) -> str:
    """Multi-series OME-TIFF (one OME <Image> per series, axes TCZYX).

    Series k encodes s = s_offset + k. Metadata parameters (DEFAULT = the
    documented default, None = attribute omitted):

    - names: channel Name per channel; default 'GFP', 'mCherry', then 'ch<c>'.
    - em_wavelens: EmissionWavelength per channel; default 510.0 + 100*c.
    - physical_sizes: (Z, Y, X) in `physical_size_unit`; default
      (0.5, 0.1, 0.1).
    - time_increment: TimeIncrement in `time_increment_unit`; default 7.5.
    - image_names: Image@Name per series; default 'Scene #<k+1>'. None
      removes the attribute (tifffile would write 'Image<k>').
    - lens_na: adds <Instrument><Objective LensNA=.../> (default: absent).
    - xml_transform: final str -> str edit of the OME-XML (any variant).

    Returns `path`.
    """
    _check_dir(path)
    default_names = ('GFP', 'mCherry') + tuple(f'ch{c}' for c in range(2, C))
    names = _resolve(names, default_names[:C])
    em_wavelens = _resolve(
        em_wavelens, tuple(510.0 + 100*c for c in range(C))
    )
    physical_sizes = _resolve(physical_sizes, (0.5, 0.1, 0.1))
    time_increment = _resolve(time_increment, 7.5)
    remove_names = image_names is None
    image_names = _resolve(
        image_names, tuple(f'Scene #{k + 1}' for k in range(series))
    )
    for label, values, size in (
            ('names', names, C), ('em_wavelens', em_wavelens, C),
            ('image_names', image_names, series)):
        if values is not None and len(values) != size:
            raise ValueError(f'len({label}) must be {size}, got {values!r}')

    with tifffile.TiffWriter(path, ome=True) as writer:
        for k in range(series):
            metadata = {'axes': 'TCZYX'}
            if k == 0:
                metadata['UUID'] = OME_UUID
                metadata['Creator'] = OME_CREATOR
            if image_names is not None:
                metadata['Name'] = image_names[k]
            if physical_sizes is not None:
                for axis, value in zip('ZYX', physical_sizes):
                    metadata[f'PhysicalSize{axis}'] = value
                    metadata[f'PhysicalSize{axis}Unit'] = physical_size_unit
            if time_increment is not None:
                metadata['TimeIncrement'] = time_increment
                metadata['TimeIncrementUnit'] = time_increment_unit
            channel = {}
            if names is not None:
                channel['Name'] = list(names)
            if em_wavelens is not None:
                channel['EmissionWavelength'] = list(em_wavelens)
            if channel:
                metadata['Channel'] = channel
            data = encoded_array(
                'TCZYX', T=T, C=C, Z=Z, Y=Y, X=X, s=s_offset + k, dtype=dtype
            )
            writer.write(data, metadata=metadata, photometric='minisblack')

    transforms = []
    if remove_names:
        transforms.append(_xml_remove_image_names)
    if lens_na is not None:
        transforms.append(lambda xml: _xml_add_lens_na(xml, lens_na))
    if xml_transform is not None:
        transforms.append(xml_transform)
    if transforms:
        def _apply(xml: str) -> str:
            for transform in transforms:
                xml = transform(xml)
            return xml
        rewrite_ome_xml(path, _apply)
    return path


def _is_ome_ext(filename: str) -> bool:
    return filename.lower().endswith(('.ome.tif', '.ome.tiff'))


def make_one_per_pos_set(
        folder: str, *,
        filenames: Sequence[str] = ('pos1.tif', 'pos2.tif', 'pos3.tif'),
        overrides: dict[str, dict] | None = None, **kwargs
    ) -> list[str]:
    """One file per position (Module 0 layout 1), file i encodes s = i.

    '.ome.tif(f)' names use `make_ome_tif(path, series=1, s_offset=i, **kw)`,
    all others `make_imagej_tif(path, s=i, **kw)`, so `kwargs` must suit the
    generator. `overrides[filename]` updates the keyword arguments of that
    file (e.g. {'pos3.tif': {'T': 5}}). Returns the file names in the given
    order.
    """
    overrides = overrides or {}
    for i, filename in enumerate(filenames):
        path = os.path.join(folder, filename)
        file_kwargs = {**kwargs, **overrides.get(filename, {})}
        if _is_ome_ext(filename):
            file_kwargs.setdefault('series', 1)
            make_ome_tif(path, s_offset=i, **file_kwargs)
        else:
            make_imagej_tif(path, s=i, **file_kwargs)
    return list(filenames)


def make_one_per_channel_set(
        folder: str, *, basename: str = 'ASY015_',
        positions: Sequence[int] = (1, 2),
        channels: Sequence[str] = ('GFP', 'mNeon-Green'), pad: int = 0,
        ext: str = '.tif', T: int = 3, Z: int = 2, Y: int = 16, X: int = 20,
        dtype: str = 'uint16', skip: Collection[tuple[int, str]] = ()
    ) -> list[str]:
    """One file per channel (Module 0 layout 2).

    File name: f'{basename}{token}_{channel}{ext}' with token =
    str(position).zfill(pad) (pad=0: no padding); `basename` is used
    verbatim (include the separator, e.g. 'ASY015_' or 'exp_'). Each file
    holds one channel (C=1) of shape T x Z x Y x X and encodes s = position
    and c = index of the channel in `channels`. '.ome.tif(f)' files are
    OME-TIFF (channel Name = token), other extensions ImageJ. `skip` lists
    (position, channel) pairs to leave out. Returns the file names in
    creation order (position-major).
    """
    created = []
    for position in positions:
        token = str(position).zfill(pad) if pad else str(position)
        for c, channel in enumerate(channels):
            if (position, channel) in skip:
                continue
            filename = f'{basename}{token}_{channel}{ext}'
            path = os.path.join(folder, filename)
            _check_dir(path)
            data = encoded_array(
                'TZYX', T=T, Z=Z, Y=Y, X=X, s=position, c0=c, dtype=dtype
            )
            if _is_ome_ext(filename):
                tifffile.imwrite(
                    path, data, ome=True, photometric='minisblack',
                    metadata={
                        'axes': 'TZYX', 'UUID': OME_UUID,
                        'Creator': OME_CREATOR, 'Channel': {'Name': [channel]}
                    }
                )
            else:
                tifffile.imwrite(
                    path, data, imagej=True, metadata={'axes': 'TZYX'}
                )
            created.append(filename)
    return created


def make_truncated_tif(
        path: str, *, keep_fraction: float = 0.5, keep_bytes: int | None = None,
        **imagej_kwargs
    ) -> str:
    """A `make_imagej_tif` file cut to `keep_bytes` (default: keep_fraction of
    its size). Small `keep_bytes` breaks the header; around 0.5 the header
    reads but the pixel data is incomplete. Returns `path`."""
    make_imagej_tif(path, **imagej_kwargs)
    size = os.path.getsize(path)
    keep = int(size*keep_fraction) if keep_bytes is None else keep_bytes
    if not 0 <= keep < size:
        raise ValueError(f'cannot truncate {size} bytes to {keep}')
    with open(path, 'r+b') as file:
        file.truncate(keep)
    return path


def make_loose_tiff_set(
        folder: str, *, pattern: str, positions: Sequence = (None,),
        channels: Sequence = (None,), timepoints: Sequence = (None,),
        z_slices: Sequence = (None,), file_axes: str = 'YX', T: int = 1,
        C: int = 1, Z: int = 1, Y: int = 4, X: int = 5,
        dtype: str = 'uint16', skip: Collection[tuple] = ()
    ) -> list[str]:
    """Loose TIFF files for Re-structure (plain tifffile, series[0] shape ==
    array shape in `file_axes` order, e.g. 'YX', 'ZYX', 'CTYX').

    One file per combination of positions x channels x timepoints x z_slices;
    the name is pattern.format(pos=..., ch=..., t=..., z=...) (None entries
    mean "not in the name"). Encoding: s = index of the position in
    `positions`, c = index of the channel in `channels` + in-file channel
    index, t = file timepoint (int, None -> 0) + in-file frame index,
    z = file z (int, None -> 0) + in-file slice index. In-file sizes come
    from T/C/Z (only for axes present in `file_axes`). `skip` lists
    (pos, ch, t, z) tuples to leave out. Returns file names in creation
    order.
    """
    created = []
    for s, pos in enumerate(positions):
        for c, ch in enumerate(channels):
            for t in timepoints:
                for z in z_slices:
                    if (pos, ch, t, z) in skip:
                        continue
                    filename = pattern.format(pos=pos, ch=ch, t=t, z=z)
                    path = os.path.join(folder, filename)
                    _check_dir(path)
                    data = encoded_array(
                        file_axes, T=T, C=C, Z=Z, Y=Y, X=X, s=s,
                        t0=t or 0, c0=c, z0=z or 0, dtype=dtype
                    )
                    tifffile.imwrite(path, data)
                    created.append(filename)
    return created
