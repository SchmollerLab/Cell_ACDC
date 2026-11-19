"""Output-tree manifests: a JSON-serializable, OS-independent description of a
folder tree, and a structural diff between two manifests.

`build_manifest(root, placeholders=...)` returns a dict keyed by the POSIX
relative path of every file (and of every empty folder; '.' is the root when
it is empty or missing), plus the informational ENV_KEY block. Entries:

- tif/tiff: kind, shape, dtype, sha256 of the decoded array (series 0),
  series axes, ImageJ keys (spacing/finterval/unit/frames/slices/channels/
  images/hyperstack), resolution ((x, y) rationals + unit), selected OME
  attributes, `other_series` (axes, shape, dtype, sha256, origins of series
  1..n) and `origins`: synthetic.decode_pixel of the first pixel of plane 0
  of every frame (one entry when there is no T axis).
- h5: every dataset (name -> shape, dtype, chunks, sha256, origins along
  axis 0).
- npy/npz: arrays (shape, dtype, sha256).
- csv: ordered rows as strings (line endings do not matter).
- ini: sections in file order, values with placeholder-normalized paths.
- json: parsed content with placeholder-normalized path strings.
- *metadataXML.txt: is_literal_None, nonempty (the text is informational).
- anything else: sha256.

Keys named INFO_KEY ('_info') and the top-level ENV_KEY ('@env') are
informational: `diff_manifests` ignores them. Unreadable files get an
`error` field instead of crashing the manifest.
"""
from __future__ import annotations

import configparser
import csv
import fnmatch
import hashlib
import io
import json
import os
import platform
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Iterable, Mapping

import numpy as np
import tifffile

from . import synthetic

ENV_KEY = '@env'
INFO_KEY = '_info'
ENV_PACKAGES = (
    'numpy', 'tifffile', 'h5py', 'pandas', 'bioio', 'bioio-base',
    'bioio-tifffile', 'bioio-czi', 'bioio-nd2', 'bioio-bioformats',
)
IMAGEJ_KEYS = (
    'images', 'channels', 'slices', 'frames', 'hyperstack', 'unit',
    'spacing', 'finterval',
)
OME_PIXELS_KEYS = (
    'DimensionOrder', 'Type', 'SizeX', 'SizeY', 'SizeZ', 'SizeC', 'SizeT',
    'PhysicalSizeX', 'PhysicalSizeXUnit', 'PhysicalSizeY',
    'PhysicalSizeYUnit', 'PhysicalSizeZ', 'PhysicalSizeZUnit',
    'TimeIncrement', 'TimeIncrementUnit',
)


class _Absent:
    def __repr__(self) -> str:
        return '<absent>'


ABSENT = _Absent()
"""`Diff.old`/`Diff.new` when the key exists on one side only."""


@dataclass(frozen=True)
class Diff:
    key: str            # '/'-joined path: '<relpath>/<field>[/<sub>...]'
    old: object
    new: object

    def __str__(self) -> str:
        return f'{self.key}: {self.old!r} -> {self.new!r}'


# ----------------------------------------------------------------- utilities
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """sha256 of the C-ordered bytes of `array` (shape/dtype stored apart)."""
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def _jsonable(value):
    """Convert numpy scalars/arrays and tuples to JSON-native values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def _decode_origin(value):
    if isinstance(value, (np.floating, float)):
        if not float(value).is_integer():
            return float(value)
    try:
        return list(synthetic.decode_pixel(value))
    except (ValueError, TypeError, OverflowError):
        return _jsonable(value)


def _path_variants(root: str) -> list[str]:
    variants = set()
    for base in (root, os.path.abspath(root), os.path.realpath(root)):
        base = base.rstrip('\\/')
        variants.update({base, base.replace('\\', '/'),
                         base.replace('/', '\\')})
    return sorted(variants, key=len, reverse=True)


def normalize_path_text(text: str,
                        placeholders: Mapping[str, str] | None) -> str:
    """Replace every occurrence of a placeholder folder in `text` by its
    token (e.g. {'<SRC>': src, '<DST>': dst}) and use '/' after the token.

    Matching is case-insensitive on Windows and accepts '/' or '\\'
    separators, abspath and realpath spellings. A value that becomes
    '<TOKEN>...' entirely has all its backslashes turned into '/'."""
    if not placeholders or not isinstance(text, str):
        return text
    flags = re.IGNORECASE if os.name == 'nt' else 0
    pairs = []
    for token, root in placeholders.items():
        pairs.extend((variant, token) for variant in _path_variants(root))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    for variant, token in pairs:
        pattern = re.escape(variant) + r'(?=[\\/]|$|[^\w.\-])'
        text = re.sub(pattern, lambda _m, t=token: t, text, flags=flags)
    tokens = sorted(placeholders, key=len, reverse=True)
    if any(text.startswith(token) for token in tokens):
        return text.replace('\\', '/')
    for token in tokens:
        text = re.sub(
            re.escape(token) + r'[^\s"\'<>|,;]*',
            lambda m: m.group(0).replace('\\', '/'), text
        )
    return text


def _normalize_nested(value, placeholders):
    if isinstance(value, str):
        return normalize_path_text(value, placeholders)
    if isinstance(value, dict):
        return {k: _normalize_nested(v, placeholders) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_nested(v, placeholders) for v in value]
    return value


# ------------------------------------------------------------ entry builders
def _frame_origins(array: np.ndarray, axes: str) -> list:
    if array.size == 0:
        return []
    if 'T' not in axes or len(axes) != array.ndim:
        return [_decode_origin(array.flat[0])]
    t_axis = axes.index('T')
    origins = []
    for t in range(array.shape[t_axis]):
        frame = np.take(array, t, axis=t_axis)
        origins.append(_decode_origin(frame.flat[0]))
    return origins


def _ome_summary(xml: str | None) -> dict | None:
    if not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {'parse_error': True}
    images = []
    for image in root.iter():
        if not image.tag.endswith('}Image'):
            continue
        pixels = next(
            (child for child in image if child.tag.endswith('}Pixels')), None
        )
        pixels_attrs = {}
        channels = []
        if pixels is not None:
            pixels_attrs = {
                key: pixels.attrib[key] for key in OME_PIXELS_KEYS
                if key in pixels.attrib
            }
            for channel in pixels:
                if channel.tag.endswith('}Channel'):
                    channels.append({
                        key: channel.attrib[key]
                        for key in ('Name', 'EmissionWavelength')
                        if key in channel.attrib
                    })
        images.append({
            'Name': image.attrib.get('Name'),
            'Pixels': pixels_attrs,
            'Channels': channels,
        })
    objectives = [
        element.attrib.get('LensNA') for element in root.iter()
        if element.tag.endswith('}Objective')
    ]
    return {'images': images, 'LensNA': objectives}


def _tif_entry(path: str) -> dict:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        array = series.asarray()
        page = tif.pages[0]
        imagej = tif.imagej_metadata
        resolution = None
        tags = page.tags
        if 'XResolution' in tags and 'YResolution' in tags:
            unit = tags['ResolutionUnit'].value if 'ResolutionUnit' in tags \
                else None
            resolution = [
                list(tags['XResolution'].value),
                list(tags['YResolution'].value),
                int(unit) if unit is not None else None,
            ]
        other_series = []
        for other in tif.series[1:]:
            other_array = other.asarray()
            other_series.append({
                'axes': other.axes, **_array_entry(other_array),
                'origins': _frame_origins(other_array, other.axes),
            })
        entry = {
            'kind': 'tif',
            'shape': list(array.shape),
            'dtype': str(array.dtype),
            'sha256': array_sha256(array),
            'axes': series.axes,
            'other_series': other_series,
            'imagej': None if imagej is None else {
                key: imagej.get(key) for key in IMAGEJ_KEYS
            },
            'resolution': resolution,
            'ome': _ome_summary(tif.ome_metadata) if tif.is_ome else None,
            'origins': _frame_origins(array, series.axes),
        }
    return _jsonable(entry)


def _array_entry(array: np.ndarray) -> dict:
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'sha256': array_sha256(array),
    }


def _h5_entry(path: str) -> dict:
    import h5py

    datasets = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            array = obj[()]
            entry = _array_entry(np.asarray(array))
            entry['chunks'] = list(obj.chunks) if obj.chunks else None
            axes = 'T' + 'Q'*(array.ndim - 1) if array.ndim > 2 else 'YX'
            entry['origins'] = _frame_origins(np.asarray(array), axes)
            datasets[name] = entry

    with h5py.File(path, 'r') as h5:
        h5.visititems(visit)
    return _jsonable({'kind': 'h5', 'datasets': datasets})


def _npy_entry(path: str) -> dict:
    if path.lower().endswith('.npz'):
        with np.load(path, allow_pickle=False) as npz:
            arrays = {name: _array_entry(npz[name]) for name in npz.files}
        return _jsonable({'kind': 'npz', 'arrays': arrays})
    array = np.load(path, allow_pickle=False)
    return _jsonable({'kind': 'npy', **_array_entry(array)})


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        return f.read()


def _csv_entry(path: str, placeholders) -> dict:
    rows = list(csv.reader(io.StringIO(_read_text(path))))
    return {'kind': 'csv', 'rows': _normalize_nested(rows, placeholders)}


def _ini_entry(path: str, placeholders) -> dict:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(_read_text(path))
    sections = {
        name: {
            key: normalize_path_text(value, placeholders)
            for key, value in parser[name].items()
        }
        for name in parser.sections()
    }
    return {
        'kind': 'ini', 'section_order': parser.sections(),
        'sections': sections,
    }


def _json_entry(path: str, placeholders) -> dict:
    content = json.loads(_read_text(path))
    return {'kind': 'json', 'content': _normalize_nested(content, placeholders)}


def _metadata_xml_entry(path: str) -> dict:
    text = _read_text(path)
    return {
        'kind': 'metadataXML',
        'is_literal_None': text == 'None',
        'nonempty': bool(text.strip()),
        INFO_KEY: {'length': len(text), 'sha256': sha256_bytes(
            text.encode('utf-8'))},
    }


def manifest_entry(path: str, *,
                   placeholders: Mapping[str, str] | None = None) -> dict:
    """Manifest entry of one file (see the module docstring)."""
    name = os.path.basename(path).lower()
    if os.path.islink(path):
        return {
            'kind': 'symlink',
            'target': normalize_path_text(os.readlink(path), placeholders),
        }
    try:
        if name.endswith(('.tif', '.tiff')):
            return _tif_entry(path)
        if name.endswith('.h5'):
            return _h5_entry(path)
        if name.endswith(('.npy', '.npz')):
            return _npy_entry(path)
        if name.endswith('.csv'):
            return _csv_entry(path, placeholders)
        if name.endswith('.ini'):
            return _ini_entry(path, placeholders)
        if name.endswith('.json'):
            return _json_entry(path, placeholders)
        if name.endswith('metadataxml.txt'):
            return _metadata_xml_entry(path)
    except Exception as err:
        return {
            'kind': 'unreadable', 'error': type(err).__name__,
            'sha256': sha256_file(path), INFO_KEY: {'message': str(err)},
        }
    return {'kind': 'file', 'sha256': sha256_file(path)}


def env_info() -> dict:
    """Python/platform and package versions (informational)."""
    versions = {}
    for package in ENV_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': versions,
    }


def build_manifest(root: str, *,
                   placeholders: Mapping[str, str] | None = None,
                   exclude: Iterable[str] = (),
                   include_env: bool = True) -> dict:
    """Manifest of the tree under `root` (see the module docstring).

    `placeholders` maps tokens to folders ({'<SRC>': src, '<DST>': dst}) for
    paths found inside files; `exclude` holds fnmatch patterns matched
    against POSIX relative paths (files and folders)."""
    manifest = {}
    exclude = tuple(exclude)

    def excluded(relpath: str) -> bool:
        return any(fnmatch.fnmatch(relpath, pattern) for pattern in exclude)

    if not os.path.isdir(root):
        manifest['.'] = {'kind': 'missing'}
    else:
        entries = {}
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root).replace(os.sep, '/')
            prefix = '' if rel_dir == '.' else rel_dir + '/'
            dirnames[:] = sorted(
                d for d in dirnames if not excluded(prefix + d)
            )
            linked = [
                d for d in dirnames if os.path.islink(os.path.join(dirpath, d))
            ]
            dirnames[:] = [d for d in dirnames if d not in linked]
            kept = [f for f in filenames if not excluded(prefix + f)] + linked
            if not dirnames and not kept:
                entries[rel_dir] = {'kind': 'empty_dir'}
            for filename in kept:
                entries[prefix + filename] = manifest_entry(
                    os.path.join(dirpath, filename),
                    placeholders=placeholders
                )
        manifest.update(sorted(entries.items()))
    if include_env:
        manifest[ENV_KEY] = env_info()
    return manifest


# ------------------------------------------------------------------- diffing
def _diff(old, new, key: str, out: list[Diff]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for name in sorted(set(old) | set(new), key=str):
            if name in (INFO_KEY, ENV_KEY):
                continue
            sub_key = f'{key}/{name}' if key else str(name)
            _diff(old.get(name, ABSENT), new.get(name, ABSENT), sub_key, out)
        return
    if (
        isinstance(old, (list, tuple)) and isinstance(new, (list, tuple))
        and len(old) == len(new)
    ):
        if list(old) == list(new):
            return
        for index, (a, b) in enumerate(zip(old, new)):
            _diff(a, b, f'{key}/{index}', out)
        return
    if old is ABSENT and new is ABSENT:
        return
    if old != new or type(old) is not type(new) and (
            isinstance(old, bool) or isinstance(new, bool)):
        out.append(Diff(key, old, new))


def diff_manifests(a: Mapping, b: Mapping) -> list[Diff]:
    """Structural differences from `a` (old/golden) to `b` (new/actual).

    Recurses into dicts (key path joined with '/') and equal-length lists
    (index as key); lists of different length differ as a whole. Missing keys
    give ABSENT. INFO_KEY and ENV_KEY entries are ignored."""
    out: list[Diff] = []
    _diff(dict(a), dict(b), '', out)
    return out


def format_diffs(diffs: Iterable[Diff], limit: int = 50) -> str:
    """Readable multi-line listing for assertion messages."""
    diffs = list(diffs)
    lines = [str(diff) for diff in diffs[:limit]]
    if len(diffs) > limit:
        lines.append(f'... and {len(diffs) - limit} more')
    return '\n'.join(lines)


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _encode(value, level: int) -> str:
    """JSON text of ``value`` as ``json.dumps(indent=1, sort_keys=True)``
    gives it, except that a list of scalars that fits in 79 columns (with
    its indent) goes on one line."""
    if isinstance(value, dict):
        if not value:
            return '{}'
        pad = ' ' * (level + 1)
        items = [
            f'{pad}{json.dumps(key, ensure_ascii=False)}: '
            f'{_encode(value[key], level + 1)}'
            for key in sorted(value)
        ]
        return '{\n' + ',\n'.join(items) + '\n' + ' ' * level + '}'
    if isinstance(value, list):
        one_line = json.dumps(value, ensure_ascii=False)
        if (
            all(_is_scalar(item) for item in value)
            and level + len(one_line) <= 79
        ):
            return one_line
        pad = ' ' * (level + 1)
        items = [f'{pad}{_encode(item, level + 1)}' for item in value]
        return '[\n' + ',\n'.join(items) + '\n' + ' ' * level + ']'
    return json.dumps(value, ensure_ascii=False)


def dump_json(obj, path: str) -> None:
    """Deterministic JSON file (sorted keys, indent 1, short lists of
    scalars on one line, UTF-8, '\\n')."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        file.write(_encode(_jsonable(obj), 0))
        file.write('\n')


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
