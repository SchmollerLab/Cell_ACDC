"""`FakeBioImage`: a stand-in for `bioio.BioImage` driven by a JSON spec file.

The fake replaces only the third-party boundary, so all of Cell-ACDC's code
(ImageReader, metadata extraction, writer, runner, worker child) runs for real
without bioio. Use it through the `ImageReader(bioimage_factory=...)` seam, or
in worker children as the dotted path `FAKE_BIOIMAGE_PATH`
('tests.datastruct.fake_bioio:FakeBioImage', honoured only with
ACDC_DATASTRUCT_TEST_HOOKS=1).

The raw "file" is a JSON spec written by `write_fake_raw(path, spec)` at a path
with any extension ('.czi' exercises EXTENSION_METADATA_ATTR_MAPPER, '.nd2' is
generic). Spec keys (all optional except `scenes`; see `make_spec`):

    scenes          list of {"T", "C", "Z", "Y", "X", "name"?, "channel_names"?}
    dims            dimension order of every scene (permutation of 'TCZYX';
                    'S' is appended when rgb), default 'TCZYX'
    dtype           numpy dtype name, default 'uint16'
    channel_names   names for every scene (a scene's own list wins); default
                    'Channel:<scene>:<c>' (what bioio-tifffile reports)
    np_str_channel_names   true -> names are np.str_ (as bioio returns them)
    physical_pixel_sizes   [Z, Y, X] (entries may be null); default all null
    time_interval_s        float or null -> `time_interval` (timedelta | None)
    timelapse_interval_s   float or null -> `standard_metadata.timelapse_interval`
                           (default: same as time_interval_s)
    metadata_text   str returned by `metadata`, default '<FakeBioImage/>'
    rgb             true -> extra trailing 'S' axis of size `samples` (3)
    samples         3 or 4
    s_offset, c_offset     added to the encoded s and c (see synthetic.py)
    noise           str printed to stdout (flushed) on open and on every read
    delay_s         sleep before every pixel read
    fail            {"stage": "open"|"read"|"metadata",
                     "how": "raise"|"exit3"|"keyboard",
                     "exception": builtin exception name (default RuntimeError),
                     "after_planes": reads that succeed before a read failure}

Pixels: (s_offset + scene)*1000 + t*100 + (c_offset + c)*10 + z, constant
per plane (`synthetic.decode_pixel`); values wrap when `dtype` is too small
(e.g. uint8 RGB). "exit3" calls os._exit(3): use it only in child processes.
"""
from __future__ import annotations

import builtins
import datetime
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from . import synthetic

FAKE_BIOIMAGE_PATH = 'tests.datastruct.fake_bioio:FakeBioImage'

SPEC_KEYS = frozenset({
    'scenes', 'dims', 'dtype', 'channel_names', 'np_str_channel_names',
    'physical_pixel_sizes', 'time_interval_s', 'timelapse_interval_s',
    'metadata_text', 'rgb', 'samples', 's_offset', 'c_offset', 'noise',
    'delay_s', 'fail',
})
SCENE_KEYS = frozenset({'T', 'C', 'Z', 'Y', 'X', 'name', 'channel_names'})
FAIL_STAGES = ('open', 'read', 'metadata')
FAIL_HOWS = ('raise', 'exit3', 'keyboard')


class PhysicalPixelSizes(NamedTuple):
    """Same fields as bioio_base.types.PhysicalPixelSizes."""
    Z: float | None
    Y: float | None
    X: float | None


class FakeDimensions:
    """Duck type of bioio_base.dimensions.Dimensions."""

    def __init__(self, order: str, shape: tuple[int, ...]) -> None:
        self._order = order
        self._shape = tuple(shape)
        for dim, size in zip(order, shape):
            setattr(self, dim, size)

    @property
    def order(self) -> str:
        return self._order

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def items(self):
        return dict(zip(self._order, self._shape)).items()

    def __getitem__(self, key) -> tuple[int, ...]:
        keys = [key] if isinstance(key, str) else list(key)
        missing = [k for k in keys if k not in self._order]
        if missing:
            raise IndexError(f'dimension(s) {missing} not in {self._order}')
        return tuple(self._shape[self._order.index(k)] for k in keys)

    def __repr__(self) -> str:
        items = ', '.join(f'{d}: {s}' for d, s in zip(self._order, self._shape))
        return f'<Dimensions [{items}]>'


class FakeLazyArray:
    """What `get_image_dask_data` returns: `.compute()` gives the array."""

    def __init__(self, owner: 'FakeBioImage', order: str | None,
                 kwargs: dict) -> None:
        self._owner = owner
        self._order = order
        self._kwargs = kwargs

    def compute(self) -> np.ndarray:
        return self._owner._read(self._order, self._kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._owner._select(self._order, self._kwargs, values=False)

    @property
    def dtype(self) -> np.dtype:
        return self._owner.dtype


def make_spec(*, scenes: int = 1, T: int = 3, C: int = 2, Z: int = 2,
              Y: int = 16, X: int = 20, scene_names=None, **extra) -> dict:
    """Spec with `scenes` identical scenes of size T x C x Z x Y x X.

    `scene_names` (sequence) sets each scene's name; `extra` adds any other
    spec key (validated by `validate_spec`)."""
    scene_list = []
    for k in range(scenes):
        scene = {'T': T, 'C': C, 'Z': Z, 'Y': Y, 'X': X}
        if scene_names is not None:
            scene['name'] = scene_names[k]
        scene_list.append(scene)
    spec = {'scenes': scene_list, **extra}
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    """ValueError on unknown keys, bad scenes, dims or fail settings."""
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f'unknown FakeBioImage spec keys: {sorted(unknown)}')
    scenes = spec.get('scenes')
    if not scenes:
        raise ValueError('spec needs a non-empty "scenes" list')
    for scene in scenes:
        bad = set(scene) - SCENE_KEYS
        missing = {'T', 'C', 'Z', 'Y', 'X'} - set(scene)
        if bad or missing:
            raise ValueError(
                f'bad scene {scene!r}: unknown {sorted(bad)}, '
                f'missing {sorted(missing)}'
            )
    dims = spec.get('dims', 'TCZYX')
    if sorted(dims) != sorted('TCZYX'):
        raise ValueError(f'dims must be a permutation of TCZYX, got {dims!r}')
    fail = spec.get('fail')
    if fail is not None:
        if fail.get('stage') not in FAIL_STAGES:
            raise ValueError(f'fail.stage must be one of {FAIL_STAGES}')
        if fail.get('how', 'raise') not in FAIL_HOWS:
            raise ValueError(f'fail.how must be one of {FAIL_HOWS}')


def write_fake_raw(path: str, spec: dict) -> str:
    """Write `spec` (validated) as the fake raw file `path`; returns `path`."""
    validate_spec(spec)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        json.dump(spec, file, indent=1, sort_keys=True)
    return path


def read_fake_spec(path: str) -> dict:
    """Load and validate the spec of a fake raw file."""
    with open(path, 'r', encoding='utf-8') as file:
        try:
            spec = json.load(file)
        except ValueError as err:
            raise ValueError(f'{path} is not a fake raw file') from err
    validate_spec(spec)
    return spec


def _seconds_to_timedelta(value) -> datetime.timedelta | None:
    if value is None:
        return None
    return datetime.timedelta(seconds=value)


class FakeBioImage:
    """Duck type of `bioio.BioImage` for the attributes Cell-ACDC uses.

    Class-level `log` records ('open', path, kwargs), ('read', path, scene,
    selection) and ('close', path) events of every instance; call
    `FakeBioImage.reset_log()` at the start of a test.
    """

    log: list[tuple] = []

    def __init__(self, image, **kwargs) -> None:
        self.path = os.fspath(image)
        self.kwargs = dict(kwargs)
        self.spec = read_fake_spec(self.path)
        self.closed = False
        self.planes_read = 0
        self._scene_index = 0
        FakeBioImage.log.append(('open', self.path, self.kwargs))
        self._noise()
        self._maybe_fail('open')

    @classmethod
    def reset_log(cls) -> None:
        cls.log.clear()

    @classmethod
    def opened_paths(cls) -> list[str]:
        """Paths of every construction recorded in `log`, in order."""
        return [event[1] for event in cls.log if event[0] == 'open']

    # ---------------------------------------------------------------- helpers
    def _noise(self) -> None:
        noise = self.spec.get('noise')
        if noise:
            print(noise, flush=True)

    def _maybe_fail(self, stage: str) -> None:
        fail = self.spec.get('fail')
        if not fail or fail.get('stage') != stage:
            return
        if stage == 'read' and self.planes_read <= fail.get('after_planes', 0):
            return
        how = fail.get('how', 'raise')
        if how == 'exit3':
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(3)
        if how == 'keyboard':
            raise KeyboardInterrupt
        exc_type = getattr(builtins, fail.get('exception', 'RuntimeError'))
        raise exc_type(
            f'FakeBioImage: injected {stage} failure for {self.path}'
        )

    def _metadata_access(self) -> None:
        self._maybe_fail('metadata')

    @property
    def _scene(self) -> dict:
        return self.spec['scenes'][self._scene_index]

    @property
    def _rgb(self) -> bool:
        return bool(self.spec.get('rgb', False))

    def _canonical(self) -> np.ndarray:
        """Current scene as TCZYX (+S) array."""
        scene = self._scene
        data = synthetic.encoded_array(
            'TCZYX', T=scene['T'], C=scene['C'], Z=scene['Z'],
            Y=scene['Y'], X=scene['X'],
            s=self.spec.get('s_offset', 0) + self._scene_index,
            c0=self.spec.get('c_offset', 0), dtype='int64'
        )
        if self._rgb:
            samples = self.spec.get('samples', 3)
            data = np.repeat(data[..., np.newaxis], samples, axis=-1)
        return data.astype(self.dtype)

    def _in_order(self) -> np.ndarray:
        data = self._canonical()
        canonical = 'TCZYX' + ('S' if self._rgb else '')
        return np.transpose(data, [canonical.index(d) for d in self.dims.order])

    def _select(self, order: str | None, kwargs: dict, values: bool = True):
        """bioio-like selection: int -> that index (dimension dropped unless
        it is in `order`), sequence -> those indices; dimensions not in
        `order` must end up with size 1. values=False returns the shape."""
        dims = self.dims.order
        sizes = dict(zip(dims, self.dims.shape))
        order = dims if order is None else order
        if (
            set(kwargs) - set(dims) or len(set(order)) != len(order)
            or set(order) - set(dims)
        ):
            raise ValueError(
                f'invalid request order={order!r} {kwargs!r} for dims {dims}'
            )
        selection = {}
        for dim, value in kwargs.items():
            indices = (
                [int(value)] if isinstance(value, (int, np.integer))
                else [int(i) for i in value]
            )
            for i in indices:
                if not 0 <= i < sizes[dim]:
                    raise IndexError(f'{dim}={i} out of range 0..{sizes[dim]-1}')
            drop = isinstance(value, (int, np.integer)) and dim not in order
            selection[dim] = indices[0] if drop else indices
            sizes[dim] = 1 if drop else len(indices)
        kept = [d for d in dims if not isinstance(selection.get(d), int)]
        for dim in kept:
            if dim not in order and sizes[dim] != 1:
                raise ValueError(f'dimension {dim} must be selected or kept')
        if not values:
            return tuple(sizes[d] for d in order)
        data = self._in_order()
        for axis, dim in enumerate(dims):
            if isinstance(selection.get(dim), list):
                data = np.take(data, selection[dim], axis=axis)
        data = data[tuple(
            selection[d] if isinstance(selection.get(d), int) else slice(None)
            for d in dims
        )]
        squeeze = tuple(i for i, d in enumerate(kept) if d not in order)
        if squeeze:
            data = np.squeeze(data, axis=squeeze)
        remaining = [d for d in kept if d in order]
        return np.transpose(data, [remaining.index(d) for d in order])

    def _read(self, order: str | None, kwargs: dict) -> np.ndarray:
        delay = self.spec.get('delay_s')
        if delay:
            time.sleep(delay)
        self._noise()
        self.planes_read += 1
        self._maybe_fail('read')
        FakeBioImage.log.append(
            ('read', self.path, self._scene_index, dict(kwargs))
        )
        return self._select(order, kwargs)

    # ------------------------------------------------------------ BioImage API
    @property
    def scenes(self) -> tuple[str, ...]:
        return tuple(
            scene.get('name', f'Image:{k}')
            for k, scene in enumerate(self.spec['scenes'])
        )

    @property
    def current_scene(self) -> str:
        return self.scenes[self._scene_index]

    @property
    def current_scene_index(self) -> int:
        return self._scene_index

    def set_scene(self, scene_id) -> None:
        if isinstance(scene_id, (int, np.integer)):
            if not 0 <= scene_id < len(self.scenes):
                raise IndexError(f'scene index {scene_id} out of range')
            self._scene_index = int(scene_id)
            return
        if scene_id not in self.scenes:
            raise ValueError(f'unknown scene {scene_id!r}')
        self._scene_index = self.scenes.index(scene_id)

    @property
    def dims(self) -> FakeDimensions:
        scene = self._scene
        order = self.spec.get('dims', 'TCZYX')
        shape = [scene[d] for d in order]
        if self._rgb:
            order += 'S'
            shape.append(self.spec.get('samples', 3))
        return FakeDimensions(order, tuple(shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dims.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.spec.get('dtype', 'uint16'))

    @property
    def data(self) -> np.ndarray:
        return self._read(None, {})

    def get_image_data(self, dimension_order_out: str | None = None,
                       **kwargs) -> np.ndarray:
        return self._read(dimension_order_out, kwargs)

    def get_image_dask_data(self, dimension_order_out: str | None = None,
                            **kwargs) -> FakeLazyArray:
        self._select(dimension_order_out, kwargs, values=False)
        return FakeLazyArray(self, dimension_order_out, kwargs)

    @property
    def channel_names(self) -> list:
        self._metadata_access()
        names = self._scene.get('channel_names', self.spec.get('channel_names'))
        if names is None:
            names = [
                f'Channel:{self._scene_index}:{c}'
                for c in range(self._scene['C'])
            ]
        if self.spec.get('np_str_channel_names'):
            return [np.str_(name) for name in names]
        return list(names)

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        self._metadata_access()
        sizes = self.spec.get('physical_pixel_sizes') or (None, None, None)
        return PhysicalPixelSizes(*sizes)

    @property
    def time_interval(self) -> datetime.timedelta | None:
        self._metadata_access()
        return _seconds_to_timedelta(self.spec.get('time_interval_s'))

    @property
    def standard_metadata(self) -> SimpleNamespace:
        self._metadata_access()
        interval = self.spec.get(
            'timelapse_interval_s', self.spec.get('time_interval_s')
        )
        scene = self._scene
        sizes = self.spec.get('physical_pixel_sizes') or (None, None, None)
        return SimpleNamespace(
            timelapse_interval=_seconds_to_timedelta(interval),
            image_size_t=scene['T'], image_size_c=scene['C'],
            image_size_z=scene['Z'], image_size_y=scene['Y'],
            image_size_x=scene['X'], pixel_size_z=sizes[0],
            pixel_size_y=sizes[1], pixel_size_x=sizes[2],
        )

    @property
    def metadata(self) -> str:
        self._metadata_access()
        return self.spec.get('metadata_text', '<FakeBioImage/>')

    def close(self) -> None:
        if not self.closed:
            FakeBioImage.log.append(('close', self.path))
        self.closed = True

    def __enter__(self) -> 'FakeBioImage':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<FakeBioImage {self.path!r}>'
