"""Persist SAM embeddings without requiring PyTorch in the client.

New files retain the existing ``.pt`` filename but contain JSON metadata and
NumPy arrays. PyTorch ZIP files are read as NumPy arrays without importing
PyTorch. Older non-ZIP files require PyTorch once to load and resave them.

Each frame/z entry retains the full-image predictor fields. An optional
``crops`` mapping stores the same fields keyed by a digest of the input pixels,
shape, dtype, and image format; legacy entries without crops remain readable.
SAM2 entries use ``_orig_hw``, ``_features`` (image and high-resolution feature
maps), ``_is_image_set``, and ``_is_batch`` instead of SAM1's predictor fields.
Predictor adapters share caching and device restoration through the mixin below.
"""

import io
import hashlib
import json
import os
import pickle
import zipfile

import numpy as np

from cellacdc.service import payload


_METADATA = 'metadata.json'
_ARRAYS = 'arrays.npz'
_SAM_FIELDS = ('original_size', 'input_size', 'features', 'is_image_set')
_SAM2_FIELDS = ('_orig_hw', '_features', '_is_image_set', '_is_batch')


def _predictor_fields(predictor):
    return _SAM2_FIELDS if hasattr(predictor, '_is_image_set') else _SAM_FIELDS


def _map_features(value, convert):
    if isinstance(value, dict):
        return {key: _map_features(item, convert) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_features(item, convert) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_features(item, convert) for item in value)
    return convert(value)


class EmbeddingPredictorMixin:
    """Cache SAM/SAM2 encoder states without changing upstream reset behavior."""

    def set_embeddings(self, embedding):
        import torch
        for name in _predictor_fields(self):
            value = embedding[name]
            if name in ('features', '_features'):
                value = _map_features(
                    value, lambda item: torch.as_tensor(item, device=self.device)
                )
            setattr(self, name, value)

    def load_embeddings(self, embedding, frame_i=0, z=0):
        if embedding is None or not all(
            name in embedding for name in _predictor_fields(self)
        ):
            self._crop_embeddings = {}
            reason = 'missing' if embedding is None else 'incompatible with this predictor'
            print(
                f'SAM embeddings {reason} for frame {frame_i}, z-slice {z}; '
                'computing image embeddings.',
                flush=True,
            )
            return False
        self.set_embeddings(embedding)
        self._crop_embeddings = dict(embedding.get('crops', {}))
        print(
            f'SAM: restored embeddings for frame {frame_i}, z-slice {z} '
            f'on {self.device}.',
            flush=True,
        )
        return True

    def set_image(self, image, *args, **kwargs):
        if getattr(self, '_reuse_embedding', False):
            self._reuse_embedding = False
            print(
                'SAM: using prepared full-image embeddings; '
                'skipping image encoder.',
                flush=True,
            )
            return
        image_format = kwargs.get('image_format', args[0] if args else 'RGB')
        key = hashlib.sha256(
            str((image.shape, image.dtype.str, image_format)).encode('ascii')
            + np.ascontiguousarray(image).tobytes()
        ).hexdigest()
        cache = getattr(self, '_crop_embeddings', {})
        embedding = cache.get(key)
        if embedding is not None:
            self.set_embeddings(embedding)
            print(
                f'SAM: using cached image/crop embeddings for {image.shape[:2]}; '
                'skipping image encoder.',
                flush=True,
            )
            return
        print(
            f'SAM: computing image/crop embeddings for {image.shape[:2]}.',
            flush=True,
        )
        super().set_image(image, *args, **kwargs)
        embedding = predictor_state(self)
        for name in ('features', '_features'):
            if name in embedding:
                embedding[name] = _map_features(
                    embedding[name],
                    lambda item: item.detach().cpu().numpy()
                    if hasattr(item, 'detach') else item,
                )
        cache[key] = embedding
        self._crop_embeddings = cache


def predictor_state(predictor):
    """Capture the state set by SAM's image encoder."""
    return {
        name: getattr(predictor, name)
        for name in _predictor_fields(predictor)
    }


def from_predictor(predictor):
    """Include cached crop states without changing the legacy full-image fields."""
    embedding = predictor_state(predictor)
    crops = getattr(predictor, '_crop_embeddings', None)
    if crops:
        embedding['crops'] = dict(crops)
    return embedding


def save(embeddings, filepath):
    metadata, arrays = payload.encode_value(embeddings)
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_METADATA, metadata)
        if arrays is not None:
            archive.writestr(_ARRAYS, arrays)


def load(filepath):
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath) as archive:
            if _METADATA in archive.namelist():
                arrays = (
                    archive.read(_ARRAYS)
                    if _ARRAYS in archive.namelist() else None
                )
                return payload.decode_value(
                    archive.read(_METADATA).decode('utf-8'), arrays
                )
            return _load_torch_zip(archive)

    try:
        import torch
    except ImportError as err:
        raise RuntimeError(
            f'Legacy SAM embeddings at {filepath!r} use an unsupported old '
            'PyTorch format. Install PyTorch once to load and resave them.'
        ) from err
    return torch.load(filepath, map_location='cpu')


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


class _Storage:
    def __init__(self, array):
        self.array = array


_STORAGE_DTYPES = {
    'ByteStorage': np.uint8,
    'CharStorage': np.int8,
    'ShortStorage': np.int16,
    'IntStorage': np.int32,
    'LongStorage': np.int64,
    'HalfStorage': np.float16,
    'FloatStorage': np.float32,
    'DoubleStorage': np.float64,
    'BoolStorage': np.bool_,
}


def _rebuild_tensor(storage, offset, size, stride, *args):
    array = storage.array
    strides = tuple(int(value) * array.dtype.itemsize for value in stride)
    return np.ndarray(
        shape=tuple(size), dtype=array.dtype, buffer=array,
        offset=int(offset) * array.dtype.itemsize, strides=strides,
    ).copy()


class _TorchUnpickler(pickle.Unpickler):
    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module, name):
        if module == 'torch._utils' and name in {
            '_rebuild_tensor', '_rebuild_tensor_v2',
        }:
            return _rebuild_tensor
        if module == 'torch' and name in _STORAGE_DTYPES:
            return _StorageType(_STORAGE_DTYPES[name])
        if module == 'collections' and name == 'OrderedDict':
            from collections import OrderedDict
            return OrderedDict
        raise pickle.UnpicklingError(
             f'Unsupported global in SAM embeddings: {module}.{name}'
         )

    def persistent_load(self, saved_id):
        kind, storage_type, key, _location, size = saved_id[:5]
        if kind != 'storage':
            raise pickle.UnpicklingError(
                f'Unsupported PyTorch persistent object: {kind!r}'
            )
        data = self.archive.read(f'{self.prefix}data/{key}')
        array = np.frombuffer(data, dtype=storage_type.dtype, count=size)
        return _Storage(array)


def _load_torch_zip(archive):
    data_pickle = next(
        (name for name in archive.namelist() if name.endswith('/data.pkl')),
        None,
    )
    if data_pickle is None:
        raise ValueError('Not a supported SAM embeddings file.')
    prefix = data_pickle[:-len('data.pkl')]
    return _TorchUnpickler(
        io.BytesIO(archive.read(data_pickle)), archive, prefix
    ).load()
