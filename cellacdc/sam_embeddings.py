import io
import json
import os
import pickle
import zipfile

import numpy as np

from cellacdc.service import payload


_METADATA = 'metadata.json'
_ARRAYS = 'arrays.npz'


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
    array = storage.array[int(offset):]
    strides = tuple(int(value) * array.dtype.itemsize for value in stride)
    return np.lib.stride_tricks.as_strided(
        array, shape=tuple(size), strides=strides
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
