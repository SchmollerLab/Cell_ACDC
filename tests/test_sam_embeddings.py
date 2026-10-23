import builtins
import pickle
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from cellacdc import load, sam_embeddings


@pytest.fixture
def without_torch(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise ModuleNotFoundError('torch intentionally unavailable', name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)


def _embeddings():
    return {
        None: {
            z: {
                'original_size': (10, 12),
                'input_size': (20, 24),
                'features': np.arange(6, dtype=np.float32).reshape(2, 3) + z,
                'is_image_set': True,
            } for z in range(500)
        }
    }


def _assert_embeddings(actual, expected):
    assert actual.keys() == expected.keys()
    for frame, slices in expected.items():
        assert actual[frame].keys() == slices.keys()
        for z, embedding in slices.items():
            restored = actual[frame][z]
            assert restored['original_size'] == embedding['original_size']
            assert restored['input_size'] == embedding['input_size']
            assert restored['is_image_set'] is True
            np.testing.assert_array_equal(restored['features'], embedding['features'])
            assert restored['features'].dtype == embedding['features'].dtype


def test_position_save_load_without_torch(tmp_path, without_torch):
    embeddings = _embeddings()
    path = tmp_path / 'image_sam_embeddings.pt'
    position = load.loadData.__new__(load.loadData)
    position.sam_embeddings = embeddings
    position.sam_embeddings_path = str(path)
    position.saveSamEmbeddings(logger_func=lambda text: None)
    del position.sam_embeddings
    position.loadSamEmbeddings()
    _assert_embeddings(position.sam_embeddings, embeddings)
    assert isinstance(position.getSamEmbeddings(None, 0)['features'], np.ndarray)
    position.sam_embeddings = {}
    position.loadSamEmbeddings()
    assert position.sam_embeddings == {}
    position.loadSamEmbeddings(force_reload=True)
    _assert_embeddings(position.sam_embeddings, embeddings)


def test_tensor_like_features_saved_without_torch(tmp_path, without_torch):
    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((2, 3), dtype=np.float32)

    path = tmp_path / 'embeddings.pt'
    sam_embeddings.save({0: {0: {'features': Tensor()}}}, path)
    restored = sam_embeddings.load(path)
    np.testing.assert_array_equal(restored[0][0]['features'], np.ones((2, 3)))


@pytest.mark.parametrize('offset,size,stride,valid', [
    (0, (2, 3), (3, 1), True),
    (0, (3, 2), (1, 3), True),
    (1, (2, 2), (3, 1), True),
    (5, (6,), (-1,), True),
    (0, (2, 3), (0, 1), True),
    (-1, (1,), (1,), False),
    (7, (1,), (1,), False),
    (0, (7,), (1,), False),
    (0, (2,), (6,), False),
    (0, (2,), (-1,), False),
    (1, (2, 3), (3, 1), False),
    (0, (-1,), (1,), False),
    (0, (2, 3), (1,), False),
])
def test_legacy_torch_zip_without_torch(
        tmp_path, without_torch, offset, size, stride, valid,
    ):
    # PyTorch's ZIP format uses a pickle tensor rebuild plus raw storage.
    def encoded(value):
        return pickle.dumps(value, protocol=2)[2:-1]

    tensor_pickle = (
        b'\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(('
        + encoded('storage') + b'ctorch\nFloatStorage\n'
        + encoded('0') + encoded('cpu') + encoded(6)
        + b'tQ' + encoded(offset) + encoded(size) + encoded(stride)
        + b'\x89ccollections\nOrderedDict\n)RtR.'
    )
    embedding_pickle = (
        b'\x80\x02}N}K\x00}'
        + encoded('original_size') + encoded((10, 12)) + b's'
        + encoded('input_size') + encoded((20, 24)) + b's'
        + encoded('is_image_set') + b'\x88s'
        + encoded('features') + tensor_pickle[2:-1] + b'sss.'
    )
    path = tmp_path / 'legacy.pt'
    storage = np.arange(6, dtype=np.float32)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', embedding_pickle)
        archive.writestr('archive/data/0', storage.tobytes())
        archive.writestr('archive/version', '3\n')
        archive.writestr('archive/byteorder', 'little')
    if not valid:
        with pytest.raises(ValueError):
            sam_embeddings.load(path)
        return
    restored = sam_embeddings.load(path)
    features = restored[None][0]['features']
    assert restored[None][0]['original_size'] == (10, 12)
    assert restored[None][0]['input_size'] == (20, 24)
    assert restored[None][0]['is_image_set'] is True
    expected = np.ndarray(
        shape=size, dtype=storage.dtype, buffer=storage,
        offset=offset * storage.dtype.itemsize,
        strides=tuple(value * storage.dtype.itemsize for value in stride),
    )
    np.testing.assert_array_equal(features, expected)
    assert features.dtype == np.float32
    portable = tmp_path / 'resaved.pt'
    sam_embeddings.save(restored, portable)
    _assert_embeddings(sam_embeddings.load(portable), restored)


def test_non_zip_legacy_file_reports_missing_torch(tmp_path, without_torch):
    path = tmp_path / 'old.pt'
    path.write_bytes(b'old PyTorch serialization')
    with pytest.raises(RuntimeError, match='load and resave'):
        sam_embeddings.load(path)


def test_non_zip_legacy_file_uses_available_torch(tmp_path, monkeypatch):
    import sys
    from unittest.mock import Mock

    path = tmp_path / 'old.pt'
    path.write_bytes(b'old PyTorch serialization')
    embeddings = _embeddings()
    torch_load = Mock(return_value=embeddings)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(load=torch_load))
    assert sam_embeddings.load(path) is embeddings
    torch_load.assert_called_once_with(path, map_location='cpu')
