import ast
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cellacdc import _types, myutils, myutils_models, sam_embeddings


@pytest.fixture
def sam2_wrapper(monkeypatch):
    import cv2
    import skimage.measure
    import sys

    class Predictor:
        device = 'cpu'

        def __init__(self, model=None, **kwargs):
            self.encoder_calls = 0
            self.options = kwargs
            self.reset_predictor()

        def set_image(self, image):
            self.reset_predictor()
            self.encoder_calls += 1
            self._orig_hw = [image.shape[:2]]
            self._features = {
                'image_embed': np.full((1, 2), self.encoder_calls, dtype=np.float32),
                'high_res_feats': [
                    np.full((1, 4), self.encoder_calls + 10, dtype=np.float32),
                    np.full((1, 8), self.encoder_calls + 20, dtype=np.float32),
                ],
            }
            self._is_image_set = True

        def predict(self, **kwargs):
            assert self._is_image_set and not self._is_batch
            assert self._features['high_res_feats'][0].shape == (1, 4)
            return np.ones((1, *self._orig_hw[0]), dtype=bool), [1], None

        def reset_predictor(self):
            self._is_image_set = False
            self._features = None
            self._orig_hw = None
            self._is_batch = False

    class Generator:
        def __init__(self, model=None, **kwargs):
            self.with_crop = True

        def generate(self, image):
            self.predictor.set_image(image)
            assert self.predictor._is_image_set
            self.predictor.reset_predictor()
            if self.with_crop:
                self.predictor.set_image(image[:4, :4])
                self.predictor.reset_predictor()
            return [{'segmentation': np.ones(image.shape[:2], dtype=bool), 'area': 64}]

    monkeypatch.setitem(
        sys.modules, 'torch', SimpleNamespace(
            as_tensor=lambda value, device: np.asarray(value),
        ),
    )
    source = (
        Path(__file__).resolve().parents[1] / 'cellacdc' / 'models'
        / 'sam2' / 'acdcSegment.py'
    )
    tree = ast.parse(source.read_text())
    classes = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.ClassDef)],
        type_ignores=[],
    )
    namespace = {
        'np': np, 'pd': pd, 'cv2': cv2, 'skimage': skimage,
        'defaultdict': defaultdict, 'myutils': myutils, '_types': _types,
        'sam_embeddings': sam_embeddings, 'model_types': {},
        'SAM2ImagePredictor': Predictor, 'SAM2AutomaticMaskGenerator': Generator,
    }
    exec(compile(classes, str(source), 'exec'), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize('automatic', [False, True])
@pytest.mark.parametrize('stack', [False, True])
def test_sam2_prediction_disk_reload_skips_all_encoders(
        sam2_wrapper, tmp_path, automatic, stack, capsys,
    ):
    from cellacdc import load

    wrapper = sam2_wrapper.Model.__new__(sam2_wrapper.Model)
    predictor = sam2_wrapper._EmbeddingPredictor()
    if automatic:
        wrapper.model = sam2_wrapper.SAM2AutomaticMaskGenerator()
        wrapper.model.predictor = predictor
        wrapper._input_points_df = None
    else:
        wrapper.model = predictor
        wrapper._input_points_df = pd.DataFrame({
            'x': [2, 2], 'y': [2, 2], 'id': [1, 1], 'z': [0, 1],
        })
    position = load.loadData.__new__(load.loadData)
    position.sam_embeddings_path = str(tmp_path / 'sam2.pt')
    image = np.arange(128 if stack else 64, dtype=np.uint8).reshape(
        (2, 8, 8) if stack else (8, 8)
    )
    first = wrapper.segment(
        image, frame_i=None, posData=position, start_z_slice=5,
        automatic_removal_of_background=False,
    )
    assert first.shape == image.shape
    assert set(position.sam_embeddings[None]) == ({5, 6} if stack else {5})
    saved = sam_embeddings.load(position.sam_embeddings_path)
    embedding = saved[None][5]
    assert set(embedding) >= {'_features', '_orig_hw', '_is_image_set', '_is_batch'}
    assert embedding['_orig_hw'] == [(8, 8)]
    assert embedding['_is_image_set'] is True
    assert embedding['_is_batch'] is False
    if automatic:
        crop = next(iter(embedding['crops'].values()))
        assert crop['_orig_hw'] == [(4, 4)]
        assert len(crop['_features']['high_res_feats']) == 2
    calls = predictor.encoder_calls
    predictor.reset_predictor()
    del position.sam_embeddings
    capsys.readouterr()
    second = wrapper.segment(
        image, frame_i=None, posData=position, start_z_slice=5,
        use_loaded_embeddings=True, automatic_removal_of_background=False,
    )
    assert predictor.encoder_calls == calls
    np.testing.assert_array_equal(second, first)
    assert predictor._orig_hw == [(8, 8)]
    assert len(predictor._features['high_res_feats']) == 2
    output = capsys.readouterr().out
    assert 'restored embeddings for frame None, z-slice 5' in output
    assert 'skipping image encoder' in output
    assert 'computing image/crop embeddings' not in output


def test_worker_stores_native_sam2_state(sam2_wrapper):
    from cellacdc.service.worker import _EmbeddingPosData

    predictor = sam2_wrapper._EmbeddingPredictor()
    predictor.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    wrapper = SimpleNamespace(model=predictor)
    position = _EmbeddingPosData(wrapper, None, {})
    position.storeSamEmbeddings(wrapper, frame_i=None, z=5)
    restored = position.getSamEmbeddings(None, z=5)
    assert 'original_size' not in restored
    assert restored['_orig_hw'] == [(8, 8)]
    predictor.reset_predictor()
    assert predictor.load_embeddings(restored, frame_i=None, z=5)
    assert predictor._is_image_set and not predictor._is_batch
    assert predictor.encoder_calls == 1
    np.testing.assert_array_equal(
        predictor._features['high_res_feats'][1], np.full((1, 8), 21),
    )


def test_sam2_rejects_sam1_embedding_with_explicit_recomputation(sam2_wrapper, capsys):
    predictor = sam2_wrapper._EmbeddingPredictor()
    assert not predictor.load_embeddings({
        'original_size': (8, 8), 'input_size': (8, 8),
        'features': np.ones((1, 2)), 'is_image_set': True,
    })
    predictor.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    assert predictor.encoder_calls == 1
    assert 'incompatible with this predictor' in capsys.readouterr().out


def test_sam2_failure_preserves_full_image_state(sam2_wrapper, monkeypatch):
    wrapper = sam2_wrapper.Model.__new__(sam2_wrapper.Model)
    wrapper.model = sam2_wrapper.SAM2AutomaticMaskGenerator()
    predictor = sam2_wrapper._EmbeddingPredictor()
    wrapper.model.predictor = predictor

    def fail(self, image):
        self.predictor.set_image(image)
        self.predictor.reset_predictor()
        self.predictor.set_image(image[:4, :4])
        raise RuntimeError('crop failed')

    monkeypatch.setattr(sam2_wrapper.SAM2AutomaticMaskGenerator, 'generate', fail)
    with pytest.raises(RuntimeError, match='crop failed'):
        wrapper._segment_2D_image(np.zeros((8, 8), dtype=np.uint8), None, None)
    assert predictor._orig_hw == [(8, 8)]
    assert predictor._is_image_set and not predictor._is_batch
    assert predictor._reuse_embedding is False
    assert predictor.encoder_calls == 2


@pytest.mark.parametrize('automatic', [False, True])
def test_sam2_constructor_preserves_predictor_configuration(sam2_wrapper, monkeypatch, automatic):
    namespace = sam2_wrapper.Model.__init__.__globals__
    monkeypatch.setitem(namespace, 'os', os)
    monkeypatch.setitem(namespace, 'sam_models_path', '')
    monkeypatch.setitem(namespace, 'model_types', {'test': ('config', 'weights')})
    monkeypatch.setitem(namespace, 'build_sam2', lambda **kwargs: object())
    points = None if automatic else pd.DataFrame({'x': [2], 'y': [2], 'id': [1]})
    wrapper = sam2_wrapper.Model(
        model_type='test', input_points_df=points, min_mask_region_area=7, gpu=False,
    )
    predictor = wrapper.model.predictor if automatic else wrapper.model
    assert isinstance(predictor, sam2_wrapper._EmbeddingPredictor)
    if automatic:
        assert predictor.options == {'max_hole_area': 7, 'max_sprinkle_area': 7}


def test_sam2_only_embeddings_reuses_saved_full_image(sam2_wrapper, tmp_path, capsys):
    from cellacdc import load

    wrapper = sam2_wrapper.Model.__new__(sam2_wrapper.Model)
    wrapper.model = sam2_wrapper._EmbeddingPredictor()
    wrapper._input_points_df = None
    position = load.loadData.__new__(load.loadData)
    position.sam_embeddings_path = str(tmp_path / 'sam2.pt')
    image = np.zeros((8, 8), dtype=np.uint8)
    labels = wrapper.segment(
        image, frame_i=0, posData=position, only_embeddings=True,
        automatic_removal_of_background=False,
    )
    assert wrapper.model.encoder_calls == 1
    np.testing.assert_array_equal(labels, np.zeros_like(image))
    wrapper.model.reset_predictor()
    del position.sam_embeddings
    capsys.readouterr()
    wrapper.segment(
        image, frame_i=0, posData=position, only_embeddings=True,
        use_loaded_embeddings=True, automatic_removal_of_background=False,
    )
    assert wrapper.model.encoder_calls == 1
    assert 'using prepared full-image embeddings' in capsys.readouterr().out


@pytest.mark.parametrize('single_slice_payload', [False, True])
def test_sam2_worker_client_roundtrip(sam2_wrapper, monkeypatch, tmp_path, single_slice_payload):
    import asyncio
    from cellacdc import load
    from cellacdc.service import worker
    from cellacdc.service.client import ModelClient

    models = []

    class Model(sam2_wrapper.Model):
        def __init__(self):
            self.model = sam2_wrapper.SAM2AutomaticMaskGenerator()
            self.model.predictor = sam2_wrapper._EmbeddingPredictor()
            self._input_points_df = None
            models.append(self)

    monkeypatch.setattr(
        myutils_models, 'import_segment_module',
        lambda name: SimpleNamespace(Model=Model),
    )
    app = worker.create_app('sam2')
    endpoint = next(
        route.endpoint for route in app.routes if getattr(route, 'path', None) == '/segment'
    )

    class Upload:
        def __init__(self, data):
            self.data = data

        async def read(self):
            return self.data

    class Session:
        def post(self, url, files, data, timeout):
            response = asyncio.run(endpoint(
                image=Upload(files['image'][1]),
                params=data['params'], params_data=None,
                embedding_meta=data['embedding_meta'],
                embeddings=Upload(files['embeddings'][1]) if 'embeddings' in files else None,
            ))
            return SimpleNamespace(
                ok=True, content=response.body, status_code=response.status_code,
                headers=response.headers, url=url,
            )

    client = ModelClient.__new__(ModelClient)
    client.model_name = 'sam2'
    client.base_url = 'http://worker'
    client.timeout = None
    client.session = Session()
    position = load.loadData.__new__(load.loadData)
    position.sam_embeddings_path = str(tmp_path / 'sam2.pt')
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    result = client.segment(
        image, frame_i=None, start_z_slice=5, posData=position,
        automatic_removal_of_background=False,
    )
    np.testing.assert_array_equal(result, np.ones((8, 8)))
    saved = sam_embeddings.load(position.sam_embeddings_path)
    assert saved[None][5]['_features']['high_res_feats'][0].shape == (1, 4)
    predictor = models[0].model.predictor
    calls = predictor.encoder_calls
    if single_slice_payload:
        position.sam_embeddings[None] = position.sam_embeddings[None][5]
    client.segment(
        image, frame_i=None, start_z_slice=5, posData=position,
        use_loaded_embeddings=True, automatic_removal_of_background=False,
    )
    assert predictor.encoder_calls == calls
    assert set(position.sam_embeddings[None]) == {5}
