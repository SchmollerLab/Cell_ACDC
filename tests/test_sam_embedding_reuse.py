import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cellacdc import _types, myutils, sam_embeddings


@pytest.fixture
def sam_wrapper(monkeypatch):
    import cv2
    import skimage.measure
    import sys
    from tqdm import tqdm

    class Predictor:
        device = 'cpu'

        def __init__(self, model=None):
            self.encoder_calls = 0

        def set_image(self, image, image_format='RGB'):
            self.encoder_calls += 1
            self.original_size = image.shape[:2]
            self.input_size = image.shape[:2]
            self.features = np.full((1, 2), self.encoder_calls)
            self.is_image_set = True

        def predict(self, **kwargs):
            assert self.is_image_set
            return np.ones((1, 8, 8), dtype=bool), [1], None

        def reset_image(self):
            self.features = None
            self.is_image_set = False

    class Generator:
        def generate(self, image):
            self.predictor.set_image(image)
            assert self.predictor.is_image_set
            self.predictor.reset_image()
            if getattr(self, 'with_crop', False):
                self.predictor.set_image(image[:4, :4])
                self.predictor.reset_image()
            return [{'segmentation': np.ones((8, 8), dtype=bool), 'area': 64}]

    monkeypatch.setitem(
        sys.modules, 'torch',
        SimpleNamespace(as_tensor=lambda array, device: array),
    )
    # Execute the wrapper classes without importing/installing SAM or weights.
    source = (
        Path(__file__).resolve().parents[1] / 'cellacdc' / 'models'
        / 'segment_anything' / 'acdcSegment.py'
    )
    tree = ast.parse(source.read_text())
    classes = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.ClassDef)],
        type_ignores=[],
    )
    namespace = {
        'np': np, 'pd': pd, 'cv2': cv2, 'myutils': myutils, '_types': _types,
        'skimage': skimage, 'tqdm': tqdm,
        'hashlib': hashlib, 'sam_embeddings': sam_embeddings,
        'model_types': {}, 'SamPredictor': Predictor,
        'SamAutomaticMaskGenerator': Generator,
    }
    exec(compile(classes, str(source), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def test_restored_embeddings_skip_prompted_image_encoder(sam_wrapper, capsys):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    predictor = sam_wrapper._EmbeddingPredictor()
    wrapper.model = predictor
    features = np.full((1, 2), 42, dtype=np.float32)
    position = SimpleNamespace(getSamEmbeddings=lambda **kwargs: {
        'original_size': (8, 8), 'input_size': (8, 8),
        'features': features, 'is_image_set': True,
    })
    restored = wrapper._get_img_embeddings(position, frame_i=None, z=5)
    assert restored
    assert predictor.features is features
    points = {1: np.array([[2, 2]])}
    labels = {1: np.array([1])}
    image = np.zeros((8, 8), dtype=np.uint8)
    result = wrapper._segment_2D_image(
        image, points, labels, embeddings_already_init=restored,
    )
    assert predictor.encoder_calls == 0
    np.testing.assert_array_equal(result, np.ones((8, 8)))
    np.testing.assert_array_equal(predictor.features, features)
    output = capsys.readouterr().out
    assert 'restored embeddings for frame None, z-slice 5 on cpu' in output
    assert 'skipping image encoder' in output
    assert 'SAM predictor state: is_image_set=True, features_shape=(1, 2).' in output
    assert 'running image encoder' not in output
    # A later uncached image must not accidentally use the restored state.
    wrapper._segment_2D_image(image, points, labels)
    assert predictor.encoder_calls == 1
    assert 'computing image/crop embeddings' in capsys.readouterr().out


@pytest.mark.parametrize('restored', [False, True])
@pytest.mark.parametrize('with_crop', [False, True])
def test_automatic_segmentation_reuses_full_image(
        sam_wrapper, capsys, restored, with_crop,
    ):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    generator = sam_wrapper.SamAutomaticMaskGenerator()
    generator.with_crop = with_crop
    predictor = sam_wrapper._EmbeddingPredictor()
    generator.predictor = predictor
    wrapper.model = generator
    features = np.full((1, 2), 42, dtype=np.float32)
    if restored:
        assert wrapper._get_img_embeddings(SimpleNamespace(
            getSamEmbeddings=lambda **kwargs: {
                'original_size': (8, 8), 'input_size': (8, 8),
                'features': features, 'is_image_set': True,
            },
        ), frame_i=3, z=5)
    result = wrapper._segment_2D_image(
        np.zeros((8, 8), dtype=np.uint8), None, None,
        embeddings_already_init=restored,
    )
    assert predictor.encoder_calls == int(not restored) + int(with_crop)
    assert predictor.is_image_set
    assert predictor.original_size == (8, 8)
    np.testing.assert_array_equal(
        predictor.features, features if restored else np.ones((1, 2))
    )
    np.testing.assert_array_equal(result, np.ones((8, 8)))
    output = capsys.readouterr().out
    assert 'using prepared full-image embeddings' in output
    assert ('computing image/crop embeddings for (8, 8)' in output) == (not restored)
    calls = predictor.encoder_calls
    wrapper._segment_2D_image(np.zeros((8, 8), dtype=np.uint8), None, None)
    assert predictor.encoder_calls == calls + 1 + int(with_crop)


def test_missing_embeddings_report_recomputation(sam_wrapper, capsys):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    wrapper.model = sam_wrapper._EmbeddingPredictor()
    assert not wrapper._get_img_embeddings(
        SimpleNamespace(getSamEmbeddings=lambda **kwargs: None), frame_i=3, z=5
    )
    assert 'frame 3, z-slice 5' in capsys.readouterr().out


@pytest.mark.parametrize('stack', [False, True])
@pytest.mark.parametrize('only_embeddings', [False, True])
def test_automatic_embedding_saving_encodes_full_image(
        sam_wrapper, stack, only_embeddings,
    ):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    wrapper.model = sam_wrapper.SamAutomaticMaskGenerator()
    wrapper.model.with_crop = True
    predictor = sam_wrapper._EmbeddingPredictor()
    wrapper.model.predictor = predictor
    wrapper._input_points_df = None
    stored = {}

    def store(model, frame_i, z):
        assert predictor.is_image_set
        assert predictor.original_size == (8, 8)
        stored[z] = predictor.features.copy()

    image = np.zeros((2, 8, 8) if stack else (8, 8), dtype=np.uint8)
    wrapper.segment(
        image, frame_i=0, start_z_slice=5,
        posData=SimpleNamespace(storeSamEmbeddings=store),
        save_embeddings=True, only_embeddings=only_embeddings,
        automatic_removal_of_background=False,
    )
    assert set(stored) == ({5, 6} if stack else {5})
    if only_embeddings:
        assert predictor.encoder_calls == 1
    else:
        assert predictor.encoder_calls == (4 if stack else 2)


def test_automatic_failure_restores_full_image(sam_wrapper, monkeypatch):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    wrapper.model = sam_wrapper.SamAutomaticMaskGenerator()
    predictor = sam_wrapper._EmbeddingPredictor()
    wrapper.model.predictor = predictor

    def fail_generate(self, image):
        self.predictor.set_image(image)
        self.predictor.reset_image()
        self.predictor.set_image(image[:4, :4])
        raise RuntimeError('crop failed')

    monkeypatch.setattr(sam_wrapper.SamAutomaticMaskGenerator, 'generate', fail_generate)
    with pytest.raises(RuntimeError, match='crop failed'):
        wrapper._segment_2D_image(np.zeros((8, 8), dtype=np.uint8), None, None)
    assert predictor.encoder_calls == 2
    assert predictor.is_image_set
    assert predictor.original_size == (8, 8)
    np.testing.assert_array_equal(predictor.features, np.ones((1, 2)))
    predictor._crop_embeddings = {}
    predictor.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    assert predictor.encoder_calls == 3


@pytest.mark.parametrize('automatic', [False, True])
def test_predictions_always_persist_and_reload_encoder_states(
        sam_wrapper, tmp_path, automatic, capsys,
    ):
    from cellacdc import load

    position = load.loadData.__new__(load.loadData)
    position.sam_embeddings_path = str(tmp_path / 'embeddings.pt')
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    predictor = sam_wrapper._EmbeddingPredictor()
    if automatic:
        wrapper.model = sam_wrapper.SamAutomaticMaskGenerator()
        wrapper.model.predictor = predictor
        wrapper.model.with_crop = True
        wrapper._input_points_df = None
    else:
        predictor = sam_wrapper._EmbeddingPredictor()
        wrapper.model = predictor
        wrapper._input_points_df = pd.DataFrame({'x': [2], 'y': [2], 'id': [1]})
        wrapper._get_input_points = lambda *args: (
            {1: np.array([[2, 2]])}, {1: np.array([1])},
        )
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    wrapper.segment(
        image, frame_i=0, posData=position, save_embeddings=False,
        automatic_removal_of_background=False,
    )
    assert Path(position.sam_embeddings_path).exists()
    saved = sam_embeddings.load(position.sam_embeddings_path)
    full = saved[0][0]
    assert full['original_size'] == (8, 8)
    if automatic:
        assert len(full['crops']) == 1
        crop = next(iter(full['crops'].values()))
        assert crop['original_size'] == (4, 4)
        np.testing.assert_array_equal(crop['features'], np.full((1, 2), 2))
    calls = predictor.encoder_calls
    # Reload from disk after SAM's regular reset; neither encoder should run.
    predictor.reset_image()
    assert predictor.features is None
    del position.sam_embeddings
    capsys.readouterr()
    wrapper.segment(
        image, frame_i=0, posData=position, use_loaded_embeddings=True,
        automatic_removal_of_background=False,
    )
    assert predictor.encoder_calls == calls
    output = capsys.readouterr().out
    assert 'restored embeddings for frame 0, z-slice 0' in output
    if automatic:
        assert 'using cached image/crop embeddings for (4, 4)' in output
    np.testing.assert_array_equal(predictor.features, full['features'])


def test_crop_cache_matches_pixels_not_just_size(sam_wrapper):
    predictor = sam_wrapper._EmbeddingPredictor()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    predictor.set_image(image)
    features = predictor.features
    predictor.reset_image()
    predictor.set_image(image.copy())
    assert predictor.encoder_calls == 1
    np.testing.assert_array_equal(predictor.features, features)
    predictor.reset_image()
    predictor.set_image(np.ones_like(image))
    assert predictor.encoder_calls == 2


def test_loaded_only_embeddings_reports_reuse(sam_wrapper, tmp_path, capsys):
    from cellacdc import load

    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    wrapper.model = sam_wrapper._EmbeddingPredictor()
    wrapper._input_points_df = None
    position = load.loadData.__new__(load.loadData)
    position.sam_embeddings_path = str(tmp_path / 'embeddings.pt')
    position.sam_embeddings = {0: {0: {
        'original_size': (8, 8), 'input_size': (8, 8),
        'features': np.ones((1, 2)), 'is_image_set': True,
    }}}
    wrapper.segment(
        np.zeros((8, 8), dtype=np.uint8), frame_i=0, posData=position,
        only_embeddings=True, use_loaded_embeddings=True,
        automatic_removal_of_background=False,
    )
    assert wrapper.model.encoder_calls == 0
    output = capsys.readouterr().out
    assert 'using prepared full-image embeddings; skipping image encoder' in output
    assert 'computing image/crop embeddings' not in output


@pytest.mark.parametrize('automatic', [False, True])
def test_model_uses_caching_predictor_in_both_modes(sam_wrapper, monkeypatch, automatic):
    namespace = sam_wrapper.Model.__init__.__globals__
    monkeypatch.setitem(namespace, 'os', os)
    monkeypatch.setitem(namespace, 'sam_models_path', '')
    monkeypatch.setitem(namespace, 'model_types', {'test': ('test', 'weights')})
    monkeypatch.setitem(namespace, 'sam_model_registry', {
        'test': lambda **kwargs: SimpleNamespace(to=lambda **kwargs: None),
    })
    monkeypatch.setattr(
        sam_wrapper.SamAutomaticMaskGenerator, '__init__', lambda *args, **kwargs: None,
    )
    points = None if automatic else pd.DataFrame({'x': [2], 'y': [2], 'id': [1]})
    wrapper = sam_wrapper.Model(model_type='test', input_points_df=points)
    predictor = wrapper.model.predictor if automatic else wrapper.model
    assert isinstance(predictor, sam_wrapper._EmbeddingPredictor)
    if not automatic:
        assert not hasattr(predictor, 'predictor')


@pytest.mark.parametrize('automatic', [False, True])
def test_init_embeddings_delegates_caching_and_logging(sam_wrapper, capsys, automatic):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    predictor = sam_wrapper._EmbeddingPredictor()
    wrapper.model = SimpleNamespace(predictor=predictor) if automatic else predictor
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    wrapper._init_embeddings(image)
    assert predictor.encoder_calls == 1
    assert 'computing image/crop embeddings' in capsys.readouterr().out
    predictor.reset_image()
    wrapper._init_embeddings(image.copy())
    assert predictor.encoder_calls == 1
    assert predictor.is_image_set
    output = capsys.readouterr().out
    assert 'skipping image encoder' in output
    assert 'computing image/crop embeddings' not in output
    changed = image.copy()
    changed[0, 0] += 1
    wrapper._init_embeddings(changed)
    assert predictor.encoder_calls == 2
    assert 'computing image/crop embeddings' in capsys.readouterr().out


@pytest.mark.parametrize('load_from_disk', [False, True])
def test_parameter_dialog_enables_embedding_reuse(monkeypatch, tmp_path, load_from_disk):
    from unittest.mock import Mock
    from cellacdc import prompts

    path = tmp_path / 'embeddings.pt'
    path.touch()
    position = SimpleNamespace(
        sam_embeddings_path=str(path), images_path=str(tmp_path),
        basename='image', chNames=[], metadata_df=None,
        loadSamEmbeddings=Mock(),
    )
    if not load_from_disk:
        position.sam_embeddings = {}
    window = SimpleNamespace(
        model_kwargs={}, cancel=False, setChannelNames=Mock(), exec_=Mock(),
    )
    monkeypatch.setattr(prompts, 'askSamLoadEmbeddings', lambda *a, **kw: (True, False, False))
    monkeypatch.setattr(prompts.apps, 'QDialogModelParams', lambda *a, **kw: window)
    save_prompt = Mock(side_effect=AssertionError('Already using saved embeddings'))
    monkeypatch.setattr(prompts, 'askSamSaveEmbeddings', save_prompt)
    prompts.init_segm_model_params(position, 'segment_anything', [], [])
    assert window.model_kwargs['use_loaded_embeddings'] is True
    position.loadSamEmbeddings.assert_called_once()


@pytest.mark.parametrize('option,value,expected', [
    (None, None, False),
    ('applied_postprocessing', 'True', True),
    ('applyPostProcessing', 'True', True),
])
def test_postprocess_settings_accept_missing_legacy_option(option, value, expected):
    from unittest.mock import Mock
    from cellacdc import apps, config

    settings = config.ConfigParser()
    settings.add_section('segment_anything.postprocess')
    if option is not None:
        settings.set('segment_anything.postprocess', option, value)
    group = SimpleNamespace(restoreFromKwargs=Mock(), setChecked=Mock())
    window = SimpleNamespace(
        model_name='segment_anything', postProcessGroupbox=group,
        configPars=settings,
    )
    apps.QDialogModelParams.loadLastSelectionPostProcess(window)
    group.setChecked.assert_called_once_with(expected)
