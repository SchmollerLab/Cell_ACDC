import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cellacdc import _types, myutils


@pytest.fixture
def sam_wrapper(monkeypatch):
    import cv2
    import sys

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
        'np': np, 'cv2': cv2, 'myutils': myutils, '_types': _types,
        'model_types': {}, 'SamPredictor': Predictor,
        'SamAutomaticMaskGenerator': Generator,
    }
    exec(compile(classes, str(source), 'exec'), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize('automatic', [False, True])
def test_restored_embeddings_skip_image_encoder(sam_wrapper, automatic, capsys):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    if automatic:
        wrapper.model = sam_wrapper._EmbeddingMaskGenerator()
        predictor = sam_wrapper._EmbeddingPredictor()
        predictor._full_image_embedding = None
        wrapper.model.predictor = predictor
    else:
        predictor = sam_wrapper.SamPredictor()
        wrapper.model = predictor
    features = np.full((1, 2), 42, dtype=np.float32)
    position = SimpleNamespace(getSamEmbeddings=lambda **kwargs: {
        'original_size': (8, 8), 'input_size': (8, 8),
        'features': features, 'is_image_set': True,
    })
    restored = wrapper._get_img_embeddings(position, frame_i=None, z=5)
    assert restored
    assert predictor.features is features
    points = None if automatic else {1: np.array([[2, 2]])}
    labels = None if automatic else {1: np.array([1])}
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
    assert 'running image encoder' in capsys.readouterr().out


def test_automatic_crops_encode_but_preserve_full_image_embedding(sam_wrapper, capsys):
    generator = sam_wrapper._EmbeddingMaskGenerator()
    generator.with_crop = True
    predictor = sam_wrapper._EmbeddingPredictor()
    generator.predictor = predictor
    generator.generate(np.zeros((8, 8, 3), dtype=np.uint8))
    assert predictor.encoder_calls == 2
    assert predictor.original_size == (8, 8)
    np.testing.assert_array_equal(predictor.features, np.ones((1, 2)))
    capsys.readouterr()
    predictor._reuse_embedding = True
    generator.generate(np.zeros((8, 8, 3), dtype=np.uint8))
    assert predictor.encoder_calls == 3
    assert predictor.original_size == (8, 8)
    np.testing.assert_array_equal(predictor.features, np.ones((1, 2)))
    output = capsys.readouterr().out
    assert output.count('skipping image encoder') == 1
    assert output.count('running image encoder') == 1
    assert 'image/crop (4, 4)' in output


def test_missing_embeddings_report_recomputation(sam_wrapper, capsys):
    wrapper = sam_wrapper.Model.__new__(sam_wrapper.Model)
    assert not wrapper._get_img_embeddings(
        SimpleNamespace(getSamEmbeddings=lambda **kwargs: None), frame_i=3, z=5
    )
    assert 'frame 3, z-slice 5' in capsys.readouterr().out


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
