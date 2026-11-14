"""ConvPaint promptable model wrapper, tested with a fake `ConvpaintModel`.

napari-convpaint is not needed: `import_convpaint_model` is replaced by a
fake that records the calls.
"""
import importlib.util
import json
import os
import sys
import types
import warnings

import numpy as np
import pytest

import cellacdc

MODULE_PATH = os.path.join(
    os.path.dirname(cellacdc.__file__), 'promptable_models', 'ConvPaint',
    'acdcPromptSegment.py'
)

INIT_PARAMS = [
    'feature_extractor', 'normalization', 'image_downsample', 'tile_image',
    'classifier_iterations', 'classifier_learning_rate', 'classifier_depth',
    'device', 'train_classifier_on_gpu', 'verbose',
]
NORMALIZATIONS = ['Whole image (z-stack)', 'Each z-slice', 'None']


def _load_module():
    """Load `acdcPromptSegment.py` by path.

    Importing it as `cellacdc.promptable_models.ConvPaint.acdcPromptSegment`
    would run the package `__init__.py` (`myutils.check_install_convpaint`,
    which may ask to install packages).
    """
    spec = importlib.util.spec_from_file_location(
        '_acdc_test_convpaint_prompt_segment', MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeConvpaintModel:
    """Records the calls. Pixels > 0 are foreground (2), the rest
    background (1), like a trained ConvPaint model (no zeros).

    `save` writes the parameters as JSON to `<path>.pkl` (and `<path>.yml`)
    and `FakeConvpaintModel(model_path=...)` reads them back, like
    ConvPaint's own save/load.
    """
    FE_NAMES = {
        'vgg': 'vgg16', 'vgg-m': 'vgg16', 'gaussian': 'gaussian_features',
    }
    # Defaults of the feature extractors (`fe_model.get_default_params()`)
    FE_DEFAULTS = {
        'vgg16': {'fe_layers': ['l0'], 'fe_scalings': [1, 2, 4]},
        'gaussian_features': {'fe_layers': None, 'fe_scalings': [1]},
    }
    # Like ConvPaint's aliases: only some FE parameters are specified
    STD_MODELS = {
        'vgg': types.SimpleNamespace(fe_name='vgg16', fe_layers=None),
        'vgg-m': types.SimpleNamespace(
            fe_name='vgg16', fe_layers=['l0', 'l2'], fe_scalings=[1, 2]
        ),
        'gaussian': types.SimpleNamespace(fe_name='gaussian_features'),
    }

    def __init__(self, alias=None, model_path=None):
        self.alias = alias
        self.model_path = model_path
        self.params = {}
        self.set_params_calls = []
        self.locked_devices = {}
        self.train_calls = []
        self.segment_calls = []
        self.classifier = None
        self.train_error = None
        self.fe_name = self.FE_NAMES.get(alias, 'vgg16')
        defaults = self.FE_DEFAULTS[self.fe_name]
        self.fe_layers = defaults['fe_layers']
        self.fe_scalings = defaults['fe_scalings']
        self.fe_order = 0
        self.classes = None
        self.num_trainings = 0
        if model_path is not None:
            assert model_path.endswith('.pkl')
            self.model_path_content = open(model_path).read()
            data = json.loads(self.model_path_content)
            self.params = data['params']
            self.fe_name = data['fe_name']
            self.fe_layers = data['fe_layers']
            self.fe_scalings = data['fe_scalings']
            self.fe_order = data['fe_order']
            if data['trained']:
                self.classifier = types.SimpleNamespace()
                if data['classes'] is not None:
                    self.classifier.classes_ = np.array(data['classes'])
        self.fe_model = types.SimpleNamespace(
            get_default_params=self._fe_default_params
        )

    def _fe_default_params(self):
        defaults = self.FE_DEFAULTS.get(self.fe_name, {})
        return types.SimpleNamespace(
            fe_name=self.fe_name, fe_layers=defaults.get('fe_layers'),
            fe_scalings=defaults.get('fe_scalings', [1]), fe_order=0,
            fe_use_min_features=False,
        )

    def get_params(self):
        params = {
            'fe_name': self.fe_name, 'fe_layers': self.fe_layers,
            'fe_scalings': self.fe_scalings, 'fe_order': self.fe_order,
            'fe_use_min_features': False, 'tile_image': None,
            'clf_iterations': None, 'clf_learning_rate': None,
            'clf_depth': None, 'normalize': None, 'image_downsample': None,
            'channel_mode': 'single',
        }
        params.update(self.params)
        return types.SimpleNamespace(**params)

    def save(self, model_path, create_pkl=True, create_yml=True):
        data = {
            'params': self.params, 'fe_name': self.fe_name,
            'fe_layers': self.fe_layers, 'fe_scalings': self.fe_scalings,
            'fe_order': self.fe_order,
            'trained': self.classifier is not None,
            'classes': self.classes,
        }
        if create_pkl:
            with open(f'{model_path}.pkl', 'w') as file:
                json.dump(data, file)
        if create_yml:
            with open(f'{model_path}.yml', 'w') as file:
                json.dump(self.params, file)

    def set_params(self, param=None, ignore_warnings=False, **kwargs):
        self.set_params_calls.append(
            dict(kwargs, ignore_warnings=ignore_warnings)
        )
        self.params.update(kwargs)

    def set_param(self, key, val, ignore_warnings=False):
        self.params[key] = val

    def lock_device(self, device, part='both'):
        self.locked_devices[part] = device

    def train(
            self, image, annotations, memory_mode=False, img_ids=None,
            use_rf=False, allow_writing_files=False, in_channels=None,
            skip_norm=False, fe_use_device=None, clf_use_device=None
        ):
        print('CatBoost: 0: learn: 0.6')
        warnings.warn('ConvPaint training warning')
        self.train_calls.append({
            'images': list(image),
            'annotations': list(annotations),
            'skip_norm': skip_norm,
            'memory_mode': memory_mode,
        })
        # Like ConvPaint: the classifier is replaced before fitting it
        self.classifier = object()
        self.num_trainings = getattr(self, 'num_trainings', 0) + 1
        if self.train_error is not None:
            raise self.train_error
        # Like ConvPaint: the (not JSON-serializable) classifier is returned
        return self.classifier

    def segment(
            self, image, in_channels=None, skip_norm=False, use_dask=False,
            fe_use_device=None
        ):
        assert self.classifier is not None
        print('FE model is designed for imagenet normalization')
        self.segment_calls.append({
            'image': image,
            'skip_norm': skip_norm,
            'seg_smoothening': self.params.get('seg_smoothening'),
        })
        return np.where(image > 0, 2, 1).astype(np.uint8)


def _make_model(monkeypatch, **init_kwargs):
    module = _load_module()
    monkeypatch.setattr(
        module, 'import_convpaint_model', lambda: FakeConvpaintModel
    )
    return module, module.Model(**init_kwargs)


def _train(model, image=None):
    if image is None:
        image = np.ones((4, 4))
    annot = np.zeros(image.shape, np.uint8)
    annot[..., 0, 0] = 1
    annot[..., -1, -1] = 2
    model.add_prompt(annot, image=image, prompt_type='brush_stroke')
    return model.train()


def test_init_configures_convpaint(monkeypatch):
    module, model = _make_model(
        monkeypatch, feature_extractor='gaussian',
        normalization='Each z-slice', image_downsample=2, tile_image=True,
        classifier_iterations=50, classifier_learning_rate=0.2,
        classifier_depth=4, device='cpu', train_classifier_on_gpu=True,
        not_a_model_param=1,
    )
    convpaint = model.model
    assert convpaint.alias == 'gaussian'
    assert convpaint.set_params_calls == [{
        'ignore_warnings': True, 'image_downsample': 2, 'tile_image': True,
        'clf_iterations': 50, 'clf_learning_rate': 0.2, 'clf_depth': 4,
        'normalize': 3,
    }]
    assert convpaint.locked_devices == {'fe': 'cpu', 'clf': 'auto'}
    assert module.Model.prompt_types == ('brush_stroke',)
    assert model.prompt_types == ('brush_stroke',)
    assert not model.is_trained
    assert model.prompts == {}

    _, model = _make_model(monkeypatch)
    assert model.model.alias == 'vgg'
    assert model.model.params['normalize'] == 2
    assert model.model.locked_devices == {'fe': 'auto', 'clf': 'cpu'}
    _, model = _make_model(monkeypatch, normalization='None')
    assert model.model.params['normalize'] == 1

    for kwargs in (
            {'feature_extractor': 'vgg-s'}, {'normalization': 'each'},
            {'device': 'tpu'}, {'image_downsample': 21},
        ):
        with pytest.raises(ValueError):
            _make_model(monkeypatch, **kwargs)


def test_model_arg_spec_exposes_gui_params():
    from cellacdc import myutils_models

    had_convpaint = 'napari_convpaint' in sys.modules
    module = _load_module()
    init_params, segment_params = myutils_models.getModelArgSpec(module)
    # Inspecting the arguments does not import ConvPaint
    assert ('napari_convpaint' in sys.modules) == had_convpaint

    assert [p.name for p in init_params] == INIT_PARAMS
    assert [p.default for p in init_params] == [
        'vgg', 'Whole image (z-stack)', 1, False, 100, 0.1, 5, 'auto',
        False, False,
    ]
    assert [p.name for p in segment_params] == [
        'seg_smoothening', 'min_object_size'
    ]
    assert [p.default for p in segment_params] == [1, 0]
    assert not myutils_models.model_init_requires_segm_data(module)

    kinds = {
        p.name: myutils_models._serialize_model_param_type(p.type)
        for p in init_params + segment_params
    }
    assert kinds['feature_extractor'] == {
        'kind': 'choices',
        'values': ['vgg', 'vgg-m', 'vgg-l', 'gaussian', 'dinov2', 'dinov3'],
        'is_exclusive_with': [], 'default_exclusive': '',
    }
    assert kinds['normalization']['values'] == NORMALIZATIONS
    assert kinds['device']['values'] == ['auto', 'cpu', 'gpu']
    for name in ('image_downsample', 'classifier_iterations',
                 'classifier_depth', 'seg_smoothening', 'min_object_size'):
        assert kinds[name] == {'kind': 'int'}
    for name in ('tile_image', 'train_classifier_on_gpu', 'verbose'):
        assert kinds[name] == {'kind': 'bool'}
    assert kinds['classifier_learning_rate'] == {'kind': 'float'}

    # Every GUI parameter has its own description in the params dialog
    for param in init_params + segment_params:
        assert param.desc and ' : ' not in param.desc, param.name
        assert 'Default is' in param.desc, param.name

    assert module.url_help() == 'https://github.com/guiwitz/napari-convpaint'


def test_add_prompt_replaces_annotation_of_the_same_frame(monkeypatch):
    _, model = _make_model(monkeypatch)
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    annot = np.zeros((4, 4), np.int64)
    annot[0, 0] = 2
    model.add_prompt(
        annot, image=image, prompt_type='brush_stroke', frame_i=1, pos_i=0
    )
    replacement = annot.copy()
    replacement[3, 3] = 1
    # Same keywords as the GUI and the other promptable models
    model.add_prompt(
        prompt=replacement, prompt_id=0, image=image,
        image_origin=(0, 0, 0), prompt_type='brush_stroke', frame_i=1,
        pos_i=0, parent_obj_id=0,
    )
    assert list(model.prompts) == [(0, 1)]
    stored_image, stored_annot = model.prompts[(0, 1)]
    np.testing.assert_array_equal(stored_image, image)
    assert stored_annot.dtype == np.uint8
    np.testing.assert_array_equal(stored_annot, replacement)
    replacement[0, 1] = 2
    assert stored_annot[0, 1] == 0

    model.add_prompt(annot, image=image, frame_i=1, pos_i=1)
    model.add_prompt(annot, image=image, frame_i=2, pos_i=0)
    assert sorted(model.prompts) == [(0, 1), (0, 2), (1, 1)]


def test_add_prompt_rejects_invalid_prompts(monkeypatch):
    _, model = _make_model(monkeypatch)
    image = np.zeros((4, 4))
    annot = np.zeros((4, 4), np.uint8)

    with pytest.raises(ValueError, match="only supports 'brush_stroke'"):
        model.add_prompt((0, 1, 1), 1, image=image, prompt_type='point')
    with pytest.raises(ValueError, match=r'Invalid brush-stroke values \[3\]'):
        model.add_prompt(np.full((4, 4), 3), image=image)
    with pytest.raises(ValueError, match='Invalid brush-stroke values'):
        model.add_prompt(np.full((4, 4), 0.5), image=image)
    with pytest.raises(ValueError, match='Invalid brush-stroke values'):
        model.add_prompt(np.full((4, 4), -1), image=image)
    with pytest.raises(ValueError, match='RGB'):
        model.add_prompt(annot, image=np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match='shape'):
        model.add_prompt(annot, image=np.zeros((5, 4)))
    with pytest.raises(ValueError, match='image'):
        model.add_prompt(annot)
    with pytest.raises(ValueError, match='2D'):
        model.add_prompt(np.zeros(4), image=np.zeros(4))
    assert model.prompts == {}

    # Float annotations with valid values are accepted
    model.add_prompt(np.full((4, 4), 2.0), image=image)
    assert model.prompts[(0, 0)][1].dtype == np.uint8


def test_train_uses_only_annotated_planes(monkeypatch):
    _, model = _make_model(monkeypatch)
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 1000, size=(4, 6, 6)).astype(np.uint16)
    annot = np.zeros(stack.shape, np.uint8)
    annot[1, 0, :3] = 2
    annot[3, 5, :2] = 1
    image_2D = rng.integers(0, 10, size=(5, 7)).astype(np.uint8)
    annot_2D = np.zeros(image_2D.shape, np.uint8)
    annot_2D[2, 2] = 1
    annot_2D[4, 6] = 2
    model.add_prompt(annot, image=stack, frame_i=0)
    model.add_prompt(annot_2D, image=image_2D, frame_i=0, pos_i=1)
    # Strokes cleared on a frame: nothing to train on
    model.add_prompt(np.zeros_like(annot), image=stack, frame_i=5)

    summary = model.train()

    assert summary == {
        'num_images': 2, 'num_annotated_planes': 3,
        'num_foreground_pixels': 4, 'num_background_pixels': 3,
    }
    assert all(type(value) is int for value in summary.values())
    json.dumps(summary)
    call, = model.model.train_calls
    assert call['skip_norm'] is True
    assert call['memory_mode'] is False
    assert [img.shape for img in call['images']] == [(6, 6), (6, 6), (5, 7)]
    assert all(img.dtype == np.float32 for img in call['images'])
    assert all(a.dtype == np.uint8 for a in call['annotations'])
    np.testing.assert_array_equal(call['annotations'][0], annot[1])
    np.testing.assert_array_equal(call['annotations'][1], annot[3])
    np.testing.assert_array_equal(call['annotations'][2], annot_2D)
    # The whole z-stack is normalized before the planes are selected
    expected = (stack - stack.mean()) / stack.std()
    np.testing.assert_allclose(call['images'][0], expected[1], atol=1e-4)
    np.testing.assert_allclose(call['images'][1], expected[3], atol=1e-4)
    assert model.prompts == {}
    assert model.is_trained


def test_train_errors_clear_the_prompts(monkeypatch):
    _, model = _make_model(monkeypatch)
    image = np.ones((4, 4))
    with pytest.raises(ValueError, match='No brush strokes'):
        model.train()

    model.add_prompt(np.zeros((4, 4), np.uint8), image=image)
    with pytest.raises(ValueError, match='empty'):
        model.train()
    assert model.prompts == {}

    only_foreground = np.zeros((4, 4), np.uint8)
    only_foreground[0] = 2
    model.add_prompt(only_foreground, image=image)
    with pytest.raises(ValueError, match='no background'):
        model.train()
    assert model.prompts == {}

    only_background = np.zeros((4, 4), np.uint8)
    only_background[0] = 1
    model.add_prompt(only_background, image=image)
    with pytest.raises(ValueError, match='no foreground'):
        model.train()
    assert model.model.train_calls == []

    both = only_foreground.copy()
    both[3] = 1
    model.add_prompt(both, image=image)
    model.model.train_error = RuntimeError('CatBoost failed')
    with pytest.raises(RuntimeError, match='CatBoost failed'):
        model.train()
    assert model.prompts == {}
    assert not model.is_trained

    model.model.train_error = None
    model.add_prompt(both, image=image)
    model.train()
    assert model.is_trained


def test_failed_training_keeps_previous_classifier(monkeypatch, tmp_path):
    """E.g., re-training a loaded model fails in CatBoost: the loaded
    classifier must still be usable (ConvPaint replaces it before fitting)."""
    _, model = _make_model(monkeypatch)
    _train(model)
    saved = model.save_model(str(tmp_path / 'model.pkl'))
    model.load_model(saved)
    classifier = model.model.classifier
    num_trainings = getattr(model.model, 'num_trainings', None)

    model.model.train_error = RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='CUDA'):
        _train(model)
    assert model.model.classifier is classifier
    assert getattr(model.model, 'num_trainings', None) == num_trainings
    assert model.is_trained
    image = np.zeros((4, 4))
    image[1, 1] = 5
    assert model.segment(image).max() == 1


def test_percentile_normalization_for_percentile_feature_extractors(
        monkeypatch
    ):
    _, model = _make_model(monkeypatch, normalization='Each z-slice')
    model.model.fe_model.norm_mode = 'percentile'
    rng = np.random.default_rng(0)
    stack = rng.normal(100, 10, size=(2, 20, 20)).astype(np.float32)
    stack[1] *= 3
    normalized = model._normalize(stack)
    assert normalized.min() == 0 and normalized.max() == 1
    for z in range(2):
        low, high = np.percentile(stack[z], (1, 99))
        expected = np.clip((stack[z] - low) / (high - low), 0, 1)
        np.testing.assert_allclose(normalized[z], expected, atol=1e-5)

    model.normalization = 'Whole image (z-stack)'
    low, high = np.percentile(stack, (1, 99))
    np.testing.assert_allclose(
        model._normalize(stack),
        np.clip((stack - low) / (high - low), 0, 1), atol=1e-5
    )
    model.normalization = 'None'
    np.testing.assert_array_equal(model._normalize(stack), stack)


@pytest.mark.parametrize('verbose', [False, True])
def test_convpaint_output_is_hidden_unless_verbose(
        monkeypatch, capsys, verbose
    ):
    _, model = _make_model(monkeypatch, verbose=verbose)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _train(model)
        model.segment(np.ones((4, 4)))
    out = capsys.readouterr().out
    assert ('CatBoost: 0: learn' in out) is verbose
    assert ('imagenet normalization' in out) is verbose
    assert bool(caught) is verbose


def test_segment_returns_instance_labels(monkeypatch):
    _, model = _make_model(monkeypatch, normalization='None')
    with pytest.raises(ValueError, match='not trained yet'):
        model.segment(np.zeros((4, 4)))
    _train(model)

    image = np.zeros((6, 8), np.uint16)
    image[1:3, 1:3] = 5
    image[4, 6] = 7
    # Extra kwargs passed by the GUI/worker are ignored
    lab = model.segment(
        image, seg_smoothening=3, lab=np.zeros((6, 8), np.uint32), frame_i=2,
        posData=object(), start_z_slice=0,
    )
    assert lab.dtype == np.uint32
    assert lab.shape == image.shape
    assert lab.max() == 2
    assert set(np.unique(lab[image > 0])) == {1, 2}
    assert not lab[image == 0].any()
    call, = model.model.segment_calls
    assert call['skip_norm'] is True
    assert call['image'].dtype == np.float32
    assert call['seg_smoothening'] == 3

    lab = model.segment(image, min_object_size=2)
    np.testing.assert_array_equal(lab, (image == 5).astype(np.uint32))
    lab = model.segment(image, min_object_size=5)
    assert not lab.any()

    with pytest.raises(ValueError, match='RGB'):
        model.segment(np.zeros((6, 8, 3)), lab=np.zeros((6, 8)))
    with pytest.raises(ValueError, match='2D'):
        model.segment(np.zeros((2, 3, 4, 5)))
    with pytest.raises(ValueError, match='seg_smoothening'):
        model.segment(image, seg_smoothening=21)
    # The classifier is kept for the next images
    assert model.is_trained


def test_segment_z_stack_plane_by_plane(monkeypatch):
    _, model = _make_model(monkeypatch, normalization='None')
    _train(model)
    model.model.segment_calls.clear()

    stack = np.zeros((3, 6, 8), np.uint16)
    stack[0, 1, 1] = stack[1, 1, 1] = 5  # One object on two z-slices
    stack[2, 4, 4] = 5
    lab = model.segment(stack, lab=np.zeros(stack.shape, np.uint32))

    assert lab.shape == stack.shape
    assert lab.dtype == np.uint32
    shapes = [call['image'].shape for call in model.model.segment_calls]
    assert shapes == [(6, 8)] * 3
    assert lab[0, 1, 1] == lab[1, 1, 1] != 0
    assert lab[2, 4, 4] not in (0, lab[0, 1, 1])
    assert lab.max() == 2

    lab = model.segment(stack, min_object_size=2)
    assert lab.max() == 1
    assert lab[2, 4, 4] == 0
    assert lab[0, 1, 1] == lab[1, 1, 1] == 1


@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_normalization_modes(monkeypatch, normalization):
    _, model = _make_model(monkeypatch, normalization=normalization)
    rng = np.random.default_rng(1)
    stack = rng.normal(100, 10, size=(3, 5, 5))
    stack[1] += 50
    stack[2] = 7  # Constant z-slice
    annot = np.zeros(stack.shape, np.uint8)
    annot[1, 0, 0] = 1
    annot[1, 4, 4] = 2
    model.add_prompt(annot, image=stack)
    model.train()
    model.segment(stack)

    planes = np.stack([c['image'] for c in model.model.segment_calls])
    assert planes.dtype == np.float32
    assert np.isfinite(planes).all()
    if normalization == 'None':
        np.testing.assert_allclose(planes, stack.astype(np.float32))
    elif normalization == 'Whole image (z-stack)':
        expected = (stack - stack.mean()) / stack.std()
        np.testing.assert_allclose(planes, expected, atol=1e-4)
    else:
        np.testing.assert_allclose(planes[:2].mean(axis=(1, 2)), 0, atol=1e-4)
        np.testing.assert_allclose(planes[:2].std(axis=(1, 2)), 1, atol=1e-4)
        np.testing.assert_array_equal(planes[2], 0)
    # Training and segmentation are normalized the same way
    train_plane, = model.model.train_calls[0]['images']
    np.testing.assert_array_equal(train_plane, planes[1])


def test_save_model_requires_training(monkeypatch, tmp_path):
    _, model = _make_model(monkeypatch)
    with pytest.raises(ValueError, match='not trained'):
        model.save_model(str(tmp_path / 'model.pkl'))
    assert list(tmp_path.iterdir()) == []


def test_save_and_load_model(monkeypatch, tmp_path):
    from cellacdc.promptable_models import saved_models

    module, trained = _make_model(
        monkeypatch, feature_extractor='gaussian',
        normalization='Each z-slice', image_downsample=2,
        classifier_iterations=50, classifier_depth=4, device='gpu',
    )
    assert module.Model.saved_model_ext == '.pkl'
    assert saved_models.get_saved_model_ext(trained) == '.pkl'
    _train(trained)

    # Missing folders are created and the extension is added
    saved = trained.save_model(str(tmp_path / 'models' / 'my_model'))
    assert saved == str(tmp_path / 'models' / 'my_model.pkl')
    assert sorted(os.listdir(tmp_path / 'models')) == [
        'my_model.pkl', 'my_model.yml'
    ]
    # A path with the extension gives the same files
    assert trained.save_model(saved) == saved

    # A model with other parameters gets the parameters of the loaded one,
    # but keeps its device options
    _, model = _make_model(
        monkeypatch, feature_extractor='vgg', normalization='None',
        classifier_iterations=5, device='cpu', train_classifier_on_gpu=True,
    )
    initial_convpaint = model.model
    params = model.load_model(saved)
    assert params == {
        'feature_extractor': 'gaussian',
        'normalization': 'Each z-slice',
        'image_downsample': 2,
        'tile_image': False,
        'classifier_iterations': 50,
        'classifier_learning_rate': 0.1,
        'classifier_depth': 4,
        'fe_name': 'gaussian_features',
        'fe_layers': [],
        'classes': None,
        'is_trained': True,
    }
    assert params == model.get_model_params()
    assert model.model is not initial_convpaint
    # Loaded from a temporary copy (deleted afterwards)
    assert model.model.model_path != saved
    assert model.model.model_path.endswith('.pkl')
    assert not os.path.exists(model.model.model_path)
    with open(saved) as file:
        assert model.model.model_path_content == file.read()
    assert model.model.locked_devices == {'fe': 'cpu', 'clf': 'auto'}
    assert model.is_trained
    assert model.normalization == 'Each z-slice'
    assert model.image_downsample == 2

    # The loaded classifier segments without training
    image = np.zeros((6, 6))
    image[1:3, 1:3] = 5
    lab = model.segment(image)
    assert lab.max() == 1
    assert model.model.train_calls == []

    # A loaded model can be saved again
    assert model.save_model(str(tmp_path / 'again.pkl')) == str(
        tmp_path / 'again.pkl'
    )


@pytest.mark.parametrize('normalize, normalization', [
    (1, 'None'), (2, 'Whole image (z-stack)'), (3, 'Each z-slice'),
    (None, 'None'),
])
def test_load_model_normalization(
        monkeypatch, tmp_path, normalize, normalization
    ):
    _, trained = _make_model(monkeypatch)
    _train(trained)
    trained.model.params['normalize'] = normalize
    saved = trained.save_model(str(tmp_path / 'model.pkl'))

    _, model = _make_model(monkeypatch, normalization='Whole image (z-stack)')
    assert model.load_model(saved)['normalization'] == normalization
    assert model.normalization == normalization


def test_load_model_feature_extractor_alias(monkeypatch, tmp_path):
    _, model = _make_model(monkeypatch)
    _train(model)
    convpaint = model.model

    cases = (
        # Defaults of VGG16 (first layer)
        ('vgg16', ['l0'], [1, 2, 4], 0, 'vgg'),
        # All the parameters of 'vgg-m' match
        ('vgg16', ['l0', 'l2'], [1, 2], 0, 'vgg-m'),
        # Custom VGG16 models (e.g., trained in napari) are not 'vgg'
        ('vgg16', ['l0', 'l5'], [1, 2, 4], 0, None),
        ('vgg16', ['l0'], [1], 0, None),
        ('vgg16', ['l0'], [1, 2, 4], 1, None),
        ('gaussian_features', [], [1], 0, 'gaussian'),
        ('gaussian_features', [], [1, 2], 0, None),
        # E.g., a model trained in napari with another FE
        ('cellpose_backbone', [], [1], 0, None),
    )
    for i, (fe_name, fe_layers, fe_scalings, fe_order, alias) in (
            enumerate(cases)
        ):
        convpaint.fe_name = fe_name
        convpaint.fe_layers = fe_layers
        convpaint.fe_scalings = fe_scalings
        convpaint.fe_order = fe_order
        saved = model.save_model(str(tmp_path / f'model_{i}.pkl'))
        params = model.load_model(saved)
        assert params['feature_extractor'] == alias, cases[i]
        assert params['fe_name'] == fe_name
        convpaint = model.model


def test_load_model_copies_file_and_reports_classes(monkeypatch, tmp_path):
    _, model = _make_model(monkeypatch)
    _train(model)
    model.model.classes = [1, 2, 3]
    saved = model.save_model(str(tmp_path / 'model'))
    # Upper-case extension (ConvPaint itself only accepts '.pkl')
    upper = tmp_path / 'MODEL.PKL'
    os.replace(saved, upper)
    content = upper.read_text()
    mtime = os.path.getmtime(upper)

    _, other = _make_model(monkeypatch)
    params = other.load_model(str(upper))
    assert params['classes'] == [1, 2, 3]
    assert other.model.model_path.endswith('.pkl')
    assert other.is_trained
    # The loaded file is not modified
    assert upper.read_text() == content
    assert os.path.getmtime(upper) == mtime


def test_load_model_errors(monkeypatch, tmp_path):
    _, model = _make_model(monkeypatch)
    _train(model)
    saved = model.save_model(str(tmp_path / 'model.pkl'))

    _, other = _make_model(monkeypatch)
    initial_convpaint = other.model

    with pytest.raises(ValueError, match='.pkl'):
        other.load_model(str(tmp_path / 'model.yml'))
    with pytest.raises(FileNotFoundError):
        other.load_model(str(tmp_path / 'missing.pkl'))

    # Models without a trained classifier
    untrained_path = str(tmp_path / 'untrained')
    initial_convpaint.save(untrained_path)
    with pytest.raises(ValueError, match='trained classifier'):
        other.load_model(f'{untrained_path}.pkl')

    # RGB or multi-channel models
    model.model.params['channel_mode'] = 'rgb'
    rgb_path = model.save_model(str(tmp_path / 'rgb.pkl'))
    with pytest.raises(ValueError, match='single-channel'):
        other.load_model(rgb_path)

    # The current model is kept after errors
    assert other.model is initial_convpaint
    assert not other.is_trained

    # ...and a valid model can still be loaded
    other.load_model(saved)
    assert other.is_trained


def _real_convpaint_available():
    return all(
        importlib.util.find_spec(name) is not None
        for name in ('napari_convpaint', 'catboost', 'torch')
    )


@pytest.mark.skipif(
    not _real_convpaint_available(),
    reason='napari-convpaint (and its dependencies) not installed'
)
def test_real_convpaint_save_and_load(tmp_path):
    """Train, save, and load with the real napari-convpaint (Gaussian
    features, nothing to download)."""
    module = _load_module()
    rng = np.random.default_rng(0)
    image = rng.normal(0, 0.2, (64, 64)).astype(np.float32)
    image[10:30, 10:30] += 3
    image[40:55, 35:60] += 3
    annot = np.zeros(image.shape, np.uint8)
    annot[15:20, 15:20] = 2
    annot[45:50, 40:50] = 2
    annot[0:5, :] = 1
    annot[:, 62:] = 1

    trained = module.Model(
        feature_extractor='gaussian', normalization='Each z-slice',
        classifier_iterations=30, classifier_depth=4, device='cpu',
    )
    trained.add_prompt(annot, image=image)
    trained.train()
    lab = trained.segment(image)
    assert lab.max() == 2
    saved = trained.save_model(str(tmp_path / 'model.pkl'))
    assert (tmp_path / 'model.yml').exists()

    model = module.Model(
        feature_extractor='gaussian', normalization='None',
        classifier_iterations=5, device='cpu',
    )
    params = model.load_model(saved)
    assert params['feature_extractor'] == 'gaussian'
    assert params['normalization'] == 'Each z-slice'
    assert params['classifier_iterations'] == 30
    np.testing.assert_array_equal(model.segment(image), lab)


def _remove_napari_convpaint_modules():
    for name in list(sys.modules):
        if name == 'napari_convpaint' or name.startswith('napari_convpaint.'):
            sys.modules.pop(name)


def test_import_convpaint_model_skips_package_init(monkeypatch, tmp_path):
    module = _load_module()
    package = tmp_path / 'napari_convpaint'
    (package / 'feature_extractors').mkdir(parents=True)
    (package / '__init__.py').write_text(
        "raise RuntimeError('napari_convpaint/__init__.py must not run')\n"
    )
    (package / 'param.py').write_text('VALUE = 42\n')
    (package / 'convpaint_model.py').write_text(
        'import importlib\n'
        'from .param import VALUE\n'
        '\n'
        'class ConvpaintModel:\n'
        '    value = VALUE\n'
        '\n'
        '    @staticmethod\n'
        '    def import_fe(name):\n'
        '        return importlib.import_module(\n'
        "            f'napari_convpaint.feature_extractors.{name}'\n"
        '        )\n'
    )
    (package / 'feature_extractors' / '__init__.py').write_text(
        'from .gaussian import NAME\n'
    )
    (package / 'feature_extractors' / 'gaussian.py').write_text(
        "NAME = 'gaussian'\n"
    )
    for name in [m for m in sys.modules if m.startswith('napari_convpaint')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK')
    try:
        ConvpaintModel = module.import_convpaint_model()
        assert ConvpaintModel.value == 42
        assert ConvpaintModel.import_fe('gaussian').NAME == 'gaussian'
        assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '1'
        # The registered package module is reused
        assert module.import_convpaint_model() is ConvpaintModel
    finally:
        _remove_napari_convpaint_modules()

    # A failed import does not leave the empty package module behind
    (package / 'convpaint_model.py').write_text(
        'import a_missing_convpaint_dependency\n'
    )
    importlib.invalidate_caches()
    try:
        with pytest.raises(ModuleNotFoundError):
            module.import_convpaint_model()
        assert 'napari_convpaint' not in sys.modules
    finally:
        _remove_napari_convpaint_modules()


def test_import_convpaint_model_not_installed(monkeypatch):
    module = _load_module()
    for name in [m for m in sys.modules if m.startswith('napari_convpaint')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(
        module.importlib.util, 'find_spec', lambda name, *args: None
    )
    with pytest.raises(ImportError, match='napari-convpaint is not installed'):
        module.import_convpaint_model()
    assert 'napari_convpaint' not in sys.modules


def test_package_import_checks_installation_only(monkeypatch):
    from cellacdc import myutils, myutils_models

    calls = []
    monkeypatch.setattr(
        myutils, 'check_install_convpaint', lambda: calls.append(True)
    )
    for name in (
            'cellacdc.promptable_models.ConvPaint',
            'cellacdc.promptable_models.ConvPaint.acdcPromptSegment',
        ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    had_convpaint = 'napari_convpaint' in sys.modules

    module = myutils_models.import_promptable_segment_module('ConvPaint')

    assert calls == [True]
    assert module.Model.prompt_types == ('brush_stroke',)
    # ConvPaint itself is imported only when the model is created
    assert ('napari_convpaint' in sys.modules) == had_convpaint
    assert 'ConvPaint' in myutils.get_list_of_promptable_models()


@pytest.mark.parametrize('installed_version', [None, '1.1', '1.2.0'])
def test_check_install_convpaint_never_imports_napari_convpaint(
        monkeypatch, installed_version
    ):
    from cellacdc import myutils

    import importlib.machinery

    calls = []
    # Module registered as `napari_convpaint` while checking/installing it
    modules_during_check = []

    def fake_check_install_package(pkg_name, **kwargs):
        calls.append((pkg_name, kwargs))
        modules_during_check.append(sys.modules.get('napari_convpaint'))

    monkeypatch.setattr(
        myutils, 'check_install_torch', lambda: calls.append('torch')
    )
    monkeypatch.setattr(
        myutils, 'check_install_package', fake_check_install_package
    )
    monkeypatch.setattr(
        myutils, 'get_package_version', lambda name: installed_version
    )
    find_spec = importlib.util.find_spec
    # Package spec without a loader: if `__init__.py` were executed,
    # `module_from_spec`/import would fail
    fake_spec = importlib.machinery.ModuleSpec(
        'napari_convpaint', None, is_package=True
    )

    def fake_find_spec(name, *args, **kwargs):
        if name == 'napari_convpaint':
            return None if installed_version is None else fake_spec
        return find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)
    monkeypatch.delitem(sys.modules, 'napari_convpaint', raising=False)

    myutils.check_install_convpaint()

    assert calls[:2] == ['torch', (myutils._CONVPAINT_PACKAGES, {})]
    if installed_version == '1.2.0':
        assert len(calls) == 2
    else:
        pkg_name, kwargs = calls[2]
        assert pkg_name == 'napari-convpaint'
        assert kwargs['import_pkg_name'] == 'napari_convpaint'
        assert kwargs['install_dependencies'] is False
        assert kwargs['min_version'] == '1.2'
        assert kwargs['include_lower_version'] is True
        assert kwargs['upgrade'] is (installed_version is not None)
        # The version check of an outdated installation must import an
        # empty package module, not the real `__init__.py` (napari and Qt)
        stub = modules_during_check[-1]
        if installed_version is None:
            assert stub is None
        else:
            assert stub is not None and stub.__spec__ is fake_spec
    # Afterwards, the next import must load the (upgraded) package files
    assert 'napari_convpaint' not in sys.modules


def test_convpaint_environment_config(monkeypatch):
    import importlib.metadata
    from cellacdc import myutils
    from cellacdc.service import environment

    config = environment.load_model_env_config('ConvPaint')
    assert config['python'] == '3.12'
    assert config['torch'] is True
    # Same dependencies in the isolated and in the main environment
    assert config['packages'] == myutils._CONVPAINT_PACKAGES
    defaults = dict(
        min_version='', max_version='', exact_version='',
        include_lower_version=False, include_higher_version=False,
    )
    specs = {
        spec['pkg_name']: spec for spec in myutils._normalize_pkg_specs(
            config['packages'], '', '', defaults
        )
    }
    # napari and Qt are not installed in the model environment
    assert not {'napari', 'napari-convpaint', 'PyQt6'} & set(specs)
    assert specs['pyyaml']['import_pkg_name'] == 'yaml'
    assert specs['scikit-image']['import_pkg_name'] == 'skimage'
    # Minimum versions are inclusive
    myutils._check_package_version_constraints(specs['catboost'], '1.2.8')
    myutils._check_package_version_constraints(specs['timm'], '1.0.20')
    with pytest.raises(ModuleNotFoundError):
        myutils._check_package_version_constraints(specs['catboost'], '1.2.7')

    step, = config['setup_steps']
    assert step['commands'] == [
        ['-m', 'pip', 'install', '--no-deps', 'napari-convpaint>=1.2']
    ]
    find_spec = importlib.util.find_spec
    for spec_found, version, ok in (
            (True, '1.2.0', True), (True, '1.3.1.dev2', True),
            (True, '1.1.0', False), (False, '1.2.0', False),
        ):
        monkeypatch.setattr(
            importlib.util, 'find_spec',
            lambda name, *args, found=spec_found: (
                (object() if found else None) if name == 'napari_convpaint'
                else find_spec(name, *args)
            )
        )
        monkeypatch.setattr(
            importlib.metadata, 'version', lambda name, v=version: v
        )
        if ok:
            exec(step['check'], {})
        else:
            with pytest.raises(AssertionError):
                exec(step['check'], {})
