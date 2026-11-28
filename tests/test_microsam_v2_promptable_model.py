"""micro-sam v2 promptable model wrapper (SAM2 models of micro-sam), tested
with a fake `micro_sam` package.

micro-sam is not needed: a fake package with the SAM2 API of micro-sam 2
(`micro_sam.v2`) that records the calls is registered in `sys.modules`.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

import cellacdc

pytestmark = pytest.mark.models

MODULE_PATH = os.path.join(
    os.path.dirname(cellacdc.__file__), 'promptable_models', 'micro-sam_v2',
    'acdcPromptSegment.py'
)


def _load_module():
    """Load `acdcPromptSegment.py` by path (importing the package would run
    `myutils.check_install_microsam`)."""
    spec = importlib.util.spec_from_file_location(
        '_acdc_test_microsam_v2_prompt_segment', MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _square(center, shape):
    """Boolean mask of a square of +-3 pixels around `center` (y, x)."""
    mask = np.zeros(shape, bool)
    yc, xc = [int(round(c)) for c in center]
    mask[max(yc-3, 0):yc+3, max(xc-3, 0):xc+3] = True
    return mask


class FakeEmbeddings(dict):
    """`micro_sam.v2.util.ImageEmbeddings`"""

    def __init__(self, shape, ndim):
        super().__init__(shape=shape, ndim=ndim)
        self.closed = False

    def close(self):
        self.closed = True


class FakeImagePredictor:
    """SAM2 image predictor: points --> square around the mean of the
    positive points; box --> the box."""

    def __init__(self, fake):
        self.fake = fake
        self.embeddings = None

    def predict(
            self, point_coords=None, point_labels=None, box=None,
            mask_input=None, multimask_output=True, return_logits=False,
            normalize_coords=True):
        assert self.embeddings is not None and not self.embeddings.closed
        self.fake.calls.append((
            'predict',
            None if point_coords is None else np.array(point_coords),
            None if point_labels is None else np.array(point_labels),
            None if box is None else np.array(box),
            multimask_output,
        ))
        shape = self.embeddings['shape'][:2]
        if box is not None:
            x0, y0, x1, y1 = box
            mask = np.zeros(shape, bool)
            mask[y0:y1, x0:x1] = True
        else:
            positive = np.asarray(point_coords)[np.asarray(point_labels) == 1]
            xc, yc = positive.mean(axis=0)
            mask = _square((yc, xc), shape)
        return mask[None].astype(np.float32), np.ones(1), np.zeros(1)


class FakeVideoPredictor:
    """SAM2 video predictor (from `get_sam2_model(input_type='videos')`)"""


class FakeMicroSam2:
    """Fake modules of micro-sam 2 (`micro_sam.util` and `micro_sam.v2`).

    `PromptableSegmentation3D.propagate_prompts` copies the mask of the
    first prompt set of each object (box or square around the positive
    points) to all the z-slices, padded to a square as the video predictor
    does.
    """

    def __init__(self):
        self.calls = []
        self.embeddings_calls = []
        self.segmenters = []
        self.propagate_error = None

    def install(self, monkeypatch):
        for name, module in self.modules().items():
            monkeypatch.setitem(sys.modules, name, module)

    def segmenter_class(self):
        fake = self

        class PromptableSegmentation3D:
            def __init__(
                    self, predictor, volume, volume_embeddings, device=None,
                    **kwargs):
                assert isinstance(predictor, FakeVideoPredictor)
                assert volume_embeddings['ndim'] == 3
                self.volume = volume
                self.volume_embeddings = volume_embeddings
                self.device = device
                self.prompt_sets = []
                self.calls = []
                fake.segmenters.append(self)

            def add_prompt_set(
                    self, frame_id, points=None, point_labels=None, box=None,
                    object_id=1, clear_old_points=True):
                self.prompt_sets.append(dict(
                    frame_id=frame_id,
                    points=None if points is None else np.array(points),
                    point_labels=(
                        None if point_labels is None
                        else np.array(point_labels)
                    ),
                    box=None if box is None else np.array(box),
                    object_id=object_id,
                ))

            def propagate_prompts(self, **kwargs):
                self.calls.append('propagate')
                if fake.propagate_error is not None:
                    raise fake.propagate_error
                Z, Y, X = self.volume.shape[:3]
                side = max(Y, X)
                masks = {}
                for prompt_set in self.prompt_sets:
                    object_id = prompt_set['object_id']
                    if object_id in masks:
                        continue
                    if prompt_set['box'] is not None:
                        y0, x0, y1, x1 = np.round(
                            prompt_set['box']
                        ).astype(int)
                        mask = np.zeros((side, side), bool)
                        mask[y0:y1, x0:x1] = True
                    else:
                        points = prompt_set['points']
                        labels = prompt_set['point_labels']
                        mask = _square(
                            points[labels == 1].mean(axis=0), (side, side)
                        )
                    masks[object_id] = mask[None]
                return {z: dict(masks) for z in range(Z)}

            def reset_tracking(self):
                self.calls.append('reset_tracking')
                self.prompt_sets = []

            def release(self):
                self.calls.append('release')

        return PromptableSegmentation3D

    def modules(self):
        fake = self

        top_util = types.ModuleType('micro_sam.util')
        top_util.get_device = lambda device=None: (
            'cuda' if device is None else device
        )

        util = types.ModuleType('micro_sam.v2.util')

        def get_sam2_model(
                model_type='hvit_t', device=None, checkpoint_path=None,
                input_type='images', peft_kwargs=None):
            fake.calls.append((
                'get_sam2_model', model_type, device, checkpoint_path,
                input_type
            ))
            if input_type == 'videos':
                return FakeVideoPredictor()
            return types.SimpleNamespace(model_type=model_type)
        util.get_sam2_model = get_sam2_model

        def get_sam2_image_predictor(model, **kwargs):
            fake.calls.append(('image_predictor', model.model_type))
            return FakeImagePredictor(fake)
        util.get_sam2_image_predictor = get_sam2_image_predictor

        def precompute_image_embeddings(
                predictor, input_, save_path=None, lazy_loading=False,
                ndim=None, verbose=True, **kwargs):
            fake.embeddings_calls.append((input_.shape, ndim, lazy_loading))
            return FakeEmbeddings(input_.shape, ndim)
        util.precompute_image_embeddings = precompute_image_embeddings

        def set_precomputed(predictor, image_embeddings, i=None, tile_id=None):
            assert image_embeddings['ndim'] == 2
            assert isinstance(predictor, FakeImagePredictor)
            predictor.embeddings = image_embeddings
            fake.calls.append(('set_precomputed', image_embeddings['shape']))
            return predictor
        util.set_precomputed = set_precomputed

        pbs = types.ModuleType('micro_sam.v2.prompt_based_segmentation')
        pbs.PromptableSegmentation3D = self.segmenter_class()

        package = types.ModuleType('micro_sam')
        package.__path__ = []
        package.util = top_util
        v2 = types.ModuleType('micro_sam.v2')
        v2.__path__ = []
        v2.util = util
        v2.prompt_based_segmentation = pbs
        package.v2 = v2
        return {
            'micro_sam': package, 'micro_sam.util': top_util,
            'micro_sam.v2': v2, 'micro_sam.v2.util': util,
            'micro_sam.v2.prompt_based_segmentation': pbs,
        }


def _make_model(monkeypatch, **init_kwargs):
    fake = FakeMicroSam2()
    fake.install(monkeypatch)
    module = _load_module()
    model = module.Model(**init_kwargs)
    fake.calls.clear()
    return fake, model


def _calls(fake, kind):
    return [call for call in fake.calls if call[0] == kind]


def test_model_arg_spec_exposes_gui_params():
    from cellacdc.myutils import model_modules

    had_micro_sam = 'micro_sam' in sys.modules
    module = _load_module()
    init_params, segment_params = model_modules.getModelArgSpec(module)
    # Inspecting the arguments does not import micro-sam
    assert ('micro_sam' in sys.modules) == had_micro_sam
    assert module.Model.prompt_types == ('point', 'box')

    assert [(p.name, p.default) for p in init_params] == [
        ('model_type', 'hvit_t_cells'), ('checkpoint_path', ''),
        ('device', 'auto'),
    ]
    assert [(p.name, p.default) for p in segment_params] == [
        ('lab', None), ('treat_other_objects_as_background', False),
        ('segment_entire_z_stack', True),
    ]
    kinds = {
        p.name: model_modules._serialize_model_param_type(p.type)
        for p in init_params + segment_params
    }
    assert kinds['lab'] == {'kind': 'hidden'}
    assert kinds['checkpoint_path'] == {'kind': 'path'}
    assert kinds['device']['values'] == ['auto', 'cpu', 'gpu']
    assert kinds['model_type']['values'][0] == 'hvit_t_cells'
    assert {
        'hvit_t_cells', 'hvit_s_cells', 'hvit_b_cells', 'hvit_l_cells',
        'hvit_s_em_mito', 'hvit_s_em_nuclei', 'hvit_t', 'hvit_l',
    } <= set(kinds['model_type']['values'])
    for name in ('treat_other_objects_as_background',
                 'segment_entire_z_stack'):
        assert kinds[name] == {'kind': 'bool'}

    # Every GUI parameter has its own description in the params dialog
    for param in init_params + segment_params:
        if param.name == 'lab':
            continue
        assert param.desc and ' : ' not in param.desc, param.name
        assert 'Default is' in param.desc, param.name

    assert 'micro-sam' in module.url_help()


@pytest.mark.parametrize('checkpoint_path', ['', 'None', None])
def test_init_loads_the_image_model(monkeypatch, checkpoint_path):
    fake = FakeMicroSam2()
    fake.install(monkeypatch)
    module = _load_module()

    model = module.Model(
        model_type='hvit_b_cells', checkpoint_path=checkpoint_path,
        device='auto'
    )
    # The video predictor is loaded only for the 3D propagation
    assert fake.calls == [
        ('get_sam2_model', 'hvit_b_cells', 'cuda', None, 'images'),
        ('image_predictor', 'hvit_b_cells'),
    ]
    assert model.device == 'cuda'

    fake.calls.clear()
    module.Model(checkpoint_path='weights.pt', device='cpu')
    assert fake.calls[0] == (
        'get_sam2_model', 'hvit_t_cells', 'cpu', 'weights.pt', 'images'
    )


@pytest.mark.parametrize('cuda, mps, expected', [
    (True, False, 'cuda'), (False, True, 'mps'), (False, False, None),
])
def test_init_gpu_device(monkeypatch, cuda, mps, expected):
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(
        is_available=lambda: mps, is_built=lambda: mps
    ))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    if expected is None:
        with pytest.raises(RuntimeError, match='no GPU is available'):
            _make_model(monkeypatch, device='gpu')
        return
    _, model = _make_model(monkeypatch, device='gpu')
    assert model.device == expected


def test_add_prompt_rejects_invalid_prompts(monkeypatch):
    _, model = _make_model(monkeypatch)
    with pytest.raises(ValueError, match='must belong to an object'):
        model.add_prompt((None, 1, 2, 5, 6), 0, prompt_type='box')
    with pytest.raises(ValueError, match='not supported by micro-sam v2'):
        model.add_prompt((1, 2, 3), 1, prompt_type='brush_stroke')
    assert not model.prompt_ids_image_mapper


def test_segment_2D_points_and_negatives(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((40, 50), np.uint16)
    model.add_prompt((None, 10, 10), 1)
    model.add_prompt((np.nan, 12, 14), 1)
    model.add_prompt((0, 30, 30), 2)
    model.add_prompt((None, 1, 1), 0)  # negative, all objects
    model.add_prompt((None, 2, 2), 0, parent_obj_id=2)  # only object 2

    lab = model.segment(image, lab=np.zeros((40, 50), np.uint32))
    assert lab.shape == (40, 50) and lab.dtype == np.uint32
    assert set(np.unique(lab)) == {0, 1, 2}
    assert (lab[8:14, 9:15] == 1).all()
    assert (lab[27:33, 27:33] == 2).all()

    (_, coords1, labels1, box1, multi1), (_, coords2, labels2, _, _) = (
        _calls(fake, 'predict')
    )
    # (x, y) coordinates of SAM2, positive points first, one mask
    assert coords1.tolist() == [[10, 10], [14, 12], [1, 1]]
    assert labels1.tolist() == [1, 1, 0]
    assert coords2.tolist() == [[30, 30], [1, 1], [2, 2]]
    assert labels2.tolist() == [1, 0, 0]
    assert box1 is None and multi1 is False
    # One embedding computation for the image, set before each prediction
    assert fake.embeddings_calls == [((40, 50), 2, False)]
    assert [call[0] for call in fake.calls] == [
        'set_precomputed', 'predict', 'set_precomputed', 'predict'
    ]
    # Prompts are reset after segmenting
    assert not model.prompt_ids_image_mapper and not model.points
    assert not model.negative_points and not model.boxes


def test_treat_other_objects_as_background(monkeypatch):
    fake, model = _make_model(monkeypatch)
    model.add_prompt((None, 10, 10), 1)
    model.add_prompt((None, 30, 20), 2)
    model.segment(np.zeros((40, 50)), treat_other_objects_as_background=True)
    (_, coords1, labels1, _, _), (_, coords2, labels2, _, _) = _calls(
        fake, 'predict'
    )
    assert coords1.tolist() == [[10, 10], [20, 30]]
    assert labels1.tolist() == [1, 0]
    assert coords2.tolist() == [[20, 30], [10, 10]]
    assert labels2.tolist() == [1, 0]


def test_segment_2D_boxes(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((40, 50))
    # Box only, coordinates sorted
    model.add_prompt((None, 20, 15, 5, 3), 1, prompt_type='box')
    # Box + point + negative point (only for object 2), box clipped to the
    # image
    model.add_prompt((None, 25, 30, 45, 55), 2, prompt_type='box')
    model.add_prompt((None, 30, 40), 2)
    model.add_prompt((None, 0, 0), 0, parent_obj_id=2)
    # Two boxes of the same object --> bounding box
    model.add_prompt((None, 0, 20, 4, 24), 3, prompt_type='box')
    model.add_prompt((None, 2, 26, 6, 30), 3, prompt_type='box')

    lab = model.segment(image)
    (_, coords1, _, box1, _), (_, coords2, labels2, box2, _), (
        _, coords3, _, box3, _) = _calls(fake, 'predict')
    # (x_min, y_min, x_max, y_max) boxes of SAM2
    assert box1.tolist() == [3, 5, 15, 20] and coords1 is None
    assert box2.tolist() == [30, 25, 50, 40]
    assert coords2.tolist() == [[40, 30], [0, 0]]
    assert labels2.tolist() == [1, 0]
    assert box3.tolist() == [20, 0, 30, 6] and coords3 is None
    assert (lab[5:20, 3:15] == 1).all() and (lab[25:40, 30:50] == 2).all()
    assert (lab[0:6, 20:30] == 3).all()


def test_objects_without_positive_prompts_are_skipped(monkeypatch):
    fake, model = _make_model(monkeypatch)
    model.add_prompt((None, 1, 1), 0)
    model.add_prompt((None, 1, 1), 0, parent_obj_id=5)
    lab = model.segment(np.zeros((10, 10)))
    assert not lab.any() and not fake.calls and not fake.embeddings_calls

    # 3D: the embeddings of the z-stack are not computed
    model.add_prompt((1, 1, 1), 0)
    lab = model.segment(np.zeros((3, 10, 10)), lab=np.zeros((3, 10, 10)))
    assert not lab.any() and not fake.calls and not fake.embeddings_calls
    assert not fake.segmenters


def test_embeddings_are_cached(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.arange(400, dtype=np.uint16).reshape(20, 20)
    for _ in range(2):
        model.add_prompt((None, 5, 5), 1)
        model.add_prompt((None, 10, 10), 2)
        model.segment(image.copy())
    assert fake.embeddings_calls == [((20, 20), 2, False)]

    image[0, 0] += 1
    model.add_prompt((None, 5, 5), 1)
    model.segment(image)
    assert fake.embeddings_calls == [((20, 20), 2, False)]*2


def test_zoom_prompts_on_a_crop(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((40, 50))
    crop = image[10:30, 20:50]
    model.add_prompt(
        (None, 2, 3, 8, 13), 4, prompt_type='box', image=crop,
        image_origin=(0, 10, 20)
    )
    lab = model.segment(image)
    assert fake.embeddings_calls == [((20, 30), 2, False)]
    assert (lab[12:18, 23:33] == 4).all() and (lab == 4).sum() == 60


def test_rgb_images(monkeypatch):
    fake, model = _make_model(monkeypatch)
    rgb = np.zeros((20, 30, 3), np.uint8)
    model.add_prompt((None, 10, 10), 1)
    lab = model.segment(rgb, lab=np.zeros((20, 30), np.uint32))
    assert lab.shape == (20, 30) and lab.any()
    assert fake.embeddings_calls == [((20, 30, 3), 2, False)]


def test_segment_z_stack_with_propagation(monkeypatch):
    fake, model = _make_model(monkeypatch)
    # Not square: the masks of the video predictor are cropped
    image = np.zeros((6, 20, 30))
    for _ in range(2):
        model.add_prompt((2, 10, 10), 1)
        model.add_prompt((4, 2, 3, 8, 13), 1, prompt_type='box')
        model.add_prompt((3, 0, 0), 0)  # negative on z-slice 3 (no prompts)
        model.add_prompt((4, 15, 25), 2)
        lab = model.segment(image, lab=np.zeros((6, 20, 30), np.uint32))

    # Video predictor loaded once, volume embeddings computed once (stored
    # in a temporary folder) and the same segmenter used for both calls
    assert _calls(fake, 'get_sam2_model') == [
        ('get_sam2_model', 'hvit_t_cells', 'cuda', None, 'videos')
    ]
    assert fake.embeddings_calls == [((6, 20, 30), 3, True)]
    segmenter, = fake.segmenters
    assert segmenter.device == 'cuda'
    # The prompts of all the objects are propagated together, then reset
    assert segmenter.calls == ['propagate', 'reset_tracking']*2
    assert not _calls(fake, 'predict')

    assert lab.shape == (6, 20, 30)
    # Propagated to all the z-slices by the fake
    for z in range(6):
        assert (lab[z, 7:13, 7:13] == 1).all()
        assert (lab[z, 12:18, 22:28] == 2).all()


def test_z_stack_prompt_sets(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((6, 20, 30))
    model.add_prompt((2, 10, 10), 1)
    model.add_prompt((4, 2, 3, 8, 13), 1, prompt_type='box')
    model.add_prompt((3, 0, 0), 0)  # negative on z-slice 3 (no prompts)
    model.add_prompt((4, 1, 1), 0, parent_obj_id=2)
    model.add_prompt((4, 15, 25), 2)
    prompt_sets = []
    original = fake.segmenter_class().add_prompt_set

    def add_prompt_set(self, **kwargs):
        prompt_sets.append(kwargs)
        return original(self, **kwargs)
    pbs = sys.modules['micro_sam.v2.prompt_based_segmentation']
    monkeypatch.setattr(
        pbs.PromptableSegmentation3D, 'add_prompt_set', add_prompt_set
    )
    model.segment(image, lab=np.zeros((6, 20, 30), np.uint32))

    set1_z2, set1_z4, set2_z4 = prompt_sets
    # (y, x) points and (y_min, x_min, y_max, x_max) box (micro-sam
    # converts them for SAM2)
    assert set1_z2['frame_id'] == 2 and set1_z2['object_id'] == 1
    assert set1_z2['points'].tolist() == [[10, 10]]
    assert set1_z2['point_labels'].tolist() == [1]
    assert set1_z2['box'] is None
    assert set1_z4['frame_id'] == 4 and set1_z4['object_id'] == 1
    assert set1_z4['points'] is None and set1_z4['point_labels'] is None
    assert set1_z4['box'].tolist() == [2, 3, 8, 13]
    assert set2_z4['frame_id'] == 4 and set2_z4['object_id'] == 2
    assert set2_z4['points'].tolist() == [[15, 25], [1, 1]]
    assert set2_z4['point_labels'].tolist() == [1, 0]


def test_volume_embeddings_are_released_for_a_new_image(monkeypatch):
    fake, model = _make_model(monkeypatch)
    volume = np.zeros((4, 20, 20))
    model.add_prompt((1, 5, 5), 1)
    model.segment(volume, lab=np.zeros(volume.shape, np.uint32))
    segmenter, = fake.segmenters
    embeddings = segmenter.volume_embeddings
    assert not embeddings.closed

    model.add_prompt((None, 5, 5), 1)
    model.segment(np.ones((20, 20)))
    # Removes the temporary folder of the embeddings of the z-stack
    assert embeddings.closed and segmenter.calls[-1] == 'release'

    model.add_prompt((1, 5, 5), 1)
    model.segment(volume, lab=np.zeros(volume.shape, np.uint32))
    assert len(fake.segmenters) == 2
    assert fake.embeddings_calls[-1] == ((4, 20, 20), 3, True)


def test_segment_z_stack_without_propagation(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((6, 20, 30))
    model.add_prompt((2, 10, 10), 1)
    model.add_prompt((4, 2, 3, 8, 13), 1, prompt_type='box')
    lab = model.segment(
        image, lab=np.zeros((6, 20, 30), np.uint32),
        segment_entire_z_stack=False
    )
    # Only the embeddings of the z-slices with prompts, no video predictor
    assert fake.embeddings_calls == [((20, 30), 2, False)]*2
    assert not _calls(fake, 'get_sam2_model') and not fake.segmenters
    assert [z for z in range(6) if lab[z].any()] == [2, 4]
    assert (lab[4, 2:8, 3:13] == 1).all()


def test_2D_masks_of_z_stack_data(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((6, 20, 30))
    model.add_prompt((2, 2, 3, 8, 13), 1, prompt_type='box')
    model.add_prompt((4, 10, 15, 18, 25), 2, prompt_type='box')
    lab = model.segment(image, lab=np.zeros((20, 30), np.uint32))
    assert lab.shape == (20, 30)
    assert (lab[2:8, 3:13] == 1).all() and (lab[10:18, 15:25] == 2).all()
    # No propagation with 2D masks
    assert not fake.segmenters


def test_single_z_slice_is_not_propagated(monkeypatch):
    fake, model = _make_model(monkeypatch)
    model.add_prompt((0, 5, 5), 1)
    lab = model.segment(np.zeros((1, 20, 20)), lab=np.zeros((1, 20, 20)))
    assert lab[0].any() and not fake.segmenters


def test_prompts_and_tracking_are_reset_after_errors(monkeypatch):
    fake, model = _make_model(monkeypatch)
    volume = np.zeros((4, 20, 20))
    model.add_prompt((1, 5, 5), 1)
    fake.propagate_error = RuntimeError('propagation failed')
    with pytest.raises(RuntimeError, match='propagation failed'):
        model.segment(volume, lab=np.zeros(volume.shape, np.uint32))
    segmenter, = fake.segmenters
    assert segmenter.calls == ['propagate', 'reset_tracking']
    assert not model.points and not model.prompt_ids_image_mapper

    fake.propagate_error = None
    model.add_prompt((1, 5, 5), 1)
    lab = model.segment(volume, lab=np.zeros(volume.shape, np.uint32))
    assert lab.any() and len(fake.segmenters) == 1


def test_incompatible_labels_shape(monkeypatch):
    _, model = _make_model(monkeypatch)
    model.add_prompt((None, 5, 5), 1)
    with pytest.raises(ValueError, match='not compatible'):
        model.segment(np.zeros((10, 10)), lab=np.zeros((5, 5), np.uint32))


def test_uses_the_sam2_api_of_micro_sam_v2():
    """The SAM2 API is imported from `micro_sam.v2` (micro-sam >= 2.0),
    never from `micro_sam.v1` (SAM1)."""
    import ast

    with open(MODULE_PATH, encoding='utf-8') as file:
        tree = ast.parse(file.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    micro_sam_modules = {
        name for name in imported if name.split('.')[0] == 'micro_sam'
    }
    assert micro_sam_modules == {
        'micro_sam.util', 'micro_sam.v2',
        'micro_sam.v2.prompt_based_segmentation',
    }


def test_microsam_v2_environment_config():
    from cellacdc.myutils import installs
    from cellacdc.service import environment

    assert (
        environment.get_model_env_name('micro-sam_v2')
        == '_acdc_micro_sam_v2'
    )
    config = environment.load_model_env_config('micro-sam_v2')
    assert config['python'] == '3.12'
    assert config['torch'] is True and config['torch_version'] == '>=2.5,<3'
    # The first initialization downloads the weights
    assert config['init_timeout'] >= 1800
    # No MobileSAM (only for the SAM1 vit_t models of micro-sam v1)
    assert 'setup_steps' not in config
    defaults = dict(
        min_version='', max_version='', exact_version='',
        include_lower_version=False, include_higher_version=False,
    )
    spec, = installs._normalize_pkg_specs(
        config['packages'], '', '', defaults
    )
    assert spec['pkg_name'] == 'micro_sam'
    assert spec['import_pkg_name'] == 'micro_sam'
    # Installed with pip (official installation), minimum version inclusive
    installs._check_package_version_constraints(spec, '2.0.1')
    installs._check_package_version_constraints(spec, '2.1.0')
    # micro-sam 2.0.0 does not declare its `sam2` dependency: the first
    # `--only-binary=:all:` pip attempt of Cell-ACDC installed it without
    # `sam2` (which has only an sdist), and existing environments with it
    # must be repaired
    for version in ('2.0.0', '1.9.0'):
        with pytest.raises(ModuleNotFoundError):
            installs._check_package_version_constraints(spec, version)


def test_pip_retries_without_only_binary_for_sdist_dependencies(monkeypatch):
    """micro-sam >= 2.0.1 depends on `sam2`, which has only an sdist: the
    first `--only-binary=:all:` attempt fails and the installation must be
    retried with `--prefer-binary` (which builds `sam2`)."""
    import subprocess

    from cellacdc.myutils import installs

    commands = []

    def run_command_streaming(command, logger=print):
        commands.append(list(command))
        if '--only-binary=:all:' in command:
            # pip: "No matching distribution found for sam2==1.1.0"
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(
        installs, '_run_command_streaming', run_command_streaming
    )
    installs._install_pip_package(
        '"micro_sam>=2.0.1"', logger=lambda *args: None, python_exec='py'
    )
    assert commands == [
        ['py', '-m', 'pip', 'install', 'micro_sam>=2.0.1',
         '--only-binary=:all:'],
        ['py', '-m', 'pip', 'install', 'micro_sam>=2.0.1', '--prefer-binary'],
    ]


def test_package_import_requires_microsam_with_sam2(monkeypatch):
    """Importing the model package (worker, or in-process in the Cell-ACDC
    environment) checks micro-sam >= 2.0.1, like the environment config."""
    import runpy

    from cellacdc.myutils import installs, model_installs
    from cellacdc.service import environment

    calls = []
    monkeypatch.setattr(model_installs, 'check_install_torch', lambda: None)
    monkeypatch.setattr(
        installs, 'check_install_package',
        lambda *args, **kwargs: calls.append((args, kwargs))
    )
    runpy.run_path(os.path.join(os.path.dirname(MODULE_PATH), '__init__.py'))
    assert calls == [(
        ('micro_sam',),
        {'pypi_name': 'micro_sam', 'min_version': '2.0.1',
         'include_lower_version': True},
    )]
    config_spec, = environment.load_model_env_config('micro-sam_v2')[
        'packages'
    ]
    assert config_spec['min_version'] == calls[0][1]['min_version']


@pytest.mark.skipif(
    importlib.util.find_spec('micro_sam') is None,
    reason='micro-sam is not installed'
)
def test_real_microsam_v2_2D_and_3D():
    import skimage.draw

    module = _load_module()
    model = module.Model(model_type='hvit_t_cells', device='auto')
    rng = np.random.default_rng(0)
    image = rng.normal(100, 8, (128, 160))
    gt = np.zeros(image.shape, bool)
    rr, cc = skimage.draw.disk((60, 70), 20, shape=image.shape)
    image[rr, cc] += 120
    gt[rr, cc] = True
    for prompt, prompt_type in (
            ((None, 60, 70), 'point'), ((None, 36, 46, 84, 94), 'box')):
        model.add_prompt(prompt, 1, prompt_type=prompt_type)
        lab = model.segment(image.astype(np.uint16))
        iou = ((lab == 1) & gt).sum() / ((lab == 1) | gt).sum()
        assert iou > 0.8, (prompt_type, iou)

    # Propagation through a small non-square z-stack
    Z, Y, X = 8, 48, 64
    zz, yy, xx = np.mgrid[:Z, :Y, :X]
    sphere = (
        ((zz - 4) / 3.5)**2 + ((yy - 24) / 14)**2 + ((xx - 30) / 14)**2 <= 1
    )
    volume = rng.normal(100, 8, (Z, Y, X))
    volume[sphere] += 120
    model.add_prompt((4, 10, 16, 39, 45), 1, prompt_type='box')
    lab = model.segment(
        volume.astype(np.uint16), lab=np.zeros(volume.shape, np.uint32)
    )
    iou = ((lab == 1) & sphere).sum() / ((lab == 1) | sphere).sum()
    assert iou > 0.6, iou
