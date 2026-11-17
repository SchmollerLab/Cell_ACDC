"""micro-sam promptable model wrapper, tested with a fake `micro_sam` package.

micro-sam is not needed: a fake package that records the calls is
registered in `sys.modules`.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

import cellacdc

MODULE_PATH = os.path.join(
    os.path.dirname(cellacdc.__file__), 'promptable_models', 'micro-sam',
    'acdcPromptSegment.py'
)


def _load_module():
    """Load `acdcPromptSegment.py` by path (importing the package would run
    `myutils.check_install_microsam`)."""
    spec = importlib.util.spec_from_file_location(
        '_acdc_test_microsam_prompt_segment', MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMicroSam:
    """Fake `micro_sam` modules.

    Masks: points --> square of +-3 pixels around the mean of the positive
    points; boxes --> the box. `segment_mask_in_volume` copies the mask of
    the first segmented z-slice to all the z-slices.
    """
    def __init__(self):
        self.calls = []
        self.embeddings_calls = []

    def _shape(self, image_embeddings, i):
        shape = image_embeddings['shape']
        if i is None:
            assert image_embeddings['ndim'] == 2
            return shape[:2]
        assert image_embeddings['ndim'] == 3
        return shape[1:3]

    def _box_mask(self, box, shape):
        mask = np.zeros((1, *shape), bool)
        y0, x0, y1, x1 = [int(round(c)) for c in box]
        mask[0, y0:y1, x0:x1] = True
        return mask

    def modules(self):
        fake = self

        util = types.ModuleType('micro_sam.util')
        util.get_device = lambda device=None: (
            'cuda' if device is None else device
        )

        def get_sam_model(model_type, device=None, checkpoint_path=None):
            fake.calls.append(
                ('get_sam_model', model_type, device, checkpoint_path)
            )
            return types.SimpleNamespace(model_type=model_type)
        util.get_sam_model = get_sam_model

        def precompute_image_embeddings(
                predictor, image, ndim=None, verbose=True):
            fake.embeddings_calls.append((image.shape, ndim))
            return {'shape': image.shape, 'ndim': ndim}
        util.precompute_image_embeddings = precompute_image_embeddings

        pbs = types.ModuleType('micro_sam.prompt_based_segmentation')

        def segment_from_points(
                predictor, points, labels, image_embeddings=None, i=None):
            fake.calls.append(
                ('points', np.array(points), np.array(labels), i)
            )
            shape = fake._shape(image_embeddings, i)
            yc, xc = np.array(points)[np.array(labels) == 1].mean(axis=0)
            return fake._box_mask((yc-3, xc-3, yc+3, xc+3), shape)

        def segment_from_box(predictor, box, image_embeddings=None, i=None):
            fake.calls.append(('box', np.array(box), i))
            return fake._box_mask(box, fake._shape(image_embeddings, i))

        def segment_from_box_and_points(
                predictor, box, points, labels, image_embeddings=None,
                i=None):
            fake.calls.append(
                ('box_and_points', np.array(box), np.array(points),
                 np.array(labels), i)
            )
            return fake._box_mask(box, fake._shape(image_embeddings, i))

        pbs.segment_from_points = segment_from_points
        pbs.segment_from_box = segment_from_box
        pbs.segment_from_box_and_points = segment_from_box_and_points

        mds = types.ModuleType('micro_sam.multi_dimensional_segmentation')

        def segment_mask_in_volume(
                segmentation, predictor, image_embeddings, segmented_slices,
                stop_lower, stop_upper, iou_threshold, projection,
                update_progress=None, box_extension=0.0, verbose=False):
            fake.calls.append((
                'volume', segmentation.copy(), list(segmented_slices),
                stop_lower, stop_upper, iou_threshold, projection,
                box_extension
            ))
            assert set(np.unique(segmentation)) <= {0, 1}
            segmentation[:] = segmentation[segmented_slices[0]]
            return segmentation, (0, len(segmentation)-1)
        mds.segment_mask_in_volume = segment_mask_in_volume

        package = types.ModuleType('micro_sam')
        package.util = util
        package.prompt_based_segmentation = pbs
        package.multi_dimensional_segmentation = mds
        return {
            'micro_sam': package,
            'micro_sam.util': util,
            'micro_sam.prompt_based_segmentation': pbs,
            'micro_sam.multi_dimensional_segmentation': mds,
        }


def _make_model(monkeypatch, **init_kwargs):
    fake = FakeMicroSam()
    for name, module in fake.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    module = _load_module()
    model = module.Model(**init_kwargs)
    fake.calls.clear()
    return fake, model


def _calls(fake, kind):
    return [call for call in fake.calls if call[0] == kind]


def test_model_arg_spec_exposes_gui_params():
    from cellacdc import myutils_models

    had_micro_sam = 'micro_sam' in sys.modules
    module = _load_module()
    init_params, segment_params = myutils_models.getModelArgSpec(module)
    # Inspecting the arguments does not import micro-sam
    assert ('micro_sam' in sys.modules) == had_micro_sam
    assert module.Model.prompt_types == ('point', 'box')

    assert [(p.name, p.default) for p in init_params] == [
        ('model_type', 'vit_b_lm'), ('checkpoint_path', ''),
        ('device', 'auto'),
    ]
    assert [(p.name, p.default) for p in segment_params] == [
        ('lab', None), ('treat_other_objects_as_background', False),
        ('segment_entire_z_stack', True), ('projection', 'single_point'),
        ('iou_threshold', 0.5), ('box_extension', 0.05),
    ]
    kinds = {
        p.name: myutils_models._serialize_model_param_type(p.type)
        for p in init_params + segment_params
    }
    assert kinds['lab'] == {'kind': 'hidden'}
    assert kinds['checkpoint_path'] == {'kind': 'path'}
    assert kinds['device']['values'] == ['auto', 'cpu', 'gpu']
    assert kinds['model_type']['values'][0] == 'vit_b_lm'
    assert {'vit_b_lm', 'vit_l_lm', 'vit_t_lm', 'vit_b', 'vit_h'} <= set(
        kinds['model_type']['values']
    )
    assert kinds['projection']['values'] == [
        'single_point', 'points', 'box', 'mask', 'points_and_mask'
    ]
    for name in ('treat_other_objects_as_background',
                 'segment_entire_z_stack'):
        assert kinds[name] == {'kind': 'bool'}
    for name in ('iou_threshold', 'box_extension'):
        assert kinds[name] == {'kind': 'float'}

    # Every GUI parameter has its own description in the params dialog
    for param in init_params + segment_params:
        if param.name == 'lab':
            continue
        assert param.desc and ' : ' not in param.desc, param.name
        assert 'Default is' in param.desc, param.name

    assert 'micro-sam' in module.url_help()


@pytest.mark.parametrize('checkpoint_path', ['', 'None', None])
def test_init_loads_the_model(monkeypatch, checkpoint_path):
    fake = FakeMicroSam()
    for name, module in fake.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    module = _load_module()

    model = module.Model(
        model_type='vit_l_lm', checkpoint_path=checkpoint_path, device='auto'
    )
    assert fake.calls == [('get_sam_model', 'vit_l_lm', 'cuda', None)]
    assert model.device == 'cuda'

    fake.calls.clear()
    module.Model(checkpoint_path='weights.pt', device='cpu')
    assert fake.calls == [('get_sam_model', 'vit_b_lm', 'cpu', 'weights.pt')]


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
    with pytest.raises(ValueError, match='3 coordinates'):
        model.add_prompt((1, 2), 1)
    with pytest.raises(ValueError, match='5 coordinates'):
        model.add_prompt((1, 2, 3), 1, prompt_type='box')
    with pytest.raises(ValueError, match='must belong to an object'):
        model.add_prompt((None, 1, 2, 5, 6), 0, prompt_type='box')
    with pytest.raises(ValueError, match='is empty'):
        model.add_prompt((None, 5, 2, 5, 9), 1, prompt_type='box')
    with pytest.raises(ValueError, match='not supported by micro-sam'):
        model.add_prompt((1, 2, 3), 1, prompt_type='brush_stroke')
    assert not model.prompt_ids_image_mapper


def test_segment_2D_points_and_negatives(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((40, 50), np.uint16)
    # z is ignored for 2D images (None, NaN or a number)
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

    (_, points1, labels1, i1), (_, points2, labels2, i2) = _calls(
        fake, 'points'
    )
    # (y, x) coordinates, positive points first
    assert points1.tolist() == [[10, 10], [12, 14], [1, 1]]
    assert labels1.tolist() == [1, 1, 0]
    assert points2.tolist() == [[30, 30], [1, 1], [2, 2]]
    assert labels2.tolist() == [1, 0, 0]
    assert i1 is None and i2 is None
    # One embedding computation for the image
    assert fake.embeddings_calls == [((40, 50), 2)]
    # Prompts are reset after segmenting
    assert not model.prompt_ids_image_mapper and not model.points
    assert not model.negative_points and not model.boxes


def test_treat_other_objects_as_background(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((40, 50))
    model.add_prompt((None, 10, 10), 1)
    model.add_prompt((None, 30, 30), 2)
    model.segment(image, treat_other_objects_as_background=True)
    (_, points1, labels1, _), (_, points2, labels2, _) = _calls(
        fake, 'points'
    )
    assert points1.tolist() == [[10, 10], [30, 30]]
    assert labels1.tolist() == [1, 0]
    assert points2.tolist() == [[30, 30], [10, 10]]
    assert labels2.tolist() == [1, 0]


def test_segment_2D_boxes(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((40, 50))
    # Box only, coordinates sorted
    model.add_prompt((None, 20, 15, 5, 3), 1, prompt_type='box')
    # Box + point + negative point (only for object 2)
    model.add_prompt((None, 25, 30, 35, 45), 2, prompt_type='box')
    model.add_prompt((None, 30, 40), 2)
    model.add_prompt((None, 0, 0), 0, parent_obj_id=2)
    # Two boxes of the same object --> bounding box
    model.add_prompt((None, 0, 20, 4, 24), 3, prompt_type='box')
    model.add_prompt((None, 2, 26, 6, 30), 3, prompt_type='box')

    lab = model.segment(image)
    (_, box1, i1), (_, box3, _) = _calls(fake, 'box')
    (_, box2, points2, labels2, i2), = _calls(fake, 'box_and_points')
    assert box1.tolist() == [5, 3, 20, 15] and i1 is None
    assert box2.tolist() == [25, 30, 35, 45]
    assert points2.tolist() == [[30, 40], [0, 0]]
    assert labels2.tolist() == [1, 0]
    assert box3.tolist() == [0, 20, 6, 30]
    assert (lab[5:20, 3:15] == 1).all() and (lab[25:35, 30:45] == 2).all()
    assert (lab[0:6, 20:30] == 3).all()
    assert not _calls(fake, 'points')


def test_negative_points_apply_to_boxes(monkeypatch):
    fake, model = _make_model(monkeypatch)
    model.add_prompt((None, 5, 3, 20, 15), 1, prompt_type='box')
    model.add_prompt((None, 0, 0), 0)  # negative, all objects
    model.segment(np.zeros((40, 50)))
    (_, box, points, labels, _), = _calls(fake, 'box_and_points')
    assert box.tolist() == [5, 3, 20, 15]
    assert points.tolist() == [[0, 0]] and labels.tolist() == [0]
    assert not _calls(fake, 'box')


def test_objects_without_positive_prompts_are_skipped(monkeypatch):
    fake, model = _make_model(monkeypatch)
    model.add_prompt((None, 1, 1), 0)
    model.add_prompt((None, 1, 1), 0, parent_obj_id=5)
    lab = model.segment(np.zeros((10, 10)))
    assert not lab.any() and not fake.calls


@pytest.mark.parametrize('order', [(1, 2), (2, 1)])
def test_smaller_objects_are_on_top(monkeypatch, order):
    _, model = _make_model(monkeypatch)
    boxes = {1: (None, 5, 5, 15, 15), 2: (None, 0, 0, 30, 30)}
    # Object 1 is inside object 2, whatever the order of the prompts
    for prompt_id in order:
        model.add_prompt(boxes[prompt_id], prompt_id, prompt_type='box')
    lab = model.segment(np.zeros((30, 30)))
    assert (lab[5:15, 5:15] == 1).all() and (lab == 2).sum() == 900 - 100

    # 3D: the masks are stored as bounding box crops
    model.add_prompt((1, 2, 3, 6, 9), 1, prompt_type='box')
    model.add_prompt((1, 0, 0, 30, 30), 2, prompt_type='box')
    lab = model.segment(
        np.zeros((3, 30, 30)), lab=np.zeros((3, 30, 30), np.uint32),
        segment_entire_z_stack=False
    )
    assert (lab[1, 2:6, 3:9] == 1).all() and (lab == 1).sum() == 24
    assert (lab[1] == 2).sum() == 900 - 24 and not lab[[0, 2]].any()


def test_embeddings_are_cached(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.arange(400, dtype=np.uint16).reshape(20, 20)
    for _ in range(2):
        model.add_prompt((None, 5, 5), 1)
        model.add_prompt((None, 10, 10), 2)
        model.segment(image.copy())
    assert fake.embeddings_calls == [((20, 20), 2)]

    image[0, 0] += 1
    model.add_prompt((None, 5, 5), 1)
    model.segment(image)
    assert fake.embeddings_calls == [((20, 20), 2)]*2


def test_zoom_prompts_on_a_crop(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((40, 50))
    crop = image[10:30, 20:50]
    model.add_prompt(
        (None, 2, 3, 8, 13), 4, prompt_type='box', image=crop,
        image_origin=(0, 10, 20)
    )
    lab = model.segment(image)
    assert fake.embeddings_calls == [((20, 30), 2)]
    assert (lab[12:18, 23:33] == 4).all() and (lab == 4).sum() == 60


def test_rgb_images(monkeypatch):
    fake, model = _make_model(monkeypatch)
    rgb = np.zeros((20, 30, 3), np.uint8)
    model.add_prompt((None, 10, 10), 1)
    lab = model.segment(rgb, lab=np.zeros((20, 30), np.uint32))
    assert lab.shape == (20, 30)
    model.add_prompt((None, 10, 10), 1)
    assert model.segment(rgb).shape == (20, 30)
    assert fake.embeddings_calls == [((20, 30, 3), 2)]

    # A z-stack with 3 z-slices is not RGB if the labels are 3D
    zstack = np.zeros((3, 20, 30))
    model.add_prompt((1, 10, 10), 1)
    lab = model.segment(
        zstack, lab=np.zeros((3, 20, 30), np.uint32),
        segment_entire_z_stack=False
    )
    assert lab.shape == (3, 20, 30) and lab[1].any() and not lab[0].any()


def test_segment_z_stack_with_propagation(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((6, 20, 30))
    model.add_prompt((2, 10, 10), 1)
    model.add_prompt((4, 2, 3, 8, 13), 1, prompt_type='box')
    model.add_prompt((3, 0, 0), 0)  # negative on z-slice 3 (no prompts)
    model.add_prompt((4, 15, 25), 2)
    lab = model.segment(
        image, lab=np.zeros((6, 20, 30), np.uint32), projection='mask',
        iou_threshold=0, box_extension=0.1
    )
    # Volume embeddings computed once and used with the z-slice index
    assert fake.embeddings_calls == [((6, 20, 30), 3)]
    (_, _, labels, i_points), = [
        c for c in _calls(fake, 'points') if c[3] == 2
    ]
    assert labels.tolist() == [1]  # the negative point is on z-slice 3
    (_, box, i_box), = _calls(fake, 'box')
    assert i_box == 4 and box.tolist() == [2, 3, 8, 13]

    volume_calls = _calls(fake, 'volume')
    assert len(volume_calls) == 2
    _, seg, slices, stop_lower, stop_upper, iou, projection, ext = (
        volume_calls[0]
    )
    assert slices == [2, 4] and not stop_lower and not stop_upper
    assert seg[2].any() and seg[4].any() and not seg[3].any()
    # A threshold <= 0 is clipped (the propagation would never stop)
    assert 0 < iou <= 1e-6
    assert projection == 'mask' and ext == 0.1
    assert lab.shape == (6, 20, 30)
    # Propagated to all the z-slices by the fake
    assert all((lab[z] == 1).any() for z in range(6))


def test_segment_z_stack_without_propagation(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((6, 20, 30))
    model.add_prompt((2, 10, 10), 1)
    model.add_prompt((4, 2, 3, 8, 13), 1, prompt_type='box')
    lab = model.segment(image, segment_entire_z_stack=False)
    # Only the embeddings of the z-slices with prompts
    assert fake.embeddings_calls == [((20, 30), 2), ((20, 30), 2)]
    assert not _calls(fake, 'volume')
    assert [z for z in range(6) if lab[z].any()] == [2, 4]


def test_2D_masks_of_z_stack_data(monkeypatch):
    fake, model = _make_model(monkeypatch)
    image = np.zeros((6, 20, 30))
    model.add_prompt((2, 2, 3, 8, 13), 1, prompt_type='box')
    model.add_prompt((4, 10, 15, 18, 25), 2, prompt_type='box')
    lab = model.segment(image, lab=np.zeros((20, 30), np.uint32))
    assert lab.shape == (20, 30)
    assert (lab[2:8, 3:13] == 1).all() and (lab[10:18, 15:25] == 2).all()
    # No propagation with 2D masks
    assert not _calls(fake, 'volume')


def test_cubic_z_stack_with_2D_masks(monkeypatch):
    """A (Z, Y, X) z-stack with Z == Y == X is not a (Y, X, C) image"""
    fake, model = _make_model(monkeypatch)
    image = np.zeros((8, 8, 8))
    model.add_prompt((3, 1, 2, 5, 6), 1, prompt_type='box')
    lab = model.segment(image, lab=np.zeros((8, 8), np.uint32))
    assert lab.shape == (8, 8) and (lab[1:5, 2:6] == 1).all()
    # The z-slice embeddings were used (not a 2D RGB image)
    assert fake.embeddings_calls == [((8, 8), 2)]
    (_, _, i), = _calls(fake, 'box')
    assert i is None

    # Same image with 3D masks
    model.add_prompt((3, 1, 2, 5, 6), 1, prompt_type='box')
    lab = model.segment(
        image, lab=np.zeros((8, 8, 8), np.uint32),
        segment_entire_z_stack=False
    )
    assert lab.shape == (8, 8, 8) and lab[3].any() and not lab[2].any()


def test_2D_masks_keep_only_the_projections(monkeypatch):
    """Overlapping objects of 2D masks of z-stack data: smaller projected
    object on top"""
    _, model = _make_model(monkeypatch)
    image = np.zeros((4, 20, 20))
    model.add_prompt((1, 0, 0, 20, 20), 1, prompt_type='box')
    model.add_prompt((2, 5, 5, 10, 10), 2, prompt_type='box')
    lab = model.segment(image, lab=np.zeros((20, 20), np.uint32))
    assert (lab[5:10, 5:10] == 2).all() and (lab == 1).sum() == 400 - 25


def test_negative_box_extension_is_clipped(monkeypatch):
    fake, model = _make_model(monkeypatch)
    model.add_prompt((1, 5, 5), 1)
    model.segment(np.zeros((3, 20, 20)), box_extension=-0.5)
    (*_, box_extension), = _calls(fake, 'volume')
    assert box_extension == 0.0


@pytest.mark.parametrize('installed', [True, False])
def test_vit_t_models_require_mobile_sam(monkeypatch, installed):
    find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args):
        if name == 'mobile_sam':
            return object() if installed else None
        return find_spec(name, *args)

    monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)
    if installed:
        _, model = _make_model(monkeypatch, model_type='vit_t_lm')
        assert model.model_type == 'vit_t_lm'
        return
    with pytest.raises(ModuleNotFoundError, match='requires MobileSAM'):
        _make_model(monkeypatch, model_type='vit_t_lm')
    # The other models do not need MobileSAM
    _make_model(monkeypatch, model_type='vit_b_lm')


def test_prompts_are_reset_after_errors(monkeypatch):
    fake, model = _make_model(monkeypatch)
    model.add_prompt((None, 5, 5), 1)

    def fail(*args, **kwargs):
        raise RuntimeError('decoder failed')

    pbs = sys.modules['micro_sam.prompt_based_segmentation']
    monkeypatch.setattr(pbs, 'segment_from_points', fail)
    with pytest.raises(RuntimeError, match='decoder failed'):
        model.segment(np.zeros((10, 10)))
    assert not model.points and not model.prompt_ids_image_mapper


def test_incompatible_labels_shape(monkeypatch):
    _, model = _make_model(monkeypatch)
    model.add_prompt((None, 5, 5), 1)
    with pytest.raises(ValueError, match='not compatible'):
        model.segment(np.zeros((10, 10)), lab=np.zeros((5, 5), np.uint32))


@pytest.mark.skipif(
    importlib.util.find_spec('micro_sam') is None,
    reason='micro-sam is not installed'
)
def test_real_microsam_points_and_box():
    import skimage.draw

    module = _load_module()
    model = module.Model(model_type='vit_b_lm', device='auto')
    image = np.random.default_rng(0).normal(100, 8, (128, 160))
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


def test_package_import_checks_installation(monkeypatch):
    from cellacdc import myutils

    calls = []
    monkeypatch.setattr(myutils, 'check_install_torch', lambda: None)
    monkeypatch.setattr(
        myutils, 'check_install_package',
        lambda *args, **kwargs: calls.append((args, kwargs))
    )
    myutils.check_install_microsam()
    assert calls == [(
        ('micro_sam',),
        {'pypi_name': 'micro_sam', 'min_version': '1.8',
         'include_lower_version': True},
    )]


def test_microsam_environment_config():
    from cellacdc import myutils
    from cellacdc.service import environment

    assert environment.get_model_env_name('micro-sam') == '_acdc_micro_sam'
    config = environment.load_model_env_config('micro-sam')
    assert config['python'] == '3.12'
    assert config['torch'] is True and config['torch_version'] == '>=2.5,<3'
    defaults = dict(
        min_version='', max_version='', exact_version='',
        include_lower_version=False, include_higher_version=False,
    )
    spec, = myutils._normalize_pkg_specs(
        config['packages'], '', '', defaults
    )
    assert spec['pkg_name'] == 'micro_sam'
    assert spec['import_pkg_name'] == 'micro_sam'
    # Installed with pip (official installation), minimum version inclusive
    myutils._check_package_version_constraints(spec, '1.8.0')
    myutils._check_package_version_constraints(spec, '1.9.0')
    with pytest.raises(ModuleNotFoundError):
        myutils._check_package_version_constraints(spec, '1.7.2')

    # MobileSAM (vit_t models) from the GitHub archive: git not required.
    # Optional: the other models work without it
    step, = config['setup_steps']
    assert step['optional'] is True
    assert step['check'] == 'import mobile_sam'
    (args,) = step['commands']
    assert args[:3] == ['-m', 'pip', 'install']
    assert args[3].startswith('https://github.com/ChaoningZhang/MobileSAM/')
    assert args[3].endswith('.zip') and 'git+' not in args[3]
    # Same URL as in the error message of the wrapper
    assert args[3] == _load_module().MOBILE_SAM_URL
