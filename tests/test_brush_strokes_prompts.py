"""Tests of the Qt-free helpers used for brush strokes prompts (e.g.,
ConvPaint) in `cellacdc.promptable_models.utils`."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cellacdc.promptable_models import utils


def _assert_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type as err:
        return err
    raise AssertionError(f'{exc_type.__name__} was not raised')


def _segm_info_df(filename, rows):
    """`rows` is a list of (frame_i, z_slice_used_gui, which_z_proj_gui)"""
    df = pd.DataFrame({
        'filename': [filename]*len(rows),
        'frame_i': [row[0] for row in rows],
        'z_slice_used_gui': [row[1] for row in rows],
        'which_z_proj_gui': [row[2] for row in rows],
    })
    return df.set_index(['filename', 'frame_i'])


def _fake_posData_zstack(rows=None, SizeT=2, SizeZ=5, Y=6, X=7):
    rng = np.random.default_rng(0)
    img_data = rng.integers(0, 255, size=(SizeT, SizeZ, Y, X)).astype(float)
    filename = 'test_s01'
    if rows is None:
        rows = [(frame_i, 1, 'single z-slice') for frame_i in range(SizeT)]
    return SimpleNamespace(
        img_data=img_data, SizeZ=SizeZ, SizeT=SizeT, filename=filename,
        segmInfo_df=_segm_info_df(filename, rows)
    )


def test_get_brush_strokes_image_2D_data():
    img_data = np.arange(2*4*5).reshape(2, 4, 5)
    posData = SimpleNamespace(img_data=img_data, SizeZ=1)
    image = utils.get_brush_strokes_image(posData, 1, 2)
    assert np.array_equal(image, img_data[1])


def test_get_brush_strokes_image_3D_annotations_returns_zstack():
    posData = _fake_posData_zstack()
    image = utils.get_brush_strokes_image(posData, 1, 3)
    assert image.shape == (5, 6, 7)
    assert np.array_equal(image, posData.img_data[1])


def test_get_brush_strokes_image_2D_segm_of_zstack_single_slice():
    posData = _fake_posData_zstack(
        rows=[(0, 3, 'single z-slice'), (1, 4, 'single z-slice')]
    )
    assert np.array_equal(
        utils.get_brush_strokes_image(posData, 0, 2), posData.img_data[0, 3]
    )
    assert np.array_equal(
        utils.get_brush_strokes_image(posData, 1, 2), posData.img_data[1, 4]
    )


def test_get_brush_strokes_image_2D_segm_of_zstack_projections():
    posData = _fake_posData_zstack(
        rows=[(0, 0, 'max z-projection'), (1, 0, 'mean z-projection')]
    )
    assert np.array_equal(
        utils.get_brush_strokes_image(posData, 0, 2),
        posData.img_data[0].max(axis=0)
    )
    assert np.allclose(
        utils.get_brush_strokes_image(posData, 1, 2),
        posData.img_data[1].mean(axis=0)
    )

    posData = _fake_posData_zstack(rows=[(0, 0, 'median z-proj.')], SizeT=1)
    assert np.allclose(
        utils.get_brush_strokes_image(posData, 0, 2),
        np.median(posData.img_data[0], axis=0)
    )


def test_get_brush_strokes_image_duplicated_index_uses_first_row():
    posData = _fake_posData_zstack(
        rows=[(0, 2, 'single z-slice'), (0, 4, 'max z-projection')],
        SizeT=1
    )
    image = utils.get_brush_strokes_image(posData, 0, 2)
    assert np.array_equal(image, posData.img_data[0, 2])


def test_get_brush_strokes_image_fallback_to_middle_slice():
    # Missing segmInfo_df entry for frame 1 --> middle z-slice
    posData = _fake_posData_zstack(rows=[(0, 1, 'single z-slice')])
    image = utils.get_brush_strokes_image(posData, 1, 2)
    assert np.array_equal(image, posData.img_data[1, 5//2])

    # NaN z-slice --> middle z-slice
    posData = _fake_posData_zstack(
        rows=[(0, np.nan, 'single z-slice'), (1, 1, 'single z-slice')]
    )
    image = utils.get_brush_strokes_image(posData, 0, 2)
    assert np.array_equal(image, posData.img_data[0, 5//2])

    # No segmInfo_df at all
    posData = _fake_posData_zstack()
    posData.segmInfo_df = None
    image = utils.get_brush_strokes_image(posData, 0, 2)
    assert np.array_equal(image, posData.img_data[0, 5//2])


def test_get_brush_strokes_image_rgb_raises():
    img_data = np.zeros((1, 4, 5, 3), dtype=np.uint8)
    posData = SimpleNamespace(img_data=img_data, SizeZ=1)
    err = _assert_raises(
        ValueError, utils.get_brush_strokes_image, posData, 0, 2
    )
    assert 'RGB' in str(err)


def test_stamp_brush_disk_inside_image():
    annot = np.zeros((20, 20), dtype=np.uint8)
    disk = utils.get_brush_disk_mask(3)
    assert disk.shape == (7, 7)
    assert utils.stamp_brush_disk(annot, 10.7, 10.2, disk, 2)
    expected = np.zeros_like(annot)
    expected[7:14, 7:14][disk] = 2
    assert np.array_equal(annot, expected)


def test_stamp_brush_disk_clipped_at_all_borders():
    disk = utils.get_brush_disk_mask(3)
    Y, X = 10, 12
    for yc, xc in [(0, 0), (0, X-1), (Y-1, 0), (Y-1, X-1), (-2, 5), (5, X+1)]:
        annot = np.zeros((Y, X), dtype=np.uint8)
        utils.stamp_brush_disk(annot, yc, xc, disk, 1)

        # Reference: paint on a padded canvas and crop
        pad = 10
        ref = np.zeros((Y+2*pad, X+2*pad), dtype=np.uint8)
        y0, x0 = int(np.floor(yc))+pad-3, int(np.floor(xc))+pad-3
        ref[y0:y0+7, x0:x0+7][disk] = 1
        assert np.array_equal(annot, ref[pad:-pad, pad:-pad]), (yc, xc)

    # Huge brush on a tiny image overflowing all borders at once
    annot = np.zeros((3, 4), dtype=np.uint8)
    utils.stamp_brush_disk(annot, 1, 1, utils.get_brush_disk_mask(50), 2)
    assert (annot == 2).all()

    # Completely outside the image
    annot = np.zeros((5, 5), dtype=np.uint8)
    assert not utils.stamp_brush_disk(annot, 50, 50, disk, 2)
    assert not annot.any()


def test_paint_brush_stroke_fills_gaps_between_positions():
    annot = np.zeros((40, 60), dtype=np.uint8)
    radius = 2
    utils.paint_brush_stroke(annot, 5, 5, 2, radius)
    # Large jump (e.g., mouse move events rate limited)
    utils.paint_brush_stroke(annot, 5, 50, 2, radius, last_yx=(5, 5))
    # Vertical move (dx == 0)
    utils.paint_brush_stroke(annot, 35, 50, 2, radius, last_yx=(5, 50))

    # Horizontal and vertical segments are continuous
    assert (annot[5, 5:51] == 2).all()
    assert (annot[5:36, 50] == 2).all()
    assert set(np.unique(annot)) == {0, 2}


def test_paint_brush_stroke_values_and_erase():
    annot = np.zeros((20, 20), dtype=np.uint8)
    utils.paint_brush_stroke(annot, 10, 10, 1, 3)
    assert (annot == 1).sum() == utils.get_brush_disk_mask(3).sum()

    # Foreground overwrites background
    utils.paint_brush_stroke(annot, 10, 10, 2, 1)
    assert annot[10, 10] == 2
    assert (annot == 1).any()

    # Erase
    utils.paint_brush_stroke(annot, 10, 10, 0, 5)
    assert not annot.any()


def test_paint_brush_stroke_on_zslice_view_of_3D_annot():
    annot = np.zeros((4, 10, 10), dtype=np.uint8)
    utils.paint_brush_stroke(annot[2], 5, 5, 2, 1)
    assert annot[2].any()
    assert not annot[[0, 1, 3]].any()


class _FakePromptableModel:
    prompt_types = ('brush_stroke',)

    def __init__(self):
        self.calls = []

    def add_prompt(self, **kwargs):
        self.calls.append(('add_prompt', kwargs))

    def train(self, **kwargs):
        self.calls.append(('train', kwargs))
        return {'num_images': 2}


def test_train_model_on_brush_strokes_forwards_prompts():
    model = _FakePromptableModel()
    annot0 = np.zeros((4, 5), dtype=np.uint8)
    annot0[0, 0] = 2
    annot1 = np.zeros((4, 5), dtype=np.uint8)
    annot1[1, 1] = 1
    img = np.ones((4, 5))
    prompts_data = [
        (0, np.int64(3), annot0, img),
        (1, 0, annot1, img)
    ]
    logs = []
    result = utils.train_model_on_brush_strokes(
        model, prompts_data, logger_func=logs.append
    )
    assert result == {'num_images': 2}
    assert [call[0] for call in model.calls] == [
        'add_prompt', 'add_prompt', 'train'
    ]
    kwargs = model.calls[0][1]
    assert kwargs['prompt'] is annot0
    assert kwargs['image'] is img
    assert kwargs['prompt_type'] == 'brush_stroke'
    assert kwargs['prompt_id'] == 0
    assert kwargs['frame_i'] == 3 and type(kwargs['frame_i']) is int
    assert kwargs['pos_i'] == 0
    assert model.calls[1][1]['pos_i'] == 1
    assert logs


def _three_objects_lab():
    lab = np.zeros((6, 8), dtype=np.uint32)
    lab[0:2, 0:2] = 1
    lab[0:2, 4:6] = 2
    lab[4:6, 5:8] = 3
    return lab


def test_add_objects_to_brush_strokes_2D():
    lab = _three_objects_lab()
    annot = np.zeros(lab.shape, dtype=np.uint8)
    num_fg, num_bg = utils.add_objects_to_brush_strokes(annot, lab, [1, 3])
    assert annot.dtype == np.uint8
    assert (annot[np.isin(lab, [1, 3])] == 2).all()
    assert (annot[lab == 0] == 1).all()
    # Not selected objects are left not annotated
    assert (annot[lab == 2] == 0).all()
    assert num_fg == np.count_nonzero(np.isin(lab, [1, 3]))
    assert num_bg == np.count_nonzero(lab == 0)
    assert set(np.unique(annot)) <= {0, 1, 2}


def test_add_objects_to_brush_strokes_keeps_existing_strokes():
    lab = _three_objects_lab()
    annot = np.zeros(lab.shape, dtype=np.uint8)
    # Foreground stroke on a cell without masks (lab == 0) and background
    # stroke on a selected cell --> both are kept
    annot[3, 0] = 2
    annot[0, 0] = 1
    num_fg, num_bg = utils.add_objects_to_brush_strokes(annot, lab, [1])
    assert annot[3, 0] == 2
    assert annot[0, 0] == 1
    assert num_fg == np.count_nonzero(lab == 1) - 1
    assert num_bg == np.count_nonzero(lab == 0) - 1

    # Second call: everything relevant is already annotated
    assert utils.add_objects_to_brush_strokes(annot, lab, [1]) == (0, 0)


def test_add_objects_to_brush_strokes_ids_types_and_missing_ids():
    lab = _three_objects_lab()
    for ids in ([1, 99], {1}, np.array([1]), [np.uint32(1)], (1, 0)):
        annot = np.zeros(lab.shape, dtype=np.uint8)
        utils.add_objects_to_brush_strokes(annot, lab, ids)
        # ID 0 (background) is never used as foreground
        assert (annot[lab == 1] == 2).all()
        assert (annot[lab == 0] == 1).all()
        assert (annot[np.isin(lab, [2, 3])] == 0).all()


def test_add_objects_to_brush_strokes_empty_selection_background_only():
    lab = _three_objects_lab()
    annot = np.zeros(lab.shape, dtype=np.uint8)
    num_fg, num_bg = utils.add_objects_to_brush_strokes(annot, lab, [])
    assert num_fg == 0
    assert num_bg == np.count_nonzero(lab == 0)
    assert not (annot == 2).any()


def test_add_objects_to_brush_strokes_select_all_covers_image():
    lab = _three_objects_lab()
    annot = np.zeros(lab.shape, dtype=np.uint8)
    utils.add_objects_to_brush_strokes(annot, lab, np.unique(lab)[1:])
    assert (annot > 0).all()


def test_add_objects_to_brush_strokes_inplace_on_views():
    # z-slice view of a 3D annotation
    lab3D = np.zeros((3, 6, 8), dtype=np.uint32)
    lab3D[1] = _three_objects_lab()
    annot3D = np.zeros(lab3D.shape, dtype=np.uint8)
    utils.add_objects_to_brush_strokes(annot3D[1], lab3D[1], [2])
    assert not annot3D[[0, 2]].any()
    assert (annot3D[1][lab3D[1] == 2] == 2).all()

    # Cutout (e.g., zoomed area)
    lab = _three_objects_lab()
    annot = np.zeros(lab.shape, dtype=np.uint8)
    region = (Ellipsis, slice(0, 3), slice(0, 3))
    utils.add_objects_to_brush_strokes(annot[region], lab[region], [1])
    outside = np.ones(lab.shape, dtype=bool)
    outside[region] = False
    assert not annot[outside].any()
    assert (annot[region][lab[region] == 1] == 2).all()
    assert (annot[region][lab[region] == 0] == 1).all()


def test_add_objects_to_brush_strokes_3D():
    lab = np.zeros((2, 4, 4), dtype=np.uint32)
    lab[0, :2, :2] = 5
    lab[1, 2:, 2:] = 6
    annot = np.zeros(lab.shape, dtype=np.uint8)
    utils.add_objects_to_brush_strokes(annot, lab, [6])
    assert (annot[lab == 6] == 2).all()
    assert (annot[lab == 5] == 0).all()
    assert (annot[lab == 0] == 1).all()


def test_add_objects_to_brush_strokes_shape_mismatch_raises():
    annot = np.zeros((4, 4), dtype=np.uint8)
    lab = np.zeros((4, 5), dtype=np.uint32)
    _assert_raises(
        ValueError, utils.add_objects_to_brush_strokes, annot, lab, [1]
    )
    assert not annot.any()


def test_add_objects_to_brush_strokes_trainable_annotation():
    lab = _three_objects_lab()
    annot = np.zeros(lab.shape, dtype=np.uint8)
    utils.add_objects_to_brush_strokes(annot, lab, [1, 2])
    model = _FakePromptableModel()
    utils.train_model_on_brush_strokes(
        model, [(0, 0, annot, np.ones(lab.shape))], logger_func=lambda x: x
    )
    assert model.calls[0][1]['prompt'] is annot


# ---------------------------------------------------------------------------
# Image source of 2D brush strokes of z-stack data
# ---------------------------------------------------------------------------

def test_get_brush_strokes_image_source():
    posData = _fake_posData_zstack(
        rows=[(0, 3, 'single z-slice'), (1, 0, 'max z-projection')]
    )
    source = utils.get_brush_strokes_image_source(posData, 0, 2)
    assert source == 3 and type(source) is int
    assert (
        utils.get_brush_strokes_image_source(posData, 1, 2)
        == 'max z-projection'
    )

    # 3D annotations and 2D data do not depend on the GUI z-slice
    assert utils.get_brush_strokes_image_source(posData, 0, 3) is None
    posData2D = SimpleNamespace(img_data=np.zeros((2, 4, 5)), SizeZ=1)
    assert utils.get_brush_strokes_image_source(posData2D, 0, 2) is None

    # Same fallbacks as `get_brush_strokes_image` (middle z-slice, clipped)
    posData = _fake_posData_zstack(
        rows=[(0, np.nan, 'single z-slice'), (1, 99, 'single z-slice')]
    )
    assert utils.get_brush_strokes_image_source(posData, 0, 2) == 5//2
    assert utils.get_brush_strokes_image_source(posData, 1, 2) == 4


def test_get_brush_strokes_image_default_source_is_gui_source():
    rows_list = [
        [(0, 3, 'single z-slice'), (1, 0, 'max z-projection')],
        [(0, 0, 'mean z-projection'), (1, 4, 'median z-proj.')],
        [(0, np.nan, 'single z-slice')],
    ]
    for rows in rows_list:
        posData = _fake_posData_zstack(rows=rows)
        for frame_i in range(posData.SizeT):
            source = utils.get_brush_strokes_image_source(posData, frame_i, 2)
            assert np.array_equal(
                utils.get_brush_strokes_image(posData, frame_i, 2),
                utils.get_brush_strokes_image(
                    posData, frame_i, 2, source=source
                )
            )


def test_get_brush_strokes_image_source_override():
    # GUI now on z-slice 3, brush strokes painted on another source
    posData = _fake_posData_zstack(
        rows=[(0, 3, 'single z-slice'), (1, 3, 'single z-slice')]
    )
    image = utils.get_brush_strokes_image(posData, 0, 2, source=1)
    assert np.array_equal(image, posData.img_data[0, 1])

    # z-slice 0 is a valid source (not "no source")
    image = utils.get_brush_strokes_image(posData, 0, 2, source=0)
    assert np.array_equal(image, posData.img_data[0, 0])

    image = utils.get_brush_strokes_image(
        posData, 0, 2, source='max z-projection'
    )
    assert np.array_equal(image, posData.img_data[0].max(axis=0))

    # Ignored with 3D annotations and 2D data
    image = utils.get_brush_strokes_image(posData, 0, 3, source=1)
    assert np.array_equal(image, posData.img_data[0])
    img_data = np.arange(2*4*5).reshape(2, 4, 5)
    posData2D = SimpleNamespace(img_data=img_data, SizeZ=1)
    image = utils.get_brush_strokes_image(posData2D, 1, 2, source=3)
    assert np.array_equal(image, img_data[1])
