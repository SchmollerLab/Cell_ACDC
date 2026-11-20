"""micro-sam (Segment Anything for Microscopy) as a Cell-ACDC promptable model.

This implements the interactive segmentation of the micro-sam "Annotator 2D"
and "Annotator 3D" with point and box prompts (the "Annotator Tracking" is
not implemented). See https://computational-cell-analytics.github.io/micro-sam

Prompts are added with `add_prompt` and grouped by object (`prompt_id`):

* Point prompts are `(z, y, x)` coordinates. `prompt_id = 0` adds a negative
  point (background), shared by all the objects or, if `parent_obj_id` is not
  0, only by the object `parent_obj_id`.
* Box prompts are `(z, y_min, x_min, y_max, x_max)` coordinates (maximum
  coordinates excluded) and must belong to an object (`prompt_id > 0`).

`z` is ignored for 2D images. Point and box prompts of the same object are
combined. Each object can have one box per z-slice (if more boxes are added
to the same object and z-slice, their bounding box is used).

`segment` then segments each object (2D) or, with z-stacks, each z-slice
with prompts of each object. With 3D segmentation masks the masks are then
propagated to the other z-slices like in the micro-sam "Annotator 3D"
(`multi_dimensional_segmentation.segment_mask_in_volume`).

The image embeddings are cached, so that segmenting the same image again
(e.g., after adding more prompts) only runs the (fast) prompt decoder.

The Segment Anything (SAM1) API of micro-sam is imported with
`_micro_sam_module`: micro-sam 2.0 moved it from the top-level modules
(`micro_sam.util.get_sam_model`, `micro_sam.prompt_based_segmentation`,
...) to `micro_sam.v1` (same functions and models), while micro-sam 1.x
only has the top-level modules.
"""
import hashlib
import importlib
import os
import threading
from collections import defaultdict

import numpy as np

def _micro_sam_module(name):
    """Return the module `name` (e.g., 'util') of the Segment Anything
    (SAM1) API of micro-sam: `micro_sam.v1.<name>` (micro-sam >= 2.0) or
    `micro_sam.<name>` (micro-sam 1.x)."""
    try:
        return importlib.import_module(f'micro_sam.v1.{name}')
    except ModuleNotFoundError as err:
        # Fall back only if `micro_sam.v1` does not exist (micro-sam 1.x),
        # not if a dependency of micro-sam is missing
        if err.name not in ('micro_sam.v1', f'micro_sam.v1.{name}'):
            raise
    return importlib.import_module(f'micro_sam.{name}')

# Models of the official micro-sam model zoo. The `vit_t` models require
# MobileSAM (installed in the isolated environment of micro-sam).
class ModelType:
    values = [
        'vit_b_lm', 'vit_l_lm', 'vit_t_lm',
        'vit_b_em_organelles', 'vit_l_em_organelles', 'vit_t_em_organelles',
        'vit_b_histopathology', 'vit_l_histopathology',
        'vit_h_histopathology',
        'vit_b_medical_imaging',
        'vit_b', 'vit_l', 'vit_h', 'vit_t',
    ]

class Device:
    values = ['auto', 'cpu', 'gpu']

# Projection modes of `segment_mask_in_volume` (3D propagation)
class Projection:
    values = ['single_point', 'points', 'box', 'mask', 'points_and_mask']

class NotParam:
    not_a_param = True

def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False

def _normalize_z(z):
    return 0 if _is_missing(z) else int(round(float(z)))

def _get_torch_device(device):
    util = _micro_sam_module('util')

    if device in (None, '', 'None', 'auto'):
        return util.get_device(None)

    if device == 'cpu':
        return 'cpu'

    if device == 'gpu':
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available() and mps.is_built():
            return 'mps'
        raise RuntimeError(
            'micro-sam: device "gpu" was selected, but no GPU is available '
            'to PyTorch (neither CUDA nor Apple MPS). Select "auto" or "cpu".'
        )

    # E.g., 'cuda:1' (validated by micro-sam)
    return util.get_device(device)

MOBILE_SAM_URL = (
    'https://github.com/ChaoningZhang/MobileSAM/archive/'
    'f706ad9c4eb7f219c00d9050e46328518ffb65d2.zip'
)

def _check_mobile_sam(model_type):
    """The `vit_t` models require MobileSAM (an optional dependency of
    micro-sam, not available on PyPI)."""
    import importlib.util
    import sys

    if not str(model_type).startswith('vit_t'):
        return
    if importlib.util.find_spec('mobile_sam') is not None:
        return
    raise ModuleNotFoundError(
        f'The micro-sam model "{model_type}" requires MobileSAM, which is '
        'not installed in the micro-sam environment (its installation from '
        'GitHub may have failed, see the log). Select a model that does not '
        'start with "vit_t", or install MobileSAM with:\n\n'
        f'"{sys.executable}" -m pip install {MOBILE_SAM_URL}'
    )

def _image_hash(image):
    image = np.ascontiguousarray(image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str((image.shape, image.dtype.str)).encode())
    digest.update(image.view(np.uint8).reshape(-1).data)
    return digest.hexdigest()

def _bbox_slices(mask):
    """Slices of the bounding box of the True values of `mask` (not
    empty)."""
    slices = []
    for axis in range(mask.ndim):
        other_axes = tuple(a for a in range(mask.ndim) if a != axis)
        idx = np.flatnonzero(mask.any(axis=other_axes))
        slices.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(slices)

def _box_union(boxes):
    boxes = np.asarray(boxes, dtype=float)
    return np.array([
        boxes[:, 0].min(), boxes[:, 1].min(),
        boxes[:, 2].max(), boxes[:, 3].max()
    ])

class Model:
    prompt_types = ('point', 'box')

    def __init__(
            self,
            model_type: ModelType = 'vit_b_lm',
            checkpoint_path: os.PathLike = '',
            device: Device = 'auto',
        ):
        """Initialize the micro-sam model (downloaded the first time)

        Parameters
        ----------
        model_type : ModelType, optional
            micro-sam model to use. The models ending with "_lm" are
            finetuned on light microscopy data, "_em_organelles" on electron
            microscopy data, "_histopathology" on histopathology images and
            "_medical_imaging" on medical images. "vit_b", "vit_l", "vit_h"
            and "vit_t" are the original Segment Anything models.
            "vit_t" models are the fastest, "vit_l" and "vit_h" models are
            the most accurate (and slowest). Default is 'vit_b_lm'
        checkpoint_path : os.PathLike, optional
            Path to the weights of a custom (e.g., finetuned) micro-sam
            model. If given, `model_type` must be the architecture of the
            model (e.g., "vit_b" for a finetuned "vit_b_lm" model). Leave
            empty to use the weights of `model_type`. Default is ''
        device : Device, optional
            Device used to run the model. 'auto' uses the GPU if available
            (CUDA or Apple MPS) and the CPU otherwise. Default is 'auto'
        """
        util = _micro_sam_module('util')

        if _is_missing(checkpoint_path) or checkpoint_path in ('', 'None'):
            checkpoint_path = None

        _check_mobile_sam(model_type)

        self.model_type = model_type
        self.device = _get_torch_device(device)
        self.predictor = util.get_sam_model(
            model_type=model_type,
            device=self.device,
            checkpoint_path=checkpoint_path
        )
        self._lock = threading.RLock()
        # Embeddings of the last image, see `_get_embeddings`
        self._embeddings_image_key = None
        self._embeddings = {}
        self._reset_prompts()

    def _reset_prompts(self):
        # {prompt_id: (image, image_origin)}
        self.prompt_ids_image_mapper = {}
        # {prompt_id: [(z, y, x), ...]}
        self.points = defaultdict(list)
        # {parent_obj_id (0 = all objects): [(z, y, x), ...]}
        self.negative_points = defaultdict(list)
        # {prompt_id: [(z, y_min, x_min, y_max, x_max), ...]}
        self.boxes = defaultdict(list)

    def _validate_point(self, prompt):
        prompt = tuple(prompt)
        if len(prompt) != 3:
            raise ValueError(
                'Point prompts must be a sequence of 3 coordinates '
                f'(z, y, x), got {prompt}.'
            )
        z, y, x = prompt
        return _normalize_z(z), float(y), float(x)

    def _validate_box(self, prompt):
        prompt = tuple(prompt)
        if len(prompt) == 4:
            prompt = (None, *prompt)
        if len(prompt) != 5:
            raise ValueError(
                'Box prompts must be a sequence of 5 coordinates '
                f'(z, y_min, x_min, y_max, x_max), got {prompt}.'
            )
        z, y0, x0, y1, x1 = prompt
        y0, y1 = sorted((float(y0), float(y1)))
        x0, x1 = sorted((float(x0), float(x1)))
        if y1 - y0 < 1 or x1 - x0 < 1:
            raise ValueError(
                f'Box prompt {prompt} is empty (width and height must be at '
                'least 1 pixel).'
            )
        return _normalize_z(z), y0, x0, y1, x1

    def add_prompt(
            self,
            prompt,
            prompt_id: int,
            *args,
            image=None,
            image_origin=(0, 0, 0),
            parent_obj_id=0,
            prompt_type='point',
            **kwargs
        ):
        """Add a prompt to the model

        Parameters
        ----------
        prompt : sequence of floats
            Point prompt (z, y, x) or box prompt (z, y_min, x_min, y_max,
            x_max), with maximum coordinates excluded. `z` is ignored for
            2D images (it can be None).
        prompt_id : int
            ID of the object. If 0, the point prompt is a negative prompt
            (background). Box prompts require `prompt_id > 0`.
        image : np.ndarray, optional
            Image the prompt coordinates refer to (e.g., a crop of the image
            passed to `segment`, see `image_origin`). If None, the image
            passed to `segment` is used. Default is None
        image_origin : tuple of (z0, y0, x0) coordinates, optional
            Origin of `image` in the image passed to `segment`. Default is
            (0, 0, 0)
        parent_obj_id : int, optional
            If not 0, a negative point prompt (`prompt_id = 0`) is used only
            for the object `parent_obj_id`. Default is 0
        prompt_type : {'point', 'box'}, optional
            Type of the prompt. Default is 'point'
        """
        prompt_id = int(prompt_id)
        parent_obj_id = int(parent_obj_id or 0)
        with self._lock:
            if prompt_type == 'point':
                point = self._validate_point(prompt)
                if prompt_id == 0:
                    self.negative_points[parent_obj_id].append(point)
                    return
                self.points[prompt_id].append(point)
            elif prompt_type == 'box':
                if prompt_id == 0:
                    raise ValueError(
                        'Box prompts must belong to an object (prompt_id > 0)'
                    )
                self.boxes[prompt_id].append(self._validate_box(prompt))
            else:
                raise ValueError(
                    f'Prompt type "{prompt_type}" is not supported by '
                    f'micro-sam (supported types: {self.prompt_types}).'
                )

            if prompt_id not in self.prompt_ids_image_mapper:
                self.prompt_ids_image_mapper[prompt_id] = (
                    image, tuple(image_origin)
                )

    def _is_zstack(self, image, lab):
        """True if `image` is a z-stack (Z, Y, X[, C]) and False if it is a
        2D image (Y, X[, C])."""
        if image.ndim == 2:
            return False
        if image.ndim == 4:
            return True
        isChannelsLast = image.shape[-1] in (3, 4)
        if lab is not None:
            lab_shape = np.asarray(lab).shape
            if lab_shape == image.shape:
                # (Z, Y, X) z-stack with 3D masks
                return True
            isRGB = isChannelsLast and lab_shape == image.shape[:2]
            if isRGB:
                # (Y, X, C) image
                return False
            if lab_shape == image.shape[1:]:
                # (Z, Y, X) z-stack with 2D masks (e.g., also Z == Y == X)
                return True
        return not isChannelsLast

    def _get_embeddings(self, image, image_key, z=None, volume=False):
        """Return `(image_embeddings, i)` for `segment_from_*` functions.

        `image` is a 2D image (z = None) or a z-stack. With z-stacks, the
        embeddings of the entire z-stack are computed if `volume` is True
        (required for 3D propagation), otherwise only the ones of z-slice
        `z`. Only the embeddings of the last image are kept.
        """
        util = _micro_sam_module('util')

        if image_key != self._embeddings_image_key:
            self._embeddings = {}
            self._embeddings_image_key = image_key

        if z is None:
            embeddings = self._embeddings.get('image')
            if embeddings is None:
                embeddings = util.precompute_image_embeddings(
                    self.predictor, image, ndim=2, verbose=False
                )
                self._embeddings['image'] = embeddings
            return embeddings, None

        embeddings = self._embeddings.get('volume')
        if embeddings is not None:
            return embeddings, z

        if volume:
            print(
                f'micro-sam: computing the image embeddings of the '
                f'{len(image)} z-slices...'
            )
            embeddings = util.precompute_image_embeddings(
                self.predictor, image, ndim=3, verbose=True
            )
            self._embeddings['volume'] = embeddings
            return embeddings, z

        key = ('z_slice', z)
        embeddings = self._embeddings.get(key)
        if embeddings is None:
            embeddings = util.precompute_image_embeddings(
                self.predictor, image[z], ndim=2, verbose=False
            )
            self._embeddings[key] = embeddings
        return embeddings, None

    def _segment_2D(
            self, image_embeddings, i, points, labels, box, shape
        ):
        """Segment one object on a 2D image or z-slice. Returns a boolean
        mask with `shape` or None if there are no positive prompts."""
        pbs = _micro_sam_module('prompt_based_segmentation')

        has_positive = box is not None or any(label == 1 for label in labels)
        if not has_positive:
            return

        if points:
            points = np.array(points, dtype=float)
            labels = np.array(labels, dtype=int)

        if box is not None and len(points) > 0:
            mask = pbs.segment_from_box_and_points(
                self.predictor, box, points, labels,
                image_embeddings=image_embeddings, i=i
            )
        elif box is not None:
            mask = pbs.segment_from_box(
                self.predictor, box, image_embeddings=image_embeddings, i=i
            )
        else:
            mask = pbs.segment_from_points(
                self.predictor, points, labels,
                image_embeddings=image_embeddings, i=i
            )
        mask = np.asarray(mask).squeeze().astype(bool)
        if mask.shape != tuple(shape):
            raise ValueError(
                f'micro-sam returned a mask with shape {mask.shape} instead '
                f'of {tuple(shape)}.'
            )
        return mask

    def _object_prompts(self, prompt_id, treat_other_objects_as_background):
        """Return the positive points, negative points and boxes of object
        `prompt_id`."""
        positive = list(self.points.get(prompt_id, []))
        negative = list(self.negative_points.get(0, []))
        negative.extend(self.negative_points.get(prompt_id, []))
        if treat_other_objects_as_background:
            for other_id, other_points in self.points.items():
                if other_id != prompt_id:
                    negative.extend(other_points)
        boxes = list(self.boxes.get(prompt_id, []))
        return positive, negative, boxes

    @staticmethod
    def _slice_prompts(positive, negative, boxes, z=None):
        """Points (y, x), labels and box (or None) on z-slice `z` (all the
        prompts if `z` is None)."""
        points, labels = [], []
        for prompts, label in ((positive, 1), (negative, 0)):
            for z_p, y, x in prompts:
                if z is not None and z_p != z:
                    continue
                points.append((y, x))
                labels.append(label)
        slice_boxes = [
            box[1:] for box in boxes if z is None or box[0] == z
        ]
        box = _box_union(slice_boxes) if slice_boxes else None
        return points, labels, box

    def _segment_object(
            self, prompt_image, image_key, positive, negative, boxes,
            is_zstack, segment_entire_z_stack, projection, iou_threshold,
            box_extension
        ):
        """Boolean mask of one object with the spatial shape of
        `prompt_image` (None if there are no positive prompts)."""
        segment_mask_in_volume = _micro_sam_module(
            'multi_dimensional_segmentation'
        ).segment_mask_in_volume

        if not is_zstack:
            shape = prompt_image.shape[:2]
            points, labels, box = self._slice_prompts(
                positive, negative, boxes
            )
            if box is None and 1 not in labels:
                return
            image_embeddings, i = self._get_embeddings(prompt_image, image_key)
            return self._segment_2D(
                image_embeddings, i, points, labels, box, shape
            )

        SizeZ = len(prompt_image)
        shape = prompt_image.shape[1:3]
        prompted_z = sorted(
            {z for z, _, _ in positive} | {box[0] for box in boxes}
        )
        prompted_z = [z for z in prompted_z if 0 <= z < SizeZ]
        if not prompted_z:
            return

        propagate = segment_entire_z_stack and SizeZ > 1
        obj_mask = np.zeros((SizeZ, *shape), dtype=bool)
        segmented_z = []
        for z in prompted_z:
            points, labels, box = self._slice_prompts(
                positive, negative, boxes, z=z
            )
            if box is None and 1 not in labels:
                continue
            image_embeddings, i = self._get_embeddings(
                prompt_image, image_key, z=z, volume=propagate
            )
            mask = self._segment_2D(
                image_embeddings, i, points, labels, box, shape
            )
            if mask is None or not mask.any():
                continue
            obj_mask[z] = mask
            segmented_z.append(z)

        if not segmented_z or not propagate:
            return obj_mask

        image_embeddings, _ = self._get_embeddings(
            prompt_image, image_key, z=segmented_z[0], volume=True
        )
        seg = obj_mask.astype(np.uint32)
        # With a threshold <= 0 the propagation would continue from empty
        # masks (which cannot be used as prompts)
        iou_threshold = max(float(iou_threshold), 1e-6)
        box_extension = max(float(box_extension), 0.0)
        seg, _ = segment_mask_in_volume(
            seg, self.predictor, image_embeddings,
            np.array(segmented_z),
            stop_lower=False,
            stop_upper=False,
            iou_threshold=iou_threshold,
            projection=projection,
            box_extension=box_extension,
        )
        return np.asarray(seg) > 0

    def segment(
            self,
            image,
            lab: NotParam = None,
            treat_other_objects_as_background: bool = False,
            segment_entire_z_stack: bool = True,
            projection: Projection = 'single_point',
            iou_threshold: float = 0.5,
            box_extension: float = 0.05,
            *args,
            **kwargs
        ):
        """Segment the objects using the prompts added

        Parameters
        ----------
        image : (Y, X), (Y, X, C), (Z, Y, X) or (Z, Y, X, C) np.ndarray
            Image to segment.
        lab : np.ndarray, optional
            Current segmentation masks. Used only to determine the shape of
            the output (e.g., 2D masks of z-stack data). Default is None
        treat_other_objects_as_background : bool, optional
            If True, the point prompts of the other objects are used as
            negative prompts for each object. Default is False
        segment_entire_z_stack : bool, optional
            Only for z-stacks with 3D segmentation masks. If True, each
            object is first segmented on the z-slices with prompts and the
            masks are then propagated to the other z-slices (micro-sam
            "Annotator 3D"). If False, only the z-slices with prompts are
            segmented. Default is True
        projection : Projection, optional
            Only with `segment_entire_z_stack = True`. Prompts derived from
            the mask of a z-slice to segment the next z-slice:
            'single_point' (one point inside the mask), 'points' (points
            inside and outside the mask), 'box' (bounding box of the mask),
            'mask' (mask and its bounding box), or 'points_and_mask'.
            Default is 'single_point'
        iou_threshold : float, optional
            Only with `segment_entire_z_stack = True`. The propagation to
            the next z-slices stops when the overlap (intersection over
            union) between the masks of two consecutive z-slices is lower
            than this value. Default is 0.5
        box_extension : float, optional
            Only with `segment_entire_z_stack = True` and `projection`
            'box', 'mask' or 'points' (practically no effect with
            'single_point' and 'points_and_mask'). Enlargement of the
            bounding box of the mask used to prompt the next z-slice:
            values lower than 1 enlarge each side by this fraction of the
            box size (e.g., 0.05 = 5%), values of 1 or higher enlarge each
            side by this number of pixels. Default is 0.05

        Returns
        -------
        np.ndarray of uint32
            Labels of the segmented objects (IDs are the prompt IDs) with
            the spatial shape of `image` (or of `lab` if given, e.g., 2D
            masks of z-stack data, where the masks of the z-slices with
            prompts are projected). Smaller objects are on top of larger
            ones.
        """
        image = np.asarray(image)
        lab_shape = None if lab is None else tuple(np.asarray(lab).shape)
        is_zstack = self._is_zstack(image, lab)
        if is_zstack:
            out_shape = image.shape[:3]
        else:
            out_shape = image.shape[:2]
        # 2D masks of z-stack data --> project the masks of the z-slices
        project_z = is_zstack and lab_shape is not None and len(lab_shape) == 2
        if project_z:
            segment_entire_z_stack = False

        with self._lock:
            try:
                # (area, prompt_id, slice in the output, mask in the slice)
                objects = []
                # Hash each prompt image only once (usually all the objects
                # share the same image)
                image_keys = {}
                for prompt_id, (prompt_image, image_origin) in (
                        self.prompt_ids_image_mapper.items()):
                    if prompt_image is None:
                        prompt_image = image
                    prompt_image = np.asarray(prompt_image)
                    positive, negative, boxes = self._object_prompts(
                        prompt_id, treat_other_objects_as_background
                    )
                    image_key = image_keys.get(id(prompt_image))
                    if image_key is None:
                        image_key = _image_hash(prompt_image)
                        image_keys[id(prompt_image)] = image_key
                    obj_mask = self._segment_object(
                        prompt_image, image_key, positive, negative, boxes,
                        is_zstack, segment_entire_z_stack, projection,
                        iou_threshold, box_extension
                    )
                    if obj_mask is None or not obj_mask.any():
                        continue

                    # Keep only the bounding box of the mask (memory)
                    bbox = _bbox_slices(obj_mask)
                    origin = [int(c) for c in image_origin]
                    origin = origin[-obj_mask.ndim:]
                    obj_slice = tuple(
                        slice(o + s.start, o + s.stop)
                        for o, s in zip(origin, bbox)
                    )
                    # Copy: a view would keep the entire `obj_mask` alive
                    local_mask = obj_mask[bbox].copy()
                    del obj_mask
                    if project_z:
                        # 2D masks of z-stack data --> keep the projection
                        local_mask = local_mask.any(axis=0)
                        obj_slice = obj_slice[1:]
                    objects.append((
                        int(local_mask.sum()), prompt_id, obj_slice,
                        local_mask
                    ))
            finally:
                self._reset_prompts()

        if project_z:
            out_shape = out_shape[1:]
        # Larger objects first, so that smaller objects are on top
        lab_out = np.zeros(out_shape, dtype=np.uint32)
        for _, prompt_id, obj_slice, local_mask in sorted(
                objects, key=lambda obj: obj[0], reverse=True):
            lab_out[obj_slice][local_mask] = prompt_id

        if lab_shape is not None and lab_out.shape != lab_shape:
            raise ValueError(
                f'The segmentation masks have shape {lab_shape}, which is '
                f'not compatible with the image shape {image.shape}.'
            )
        return lab_out

def url_help():
    return 'https://computational-cell-analytics.github.io/micro-sam/micro_sam.html'
