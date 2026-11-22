"""Prompts of the micro-sam promptable models: `micro-sam_v1` (original
Segment Anything models, SAM1, `micro_sam.v1`) and `micro-sam_v2` (SAM2
models, `micro_sam.v2`).

Prompts are added with `add_prompt` and grouped by object (`prompt_id`):

* Point prompts are `(z, y, x)` coordinates. `prompt_id = 0` adds a negative
  point (background), shared by all the objects or, if `parent_obj_id` is not
  0, only by the object `parent_obj_id`.
* Box prompts are `(z, y_min, x_min, y_max, x_max)` coordinates (maximum
  coordinates excluded) and must belong to an object (`prompt_id > 0`).

`z` is ignored for 2D images. Point and box prompts of the same object are
combined. Each object can have one box per z-slice (if more boxes are added
to the same object and z-slice, their bounding box is used).

`MicroSamPromptsModel._segment` (called by the `segment` method of the
models) passes the prompts of the objects of each prompt image (the image
passed to `segment` or the image given with the prompts, e.g., a crop) to
the `_segment_image_objects` generator of the model, which yields the mask of
each object. The prompts are then reset, also after errors. The masks are
returned as labels (IDs are the prompt IDs, smaller objects on top of larger
ones). With 2D masks of z-stack data (`lab` is 2D) the masks of the z-slices
with prompts are projected.

The models cache the image embeddings of the last image in `_embeddings`
(see `_use_embeddings_of`), so that segmenting the same image again (e.g.,
after adding more prompts) only runs the (fast) prompt decoder.

This module must not import micro-sam: the GUI reads the parameters of the
models in the main Cell-ACDC environment, where micro-sam is not installed.
"""
import hashlib
import threading
from collections import defaultdict

import numpy as np

class Device:
    values = ['auto', 'cpu', 'gpu']

class NotParam:
    not_a_param = True

def is_missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False

def checkpoint_path_or_none(checkpoint_path):
    """None if `checkpoint_path` is not set (e.g., empty in the GUI)."""
    if is_missing(checkpoint_path) or checkpoint_path in ('', 'None'):
        return None
    return checkpoint_path

def normalize_z(z):
    return 0 if is_missing(z) else int(round(float(z)))

def get_torch_device(device, get_device, model_name='micro-sam'):
    """PyTorch device of the `Device` parameter (or e.g., 'cuda:1').

    `get_device` is the function of micro-sam that returns its default
    device (argument None) or validates a device.
    """
    if device in (None, '', 'None', 'auto'):
        return get_device(None)

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
            f'{model_name}: device "gpu" was selected, but no GPU is '
            'available to PyTorch (neither CUDA nor Apple MPS). Select '
            '"auto" or "cpu".'
        )

    # E.g., 'cuda:1' (validated by micro-sam)
    return get_device(device)

def image_hash(image):
    image = np.ascontiguousarray(image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str((image.shape, image.dtype.str)).encode())
    digest.update(image.view(np.uint8).reshape(-1).data)
    return digest.hexdigest()

def bbox_slices(mask):
    """Slices of the bounding box of the True values of `mask` (not
    empty)."""
    slices = []
    for axis in range(mask.ndim):
        other_axes = tuple(a for a in range(mask.ndim) if a != axis)
        idx = np.flatnonzero(mask.any(axis=other_axes))
        slices.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(slices)

def box_union(boxes):
    boxes = np.asarray(boxes, dtype=float)
    return np.array([
        boxes[:, 0].min(), boxes[:, 1].min(),
        boxes[:, 2].max(), boxes[:, 3].max()
    ])

class MicroSamPromptsModel:
    """Base class of the micro-sam models (see module docstring).

    Subclasses call `_init_prompts` in `__init__`, implement
    `_segment_image_objects` and call `_segment` in `segment`.
    """
    prompt_types = ('point', 'box')
    # Name used in the messages
    display_name = 'micro-sam'

    def _init_prompts(self):
        self._lock = threading.RLock()
        # Embeddings of the last image, see `_use_embeddings_of`
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

    def _use_embeddings_of(self, image_key):
        """Keep only the cached embeddings of the image `image_key`."""
        if image_key == self._embeddings_image_key:
            return
        self._release_embeddings()
        self._embeddings_image_key = image_key

    def _release_embeddings(self):
        """Drop the cached embeddings (subclasses also release the resources
        of the embeddings, e.g., temporary files)."""
        self._embeddings = {}

    def _validate_point(self, prompt):
        prompt = tuple(prompt)
        if len(prompt) != 3:
            raise ValueError(
                'Point prompts must be a sequence of 3 coordinates '
                f'(z, y, x), got {prompt}.'
            )
        z, y, x = prompt
        return normalize_z(z), float(y), float(x)

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
        return normalize_z(z), y0, x0, y1, x1

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
                    f'{self.display_name} (supported types: '
                    f'{self.prompt_types}).'
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
        """Points (y, x), labels and box (y_min, x_min, y_max, x_max) or
        None on z-slice `z` (all the prompts if `z` is None)."""
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
        box = box_union(slice_boxes) if slice_boxes else None
        return points, labels, box

    @staticmethod
    def _prompted_z(positive, boxes, SizeZ):
        """Sorted z-slices (in the z-stack) with positive points or boxes."""
        prompted_z = sorted(
            {z for z, _, _ in positive} | {box[0] for box in boxes}
        )
        return [z for z in prompted_z if 0 <= z < SizeZ]

    def _prompt_image_groups(self, image):
        """`(prompt_image, image_origin, prompt_ids)` of the objects that
        share the same prompt image (usually all of them)."""
        groups = {}
        for prompt_id, (prompt_image, image_origin) in (
                self.prompt_ids_image_mapper.items()):
            if prompt_image is None:
                prompt_image = image
            key = (id(prompt_image), image_origin)
            if key not in groups:
                groups[key] = (np.asarray(prompt_image), image_origin, [])
            groups[key][2].append(prompt_id)
        return list(groups.values())

    def _segment_image_objects(
            self, prompt_image, image_key, objects, is_zstack,
            segment_entire_z_stack, **kwargs
        ):
        """Yield `(prompt_id, mask)` of the objects of `prompt_image`.

        Parameters
        ----------
        prompt_image : np.ndarray
            Image of the prompts, (Y, X[, C]) or (Z, Y, X[, C]) z-stack.
        image_key : str
            Hash of `prompt_image` (see `_use_embeddings_of`).
        objects : dict
            `{prompt_id: (positive, negative, boxes)}` (see
            `_object_prompts`).
        is_zstack : bool
            Whether `prompt_image` is a z-stack.
        segment_entire_z_stack : bool
            Whether the objects are propagated to all the z-slices (always
            False with 2D masks of z-stack data).
        **kwargs
            Other parameters of `segment` of the model.

        Yields
        ------
        (int, np.ndarray or None)
            Prompt ID and boolean mask with the spatial shape of
            `prompt_image` (None or empty mask if there is no object).
        """
        raise NotImplementedError

    def _segment(
            self, image, lab, treat_other_objects_as_background,
            segment_entire_z_stack, **kwargs
        ):
        """Segment the objects of the prompts added (then reset). Returns
        the labels with the spatial shape of `image` (or of `lab` if
        given). See `segment` of the models."""
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
                for prompt_image, image_origin, prompt_ids in (
                        self._prompt_image_groups(image)):
                    image_key = image_hash(prompt_image)
                    objects_prompts = {
                        prompt_id: self._object_prompts(
                            prompt_id, treat_other_objects_as_background
                        )
                        for prompt_id in prompt_ids
                    }
                    masks = self._segment_image_objects(
                        prompt_image, image_key, objects_prompts, is_zstack,
                        segment_entire_z_stack, **kwargs
                    )
                    for prompt_id, obj_mask in masks:
                        if obj_mask is None or not obj_mask.any():
                            continue

                        # Keep only the bounding box of the mask (memory)
                        bbox = bbox_slices(obj_mask)
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
