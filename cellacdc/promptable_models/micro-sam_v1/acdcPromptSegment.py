"""micro-sam v1 (Segment Anything for Microscopy with the original Segment
Anything models, SAM1) as a Cell-ACDC promptable model.

This implements the interactive segmentation of the micro-sam "Annotator 2D"
and "Annotator 3D" with point and box prompts (the "Annotator Tracking" is
not implemented). See https://computational-cell-analytics.github.io/micro-sam

The prompts (points and boxes, grouped by object) are described in
`cellacdc/promptable_models/micro_sam_base.py` (shared with `micro-sam_v2`).

`segment` segments each object (2D) or, with z-stacks, each z-slice with
prompts of each object. With 3D segmentation masks the masks are then
propagated to the other z-slices like in the micro-sam "Annotator 3D"
(`multi_dimensional_segmentation.segment_mask_in_volume`).

The image embeddings are cached, so that segmenting the same image again
(e.g., after adding more prompts) only runs the (fast) prompt decoder.

micro-sam >= 2.0 is required: its SAM1 API is the `micro_sam.v1`
subpackage (`micro_sam.v1.util.get_sam_model`,
`micro_sam.v1.prompt_based_segmentation`, ...), while the top-level modules
of micro-sam 2 are for SAM2. micro-sam is imported only when the model is
used, because the GUI reads the parameters of this module in the main
Cell-ACDC environment, where micro-sam is not installed.
"""
import os

import numpy as np

from cellacdc.promptable_models.micro_sam_base import (
    Device, MicroSamPromptsModel, NotParam, checkpoint_path_or_none,
    get_torch_device
)

# Models of the official micro-sam model zoo for SAM1. The `vit_t` models
# require MobileSAM (installed in the isolated environment of micro-sam v1).
class ModelType:
    values = [
        'vit_b_lm', 'vit_l_lm', 'vit_t_lm',
        'vit_b_em_organelles', 'vit_l_em_organelles', 'vit_t_em_organelles',
        'vit_b_histopathology', 'vit_l_histopathology',
        'vit_h_histopathology',
        'vit_b_medical_imaging',
        'vit_b', 'vit_l', 'vit_h', 'vit_t',
    ]

# Projection modes of `segment_mask_in_volume` (3D propagation)
class Projection:
    values = ['single_point', 'points', 'box', 'mask', 'points_and_mask']

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
        'not installed in the micro-sam v1 environment (its installation from '
        'GitHub may have failed, see the log). Select a model that does not '
        'start with "vit_t", or install MobileSAM with:\n\n'
        f'"{sys.executable}" -m pip install {MOBILE_SAM_URL}'
    )

class Model(MicroSamPromptsModel):
    display_name = 'micro-sam v1'

    def __init__(
            self,
            model_type: ModelType = 'vit_b_lm',
            checkpoint_path: os.PathLike = '',
            device: Device = 'auto',
        ):
        """Initialize the micro-sam v1 model (downloaded the first time)

        Parameters
        ----------
        model_type : ModelType, optional
            micro-sam model (SAM1) to use. The models ending with "_lm" are
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
        from micro_sam.v1 import util

        checkpoint_path = checkpoint_path_or_none(checkpoint_path)
        _check_mobile_sam(model_type)

        self.model_type = model_type
        self.device = get_torch_device(
            device, util.get_device, self.display_name
        )
        self.predictor = util.get_sam_model(
            model_type=model_type,
            device=self.device,
            checkpoint_path=checkpoint_path
        )
        self._init_prompts()

    def _get_embeddings(self, image, image_key, z=None, volume=False):
        """Return `(image_embeddings, i)` for `segment_from_*` functions.

        `image` is a 2D image (z = None) or a z-stack. With z-stacks, the
        embeddings of the entire z-stack are computed if `volume` is True
        (required for 3D propagation), otherwise only the ones of z-slice
        `z`. Only the embeddings of the last image are kept.
        """
        from micro_sam.v1 import util

        self._use_embeddings_of(image_key)

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
                f'{self.display_name}: computing the image embeddings of '
                f'the {len(image)} z-slices...'
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
        from micro_sam.v1 import prompt_based_segmentation as pbs

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
                f'{self.display_name} returned a mask with shape '
                f'{mask.shape} instead of {tuple(shape)}.'
            )
        return mask

    def _segment_object(
            self, prompt_image, image_key, positive, negative, boxes,
            is_zstack, segment_entire_z_stack, projection, iou_threshold,
            box_extension
        ):
        """Boolean mask of one object with the spatial shape of
        `prompt_image` (None if there are no positive prompts)."""
        from micro_sam.v1.multi_dimensional_segmentation import (
            segment_mask_in_volume
        )

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
        prompted_z = self._prompted_z(positive, boxes, SizeZ)
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

    def _segment_image_objects(
            self, prompt_image, image_key, objects, is_zstack,
            segment_entire_z_stack, projection='single_point',
            iou_threshold=0.5, box_extension=0.05
        ):
        for prompt_id, (positive, negative, boxes) in objects.items():
            obj_mask = self._segment_object(
                prompt_image, image_key, positive, negative, boxes,
                is_zstack, segment_entire_z_stack, projection, iou_threshold,
                box_extension
            )
            yield prompt_id, obj_mask

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
        return self._segment(
            image, lab, treat_other_objects_as_background,
            segment_entire_z_stack, projection=projection,
            iou_threshold=iou_threshold, box_extension=box_extension
        )

def url_help():
    return 'https://computational-cell-analytics.github.io/micro-sam/micro_sam.html'
