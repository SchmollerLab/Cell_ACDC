"""micro-sam v2 (Segment Anything for Microscopy with the Segment Anything 2
models, SAM2) as a Cell-ACDC promptable model.

This implements the interactive segmentation of the micro-sam 2
"Segmentation Annotator" with point and box prompts, for 2D images and
z-stacks (the "Tracking Annotator" is not implemented). See
https://computational-cell-analytics.github.io/micro-sam

The prompts (points and boxes, grouped by object) are described in
`cellacdc/promptable_models/micro_sam_base.py` (shared with `micro-sam_v1`).

`segment` segments each object of 2D images with the SAM2 image predictor
(`micro_sam.v2.util.get_sam2_image_predictor`). With z-stacks, the image
predictor segments the z-slices with prompts of each object, unless the
segmentation masks are 3D and `segment_entire_z_stack` is True: then the
objects are segmented on the z-slices with prompts and propagated through
the z-stack by the SAM2 video predictor, like "Apply to Volume" of the
micro-sam annotator (`micro_sam.v2.prompt_based_segmentation.
PromptableSegmentation3D`, all the objects of the prompt image together).
The video predictor is loaded the first time that it is needed.

The image embeddings of the last image are cached, so that segmenting the
same image again (e.g., after adding more prompts) only runs the (fast)
prompt decoder (and the propagation). As in the micro-sam annotator, the
embeddings of z-stacks for the video predictor are stored in a temporary
folder of the micro-sam cache (removed when the embeddings of another image
are computed or at exit), to keep the memory usage low.

micro-sam >= 2.0 is required (SAM2 API in `micro_sam.v2`). micro-sam is
imported only when the model is used, because the GUI reads the parameters
of this module in the main Cell-ACDC environment, where micro-sam is not
installed.
"""
import os

import numpy as np

from cellacdc.promptable_models.micro_sam_base import (
    Device, MicroSamPromptsModel, NotParam, checkpoint_path_or_none,
    get_torch_device
)

# Models of the official micro-sam model zoo for SAM2 (SAM2.1 backbones:
# hvit_t = tiny, hvit_s = small, hvit_b = base+, hvit_l = large)
class ModelType:
    values = [
        'hvit_t_cells', 'hvit_s_cells', 'hvit_b_cells', 'hvit_l_cells',
        'hvit_s_em_mito', 'hvit_s_em_nuclei',
        'hvit_t', 'hvit_s', 'hvit_b', 'hvit_l',
    ]

def _sam_box(box, shape):
    """(y_min, x_min, y_max, x_max) box --> (x_min, y_min, x_max, y_max)
    box of SAM2, clipped to the image `shape` and rounded (as micro-sam
    processes the box prompts)."""
    y0, x0, y1, x1 = box
    height, width = shape
    sam_box = (max(x0, 0), max(y0, 0), min(x1, width), min(y1, height))
    return np.round(sam_box).astype(int)

def _as_2D(mask):
    """(..., Y, X) mask with one object --> (Y, X) boolean mask."""
    mask = np.asarray(mask)
    return mask.reshape(mask.shape[-2:]) > 0

class Model(MicroSamPromptsModel):
    display_name = 'micro-sam v2'

    def __init__(
            self,
            model_type: ModelType = 'hvit_t_cells',
            checkpoint_path: os.PathLike = '',
            device: Device = 'auto',
        ):
        """Initialize the micro-sam v2 model (downloaded the first time)

        Parameters
        ----------
        model_type : ModelType, optional
            micro-sam model (SAM2) to use. The models ending with "_cells"
            are finetuned on cells and nuclei in light microscopy (and on
            neurites in electron microscopy), "_em_mito" on mitochondria and
            "_em_nuclei" on nuclei in electron microscopy. "hvit_t",
            "hvit_s", "hvit_b" and "hvit_l" are the original Segment
            Anything 2 models. "hvit_t" models are the fastest, "hvit_b" and
            "hvit_l" models are the most accurate (and slowest). Default is
            'hvit_t_cells'
        checkpoint_path : os.PathLike, optional
            Path to the weights of a custom (e.g., finetuned) micro-sam v2
            model. The architecture (e.g., "hvit_t") is read from the
            weights. If they contain only the image encoder, the other
            weights are the ones of `model_type`. Leave empty to use the
            weights of `model_type`. Default is ''
        device : Device, optional
            Device used to run the model. 'auto' uses the GPU if available
            (CUDA or Apple MPS) and the CPU otherwise. Default is 'auto'
        """
        from micro_sam.util import get_device
        from micro_sam.v2 import util

        self.model_type = model_type
        self.checkpoint_path = checkpoint_path_or_none(checkpoint_path)
        self.device = get_torch_device(device, get_device, self.display_name)
        model = util.get_sam2_model(
            model_type=model_type,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            input_type='images',
        )
        # Image predictor with the preprocessing of micro-sam
        self.predictor = util.get_sam2_image_predictor(model)
        # Video predictor (3D propagation), see `_get_video_predictor`
        self._video_predictor = None
        self._init_prompts()

    def _get_video_predictor(self):
        if self._video_predictor is None:
            from micro_sam.v2 import util

            print(
                f'{self.display_name}: loading the SAM2 video predictor '
                '(propagation through the z-slices)...'
            )
            self._video_predictor = util.get_sam2_model(
                model_type=self.model_type,
                device=self.device,
                checkpoint_path=self.checkpoint_path,
                input_type='videos',
            )
        return self._video_predictor

    def _release_embeddings(self):
        segmenter = self._embeddings.get('volume_segmenter')
        if segmenter is not None:
            # Tracked objects and features of the z-slices
            segmenter.release()
        embeddings = self._embeddings.get('volume')
        if embeddings is not None:
            # Removes the temporary folder of the embeddings
            embeddings.close()
        super()._release_embeddings()

    def _set_image_embeddings(self, image, image_key, z=None):
        """Set the embeddings of the 2D image (z = None) or of the z-slice
        `z` of the z-stack `image` on the image predictor (computed the first
        time)."""
        from micro_sam.v2 import util

        self._use_embeddings_of(image_key)
        key = 'image' if z is None else ('z_slice', z)
        embeddings = self._embeddings.get(key)
        if embeddings is None:
            image_2D = image if z is None else image[z]
            embeddings = util.precompute_image_embeddings(
                self.predictor, image_2D, ndim=2, verbose=False
            )
            self._embeddings[key] = embeddings
        util.set_precomputed(self.predictor, embeddings)

    def _get_volume_segmenter(self, image, image_key):
        """`PromptableSegmentation3D` of the z-stack `image` (embeddings
        computed the first time)."""
        from micro_sam.v2 import util
        from micro_sam.v2.prompt_based_segmentation import (
            PromptableSegmentation3D
        )

        self._use_embeddings_of(image_key)
        segmenter = self._embeddings.get('volume_segmenter')
        if segmenter is not None:
            return segmenter

        predictor = self._get_video_predictor()
        print(
            f'{self.display_name}: computing the image embeddings of the '
            f'{len(image)} z-slices...'
        )
        # Stored in a temporary folder (lazy loading), as in the annotator
        embeddings = util.precompute_image_embeddings(
            predictor, image, ndim=3, verbose=True, lazy_loading=True
        )
        self._embeddings['volume'] = embeddings
        segmenter = PromptableSegmentation3D(
            predictor=predictor, volume=image,
            volume_embeddings=embeddings, device=self.device
        )
        self._embeddings['volume_segmenter'] = segmenter
        return segmenter

    def _segment_2D(self, points, labels, box, shape):
        """Segment one object on the image (or z-slice) of the embeddings
        set on the image predictor. Returns a boolean mask with `shape`."""
        kwargs = {}
        if points:
            # (y, x) --> (x, y) of SAM2
            kwargs['point_coords'] = (
                np.array(points, dtype=np.float32)[:, ::-1].copy()
            )
            kwargs['point_labels'] = np.array(labels, dtype=np.int32)
        if box is not None:
            kwargs['box'] = _sam_box(box, shape)
        masks, _, _ = self.predictor.predict(multimask_output=False, **kwargs)
        mask = _as_2D(masks)
        if mask.shape != tuple(shape):
            raise ValueError(
                f'{self.display_name} returned a mask with shape '
                f'{mask.shape} instead of {tuple(shape)}.'
            )
        return mask

    def _segment_volume_objects(self, volume, image_key, objects):
        """Yield `(prompt_id, mask)` of the objects segmented on the z-slices
        with prompts and propagated through the z-stack `volume`."""
        SizeZ = len(volume)
        height, width = volume.shape[1:3]
        # Keyword arguments of `add_prompt_set` (one per object and z-slice)
        prompt_sets = []
        for prompt_id, (positive, negative, boxes) in objects.items():
            for z in self._prompted_z(positive, boxes, SizeZ):
                points, labels, box = self._slice_prompts(
                    positive, negative, boxes, z=z
                )
                if box is None and 1 not in labels:
                    continue
                prompt_sets.append(dict(
                    frame_id=z,
                    points=np.array(points, dtype=np.float32) if points
                        else None,
                    point_labels=np.array(labels, dtype=np.int32) if points
                        else None,
                    box=box,
                    object_id=prompt_id,
                ))
        if not prompt_sets:
            return

        segmenter = self._get_volume_segmenter(volume, image_key)
        try:
            for prompt_set in prompt_sets:
                segmenter.add_prompt_set(**prompt_set)
            # {z: {prompt_id: mask}}
            video_segments = segmenter.propagate_prompts()
        finally:
            # Keep the features of the z-slices for the next prompts
            segmenter.reset_tracking()

        prompted_ids = dict.fromkeys(
            prompt_set['object_id'] for prompt_set in prompt_sets
        )
        for prompt_id in prompted_ids:
            obj_mask = np.zeros((SizeZ, height, width), dtype=bool)
            for z, masks in video_segments.items():
                mask = masks.get(prompt_id)
                if mask is None:
                    continue
                # The video predictor pads the z-slices to a square
                obj_mask[z] = _as_2D(mask)[:height, :width]
            yield prompt_id, obj_mask

    def _segment_image_objects(
            self, prompt_image, image_key, objects, is_zstack,
            segment_entire_z_stack
        ):
        if not is_zstack:
            shape = prompt_image.shape[:2]
            for prompt_id, (positive, negative, boxes) in objects.items():
                points, labels, box = self._slice_prompts(
                    positive, negative, boxes
                )
                if box is None and 1 not in labels:
                    continue
                self._set_image_embeddings(prompt_image, image_key)
                yield prompt_id, self._segment_2D(points, labels, box, shape)
            return

        SizeZ = len(prompt_image)
        if segment_entire_z_stack and SizeZ > 1:
            yield from self._segment_volume_objects(
                prompt_image, image_key, objects
            )
            return

        shape = prompt_image.shape[1:3]
        for prompt_id, (positive, negative, boxes) in objects.items():
            obj_mask = None
            for z in self._prompted_z(positive, boxes, SizeZ):
                points, labels, box = self._slice_prompts(
                    positive, negative, boxes, z=z
                )
                if box is None and 1 not in labels:
                    continue
                self._set_image_embeddings(prompt_image, image_key, z=z)
                if obj_mask is None:
                    obj_mask = np.zeros((SizeZ, *shape), dtype=bool)
                obj_mask[z] = self._segment_2D(points, labels, box, shape)
            yield prompt_id, obj_mask

    def segment(
            self,
            image,
            lab: NotParam = None,
            treat_other_objects_as_background: bool = False,
            segment_entire_z_stack: bool = True,
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
            Only for z-stacks with 3D segmentation masks. If True, the
            objects are segmented on the z-slices with prompts and then
            propagated to the other z-slices by the SAM2 video predictor
            (micro-sam "Apply to Volume"). If False, only the z-slices with
            prompts are segmented. Default is True

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
            segment_entire_z_stack
        )

def url_help():
    return 'https://computational-cell-analytics.github.io/micro-sam/micro_sam.html'
