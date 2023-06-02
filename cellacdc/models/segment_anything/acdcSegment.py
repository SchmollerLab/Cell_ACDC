import os
import cv2

import numpy as np

import skimage.measure

from . import model_types, sam_models_path

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from cellacdc import myutils

class AvailableModels:
    values = list(model_types.keys())

class Model:
    def __init__(
            self, 
            model_type: AvailableModels='Large', 
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1,
            gpu=False
        ):
        if gpu:
            from cellacdc import is_mac_arm64
            if is_mac_arm64:
                device = 'cpu'
            else:
                device = 'cuda'
        else:
            device = 'cpu'
        
        model_type, sam_checkpoint = model_types[model_type]
        sam_checkpoint = os.path.join(sam_models_path, sam_checkpoint)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            
        sam.to(device=device)
        self.model = SamAutomaticMaskGenerator(
            sam, 
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
    
    def segment(self, image: np.ndarray, automatic_removal_of_background: bool=True) -> np.ndarray:
        is_rgb_image = image.shape[-1] == 3 or image.shape[-1] == 4
        is_z_stack = (image.ndim==3 and not is_rgb_image) or (image.ndim==4)
        if is_rgb_image:
            labels = np.zeros(image.shape[:-1], dtype=np.uint32)
        else:
            labels = np.zeros(image.shape, dtype=np.uint32)
        if is_z_stack:
            for z, img in enumerate(image):
                labels[z] = self._segment_2D_image(img)
            labels = skimage.measure.label(labels>0)
        else:
            labels = self._segment_2D_image(image)
        if automatic_removal_of_background:
            labels = self._remove_background(labels)
        return labels

    def _segment_2D_image(self, image: np.ndarray) -> np.ndarray:
        img = myutils.to_uint8(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = self.model.generate(img)
        labels = np.zeros(image.shape[:3], dtype=np.uint32)
        for id, mask in enumerate(masks):
            obj_image = mask['segmentation']
            labels[obj_image] = id+1
        return labels

    def _remove_background(self, labels: np.ndarray) -> np.ndarray:
        border_mask = np.ones(labels.shape, dtype=bool)
        border_slice = tuple([slice(2,-2) for _ in range(labels.ndim)])
        border_mask[border_slice] = False
        border_ids, counts = np.unique(labels[border_mask], return_counts=True)
        max_count_idx = list(counts).index(counts.max())
        largest_border_id = border_ids[max_count_idx]
        labels[labels == largest_border_id] = 0
        return labels


def url_help():
    return 'https://github.com/facebookresearch/segment-anything'
