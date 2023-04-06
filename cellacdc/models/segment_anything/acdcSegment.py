import os
import cv2

import numpy as np

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
            from cellacdc import is_mac
            import platform
            cpu = platform.processor()
            if is_mac and cpu == 'arm':
                device = 'mps'
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
    
    def segment(self, image, automatic_remove_background=True):
        isRGB = image.shape[-1] == 3 or image.shape[-1] == 4
        isZstack = (image.ndim==3 and not isRGB) or (image.ndim==4)

        print('Generating masks...')

        labels = np.zeros(image.shape, dtype=np.uint32)
        if isZstack:   
            for z, img in enumerate(image):
                img = myutils.to_uint8(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                masks = self.model.generate(img)
                for id, mask in enumerate(masks):
                    obj_image = mask['segmentation']
                    labels[z][obj_image] = id+1
        else:
            img = myutils.to_uint8(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            masks = self.model.generate(img)
            for id, mask in enumerate(masks):
                obj_image = mask['segmentation']
                labels[obj_image] = id+1
        
        if automatic_remove_background:
            border_mask = np.ones(labels.shape, dtype=bool)
            border_slice = tuple([slice(1,-1) for _ in range(labels.ndim)])
            border_mask[border_slice] = False
            border_ids, counts = np.unique(labels[border_mask])
            max_count_idx = list(counts).index(counts.max())
            largest_border_id = border_ids[max_count_idx]
            labels[labels == largest_border_id] = 0
        return labels

def url_help():
    return 'https://github.com/facebookresearch/segment-anything'