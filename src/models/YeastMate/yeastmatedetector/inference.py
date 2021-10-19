import os
import json
import torch
import numpy as np
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from skimage.exposure import rescale_intensity

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN

from .models import MultiMaskRCNNConvUpsampleHead as MultiR
from .multimaskrcnn import MultiMaskMRCNN
from .postprocessing import postproc_multimask
from .utils import initialize_new_config_values

class YeastMatePredictor():
    def __init__(self, cfg, weights=None):
        self.cfg = get_cfg()
        
        self.cfg = initialize_new_config_values(self.cfg)

        self.cfg.merge_from_file(cfg)

        if not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = 'cpu'

        if weights is not None:
            self.cfg.MODEL.WEIGHTS = weights

        self.model = MultiMaskMRCNN(self.cfg)
        self.model.to(torch.device(self.cfg.MODEL.DEVICE))
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

    @staticmethod
    def image_to_tensor(image):
        height, width = image.shape

        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)

        image = torch.as_tensor(image)  
        image = {"image": image, "height": height, "width": width}

        return image

    @staticmethod
    def rescale_and_normalize(image, pixel_size, reference_pixel_size=110, lower_quantile=1.5, upper_quantile=98.5):
        lq, uq = np.percentile(image, [lower_quantile, upper_quantile])
        image = rescale_intensity(image, in_range=(lq,uq), out_range=(0,1))
        image = image.astype(np.float32)

        scale_factor = pixel_size/reference_pixel_size

        if scale_factor != 1.0:
            image = rescale(image, scale_factor, preserve_range=True)
        
        return image

    @staticmethod
    def postprocess_instances(instances, possible_comps, optional_object_score_threshold=0.15, parent_override_threshold=2, score_thresholds={0:0.9, 1:0.5, 2:0.5}):
        possible_comps_dict = {}
        for n in range(len(possible_comps)):
            new_comps = {}
            for key in possible_comps[n]:
                new_comps[int(key)] = possible_comps[n][key]

            possible_comps_dict[n+1] = new_comps

        things, mask = postproc_multimask(instances, possible_comps_dict, \
            optional_object_score_threshold=optional_object_score_threshold,\
                 parent_override_thresh=parent_override_threshold, \
                     score_thresholds=score_thresholds)

        return things, mask

    @staticmethod
    def unscale_results(things, mask, original_shape):
        scale_factor = original_shape[0]/mask.shape[0]

        if scale_factor != 1.0:
            for key, thing in things.items():
                things[key]['box'] = [int(x*scale_factor) for x in thing['box']]

            mask = resize(mask, original_shape, preserve_range=True, anti_aliasing=False, order=0)

        return things, mask


    def inference(self, image, score_thresholds = {0:0.9, 1:0.75, 2:0.75}, pixel_size=110, reference_pixel_size=110, lower_quantile=1.5, upper_quantile=98.5):

        original_shape = image.shape
        image = self.rescale_and_normalize(image, pixel_size, reference_pixel_size, lower_quantile, upper_quantile)
        image = self.image_to_tensor(image)

        with torch.no_grad():
            instances = self.model([image])[0]['instances']

        possible_comps = self.cfg.POSTPROCESSING.POSSIBLE_COMPS
        optional_object_score_threshold = self.cfg.POSTPROCESSING.OPTIONAL_OBJECT_SCORE_THRESHOLD
        parent_override_threshold = self.cfg.POSTPROCESSING.PARENT_OVERRIDE_THRESHOLD

        things, mask = things, mask = self.postprocess_instances(instances, possible_comps, \
            optional_object_score_threshold=optional_object_score_threshold, \
                parent_override_threshold=parent_override_threshold, \
                    score_thresholds=score_thresholds)

        things, mask = self.unscale_results(things, mask, original_shape)
        
        return things, mask
