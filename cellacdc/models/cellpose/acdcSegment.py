import os
import pathlib

import numpy as np

import skimage.exposure
import skimage.filters

from cellpose import models
from cellacdc.models import CELLPOSE_MODELS

help_url = 'https://cellpose.readthedocs.io/en/latest/api.html'

class Model:
    def __init__(self, model_type='cyto', net_avg=False, gpu=False):
        if model_type not in CELLPOSE_MODELS:
            err_msg = (
                f'"{model_type}" not available. '
                f'Available models are {CELLPOSE_MODELS}'
            )
            raise NameError(err_msg)
        try:
            device, gpu = models.assign_device(True, False)
            self.model = models.Cellpose(
                gpu=gpu, device=device, model_type=model_type, torch=True
            )
        except Exception as e:
            self.model = models.Cellpose(
                gpu=gpu, model_type=model_type, net_avg=net_avg
            )

    def segment(
            self, image,
            diameter=0.0,
            flow_threshold=0.4,
            cellprob_threshold=0.0
        ):
        # Preprocess image
        image = image/image.max()
        image = skimage.filters.gaussian(image, sigma=1)
        image = skimage.exposure.equalize_adapthist(image)

        # Run cellpose eval
        lab, flows, _, _ = self.model.eval(
            image,
            channels=[0,0],
            diameter=diameter,
            mask_threshold=mask_threshold
        )
        return lab
