import os
import pathlib

import numpy as np

import skimage.exposure
import skimage.filters

from . import models

CELLPOSE_MODELS = [
    'cyto',
    'nuclei',
    'cyto2',
    'bact',
    'bact_omni',
    'cyto2_omni'
]

class Model:
    def __init__(self, model_type='cyto'):
        if model_type not in CELLPOSE_MODELS:
            err_msg = (
                f'"{model_type}" not available. '
                f'Avilable models are {CELLPOSE_MODELS}'
            )
            raise NameError(err_msg)
        device, gpu = models.assign_device(True, False)
        self.model = models.Cellpose(
            gpu=gpu, device=device, model_type=model_type, torch=True
        )

    def segment(
            self, image,
            diameter=0.0,
            mask_threshold=0.0
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

def url_help():
    return 'https://colab.research.google.com/github/MouseLand/cellpose/blob/master/notebooks/Cellpose_2D_v0_1.ipynb#scrollTo=Rr0UozRm42CA'
