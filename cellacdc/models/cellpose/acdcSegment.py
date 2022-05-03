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
        if model_type=='cyto':
            self.model = models.Cellpose(
                gpu=gpu, net_avg=net_avg, model_type=model_type
            )
        else:
            self.model = models.CellposeModel(
                gpu=gpu, net_avg=net_avg, model_type=model_type
            )

    def segment(
            self, image,
            diameter=0.0,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            net_avg=False
        ):
        # Preprocess image
        # image = image/image.max()
        # image = skimage.filters.gaussian(image, sigma=1)
        # image = skimage.exposure.equalize_adapthist(image)

        # Run cellpose eval
        lab = self.model.eval(
            image.astype(np.float32),
            channels=[0,0],
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )[0]
        return lab
