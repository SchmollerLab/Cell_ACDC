import os
import pathlib

import numpy as np

import skimage.exposure
import skimage.filters
import skimage.measure

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
            min_size=-1,
            normalize=True,
            segment_3D_volume=False
        ):
        # Preprocess image
        # image = image/image.max()
        # image = skimage.filters.gaussian(image, sigma=1)
        # image = skimage.exposure.equalize_adapthist(image)

        # Run cellpose eval
        if not segment_3D_volume and image.ndim == 3:
            labels = np.zeros(image.shape, dtype=np.uint16)
            for i, _img in enumerate(image):
                lab = self.model.eval(
                    _img.astype(np.float32),
                    channels=[0,0],
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    min_size=min_size,
                    normalize=normalize,
                    do_3D=segment_3D_volume
                )[0]
                labels[i] = _lab2D
            labels = skimage.measure.label(labels>0)
        else:
            labels = self.model.eval(
                image.astype(np.float32),
                channels=[0,0],
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
                normalize=normalize,
                do_3D=segment_3D_volume
            )[0]
        return labels

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'
