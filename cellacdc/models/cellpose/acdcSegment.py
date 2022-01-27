import os
import pathlib

import numpy as np

import skimage.exposure
import skimage.filters

from . import models

class Model:
    def __init__(self, model_type='cyto'):
        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_path, 'model')

        model_dir = pathlib.Path(model_path)

        device, gpu = models.assign_device(True, False)
        self.model = models.Cellpose(
            gpu=gpu, device=device, model_type=model_type, torch=True,
            model_dir=model_dir
        )

    def segment(self, image,
                diameter=0.0,
                flow_threshold=0.4,
                cellprob_threshold=0.0):
        # Preprocess image
        image = image/image.max()
        image = skimage.filters.gaussian(image, sigma=1)
        image = skimage.exposure.equalize_adapthist(image)

        # Run cellpose eval
        lab, flows, _, _ = self.model.eval(
            image,
            channels=[0,0],
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        return lab

def url_help():
    return 'https://colab.research.google.com/github/MouseLand/cellpose/blob/master/notebooks/Cellpose_2D_v0_1.ipynb#scrollTo=Rr0UozRm42CA'
