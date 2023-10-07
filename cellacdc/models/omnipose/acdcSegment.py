import os
import pathlib

import numpy as np

import skimage.exposure
import skimage.filters
import skimage.measure

from cellpose_omni import models
from cellacdc import printl

from omnipose.core import OMNI_MODELS

class AvailableModels:
    values = OMNI_MODELS

class Model:
    def __init__(
            self, 
            model_type: AvailableModels='bact_phase_omni', 
            net_avg=False, 
            gpu=False
        ):
        if model_type not in OMNI_MODELS:
            err_msg = (
                f'"{model_type}" not available. '
                f'Available models are {OMNI_MODELS}'
            )
            raise NameError(err_msg)
        self.model = models.Cellpose(
            gpu=gpu, net_avg=net_avg, model_type=model_type
        )
    
    def _eval(self, image, **kwargs):
        kwargs['omni'] = True
        return self.model.eval(image.astype(np.float32), **kwargs)[0]
    
    def _initialize_image(self, image):
        # See cellpose.io._initialize_images
        if image.ndim == 2:
            image = image[np.newaxis,...]      
        
        img_min = image.min() 
        img_max = image.max()
        image = image.astype(np.float32)
        image -= img_min
        if img_max > img_min + 1e-3:
            image /= (img_max - img_min)
        image *= 255
        if image.ndim < 4:
            image = image[:,:,:,np.newaxis]
        return image
        
    def segment(
            self, image,
            diameter=0.0,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            stitch_threshold=0.0,
            min_size=15,
            anisotropy=0.0,
            normalize=True,
            resample=True,
            segment_3D_volume=False            
        ):
        # Preprocess image
        # image = image/image.max()
        # image = skimage.filters.gaussian(image, sigma=1)
        # image = skimage.exposure.equalize_adapthist(image)
        if anisotropy == 0 or image.ndim == 2:
            anisotropy = None
        
        do_3D = segment_3D_volume
        if image.ndim == 2:
            stitch_threshold = 0.0
            segment_3D_volume = False
            do_3D = False
        
        if stitch_threshold > 0:
            do_3D = False
        
        if flow_threshold==0.0 or image.ndim==3:
            flow_threshold = None

        eval_kwargs = {
            'channels': [0,0],
            'diameter': diameter,
            'flow_threshold': flow_threshold,
            'cellprob_threshold': cellprob_threshold,
            'stitch_threshold': stitch_threshold,
            'min_size': min_size,
            'normalize': normalize,
            'do_3D': do_3D,
            'anisotropy': anisotropy,
            'resample': resample
        }

        # Run cellpose eval
        if not segment_3D_volume and image.ndim == 3:
            labels = np.zeros(image.shape, dtype=np.uint32)
            for i, _img in enumerate(image):
                _img = self._initialize_image(_img)
                lab = self._eval(_img, **eval_kwargs)
                labels[i] = lab
            labels = skimage.measure.label(labels>0)
        else:
            image = self._initialize_image(image)  
            labels = self._eval(image, **eval_kwargs)
        return labels

def url_help():
    return 'https://omnipose.readthedocs.io/'
