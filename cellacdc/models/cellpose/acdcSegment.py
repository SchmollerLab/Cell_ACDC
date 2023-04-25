import os
import pathlib

import numpy as np

import skimage.exposure
import skimage.filters
import skimage.measure

from cellpose import models
from cellacdc.models import CELLPOSE_MODELS
from cellacdc import printl

class AvailableModels:
    values = CELLPOSE_MODELS

class Model:
    def __init__(self, model_type: AvailableModels='cyto', net_avg=False, gpu=False):
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
        
    def setupLogger(self, logger):
        models.models_logger = logger
    
    def closeLogger(self):
        handlers = models.models_logger.handlers[:]
        for handler in handlers:
            handler.close()
            models.models_logger.removeHandler(handler)
    
    def _eval(self, image, **kwargs):
        return self.model.eval(image.astype(np.float32), **kwargs)[0]
    
    def _initialize_image(self, image):
        # See cellpose.gui.io._initialize_images
        if image.ndim > 3:
            # make tiff Z x channels x W x H
            if image.shape[0]<4:
                # tiff is channels x Z x W x H
                image = np.transpose(image, (1,0,2,3))
            elif image.shape[-1]<4:
                # tiff is Z x W x H x channels
                image = np.transpose(image, (0,3,1,2))
            # fill in with blank channels to make 3 channels
            if image.shape[1] < 3:
                shape = image.shape
                shape_to_concat = (shape[0], 3-shape[1], shape[2], shape[3])
                to_concat = np.zeros(shape_to_concat, dtype=np.uint8)
                image = np.concatenate((image, to_concat), axis=1)
            image = np.transpose(image, (0,2,3,1))
        elif image.ndim==3:
            if image.shape[0] < 5:
                image = np.transpose(image, (1,2,0))
            if image.shape[-1] < 3:
                shape = image.shape
                #if parent.autochannelbtn.isChecked():
                #    image = normalize99(image) * 255
                shape_to_concat = (shape[0], shape[1], 3-shape[2])
                to_concat = np.zeros(shape_to_concat,dtype=type(image[0,0,0]))
                image = np.concatenate((image, to_concat), axis=-1)
                image = image[np.newaxis,...]
            elif image.shape[-1]<5 and image.shape[-1]>2:
                image = image[:,:,:3]
                #if parent.autochannelbtn.isChecked():
                #    image = normalize99(image) * 255
                image = image[np.newaxis,...]
        else:
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
    
    def to_rgb_stack(self, first_ch_data, second_ch_data):
        # The 'cyto' model can work with a second channel (e.g., nucleus).
        # However, it needs to be encoded into one of the RGB channels
        # Here we put the first channel in the 'red' channel and the 
        # second channel in the 'green' channel. We then pass
        # `channels = [1,2]` to the segment method
        rgb_stack = np.zeros((*first_ch_data.shape, 3), dtype=first_ch_data.dtype)
        
        R_slice = [slice(None)]*(rgb_stack.ndim)
        R_slice[-1] = 0
        R_slice = tuple(R_slice)
        rgb_stack[R_slice] = first_ch_data

        G_slice = [slice(None)]*(rgb_stack.ndim)
        G_slice[-1] = 1
        G_slice = tuple(G_slice)

        rgb_stack[G_slice] = second_ch_data

        return rgb_stack
        
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

        isRGB = image.shape[-1] == 3 or image.shape[-1] == 4
        isZstack = (image.ndim==3 and not isRGB) or (image.ndim==4)

        if anisotropy == 0 or not isZstack:
            anisotropy = None
        
        do_3D = segment_3D_volume
        if not isZstack:
            stitch_threshold = 0.0
            segment_3D_volume = False
            do_3D = False
        
        if stitch_threshold > 0:
            do_3D = False

        if flow_threshold==0.0 or isZstack:
            flow_threshold = None

        channels = [0,0] if not isRGB else [1,2]

        eval_kwargs = {
            'channels': channels,
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

        if not segment_3D_volume and isZstack and stitch_threshold>0:
            raise TypeError(
                "`stitch_threshold` must be 0 when segmenting slice-by-slice. "
                "Alternatively, set `segment_3D_volume = True`."
            )

        # Run cellpose eval
        if not segment_3D_volume and isZstack:
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
    return 'https://cellpose.readthedocs.io/en/latest/api.html'
