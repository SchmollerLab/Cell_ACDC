import os
import pathlib
import numpy as np

from skimage.measure import label as skiLabel
import math
import cv2
import skimage
import scipy
import scipy.ndimage
import cellpose

import skimage.exposure
import skimage.filters
import skimage.measure

from cellpose import models
from cellacdc import printl
from cellacdc.models import CELLPOSE_MODELS

class AvailableModels:
    values = CELLPOSE_MODELS


class Model:
    def __init__(self, model_name='cellpose_germlineNuclei_2023', gpu=False):
        script_path = os.path.abspath(__file__)
        model_path = os.path.join(os.path.dirname(script_path), model_name)
        self.model = models.CellposeModel(
                gpu=gpu, diam_mean=30, pretrained_model=model_path
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
        
        if image.ndim < 4:
            image = image[:,:,:,np.newaxis]
        return image
    
        
    def segment(
            self, image,
            diameter_um=3.5,
            blurfactor=2.50,
            PhysicalSizeZ = 1.0001,
            PhysicalSizeY = 1.0001,
            PhysicalSizeX = 1.0001,
            cellprob_threshold=0.0,
            clean_borders=False 
        ):               
        """Simone's custom cellpose model.
        
        Parameters
        ----------
        blurfactor : float
            Sigma value of the gaussian filter.

        Notes
        -----
        This model works on single channel only. 
        """        


        # Preprocess image
        # image = image/image.max()
        # image = skimage.filters.gaussian(image, sigma=1)
        # image = skimage.exposure.equalize_adapthist(image)
        zspacing = PhysicalSizeZ
        xysize = np.mean([PhysicalSizeX, PhysicalSizeY])
        
        isRGB = image.shape[-1] == 3 or image.shape[-1] == 4
        if isRGB:
            raise TypeError(
                "This model was trained for 1 channel only. Please specify a single channel (DNA or synaptonemal complex/axis staining). "
            )
        
        isZstack = (image.ndim==3 and not isRGB) or (image.ndim==4)

        anisotropy = math.ceil(abs(zspacing/xysize))
        pxScale=xysize*30/diameter_um

        
        do_3D = True
        
        #if stitch_threshold > 0:
        #    do_3D = False


        channels = [0,0] 



        # Run cellpose eval
        if not isZstack:
            raise TypeError(
                "This script is for 3D data (at least 5 slices) only. If needed, please modify the script to segment 2D data."
            )
        else:
            img_scaled=np.zeros((image.shape[0],round(image.shape[1]*pxScale),round(image.shape[2]*pxScale)))
            img_blur=np.zeros((img_scaled.shape))
            image[image==0] = np.quantile(image[image>0],0.01)

            if pxScale > 1:
                for i in range(image.shape[0]):
                    #img_blur[i,:,:]=np.uint16(cv2.GaussianBlur(image[i,:,:],(0,0),blurfactor,cv2.BORDER_DEFAULT))
                    img_scaled[i,:,:] = scipy.ndimage.zoom(image[i,:,:],pxScale, order=3)
                    img_blur[i,:,:]=scipy.ndimage.gaussian_filter(image_scaled[i,:,:],blurfactor)

            else: 
                for i in range(image.shape[0]):
                    #img_blur[i,:,:]=np.uint16(cv2.GaussianBlur(image[i,:,:],(0,0),blurfactor,cv2.BORDER_DEFAULT))
                    img_scaled[i,:,:] = scipy.ndimage.zoom(image[i,:,:],pxScale, order=3)
                    img_blur[i,:,:]=scipy.ndimage.gaussian_filter(img_scaled[i,:,:],blurfactor)
            img_blur = self._initialize_image(img_blur)
            labels_scaled, flows_blur, styles_blur = self.model.eval(img_blur.astype(np.uint16),
                                                diameter=30, 
                                                channels=channels, do_3D=True, 
                                                anisotropy=anisotropy,
                                                batch_size=3,
                                                cellprob_threshold=cellprob_threshold)
            
            labels=np.zeros(image.shape,dtype=labels_scaled.dtype)
            for i in range(image.shape[0]):
                labels[i,:,:]=scipy.ndimage.zoom(labels_scaled[i,:,:],(image.shape[1]/labels_scaled.shape[1],image.shape[2]/labels_scaled.shape[2]),order=0)

            if clean_borders:
                idx = np.unique(np.concatenate([np.unique(labels[-1,:,:][labels[-1,:,:]>0]),np.unique(labels[0,:,:][labels[0,:,:]>0]),
                                    np.unique(labels[:,0:2,:][labels[:,0:2,:]>0]),np.unique(labels[:,-3:-1,:][labels[:,-3:-1,:]>0]),
                                    np.unique(labels[:,:,0:2][labels[:,:,0:2]>0]),np.unique(labels[:,:,-3:-1][labels[:,:,-3:-1]>0]),]))
    
                labels[np.isin(labels,idx)] = 0

        return labels

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'
