import os
import numpy as np

from skimage.measure import label as skiLabel
import math
import scipy
import scipy.ndimage

import skimage.exposure
import skimage.filters
import skimage.measure

from cellpose import models
from cellacdc import user_profile_path

default_model_path = os.path.join(
    user_profile_path, 
    'acdc-Cellpose_germlineNuclei', 
    'cellpose_germlineNuclei_2023'
)

class Model:
    def __init__(
            self, 
            model_path: os.PathLike=default_model_path, 
            gpu=False
        ):
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
        """ Cellpose model for C. elegans germline nuclei. This model works on a single channel only.
        
        Parameters
        ----------
        diameter_um : float
            Expected diameter of a nucleus in micrometer
        blurfactor : float
            Sigma value of the gaussian filter used for blurring of the data.
        PhysicalSizeZ : float
            Spacing of slices in z (unit: micrometer/slice). Prepopulated from image metadata
        PhysicalSizeY : float
            Pixelsize in y (unit: micrometer/pixel). Prepopulated from image metadata
        PhysicalSizeX : float
            Pixelsize in x (unit: micrometer/pixel). Prepopulated from image metadata
        cellprob_threshold : float
            cellprob_threshold for cellpose.
        clean_borders : bool
            Remove masks that touch the top or bottom slice in z, or that are closer than 2 pixels to the edges in x or y.
                
        Returns
        -----
        np.ndarray
            Instance segmentation array with the same shape as the input image.
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
                    img_scaled[i,:,:] = scipy.ndimage.zoom(image[i,:,:],pxScale, order=3)
                    img_blur[i,:,:]=scipy.ndimage.gaussian_filter(img_scaled[i,:,:],blurfactor)

            else: 
                for i in range(image.shape[0]):
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
