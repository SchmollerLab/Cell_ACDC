import cv2

import numpy as np

from PIL import Image

from deepsea.model import DeepSeaTracker
from deepsea.utils import track_cells

from cellacdc import myutils, printl
from cellacdc.models.DeepSea import _init_model, _resize_img
from cellacdc.models.DeepSea import image_size as segm_image_size
from cellacdc.models.DeepSea import image_means as segm_image_means
from cellacdc.models.DeepSea import image_stds as segm_image_stds

from . import image_size, image_means, image_stds

class tracker:
    def __init__(self, gpu=False):
        _transforms, torch_device, checkpoint, model = _init_model(
            'tracking.pth', DeepSeaTracker, image_size, 
            image_means, image_stds
        )
        self.torch_device = torch_device
        self._transforms = _transforms
        self._checkpoint = checkpoint
        self.model = model
    
    def track(self, segm_video, image, signals=None):
        labels_list = []
        resize_img_list = []
        for img, lab in image:
            img = myutils.to_uint8(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            resized_img = _resize_img(
                img, segm_image_size, segm_image_means, segm_image_stds,
                self.torch_device
            )
            resize_lab = _resize_img(
                lab, segm_image_size, segm_image_means, segm_image_stds,
                self.torch_device
            ).round().astype(np.uint32)
            
            resize_img_list.append(resize_lab)
            labels_list.append(resize_lab)
        
        tracked_video, tracked_centroids, tracked_imgs = (
            track_cells(
                labels_list, resize_img_list, self.model, 
                self.torch_device, transforms=self._transforms
            )
        )
        return tracked_video