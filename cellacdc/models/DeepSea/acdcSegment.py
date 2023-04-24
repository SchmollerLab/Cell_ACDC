import os
import cv2

import numpy as np
import torch
import torchvision.transforms as transforms
import random
from PIL import Image

import skimage.measure

from deepsea.model import DeepSeaSegmentation
from cellacdc import myutils, printl

from . import _init_model
from . import _get_segm_transforms

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Model:
    def __init__(self, gpu=False):
        torch_device, checkpoint, model = _init_model(
            'segmentation.pth', DeepSeaSegmentation, gpu=gpu
        )
        self.torch_device = torch_device
        self._transforms = _get_segm_transforms()
        self._checkpoint = checkpoint
        self.model = model
    
    def segment(self, image: np.ndarray):
        is_rgb_image = image.shape[-1] == 3 or image.shape[-1] == 4
        is_z_stack = (image.ndim==3 and not is_rgb_image) or (image.ndim==4)
        labels = np.zeros(image.shape, dtype=np.uint32)
        if is_rgb_image:
            labels = np.zeros(image.shape[:-1], dtype=np.uint32)
        else:
            labels = np.zeros(image.shape, dtype=np.uint32)

        Y, X = labels.shape[-2:]

        if is_z_stack:
            for z, img in enumerate(image):
                labels[z] = self._segment_2D_image(img, (Y, X))
        else:
            labels = self._segment_2D_image(image, (Y, X))
        return labels
        
    def _segment_2D_image(self, img: np.ndarray, grayscale_img_shape):
        img = (255 * ((img - img.min()) / img.ptp())).astype(np.uint8)
        tensor_img = (
            self._transforms(img)
            .to(device=self.torch_device, dtype=torch.float32)
        )
        _eval = self.model.eval()
        mask_pred, edge_pred = _eval(tensor_img.unsqueeze(0))

        mask_pred = transforms.Resize(
            grayscale_img_shape, antialias=True
        ).forward(mask_pred)
        mask_pred = mask_pred.argmax(dim=1).cpu().numpy()[0, :, :]
        mask_bool = mask_pred > 0
        lab = skimage.measure.label(np.squeeze(mask_bool))
        
        return lab
