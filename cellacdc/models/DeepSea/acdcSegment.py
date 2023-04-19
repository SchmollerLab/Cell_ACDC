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

from . import deepsea_models_path, _init_model

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Model:
    def __init__(self, gpu=False):
        _transforms, torch_device, checkpoint, model = _init_model(
            'segmentation.pth', DeepSeaSegmentation, [383,512], 
            [0.5], [0.5]
        )
        self.torch_device = torch_device
        self._transforms = _transforms
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

        if is_z_stack:
            for z, img in enumerate(image):
                labels[z] = self._segment_2D_image(img)
        else:
            labels = self._segment_2D_image(image)
        return labels
        
    def _segment_2D_image(self, image: np.ndarray):
        img = myutils.to_uint8(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        tensor_img = (
            self._transforms(img)
            .to(device=self.torch_device, dtype=torch.float32)
        )
        _eval = self.model.eval()
        mask_pred, edge_pred = _eval(tensor_img.unsqueeze(0))
        mask_pred = transforms.Resize(
            image.shape, antialias=True
        ).forward(mask_pred)
        mask_pred = mask_pred.argmax(dim=1).cpu().numpy()
        lab = skimage.measure.label(np.squeeze(mask_pred))
        return lab
