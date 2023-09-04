import os
import numpy as np

import skimage.exposure
import skimage.measure

import torch
import gc

from yeaz.unet.model_pytorch import UNet
from yeaz.unet import segment as yeaz_segment
import yeaz.unet.neural_network as nn

from cellacdc import myutils, printl

class AvailableModels:
    values = [
        'Phase contrast',
        'Bright-field',
        'Fission yeast'
    ]
    mapper = {
        'Phase contrast': 'weights_budding_PhC_multilab_0_1',
        'Bright-field': 'weights_budding_BF_multilab_0_1',
        'Fission yeast': 'weights_fission_multilab_0_2'
    }

class Model:
    def __init__(self, model_type: AvailableModels='Phase contrast'):
        # Initialize model
        filename = AvailableModels.mapper[model_type]
        _, model_folderpath = myutils.get_model_path(
            'YeaZ_v2', create_temp_dir=False
        )
        pretrained_weights = os.path.join(model_folderpath, filename)
        
        self.model = UNet()
        self.model.load_state_dict(torch.load(pretrained_weights))
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self._is_gpu = True
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            self._is_gpu = True
        else:
            device = torch.device('cpu')
            self._is_gpu = False
        
        self.device = device
        self.model = self.model.to(device)
    
    def _segment_img_2D(self, image, thresh_val=0.0, min_distance=10):
        # Preprocess image
        image = self._preprocess_image(image).astype(np.float32)
        
        # pad with zeros such that is divisible by 16
        (nrow, ncol) = image.shape
        row_add = 16-nrow%16
        col_add = 16-ncol%16
        pad_width = ((0, row_add), (0, col_add))
        padded = np.pad(image, pad_width)
        
        padded = torch.from_numpy(padded)
        if self._is_gpu:
            padded = padded.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            # Convert input tensor to PyTorch tensor
            input_tensor = padded.unsqueeze(0).unsqueeze(0).float()
            # Pass input through the model
            output_tensor = self.model.forward(input_tensor)
            # Convert output tensor to NumPy array
            output_array = output_tensor.cpu().detach().numpy()
        result = output_array[0, 0, :, :]
        
        if self._is_gpu:
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                pass
        prediction = result[:nrow, :ncol]
        
        if thresh_val == 0:
            thresh_val = None
        
        thresholded = nn.threshold(prediction, th=thresh_val)
        lab = yeaz_segment.segment(
            thresholded, prediction, min_distance=min_distance
        )
        return lab.astype(np.uint32)
    
    def _preprocess_image(self, image, tqdm_pbar=None):
        image = myutils.img_to_float(image)
        image = skimage.exposure.equalize_adapthist(image)
        if tqdm_pbar is not None:
            tqdm_pbar.emit(1)
        return image

    def segment(self, image, thresh_val=0.0, min_distance=10):
        if image.ndim == 3:
            labels = np.zeros(image.shape, dtype=np.uint32)
            for z, img in enumerate(image):
                lab = self._segment_img_2D(
                    img, thresh_val=thresh_val, min_distance=min_distance
                )
                labels[z] = lab
            labels = skimage.measure.label(labels>0)
        else:
            labels = self._segment_img_2D(
                image, thresh_val=thresh_val, min_distance=min_distance
            )
        return labels

def url_help():
    return 'https://github.com/rahi-lab/YeaZ-GUI'