import os

import numpy as np

import skimage.exposure

from .unet import model
from .unet import neural_network
from .unet import segment

class Model:
    def __init__(self, is_phase_contrast=True):
        # Initialize model
        self.model = model.unet(
            pretrained_weights=None,
            input_size=(None,None,1)
        )

        # Get the path where the weights are saved.
        # We suggest saving the weights files into a 'model' subfolder
        script_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_path, 'model')

        if is_phase_contrast:
            weights_fn = 'unet_weights_batchsize_25_Nepochs_100_SJR0_10.hdf5'
        else:
            weights_fn = 'unet_weights_BF_batchsize_25_Nepochs_100_SJR_0_1.hdf5'

        weights_path = os.path.join(model_path, weights_fn)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Weights file not found in {model_path}')

        self.model.load_weights(weights_path)

    def segment(self, image, thresh_val=0.0, min_distance=10):
        # Preprocess image
        image = image/image.max()
        image = skimage.exposure.equalize_adapthist(image)

        if thresh_val == 0:
            thresh_val = None

        # pad with zeros such that is divisible by 16
        (nrow, ncol) = image.shape
        row_add = 16-nrow%16
        col_add = 16-ncol%16
        pad_info = ((0, row_add), (0, col_add))
        padded = np.pad(image, pad_info, 'constant')
        x = padded[np.newaxis,:,:,np.newaxis]

        prediction = self.model.predict(x, batch_size=1)[0,:,:,0]

        # remove padding with 0s
        prediction = prediction[0:-row_add, 0:-col_add]
        thresh = neural_network.threshold(prediction, thresh_val=thresh_val)
        lab = segment.segment(
            thresh, prediction, min_distance=min_distance
        ).astype(np.uint16)
        return lab
