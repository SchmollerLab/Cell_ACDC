"""
This script runs the segmentation U-Net
for 2D images.

Initializes Model for

@author: jroberts / jamesr787
"""

import numpy as np
from typing import Tuple

import skimage.transform as trans

from delta import utilities as utils
from delta import config as cfg
from delta.model import unet_seg
from delta.assets import download_assets


class Model:

    def __init__(self,
                 model_type='2D or mothermachine'):
        """
        Configures data, initializes model, loads weights for model.

        Parameters
        ----------
        model_type : string
            The model name to be used for segmenting.
            2D or mothermachine

        Returns
        -------
        None.
        """

        while True:
            try:
                # Loads Presets for '2D' or 'mothermachine' images
                cfg.load_config(presets=model_type)
                self.target_size = cfg.target_size_seg

                # Initialize Model and Load Weights with 2D presets
                self.model = unet_seg(input_size=self.target_size + (1,))
                self.model.load_weights(cfg.model_file_seg)

                break

            except ValueError:
                # Downloads model weights and configuration files for 2D and mothermachine
                download_assets(load_models=True,
                                load_sets=False,
                                load_evals=False,
                                config_level='local')

    def delta_preprocess(self,
                         image,
                         target_size: Tuple[int, int] = (256, 32),
                         order: int = 1,
                         rangescale: bool = True,
                         crop: bool = False,
                         ):
        """
        Takes image and reformat it

        (A lot like delta.data.readreshape)

        Parameters
        ----------
        image: numpy.array
            Supplied by acdc (2D)
        target_size : tupe of int or None, optional
            Size to reshape the image.
            The default is (256,32).
        order : int, optional
            interpolation order (see skimage.transform.warp doc).
            0 is nearest neighbor
            1 is bilinear
            The default is 1.
        rangescale : bool, optional
            Scale array image values to 0-1 if True.
            The default is True.
        crop : bool
            Will resize image if True.
            The default is False.

        Returns
        -------
        img : numpy 2d array of floats
            Loaded array.

        """
        i = image

        # For DeLTA mothermachine, all images are resized in 256x32
        if not crop:
            img = trans.resize(i, target_size, anti_aliasing=True, order=order)
        # For DeLTA 2D, black space is added if img is smaller than target_size
        else:
            fill_shape = [
                target_size[j] if i.shape[j] < target_size[j] else i.shape[j]
                for j in range(2)
            ]
            img = np.zeros((fill_shape[0], fill_shape[1]))
            img[0: i.shape[0], 0: i.shape[1]] = i

        if rangescale:
            if np.ptp(img) != 0:
                img = (img - np.min(img)) / np.ptp(img)
        if np.max(img) == 255:
            img = img / 255
        return img

    def segment(self,
                image):
        """
        Uses initialized model with weights and image to
        label cells in segmentation mask.

        (Much like delta.segmentation.py)

        Parameters
        ----------
        image : numpy array
            single image from input.

        Returns
        -------
        lab : 2D numpy array of uint16
                Labelled image. Each cell in the image is marked by adjacent pixels
                with values given by cell number.
        """

        # Find original shape of image before processing
        original_shape = image.shape

        if image.ndim != 2:
            raise ValueError(
                f"""Delta only works with 2 dimensional images."""
            )

        # 2D: Cut into overlapping windows
        img = self.delta_preprocess(image=image,
                                    target_size=self.target_size,
                                    crop=True)

        # Process image to use for delta
        image = self.delta_preprocess(image=image,
                                      target_size=self.target_size,
                                      crop=cfg.crop_windows)

        # Change Dimensions to 4D numpy array
        image = np.reshape(image, (1,) + image.shape + (1,))

        # mother machine: Don't crop images into windows
        if not cfg.crop_windows:

            # Predictions:
            results = self.model.predict(image, verbose=1)[0, :, :, 0]

            # Resize to the original shape
            results = trans.resize(results, original_shape, anti_aliasing=True, order=1)

            # Label the cells using prediction
            lab = utils.label_seg(seg=results)

        # For 2D images
        else:
            # Create array to store predictions
            results = np.zeros((1, img.shape[0], img.shape[1], 1))

            # Crop, segment, stitch and store predictions in results
            # Crop each frame into overlapping windows:
            windows, loc_y, loc_x = utils.create_windows(
                image[0, :, :, 0], target_size=self.target_size
            )
            windows = windows[:, :, :, np.newaxis]

            # Predictions:
            pred = self.model.predict(windows, verbose=1, steps=windows.shape[0])
            # Stich prediction frames back together:
            pred = utils.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
            pred = pred[:, :, np.newaxis]

            results[0] = pred

            # Label the cells using prediction
            lab = utils.label_seg(seg=results[0, :, :, 0])

        return lab.astype(np.uint32)

def url_help():
    return 'https://gitlab.com/dunloplab/delta'