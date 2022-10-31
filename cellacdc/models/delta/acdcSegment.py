"""
This script runs the segmentation U-Net
For mother machine data, it runs on images of cropped out and resized single
chambers as fed to it in Pipeline processing.

This script enables the use of delta's segmenting
in acdc.

@author: jamesroberts
"""

import os
import numpy as np
import pathlib

import delta.utilities as utils
import delta.config as cfg
from delta.model import unet_seg
from delta.data import binarizerange
from delta import assets
import skimage.transform as trans
from typing import Tuple


class Model:

    def __init__(self, model_type='2D'):
        """
        Configures data, initializes model, loads weights for model.

        Parameters
        ----------
        model_type : string
            The model name to be used for segmenting.
        """

        valid = False
        while valid == False:
            try:
                if model_type == '2D':
                    cfg.load_config(presets='2D')
                    weights = 'unet_pads_seg.hdf5'
                else:
                    cfg.load_config(presets='mothermachine')
                    weights = 'unet_moma_seg.hdf5'

                valid = True

            except ValueError:
                assets.download_assets(
                    load_models=True,
                    load_sets=False,
                    load_evals=False,
                    config_level='local'
                )

        user_path = pathlib.Path.home()
        model_path = os.path.join(str(user_path), f'acdc-delta')

        self.weights_path = os.path.join(model_path, weights)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Weights file not found in {model_path}')

        # Initialize Model
        self.model = unet_seg(input_size=cfg.target_size_seg+(1,))
        self.model.load_weights(self.weights_path)


    def delta_preprocess(self,
                         image,
                         target_size: Tuple[int, int] = (256, 32),
                         binarize: bool = False,
                         order: int = 1,
                         rangescale: bool = True,
                         crop: bool = False,
                         ):
        """
        Takes image and reformat it

        (Much like delta.data.readreshape)

        Parameters
        ----------
        image: numpy.array
            Supplied by acdc (2D)
        target_size : tupe of int or None, optional
            Size to reshape the image.
            The default is (256,32).
        binarize : bool, optional
            Use the binarizerange() function on the image.
            The default is False.
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

        if binarize:
            img = binarizerange(img)
        if rangescale:
            if np.ptp(img) != 0:
                img = (img - np.min(img)) / np.ptp(img)
        if np.max(img) == 255:
            img = img / 255
        return img

    def segment(self,
                image,
                model_type='2D'):
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

        # 2D: Cut into overlapping windows
        img = self.delta_preprocess(image=image,
                                        target_size=cfg.target_size_seg,
                                        crop=True)

        if image.ndim == 2:

            # Process image to use for delta
            image = self.delta_preprocess(image=image,
                                          target_size=cfg.target_size_seg,
                                          crop=cfg.crop_windows)

            # Change Dimensions to 4D numpy array
            image = np.reshape(image, (1,) + image.shape + (1,))

            # mother machine: Don't crop images into windows
            if not cfg.crop_windows:
                # Predictions:
                results = self.model.predict(image, verbose=1)[0, :, :, 0]
                results = trans.resize(results, original_shape, anti_aliasing=True, order=1)

                # Post process results (binarize + light morphology-based cleaning):
                # prediction = postprocess(results, crop=cfg.crop_windows)

                # Label the cells using prediction
                lab = utils.label_seg(seg=results)
                print(np.unique(lab))

            else:

                # Create array to store predictions
                results = np.zeros((1, img.shape[0], img.shape[1], 1))
                print(img.shape)
                print(image.shape)
                print(results.shape)

                # Crop, segment, stitch and store predictions in results
                # This will only happen once because the input is a single image
                for i in range(1):

                    # Crop each frame into overlapping windows:
                    windows, loc_y, loc_x = utils.create_windows(
                        image[0, :, :, 0], target_size=cfg.target_size_seg
                    )
                    windows = windows[:, :, :, np.newaxis]

                    # Predictions:
                    pred = self.model.predict(windows, verbose=1, steps=windows.shape[0])
                    # Stich prediction frames back together:
                    pred = utils.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
                    pred = pred[:, :, np.newaxis]

                    results[i] = pred

                # Label the cells using prediction
                lab = utils.label_seg(seg=results[0, :, :, 0])
                print(np.unique(lab))

        return lab.astype(np.uint16)
