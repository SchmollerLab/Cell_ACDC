"""


@author: jblugagne

@author: jamesroberts
"""

import os
import sys
import glob
import numpy as np

import delta
import delta.delta.utilities as utils
from delta.delta.utilities import cfg
from delta.delta.model import unet_seg
from delta.delta import model
from delta.delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape


class Model:

    def __init__(self, **kwargs):

        cfg.load_config(presets="2D")

        self.model = unet_seg(input_size=cfg.target_size_seg + (1,))
        self.model.load_weights(cfg.model_file_seg)

    def segment(self, image, **kwargs):

        inputs_folder = image

        unprocessed = sorted(
            glob.glob(inputs_folder + "/*.tif") + glob.glob(inputs_folder + "/*.png")
        )

        # Process
        while unprocessed:
            # Pop out filenames
            ps = min(4096, len(unprocessed))  # 4096 at a time
            to_process = unprocessed[0:ps]
            del unprocessed[0:ps]

            # Input data generator:
            predGene = predictGenerator_seg(
                inputs_folder,
                files_list=to_process,
                target_size=cfg.target_size_seg,
                crop=cfg.crop_windows,
            )

            # mother machine: Don't crop images into windows
            if not cfg.crop_windows:
                # Predictions:
                results = self.model.predict(predGene, verbose=1)[:, :, :, 0]

            # 2D: Cut into overlapping windows
            else:
                img = readreshape(
                    os.path.join(inputs_folder, to_process[0]),
                    target_size=cfg.target_size_seg,
                    crop=True,
                )
                # Create array to store predictions
                results = np.zeros((len(to_process), img.shape[0], img.shape[1], 1))
                # Crop, segment, stitch and store predictions in results
                for i in range(len(to_process)):
                    # Crop each frame into overlapping windows:
                    windows, loc_y, loc_x = utils.create_windows(
                        next(predGene)[0, :, :], target_size=cfg.target_size_seg
                    )
                    # We have to play around with tensor dimensions to conform to
                    # tensorflow's functions:
                    windows = windows[:, :, :, np.newaxis]
                    # Predictions:
                    pred = self.model.predict(windows, verbose=1, steps=windows.shape[0])
                    # Stich prediction frames back together:
                    pred = utils.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
                    pred = pred[np.newaxis, :, :, np.newaxis]  # Mess around with dims

                    results[i] = pred

            # Post process results (binarize + light morphology-based cleaning):
            results = postprocess(results, crop=cfg.crop_windows)

        return results