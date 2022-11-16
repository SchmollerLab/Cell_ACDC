"""
This script runs the tracking U-Net
for 2D images.

@author: jroberts / jamesr787
"""

# Modules:
import os
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, Tuple, Union

import delta.utilities as utils
import delta.config as cfg
from delta.assets import download_assets
from delta import pipeline


class FakeReader:

    def __init__(self,
                 x,
                 y,
                 channels,
                 timepoints,
                 filename,
                 original_video,
                 starting_frame):
        """
        Initialize experiment reader

        Parameters
        ----------
        x : int
            Size of images along X axis
        y : int
            Size of images along Y axis
        channels : int
            Number of imaging channels
        timepoints : int
            Number of frames
        filename : str
            File or folder name for the experiment
        original_video : np.array
            3D numpy array of original images
        starting_frame : int
            starting frame to track

        Returns
        -------
        None.

        """
        self.x = x
        self.y = y
        self.channels = channels
        self.timepoints = timepoints
        self.filename = filename
        self.original_video = original_video
        self.starting_frame = starting_frame

    def getframes(self,
                  squeeze_dimensions: bool = True,
                  resize: Tuple[int, int] = None,
                  rescale: Tuple[float, float] = None,
                  globalrescale: Tuple[float, float] = None,
                  rotate: float = None,
                  **kwargs
                  ):
        """
        Get frames from experiment.

        Parameters
        ----------
        squeeze_dimensions : bool, optional
            If True, the numpy squeeze function is applied to the output array,
            removing all singleton dimensions.
            The default is True.
        resize : None or tuple/list of 2 ints, optional
            Dimensions to resize the frames. If None, no resizing is performed.
            The default is None.
        rescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in each frame to be within the given range.
            The default is None.
        globalrescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in all frames to be within the given range.
            The default is None.
        rotate : None or float, optional
            Rotation to apply to the image (in degrees).
            The default is None.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 5-dimensional, with the
            dimensions order being: Position, Time, Channel, Y, X

        """

        dt: Union[str, type] = self.dtype if rescale is None else np.float32
        if resize is None:
            output = np.empty(
                [self.timepoints, self.y, self.x], dtype=dt
            )
        else:
            output = np.empty(
                [self.timepoints, resize[0], resize[1]],
                dtype=dt,
            )

        # Load images:
        for f in range(self.timepoints):

            idx = f + self.starting_frame
            frame = self.original_video[idx].astype(np.uint16)

            # Optionally resize and rescale:
            if rotate is not None:
                frame = utils.imrotate(frame, rotate)
            if resize is not None:
                frame = cv2.resize(frame, resize[::-1])  # cv2 inverts shape
            if rescale is not None:
                frame = utils.rangescale(frame, rescale)
            # Add to output array:
            output[f] = frame

        # Rescale all images:
        if globalrescale is not None:
            output = utils.rangescale(output, globalrescale)

        output = output[np.newaxis, :, np.newaxis, :, :]

        # Return:
        return np.squeeze(output)[0, :, :] if squeeze_dimensions else output


class tracker:

    def __init__(self, **params):
        """
        Initializes Tracker

        Parameters
        ----------
        params : dict
            Dictionary of parameters:
            'model_type' (2D or mothermachine),
            'original_images_path' (path to original images .tif),
            'single mothermachine chamber' (Bool),
            'verbose' (Bool),
            'legacy' (Bool),
            'pickle' (Bool),
            'movie' (Bool).

        Returns
        -------
        None.

        """

        self.params = params

    def __read_tiff(self,
                    path):
        """
        Reads multipage tiff to numpy array.

        Parameters
        ----------
        path : string
            Path to tiff file.

        Returns
        -------
        images : np.array
            Images as 3D numpy array.

        """
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
        return np.array(images)

    def __load_model_and_presets(self,
                                 model_type):
        """
        Loads Presets for 2D or mothermachine, initializes model for tracking
        and loads model weights.

        Parameters
        ----------
        model_type : string
            Either '2D' or 'mothermachine'.

        Raises
        ------
        ValueError
            Configuration files or models have not yet been downloaded.
            Downloads the configuration files and model weights.

        Returns
        -------
        None.

        """
        # Load Presets
        while True:
            try:
                # Loads Presets images
                cfg.load_config(presets=model_type)

                # Load models:
                self.models = utils.loadmodels()
                "Dictionary of Tensorflow models"

                break

            except ValueError:
                # Downloads model weights and configuration files for 2D and mothermachine
                download_assets(load_models=True,
                                load_sets=False,
                                load_evals=False,
                                config_level='local')

        if self.params['single mothermachine chamber'] and model_type == 'mothermachine':
            self.models.pop('rois')

    def track(self,
              segm_video,
              signals=None,
              export_to: os.PathLike=None):
        """
        Tracks Cells

        Parameters
        ----------
        segm_video : np.array
            3D numpy array of segmentation mask.

        Returns
        -------
        tracked_video : np.array
            3D numpy array of tracked and labeled video.
        """

        # Loads Presets and Initializes Model
        self.__load_model_and_presets(model_type=self.params['model_type'])

        # Original Shape
        original_shape = segm_video[0].shape

        # Get original video and original image size
        original_video = self.__read_tiff(self.params['original_images_path'])
        reference = utils.rangescale(original_video[0], (0, 1))

        # Preprocess Segmentation Video
        seg_stack = []
        for idx in range(len(segm_video)):
            img = segm_video[idx, :, :]
            if not cfg.crop_windows:
                img = cv2.resize(img, cfg.target_size_seg[::-1])
            img_sm = (img > 0.5).astype(np.uint8)
            if cfg.crop_windows:
                img_sm = img_sm[: original_shape[0], : original_shape[1]].astype(np.uint8)
            seg_stack.append(img_sm)
        segm_video = seg_stack

        # Preprocess Original Video
        box = utils.CroppingBox(
            xtl=0, ytl=0,
            xbr=reference.shape[1], ybr=reference.shape[0],
        )
        img_stack = []
        if len(original_video) != len(segm_video):
            starting_frame = len(original_video) - len(segm_video) - 1
        else:
            starting_frame = 0
        for frame in range(len(segm_video)):
            idx = frame + starting_frame
            # Crop and scale:
            i = utils.rangescale(utils.cropbox(original_video[idx], box), rescale=(0, 1))
            # Append i as is to input images stack:
            img_stack.append(i)

        # Get Save Path (File Name is same as Original Images + .format)
        savepath = self.params['original_images_path']
        filename = savepath.replace('.tif', '')

        # Init reader
        xpreader = FakeReader(x=original_shape[1],
                              y=original_shape[0],
                              channels=0,
                              timepoints=len(segm_video),
                              filename=savepath,
                              original_video=original_video,
                              starting_frame=starting_frame
                              )

        # Init Position
        xp = pipeline.Position(position_nb=0,
                               reader=xpreader,
                               models=self.models,
                               drift_correction=False,
                               crop_windows=cfg.crop_windows)

        # Preprocess
        xp.preprocess(rotation_correction=False)

        # Replace img_stack and seg_stack in ROI
        xp.rois[0].img_stack = img_stack
        xp.rois[0].seg_stack = segm_video

        # Track
        xp.track(frames=list(range(len(segm_video))))

        # Label Cells
        xp.features(frames=list(range(len(segm_video))))

        # Get labels
        tracked_video = np.array(xp.rois[0].label_stack, dtype=np.uint8)

        # Save Results
        if self.params['legacy']:
            xp.legacysave(filename + ".mat")
        if self.params['pickle']:
            import pickle
            with open(filename + ".pkl", "wb") as file:
                pickle.dump(self, file)
        if self.params['movie']:
            movie = xp.results_movie(frames=list(range(len(segm_video))))
            utils.vidwrite(movie, filename + ".mp4", verbose=False)

        # Return tracked and labeled video
        return tracked_video
