"""
Experiment reader class.

@author: jroberts / jamesr787
"""

# Modules:
from typing import Tuple, Union

import cv2
import numpy as np

from delta import utilities as utils


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
                  positions=None,
                  channels=None,
                  frames=None,
                  squeeze_dimensions: bool = True,
                  resize: Tuple[int, int] = None,
                  rescale: Tuple[float, float] = None,
                  globalrescale: Tuple[float, float] = None,
                  rotate: float = None,
                  ):
        """
        Get frames from experiment.

        Parameters
        ----------
        positions : None, int, tuple/list of ints, optional
            The frames from the position index or indexes passed as an integer
            or a tuple/list will be returned. If None is passed, all positions
            are returned.
            The default is None.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer
            or a tuple/list will be returned. If the channel names have been
            defined, the channel(s) can be passed as a string or tuple/list of
            strings. If an empty list is passed, None is returned. If None is
            passed, all channels are returned.
            The default is None.
        frames : None, int, tuple/list of ints, optional
            The frame index or indexes passed as an integer or a tuple/list
            will be returned. If None is passed, all frames are returned. If -1
            is passed and the file watcher is activated, only new frames are
            read. Works only for one position and channel at a time.
            The default is None.
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
