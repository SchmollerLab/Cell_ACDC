import numpy as np
import cv2
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch, Path

from tqdm import tqdm

# Custom modules
from MyWidgets import Slider, Button, RadioButtons
import apps

def align_frames_3D(data, slices=None, register=True,
                          user_shifts=None, pbar=False):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        slice = slices[frame_i]
        if frame_i != 0:  # skip first frame
            curr_frame_img = frame_V[slice]
            prev_frame_img = data_aligned[frame_i-1, slice] #previously aligned frame, slice
            if user_shifts is None:
                shifts = skimage.registration.phase_cross_correlation(
                    prev_frame_img, curr_frame_img
                    )[0]
            else:
                shifts = user_shifts[frame_i]
            shifts = shifts.astype(int)
            aligned_frame_V = np.copy(frame_V)
            aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(1,2))
            # Pad rolled sides with 0s
            y, x = shifts
            if y>0:
                aligned_frame_V[:, :y] = 0
            else:
                aligned_frame_V[:, y:] = 0
            if x>0:
                aligned_frame_V[:, :, :x] = 0
            else:
                aligned_frame_V[:, :, x:] = 0
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts


def align_frames_2D(data, slices=None, register=True,
                          user_shifts=None, pbar=False):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        if frame_i != 0:  # skip first frame
            curr_frame_img = frame_V
            prev_frame_img = data_aligned[frame_i-1] #previously aligned frame, slice
            if user_shifts is None:
                shifts = skimage.registration.phase_cross_correlation(
                    prev_frame_img, curr_frame_img
                    )[0]
            else:
                shifts = user_shifts[frame_i]
            shifts = shifts.astype(int)
            aligned_frame_V = np.copy(frame_V)
            aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(0,1))
            y, x = shifts
            if y>0:
                aligned_frame_V[:y] = 0
            else:
                aligned_frame_V[y:] = 0
            if x>0:
                aligned_frame_V[:, :x] = 0
            else:
                aligned_frame_V[:, x:] = 0
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts
