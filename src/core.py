import numpy as np
import cv2
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import skimage.filters
import scipy.ndimage.morphology
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch, Path

import pandas as pd
from numba import jit

from tqdm import tqdm

# Custom modules
import apps

@jit(nopython=True, parallel=True)
def numba_max(arr):
    return arr.max()

@jit(nopython=True, parallel=True)
def numba_min(arr):
    return arr.min()

def np_replace_values(arr, old_values, new_values):
    # See method_jdehesa https://stackoverflow.com/questions/45735230/how-to-replace-a-list-of-values-in-a-numpy-array
    old_values = np.asarray(old_values)
    new_values = np.asarray(new_values)
    n_min, n_max = arr.min(), arr.max()
    replacer = np.arange(n_min, n_max + 1)
    # Mask replacements out of range
    mask = (old_values >= n_min) & (old_values <= n_max)
    replacer[old_values[mask] - n_min] = new_values[mask]
    arr = replacer[arr - n_min]
    return arr

def lab_replace_values(lab, rp, oldIDs, newIDs):
    for obj in rp:
        try:
            idx = oldIDs.index(obj.label)
        except ValueError:
            continue
        lab[obj.slice][obj.image] = newIDs[idx]
    return lab

def remove_artefacts(
        lab, min_solidity=0.5, min_area=15, max_elongation=3,
        return_delIDs=False
    ):
    """
    function to remove cells with area<min_area or solidity<min_solidity
    or elongation>max_elongation
    """
    rp = skimage.measure.regionprops(lab.astype(int))
    delIDs = []
    for obj in rp:
        if obj.area<min_area:
            lab[obj.slice][obj.image] = 0
            delIDs.append(obj.label)
            continue

        if obj.solidity<min_solidity:
            lab[obj.slice][obj.image] = 0
            delIDs.append(obj.label)
            continue

        # NOTE: single pixel horizontal or vertical lines minor_axis_length=0
        minor_axis_length = max(1, obj.minor_axis_length)
        print(obj.minor_axis_length, minor_axis_length)
        elongation = obj.major_axis_length/minor_axis_length
        if obj.solidity<min_solidity:
            lab[obj.slice][obj.image] = 0
            delIDs.append(obj.label)

    if return_delIDs:
        return lab, delIDs
    else:
        return lab

def align_frames_3D(
        data, slices=None, register=True,
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
            elif y<0:
                aligned_frame_V[:, y:] = 0
            if x>0:
                aligned_frame_V[:, :, :x] = 0
            elif x<0:
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
            elif y<0:
                aligned_frame_V[y:] = 0
            if x>0:
                aligned_frame_V[:, :x] = 0
            elif x<0:
                aligned_frame_V[:, x:] = 0
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts

def get_objContours(obj):
    contours, _ = cv2.findContours(
                           obj.image.astype(np.uint8),
                           cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_NONE
    )
    min_y, min_x, _, _ = obj.bbox
    cont = np.squeeze(contours[0], axis=1)
    cont = np.vstack((cont, cont[0]))
    cont += [min_x, min_y]
    return cont

def smooth_contours(lab, radius=2):
    sigma = 2*radius + 1
    smooth_lab = np.zeros_like(lab)
    for obj in skimage.measure.regionprops(lab):
        cont = get_objContours(obj)
        x = cont[:,0]
        y = cont[:,1]
        x = np.append(x, x[0:sigma])
        y = np.append(y, y[0:sigma])
        x = np.round(skimage.filters.gaussian(x, sigma=sigma,
                                              preserve_range=True)).astype(int)
        y = np.round(skimage.filters.gaussian(y, sigma=sigma,
                                              preserve_range=True)).astype(int)
        temp_mask = np.zeros(lab.shape, bool)
        temp_mask[y, x] = True
        temp_mask = scipy.ndimage.morphology.binary_fill_holes(temp_mask)
        smooth_lab[temp_mask] = obj.label
    return smooth_lab

def getBaseCca_df(IDs):
    cc_stage = ['G1' for ID in IDs]
    num_cycles = [2]*len(IDs)
    relationship = ['mother' for ID in IDs]
    related_to = [-1]*len(IDs)
    emerg_frame_i = [-1]*len(IDs)
    division_frame_i = [-1]*len(IDs)
    is_history_known = [False]*len(IDs)
    corrected_assignment = [False]*len(IDs)
    cca_df = pd.DataFrame({
                       'cell_cycle_stage': cc_stage,
                       'generation_num': num_cycles,
                       'relative_ID': related_to,
                       'relationship': relationship,
                       'emerg_frame_i': emerg_frame_i,
                       'division_frame_i': division_frame_i,
                       'is_history_known': is_history_known,
                       'corrected_assignment': corrected_assignment},
                        index=IDs)
    cca_df.index.name = 'Cell_ID'
    return cca_df

def cca_df_to_acdc_df(cca_df, rp, acdc_df=None):
    if acdc_df is None:
        IDs = []
        is_cell_dead_li = []
        is_cell_excluded_li = []
        xx_centroid = []
        yy_centroid = []
        editIDclicked_x = []
        editIDclicked_y = []
        editIDnewID = []
        for obj in rp:
            IDs.append(obj.label)
            is_cell_dead_li.append(0)
            is_cell_excluded_li.append(0)
            xx_centroid.append(int(obj.centroid[1]))
            yy_centroid.append(int(obj.centroid[0]))
            editIDclicked_x.append(np.nan)
            editIDclicked_y.append(np.nan)
            editIDnewID.append(-1)
        acdc_df = pd.DataFrame({
            'Cell_ID': IDs,
            'is_cell_dead': is_cell_dead_li,
            'is_cell_excluded': is_cell_excluded_li,
            'x_centroid': xx_centroid,
            'y_centroid': yy_centroid,
            'editIDclicked_x': editIDclicked_x,
            'editIDclicked_y': editIDclicked_y,
            'editIDnewID': editIDnewID
        }).set_index('Cell_ID')

    acdc_df = acdc_df.join(cca_df, how='left')
    return acdc_df
