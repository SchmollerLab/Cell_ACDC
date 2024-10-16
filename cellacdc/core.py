import traceback
import os
import time
from importlib import import_module
import numpy as np
import cv2
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import skimage.filters
import skimage.segmentation
import scipy.ndimage.morphology
from itertools import product

from math import sqrt
from scipy.stats import norm

import pandas as pd

from tqdm import tqdm

from . import load, myutils
from . import cca_df_colnames, printl, base_cca_dict, base_cca_tree_dict
from . import features
from . import error_up_str
from . import issues_url
from . import exception_handler_cli
from . import all_non_metrics_cols
from . import cca_functions
from . import config
from . import preprocess
from .config import PREPROCESS_MAPPER

class HeadlessSignal:
    def __init__(self, *args):
        pass
    
    def emit(self, *args, **kwargs):
        pass

class ProgressCliSignal:
    def __init__(self, logger_func):
        self.logger_func = logger_func
    
    def emit(self, text):
        self.logger_func(text)

class KernelCliSignals:
    def __init__(self, logger_func):
        self.finished = HeadlessSignal(float)
        self.progress = ProgressCliSignal(logger_func)
        self.progressBar = HeadlessSignal(int)
        self.innerProgressBar = HeadlessSignal(int)
        self.resetInnerPbar = HeadlessSignal(int)
        self.progress_tqdm = HeadlessSignal(int)
        self.signal_close_tqdm = HeadlessSignal()
        self.create_tqdm = HeadlessSignal(int)
        self.debug = HeadlessSignal(object)
        self.critical = HeadlessSignal(object)

def get_indices_dash_pattern(arr, line_length, gap):
    n = len(arr)
    sampling_rate = (line_length+gap)
    n_lines = n // sampling_rate
    tot_len = n_lines*sampling_rate
    indices_2D = np.arange(tot_len).reshape((n_lines,sampling_rate))
    indices = (indices_2D[:, :line_length]).flatten()
    return indices

def get_line(r0, c0, r1, c1, dashed=True):
    x1, x2 = sorted((c0, c1))
    Dc = (c0-c1)
    Dr = (r0-r1)
    dist = np.ceil(np.sqrt(np.square(Dr)+np.square(Dc)))
    
    if Dc == 0:
        xx = np.array([c0]*int(dist))
        y1, y2 = sorted((r0, r1))
        yy = np.linspace(y1, y2, len(xx))
    else:
        xx = np.linspace(x1, x2, int(dist))
        m = Dr/Dc
        q = (c0*r1 - c1*r0)/Dc
        yy = xx*m+q
    if dashed:
        indices = get_indices_dash_pattern(xx, 4, 3)
        xx = xx[indices]
        yy = yy[indices]
    return xx, yy

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

def nearest_nonzero_2D(a, y, x, max_dist=None, return_coords=False):
    value = a[round(y), round(x)]
    if value > 0:
        if return_coords:
            return value, round(y), round(x)
        else:
            return value
    r, c = np.nonzero(a)
    dist = ((r - y)**2 + (c - x)**2)
    if max_dist is not None:
        if dist.min() > max_dist:
            if return_coords:
                return 0, 0, 0
            else:
                return 0
    min_idx = dist.argmin()
    y_nearest, x_nearest = r[min_idx], c[min_idx]
    if return_coords:
        return a[y_nearest, x_nearest], y_nearest, x_nearest
    else:
        return a[y_nearest, x_nearest]

def nearest_nonzero_1D(arr, x, return_index=False):
    if arr[x] > 0:
        if return_index:
            return arr[x], x
        else:
            return arr[x]
    nonzero_idxs, = np.nonzero(arr)
    dist = (nonzero_idxs - x)**2
    min_idx = dist.argmin()
    nearest_nonzero_idx = nonzero_idxs[min_idx]
    val = arr[nearest_nonzero_idx]
    
    if return_index:
        return val, nearest_nonzero_idx
    else:
        return val

def nearest_nonzero_z_idx_from_z_centroid(obj, current_z=-1):
    zc = obj.local_centroid[0]
    z_obj_local = int(zc)
    is_obj_slice_not_empty = np.any(obj.image[z_obj_local])
    z_obj_global = z_obj_local + obj.bbox[0]
    if current_z == z_obj_global and is_obj_slice_not_empty:
        return current_z
    
    zslices_not_empty_arr = np.any(obj.image, axis=(1,2)).astype(np.uint8)
    _, nearest_nonzero_z_local = nearest_nonzero_1D(
        zslices_not_empty_arr, z_obj_local, return_index=True
    )
    nearest_nonzero_z_global = nearest_nonzero_z_local + obj.bbox[0]
    return nearest_nonzero_z_global

def compute_twoframes_velocity(prev_lab, lab, spacing=None):
    prev_rp = skimage.measure.regionprops(prev_lab)
    rp = skimage.measure.regionprops(lab)
    prev_IDs = [obj.label for obj in prev_rp]
    velocities_pxl = [0]*len(rp)
    velocities_um = [0]*len(rp)
    for i, obj in enumerate(rp):
        if obj.label not in prev_IDs:
            continue

        prev_obj = prev_rp[prev_IDs.index(obj.label)]
        diff = np.subtract(obj.centroid, prev_obj.centroid)
        v_pixel = np.linalg.norm(diff)
        velocities_pxl[i] = v_pixel
        if spacing is not None:
            v_um = np.linalg.norm(diff*spacing)
            velocities_um[i] = v_um
    return velocities_pxl, velocities_um


def lab_replace_values(lab, rp, oldIDs, newIDs, in_place=True):
    if not in_place:
        lab = lab.copy()
    for obj in rp:
        try:
            idx = oldIDs.index(obj.label)
        except ValueError:
            continue

        if obj.label == newIDs[idx]:
            # Skip assigning ID to same ID
            continue

        lab[obj.slice][obj.image] = newIDs[idx]
    return lab

def post_process_segm(labels, return_delIDs=False, **kwargs):
    min_solidity = kwargs.get('min_solidity')
    min_area = kwargs.get('min_area')
    max_elongation = kwargs.get('max_elongation')
    min_obj_no_zslices = kwargs.get('min_obj_no_zslices')
    if labels.ndim == 3:
        delIDs = set()
        if min_obj_no_zslices is not None:
            for obj in skimage.measure.regionprops(labels):
                obj_no_zslices = np.sum(
                    np.count_nonzero(obj.image, axis=(1, 2)).astype(bool)
                )
                if obj_no_zslices < min_obj_no_zslices:
                    labels[obj.slice][obj.image] = 0
                    delIDs.add(obj.label)
        
        for z, lab in enumerate(labels):
            _result = post_process_segm_lab2D(
                lab, min_solidity, min_area, max_elongation,
                return_delIDs=return_delIDs
            )
            if return_delIDs:
                lab, _delIDs = _result
                delIDs.update(_delIDs)
            else:
                lab = _result
            labels[z] = lab
        if return_delIDs:
            result = labels, delIDs
        else:
            result = labels
    else:
        result = post_process_segm_lab2D(
            labels, min_solidity, min_area, max_elongation,
            return_delIDs=return_delIDs
        )

    if return_delIDs:
        labels, delIDs = result
        return labels, delIDs
    else:
        labels = result
        return labels

def post_process_segm_lab2D(
        lab, min_solidity=None, min_area=None, max_elongation=None,
        return_delIDs=False
    ):
    """
    function to remove cells with area<min_area or solidity<min_solidity
    or elongation>max_elongation
    """
    rp = skimage.measure.regionprops(lab.astype(int))
    if return_delIDs:
        delIDs = []
    for obj in rp:
        if min_area is not None:
            if obj.area < min_area:
                lab[obj.slice][obj.image] = 0
                if return_delIDs:
                    delIDs.append(obj.label)
                continue

        if min_solidity is not None:
            if obj.solidity < min_solidity:
                lab[obj.slice][obj.image] = 0
                if return_delIDs:
                    delIDs.append(obj.label)
                continue

        if max_elongation is not None:
            # NOTE: single pixel horizontal or vertical lines minor_axis_length=0
            minor_axis_length = max(1, obj.minor_axis_length)
            elongation = obj.major_axis_length/minor_axis_length
            if elongation > max_elongation:
                lab[obj.slice][obj.image] = 0
                if return_delIDs:
                    delIDs.append(obj.label)

    if return_delIDs:
        return lab, delIDs
    else:
        return lab

def connect_3Dlab_zboundaries(lab):
    connected_lab = np.zeros_like(lab)
    rp = skimage.measure.regionprops(lab)
    for obj in rp:
        if len(obj.image) == 1:
            lab[obj.slice][obj.image] = obj.label
            continue
        
        # Take the center non-zero z-area as reference object
        z_areas = [np.count_nonzero(z_img) for z_img in obj.image]
        nonzero_z_areas = [z_area for z_area in z_areas if z_area > 0]
        nonzero_center_idx = int(len(nonzero_z_areas)/2)
        nonzero_center_z_area = nonzero_z_areas[nonzero_center_idx]
        center_idx = z_areas.index(nonzero_center_z_area)
        max_obj_image = obj.image[center_idx]
        num_zslices = len(obj.image)
        
        for z in range(num_zslices):
            connected_lab[obj.slice][z][max_obj_image] = obj.label
        
    return connected_lab

def stack_2Dlab_to_3D(lab, SizeZ):
    return np.tile(lab, (SizeZ, 1, 1))

def track_sub_cell_objects_third_segm_acdc_df(
        track_parent_objs_segm_data, parent_objs_acdc_df
    ):
    if parent_objs_acdc_df is None:
        return
    
    keys = []
    dfs = []
    for frame_i, lab in enumerate(track_parent_objs_segm_data):
        rp = skimage.measure.regionprops(lab)
        IDs = [obj.label for obj in rp]
        if frame_i not in parent_objs_acdc_df.index.get_level_values(0):
            acdc_df_frame_i = myutils.getBaseAcdcDf(rp)
        else:
            acdc_df_frame_i = parent_objs_acdc_df.loc[frame_i]
            cols = acdc_df_frame_i.columns.intersection(all_non_metrics_cols)
            acdc_df_frame_i = acdc_df_frame_i[cols]
        
        dfs.append(acdc_df_frame_i)
        keys.append(frame_i)
    third_segm_acdc_df = pd.concat(
        dfs, keys=keys, names=['frame_i', 'Cell_ID']
    )
    return third_segm_acdc_df
    
def track_sub_cell_objects_acdc_df(
        tracked_subobj_segm_data, subobj_acdc_df, all_old_sub_ids,
        all_num_objects_per_cells, SizeT=None, sigProgress=None, 
        tracked_cells_segm_data=None, cells_acdc_df=None
    ):
    if SizeT == 1:
        tracked_subobj_segm_data = tracked_subobj_segm_data[np.newaxis]
        if tracked_cells_segm_data is not None:
            tracked_cells_segm_data = tracked_cells_segm_data[np.newaxis]
    
    if tracked_cells_segm_data is not None:
        acdc_df_list = []
    sub_acdc_df_list = []
    keys_cells = []
    keys_sub = []
    for frame_i, lab_sub in enumerate(tracked_subobj_segm_data):
        rp_sub = skimage.measure.regionprops(lab_sub)
        sub_ids = [sub_obj.label for sub_obj in rp_sub]
        old_sub_ids = all_old_sub_ids[frame_i]
        if subobj_acdc_df is None:
            sub_acdc_df_frame_i = myutils.getBaseAcdcDf(rp_sub)
        elif frame_i not in subobj_acdc_df.index.get_level_values(0):
            sub_acdc_df_frame_i = myutils.getBaseAcdcDf(rp_sub)
        else:
            sub_acdc_df_frame_i = (
                subobj_acdc_df.loc[frame_i].rename(index=old_sub_ids) 
            )
            if 'relative_ID' in sub_acdc_df_frame_i.columns:
                sub_acdc_df_frame_i['relative_ID'] = (
                    sub_acdc_df_frame_i['relative_ID'].replace(old_sub_ids)
                )

        cols = sub_acdc_df_frame_i.columns.intersection(all_non_metrics_cols)
        sub_acdc_df_list.append(sub_acdc_df_frame_i.loc[sub_ids, cols])
        keys_sub.append(frame_i)
        
        if tracked_cells_segm_data is not None:
            num_objects_per_cells = all_num_objects_per_cells[frame_i]
            lab = tracked_cells_segm_data[frame_i]
            rp = skimage.measure.regionprops(lab)
            # Untacked sub-obj (if kept) are not present in acdc_df of the cells
            # --> check with `IDs_with_sub_obj = ... if id in lab`
            IDs_with_sub_obj = [id for id in sub_ids if id in lab]
            if cells_acdc_df is None:
                acdc_df_frame_i = myutils.getBaseAcdcDf(rp)
            else:
                acdc_df_frame_i = cells_acdc_df.loc[frame_i].copy()

            cols = acdc_df_frame_i.columns.intersection(all_non_metrics_cols)
            acdc_df_frame_i = acdc_df_frame_i[cols]
            
            acdc_df_frame_i['num_sub_cell_objs_per_cell'] = 0
            acdc_df_frame_i.loc[IDs_with_sub_obj, 'num_sub_cell_objs_per_cell'] = ([
                num_objects_per_cells[id] for id in IDs_with_sub_obj
            ])
            acdc_df_list.append(acdc_df_frame_i)
            keys_cells.append(frame_i)

        if sigProgress is not None:
            sigProgress.emit(1)
            
    sub_tracked_acdc_df = pd.concat(
        sub_acdc_df_list, keys=keys_sub, names=['frame_i', 'Cell_ID']
    )
    
    tracked_acdc_df = None
    if tracked_cells_segm_data is not None:
        tracked_acdc_df = pd.concat(
            acdc_df_list, keys=keys_cells, names=['frame_i', 'Cell_ID']
        )
    
    return sub_tracked_acdc_df, tracked_acdc_df
        
        
def track_sub_cell_objects(
        cells_segm_data, subobj_segm_data, IoAthresh, 
        how='delete_sub', SizeT=None, sigProgress=None,
        relabel_sub_obj_lab=False
    ):  
    """Function used to track sub-cellular objects and assign the same ID of 
    the cell they belong to. 
    
    For each sub-cellular object calculate the interesection over area  with cells 
    --> get max IoA in case it is touching more than one cell 
    --> assign that cell if IoA >= IoA thresh

    Args:
        cells_segm_data (ndarray): 2D, 3D or 4D array of `int` type cotaining 
            the cells segmentation masks.

        subobj_segm_data (ndarray): 2D, 3D or 4D array of `int` type cotaining 
            the sub-cellular segmentation masks (e.g., nuclei).

        IoAthresh (float): Minimum percentage (0-1) of the sub-cellular object's 
            area to assign it to a cell

        how (str, optional): Strategy to take with untracked objects. 
            Options are 'delete_sub' to delete untracked sub-cellular objects, 
            'delete_cells' to delete cells that do not have any sub-cellular 
            object assigned to it, 'delete_both', and 'only_track' to keep 
            untracked objects. Note that 'delete_sub' is actually not used 
            because we add tracked sub-objects to an array initialized with 
            zeros. Defaults to 'delete_sub'.

        SizeT (int, optional): Number of frames. Pass `SizeT=1` for non-timelapse
            data. Defaults to None --> assume first dimension of segm data is SizeT.

        sigProgress (PyQt5.QtCore.Signal, optional): If provided it will emit 
            1 for each complete frame. Used to update GUI progress bars. 
            Defaults to None --> do not emit signal.
        
        relabel_sub_obj_lab (bool, optional): Re-label sub-cellular objects 
            segmentation labels before tracking them.
    
    Returns:
        tuple: A tuple `(tracked_subobj_segm_data, tracked_cells_segm_data, 
            all_num_objects_per_cells, old_sub_ids)` where `tracked_subobj_segm_data` is the 
            segmentation mask of the sub-cellular objects with the same IDs of 
            the cells they belong to, `tracked_cells_segm_data` is the segmentation 
            masks of the cells that do have at least on sub-cellular object 
            (`None` if `how != 'delete_sub'`), `all_num_objects_per_cells` is 
            a list of dictionary (one per frame) where the dictionaries have 
            cell IDs as keys and the number of sub-cellular objects per cell as
            values, and `all_old_sub_ids` is a list of dictionaries (one per frame)
            where each dictionary has the new sub-cellular objects' ids as keys and 
            the old (replaced) ids.
    """    
    if SizeT == 1:
        cells_segm_data = cells_segm_data[np.newaxis]
        subobj_segm_data = subobj_segm_data[np.newaxis]

    tracked_cells_segm_data = None
    tracked_subobj_segm_data = np.zeros_like(subobj_segm_data)        

    segm_data_zip = zip(cells_segm_data, subobj_segm_data)
    old_tracked_sub_obj_IDs = set()
    cells_IDs_with_sub_obj = []
    all_num_objects_per_cells = []
    all_cells_IDs_with_sub_obj = []
    all_old_sub_ids = [{} for _ in range(len(cells_segm_data))]
    for frame_i, (lab, lab_sub) in enumerate(segm_data_zip):
        rp = skimage.measure.regionprops(lab)
        num_objects_per_cells = {obj.label:0 for obj in rp}
        if relabel_sub_obj_lab:
            lab_sub = skimage.measure.label(lab_sub)
        rp_sub = skimage.measure.regionprops(lab_sub)
        tracked_lab_sub = tracked_subobj_segm_data[frame_i]
        cells_IDs_with_sub_obj = []
        tracked_sub_obj_original_IDs = []
        untracked_sub_objs_frame_i = set()
        for sub_obj in rp_sub:
            intersect_mask = lab[sub_obj.slice][sub_obj.image]
            intersect_IDs, intersections = np.unique(
                intersect_mask, return_counts=True
            )
            argmax = intersections.argmax()
            intersect_ID = intersect_IDs[argmax]
            intersection = intersections[argmax]
            
            if intersect_ID == 0:
                untracked_sub_objs_frame_i.add(sub_obj.label)
                continue
            
            IoA = intersection/sub_obj.area
            if IoA < IoAthresh:
                # Do not add untracked sub-obj
                untracked_sub_objs_frame_i.add(sub_obj.label)
                continue
            
            all_old_sub_ids[frame_i][sub_obj.label] = intersect_ID
            tracked_lab_sub[sub_obj.slice][sub_obj.image] = intersect_ID
            num_objects_per_cells[intersect_ID] += 1
            old_tracked_sub_obj_IDs.add(sub_obj.label)
            cells_IDs_with_sub_obj.append(intersect_ID)
            tracked_sub_obj_original_IDs.append(sub_obj.label)
        
        # assignments = []
        # for sub_obj_ID, cell_ID in zip(tracked_sub_obj_original_IDs, cells_IDs_with_sub_obj):
        #     assignments.append(f'  * {sub_obj_ID} --> {cell_ID}')
        # assignments_format = '\n'.join(assignments)
        # printl(f'Assignments in frame_i = {frame_i}:\n{assignments_format}')
        # printl(f'Untracked sub-objs = {untracked_sub_objs_frame_i}')
        # from acdctools.plot import imshow
        # imshow(lab, subobj_segm_data[frame_i], lab_sub, tracked_lab_sub)
        # import pdb; pdb.set_trace()
        
        all_num_objects_per_cells.append(num_objects_per_cells)
        all_cells_IDs_with_sub_obj.append(cells_IDs_with_sub_obj)
        
        if sigProgress is not None:
            sigProgress.emit(1)
    
    if how == 'delete_both' or how == 'delete_cells':
        # Delete cells that do not have a sub-cellular object
        tracked_cells_segm_data = cells_segm_data.copy()
        for frame_i, lab in enumerate(tracked_cells_segm_data):
            rp = skimage.measure.regionprops(lab)
            tracked_lab = tracked_cells_segm_data[frame_i]
            cells_IDs_with_sub_obj = all_cells_IDs_with_sub_obj[frame_i]
            for obj in rp:
                if obj.label in cells_IDs_with_sub_obj:
                    # Cell has sub-object do not delete
                    continue
                    
                tracked_lab[obj.slice][obj.image] = 0
    
    if how == 'only_track' or how == 'delete_cells':
        # Assign unique IDs to untracked sub-cellular objects and add them 
        # to all_old_sub_ids
        maxSubObjID = tracked_subobj_segm_data.max() + 1
        for sub_obj_ID in np.unique(subobj_segm_data):
            if sub_obj_ID == 0:
                continue

            if sub_obj_ID in old_tracked_sub_obj_IDs:
                # sub_obj_ID has already ben tracked
                continue
            
            tracked_subobj_segm_data[subobj_segm_data == sub_obj_ID] = maxSubObjID
            
            for frame_i, lab_sub in enumerate(subobj_segm_data):
                if sub_obj_ID not in lab_sub:
                    continue
                all_old_sub_ids[frame_i][sub_obj_ID] = maxSubObjID
            maxSubObjID += 1

    if SizeT == 1:
        tracked_subobj_segm_data = tracked_subobj_segm_data[0]
        if how == 'delete_both':
            tracked_cells_segm_data = tracked_cells_segm_data[0]
    
    return (
        tracked_subobj_segm_data, tracked_cells_segm_data, 
        all_num_objects_per_cells, all_old_sub_ids
    )

def _calc_airy_radius(wavelen, NA):
    airy_radius_nm = (1.22 * wavelen)/(2*NA)
    airy_radius_um = airy_radius_nm*1E-3 #convert nm to µm
    return airy_radius_nm, airy_radius_um

def calc_resolution_limited_vol(
        wavelen, NA, yx_resolution_multi, zyx_vox_dim, z_resolution_limit
    ):
    airy_radius_nm, airy_radius_um = _calc_airy_radius(wavelen, NA)
    yx_resolution = airy_radius_um*yx_resolution_multi
    zyx_resolution = np.asarray(
        [z_resolution_limit, yx_resolution, yx_resolution]
    )
    zyx_resolution_pxl = zyx_resolution/np.asarray(zyx_vox_dim)
    return zyx_resolution, zyx_resolution_pxl, airy_radius_nm

def align_frames_3D(data, slices=None, user_shifts=None, sigPyqt=None):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        if frame_i == 0:
            # skip first frame
            continue   
        if user_shifts is None:
            slice = slices[frame_i]
            curr_frame_img = frame_V[slice]
            prev_frame_img = data_aligned[frame_i-1, slice] 
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
        if sigPyqt is not None:
            sigPyqt.emit(1)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(z_proj_max(frame_V))
        # ax[1].imshow(z_proj_max(aligned_frame_V))
        # plt.show()
    return data_aligned, registered_shifts

def revert_alignment(saved_shifts, img_data, sigPyqt=None):
    shifts = -saved_shifts
    reverted_data = np.zeros_like(img_data)
    for frame_i, shift in enumerate(saved_shifts):
        if frame_i >= len(img_data):
            break
        img = img_data[frame_i]
        axis = tuple(range(img.ndim))[-2:]
        reverted_img = np.roll(img, tuple(shift), axis=axis)
        reverted_data[frame_i] = reverted_img
        if sigPyqt is not None:
            sigPyqt.emit(1)
    return reverted_data

def align_frames_2D(
        data, slices=None, register=True, user_shifts=None, pbar=False,
        sigPyqt=None
    ):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(tqdm(data, ncols=100)):
        if frame_i == 0:
            # skip first frame
            continue 
        
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
        if sigPyqt is not None:
            sigPyqt.emit(1)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(z_proj_max(frame_V))
        # ax[1].imshow(z_proj_max(aligned_frame_V))
        # plt.show()
    return data_aligned, registered_shifts

def label_3d_segm(labels):
    """Label objects in 3D array that is the result of applying
    2D segmentation model on each z-slice.

    Parameters
    ----------
    labels : Numpy array
        Array of labels with shape (Z, Y, X).

    Returns
    -------
    Numpy array
        Labelled array with shape (Z, Y, X).

    """
    rp_split_lab = skimage.measure.regionprops(labels)
    merge_lab = skimage.measure.label(labels)
    rp_merge_lab = skimage.measure.regionprops(merge_lab)
    for obj in rp_split_lab:
        pass

    return labels

def get_obj_contours(
        obj=None, 
        obj_image=None, 
        obj_bbox=None, 
        all_external=False, 
        all=False, 
        only_longest_contour=True, 
        local=False,
    ):
    if all:
        retrieveMode = cv2.RETR_CCOMP
    else:
        retrieveMode = cv2.RETR_EXTERNAL
    
    if obj_image is None:
        obj_image = obj.image
    
    obj_image = obj_image.astype(np.uint8)
    
    if obj_bbox is None and not local:
        obj_bbox = obj.bbox
    
    contours, _ = cv2.findContours(
        obj_image, retrieveMode, cv2.CHAIN_APPROX_NONE
    )
    if all or all_external:
        if local:
            return [np.squeeze(cont, axis=1) for cont in contours]
        else:
            min_y, min_x, _, _ = obj_bbox
            return [np.squeeze(cont, axis=1)+[min_x, min_y] for cont in contours]
    
    if len(contours) > 1 and only_longest_contour:
        contours_len = [len(c) for c in contours]
        max_len_idx = contours_len.index(max(contours_len))
        contour = contours[max_len_idx]
    else:
        contour = contours[0]
    contour = np.squeeze(contour, axis=1)
    contour = np.vstack((contour, contour[0]))
    if not local:
        min_y, min_x, _, _ = obj_bbox
        contour += [min_x, min_y]
    return contour

def smooth_contours(lab, radius=2):
    sigma = 2*radius + 1
    smooth_lab = np.zeros_like(lab)
    for obj in skimage.measure.regionprops(lab):
        cont = get_obj_contours(obj)
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

def get_labels_to_IDs_mapper(tracked_labels):
    labels_to_IDs_mapper = {}
    uniqueID = 1
    for frame_i, tracked_frame_labels in enumerate(tracked_labels):
        for tracked_label in tracked_frame_labels:
            if tracked_label in labels_to_IDs_mapper:
                # Cell existed in the past, ID already stored
                continue
            
            parent_label, _, sister_label = tracked_label.rpartition('_')
            if not parent_label:
                # Single-cell that was not mapped yet
                ID = uniqueID
                uniqueID += 1
            elif sister_label == '0':
                # Sister label == 0 --> keep mother ID
                ID = labels_to_IDs_mapper[parent_label].split('_')[0]
            elif (
                    sister_label == '1' 
                    and f'{parent_label}_0' not in tracked_frame_labels
                ):
                # Daughter cell without a sister --> keep mother ID
                ID = labels_to_IDs_mapper[parent_label].split('_')[0]
            else:
                # Sister label == 1 --> assign new ID
                ID = uniqueID
                uniqueID += 1
            labels_to_IDs_mapper[tracked_label] = f'{ID}_{frame_i}'

    return labels_to_IDs_mapper

def annotate_lineage_tree_from_labels(tracked_labels, labels_to_IDs_mapper):
    IDs_to_labels_mapper = {
        ID:label for label, ID in labels_to_IDs_mapper.items()
    }
    cca_dfs = []
    keys = []
    pbar = tqdm(total=len(tracked_labels), ncols=100)
    for frame_i, tracked_frame_labels in enumerate(tracked_labels):
        keys.append(frame_i)
        IDs = [
            int(labels_to_IDs_mapper[label].split('_')[0]) 
            for label in tracked_frame_labels
        ]
        if frame_i == 0:
            cca_df = getBaseCca_df(IDs)
            cca_dfs.append(cca_df)
            pbar.update()
            continue
        
        # Get cca_df from previous frame for existing cells
        cca_df = cca_dfs[frame_i-1]
        is_in_index = cca_df.index.isin(IDs)
        cca_df = cca_df[is_in_index]
        new_cells_cca_dfs = []

        for ID in IDs:
            if ID in cca_df.index:
                continue
            
            newID = ID
            # New cell --> store cca info
            label = IDs_to_labels_mapper[f'{newID}_{frame_i}']
            parent_label, _, sister_label = label.rpartition('_')
            if not parent_label:
                # New single-cell --> check if it existed in past frames
                for i in range(frame_i-2, -1, -1):
                    past_cca_df = cca_dfs[frame_i-1]
                    if newID in past_cca_df.index:
                        cca_df_single_ID = past_cca_df.loc[[newID]]
                        break
                else:
                    cca_df_single_ID = getBaseCca_df([newID])
                    cca_df_single_ID.loc[newID, 'emerg_frame_i'] = frame_i
            else:
                # New cell resulting from division --> store division
                mothID = int(labels_to_IDs_mapper[parent_label].split('_')[0])
                cca_df_single_ID = getBaseCca_df([newID])
                try:
                    cca_df.at[mothID, 'generation_num'] += 1
                except Exception as e:
                    import pdb; pdb.set_trace()
                cca_df.at[mothID, 'division_frame_i'] = frame_i
                cca_df.at[mothID, 'relative_ID'] = newID
                cca_df_single_ID.at[newID, 'emerg_frame_i'] = frame_i   
                cca_df_single_ID.at[newID, 'division_frame_i'] = frame_i
                cca_df_single_ID.at[newID, 'generation_num'] = 1  
                cca_df_single_ID.at[newID, 'relative_ID'] = mothID

            new_cells_cca_dfs.append(cca_df_single_ID)
        
        cca_df = pd.concat([cca_df, *new_cells_cca_dfs]).sort_index()
        cca_dfs.append(cca_df)
        pbar.update()
    pbar.close()
    return cca_dfs

def getBaseCca_df(IDs, with_tree_cols=False):
    row_data = base_cca_dict
    if with_tree_cols:
        row_data = {**base_cca_dict, **base_cca_tree_dict}
    data = [row_data]*len(IDs)
    cca_df = pd.DataFrame(data, index=IDs)    
    if with_tree_cols:
       cca_df['Cell_ID_tree'] = IDs
    cca_df.index.name = 'Cell_ID'
    return cca_df

def apply_tracking_from_table(
        segmData, trackColsInfo, src_df, signal=None, logger=print, 
        pbarMax=None, debug=False
    ):
    frameIndexCol = trackColsInfo['frameIndexCol']

    if trackColsInfo['isFirstFrameOne']:
        # Zeroize frames since first frame starts at 1
        src_df[frameIndexCol] = src_df[frameIndexCol] - 1

    logger('Applying tracking info...')  

    grouped = src_df.groupby(frameIndexCol)
    iterable = grouped if signal is not None else tqdm(grouped, ncols=100)
    trackIDsCol = trackColsInfo['trackIDsCol']
    maskIDsCol = trackColsInfo['maskIDsCol']
    xCentroidCol = trackColsInfo['xCentroidCol']
    yCentroidCol = trackColsInfo['yCentroidCol']
    deleteUntrackedIDs = trackColsInfo['deleteUntrackedIDs']
    trackedIDsMapper = {}
    deleteIDsMapper = {}

    for frame_i, df_frame in iterable:
        if frame_i == len(segmData):
            print('')
            logger(
                '[WARNING]: segmentation data has less frames than the '
                f'frames in the "{frameIndexCol}" column.'
            )
            if signal is not None and pbarMax is not None:
                signal.emit(pbarMax-frame_i)
            break

        lab = segmData[frame_i]
        if debug:
            origLab = segmData[frame_i].copy()

        maxTrackID = df_frame[trackIDsCol].max()
        trackIDs = df_frame[trackIDsCol].values

        # print('')
        # print('='*40)
        # print(f'Unique IDs = {np.unique(lab)}')
        # if xCentroidCol == 'None':
        #     print(f'Mask IDs = {df_frame[maskIDsCol].values}')
        # print(f'Frame_i = {frame_i}')

        deleteIDs = []
        if deleteUntrackedIDs:
            if xCentroidCol == 'None':
                maskIDsTracked = df_frame[maskIDsCol].dropna().apply(round).values
            else:
                xx = df_frame[xCentroidCol].dropna().apply(round).values
                yy = df_frame[yCentroidCol].dropna().apply(round).values
                maskIDsTracked = lab[yy, xx]
            for obj in skimage.measure.regionprops(lab):
                if obj.label in maskIDsTracked:
                    continue
                lab[obj.slice][obj.image] = 0
                deleteIDs.append(obj.label)

        if deleteIDs:
            deleteIDsMapper[str(frame_i)] = deleteIDs

        firstPassMapper_i = {}
        # First iterate IDs and make sure there are no overlapping IDs
        for row in df_frame.itertuples():
            trackedID = getattr(row, trackIDsCol)
            if xCentroidCol == 'None':
                maskID = getattr(row, maskIDsCol)
            else:
                xc = getattr(row, xCentroidCol)
                yc = getattr(row, yCentroidCol)
                maskID = lab[round(yc), round(xc)]
            
            if not maskID > 0:
                continue
            
            if maskID == trackedID:
                continue

            if trackedID == 0:
                continue

            if trackedID not in lab:
                continue

            # Assign unique ID to existing tracked ID
            uniqueID = lab.max() + 1
            if uniqueID in trackIDs:
                uniqueID = maxTrackID + 1
                maxTrackID += 1
            
            lab[lab==trackedID] = uniqueID
            firstPassMapper_i[int(trackedID)] = int(uniqueID)
            if xCentroidCol == 'None':
                mask = df_frame[maskIDsCol]==trackedID
                df_frame.loc[mask, maskIDsCol] = int(uniqueID)
            
            # print(f'First = {int(trackedID)} --> {int(uniqueID)}')

        if firstPassMapper_i:
            trackedIDsMapper[str(frame_i)] = {'first_pass': firstPassMapper_i}

        secondPassMapper_i = {}
        for row in df_frame.itertuples():
            trackedID = getattr(row, trackIDsCol)
            if xCentroidCol == 'None':
                maskID = getattr(row, maskIDsCol)
            else:
                xc = getattr(row, xCentroidCol)
                yc = getattr(row, yCentroidCol)
                maskID = lab[round(yc), round(xc)]
            
            if not maskID > 0:
                continue
            
            if maskID == trackedID:
                continue

            lab[lab==maskID] = trackedID
            secondPassMapper_i[int(maskID)] = int(trackedID)   

            # print(f'Second = {int(maskID)} --> {int(trackedID)}')    
        
        if secondPassMapper_i:
            if firstPassMapper_i:
                trackedIDsMapper[str(frame_i)]['second_pass'] = secondPassMapper_i
            else:
                trackedIDsMapper[str(frame_i)] = {'second_pass': secondPassMapper_i}

        if signal is not None:
            signal.emit(1)

        # print('*'*40)
        # import pdb; pdb.set_trace()
  
    return segmData, trackedIDsMapper, deleteIDsMapper

def apply_trackedIDs_mapper_to_acdc_df(
        tracked_IDs_mapper, deleted_IDs_mapper, acdc_df
    ):
    acdc_dfs_renamed = []
    for frame_i, acdc_df_i in acdc_df.groupby(level=0):
        df_renamed = acdc_df_i

        deletedIDs = deleted_IDs_mapper.get(str(frame_i))
        if deletedIDs is not None:
            df_renamed = df_renamed.drop(index=deletedIDs, level=1)

        mapper_i = tracked_IDs_mapper.get(str(frame_i))
        if mapper_i is None:
            acdc_dfs_renamed.append(df_renamed)
            continue
        
        first_pass = mapper_i.get('first_pass')
        if first_pass is not None:
            first_pass = {int(k):int(v) for k,v in first_pass.items()}
            # Substitute mask IDs with tracked IDs
            df_renamed = df_renamed.rename(index=first_pass, level=1)
            if 'relative_ID' in df_renamed.columns:
                relIDs = df_renamed['relative_ID']
                df_renamed['relative_ID'] = relIDs.replace(tracked_IDs_mapper)
            
        second_pass = mapper_i.get('second_pass')
        if second_pass is not None:
            second_pass = {int(k):int(v) for k,v in second_pass.items()}
            # Substitute mask IDs with tracked IDs
            df_renamed = df_renamed.rename(index=second_pass, level=1)
            if 'relative_ID' in df_renamed.columns:
                relIDs = df_renamed['relative_ID']
                df_renamed['relative_ID'] = relIDs.replace(tracked_IDs_mapper)
        
        acdc_dfs_renamed.append(df_renamed)
    
    acdc_df = pd.concat(acdc_dfs_renamed).sort_index()
    return acdc_df

def _get_cca_info_warn_text(
        newID, parentID, frame_i, maskID_colname, x_colname, y_colname,
        df_frame, src_df, frame_idx_colname, trackID_colname
    ):
    txt = (
        f'\n[WARNING]: The parent ID of {newID} at frame index '
        f'{frame_i} is {parentID}, but this parent {parentID} '
        f'does not exist at previous frame {frame_i-1} -->\n'
        f'           --> Setting ID {newID} as a new cell without a parent.\n\n'
        'More details:\n'
    )
    try:
        df_prev_frame = src_df[src_df[frame_idx_colname] == frame_i-1]
        df_prev_frame = df_prev_frame.set_index(trackID_colname)

        if maskID_colname != 'None':
            maskID_of_newID = df_frame.at[newID, maskID_colname]
            maskID_of_parentID = df_prev_frame.at[parentID, maskID_colname]
            details_txt = (
                f'  - "{maskID_colname}" of ID {newID} = {maskID_of_newID}\n'
                f'  - "{maskID_colname}" of ID {parentID} = {maskID_of_parentID}\n'
            )
            txt = f'{txt}{details_txt}'
        if x_colname != 'None':
            xc_of_newID = df_frame.at[newID, x_colname]
            xc_of_parentID = df_prev_frame.at[parentID, x_colname]
            yc_of_newID = df_frame.at[newID, y_colname]
            yc_of_parentID = df_prev_frame.at[parentID, y_colname]
            details_txt = (
                f'  - (x,y) coordinates of ID {newID} = {(xc_of_newID, yc_of_newID)}\n'
                f'  - (x,y) coordinates of ID {parentID} = {(xc_of_parentID, yc_of_parentID)}\n'
            )
            txt = f'{txt}{details_txt}'
    except Exception as e:
        # import pdb; pdb.set_trace()
        pass
    return txt

def add_cca_info_from_parentID_col(
        src_df, acdc_df, frame_idx_colname, IDs_colname, parentID_colname, 
        SizeT, signal=None, trackedData=None, logger=print, 
        maskID_colname='None', x_colname='None', y_colname='None'
    ):
    grouped = src_df.groupby(frame_idx_colname)
    acdc_dfs = []
    keys = []
    iterable = grouped if signal is not None else tqdm(grouped, ncols=100)            
    for frame_i, df_frame in iterable:
        if frame_i == SizeT:
            break
        
        if trackedData is not None:
            lab = trackedData[frame_i]

        df_frame = df_frame.set_index(IDs_colname)
        acdc_df_i = acdc_df.loc[frame_i].copy()
        IDs = acdc_df_i.index.values
        cca_df = getBaseCca_df(IDs, with_tree_cols=True)
        if frame_i == 0:
            prevIDs = []
            oldIDs = []
            newIDs = IDs
        else:
            prevIDs = acdc_df.loc[frame_i-1].index.values
            newIDs = [ID for ID in IDs if ID not in prevIDs]
            oldIDs = [ID for ID in IDs if ID in prevIDs]
        
        if oldIDs:
            # For the oldIDs copy from previous cca_df
            prev_acdc_df = acdc_dfs[frame_i-1].filter(oldIDs, axis=0)
            cca_df.loc[prev_acdc_df.index] = prev_acdc_df

        for newID in newIDs:
            try:
                parentID = int(df_frame.at[newID, parentID_colname])
            except Exception as e:
                parentID = -1
            
            parentGenNum = None
            if parentID > 1:
                prev_acdc_df = acdc_dfs[frame_i-1]
                try:
                    parentGenNum = prev_acdc_df.at[parentID, 'generation_num']
                except Exception as e:
                    parentGenNum = None
                    logger('*'*40)
                    warn_txt = _get_cca_info_warn_text(
                        newID, parentID, frame_i, maskID_colname, x_colname, 
                        y_colname, df_frame, src_df, frame_idx_colname, 
                        IDs_colname
                    )
                    logger(warn_txt)
                    logger('*'*40)
            
            if parentGenNum is not None:
                prentGenNumTree = (
                    prev_acdc_df.at[parentID, 'generation_num_tree']
                )
                newGenNumTree = prentGenNumTree+1
                parentRootID = (
                    prev_acdc_df.at[parentID, 'root_ID_tree']
                )
                cca_df.at[newID, 'is_history_known'] = True
                cca_df.at[newID, 'cell_cycle_stage'] = 'G1'
                cca_df.at[newID, 'generation_num'] = parentGenNum+1
                cca_df.at[newID, 'emerg_frame_i'] = frame_i
                cca_df.at[newID, 'division_frame_i'] = frame_i
                cca_df.at[newID, 'relationship'] = 'mother'
                cca_df.at[newID, 'generation_num_tree'] = newGenNumTree
                cca_df.at[newID, 'Cell_ID_tree'] = newID
                cca_df.at[newID, 'root_ID_tree'] = parentRootID
                cca_df.at[newID, 'parent_ID_tree'] = parentID
                # sister ID is the other cell with the same parent ID
                sisterIDmask = (
                    (df_frame[parentID_colname] == parentID)
                    & (df_frame.index != newID)
                )
                sisterID_df = df_frame[sisterIDmask]
                if len(sisterID_df) == 1:
                    sisterID = sisterID_df.index[0]
                else:
                    sisterID = -1
                cca_df.at[newID, 'sister_ID_tree'] = sisterID
            else:
                # Set new ID without a parent as history unknown
                cca_df.at[newID, 'is_history_known'] = False
                cca_df.at[newID, 'cell_cycle_stage'] = 'G1'
                cca_df.at[newID, 'generation_num'] = 2
                cca_df.at[newID, 'emerg_frame_i'] = frame_i
                cca_df.at[newID, 'division_frame_i'] = -1
                cca_df.at[newID, 'relationship'] = 'mother'
                cca_df.at[newID, 'generation_num_tree'] = 1
                cca_df.at[newID, 'Cell_ID_tree'] = newID
                cca_df.at[newID, 'root_ID_tree'] = newID
                cca_df.at[newID, 'parent_ID_tree'] = -1
                cca_df.at[newID, 'sister_ID_tree'] = -1
        
        acdc_df_i[cca_df.columns] = cca_df
        acdc_dfs.append(acdc_df_i)
        keys.append(frame_i)
        if signal is not None:
            signal.emit(1)
    
    if acdc_dfs:
        acdc_df_with_cca = pd.concat(
            acdc_dfs, keys=keys, names=['frame_i', 'Cell_ID']
        )
        if len(acdc_df_with_cca) == len(acdc_df):
            # All frames from existing acdc_df were cca annotated in src_table
            acdc_df_with_cca = pd.concat(
                acdc_dfs, keys=keys, names=['frame_i', 'Cell_ID']
            )
            return acdc_df_with_cca
        else:
            # Only a subset of frames already present in acdc_df were annotated in src_table
            acdc_df[cca_df.columns] = np.nan
            for frame_i, acdc_df_i_with_cca in acdc_df_with_cca.groupby(level=0):
                acdc_df.loc[acdc_df_i_with_cca.index] = acdc_df_i_with_cca
    else:
        # No annotations present in src_table
        return acdc_df
    
    
    return acdc_df

def cca_df_to_acdc_df(cca_df, rp, acdc_df=None):
    if acdc_df is None:
        IDs = []
        is_cell_dead_li = []
        is_cell_excluded_li = []
        xx_centroid = []
        yy_centroid = []
        for obj in rp:
            IDs.append(obj.label)
            is_cell_dead_li.append(0)
            is_cell_excluded_li.append(0)
            xx_centroid.append(int(obj.centroid[1]))
            yy_centroid.append(int(obj.centroid[0]))
        acdc_df = pd.DataFrame({
            'Cell_ID': IDs,
            'is_cell_dead': is_cell_dead_li,
            'is_cell_excluded': is_cell_excluded_li,
            'x_centroid': xx_centroid,
            'y_centroid': yy_centroid,
            'was_manually_edited': is_cell_excluded_li.copy()
        }).set_index('Cell_ID')

    acdc_df = acdc_df.join(cca_df, how='left')
    return acdc_df

class LineageTree:
    def __init__(self, acdc_df, logging_func=print, debug=False) -> None:
        acdc_df = load.pd_bool_to_int(acdc_df).reset_index()
        acdc_df = self._normalize_gen_num(acdc_df).reset_index()
        acdc_df = acdc_df.drop(columns=['index', 'level_0'], errors='ignore')
        self.acdc_df = acdc_df.set_index(['frame_i', 'Cell_ID'])
        self.df = acdc_df.copy()
        self.cca_df_colnames = cca_df_colnames
        self.log = logging_func
        self.debug = debug
    
    def build(self):
        self.log('Building lineage tree...')
        try:
            df_G1 = self.acdc_df[self.acdc_df['cell_cycle_stage'] == 'G1']
            self.df_G1 = df_G1[self.cca_df_colnames].copy()
            self.new_col_loc = df_G1.columns.get_loc('division_frame_i') + 1
        except Exception as error:
            return error
        
        self.df = self.add_lineage_tree_table_to_acdc_df()
        self.log('Lineage tree built successfully!')
    
    def _normalize_gen_num(self, acdc_df):
        '''
        Since the user is allowed to start the generation_num of unknown mother
        cells with any number we need to normalise this to 2 -->
        Create a new 'normalized_gen_num' column where we make sure that mother
        cells with unknown history have 'normalized_gen_num' starting from 2
        (required by the logic of _build_tree)
        '''
        acdc_df = acdc_df.drop(columns=['level_0', 'index'], errors='ignore')
        acdc_df = (
            acdc_df.reset_index()
            .drop(columns='index', errors='ignore')
        )

        # Get the starting generation number of the unknown mother cells
        df_emerg = acdc_df.groupby('Cell_ID').agg('first')
        history_unknown_mask = df_emerg['is_history_known'] == 0
        moth_mask = df_emerg['relationship'] == 'mother'
        df_emerg_moth_uknown = df_emerg[(history_unknown_mask) & (moth_mask)]

        # Get the difference from 2
        df_diff = 2 - df_emerg_moth_uknown['generation_num']

        # Build a normalizing df with the number to be added for each cell
        normalizing_df = pd.DataFrame(
            data=acdc_df[['frame_i', 'Cell_ID']]
        ).set_index('Cell_ID')
        normalizing_df['gen_num_diff'] = 0
        normalizing_df.loc[df_emerg_moth_uknown.index, 'gen_num_diff'] = (
            df_diff
        )

        # Add the normalising_df to create the new normalized_gen_num col
        normalizing_df = normalizing_df.reset_index().set_index(
            ['frame_i', 'Cell_ID']
        )
        acdc_df = acdc_df.set_index(['frame_i', 'Cell_ID'])
        acdc_df['normalized_gen_num'] = (
            acdc_df['generation_num'] + normalizing_df['gen_num_diff']
        )
        return acdc_df
    
    def _build_tree(self, gen_df, ID):
        current_ID = gen_df.index.get_level_values(1)[0]
        if current_ID != ID:
            return gen_df

        '''
        Add generation number tree:
        --> At the start of a branch we set the generation number as either 
            0 (if also start of tree) or relative ID generation number tree
            --> This value called gen_num_relID_tree is added to the current 
                generation_num
        '''
        ID_slice = pd.IndexSlice[:, ID]
        relID = gen_df.loc[ID_slice, 'relative_ID'].iloc[0]
        relID_slice = pd.IndexSlice[:, relID]
        gen_nums_tree = gen_df['generation_num_tree'].values
        start_frame_i = gen_df.index.get_level_values(0)[0]
        if self.is_new_tree:
            try:  
                gen_num_relID_tree = self.df_G1.at[
                    (start_frame_i, relID), 'generation_num_tree'
                ] - 1
            except Exception as e:
                gen_num_relID_tree = 0
            self.branch_start_gen_num[ID] = gen_num_relID_tree
        else:
            gen_num_relID_tree = self.branch_start_gen_num[ID]
        
        updated_gen_nums_tree = gen_nums_tree + gen_num_relID_tree
        gen_df['generation_num_tree'] = updated_gen_nums_tree       
        
        '''Assign unique ID every consecutive division'''
        if self.is_new_tree:
            # Keep start ID for cell at the top of the branch
            Cell_ID_tree = ID
            gen_df['Cell_ID_tree'] = [ID]*len(gen_df)
        else:
            Cell_ID_tree = self.uniqueID
            self.uniqueID += 1
        
        gen_df['Cell_ID_tree'] = [Cell_ID_tree]*len(gen_df)

        '''
        Assign parent ID --> existing ID between relID and ID in prev gen_num_tree      
        '''
        gen_num_tree = gen_df.loc[ID_slice, 'generation_num_tree'].iloc[0]
        
        prev_gen_G1_existing = True
        if gen_num_tree > 1:        
            prev_gen_num_tree = gen_num_tree - 1
            try:
                # Parent ID is the Cell_ID_tree that current ID had in prev gen
                prev_gen_df = self.gen_dfs[(ID, prev_gen_num_tree)]
            except Exception as e:
                # Parent ID is the Cell_ID_tree that the relative of the 
                # current ID had in prev gen
                try:
                    prev_gen_df = self.gen_dfs[(relID, prev_gen_num_tree)]
                except Exception as e:
                    # Cell has not previous gen because the gen_num_tree
                    # starts at 2 (cell appeared in S and then started G1)
                    prev_gen_G1_existing = False
                    pass
            
            if prev_gen_G1_existing:
                try:
                    parent_ID = prev_gen_df.loc[relID_slice, 'Cell_ID_tree'].iloc[0]
                except Exception as e:
                    parent_ID = prev_gen_df.loc[ID_slice, 'Cell_ID_tree'].iloc[0]
                gen_df['parent_ID_tree'] = parent_ID
            else:
                # Cell appeared in S in previous frame
                idx = (start_frame_i-1, ID)
                # import pdb; pdb.set_trace()
                was_bud = self.acdc_df.loc[idx, 'relationship'] == 'bud'
                if was_bud:
                    # This is a bud of the first frame where the algorithm 
                    # thinks is a new tree --> correct
                    parent_ID = self.acdc_df.loc[idx, 'relative_ID']
                    self.branch_start_gen_num[ID] = (
                        self.branch_start_gen_num[parent_ID] + 2
                    )
                else:
                    parent_ID = ID
                
                Cell_ID_tree = self.uniqueID
                self.uniqueID += 1
                gen_df['Cell_ID_tree'] = [Cell_ID_tree]*len(gen_df)
        else:
            parent_ID = -1
        
        '''
        Assign root ID --> 
            at start of branch (self.is_new_tree is True) the root_ID
            is ID if gen_num_tree == 1 otherwise we go back until 
            the parent_ID == -1
            --> store this and use when traversing branch
        '''
        if self.is_new_tree:
            if gen_num_tree == 2 and prev_gen_G1_existing:
                root_ID_tree = parent_ID
            elif gen_num_tree > 2:
                prev_gen_num_tree = gen_num_tree - 1
                prev_gen_idx = parent_ID
                parent_ID_df = self.gen_dfs_by_ID_tree[prev_gen_idx]
                root_ID_tree = parent_ID_df['parent_ID_tree'].iloc[0]
                while prev_gen_num_tree > 2:
                    prev_gen_num_tree -= 1
                    prev_gen_idx = root_ID_tree
                    parent_ID_df = self.gen_dfs_by_ID_tree[prev_gen_idx]
                    root_ID_tree = parent_ID_df['parent_ID_tree'].iloc[0]    
                if root_ID_tree == -1:
                    root_ID_tree = parent_ID_df['root_ID_tree'].iloc[0] 
            elif parent_ID > 0:
                # We started a new tree of a bud that appeared already in S
                # --> the root ID is the parent_ID (mother cell)
                root_ID_tree = parent_ID
            else:
                root_ID_tree = ID
            self.root_IDs_trees[ID] = root_ID_tree
        else:
            root_ID_tree = self.root_IDs_trees[ID]
        
        gen_df['root_ID_tree'] = root_ID_tree     

        if self.debug:
            printl(
                f'Traversing ID: {ID}\n'
                f'Parent ID: {parent_ID}\n'
                f'Is new tree: {self.is_new_tree}\n'
                f'Relative ID: {relID}\n'
                f'Relative ID generation num tree: {gen_num_relID_tree}\n'
                f'Generation number tree: {gen_num_tree}\n'
                f'New cell ID tree: {Cell_ID_tree}\n'
                f'Start branch gen number: {self.branch_start_gen_num[ID]}\n'
                f'Start of tree frame n.: {start_frame_i+1}\n'
                f'root_ID_tree: {root_ID_tree}'
            )
            import pdb; pdb.set_trace()
            
        self.gen_dfs[(ID, gen_num_tree)] = gen_df
        self.gen_dfs_by_ID_tree[Cell_ID_tree] = gen_df
        
        self.is_new_tree = False
        return gen_df
             
    def add_lineage_tree_table_to_acdc_df(self):
        Cell_ID_tree_vals = self.df_G1.index.get_level_values(1)
        self.df_G1['Cell_ID_tree'] = Cell_ID_tree_vals
        self.df_G1['parent_ID_tree'] = -1
        self.df_G1['root_ID_tree'] = -1
        self.df_G1['sister_ID_tree'] = -1
            
        self.df_G1['generation_num_tree'] = self.df_G1['generation_num']
        
        # For cells that starts at ccs = 2 subtract 1
        history_unknown_mask = self.df_G1['is_history_known'] == 0
        ccs_greater_one_mask = self.df_G1['generation_num'] > 1
        subtract_gen_num_mask = (history_unknown_mask) & (ccs_greater_one_mask)
        self.df_G1.loc[subtract_gen_num_mask, 'generation_num_tree'] = (
            self.df_G1.loc[subtract_gen_num_mask, 'generation_num'] - 1
        )
        
        cols_tree = [col for col in self.df_G1.columns if col.endswith('_tree')]

        frames_idx = self.df_G1.dropna().index.get_level_values(0).unique()
        not_annotated_IDs = self.df_G1.index.get_level_values(1).unique().to_list()

        self.uniqueID = max(not_annotated_IDs) + 1

        self.gen_dfs = {}
        self.gen_dfs_by_ID_tree = {}
        self.root_IDs_trees = {}
        self.branch_start_gen_num = {}
        for frame_i in frames_idx:
            if not not_annotated_IDs:
                # Built tree for every ID --> exit
                break
            
            df_i = self.df_G1.loc[frame_i]
            IDs = df_i.index.array
            for ID in IDs:
                if ID not in not_annotated_IDs:
                    # Tree already built in previous frame iteration --> skip
                    continue
                      
                self.is_new_tree = True
                # Iterate the branch till the end
                df_tree_iter = (
                    self.df_G1
                    .groupby(['Cell_ID', 'generation_num'], group_keys=False)
                    .apply(self._build_tree, ID)
                )
                self.df_G1 = df_tree_iter
                not_annotated_IDs.remove(ID)
        
        self._add_sister_ID()

        for c, col_tree in enumerate(cols_tree):
            if col_tree in self.acdc_df.columns:
                self.acdc_df.pop(col_tree)
            self.acdc_df.insert(self.new_col_loc, col_tree, 0)

        self.acdc_df.loc[self.df_G1.index, self.df_G1.columns] = self.df_G1
        self._build_tree_S(cols_tree)

        return self.acdc_df
    
    def _err_msg_add_sister_ID(self, relative_ID, frame_i, df):
        ID = df.index.get_level_values(1)[0]
        txt = (
            f'There is a problem with Cell ID {relative_ID} '
            f'at frame n. {frame_i+1}. '
            'Make sure that annotations are correct before trying again.\n\n'
            'More info: error happened when trying to set the `sister_ID` of '
            f'cell ID {ID} to {relative_ID}. It might be that ID {relative_ID} '
            f'is not in G1 at frame n. {frame_i+1}'
        )
        return txt
    
    def _add_sister_ID(self):
        grouped_ID_tree = self.df_G1.groupby('Cell_ID_tree')
        for Cell_ID_tree, df in grouped_ID_tree:
            relative_ID = df['relative_ID'].iloc[0]
            if relative_ID == -1:
                continue
            start_frame_i = df.index.get_level_values(0)[0]
            try:
                sister_ID_tree = self.df_G1.at[
                    (start_frame_i, relative_ID), 'Cell_ID_tree'
                ]
            except KeyError as error:
                raise KeyError(
                    self._err_msg_add_sister_ID(relative_ID, start_frame_i, df)
                ) from error
                
            self.df_G1.loc[df.index, 'sister_ID_tree'] = sister_ID_tree
    
    def _build_tree_S(self, cols_tree):
        '''In S we consider the bud still the same as the mother in the tree
        --> either copy the tree information from the G1 phase or, in case 
        the cell doesn't have a G1 (before S) because it appeared already in S,
        copy from the current S phase (e.g., Cell_ID_tree = Cell_ID)
        '''
        S_mask = self.acdc_df['cell_cycle_stage'] == 'S'
        df_S = self.acdc_df[S_mask].copy()
        gen_acdc_df = self.acdc_df.reset_index().set_index(
            ['Cell_ID', 'generation_num', 'cell_cycle_stage']
        ).sort_index()
        for row_S in df_S.itertuples():
            relationship = row_S.relationship
            if relationship == 'mother':
                idx_ID = row_S.Index[1]
                idx_gen_num = row_S.generation_num
            else:
                idx_ID = row_S.relative_ID
                frame_i = row_S.Index[0]
                idx_gen_num = self.acdc_df.at[(frame_i, idx_ID), 'generation_num']
            cc_df = gen_acdc_df.loc[(idx_ID, idx_gen_num)]
            if 'G1' in cc_df.index:
                row_G1 = cc_df.loc[['G1']].iloc[0]
                for col_tree in cols_tree:
                    self.acdc_df.loc[row_S.Index, col_tree] = row_G1[col_tree]
            else:
                # Cell that was already in S at appearance --> There is not G1 to copy from
                sister_ID = cc_df.iloc[0]['relative_ID']
                self.acdc_df.loc[row_S.Index, 'Cell_ID_tree'] = idx_ID
                self.acdc_df.loc[row_S.Index, 'parent_ID_tree'] = -1
                self.acdc_df.loc[row_S.Index, 'root_ID_tree'] = idx_ID
                self.acdc_df.loc[row_S.Index, 'generation_num_tree'] = 1
                self.acdc_df.loc[row_S.Index, 'sister_ID_tree'] = sister_ID
    
    def newick(self):
        if 'Cell_ID_tree' not in self.acdc_df.columns:
            self.build()
        
        df = self.df.reset_index()
    
    def plot(self):
        if 'Cell_ID_tree' not in self.acdc_df.columns:
            self.build()
        
        df = self.df.reset_index()
    
    def to_arboretum(self, rebuild=False):
        # See https://github.com/lowe-lab-ucl/arboretum/blob/main/examples/show_sample_data.py
        if 'Cell_ID_tree' not in self.acdc_df.columns or rebuild:
            self.build()

        df = self.df.reset_index()
        tracks_cols = ['Cell_ID_tree', 'frame_i', 'y_centroid', 'x_centroid']
        tracks_data = df[tracks_cols].to_numpy()

        graph_df = df.groupby('Cell_ID_tree').agg('first').reset_index()
        graph_df = graph_df[graph_df.parent_ID_tree > 0]
        graph = {
            child_ID:[parent_ID] for child_ID, parent_ID 
            in zip(graph_df.Cell_ID_tree, graph_df.parent_ID_tree)
        }

        properties = pd.DataFrame({
            't': df.frame_i,
            'root': df.root_ID_tree,
            'parent': df.parent_ID_tree
        })

        return tracks_data, graph, properties

def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def preprocess_image_from_recipe(image, recipe: dict):
    for step in recipe:
        method = step['method']
        func = PREPROCESS_MAPPER[method]['function']
        kwargs = step['kwargs']
        image = func(image, **kwargs)
    
    return image

def segm_model_segment(
        model, image, model_kwargs, frame_i=None, preproc_recipe=None, 
        is_timelapse_model=False, posData=None, start_z_slice=0
    ):
    if preproc_recipe is not None:
        if is_timelapse_model:
            filtered_image = np.zeros(image.shape)
            for i, img in enumerate(image):
                img = preprocess_image_from_recipe(img, preproc_recipe)
                filtered_image[i] = img
            image = filtered_image # .astype(image.dtype)
        else:
            image = preprocess_image_from_recipe(image, preproc_recipe)
    
    if is_timelapse_model:
        segm_data = model.segment3DT(image, **model_kwargs)
        return segm_data             
    
    # Some models have `start_z_slice` kwarg
    try:
        lab = model.segment(
            image, frame_i, posData=posData, start_z_slice=start_z_slice, 
            **model_kwargs
        )
        return lab
    except TypeError as err:
        if str(err).find('unexpected keyword argument') == -1:
            # Raise error since it's not about the missing posData kwarg
            raise err
    
    # Some models have posData as kwarg and frame_i as second arg
    try:
        lab = model.segment(image, frame_i, posData=posData, **model_kwargs)
        return lab
    except TypeError as err:
        if str(err).find('unexpected keyword argument') == -1:
            # Raise error since it's not about the missing posData kwarg
            raise err
    
    # Some models have frame_i as second arg
    try:
        lab = model.segment(image, frame_i, **model_kwargs)
        return lab
    except TypeError as err:
        pass

    lab = model.segment(image, **model_kwargs)
    return lab

class _WorkflowKernel:
    def __init__(self, logger, log_path, is_cli=False):
        self.logger = logger
        self.log_path = log_path
        self.is_cli = True
    
    def quit(self, error=None):
        if not self.is_cli and error is not None:
            raise error
        
        self.logger.info('='*50)
        if error is not None:
            self.logger.exception(traceback.format_exc())
            print('-'*60)
            self.logger.info(f'[ERROR]: {error}{error_up_str}')
            err_msg = (
                'Cell-ACDC aborted due to **error**. '
                'More details above or in the following log file:\n\n'
                f'{self.log_path}\n\n'
                'If you cannot solve it, you can report this error by opening '
                'an issue on our '
                'GitHub page at the following link:\n\n'
                f'{issues_url}\n\n'
                'Please **send the log file** when reporting a bug, thanks!'
            )
            self.logger.info(err_msg)
        else:
            self.logger.info(
                'Cell-ACDC command-line interface closed. '
                f'{myutils.get_salute_string()}'
            )
        self.logger.info('='*50)
        exit()

class SegmKernel(_WorkflowKernel):
    def __init__(self, logger, log_path, is_cli):
        super().__init__(logger, log_path, is_cli=is_cli)
    
    @exception_handler_cli
    def parse_paths(self, workflow_params):
        paths_to_segm = workflow_params['paths_to_segment']['paths']
        ch_name = workflow_params['initialization']['user_ch_name']
        parsed_paths = []
        for path in paths_to_segm:
            if os.path.isfile(path):
                parsed_paths.append(path)
                continue
            
            images_paths = load.get_images_paths(path)
            ch_filepaths = load.get_user_ch_paths(images_paths, ch_name)
            parsed_paths.extend(ch_filepaths)
        return parsed_paths

    @exception_handler_cli
    def parse_stop_frame_numbers(self, workflow_params):
        stop_frames_param = (
            workflow_params['paths_to_segment']['stop_frame_numbers']
        )
        return [int(n) for n in stop_frames_param]

    @exception_handler_cli
    def parse_custom_postproc_features_grouped(self, workflow_params):
        custom_postproc_grouped_features = {}
        for section, options in workflow_params.items():
            if not section.startswith('postprocess_features.'):
                continue
            category = section.split('.')[-1]
            for option, value in options.items():
                if option == 'names':
                    values = value.strip('\n')
                    values = value.split('\n')
                    custom_postproc_grouped_features[category] = values
                    continue
                channel = option
                if category not in custom_postproc_grouped_features:
                    custom_postproc_grouped_features[category] = {
                        channel: [value]
                    }
                elif channel not in custom_postproc_grouped_features[category]:
                    custom_postproc_grouped_features[category][channel] = (
                        [value]
                    )
                else:
                    custom_postproc_grouped_features[category][channel].append(value)
        return custom_postproc_grouped_features
    
    @exception_handler_cli      
    def init_args_from_params(self, workflow_params, logger_func):
        args = workflow_params['initialization'].copy()
        args['use3DdataFor2Dsegm'] = workflow_params.get(
            'use3DdataFor2Dsegm', False
        )
        args['model_kwargs'] = workflow_params['segmentation_model_params']
        args['track_params'] = workflow_params.get('tracker_params', {})
        args['standard_postrocess_kwargs'] = (
            workflow_params.get('standard_postprocess_features', {})
        )
        args['custom_postproc_features'] = (
            workflow_params.get('custom_postprocess_features', {})
        )
        args['custom_postproc_grouped_features'] = (
            self.parse_custom_postproc_features_grouped(workflow_params)
        )
        
        args['SizeT'] = workflow_params['metadata']['SizeT']
        args['SizeZ'] = workflow_params['metadata']['SizeZ']
        args['logger_func'] = logger_func
        args['init_model_kwargs'] = (
            workflow_params.get('init_segmentation_model_params', {})
        )
        args['init_tracker_kwargs'] = (
            workflow_params.get('init_tracker_params', {})
        )
        
        args['preproc_recipe'] = config.preprocess_ini_items_to_recipe(
            workflow_params
        )
        
        self.init_args(**args)
    
    @exception_handler_cli
    def init_args(
            self, 
            user_ch_name, 
            segm_endname,
            model_name, 
            do_tracking,
            do_postprocess, 
            do_save,
            image_channel_tracker,
            standard_postrocess_kwargs,
            custom_postproc_grouped_features,
            custom_postproc_features,
            isSegm3D,
            use_ROI,
            second_channel_name,
            use3DdataFor2Dsegm,
            model_kwargs, 
            track_params,
            SizeT, 
            SizeZ,
            tracker_name='',
            model=None,
            preproc_recipe=None,
            init_model_kwargs=None,
            init_tracker_kwargs=None,
            tracker=None,
            signals=None,
            logger_func=print,
            innerPbar_available=False,
            is_segment3DT_available=False, 
        ):
        self.user_ch_name = user_ch_name
        self.segm_endname = segm_endname
        self.model_name = model_name
        self.do_postprocess = do_postprocess
        self.standard_postrocess_kwargs = standard_postrocess_kwargs
        self.custom_postproc_grouped_features = custom_postproc_grouped_features
        self.custom_postproc_features = custom_postproc_features
        self.do_tracking = do_tracking
        self.do_save = do_save
        self.image_channel_tracker = image_channel_tracker
        self.isSegm3D = isSegm3D
        self.use3DdataFor2Dsegm = use3DdataFor2Dsegm
        self.use_ROI = use_ROI
        self.second_channel_name = second_channel_name
        self.logger_func = logger_func
        self.innerPbar_available = innerPbar_available
        self.SizeT = SizeT
        self.SizeZ = SizeZ
        self.init_model_kwargs = init_model_kwargs
        self.init_tracker_kwargs = init_tracker_kwargs
        self.is_segment3DT_available = is_segment3DT_available
        self.preproc_recipe = preproc_recipe
        if signals is None:
            self.signals = KernelCliSignals(logger_func)
        else:
            self.signals = signals
        self.model = model
        self.model_kwargs = model_kwargs
        self.tracker_name = tracker_name
        self.init_tracker(
            self.do_tracking, track_params, tracker_name=tracker_name, 
            tracker=tracker
        )
    
    @exception_handler_cli
    def init_segm_model(self, posData):
        self.signals.progress.emit(
            f'\nInitializing {self.model_name} segmentation model...'
        )
        acdcSegment = myutils.import_segment_module(self.model_name)
        init_argspecs, segment_argspecs = myutils.getModelArgSpec(acdcSegment)
        self.init_model_kwargs = myutils.parse_model_params(
            init_argspecs, self.init_model_kwargs
        )
        self.model_kwargs = myutils.parse_model_params(
            segment_argspecs, self.model_kwargs
        )
        self.model = myutils.init_segm_model(
            acdcSegment, posData, self.init_model_kwargs
        )
        self.is_segment3DT_available = any(
            [name=='segment3DT' for name in dir(self.model)]
        )
    
    @exception_handler_cli
    def init_tracker(
            self, do_tracking, track_params, tracker_name='', tracker=None
        ):
        if not do_tracking:
            self.tracker = None
            return
        
        if tracker is None:
            self.signals.progress.emit(f'Initializing {tracker_name} tracker...')
            tracker_module = myutils.import_tracker_module(tracker_name)
            init_argspecs, track_argspecs = myutils.getTrackerArgSpec(
                tracker_module, realTime=False
            )
            self.init_tracker_kwargs = myutils.parse_model_params(
                init_argspecs, self.init_tracker_kwargs
            )
            self.init_tracker_kwargs = myutils.parse_model_params(
                init_argspecs, self.init_tracker_kwargs
            )
            track_params = myutils.parse_model_params(
                track_argspecs, track_params
            )
            tracker = tracker_module.tracker(**self.init_tracker_kwargs)
            
        self.track_params = track_params
        self.tracker = tracker
    
    @exception_handler_cli
    def run(
            self,
            img_path,  
            stop_frame_n
        ):    
        posData = load.loadData(img_path, self.user_ch_name)

        self.logger_func(f'Loading {posData.relPath}...')

        posData.getBasenameAndChNames()
        posData.buildPaths()
        posData.loadImgData()
        posData.loadOtherFiles(
            load_segm_data=False,
            load_acdc_df=False,
            load_shifts=True,
            loadSegmInfo=True,
            load_delROIsInfo=False,
            load_dataPrep_ROIcoords=True,
            load_bkgr_data=True,
            load_last_tracked_i=False,
            load_metadata=True,
            end_filename_segm=self.segm_endname
        )
        # Get only name from the string 'segm_<name>.npz'
        endName = (
            self.segm_endname.replace('segm', '', 1)
            .replace('_', '', 1)
            .split('.')[0]
        )
        if endName:
            # Create a new file that is not the default 'segm.npz'
            posData.setFilePaths(endName)

        segmFilename = os.path.basename(posData.segm_npz_path)
        if self.do_save:
            self.logger_func(f'\nSegmentation file {segmFilename}...')

        posData.SizeT = self.SizeT
        if self.SizeZ > 1:
            SizeZ = posData.img_data.shape[-3]
            posData.SizeZ = SizeZ
        else:
            posData.SizeZ = 1

        posData.isSegm3D = self.isSegm3D
        posData.saveMetadata()
        
        isROIactive = False
        if posData.dataPrep_ROIcoords is not None and self.use_ROI:
            df_roi = posData.dataPrep_ROIcoords.loc[0]
            isROIactive = df_roi.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = df_roi['value'].astype(int)[:4]
            Y, X = posData.img_data.shape[-2:]
            x0 = x0 if x0>0 else 0
            y0 = y0 if y0>0 else 0
            x1 = x1 if x1<X else X
            y1 = y1 if y1<Y else Y

        # Note that stop_i is not used when SizeT == 1 so it does not matter
        # which value it has in that case
        stop_i = stop_frame_n

        if self.second_channel_name is not None:
            self.logger_func(
                f'Loading second channel "{self.second_channel_name}"...'
            )
            secondChFilePath = load.get_filename_from_channel(
                posData.images_path, self.second_channel_name
            )
            secondChImgData = load.load_image_file(secondChFilePath)

        if posData.SizeT > 1:
            self.t0 = 0
            if posData.SizeZ > 1 and not self.isSegm3D and not self.use3DdataFor2Dsegm:
                # 2D segmentation on 3D data over time
                img_data = posData.img_data
                if self.second_channel_name is not None:
                    second_ch_data_slice = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data_slice = second_ch_data_slice[:, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))

                img_data_slice = img_data[self.t0:stop_i]
                postprocess_img = img_data
                
                Y, X = img_data.shape[-2:]
                newShape = (stop_i, Y, X)
                img_data = np.zeros(newShape, img_data.dtype)
                
                if self.second_channel_name is not None:
                    second_ch_data = np.zeros(newShape, secondChImgData.dtype)
                df = posData.segmInfo_df.loc[posData.filename]
                for z_info in df[:stop_i].itertuples():
                    i = z_info.Index
                    z = z_info.z_slice_used_dataPrep
                    zProjHow = z_info.which_z_proj
                    img = img_data_slice[i]
                    if self.second_channel_name is not None:
                        second_ch_img = second_ch_data_slice[i]
                    if zProjHow == 'single z-slice':
                        img_data[i] = img[z]
                        if self.second_channel_name is not None:
                            second_ch_data[i] = second_ch_img[z]
                    elif zProjHow == 'max z-projection':
                        img_data[i] = img.max(axis=0)
                        if self.second_channel_name is not None:
                            second_ch_data[i] = second_ch_img.max(axis=0)
                    elif zProjHow == 'mean z-projection':
                        img_data[i] = img.mean(axis=0)
                        if self.second_channel_name is not None:
                            second_ch_data[i] = second_ch_img.mean(axis=0)
                    elif zProjHow == 'median z-proj.':
                        img_data[i] = np.median(img, axis=0)
                        if self.second_channel_name is not None:
                            second_ch_data[i] = np.median(second_ch_img, axis=0)
            elif posData.SizeZ > 1 and (self.isSegm3D or self.use3DdataFor2Dsegm):
                # 3D segmentation on 3D data over time
                img_data = posData.img_data[self.t0:stop_i]
                postprocess_img = img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, :, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (0, 0), (y0, Y-y1), (x0, X-x1))
            else:
                # 2D data over time
                img_data = posData.img_data[self.t0:stop_i]
                postprocess_img = img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
        else:
            if posData.SizeZ > 1 and not self.isSegm3D:
                img_data = posData.img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]

                postprocess_img = img_data
                # 2D segmentation on single 3D image
                z_info = posData.segmInfo_df.loc[posData.filename].iloc[0]
                z = z_info.z_slice_used_dataPrep
                zProjHow = z_info.which_z_proj
                if zProjHow == 'single z-slice':
                    img_data = img_data[z]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[z]
                elif zProjHow == 'max z-projection':
                    img_data = img_data.max(axis=0)
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data.max(axis=0)
                elif zProjHow == 'mean z-projection':
                    img_data = img_data.mean(axis=0)
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data.mean(axis=0)
                elif zProjHow == 'median z-proj.':
                    img_data = np.median(img_data, axis=0)
                    if self.second_channel_name is not None:
                        second_ch_data[i] = np.median(second_ch_data, axis=0)
            elif posData.SizeZ > 1 and self.isSegm3D:
                # 3D segmentation on 3D z-stack
                img_data = posData.img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, y0:y1, x0:x1]
                postprocess_img = img_data
            else:
                # Single 2D image
                img_data = posData.img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[y0:y1, x0:x1]
                postprocess_img = img_data

        self.logger_func(f'\nImage shape = {img_data.shape}')

        if self.model is None:
            self.init_segm_model(posData)
        
        """Segmentation routine"""
        self.logger_func(f'\nSegmenting with {self.model_name}...')
        t0 = time.perf_counter()
        # self.logger_func(f'Segmenting with {model} (Ctrl+C to abort)...')
        if posData.SizeT > 1:
            if self.innerPbar_available and self.signals is not None:
                self.signals.resetInnerPbar.emit(len(img_data))
            
            if self.is_segment3DT_available:
                self.model_kwargs['signals'] = (
                    self.signals, self.innerPbar_available
                )
                if self.second_channel_name is not None:
                    img_data = self.model.to_rgb_stack(img_data, second_ch_data)
                lab_stack = segm_model_segment(
                    self.model, img_data, self.model_kwargs, 
                    is_timelapse_model=True, 
                    preproc_recipe=self.preproc_recipe, 
                    posData=posData
                )
                if self.innerPbar_available:
                    # emit one pos done
                    self.signals.progressBar.emit(1)
            else:
                lab_stack = []
                pbar = tqdm(total=len(img_data), ncols=100)
                for t, img in enumerate(img_data):
                    if self.second_channel_name is not None:
                        img = self.model.to_rgb_stack(img, second_ch_data[t])
                        
                    lab = segm_model_segment(
                        self.model, img, self.model_kwargs, frame_i=t, 
                        preproc_recipe=self.preproc_recipe, 
                        posData=posData
                    )
                    lab_stack.append(lab)
                    if self.innerPbar_available:
                        self.signals.innerProgressBar.emit(1)
                    else:
                        self.signals.progressBar.emit(1)
                    pbar.update()
                pbar.close()
                lab_stack = np.array(lab_stack, dtype=np.uint32)
                if self.innerPbar_available:
                    # emit one pos done
                    self.signals.progressBar.emit(1)
        else:
            if self.second_channel_name is not None:
                img_data = self.model.to_rgb_stack(img_data, second_ch_data)

            lab_stack = segm_model_segment(
                self.model, img_data, self.model_kwargs, frame_i=0, 
                preproc_recipe=self.preproc_recipe, 
                posData=posData
            )
            self.signals.progressBar.emit(1)
            # lab_stack = smooth_contours(lab_stack, radius=2)

        posData.saveSamEmbeddings(logger_func=self.logger_func)
        
        if self.do_postprocess:
            if posData.SizeT > 1:
                pbar = tqdm(total=len(lab_stack), ncols=100)
                for t, lab in enumerate(lab_stack):
                    lab_cleaned = post_process_segm(
                        lab, **self.standard_postrocess_kwargs
                    )
                    lab_stack[t] = lab_cleaned
                    if self.custom_postproc_features:
                        lab_filtered = features.custom_post_process_segm(
                            posData, self.custom_postproc_grouped_features, 
                            lab_cleaned, postprocess_img, t, posData.filename, 
                            posData.user_ch_name, self.custom_postproc_features
                        )
                        lab_stack[t] = lab_filtered
                    pbar.update()
                pbar.close()
            else:
                lab_stack = post_process_segm(
                    lab_stack, **self.standard_postrocess_kwargs
                )
                if self.custom_postproc_features:
                    lab_stack = features.custom_post_process_segm(
                        posData, self.custom_postproc_grouped_features, 
                        lab_stack, postprocess_img, 0, posData.filename, 
                        posData.user_ch_name, self.custom_postproc_features
                    )

        if posData.SizeT > 1 and self.do_tracking:     
            self.logger_func(f'\nTracking with {self.tracker_name} tracker...')       
            if self.do_save:
                # Since tracker could raise errors we save the not-tracked 
                # version which will eventually be overwritten
                self.logger_func(f'Saving NON-tracked masks of {posData.relPath}...')
                np.savez_compressed(posData.segm_npz_path, lab_stack)

            self.signals.innerPbar_available = self.innerPbar_available
            self.track_params['signals'] = self.signals
            if self.image_channel_tracker is not None:
                # Check if loading the image for the tracker is required
                if 'image' in self.track_params:
                    trackerInputImage = self.track_params.pop('image')
                else:
                    self.logger_func(
                        'Loading image data of channel '
                        f'"{self.image_channel_tracker}"')
                    trackerInputImage = posData.loadChannelData(
                        self.image_channel_tracker)
                try:
                    tracked_stack = self.tracker.track(
                        lab_stack, trackerInputImage, **self.track_params
                    )
                except Exception as e:
                    # Check if user accidentally passed the image even if 
                    # the tracker doesn't need it
                    self.logger_func(
                        'Image data is not required by this tracker, ignoring it...'
                    )
                    tracked_stack = self.tracker.track(
                        lab_stack, **self.track_params
                    )
            else:
                tracked_stack = self.tracker.track(
                    lab_stack, **self.track_params
                )
            posData.fromTrackerToAcdcDf(self.tracker, tracked_stack, save=True)
        else:
            tracked_stack = lab_stack
            try:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(stop_frame_n)
                else:
                    self.signals.progressBar.emit(stop_frame_n)
            except AttributeError:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(1)
                else:
                    self.signals.progressBar.emit(1)

        if isROIactive:
            self.logger_func(f'Padding with zeros {pad_info}...')
            tracked_stack = np.pad(tracked_stack, pad_info, mode='constant')

        if self.do_save:
            self.logger_func(f'Saving {posData.relPath}...')
            np.savez_compressed(posData.segm_npz_path, tracked_stack)

        t_end = time.perf_counter()

        self.logger_func(f'\n{posData.relPath} done.')
        self.signals.finished.emit(t_end-t0)

def filter_segm_objs_from_table_coords(lab, df):
    cols = []
    if lab.ndim == 3:
        cols = ['z']
    cols.extend(('y', 'x'))
    coords = df[cols].values.T
    IDs_to_keep = lab[tuple(coords)]
    mask_to_keep = np.isin(lab, IDs_to_keep)
    filtered_lab = lab.copy()
    filtered_lab[~mask_to_keep] = 0
    return filtered_lab

def tracker_track(
        segm_data, tracker, track_params, intensity_img=None,
        logger_func=print
    ):
    if intensity_img is not None:
        args_to_try = (tuple(), (intensity_img,))
    else:
        args_to_try = (tuple(),)

    kwargs_to_remove = ('', 'signals')
    for args, kwarg_to_remove in product(args_to_try, kwargs_to_remove):
        kwargs = track_params.copy()
        kwargs.pop(kwarg_to_remove, None)
        try:
            tracked_video = tracker.track(segm_data, *args, **kwargs)
            return tracked_video
        except Exception as err:
            is_unexpected_kwarg = (str(err).find(
                "got an unexpected keyword argument 'signals'"
            ) != -1)
            is_missing_arg = (str(err).find(
                "missing 1 required positional argument:"
            ) != -1)
            if is_unexpected_kwarg or is_missing_arg:
                continue
            else:
                raise err

def _relabel_sequential(segm_data):
    relabelled, fw, inv = skimage.segmentation.relabel_sequential(segm_data)
    newIDs = list(inv.in_values)
    oldIDs = list(inv.out_values)
    newIDs.append(-1)
    oldIDs.append(-1)
    return relabelled, oldIDs, newIDs

def _relabel_sequential_timelapse(segm_data):
    """Relabel IDs sequentially frame-by-frame

    Parameters
    ----------
    segm_data : (T, Z, Y, X) or (T, Y, X) numpy.ndarray of ints
        Timelapse segmentation data to relabel.

    Returns
    -------
    3-tuple of (numpy.ndarray, list, list)
        First element is the relabelled segmentation data. 
        Second element is the list of the old IDs.
        Third element is the list of the new IDs.
    """    
    mapper_old_to_new_IDs = {-1: -1}
    relabelled = np.zeros_like(segm_data)
    lastID = 0
    pbar = tqdm(total=len(segm_data), ncols=100, unit=' frame')
    for frame_i, lab in enumerate(segm_data):
        if frame_i == 0:
            relab, oldIDs_i, newIDs_i = _relabel_sequential(lab)
            mapper_old_to_new_IDs = dict(zip(oldIDs_i, newIDs_i))
            lastID = max(newIDs_i)
            relabelled[frame_i] = relab
            continue
        
        rp = skimage.measure.regionprops(lab)
        for obj in rp:
            newID = mapper_old_to_new_IDs.get(obj.label)
            if newID is not None:
                # ID was already mapped in prev iter --> use it
                relabelled[frame_i][obj.slice][obj.image] = newID
            else:
                newID = lastID + 1
                mapper_old_to_new_IDs[obj.label] = newID
                lastID += 1
            relabelled[frame_i][obj.slice][obj.image] = newID
        pbar.update()
    pbar.close()
    oldIDs = list(mapper_old_to_new_IDs.keys())
    newIDs = list(mapper_old_to_new_IDs.values())
    return relabelled, oldIDs, newIDs

def relabel_sequential(segm_data, is_timelapse=False):
    if is_timelapse:
        relabelled, oldIDs, newIDs = _relabel_sequential_timelapse(segm_data)
    else:
        relabelled, oldIDs, newIDs = _relabel_sequential(segm_data)
    return relabelled, oldIDs, newIDs

class CcaIntegrityChecker:
    def __init__(self, cca_df, lab, lab_IDs):
        self.lab = lab
        self.lab_IDs = lab_IDs
        self.cca_df = cca_df
        self.cca_df_S = cca_df[cca_df['cell_cycle_stage'] == 'S']
        self.cca_df_G1 = cca_df[cca_df['cell_cycle_stage'] == 'G1']

    def get_num_mothers_and_buds_in_S(self):
        cca_df_S = self.cca_df_S
        cca_df_S_buds = cca_df_S[cca_df_S['relationship'] == 'bud']
        cca_df_S_mothers = cca_df_S[cca_df_S['relationship'] == 'mother']
        num_buds = len(cca_df_S_buds)
        num_mothers = len(cca_df_S_mothers)
        return num_mothers, num_buds
    
    def get_mother_IDs_with_multiple_buds(self):
        cca_df_S = self.cca_df_S
        cca_df_S_buds = cca_df_S[cca_df_S['relationship'] == 'bud']
        mothers_of_buds = cca_df_S_buds['relative_ID']
        mother_IDs_with_multiple_buds = (
            mothers_of_buds[mothers_of_buds.duplicated()]
        )
        return mother_IDs_with_multiple_buds.values
    
    def get_IDs_cycles_without_G1(self, global_cca_df):
        global_cca_df_moths_hist_known = (
            global_cca_df[
                (global_cca_df['relationship'] == 'mother') 
                & (global_cca_df['is_history_known'] > 0)
            ]
        )
        grouped_cycles = global_cca_df_moths_hist_known.reset_index().groupby(
            ['Cell_ID', 'generation_num']
        )
        G1_not_present_mask = (
            grouped_cycles['cell_cycle_stage']
            .agg(lambda x: ~x.eq('G1').any())
        )
        IDs_cycles_without_G1 = (
            G1_not_present_mask[G1_not_present_mask].index.to_list()
        )
        return IDs_cycles_without_G1

    def get_IDs_gen_num_will_divide_wrong(self, global_cca_df):
        IDs_will_divide_wrong = cca_functions.get_IDs_gen_num_will_divide_wrong(
            global_cca_df
        )
        return IDs_will_divide_wrong
    
    def get_bud_IDs_gen_num_nonzero(self):
        cca_df_S = self.cca_df_S
        cca_df_S_buds = cca_df_S[cca_df_S['relationship'] == 'bud']
        bud_IDs_gen_num_nonzero = (
            cca_df_S_buds[cca_df_S_buds['generation_num'] != 0]
            .index.to_list()
        )
        return bud_IDs_gen_num_nonzero
    
    def get_moth_IDs_gen_num_non_greater_one(self):
        cca_df_S = self.cca_df_S
        cca_df_S_moths = cca_df_S[cca_df_S['relationship'] == 'mother']
        moth_IDs_gen_num_non_greater_one = (
            cca_df_S_moths[cca_df_S_moths['generation_num'] < 1]
            .index.to_list()
        )
        return moth_IDs_gen_num_non_greater_one
    
    def get_buds_G1(self):
        cca_df_S = self.cca_df_S
        cca_df_S_buds = cca_df_S[cca_df_S['relationship'] == 'bud']
        buds_G1 = (
            cca_df_S_buds[cca_df_S_buds['cell_cycle_stage'] == 'G1']
            .index.to_list()
        )
        return buds_G1
    
    def get_cell_S_rel_ID_zero(self):
        cca_df_S = self.cca_df_S
        cell_S_rel_ID_zero = (
            cca_df_S[cca_df_S['relative_ID'] < 1]
            .index.to_list()
        )
        return cell_S_rel_ID_zero
    
    def get_ID_rel_ID_mismatches(self):
        ID_rel_ID_mismatches = []
        for row in self.cca_df_S.itertuples():
            ID = row.Index
            relID = row.relative_ID
            relID_of_relID = self.cca_df.at[relID, 'relative_ID']
            
            if relID_of_relID != ID:
                ID_rel_ID_mismatches.append((ID, relID, relID_of_relID))
        
        return ID_rel_ID_mismatches

    def get_lonely_cells_in_S(self):
        lonely_cells_in_S = []
        for row in self.cca_df_S.itertuples():
            ID = row.Index            
            if row.relative_ID in self.lab_IDs:
                continue
            
            # ID is in S but its relative_ID does not exist in lab
            lonely_cells_in_S.append(ID)
        
        return lonely_cells_in_S

def cellpose_v3_run_denoise(
        image,
        run_params,
        denoise_model=None, 
        init_params=None,
    ):
    if denoise_model is None:
        from cellacdc.models.cellpose_v3 import _denoise
        denoise_model = _denoise.CellposeDenoiseModel(**init_params)
    
    denoised_img = denoise_model.run(image, **run_params)
    return denoised_img

def closest_n_divisible_by_m(n, m) :
    # Find the quotient
    q = int(n / m)
     
    # 1st possible closest number
    n1 = m * q
     
    # 2nd possible closest number
    if((n * m) > 0) :
        n2 = (m * (q + 1)) 
    else :
        n2 = (m * (q - 1))
     
    # if true, then n1 is the required closest number
    if (abs(n - n1) < abs(n - n2)) :
        return n1
     
    # else n2 is the required closest number 
    return n2