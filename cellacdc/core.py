import traceback
import inspect
from typing import List, Dict, Any, Iterable, Tuple, Callable, Union, Literal
import os
import time
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, as_completed
)
from functools import partial
from importlib import import_module
import numpy as np
import math
import cv2
import re
from collections import defaultdict
import skimage.exposure
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
import pathlib
from natsort import natsorted
import sympy as sp
import pickle

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
from . import io
from . import measurements
from . import favourite_func_metrics_csv_path
from . import default_index_cols
from . import regionprops

from ._types import (
    ChannelsDict
)

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
    if dist.size == 0:
        if return_coords:
            return 0, 0, 0
        else:
            return 0
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

def nearest_points_objects(objs_arr: np.ndarray, other_obj: np.ndarray):
    """Find the nearest points between all objects in objs_arr and other_obj

    Parameters
    ----------
    objs_arr : (N, P, 2) np.ndarray of floats 
        Array with N pages (one for each object), P rows (number of points) 
        and 2 columns for y, x coordinates
    other_obj : (P1, 2) np.ndarray
        Array with P1 rows (number of points) and 2 columns for y, x coordinates

    Returns
    -------
    (N,) np.ndarray
        Array with N elements where the ith element is the minimum distance 
        between object objs_arr[i] and other_obj
    """    
    # diff[l, k, i] = objs_arr[l][k] - other_obj[i]
    diff = objs_arr[:, :, np.newaxis] - other_obj
    
    # dist[l, i, j] = math.dist(objs_arr[l][i], other_obj[j])
    dist = np.linalg.norm(diff, axis=3)
    
    # min_dist[l] = min_dist(objs_arr[l], other_obj)
    min_dist = np.nanmin(dist, axis=(1, 2))
    
    return min_dist

def nearest_point_2Dyx(points, all_others):
    """
    Given 2D array of [y, x] coordinates points and all_others return the
    [y, x] coordinates of the two points (one from points and one from all_others)
    that have the absolute minimum distance
    """
    # Compute 3D array where each ith row of each kth page is the element-wise
    # difference between kth row of points and ith row in all_others array.
    # (i.e. diff[k,i] = points[k] - all_others[i])
    diff = points[:, np.newaxis] - all_others
    # Compute 2D array of distances where
    # dist[i, j] = euclidean_dist(points[i],all_others[j])
    dist = np.linalg.norm(diff, axis=2)
    # Compute i, j indexes of the absolute minimum distance
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    nearest_point = all_others[j]
    point = points[i]
    min_dist = np.min(dist)
    return min_dist, nearest_point

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

        try:
            lab[obj.slice][obj.image] = newIDs[idx]
        except OverflowError:
            # it should be uint32 already but sometimes it was not
            lab = lab.astype(np.uint32) 
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
                acdc_df_frame_i = cells_acdc_df.loc[[frame_i]].copy()
           
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
        cells_segm_data, 
        subobj_segm_data, 
        IoAthresh, 
        how: Literal[
            'delete_sub', 'delete_cells', 'delete_both', 'only_track'
        ]='delete_sub', 
        SizeT: int | None =None, 
        sigProgress=None,
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

        sigProgress (qtpy.QtCore.Signal, optional): If provided it will emit 
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
            if intersect_IDs[0] == 0:
                intersect_IDs = intersect_IDs[1:]
                intersections = intersections[1:]
            
            if len(intersect_IDs) == 0:
                untracked_sub_objs_frame_i.add(sub_obj.label)
                continue
            
            argmax = intersections.argmax()
            intersect_ID = intersect_IDs[argmax]
            intersection = intersections[argmax]
            
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
    if obj is not None and obj_image is None and not local:
        if all_external and not all:
            obj_image = np.ascontiguousarray(obj.image, dtype=np.uint8)
            contours, _ = cv2.findContours(
                obj_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            min_y, min_x, _, _ = obj.bbox
            return [np.squeeze(cont, axis=1)+[min_x, min_y] for cont in contours]
        if all and hasattr(obj, 'contour_all'):
            return obj.contour_all
        if only_longest_contour and hasattr(obj, 'contour'):
            return obj.contour

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
            if isinstance(rp, regionprops.acdcRegionprops):
                centroid = rp.get_centroid(obj.label, exact=True)
            else:
                centroid = obj.centroid
            xx_centroid.append(int(centroid[1]))
            yy_centroid.append(int(centroid[0]))
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
        acdc_df = load.pd_bool_and_float_to_int_to_str(acdc_df, colsToCastInt=[]).reset_index()
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
                was_bud = self.acdc_df.loc[idx, 'relationship'] == 'bud'
                if was_bud:
                    # This is a bud of the first frame where the algorithm 
                    # thinks is a new tree --> correct
                    parent_ID = self.acdc_df.loc[idx, 'relative_ID']
                    try:
                        self.branch_start_gen_num[ID] = (
                            self.branch_start_gen_num[parent_ID] + 2
                        )
                    except KeyError as e:
                        gen_num_parentID_tree = 2
                        self.branch_start_gen_num[ID] = gen_num_parentID_tree
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
            IDs = np.sort(df_i.index.array)
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

def preprocess_multi_pos_from_recipe(
        image_data: Iterable[np.ndarray], 
        recipe: List[Dict[str, Any]]
    ):
    pbar = tqdm(total=len(image_data), unit='Position', ncols=100)
    preprocessed_data = []
    for pos_i, image in enumerate(image_data):
        preprocessed_image = preprocess_zstack_from_recipe(
            image, recipe, pbar_pos=1
        )
        preprocessed_data.append(preprocessed_image)
        pbar.update()
    pbar.close()
    return preprocessed_data

def preprocess_video_from_recipe(
        image, recipe: List[Dict[str, Any]], pbar_pos=0
    ):
    if image.ndim < 3:
        raise TypeError(
            'Only 3D or 4D videos allowed. '
            f'Input image has {image.ndim} dimensions!'
        )

    preprocessed_image = image
    for step in recipe:
        method = step['method']
        func = PREPROCESS_MAPPER[method]['function']
        kwargs = step['kwargs']
        argspecs = inspect.getfullargspec(func)
        is_func_time_capable = False
        is_func_zstack_capable = False
        for arg in argspecs.args:
            if arg == 'apply_to_all_frames':
                is_func_time_capable = True
            elif arg == 'apply_to_all_zslices':
                is_func_zstack_capable = True
        
        if is_func_time_capable and is_func_zstack_capable:
            kwargs["apply_to_all_zslices"] = True
            kwargs["apply_to_all_frames"] = True
            preprocessed_image = func(
                preprocessed_image, 
                **kwargs
            )
        else:
            pbar = tqdm(
                total=len(preprocessed_image), unit='frame', ncols=100, 
                position=pbar_pos
            )
            for frame_i, frame_img in enumerate(preprocessed_image):
                if frame_img.ndim == 3:
                    preprocessed_img = preprocess_zstack_from_recipe(
                        frame_img, (step,), pbar_pos=pbar_pos+1
                    )
                    if preprocessed_img.dtype != preprocessed_image.dtype:
                        preprocessed_image = (
                            preprocessed_image.astype(preprocessed_img.dtype)
                        )
                    preprocessed_image[frame_i] = preprocessed_img
                else:
                    preprocessed_img = preprocess_image_from_recipe(
                        frame_img, (step,)
                    )
                    if preprocessed_img.dtype != preprocessed_image.dtype:
                        preprocessed_image = (
                            preprocessed_image.astype(preprocessed_img.dtype)
                        )
                    preprocessed_image[frame_i] = preprocessed_img
                pbar.update()
            pbar.close()
    
    return preprocessed_image
    
def preprocess_zstack_from_recipe(
        image, recipe: List[Dict[str, Any]], pbar_pos=0
    ):
    if image.ndim != 3:
        raise TypeError(
            'Only 3D z-stack images allowed. '
            f'Input image has {image.ndim} dimensions!'
        )
        
    preprocessed_image = image
    for step in recipe:
        method = step['method']
        func = PREPROCESS_MAPPER[method]['function']
        kwargs = step['kwargs']
        argspecs = inspect.getfullargspec(func)
        is_func_zstack_capable = False
        for arg in argspecs.args:
            if arg == 'apply_to_all_zslices':
                is_func_zstack_capable = True
                break
        
        if is_func_zstack_capable:
            kwargs['apply_to_all_zslices'] = True
            preprocessed_image = func(
                preprocessed_image, **kwargs
            )
        else:
            pbar = tqdm(
                total=len(preprocessed_image), unit='z-slice', ncols=100, 
                position=pbar_pos
            )
            for z_slice, img in enumerate(preprocessed_image):
                preprocessed_img = func(img, **kwargs)
                if preprocessed_img.dtype != preprocessed_image.dtype:
                    preprocessed_image = (
                        preprocessed_image.astype(preprocessed_img.dtype)
                    )
                preprocessed_image[z_slice] = preprocessed_img
                pbar.update()
            pbar.close()
    
    return preprocessed_image

all_kwargs_to_pop = (
    ('apply_to_all_zslices',), 
    ('apply_to_all_frames',), 
    ('apply_to_all_frames', 'apply_to_all_zslices'), 
)
def preprocess_image_from_recipe(image, recipe: List[Dict[str, Any]]):
    preprocessed_image = image
    for step in recipe:
        method = step['method']
        func = PREPROCESS_MAPPER[method]['function']
        kwargs = step['kwargs']
        for kwargs_to_pop in all_kwargs_to_pop:
            test_kwargs = kwargs.copy()
            try:
                preprocessed_image = func(preprocessed_image, **test_kwargs)
                break
            except TypeError as err:
                if not 'unexpected keyword argument' in str(err):
                    raise err
            
            for kwarg_to_pop in kwargs_to_pop:
                test_kwargs.pop(kwarg_to_pop, None)
        
    return preprocessed_image

def pop_signals_kwarg_if_not_needed(func, kwargs):
    args = inspect.getfullargspec(func).args
    if 'signals' in args:
        return kwargs
    
    kwargs.pop('signals', None)
    return kwargs

def segm_model_segment(
        model, image, model_kwargs, frame_i=None, preproc_recipe=None, 
        is_timelapse_model_and_data=False, posData=None, start_z_slice=0,
    ):
    if preproc_recipe is not None:
        if is_timelapse_model_and_data:
            filtered_image = np.zeros(image.shape)
            for i, img in enumerate(image):
                img = preprocess_image_from_recipe(img, preproc_recipe)
                filtered_image[i] = img
            image = filtered_image # .astype(image.dtype)
        else:
            image = preprocess_image_from_recipe(image, preproc_recipe)
    
    if is_timelapse_model_and_data:
        model_kwargs = pop_signals_kwarg_if_not_needed(
            model.segment3DT, model_kwargs
        )
        segm_data = model.segment3DT(image, **model_kwargs)
        return segm_data             
    
    model_kwargs = pop_signals_kwarg_if_not_needed(
        model.segment, model_kwargs
    )

    # Some models have `start_z_slice` kwarg
    try:
        lab = model.segment(
            image, 
            frame_i=frame_i, 
            posData=posData, 
            start_z_slice=start_z_slice, 
            **model_kwargs
        )
        return lab
    except TypeError as err:
        if str(err).find('unexpected keyword argument') == -1:
            # Raise error since it's not about the missing posData kwarg
            raise err
    
    # Some models have posData as kwarg and frame_i as second arg
    try:
        lab = model.segment(
            image, 
            frame_i=frame_i, 
            posData=posData, 
            **model_kwargs
        )
        return lab
    except TypeError as err:
        if str(err).find('unexpected keyword argument') == -1:
            # Raise error since it's not about the missing posData kwarg
            raise err
    
    # Some models have frame_i as second arg
    try:
        lab = model.segment(
            image, 
            frame_i=frame_i, 
            **model_kwargs
        )
        return lab
    except TypeError as err:
        pass
    
    lab = model.segment(image, **model_kwargs)
    return lab

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
            
            if ID not in self.lab_IDs:
                # Mother-bud pair gone entirely
                continue
            
            # ID is in S but its relative_ID does not exist in lab
            lonely_cells_in_S.append(ID)
        
        return lonely_cells_in_S

def cellpose_v3_run_denoise(
        image,
        run_params,
        denoise_model=None, 
        init_params=None,
        timelapse=False,
        isZstack=False
    ):
    if denoise_model is None:
        from cellacdc.models.cellpose_v3 import _denoise
        denoise_model = _denoise.CellposeDenoiseModel(**init_params)
    
    denoised_img = denoise_model.run(image, timelapse=timelapse,isZstack=isZstack, **run_params)# may have to give rgb stuff too!
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

def fucci_pipeline_executor_map(input, **filter_kwargs):
    frame_i, (ch1_img, ch2_img) = input
    
    ch1_img = skimage.exposure.rescale_intensity(
        ch1_img, out_range=(0, 0.5)
    )
    ch2_img = skimage.exposure.rescale_intensity(
        ch2_img, out_range=(0, 0.5)
    )
    
    sum_img = ch1_img + ch2_img
    
    processed_img = preprocess.fucci_filter(sum_img, **filter_kwargs)
    
    return frame_i, processed_img

def preprocess_exceutor_map(
        input: Tuple[int, np.ndarray],
        recipe: List[Dict[str, Any]]=None,
    ):
    if recipe is None:
        return input
    
    frame_i, image = input
    if image.ndim == 3:
        preprocessed_image = preprocess_zstack_from_recipe(image, recipe)
    else:
        preprocessed_image = preprocess_image_from_recipe(image, recipe)
    
    return frame_i, preprocessed_image

def preprocess_image_from_recipe_multithread(
        image: np.ndarray, 
        recipe: List[Dict[str, Any]], 
        n_threads: int=None
    ):
    preprocessed_image = image
    for step in recipe:
        method = step['method']
        func = PREPROCESS_MAPPER[method]['function']
        kwargs = step['kwargs']
        argspecs = inspect.getfullargspec(func)
        is_func_time_capable = False
        for arg in argspecs.args:
            if arg == 'apply_to_all_frames':
                is_func_time_capable = True
                break

        if is_func_time_capable:
            preprocessed_image = preprocess_video_from_recipe(
                preprocessed_image, (step,)
            )
        else:
            num_frames = len(preprocessed_image)
            pbar = tqdm(total=num_frames, ncols=100)
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                iterable = enumerate(preprocessed_image)
                func = partial(
                    preprocess_exceutor_map,
                    recipe=(step,)
                )
                futures = {executor.submit(func, arg) for arg in iterable}
                for future in as_completed(futures):
                    try:
                        frame_i, processed_img = future.result()
                        preprocessed_image[frame_i] = processed_img
                        pbar.update()
                    except Exception as e:
                        printl(e)
                        raise e
            pbar.close()
    
    return preprocessed_image

def combine_channels_multithread(
        steps: Dict[str, Dict[str, Any]],
        formula: str,
        images_paths: List[str],
        keep_input_data_type: bool,
        save_filepaths: List[str]=None,
        n_threads: int=None,
        signals=None,
        logger_func: Callable=None,
        output_as_segm: bool = False,
    ):
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        if signals:
            signals.initProgressBar.emit(len(images_paths))
        else:
            pbar = tqdm(
                total=len(images_paths), ncols=100, desc='Combining channels'
            )
        
        func = partial(
            combine_channels_executor_map,
            keep_input_data_type=keep_input_data_type,
            return_img=False,
            logger_func=logger_func,
            output_as_segm=output_as_segm,
            formula=formula,
            steps=steps,
        )
        iterable = zip(images_paths, save_filepaths)
        result = executor.map(func, iterable)
        for res in result:
            if signals:
                signals.progressBar.emit(1)
            else:
                pbar.update()

def combine_channels_multithread_return_imgs(
        steps: Dict[str, Dict[str, Any]],
        data: list['load.loadData'],
        keep_input_data_type: bool,
        keys: List[Tuple[Union[int, None], Union[int, None], Union[int, None]]],
        n_threads: int=None,
        signals=None,
        logger_func: Callable=None,
        output_as_segm: bool = False,
        formula: str = None,
    ):


    total = len(keys)
    
    output_imgs = [None] * total
    keys_out = [0] * total
    res_i = 0
    txts = set()

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        if signals:
            signals.initProgressBar.emit(total)
        else:
            pbar = tqdm(total=len(total), ncols=100, desc='Combining channels')
        func = partial(
            combine_channels_executor_map_return_img,
            data=data,
            keep_input_data_type=keep_input_data_type,
            return_img=True,
            logger_func=logger_func,
            output_as_segm=output_as_segm,
            formula=formula,
            steps=steps,
        )
        iterable = keys
        result = executor.map(func, iterable)
        for res in result:
            output_img, key, txt = res
            output_imgs[res_i] = output_img
            keys_out[res_i] = key
            res_i += 1
            txts.add(txt)

            if signals:
                signals.progressBar.emit(1)
            else:
                pbar.update()

    if not logger_func:
        for txt in txts:
            print(txt)
    else:
        for txt in txts:
            try:
                logger_func(txt)
            except Exception as err:
                print(txt)

    return output_imgs, keys_out

def combine_channels_executor_map(args, **kwargs):
    images_path, save_filepath = args
    kwargs['save_filepath'] = save_filepath
    kwargs['images_path'] = images_path
    return combine_channels_func(**kwargs)

def combine_channels_executor_map_return_img(args, **kwargs):
    key = args
    kwargs['key'] = key
    return combine_channels_func(**kwargs)

def _combine_channels_multiplier_apply(binarize, input_img):
    if binarize == 'binarize':
        input_img = (input_img > 0)
    elif binarize == 'inverse binarize':
        input_img = ~(input_img > 0)
    return input_img

def _get_img_from_data_key(data, key, num_dim, seg=False):
    n_dim_data = data.ndim - 1 # - 1 dim for x y
    n_dim_key = len(key)
    if seg and n_dim_key == n_dim_data + 1:
        # here a 2D segmentation is used for 3D image
        return data[key[1]]
    if num_dim == 3: # t x y
        return data[key[1]]
    elif num_dim == 4: # t z x y
        return data[key[1]][key[2]]
    elif num_dim == 2: # z x y, but t is always there
        return data[0]
    else:
        raise ValueError(
            f'Invalid number of dimensions in img_data. {num_dim}'
        )
        
def _log_printl_fallback(txt, logger_func):
    if logger_func is not None:
        try:
            logger_func(txt)
        except:
            printl(txt)
            pass
    else:
        printl(txt)
        
def _add_missing_dims(segm, target_shape, use_broadcast=False):
    """
    Expand segmentation by replicating existing data along missing dims.

    Parameters
    ----------
    segm : np.ndarray
    target_shape : tuple
        Desired output shape (e.g. (Z, Y, X) or (T, Z, Y, X))
    use_broadcast : bool
        If True, uses zero-copy broadcast (read-only view)

    Returns
    -------
    segm_expanded : np.ndarray
    text : str
    """
    text = ''

    if segm.shape == target_shape:
        return segm, text

    # 2D -> 3D (Y,X -> Z,Y,X)
    if segm.ndim == 2 and len(target_shape) == 3:
        text = (
            'The segmentation mask is 2D but the image data is 3D. '
            'Replicating mask across Z.'
        )
        y, x = segm.shape
        z = target_shape[0]

        segm = segm[np.newaxis, ...]  # (1, Y, X)

        if use_broadcast:
            segm_expanded = np.broadcast_to(segm, (z, y, x))
        else:
            segm_expanded = np.repeat(segm, z, axis=0)

        return segm_expanded, text

    # 3D -> 4D (T,Y,X -> T,Z,Y,X)
    if segm.ndim == 3 and len(target_shape) == 4:
        text = (
            'The segmentation mask is 2Dt but the image data is 3Dt. '
            'Replicating mask across Z.'
        )
        t, y, x = segm.shape
        z = target_shape[1]

        segm = segm[:, np.newaxis, ...]  # (T,1,Y,X)

        if use_broadcast:
            segm_expanded = np.broadcast_to(segm, (t, z, y, x))
        else:
            segm_expanded = np.repeat(segm, z, axis=1)

        return segm_expanded, text

    raise ValueError(
        f'Invalid shape. segm: {segm.shape}, target: {target_shape}'
    )
        
def _verify_shape_ndim(img_data, target_dims, target_shape, is_segm=False):
    def _shape_mismatch_error(indices, img_data, target_shape):
        mismatches = [
            f'  axis {i}: got {img_data.shape[idx]}, expected {target_shape[i]}'
            for idx, i in indices
            if img_data.shape[idx] != target_shape[i]
        ]
        if mismatches:
            raise ValueError(
                f'Shape mismatch:\n' + '\n'.join(mismatches) + '\n'
                f'img shape={img_data.shape}, target shape={target_shape}'
            )

    if img_data.ndim == target_dims:
        # Check all axes directly
        _shape_mismatch_error([(i, i) for i in range(target_dims)], img_data, target_shape)

    elif is_segm and img_data.ndim + 1 == target_dims:
        if target_dims == 3:
            # 2D segm on 3D image: check Y, X (axes 1, 2)
            _shape_mismatch_error([(1, 1), (2, 2)], img_data, target_shape)
        elif target_dims == 4:
            # 2D+t segm (T, Y, X) on 3D+t image (T, Z, Y, X): check T, Y, X
            _shape_mismatch_error([(0, 0), (1, 2), (2, 3)], img_data, target_shape)
        else:
            raise ValueError(
                f'Invalid segmentation mask dimensions: '
                f'got {img_data.ndim}D mask for {target_dims}D target.'
            )
    else:
        raise ValueError(
            f'Invalid image dimensions: '
            f'got {img_data.ndim}D data, expected {target_dims}D.'
        )

def _update_target_shape_target_dims(target_dims, target_shape, img_data):
    if target_dims < img_data.ndim:
        target_dims_old = target_dims
        target_dims = img_data.ndim

    img_shape = list(img_data.shape)
    if img_data.ndim == 2:
        img_shape = [0, 0] + img_shape
    elif img_data.ndim == 3:
        img_shape = img_shape[:1] + [0] + img_shape[1:]
    
    try:
        for i in range(len(target_shape)):
            if target_shape[i] < img_shape[i]:
                target_shape[i] = img_shape[i]
    except Exception as err:
        printl(target_shape, img_shape)
        raise err

    return target_dims, target_shape
        

def combine_channels_func(
        steps: Dict[str, Dict[str, Any]],
        keep_input_data_type: bool,
        save_filepath: str=None,
        return_img: bool=False,
        logger_func: Callable=None,
        images_path: str = None,
        key: str = None,
        data = None,
        output_as_segm: bool = False,
        formula: str = None,
    ):# -> tuple[Any | ndarray, str, str | Any] | None:
    if not save_filepath and not return_img:
        raise ValueError('Either save_filepath must be provided or return_img must be true')
    
    provided = sum(x is not None for x in (data, key))
    provided += return_img
    if provided not in (0, 3):
        raise ValueError('return_img, data, and key must all be provided together or not at all')
    
    fluo_ch_data_list = dict()
    segm_ch_data_list = dict()
    
    channel_names = [step['channel'] for step in steps.values()]
    channel_keys = steps.keys()
    segm_channels, fluo_channel_names, current_segm = myutils.separate_fluo_segment_channels(channel_names)
    original_dtype = None
    
    target_dims = 0
    target_shape = [0, 0, 0, 0]

    if data is None:
        for channel in fluo_channel_names:
            ch_filepath = load.get_filepath_from_endname(
                images_path, channel
            )
            ch_image_data = load.load_image_file(ch_filepath)
            if original_dtype is None:
                original_dtype = ch_image_data.dtype
            
            ch_image_data = myutils.img_to_float(ch_image_data)
            target_dims, target_shape = _update_target_shape_target_dims(
                 target_dims, target_shape, ch_image_data
            )
            fluo_ch_data_list[channel] = ch_image_data
        for channel in segm_channels:
            ch_filepath = load.get_filepath_from_endname(
                images_path, channel
            )
            ch_image_data = load.load_image_file(ch_filepath)
            if original_dtype is None:
                original_dtype = ch_image_data.dtype

            ch_image_data = ch_image_data.astype(np.uint32)
            target_dims, target_shape = _update_target_shape_target_dims(
                 target_dims, target_shape, ch_image_data
            )
            segm_ch_data_list[channel] = ch_image_data
    else:
        posData = data[key[0]]
        n_dim = 4
        if posData.SizeZ == 1:
            n_dim -= 1
        # if posData.SizeT == 1: # actually t is always there, we only need to subtract for curr. segm
        #     n_dim -= 1
        is_2D_segm_on_3D = posData.SizeZ != 1 and posData.allData_li[0]['labels'].ndim == 2
        fluo_data_dict = posData.fluo_data_dict
        segm_data_dict = posData.ol_labels_data
        imgs_path = posData.images_path

        for channel in fluo_channel_names:
            channel_path = load.get_filename_from_channel(
                imgs_path, channel, basename=posData.basename
            )
            channel_full_name = pathlib.Path(channel_path).stem
            # remove the file extension
            
            channel_img_data = _get_img_from_data_key(fluo_data_dict[channel_full_name], key, n_dim)
            if original_dtype is None:
                original_dtype = channel_img_data.dtype
            channel_img_data_float = myutils.img_to_float(channel_img_data)
            target_dims, target_shape = _update_target_shape_target_dims(
                target_dims, target_shape, channel_img_data_float
            )
            fluo_ch_data_list[channel] = channel_img_data_float
        for channel in segm_channels:
            channel_img_data = _get_img_from_data_key(segm_data_dict[channel], key, n_dim, seg=True)
            if original_dtype is None:
                original_dtype = channel_img_data.dtype
            channel_img_data_int = channel_img_data.astype(np.uint32)
            target_dims, target_shape = _update_target_shape_target_dims(
                target_dims, target_shape, channel_img_data_int
            )
            segm_ch_data_list[channel] = channel_img_data_int
        if current_segm: # here we dont need to get/appply target dim, as we already ignore z slice key if segm is 2D and image 3D (time is always treated differently!)
            if posData.SizeT == 1: # actually t is always there, we only need to subtract for curr. segm
                n_dim -= 1
            if posData.frame_i != key[1]:
                if n_dim == 4 and not is_2D_segm_on_3D:
                    channel_img_data = posData.allData_li[key[1]]['labels'][key[2]]
                elif n_dim == 4 and is_2D_segm_on_3D:
                    channel_img_data = posData.allData_li[key[1]]['labels']
                elif n_dim == 3 and posData.SizeZ == 1:
                    channel_img_data = posData.allData_li[key[1]]['labels']
                elif n_dim == 3 and posData.SizeT == 1 and not is_2D_segm_on_3D:
                    channel_img_data = posData.allData_li[0]['labels'][key[2]]
                elif n_dim == 3 and posData.SizeT == 1 and is_2D_segm_on_3D:
                    channel_img_data = posData.allData_li[0]['labels']
                else:
                    channel_img_data = posData.allData_li[0]['labels']
            else:
                if n_dim == 4 and not is_2D_segm_on_3D:
                    channel_img_data = posData.lab[key[2]]
                elif n_dim == 4 and is_2D_segm_on_3D:
                    channel_img_data = posData.lab
                elif n_dim == 3 and posData.SizeZ != 1 and not is_2D_segm_on_3D:
                    channel_img_data = posData.lab[key[2]]
                else:
                    channel_img_data = posData.lab
            if original_dtype is None:
                original_dtype = channel_img_data.dtype
            channel_img_data_int = channel_img_data.astype(np.uint32)
            target_dims, target_shape = _update_target_shape_target_dims(
                target_dims, target_shape, channel_img_data_int
            )
            segm_ch_data_list['current segm.'] = channel_img_data_int
    
    target_shape_new = []
    for dim in target_shape:
        if dim == 0:
            continue
        target_shape_new.append(dim)
        
    target_shape = tuple(target_shape_new)
    # _log_printl_fallback(f'target shape: {target_shape}', logger_func)
    for i, ch in zip(channel_keys, channel_names):
        if ch in fluo_ch_data_list:
            ch_image_data = fluo_ch_data_list[ch]
            _verify_shape_ndim(ch_image_data, target_dims, target_shape, is_segm=False)

        elif ch in segm_ch_data_list:
            ch_image_data = segm_ch_data_list[ch]
            ch_image_data, text = _add_missing_dims(ch_image_data, target_shape)
            if text:
                _log_printl_fallback(text, logger_func)
            _verify_shape_ndim(ch_image_data, target_dims, target_shape, is_segm=False) # false since we already expanded
            segm_ch_data_list[ch] = ch_image_data
        else:
            raise ValueError(f'Channel "{ch}" not found.')
        if steps[i]['channel'] != ch:
            raise ValueError(f'Channel "{ch}" not found.')        
        steps[i]['channel_data'] = ch_image_data
    
    for i, step_info in steps.items():
        binarize = step_info['binarize']
        steps[i]['channel_data'] = _combine_channels_multiplier_apply(
            binarize, step_info['channel_data']
        )
        norm_min, norm_max = step_info['min_val'], step_info['max_val']
        # use rescale_intensity to normalize
        if norm_min == 0 and norm_max == 1:
            continue # cases where either the fields where disabled/reset or default, where we already normalized
        steps[i]['channel_data'] = skimage.exposure.rescale_intensity(
            steps[i]['channel_data'], 
            out_range=(norm_min, norm_max)
        )
    
    if formula != '':
        input_img_data = {step['name']: step['channel_data'] for step in steps.values()}
        
        symbols = {name: sp.Symbol(name) for name in input_img_data}
        expr = sp.sympify(formula, locals=symbols)

        used_vars = sorted(str(s) for s in expr.free_symbols)
        func = sp.lambdify(
            [symbols[v] for v in used_vars],  # fixed order!
            expr,
            modules="numpy"
        )

        args = [input_img_data[v] for v in used_vars]
        output_img = func(*args)
    else:
        key0 = list(steps.keys())[0]
        output_img = steps[key0]['channel_data']

    if not output_as_segm:
        output_img = skimage.exposure.rescale_intensity(
            output_img, out_range=(0, 1)
        )
    
    txt = ''
    if keep_input_data_type and not output_as_segm:
        try:
            output_img = myutils.convert_to_dtype(
                output_img, original_dtype
            )
            method = 'cellacdc.myutils.convert_to_dtype'
            warning = 'safe'
            prefix = ''
        except Exception as err:
            dtype_info = np.iinfo(original_dtype)
            dtype_max = dtype_info.max
            dtype_min = dtype_info.min
            is_in_bounds = (
                output_img.max() <= dtype_max
                and output_img.min() >= dtype_min
            )
            if is_in_bounds:
                output_img = output_img.astype(original_dtype)
                method = 'output_img.astype(original_dtype)'
                warning = 'safe' # if weights were set correctly'
                prefix = '[WARNING]: '
            else:
                output_img = skimage.exposure.rescale_intensity(
                    output_img, out_range=(dtype_min, dtype_max)
                )
                output_img = output_img.astype(original_dtype)
                method = (
                    'skimage.exposure.rescale_intensity '
                    '-> output_img.astype (original_dtype)'
                )
                warning = '!RESCALING! the image data'
                prefix = '[WARNING]: '

        txt = (
            f'{prefix}Converted output image to {original_dtype} '
            f'using {method}, which is {warning}'
        )

        if not return_img:
            _log_printl_fallback(txt, logger_func)
    elif output_as_segm:
        output_img[output_img<0] = 0
        output_img = output_img.astype(np.uint32)
    
    if return_img:
        return output_img, key, txt
    
    txt = f'Saving combined {"segmentation" if output_as_segm else "image"} to {save_filepath}'
    _log_printl_fallback(txt, logger_func)
    

    io.save_image_data( # handles saving img and segm
        save_filepath, output_img
    )
    return None

def get_selected_channels(steps):
    selected_channel = set()
    for step in steps.values():
        ch = step['channel']
        if ch == 'current segm.':
            continue
        selected_channel.add(ch)
    
    return selected_channel

def split_segm_masks_mother_bud_line(
        cells_segm_data, segm_data_to_split, acdc_df, 
        debug=False
    ):
    acdc_df = acdc_df.set_index(['frame_i', 'Cell_ID'])
    split_segm_away = np.zeros_like(segm_data_to_split)
    split_segm_close = np.zeros_like(segm_data_to_split)
    
    pbar = tqdm(total=len(cells_segm_data), ncols=100, position=1, leave=False)
    for frame_i, lab in enumerate(cells_segm_data):
        rp = skimage.measure.regionprops(lab)
        rp_mapper = {obj.label:obj for obj in rp}
        for obj in rp:
            try:
                ccs = acdc_df.at[(frame_i, obj.label), 'cell_cycle_stage']
            except Exception as err:
                continue
            
            if ccs != 'S':
                continue
            
            try:
                relationship = acdc_df.at[(frame_i, obj.label), 'relationship']
            except Exception as err:
                continue
            
            if relationship == 'bud':
                continue
            
            bud_ID = int(acdc_df.at[(frame_i, obj.label), 'relative_ID'])
            obj_bud = rp_mapper[bud_ID]
            
            moth_ID = obj.label
            yc_m, xc_m = obj.centroid
            yc_b, xc_b = obj_bud.centroid
            
            slope_mb = (yc_b - yc_m)/(xc_b - yc_b)
            if slope_mb != 0:
                slope_perp = -1/slope_mb
                interc_perp = yc_m - xc_m*slope_perp
            else:
                slope_perp = np.inf
                interc_perp = np.nan
            
            ref_p1, ref_p2 = get_split_line_ref_points_img(
                lab, slope_perp, interc_perp, xc_m, yc_m
            )
            
            if debug:
                from cellacdc import _debug
                _debug.split_segm_masks_mother_bud_line(
                    lab, obj, obj_bud, ref_p1, ref_p2
                )
            
            for z, lab_split in enumerate(segm_data_to_split[frame_i]):
                lab_split_yy, lab_split_xx = np.nonzero(lab_split==obj.label)
                if len(lab_split_yy) == 0:
                    continue
                
                query_points = np.column_stack((lab_split_xx, lab_split_yy))
                close_to_bud_mask = classify_points_plane_split_by_line(
                    ref_p1, ref_p2, query_points, (xc_b, yc_b)
                )
                
                split_close_yy = lab_split_yy[close_to_bud_mask]
                split_close_xx = lab_split_xx[close_to_bud_mask]
                
                split_segm_close[frame_i, z, split_close_yy, split_close_xx] = (
                    obj.label
                )
                
                split_away_yy = lab_split_yy[~close_to_bud_mask]
                split_away_xx = lab_split_xx[~close_to_bud_mask]
                
                split_segm_away[frame_i, z, split_away_yy, split_away_xx] = (
                    obj.label
                )
                
        pbar.update()
    pbar.close()  
    
    return split_segm_close, split_segm_away

def classify_points_plane_split_by_line(
        p1, p2, query_points: np.ndarray, relative_to_p
    ):
    """Classify points on plane crossed by a line connecting p1 and p2 relative
    to `relative_to_p` point

    Parameters
    ----------
    p1 : (x, y) of floats
        First point of the line
    p2 : (x, y) of floats
        Second point 
    query_points : (N, 2) np.ndarray
        (x, y) coordinates of the points to classify
    
    References
    ----------
    https://stackoverflow.com/questions/45766534/finding-cross-product-to-find-points-above-below-a-line-in-matplotlib
    """    
    relative_p_arr = np.array([relative_to_p])
    a = np.array(p1)
    b = np.array(p2)
    
    class_relative_p = (np.cross(relative_p_arr-a, b-a) <= 0).astype(int)[0]
    class_query_points = (np.cross(query_points-a, b-a) <= 0).astype(int)
    query_points_mask = class_query_points == class_relative_p
    
    return query_points_mask   
    

def get_split_line_ref_points_img(img, slope, interc, xc, yc):
    Y, X = img.shape
    if slope == np.inf:
        x_ref_0 = xc
        y_ref_0 = 0
        x_ref1 = xc
        y_ref1 = Y
    elif slope == 0:
        x_ref_0 = 0
        y_ref_0 = yc
        x_ref1 = X
        y_ref1 = yc
    else:
        y0 = 0
        x0 = y0 - interc/slope
    
        x1 = X
        y1 = slope*x1 + interc
        
        x2 = 0
        y2 = interc
        
        y3 = Y
        x3 = (y3 - interc)/slope
        
        if x0 < X:
            x_ref_0 = x0
            y_ref_0 = y0
        else:
            x_ref_0 = x1
            y_ref_0 = y1
        
        if x3 > 0:
            x_ref1 = x3
            y_ref1 = y3
        else:
            x_ref1 = x2
            y_ref1 = y2
    
    return (x_ref_0, y_ref_0), (x_ref1, y_ref1)

def _compute_obj_to_all_objs_contour_dist_pairs(
        input, all_objs_contours_arr=None, all_contours=None, pbar=None
    ):
    j, other_obj = input
    other_obj_contours = all_contours[(other_obj.label, 'None', False, False)]
    min_distances_to_other = nearest_points_objects(
        all_objs_contours_arr, other_obj_contours
    )       
    return other_obj.label, min_distances_to_other

def _compute_all_obj_to_obj_contour_dist_pairs(
        all_contours: dict, rp, prev_rp=None, restrict_search=True
    ):
    if prev_rp is not None:
        prev_IDs = set([obj.label for obj in prev_rp])
        new_IDs = set([obj.label for obj in rp if obj.label not in prev_IDs])
        current_rp = [obj for obj in rp if obj.label not in new_IDs]
        other_rp = [obj for obj in rp if obj.label not in prev_IDs]
        num_cols = len(new_IDs)
    else:
        current_rp = rp
        other_rp = rp
        num_cols = len(current_rp)
    
    max_distance = np.inf
    if restrict_search:
        max_distance = 3*np.max([obj.major_axis_length for obj in rp])
    
    calculated_pairs = {}
    num_rows = len(current_rp)
    num_objs = len(rp)
    IDs = [obj.label for obj in rp]
    dist_matrix_df = pd.DataFrame(
        index=IDs, 
        columns=IDs, 
        data=np.full((num_objs, num_objs), np.inf)
    )
    len_longest_contour = np.max(
        [len(contours) for contours in all_contours.values()]
    )
    all_objs_contours_arr = np.full((num_rows, len_longest_contour, 2), np.nan)
    current_rp_mapper = {}
    for o, obj in enumerate(current_rp):
        obj_contours = all_contours[(obj.label, 'None', False, False)]
        all_objs_contours_arr[o, :len(obj_contours)] = obj_contours
        current_rp_mapper[o] = obj
    
    pbar = tqdm(total=num_rows*num_cols, ncols=100, leave=False)
    with ThreadPoolExecutor() as executor:
        iterable = enumerate(other_rp)
        
        func = partial(
            _compute_obj_to_all_objs_contour_dist_pairs, 
            all_objs_contours_arr=all_objs_contours_arr,
            all_contours=all_contours,
            pbar=pbar,
        )
        result = executor.map(func, iterable)
        for j, min_distances_to_other in result:
            for o, min_dist in enumerate(min_distances_to_other):
                i = current_rp_mapper[o].label
                dist_matrix_df.loc[i, j] = min_dist

    return dist_matrix_df

def convexity_defects(img, eps_percent):
    img = img.astype(np.uint8)
    contours, _ = cv2.findContours(img,2,1)
    cnt = contours[0]
    cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
    hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    return cnt, defects

def split_connected_components(lab, rp=None, max_ID=None):  
    if rp is None:
        lab = skimage.measure.regionprops(lab)
    
    if max_ID is None:
        max_ID = max([obj.label for obj in rp], default=1)
        
    split_occured = False
    for obj in rp:
        lab_obj = skimage.measure.label(obj.image)
        rp_lab_obj = skimage.measure.regionprops(lab_obj)
        if len(rp_lab_obj)<=1:
            continue
        lab_obj += max_ID
        _slice = obj.slice # self.getObjSlice(obj.slice)
        _objMask = obj.image # self.getObjImage(obj.image)
        lab[_slice][_objMask] = lab_obj[_objMask]
        split_occured = True
        max_ID += 1
    return split_occured

def split_along_convexity_defects(
        ID, lab, max_ID, max_i=1, eps_percent=0.01, rp=None
    ):
    if rp is not None:
        obj = rp.get_obj_from_ID(ID)
        lab_ID_bool = np.zeros_like(lab[obj.slice], dtype=bool)
        lab_ID_bool[obj.image] = True
    else:
        lab_ID_bool = lab == ID
    # First try separating by labelling
    lab_ID = lab_ID_bool.astype(int)
    rp_ID = skimage.measure.regionprops(lab_ID)
    split_occured = split_connected_components(lab_ID, rp=rp_ID, max_ID=max_ID)
    if split_occured:
        success = True
        if rp is not None:
            lab[obj.slice][obj.image] = lab_ID[obj.image]
        else:
            lab[lab_ID_bool] = lab_ID[lab_ID_bool]
            
        rp_ID = skimage.measure.regionprops(lab_ID)
        separateIDs = [obj.label for obj in rp_ID]
        return lab, success, separateIDs

    cnt, defects = convexity_defects(lab_ID_bool, eps_percent)
    success = False
    if defects is None:
        return lab, success, []

    if len(defects) != 2:
        return lab, success, []

    defects_points = [0]*len(defects)
    for i, defect in enumerate(defects):
        s,e,f,d = defect[0]
        x,y = tuple(cnt[f][0])
        defects_points[i] = (y,x)
    (r0, c0), (r1, c1) = defects_points
    rr, cc, _ = skimage.draw.line_aa(r0, c0, r1, c1)
    sep_bud_img = np.copy(lab_ID_bool)
    sep_bud_img[rr, cc] = False
    
    sep_bud_label = skimage.measure.label(
        sep_bud_img, connectivity=2
    )
    
    rp_sep = skimage.measure.regionprops(sep_bud_label)
    IDs_sep = [obj.label for obj in rp_sep]
    areas = [obj.area for obj in rp_sep]
    curr_ID_bud = IDs_sep[areas.index(min(areas))]
    curr_ID_moth = IDs_sep[areas.index(max(areas))]
    orig_sblab = np.copy(sep_bud_label)
    # sep_bud_label = np.zeros_like(sep_bud_label)
    ID1 = ID
    ID2 = max_ID+max_i
    sep_bud_label[orig_sblab==curr_ID_moth] = ID1
    sep_bud_label[orig_sblab==curr_ID_bud] = ID2
    splittedIDs = [ID1, ID2]
    # sep_bud_label *= (max_ID+max_i)
    temp_sep_bud_lab = sep_bud_label.copy()
    for r, c in zip(rr, cc):
        if lab_ID_bool[r, c]:
            nearest_ID = nearest_nonzero_2D(sep_bud_label, r, c)
            temp_sep_bud_lab[r,c] = nearest_ID
    sep_bud_label = temp_sep_bud_lab
    sep_bud_label_mask = sep_bud_label != 0
    # plt.imshow_tk(sep_bud_label, dots_coords=np.asarray(defects_points))
    if rp is not None:
        lab[obj.slice][sep_bud_label_mask] = sep_bud_label[sep_bud_label_mask]
    else:
        lab[sep_bud_label_mask] = sep_bud_label[sep_bud_label_mask]
    max_i += 1
    success = True
    return lab, success, splittedIDs

def validate_multidimensional_recipe(
        recipe: List[Dict[str, Any]], 
        apply_to_all_zslices=False,
        apply_to_all_frames=False
    ):
    for step in recipe:
        method = step['method']
        func = PREPROCESS_MAPPER[method]['function']
        kwargs = step['kwargs']
        
        argspecs = inspect.getfullargspec(func)
        for arg in argspecs.args:
            if arg == 'apply_to_all_frames':
                kwargs['apply_to_all_frames'] = apply_to_all_frames
            if arg == 'apply_to_all_zslices':
                kwargs['apply_to_all_zslices'] = apply_to_all_zslices
    
    return recipe

def insert_missing_object(lab_dst, obj, all_dst_IDs, assignments_mapper):
    added_ID = assignments_mapper.get(obj.label)
    if obj.label not in all_dst_IDs:
        # First time we insert the missing ID and not existing in dst
        # --> safe to assign the same ID
        lab_dst[obj.slice][obj.image] = obj.label
        all_dst_IDs.add(obj.label)
    elif added_ID is None:
        # First time we insert the missing ID but already existing in dst
        # --> need to assign a new unique ID
        new_unique_ID = max(all_dst_IDs) + 1
        lab_dst[obj.slice][obj.image] = new_unique_ID
        assignments_mapper[obj.label] = new_unique_ID
        all_dst_IDs.add(new_unique_ID)
    else:
        # Already inserted the missing ID and already existing in dst
        # --> need to assign the same ID as before
        lab_dst[obj.slice][obj.image] = added_ID
        all_dst_IDs.add(added_ID)
    
    return lab_dst, assignments_mapper, all_dst_IDs

def insert_missing_objects(
        segm_dst, segm_src, is_timelapse=True, display_pbar=True
    ):
    if not is_timelapse:
        segm_dst = segm_dst[np.newaxis]
        segm_src = segm_src[np.newaxis]
    
    all_dst_IDs = set()
    for lab_dst in segm_dst:
        rp = skimage.measure.regionprops(lab_dst)
        all_dst_IDs.update([obj.label for obj in rp])
    
    if display_pbar:
        pbar = tqdm(total=len(segm_src), ncols=100, leave=False)
    
    assignments_mapper = {}
    for frame_i, (lab_src, lab_dst) in enumerate(zip(segm_src, segm_dst)):
        rp = skimage.measure.regionprops(lab_src)
        rp_dst = skimage.measure.regionprops(lab_dst)
        rp_dst_mapper = {obj.label: obj for obj in rp_dst}
        for obj in rp:
            obj_dst_values = lab_dst[obj.slice][obj.image]
            obj_dst_ID = obj_dst_values[0]
            is_missing = obj_dst_ID == 0
            if is_missing:
                out = insert_missing_object(
                    lab_dst, obj, all_dst_IDs, assignments_mapper
                )
                lab_dst, assignments_mapper, all_dst_IDs = out
                segm_dst[frame_i] = lab_dst
                continue
            
            # Check if merged --> the masks do not coincide
            obj_dst = rp_dst_mapper[obj_dst_ID]
            is_merged = not (
                len(obj_dst.coords) == len(obj.coords)
                and np.all(obj_dst.coords == obj.coords)
            )
            
            if not is_merged:
                continue
            
            lab_dst, assignments_mapper, all_dst_IDs = insert_missing_object(
                lab_dst, obj, all_dst_IDs, assignments_mapper
            )
            segm_dst[frame_i] = lab_dst
            
        if display_pbar:
            pbar.update()
    
    if display_pbar:
        pbar.close()
        
    return segm_dst
    
### out of date
# def process_lab(task):
#     i, lab = task
#     # Assuming this function processes each lab independently
#     data_dict = {}
#     rp = skimage.measure.regionprops(lab)
#     IDs = [obj.label for obj in rp]
#     data_dict['IDs'] = IDs
#     data_dict['regionprops'] = rp
#     data_dict['IDs_idxs'] = {ID: idx for idx, ID in enumerate(IDs)}
    
#     return i, data_dict, IDs  # Return index, data_dict, and IDs

# def parallel_count_objects(posData, logger_func):
#     benchmark = True
#     #futile attempt to use multiprocessing to speed things up
#     logger_func('Counting total number of segmented objects...')
    
#     allIDs = set()
#     seg_data = posData.segm_data
    
#     # Initialize empty data dictionary to avoid recalculating each time
#     tasks = [(i, lab) for i, lab in enumerate(seg_data)]

#     if benchmark:
#         t0 = time.perf_counter()
#     # Process in batches to optimize memory usage and control parallelism
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_lab, task) for task in tasks]
        
#         # Process results as they are completed
#         for future in tqdm(as_completed(futures), total=len(futures), ncols=100):
#             i, data_dict, IDs = future.result()
#             posData.allData_li[i] = myutils.get_empty_stored_data_dict() # or directly assign if it's mutable
#             posData.allData_li[i]['IDs'] = data_dict['IDs']
#             posData.allData_li[i]['regionprops'] = data_dict['regionprops']
#             posData.allData_li[i]['IDs_idxs'] = data_dict['IDs_idxs']
#             allIDs.update(IDs)
    
#     if benchmark:
#         t1 = time.perf_counter()
#         logger_func(f'Counting objects took {(t1 - t0)*1000:.2f} ms')

#     return allIDs, posData

def check_file_time_proximity(file1, file2, max_seconds=300, logger_func=print):
    if not os.path.isfile(file1):
        return False
    
    if not os.path.isfile(file2):
        return False
    
    mtime1 = os.path.getmtime(file1)
    mtime2 = os.path.getmtime(file2)
    
    if abs(mtime1 - mtime2) <= max_seconds:
        return True
    else:
        logger_func(f'Warning: The files "{file1}" and "{file2}" were not saved within {max_seconds} seconds of each other.')
        return False

def verify_acdc_df_segm(posData: load.loadData, logger_func=print):
    if posData.segmMetadata is None:
        return None
    segm_info = posData.segmMetadata[os.path.basename(posData.segm_npz_path)]
    imgs_folder = posData.images_path
    csv_name = segm_info['acdc_df_segm'] if 'acdc_df_segm' in segm_info else None
    if csv_name is None:
        return None
    csv_filepath = os.path.join(imgs_folder, csv_name)
    
    # verify that that both files exist and are within the allowed time proximity
    success = check_file_time_proximity(
        posData.segm_npz_path, csv_filepath, max_seconds=120, logger_func=logger_func
    )
    if not success:
        return None
    
    return csv_filepath

def verify_add_data_segm_proximity(posData: load.loadData, logger_func=print):
    segm_path = posData.segm_npz_path
    segm_filename = os.path.basename(segm_path).replace('.npz', '')
    add_data_folder = os.path.join(posData.images_path, segm_filename)
    
    centroids_path = os.path.join(add_data_folder, 'centroids.pkl')
    # IDs_path = os.path.join(add_data_folder, 'IDs.pkl')
    centroids_IDs_exact_path = os.path.join(add_data_folder, 'centroids_IDs_exact.pkl')
    # ID_to_idx_path = os.path.join(add_data_folder, 'ID_to_idx.pkl')
    
    ok = [True] * 2
    for idx, file in enumerate([centroids_path, centroids_IDs_exact_path]):
        success = check_file_time_proximity(
            segm_path, file, max_seconds=120, logger_func=logger_func
        )
        if not success:
            ok[idx] = False
            
    return {
        'centroids': centroids_path if ok[0] else None,
        # 'IDs': IDs_path if ok[1] else None,
        'centroids_IDs_exact': centroids_IDs_exact_path if ok[1] else None,
        # 'ID_to_idx': ID_to_idx_path if ok[3] else None,
    }
        
    
# WARNING: this function has been attempted to be optimized by
#    parallelization, loading data from last session
# The main bottleneck seams to be the rp creation (not even for example getting the IDs or centorids)
# Total time spend optimising here
#     >5 hrs
# please update this if you try to optimize again
def count_objects_and_init_rps(posData: load.loadData, logger_func=print):
    allIDs = set()

    segm_data = posData.segm_data
    if not np.any(segm_data):
        allIDs = []
        return allIDs, posData
    
    # check if csv is usable
    # csv_filepath = verify_acdc_df_segm(posData, logger_func)
    
    logger_func('Counting total number of segmented objects...')

    # frames_in_csv = set()
    # if csv_filepath is not None:
    #     logger_func(f'Loading ACDC dataframe from "{csv_filepath}"...')
    #     acdc_df = pd.read_csv(csv_filepath)
    #     frames_in_csv = set(acdc_df['frame_i'])
    #     pbar = tqdm(total=len(frames_in_csv), ncols=100, leave=False)
    #     for i in frames_in_csv:
    #         posData.allData_li[i] = myutils.get_empty_stored_data_dict()
    #         relevant_df = acdc_df[acdc_df['frame_i'] == i]
    #         rp = regionprops.acdcRegionprops(segm_data[i], relevant_df)
    #         posData.allData_li[i]['regionprops'] = rp
    #         allIDs.update(rp.IDs_set)
    #         pbar.update()
    #     pbar.close()
    
    # # missing_i = set(range(len(segm_data))) - frames_in_csv if csv_filepath is not None else set(range(len(segm_data)))
    # if len(missing_i) == 0:
    #     return allIDs, posData
    
    # add_data_dict = verify_add_data_segm_proximity(posData, logger_func)
    # centroids_loaded = ID_to_idx_loaded = centroids_IDs_exact_loaded = IDs_loaded = None
    # if add_data_dict['centroids'] is not None:
    #     with open(add_data_dict['centroids'], 'rb') as f:
    #         centroids_loaded = pickle.load(f)
    # # if add_data_dict['IDs'] is not None:
    # #     with open(add_data_dict['IDs'], 'rb') as f:
    # #         IDs_loaded = pickle.load(f)
    # if add_data_dict['centroids_IDs_exact'] is not None:
    #     with open(add_data_dict['centroids_IDs_exact'], 'rb') as f:
    #         centroids_IDs_exact_loaded = pickle.load(f)
    # # if add_data_dict['ID_to_idx'] is not None:
    #     with open(add_data_dict['ID_to_idx'], 'rb') as f:
    #         ID_to_idx_loaded = pickle.load(f)
    # other ids:
    pbar = tqdm(total=len(segm_data), ncols=100)
    for i, lab in enumerate(segm_data):
        # if i in frames_in_csv:
        #     continue # skip frames already processed with csv
        # _centroids_loaded = centroids_loaded[i] if centroids_loaded is not None and i in centroids_loaded else None
        # # _IDs_loaded = IDs_loaded[i] if IDs_loaded is not None and i in IDs_loaded else None
        # _centroids_IDs_exact_loaded = centroids_IDs_exact_loaded[i] if centroids_IDs_exact_loaded is not None and i in centroids_IDs_exact_loaded else None
        # _ID_to_idx_loaded = ID_to_idx_loaded[i] if ID_to_idx_loaded is not None and i in ID_to_idx_loaded else None
        posData.allData_li[i] = myutils.get_empty_stored_data_dict()
        rp = regionprops.acdcRegionprops(lab,
                                        #  centroids_loaded=_centroids_loaded, 
                                        #  IDs_loaded=_IDs_loaded, 
                                        #  centroids_IDs_exact_loaded=_centroids_IDs_exact_loaded, 
                                        #  ID_to_idx_loaded=_ID_to_idx_loaded
                                        )
        IDs = rp.IDs_set      
        posData.allData_li[i]['regionprops'] = rp
        allIDs.update(IDs)
        pbar.update()
    pbar.close()
    return allIDs, posData

def fix_sparse_directML(verbose=True):
    """DirectML does not support sparse tensors, so we need to fallback to CPU.
    This function replaces `torch.sparse_coo_tensor`, `torch._C._sparse_coo_tensor_unsafe`,
    `torch._C._sparse_coo_tensor_with_dims_and_tensors`, `torch.sparse.SparseTensor`
     with a wrapper that falls back to CPU.

    In the end, this could be handled better in the future. It would probably run faster if we
    just manually set the device to CPU, but my goal was to not modify the code too much,
    and this runs suprisingly fast.
    """
    import torch
    import functools
    import warnings

    def fallback_to_cpu_on_sparse_error(func, verbose=True):
        @functools.wraps(func) # wrapper shinanigans (thanks chatgpt)
        def wrapper(*args, **kwargs):
            device_arg = kwargs.get('device', None) # get desired device from kwargs

            # Ensure indices are int64 if args[0] looks like indices,
            # I got random errors from it not being int64
            if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                if args[0].dtype != torch.int64:
                    args = (args[0].to(dtype=torch.int64),) + args[1:]

            try: # try to perform the operation and move to dml if possible
                result = func(*args, **kwargs) # run function with current args and kwargs
                if device_arg is not None and str(device_arg).lower() == "dml":
                    try: # try to move result to dml
                        result.to("dml")
                    except RuntimeError as e: # moving failed, falling back to cpu 
                        if verbose:
                            warnings.warn(f"Sparse op failed on DirectML, falling back to CPU: {e}")
                        kwargs['device'] = torch.device("cpu")
                        return func(*args, **kwargs) # try again, after setting device to cpu
                return result # just return result if all worked well

            except RuntimeError as e: # try and run on dlm, if it fails, fallback to cpu
                if "sparse" in str(e).lower() or "not implemented" in str(e).lower():
                    if verbose:
                        warnings.warn(f"Sparse op failed on DirectML, falling back to CPU: {e}")
                    kwargs['device'] = torch.device("cpu") # if rutime warning caused by sparse tensor, set device to cpu

                    # Re-apply indices dtype correction before retrying on CPU. Just in case (maybe first one not needed?)
                    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                        if args[0].dtype != torch.int64:
                            args = (args[0].to(dtype=torch.int64),) + args[1:]

                    return func(*args, **kwargs) # run function again with cpu device
                else:
                    raise e # catch and other runtime errors

        return wrapper

    # --- Patch Sparse Tensor Constructors ---

    # High-level API
    torch.sparse_coo_tensor = fallback_to_cpu_on_sparse_error(torch.sparse_coo_tensor, verbose=verbose)

    # Low-level API
    if hasattr(torch._C, "_sparse_coo_tensor_unsafe"):
        torch._C._sparse_coo_tensor_unsafe = fallback_to_cpu_on_sparse_error(torch._C._sparse_coo_tensor_unsafe, verbose=verbose)

    if hasattr(torch._C, "_sparse_coo_tensor_with_dims_and_tensors"):
        torch._C._sparse_coo_tensor_with_dims_and_tensors = fallback_to_cpu_on_sparse_error(
            torch._C._sparse_coo_tensor_with_dims_and_tensors, verbose=verbose
        )

    if hasattr(torch.sparse, 'SparseTensor'):
        torch.sparse.SparseTensor = fallback_to_cpu_on_sparse_error(torch.sparse.SparseTensor, verbose=verbose)
    
    # suppress warnings
    if not verbose:
        import warnings
        warnings.filterwarnings("once", message="Sparse op failed on DirectML*")

def connected_components_in_undirected_graph(undirected_graph:dict):
    # Build undirected graph
    graph = defaultdict(set)
    for key, val in undirected_graph.items():
        for other in val:
            graph[key].add(other)
            graph[other].add(key)  # Make it bidirectional
    
    visited = set()
    groups = []

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group) # recursive call to visit neighbors

    for key in graph:
        if key not in visited:
            group = []
            dfs(key, group)
            groups.append(group)
    
    return groups

def apply_func_to_imgs(image:np.ndarray, 
                       func: Callable, 
                       *args,
                       workers: int = 10,
                       iter_axis:List[int]|int= None, 
                       target_shape:List[int] = None,
                       target_type: type = None,
                       target_axis_iter: List[int]|int = None,
                       parallel: bool = True,
                       benchmark: bool = False,
                       processpool: bool = False,
                       **kwargs):        
    """Apply a function to each image. This is done along the iter_axis (can also be a single int).
    Then the processed image is put in the target_axis_iter (can also be a single int).
    (If target_axis_iter, target_shape or target_type are None, 
    they are taken from the input image).
    Example of iter_axis: [0, 1] and target_axis_iter: [1, 0] means that the function is applied to each
    [0, 1, ...] slice of the input and the processed image is put in the [1, 0, ...] slice of the output image.

    THe function should follow the signature:
    frame_index_out, image_out = func(image, *args, frame_index_out=None, **kwargs)

    Parameters
    ----------
    image : np.ndarray
        Image to be processed
        
    func : Callable
        Function to be applied to each image. First argument
        should be the image itself, one kwarg should be `frame_index_out`,
        should `return processed_image, frame_index_out`. `frame_index_out` just needs to 
        be passed along, no need to slice the image in `func`.
    *args : tuple
        Additional arguments to be passed to the function
    workers : int, optional
        Number of workers to use, by default 10
    iter_axis : List[int]|int, optional
        Axis along which to iterate, by default None
        If None, the function is applied to the entire image.
    target_shape : List[int], optional
        Shape of the output image, by default None
        If None, the shape of the input image is used.
    target_type : type, optional
        Type of the output image, by default None
        If None, the type of the input image is used.
    target_axis_iter : List[int]|int, optional
        Axis along to which to put the processed image
        Must be same length as `iter_axis`, by default None
        If None, the processed image is put in the same axis as the input image.
    parallel : bool, optional
        Whether to use parallel processing, by default True
        If False, the function is applied to each image sequentially.
    benchmark : bool, optional
        Whether to benchmark the function, by default False
        If True, prints the execution time of the function.
    processpool : bool, optional
        Whether to use process pool, by default False
        If True, uses process pool instead of thread pool.
    **kwargs : dict
        Additional keyword arguments to be passed to the function

    Returns
    -------
    np.ndarray
        Processed image
    """
    if benchmark:
        t0 = time.perf_counter()
    image_shape = image.shape
    if iter_axis is None:
        out = func(image, *args, **kwargs, frame_index_out=None)[1]
        if benchmark:
            t1 = time.perf_counter()
            printl(f"Processing time: {(t1 - t0)*1000:.2f} ms, no parallel since iter_axis is None")
        return out 
    
    if isinstance(iter_axis, int):
        iter_axis = [iter_axis]
    
    if isinstance(target_axis_iter, int):
        iter_axis = [target_axis_iter]
    
    if target_axis_iter is None:
        target_axis_iter = iter_axis
    
    if target_shape is None:
        target_shape = image_shape
    
    if target_type is None:
        target_type = type(image.flat[0])
    

    image_out = np.empty(
        target_shape, dtype=target_type
    )

    input_output_mapper = myutils.get_input_output_mapper(
        image_shape, iter_axis, target_shape, target_axis_iter
    )

    if parallel:
        if processpool:
            executor_func = ProcessPoolExecutor
        else:
            executor_func = ThreadPoolExecutor
        with executor_func() as executor:
            futures = {
                executor.submit(func, image[i_in], *args, frame_index_out=i_out, **kwargs)
                for i_in, i_out in input_output_mapper
            }

            for future in tqdm(
                    as_completed(futures), 
                    total=len(futures), 
                    desc="Processing frames"
                ):
                i, processed = future.result()
                image_out[i] = processed
    else:
        for i_in, i_out in tqdm(input_output_mapper, desc="Processing frames"):
            processed = func(image[i_in], *args, frame_index_out=i_out, **kwargs)[1]
            image_out[i_out] = processed
    
    if benchmark:
        t1 = time.perf_counter()
        printl(f"Processing time: {(t1 - t0)*1000:.2f} ms")

    return image_out


def fill_holes_in_segmentation(labels):
    filled = np.zeros_like(labels)
    for obj in skimage.measure.regionprops(labels):
        label_id = obj.label
        mask_filled = scipy.ndimage.binary_fill_holes(obj.image)
        
        region = filled[obj.slice]
        # Only fill where mask_filled is True and region is still background
        fill_mask = mask_filled & (region == 0)
        region[fill_mask] = label_id
        filled[obj.slice] = region

    return filled

def natsort_acdc_columns(
        columns: Iterable[str], 
        prepend_default_index_cols=True
    ):
    sorted_cols = natsorted(columns, key=str.casefold)
    if not prepend_default_index_cols:
        return sorted_cols
    
    cols_to_prepend = []
    for col in default_index_cols:
        if col not in sorted_cols:
            continue
        
        sorted_cols.remove(col)
        cols_to_prepend.append(col)
    
    sorted_cols = [*cols_to_prepend, *sorted_cols]
    return sorted_cols

def linear_fit_3d(xx, yy, zz):
    points = np.column_stack((xx, yy, zz))
    centroid = points.mean(axis=0)

    _, _, vh = np.linalg.svd(points - centroid)
    d = vh[0]

    return centroid, d

def binary_fill_holes(mask, slice_by_slice=True):
    if not slice_by_slice:
        mask = scipy.ndimage.binary_fill_holes(mask)
        return mask
    
    if mask.ndim == 2:
        mask = scipy.ndimage.binary_fill_holes(mask)
        return mask
    
    for z, mask_z in enumerate(mask):
        if not np.any(mask_z):
            continue
        
        mask[z] = scipy.ndimage.binary_fill_holes(mask_z)
    
    return mask

def convex_hull_mask(mask: np.ndarray, slice_by_slice=True):
    if not slice_by_slice:
        mask = skimage.morphology.convex_hull_image(mask)
        return mask
    
    if mask.ndim == 2:
        mask = skimage.morphology.convex_hull_image(mask)
        return mask
    
    mask_rp = skimage.measure.regionprops(mask.astype(np.uint8))
    if len(mask_rp) == 0:
        return mask
    
    mask_obj = mask_rp[0]
    for z, mask_obj_img_z in enumerate(mask_obj.image):
        if not np.any(mask_obj_img_z):
            continue
        
        mask_obj_hull_z = skimage.morphology.convex_hull_image(mask_obj_img_z)
        mask[mask_obj.slice][z] = mask_obj_hull_z
    
    return mask