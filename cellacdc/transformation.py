import xml.etree.ElementTree as ET 

import math
import pandas as pd
import numpy as np

from skimage.transform import resize
import skimage.measure

from tqdm import tqdm

from . import printl, core

def resize_lab(lab, output_shape, rp=None):
    if rp is None:
        rp = skimage.measure.regionprops(lab)
    _lab_obj_to_resize = np.zeros(lab.shape, dtype=np.float16)
    lab_resized = np.zeros(output_shape, dtype=np.uint32)
    for obj in rp:
        _lab_obj_to_resize[obj.slice][obj.image] = 1.0
        _lab_obj_resized = resize(
            _lab_obj_to_resize, output_shape, anti_aliasing=True,
            preserve_range=True
        ).round()
        lab_resized[_lab_obj_resized == 1.0] = obj.label
        _lab_obj_to_resize[:] = 0.0
    return lab_resized

def crop_2D(img, xy_range, tolerance=0, return_copy=True):
    (xmin, xmax), (ymin, ymax) = xy_range
    Y, X = img.shape
    xmin -= tolerance
    xmax += tolerance
    ymin -= tolerance
    ymax += tolerance
    xmin = 0 if xmin < 0 else round(xmin)
    xmax = X if xmax > X else round(xmax)
    ymin = 0 if ymin < 0 else round(ymin)
    ymax = Y if ymax > Y else round(ymax)
    crop_shape = (ymax-ymin, xmax-xmin)
    crop_slice = (slice(ymin, ymax, None), slice(xmin, xmax, None))
    if return_copy:
        cropped = np.zeros(crop_shape, dtype=img.dtype)
        cropped[:] = img[crop_slice]
    else:
        cropped = img[crop_slice]
    return cropped, crop_slice

def del_objs_outside_segm_roi(segm_roi, segm):
    del_IDs = np.unique(segm[segm_roi==0])
    cleared_segm = segm.copy()
    clearedIDs = []
    for del_ID in del_IDs:
        if del_ID == 0:
            continue
        cleared_segm[segm==del_ID] = 0
        clearedIDs.append(del_ID)
    return cleared_segm, clearedIDs

def trackmate_xml_to_df(xml_file):
    IDs = []
    xx = []
    yy = []
    zz = []
    frame_idxs = []
    tree = ET.parse(xml_file)
    Tracks = tree.getroot()

    for i, particle in enumerate(Tracks):
        ID = i+1
        for t, detection in enumerate(particle):
            attrib = detection.attrib
            IDs.append(ID)
            xx.append(attrib['x'])
            yy.append(attrib['y'])
            zz.append(attrib['z'])
            frame_idxs.append(attrib['t'])
    
    df = pd.DataFrame({
        'frame_i': frame_idxs,
        'ID': IDs, 
        'x': xx,
        'y': yy,
        'z': zz
    })
    return df

def retrack_based_on_untracked_first_frame(
        tracked_video, first_untracked_lab, uniqueID=None        
    ):
    """Re-tack the objects in the first frame of `tracked_video` to have the 
    same IDs as in `first_untracked_lab`

    Parameters
    ----------
    tracked_video : (T, Y, X) or (T, Z, Y, X) of ints
        Array with the segmentation instances of the tracked objects
    first_untracked_lab : (Y, X) or (Z, Y, X) of ints
        Array with the segmentation instances of the objects in the first 
        frame before they were tracked
    uniqueID : int, optional
        If not None, it will be used as first of the unique IDs. 
        If None, this will be initialized to the maximum in `tracked_video`. 
        Default is None.

    Returns
    -------
    (T, Y, X) or (T, Z, Y, X) of ints
        Tracked video where the objects in the first frame has the same IDs as 
        in `first_untracked_lab`. 
    
    Notes
    -----
    The idea of this function is to ensure that objects in the first frame 
    before and after tracking have the same IDs. This is needed to ensure 
    continuity of obejct IDs when tracking portions of the video in 
    different batches.    
    """    
    
    first_tracked_lab = tracked_video[0]
    first_tracked_rp = skimage.measure.regionprops(first_tracked_lab)
    
    tracked_to_untracked_mapper = {}
    for obj in first_tracked_rp:
        untracked_ID = first_untracked_lab[obj.slice][obj.image][0]
        if untracked_ID == obj.label:
            continue
        tracked_to_untracked_mapper[obj.label] = untracked_ID

    if not tracked_to_untracked_mapper:
        return tracked_video
    
    first_untracked_rp = skimage.measure.regionprops(first_untracked_lab)
    first_untracked_IDs = [obj.label for obj in first_untracked_rp]
    
    if uniqueID is None:
        uniqueID = np.max(tracked_video) + 1
    uniqueIDs = np.arange(uniqueID, uniqueID+len(first_untracked_IDs))

    untracked_to_unique_mapper = (
        dict(zip(first_untracked_IDs, uniqueIDs))
    )
    
    pbar = tqdm(total=len(tracked_video), ncols=100)
    for frame_i, tracked_lab in enumerate(tracked_video):
        rp_tracked = skimage.measure.regionprops(tracked_lab)
        for obj_tracked in rp_tracked:
            new_unique_ID = untracked_to_unique_mapper.get(obj_tracked.label)
            if new_unique_ID is None:
                # Untracked ID not present in tracked labels
                continue
            
            untracked_ID = tracked_to_untracked_mapper.get(obj_tracked.label)
            if untracked_ID is None:
                # No need to make ID unique because it will not change later
                continue
            
            # Replace untracked ID with a unique ID to prevent merging when later 
            # we will replace tracked IDs of first frame to their corresponding 
            # untracked ID
            tracked_video[tracked_video==obj_tracked.label] = new_unique_ID

            # Update tracked to untracked mapper because now tracked_video 
            # changed and we would not find the same ID later
            tracked_to_untracked_mapper[new_unique_ID] = (
                tracked_to_untracked_mapper.pop(obj_tracked.label)
            )
            
        pbar.update()
    pbar.close()
    
    uniqueID = np.max(tracked_video) + 1
    
    untracked_to_unique_mapper = {}
    pbar = tqdm(total=len(tracked_video), ncols=100)
    for frame_i, tracked_lab in enumerate(tracked_video):
        rp_tracked = skimage.measure.regionprops(tracked_lab)
        rp_tracked_dict = {obj.label:obj for obj in rp_tracked}
        for obj_tracked in rp_tracked:
            untracked_ID = tracked_to_untracked_mapper.get(obj_tracked.label)
            if untracked_ID is None:
                # Untracked ID not present in tracked labels
                continue
            
            untracked_obj = rp_tracked_dict.get(untracked_ID)
            if untracked_obj is not None:
                new_unique_ID = untracked_to_unique_mapper.get(untracked_ID)
                if new_unique_ID is None:
                    new_unique_ID = uniqueID
                    untracked_to_unique_mapper[untracked_ID] = new_unique_ID
                    uniqueID += 1
                
                # Make sure to change existing IDs to unique
                lab = tracked_video[frame_i]
                lab[untracked_obj.slice][untracked_obj.image] = (
                    new_unique_ID
                )
            
            # Replace tracked ID of first frame to the untracked ID of the 
            # reference 
            tracked_video[frame_i][obj_tracked.slice][obj_tracked.image] = (
                untracked_ID
            )
        pbar.update()
    pbar.close()
    
    return tracked_video

def remove_padding_2D(arr, val=0, return_crop_slice=False):
    crop_slice = []
    for a, ax in enumerate((1, 0)):
        pad_ax = arr.sum(axis=ax)
        if np.isnan(val):
            pad_ax_mask = np.isnan(pad_ax)
        else:
            pad_ax_mask = pad_ax == val
            
        pad_ax_left = 0
        for i, val in enumerate(pad_ax_mask):
            if not val:
                pad_ax_left = i
                break  
        
        pad_ax_right = arr.shape[a]
        for j, val in enumerate(pad_ax_mask[::-1]):
            if not val:
                pad_ax_right -= j
                break  
        
        crop_slice.append(slice(pad_ax_left, pad_ax_right))
    
    crop_slice = tuple(crop_slice)
    if return_crop_slice:
        return arr[crop_slice], crop_slice
    
    return arr[tuple(crop_slice)]
    
def snap_xy_to_closest_angle(x0, y0, x1, y1, angle_factor=15):
    # Snap to closest angle divisible by angle_factor degrees
    angle = math.degrees(math.atan2(y1-y0, x1-x0))
    snap_angle = math.radians(core.closest_n_divisible_by_m(angle, angle_factor))
    dist = math.dist((x0, y0), (x1, y1))
    dx = dist * math.cos(snap_angle)
    dy = dist * math.sin(snap_angle)
    x1, y1 = x0 + dx, y0 + dy
    return x1, y1