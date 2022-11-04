import traceback
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

from tqdm import tqdm

# Custom modules
from . import apps, base_cca_df, printl
from . import load, myutils

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

def remove_artefacts(
        labels, min_solidity=0.5, min_area=15, max_elongation=3,
        return_delIDs=False
    ):
    if labels.ndim == 3:
        delIDs = set()
        for z, lab in enumerate(labels):
            _result = remove_artefacts_lab2D(
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
        result = remove_artefacts_lab2D(
            labels, min_solidity, min_area, max_elongation,
            return_delIDs=return_delIDs
        )

    if return_delIDs:
        labels, delIDs = result
        return labels, delIDs
    else:
        labels = result
        return labels

def remove_artefacts_lab2D(
        lab, min_solidity, min_area, max_elongation,
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
        if obj.area < min_area:
            lab[obj.slice][obj.image] = 0
            if return_delIDs:
                delIDs.append(obj.label)
            continue

        if obj.solidity < min_solidity:
            lab[obj.slice][obj.image] = 0
            if return_delIDs:
                delIDs.append(obj.label)
            continue

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
        else:
            sub_acdc_df_frame_i = (
                subobj_acdc_df.loc[frame_i].rename(index=old_sub_ids) 
            )
            if 'relative_ID' in sub_acdc_df_frame_i.columns:
                sub_acdc_df_frame_i['relative_ID'] = (
                    sub_acdc_df_frame_i['relative_ID'].replace(old_sub_ids)
                )
        
        num_objects_per_cells = all_num_objects_per_cells[frame_i]

        # For sub-obj that do not have a parent cell the num_sub_cell_objs_per_cell = 0        
        sub_acdc_df_frame_i['num_sub_cell_objs_per_cell'] = (
            [num_objects_per_cells.get(sub_id, 0) for sub_id in sub_ids]
        )
        
        sub_acdc_df_list.append(sub_acdc_df_frame_i)
        keys_sub.append(frame_i)
        
        if tracked_cells_segm_data is not None:
            lab = tracked_cells_segm_data[frame_i]
            rp = skimage.measure.regionprops(lab)
            # Untacked sub-obj (if kept) are not present in acdc_df of the cells
            # --> check with `IDs_with_sub_obj = ... if id in lab`
            IDs_with_sub_obj = [id for id in sub_ids if id in lab]
            if cells_acdc_df is None:
                acdc_df_frame_i = myutils.getBaseAcdcDf(rp)
            else:
                acdc_df_frame_i = cells_acdc_df.loc[frame_i].copy()

            acdc_df_frame_i['num_sub_cell_objs_per_cell'] = 0
            acdc_df_frame_i.loc[IDs_with_sub_obj, 'num_sub_cell_objs_per_cell'] = ([
                num_objects_per_cells[id] for id in IDs_with_sub_obj
            ])
            acdc_df_list.append(acdc_df_frame_i)
            keys_cells.append(frame_i)

        if sigProgress is not None:
            sigProgress.emit(1)
            
    tracked_acdc_df = None
    sub_tracked_acdc_df = pd.concat(
        sub_acdc_df_list, keys=keys_sub, names=['frame_i', 'Cell_ID']
    )
    if tracked_cells_segm_data is not None:
        tracked_acdc_df = pd.concat(
            acdc_df_list, keys=keys_cells, names=['frame_i', 'Cell_ID']
        )
    
    return sub_tracked_acdc_df, tracked_acdc_df
        
        
def track_sub_cell_objects(
        cells_segm_data, subobj_segm_data, IoAthresh, 
        how='delete_sub', SizeT=None, sigProgress=None
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

        how (str, optional): _description_. Defaults to 'delete_sub'.

        SizeT (int, optional): Number of frames. Pass `SizeT=1` for non-timelapse
            data. Defaults to None --> assume first dimension of segm data is SizeT.

        sigProgress (PyQt5.QtCore.pyqtSignal, optional): If provided it will emit 
            1 for each complete frame. Used to update GUI progress bars. 
            Defaults to None --> do not emit signal.
    
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
    tracked_IDs = set()
    all_num_objects_per_cells = []
    all_old_sub_ids = [{} for _ in range(len(cells_segm_data))]
    for frame_i, (lab, lab_sub) in enumerate(segm_data_zip):
        rp = skimage.measure.regionprops(lab)
        num_objects_per_cells = {obj.label:0 for obj in rp}
        rp_sub = skimage.measure.regionprops(lab_sub)
        tracked_lab_sub = tracked_subobj_segm_data[frame_i]
        old_sub_ids = all_old_sub_ids[frame_i]
        for sub_obj in rp_sub:
            intersect_mask = lab[sub_obj.slice][sub_obj.image]
            intersect_IDs, I_counts = np.unique(
                intersect_mask, return_counts=True
            )
            for intersect_ID, I in zip(intersect_IDs, I_counts):
                if intersect_ID == 0:
                    continue
                
                IoA = I/sub_obj.area
                if IoA < IoAthresh:
                    # Do not add untracked sub-obj
                    continue
                
                old_sub_ids[intersect_ID] = sub_obj.label
                tracked_lab_sub[sub_obj.slice][sub_obj.image] = intersect_ID
                num_objects_per_cells[intersect_ID] += 1
                tracked_IDs.add(intersect_ID)
        
        all_num_objects_per_cells.append(num_objects_per_cells)
        
        if sigProgress is not None:
            sigProgress.emit(1)
    
    if how == 'delete_both' or how == 'delete_cells':
        # Delete cells that do not have a sub-cellular object
        tracked_cells_segm_data = cells_segm_data.copy()
        for frame_i, lab in enumerate(tracked_cells_segm_data):
            rp = skimage.measure.regionprops(lab)
            tracked_lab = tracked_cells_segm_data[frame_i]
            for obj in rp:
                if obj.label in tracked_IDs:
                    continue
                    
                tracked_lab[obj.slice][obj.image] = 0
    
    if how == 'only_track' or how == 'delete_cells':
        # Assign unique IDs to untracked sub-cellular objects and add them 
        # to all_old_sub_ids
        maxSubObjID = tracked_subobj_segm_data.max() + 1
        for sub_obj_ID in np.unique(subobj_segm_data):
            if sub_obj_ID == 0:
                continue

            if sub_obj_ID in tracked_IDs:
                continue
            
            tracked_subobj_segm_data[subobj_segm_data == sub_obj_ID] = maxSubObjID
            for frame_i, lab_sub in enumerate(subobj_segm_data):
                if sub_obj_ID not in lab_sub:
                    continue
                all_old_sub_ids[frame_i][maxSubObjID] = sub_obj_ID
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

def get_objContours(obj, all=False):
    if all:
        retrieveMode = cv2.RETR_CCOMP
    else:
        retrieveMode = cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(
        obj.image.astype(np.uint8), retrieveMode, cv2.CHAIN_APPROX_NONE
    )
    min_y, min_x, _, _ = obj.bbox
    if all:
        return [np.squeeze(cont, axis=1)+[min_x, min_y] for cont in contours]
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
    def __init__(self, acdc_df) -> None:
        acdc_df = load.pd_bool_to_int(acdc_df).reset_index()
        acdc_df = self._normalize_gen_num(acdc_df).reset_index()
        acdc_df = acdc_df.drop(columns=['index', 'level_0'], errors='ignore')
        self.acdc_df = acdc_df.set_index(['frame_i', 'Cell_ID'])
        self.df = acdc_df.copy()
        self.cca_df_colnames = list(base_cca_df.keys())
    
    def build(self):
        print('Building lineage tree...')
        try:
            df_G1 = self.acdc_df[self.acdc_df['cell_cycle_stage'] == 'G1']
            self.df_G1 = df_G1[self.cca_df_colnames].copy()
            self.new_col_loc = df_G1.columns.get_loc('division_frame_i') + 1
        except Exception as error:
            return error
        
        self.df = self.add_lineage_tree_table_to_acdc_df()
        print('Lineage tree built successfully!')
    
    def _normalize_gen_num(self, acdc_df):
        '''
        Since the user is allowed to start the generation_num of unknown mother
        cells with any number we need to normalise this to 2 -->
        Create a new 'normalized_gen_num' column where we make sure that mother
        cells with unknown history have 'normalized_gen_num' starting from 2
        (required by the logic of _build_tree)
        '''
        acdc_df = acdc_df.reset_index().drop(columns='index', errors='ignore')

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
        if not self.traversing_branch_ID:
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
        if not self.traversing_branch_ID:
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
                parent_ID = -1
        else:
            parent_ID = -1
        
        '''
        Assign root ID --> 
            at start of branch (self.traversing_branch_ID is True) the root_ID
            is ID if gen_num_tree == 1 otherwise we go back until 
            the parent_ID == -1
            --> store this and use when traversing branch
        '''
        if not self.traversing_branch_ID:
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
            else:
                root_ID_tree = ID
            self.root_IDs_trees[ID] = root_ID_tree
        else:
            root_ID_tree = self.root_IDs_trees[ID]
            
        gen_df['root_ID_tree'] = root_ID_tree     

        # printl(
        #     f'Traversing ID: {ID}\n'
        #     f'Parent ID: {parent_ID}\n'
        #     f'Started traversing: {self.traversing_branch_ID}\n'
        #     f'Relative ID: {relID}\n'
        #     f'Relative ID generation num tree: {gen_num_relID_tree}\n'
        #     f'Generation number tree: {gen_num_tree}\n'
        #     f'New cell ID tree: {Cell_ID_tree}\n'
        #     f'Start branch gen number: {self.branch_start_gen_num[ID]}\n'
        #     f'root_ID_tree: {root_ID_tree}'
        # )
        # import pdb; pdb.set_trace()
            
        self.gen_dfs[(ID, gen_num_tree)] = gen_df
        self.gen_dfs_by_ID_tree[Cell_ID_tree] = gen_df
        
        self.traversing_branch_ID = True
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
                      
                self.traversing_branch_ID = False
                # Iterate the branch till the end
                self.df_G1 = (
                    self.df_G1
                    .groupby(['Cell_ID', 'generation_num'])
                    .apply(self._build_tree, ID)
                )
                not_annotated_IDs.remove(ID)
        
        self._add_sister_ID()

        for c, col_tree in enumerate(cols_tree):
            if col_tree in self.acdc_df.columns:
                self.acdc_df.pop(col_tree)
            self.acdc_df.insert(self.new_col_loc, col_tree, 0)

        self.acdc_df.loc[self.df_G1.index, self.df_G1.columns] = self.df_G1
        self._build_tree_S(cols_tree)

        return self.acdc_df
    
    def _add_sister_ID(self):
        grouped_ID_tree = self.df_G1.groupby('Cell_ID_tree')
        for Cell_ID_tree, df in grouped_ID_tree:
            relative_ID = df['relative_ID'].iloc[0]
            if relative_ID == -1:
                continue
            start_frame_i = df.index.get_level_values(0)[0]
            sister_ID_tree = self.df_G1.at[
                (start_frame_i, relative_ID), 'Cell_ID_tree'
            ]
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

