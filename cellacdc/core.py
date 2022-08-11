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

def align_frames_3D(data, slices=None, user_shifts=None):
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
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(z_proj_max(frame_V))
        # ax[1].imshow(z_proj_max(aligned_frame_V))
        # plt.show()
    return data_aligned, registered_shifts


def align_frames_2D(
        data, slices=None, register=True, user_shifts=None, pbar=False
    ):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(tqdm(data, ncols=100)):
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
    contours, _ = cv2.findContours(
        obj.image.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
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

class AddLineageTreeTable:
    def __init__(self, acdc_df) -> None:
        self.acdc_df = acdc_df
    
    def build(self):
        cca_df_colnames = list(base_cca_df.keys())[:-2]
        try:
            cca_df = self.acdc_df[cca_df_colnames]
            self.new_col_loc = self.acdc_df.columns.get_loc('division_frame_i') + 1
        except Exception as error:
            return error
        
        self.df = self.add_lineage_tree_table_to_acdc_df()
    
    def _build_tree(self, gen_df, ID, relID_gen_num):
        current_ID = gen_df.index.get_level_values(1)[0]
        if current_ID != ID:
            return gen_df

        '''
        Add generation number tree:
        --> gen_num - 1 for new cells unknown cell
        --> Mother generation number + gen_num for known relative cell
        '''
        gen_nums = gen_df['generation_num_tree'].values
        gen_df['generation_num_tree'] = gen_nums + relID_gen_num

        
        '''Assign unique ID every consecutive division'''
        if not self.gen_dfs:
            # Keep start ID for cell at the top of the branch
            gen_df['Cell_ID_tree'] = [ID]*len(gen_df)
        else:
            gen_df['Cell_ID_tree'] = [self.uniqueID]*len(gen_df)
            self.uniqueID += 1

        '''
        Assign parent ID: --> existing ID between relID and ID in prev gen_num_tree      
        '''
        gen_num_tree = gen_df.loc[pd.IndexSlice[:, ID], 'generation_num_tree'].iloc[0]

        if gen_num_tree > 1:
            ID_idx = pd.IndexSlice[:, ID]
            relID = gen_df.loc[ID_idx, 'relative_ID'].iloc[0]
            if not self.gen_dfs:
                # Start of the branch of a new cell
                parent_ID = relID
            else:
                prev_gen_num_df = self.gen_dfs[-1]
                ID_idx = pd.IndexSlice[:, ID]
                try:
                    parent_ID = prev_gen_num_df.loc[ID_idx, 'Cell_ID_tree'].iloc[0]
                except KeyError:
                    prev_gen_frame_i = prev_gen_num_df.index.get_level_values(0)[0]
                    parent_ID = self.acdc_df.at[
                        (prev_gen_frame_i, relID), 'Cell_ID_tree'
                    ].iloc[0]
                
            gen_df['parent_ID_tree'] = parent_ID

        self.gen_dfs.append(gen_df)       

        return gen_df
             
    def add_lineage_tree_table_to_acdc_df(self):
        acdc_df = self.acdc_df    
        acdc_df.insert(self.new_col_loc, 'Cell_ID_tree', 0)
        acdc_df.insert(self.new_col_loc+1, 'parent_ID_tree', -1)
        # acdc_df.insert(self.new_col_loc+2, 'relative_ID_tree', -1)
        gen_nums = acdc_df['generation_num']
        acdc_df.insert(self.new_col_loc+3, 'generation_num_tree', gen_nums)

        frames_idx = acdc_df.index.get_level_values(0).unique()
        not_annotated_IDs = acdc_df.index.get_level_values(1).unique().to_list()
        annotated_IDs = []

        self.uniqueID = max(not_annotated_IDs) + 1

        for frame_i in frames_idx:
            if not not_annotated_IDs:
                # Built tree for every ID --> exit
                break
            
            acdc_df_i = acdc_df.loc[frame_i]
            IDs = acdc_df_i.index.array
            for ID in IDs:
                if ID not in not_annotated_IDs:
                    # Tree already built in previous frame iteration --> skip
                    continue
                
                relID = acdc_df_i.at[ID, 'relative_ID']
                try:
                    relID_gen_num = acdc_df_i.at[relID, 'generation_num']
                except Exception as e:
                    relID_gen_num = -1
                
                self.gen_dfs = []
                # Iterate the branch till the end
                self.acdc_df = (
                    self.acdc_df
                    .groupby(['Cell_ID', 'generation_num'])
                    .apply(self._build_tree, ID, relID_gen_num)
                )
                not_annotated_IDs.remove(ID)
        return self.acdc_df
    
    def newick(self):
        if 'Cell_ID_tree' not in self.acdc_df.columns:
            self.build()
        
        pass

    
