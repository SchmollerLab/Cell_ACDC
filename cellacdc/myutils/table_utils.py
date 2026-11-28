"""Helpers for acdc_output tables (acdc_df) and other DataFrames: base and
empty tables, checked index resets, the per-frame stored-data dict, the
default segmInfo table, and converting Cell Tracking Challenge tracking
output to an acdc_df. Plugin API: df_ctc_to_acdc_df,
checked_reset_index, checked_reset_index_Cell_ID.
"""
from tqdm import tqdm
import pandas as pd
import skimage
import skimage.io
import skimage.measure
from cellacdc import core


def checked_reset_index(df):
    if df.index.names is None or df.index.names == [None]:
        return df.reset_index(drop=True)
    else:
        return df.reset_index()

def checked_reset_index_Cell_ID(df):
    if df.index.names == ['Cell_ID']:
        return df
    df = checked_reset_index(df)
    return df.set_index('Cell_ID')


def getBaseAcdcDf(rp):
    zeros_list = [0]*len(rp)
    nones_list = [None]*len(rp)
    minus1_list = [-1]*len(rp)
    IDs = [0]*len(rp)
    xx_centroid = [0]*len(rp)
    yy_centroid = [0]*len(rp)
    zz_centroid = [0]*len(rp)
    
    for i, obj in enumerate(rp):
        centroid = obj.centroid
        xc, yc = centroid[-2:]
        IDs[i] = obj.label
        xx_centroid[i] = xc
        yy_centroid[i] = yc
        if len(centroid) == 3:
            zc = centroid[0]
            zz_centroid[i] = zc
            
    df = pd.DataFrame(
        {
            'Cell_ID': IDs,
            'is_cell_dead': zeros_list,
            'is_cell_excluded': zeros_list,
            'x_centroid': xx_centroid,
            'y_centroid': yy_centroid,
            'was_manually_edited': minus1_list
        }
    ).set_index('Cell_ID')
    if len(centroid) == 3:
        df['z_centroid'] = zz_centroid
        
    return df


def getDefault_SegmInfo_df(posData, filename):
    mid_slice = int(posData.SizeZ/2)
    df = pd.DataFrame({
        'filename': [filename]*posData.SizeT,
        'frame_i': range(posData.SizeT),
        'z_slice_used_dataPrep': [mid_slice]*posData.SizeT,
        'which_z_proj': ['single z-slice']*posData.SizeT,
        'z_slice_used_gui': [mid_slice]*posData.SizeT,
        'which_z_proj_gui': ['single z-slice']*posData.SizeT,
        'resegmented_in_gui': [False]*posData.SizeT,
        'is_from_dataPrep': [False]*posData.SizeT
    }).set_index(['filename', 'frame_i'])
    return df


def _relabel_cca_dfs_and_segm_data(
        cca_dfs,
        IDs_mapper,
        asymm_tracked_segm,
        progressbar=True,
    ):
    # Rename Cell_ID index according to asymmetric cell div convention
    if progressbar:
        pbar = tqdm(
            desc='Applying asymmetric division', 
            total=len(IDs_mapper), ncols=100
        )
    for key, (root_ID, parent_ID) in IDs_mapper.items():
        div_frame_i, daughter_ID = key
        for frame_i in range(div_frame_i, len(asymm_tracked_segm)):
            
            
            lab = asymm_tracked_segm[frame_i]
            rp = skimage.measure.regionprops(lab)
            rp_mapper = {obj.label: obj for obj in rp}
            obj_daught = rp_mapper.get(daughter_ID)
            mother_ID = root_ID if rp_mapper.get(root_ID) is None else parent_ID
            
            cca_dfs[frame_i].rename(
                index={daughter_ID: mother_ID}, inplace=True
            )
            
            if obj_daught is None:
                continue
    
            lab[obj_daught.slice][obj_daught.image] = mother_ID
        
        if progressbar:
            pbar.update()
    
    if progressbar:
        pbar.close()

def df_ctc_to_acdc_df(
        df_ctc, tracked_segm, cell_division_mode='Normal', return_list=False, 
        progressbar=True
    ):
    """Convert Cell Tracking Challenge DataFrame with annotated division to
    Cell-ACDC cell cycle annotations DataFrame.

    Parameters
    ----------
    df_ctc : pd.DataFrame
        DataFrame with {'label', 't1', 't2', 'parent'} columns where 
        't1' is the frame index of cell division.
    tracked_segm : (T, Y, X) array of ints
        Array of tracked segmentation labels.
    cell_division_mode : {'Normal', 'Asymmetric'}, optional
        Type of cell division. `Normal` is the standard cell division, 
        where the mother cell divides into two daughter cells. For the 
        tracking, that means the two daughter cells get a new, unique ID 
        each. 
        
        `Asymmetric` means that the mother cell grows one daughter 
        cell that eventually divides from the mother (e.g., budding yeast). 
        For the tracking, this means that the mother cell ID keeps 
        existing after division and the daughter cell gets a new, unique ID.
        
        If `Asymmetric`, the third returned element is the segmentation data 
        with the asymmetric Cell IDs.  
    return_list : bool, optional
        If `True`, the second returned element is the list of created dataframes, 
        one per frame. Default is False
    progressbar : bool, optional
        If `True`, displays a tqdm progressbar. Default is True
    """    
    cca_dfs = []
    keys = []
    df_ctc = df_ctc.set_index(['t1', 'parent'])
    
    if cell_division_mode == 'Asymmetric':
        asymm_tracked_segm = tracked_segm.copy()
    
    asymmetric_IDs_rename_mapper = {}
    if progressbar:
        pbar = tqdm(
            desc='Converting to Cell-ACDC format', 
            total=len(tracked_segm), ncols=100
        )
    for frame_i, lab in enumerate(tracked_segm):
        rp = skimage.measure.regionprops(lab)
        IDs = [obj.label for obj in rp]
        cca_df = core.getBaseCca_df(IDs, with_tree_cols=True)
        keys.append(frame_i)
        if frame_i == 0:
            cca_dfs.append(cca_df)
            if progressbar:
                pbar.update()
            continue
        
        # Copy annotations from previous frames
        prev_cca_df = cca_dfs[frame_i-1]
        old_IDs = cca_df.index.intersection(prev_cca_df.index)
        cca_df.loc[old_IDs] = prev_cca_df.loc[old_IDs]
        
        try:
            df_ctc_i = df_ctc.loc[frame_i]
        except KeyError as err:
            # No division detected --> nothing to annotate
            cca_dfs.append(cca_df)
            if progressbar:
                pbar.update()
            continue
        
        for parent_ID, df_ctc_i_pID in df_ctc_i.groupby(level=0):
            daughter_IDs = df_ctc_i_pID['label'].to_list()     
            
            if parent_ID == 0:
                continue
            
            cca_df.loc[daughter_IDs, 'parent_ID_tree'] = parent_ID
            cca_df.loc[daughter_IDs, 'emerg_frame_i'] = frame_i
            cca_df.loc[daughter_IDs, 'division_frame_i'] = frame_i
            
            root_ID = prev_cca_df.at[parent_ID, 'root_ID_tree']
            if root_ID == -1:
                root_ID = parent_ID
            cca_df.loc[daughter_IDs, 'root_ID_tree'] = root_ID
            
            cca_df.loc[daughter_IDs[0], 'sister_ID_tree'] = daughter_IDs[1]
            cca_df.loc[daughter_IDs[1], 'sister_ID_tree'] = daughter_IDs[0]
            
            prev_gen_num = prev_cca_df.loc[parent_ID, 'generation_num_tree']
            cca_df.loc[daughter_IDs, 'generation_num_tree'] = prev_gen_num + 1
            
            # Annotate division from df_ctc_i into 
            if cell_division_mode == 'Asymmetric':
                # Recycle the root_ID and assign it to one of the daughters
                replaced_daught_ID = daughter_IDs[1]
                key = (frame_i, replaced_daught_ID)
                asymmetric_IDs_rename_mapper[key] = (root_ID, parent_ID)    
        
        cca_dfs.append(cca_df)
        
        if progressbar:
            pbar.update()
    
    if progressbar:
        pbar.close()
    
    if asymmetric_IDs_rename_mapper:
        _relabel_cca_dfs_and_segm_data(
            cca_dfs,
            asymmetric_IDs_rename_mapper,
            asymm_tracked_segm,
            progressbar=True,
        )
    
    cca_df = pd.concat(cca_dfs, keys=keys, names=['frame_i'])
    
    out = [cca_df, None, None]
    
    if return_list:
        out[1] = cca_dfs
        
    if cell_division_mode == 'Asymmetric':
        out[2] = asymm_tracked_segm
    
    return out


def get_empty_stored_data_dict():
    return {
            'regionprops': None,
            'labels': None,
            'acdc_df': None,
            'delROIs_info': {
                    'rois': [], 'delMasks': [], 'delIDsROI': [], 'state': [],
                    'delMasksCoords': []
                    
                },
            'IDs': [],
            'manually_edited_lab': {'lab': {}, 'zoom_slice': None},
            'single_moth_bud_pair_cca': None,
        }
