import os
from typing import List

from tqdm import tqdm

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

from cellacdc import core, printl, debugutils

try:
    from cellacdc.precompiled_functions import (
        calc_IoA_matrix_2D as _calc_IoA_matrix_2D_cython,
        calc_IoA_matrix_3D as _calc_IoA_matrix_3D_cython,
    )
    _HAS_CYTHON_IOA = True
except ImportError:
    _HAS_CYTHON_IOA = False

DEBUG = False

def _normalize_specific_IDs(specific_IDs):
    if specific_IDs is None:
        return None
    if isinstance(specific_IDs, (list, tuple, set, np.ndarray)):
        return set(specific_IDs)
    return {specific_IDs}

def _filter_subset_assignments(old_IDs, tracked_IDs, all_curr_IDs, specific_IDs):
    if specific_IDs is None:
        return old_IDs, tracked_IDs

    selected_curr_IDs = set(specific_IDs)
    other_curr_IDs = set(all_curr_IDs).difference(selected_curr_IDs)
    filtered_old_IDs = []
    filtered_tracked_IDs = []
    for old_ID, tracked_ID in zip(old_IDs, tracked_IDs):
        if tracked_ID in other_curr_IDs:
            continue
        filtered_old_IDs.append(old_ID)
        filtered_tracked_IDs.append(tracked_ID)

    return filtered_old_IDs, filtered_tracked_IDs

@debugutils.line_benchmark
def calc_Io_matrix(lab, prev_lab, rp, prev_rp, IDs_curr_untracked=None,
                   specific_IDs=None,
                   denom:str='area_prev'):
    specific_IDs = _normalize_specific_IDs(specific_IDs)
    if IDs_curr_untracked is None:
        IDs_curr_untracked = [obj.label for obj in rp]
    elif not isinstance(IDs_curr_untracked, list):
        IDs_curr_untracked = list(IDs_curr_untracked)

    if specific_IDs is not None:
        IDs_curr_untracked = [
            ID for ID in IDs_curr_untracked if ID in specific_IDs
        ]

    IDs_prev = [obj.label for obj in prev_rp]

    if not IDs_curr_untracked:
        return np.zeros((0, len(prev_rp))), IDs_curr_untracked, IDs_prev

    if denom not in ('area_prev', 'union'):
        raise ValueError(
            "Invalid denom value. Use 'area_prev' or 'union'."
        )

    if _HAS_CYTHON_IOA:
        use_union = denom == 'union'
        curr_IDs_arr  = np.array(IDs_curr_untracked, dtype=np.uint32)
        prev_IDs_arr  = np.array(IDs_prev,           dtype=np.uint32)
        prev_areas_arr = np.array([obj.area for obj in prev_rp], dtype=np.uint32)
        if use_union:
            rp_mapper = {obj.label: obj for obj in rp}
            curr_areas_arr = np.array(
                [rp_mapper[ID].area for ID in IDs_curr_untracked], dtype=np.uint32
            )
        else:
            curr_areas_arr = np.empty(0, dtype=np.uint32)
        lab_u32      = np.asarray(lab,      dtype=np.uint32)
        prev_lab_u32 = np.asarray(prev_lab, dtype=np.uint32)
        if lab.ndim == 2:
            IoA_matrix = _calc_IoA_matrix_2D_cython(
                lab_u32, prev_lab_u32, curr_IDs_arr, prev_IDs_arr,
                prev_areas_arr, curr_areas_arr, use_union,
            )
        else:
            IoA_matrix = _calc_IoA_matrix_3D_cython(
                lab_u32, prev_lab_u32, curr_IDs_arr, prev_IDs_arr,
                prev_areas_arr, curr_areas_arr, use_union,
            )
        return IoA_matrix, IDs_curr_untracked, IDs_prev

    # --- pure-Python fallback (used when Cython extension is not compiled) ---
    IoA_matrix = np.zeros((len(IDs_curr_untracked), len(prev_rp)))
    rp_mapper = {obj.label: obj for obj in rp}
    idx_mapper = {ID: i for i, ID in enumerate(IDs_curr_untracked)}
    for j, obj_prev in enumerate(prev_rp):
        if denom == 'area_prev':
            denom_val = obj_prev.area
        intersect_IDs, intersects = np.unique(
            lab[obj_prev.slice][obj_prev.image], return_counts=True
        )
        for intersect_ID, I in zip(intersect_IDs, intersects):
            if intersect_ID == 0 or I == 0:
                continue
            if denom == 'union':
                if intersect_ID not in rp_mapper:
                    continue
                obj_curr = rp_mapper[intersect_ID]
                denom_val = obj_prev.area + obj_curr.area - I
                if denom_val == 0:
                    continue
            idx = idx_mapper.get(intersect_ID)
            if idx is None:
                continue
            IoA_matrix[idx, j] = I / denom_val
    return IoA_matrix, IDs_curr_untracked, IDs_prev

def assign(
        IoA_matrix, IDs_curr_untracked, IDs_prev, IoA_thresh=0.4, 
        aggr_track=None, IoA_thresh_aggr=0.4, daughters_list=None,
        specific_IDs=None):
    # Determine max IoA between IDs and assign tracked ID if IoA >= IoA_thresh
    if IoA_matrix.size == 0:
        return [], []
        
    max_IoA_col_idx = IoA_matrix.argmax(axis=1)
    unique_col_idx, counts = np.unique(max_IoA_col_idx, return_counts=True)
    counts_dict = dict(zip(unique_col_idx, counts))
    tracked_IDs = []
    old_IDs = []

    if DEBUG:
        printl(f'IDs in previous frame: {IDs_prev}')

    for i, j in enumerate(max_IoA_col_idx):
        if daughters_list is not None:
            if i in daughters_list:
                continue

        if aggr_track:
            if i in aggr_track:
                IoA_thresh_temp = IoA_thresh_aggr
            else:
                IoA_thresh_temp = IoA_thresh
        else:
            IoA_thresh_temp = IoA_thresh
        max_IoU = IoA_matrix[i,j]
        count = counts_dict[j]
        if max_IoU >= IoA_thresh_temp:
            tracked_ID = IDs_prev[j]
            if count == 1:
                old_ID = IDs_curr_untracked[i]
            elif count > 1:
                old_ID_idx = IoA_matrix[:,j].argmax()
                old_ID = IDs_curr_untracked[old_ID_idx]
            tracked_IDs.append(tracked_ID)
            old_IDs.append(old_ID)

    return old_IDs, tracked_IDs

def log_debugging(what, **kwargs):
    if not DEBUG:
        return
    
    if what == 'start':
        printl('----------------START INDEX ASSIGNMENT----------------')
        printl(
            f'Current IDs: {kwargs["IDs_curr_untracked"]}\n'
            f'Previous IDs: {kwargs["old_IDs"]}'
        )
    if what == 'assign_unique':
        assign_unique_new_IDs = kwargs['assign_unique_new_IDs']
        txt = (
            f'Assign new IDs uniquely = {assign_unique_new_IDs}'
        )
        printl(txt)
    elif what == 'new_untracked_and_assign_unique':
        new_untracked_IDs = kwargs['new_untracked_IDs']
        new_tracked_IDs = kwargs['new_tracked_IDs']
        IDs_curr_untracked = kwargs['IDs_curr_untracked']
        old_IDs = kwargs['old_IDs']
        txt = (
            f'Current IDs: {IDs_curr_untracked}\n'
            f'Previous IDs: {old_IDs}\n'
            f'New objects that get a new big ID: {new_untracked_IDs}\n'
            f'New unique IDs for the new objects: {new_tracked_IDs}'
        )
        printl(txt)
        txt = ''
        for _ID, replacingID in zip(new_untracked_IDs, new_tracked_IDs):
            txt = f'{txt}{_ID} --> {replacingID}\n'
        printl(txt)
    elif what == 'new_untracked_and_tracked':
        new_untracked_IDs = kwargs['new_untracked_IDs']
        new_tracked_IDs = kwargs['new_tracked_IDs']
        new_IDs_in_trackedIDs = kwargs['new_IDs_in_trackedIDs']
        old_IDs = kwargs['old_IDs']
        txt = (
            f'New tracked IDs that already exists: {new_IDs_in_trackedIDs}\n'
            f'Previous IDs: {old_IDs}\n'
            f'New objects that get a new big ID: {new_untracked_IDs}\n'
            f'New unique IDs for the new objects: {new_tracked_IDs}'
        )
        printl(txt)
        txt = ''
        for _ID, replacingID in zip(new_IDs_in_trackedIDs, new_tracked_IDs):
            txt = f'{txt}{_ID} --> {replacingID}\n'
        printl(txt)
    elif what == 'tracked':
        old_IDs = kwargs['old_IDs']
        tracked_IDs = kwargs['tracked_IDs']
        txt = (
            f'Old IDs to be tracked: {old_IDs}\n'
            f'New IDs replacing old IDs: {tracked_IDs}'
        )
        printl(txt)
        txt = ''
        for _ID, replacingID in zip(old_IDs, tracked_IDs):
            txt = f'{txt}{_ID} --> {replacingID}\n'
        printl(txt)

def indexAssignment(
        old_IDs: List[int], 
        tracked_IDs: List[int], 
        IDs_curr_untracked: List[int], 
        lab: 'np.ndarray[int]', 
        rp: 'regionprops', 
        uniqueID: int,
        remove_untracked=False, 
        assign_unique_new_IDs=True, 
        return_assignments=False,
        dont_return_tracked_lab=False,
        specific_IDs=None,
        all_curr_IDs=None,
        IDs=None,
    ):
    """Replace `old_IDs` in `lab` with `tracked_IDs` while making sure to 
    avoid merging IDs.

    Parameters
    ----------
    old_IDs : list of ints
        IDs that must be replaced with `tracked_IDs`
    tracked_IDs : list of ints
        IDs that replace `old_IDs`
    IDs_curr_untracked : list of ints
        All IDs in `lab` (including IDs that are not tracked)
    lab : (Y, X) or (Z, Y, X) array of ints
        Segmentation masks with `IDs_curr_untracked` objects
    rp : list of skimage.measure._regionprops.RegionProperties
        List of RegionProperties of the objects in `lab`
    uniqueID : int
        Starting unique ID that is going to replace those objects whose ID is 
        not tracked but they might require a new (unique) one to avoid merging.
    remove_untracked : bool, optional
        If True, those objects that were not tracked will be removed. 
        Default is False
    assign_unique_new_IDs : bool, optional
        If True, uses `uniqueID` to replace the ID of the untracked objects. 
        Default is True
    return_assignments : bool, optional
        If True, returns a dictionary where the keys are the untracked 
        IDs and the values are the unique IDs that replaced untracked IDs. 
        Default is False
    IDs : list of ints, optional
        IDs to be used for the calculation of the IoA matrix. If None,
        all IDs in `lab` are used. Default is None.

    Returns
    -------
    tracked_lab : (Y, X) or (Z, Y, X) array of ints
        Segmentation masks with IDs replaced according to input tracking 
        information.
    assignments: dict
        Returned only if `return_assignments` is True.
    """    
    specific_IDs = _normalize_specific_IDs(specific_IDs)
    log_debugging(
        'start', 
        IDs_curr_untracked=IDs_curr_untracked,
        old_IDs=old_IDs
    )
    
    if all_curr_IDs is None:
        all_curr_IDs = list(IDs_curr_untracked)
    old_IDs, tracked_IDs = _filter_subset_assignments(
        old_IDs, tracked_IDs, all_curr_IDs, specific_IDs
    )

    # Replace untracked IDs with tracked IDs and new IDs with increasing num.
    # When tracking only a subset of current IDs, leave unrelated labels untouched.
    new_untracked_IDs = [ID for ID in IDs_curr_untracked if ID not in old_IDs]
    
    if not dont_return_tracked_lab:
        tracked_lab = lab
    assignments = {}
    log_debugging(
        'assign_unique', 
        assign_unique_new_IDs=assign_unique_new_IDs
    )
    if new_untracked_IDs and assign_unique_new_IDs:
        # Relabel new untracked IDs (i.e., new cells) unique IDs
        if remove_untracked:
            new_tracked_IDs = [0]*len(new_untracked_IDs)
        else:
            new_tracked_IDs = [
                uniqueID+i for i in range(len(new_untracked_IDs))
            ]
        if not dont_return_tracked_lab:
            core.lab_replace_values(
                tracked_lab, rp, new_untracked_IDs, new_tracked_IDs
            )
        assignments.update(dict(zip(new_untracked_IDs, new_tracked_IDs)))
        log_debugging(
            'new_untracked_and_assign_unique', 
            IDs_curr_untracked=IDs_curr_untracked,
            old_IDs=old_IDs,
            new_untracked_IDs=new_untracked_IDs,
            new_tracked_IDs=new_tracked_IDs
        )
    elif new_untracked_IDs and tracked_IDs:
        # If we don't replace unique new IDs we check that tracked IDs are
        # not already existing to avoid duplicates
        new_IDs_in_trackedIDs = [
            ID for ID in new_untracked_IDs if ID in tracked_IDs
        ]
        new_tracked_IDs = [
            uniqueID+i for i in range(len(new_IDs_in_trackedIDs))
        ]
        if not dont_return_tracked_lab:
            core.lab_replace_values(
                tracked_lab, rp, new_IDs_in_trackedIDs, new_tracked_IDs
            )
        assignments.update(dict(zip(new_IDs_in_trackedIDs, new_tracked_IDs)))
        log_debugging(
            'new_untracked_and_tracked', 
            new_IDs_in_trackedIDs=new_IDs_in_trackedIDs,
            old_IDs=old_IDs,
            new_untracked_IDs=new_untracked_IDs,
            new_tracked_IDs=new_tracked_IDs
        )
    if tracked_IDs:
        if not dont_return_tracked_lab:
            core.lab_replace_values(
                tracked_lab, rp, old_IDs, tracked_IDs, in_place=True
            )
        assignments.update({
            old_ID: tracked_ID
            for old_ID, tracked_ID in zip(old_IDs, tracked_IDs)
            if old_ID != tracked_ID
        })
        log_debugging(
            'tracked', 
            tracked_IDs=tracked_IDs,
            old_IDs=old_IDs,
        )

    if not return_assignments:
        return tracked_lab
    elif dont_return_tracked_lab:
        return assignments
    else: 
        return tracked_lab, assignments

def track_frame(
        prev_lab, prev_rp, lab, rp, IDs_curr_untracked=None,
        unique_ID=None, setBrushID_func=None, posData=None,
        assign_unique_new_IDs=True, IoA_thresh=0.4, debug=False,
        return_all=False, aggr_track=None, IoA_matrix=None, 
        IoA_thresh_aggr=None, IDs_prev=None, return_prev_IDs=False,
        mother_daughters=None, denom_overlap_matrix = 'area_prev',
        return_assignments=False, specific_IDs=None, dont_return_tracked_lab=False
    ):
    if not np.any(lab):
        # Skip empty frames
        return lab

    all_curr_IDs = (
        list(IDs_curr_untracked)
        if IDs_curr_untracked is not None else [obj.label for obj in rp]
    )

    if IoA_matrix is None:
        IoA_matrix, tracked_curr_IDs, IDs_prev = calc_Io_matrix(
            lab, prev_lab, rp, prev_rp, IDs_curr_untracked=IDs_curr_untracked,
            denom=denom_overlap_matrix,specific_IDs=specific_IDs,
        )
    else:
        tracked_curr_IDs = IDs_curr_untracked

    daughters_list = []
    if mother_daughters:
        for _, daughters in mother_daughters:
            daughters_list.extend(daughters)

    old_IDs, tracked_IDs = assign(
        IoA_matrix, tracked_curr_IDs, IDs_prev,
        IoA_thresh=IoA_thresh, aggr_track=aggr_track, 
        IoA_thresh_aggr=IoA_thresh_aggr, daughters_list=daughters_list,
        specific_IDs=specific_IDs,
    )
    
    if posData is None and unique_ID is None:
        unique_ID = max(
            (max(IDs_prev, default=0), max(all_curr_IDs, default=0))
        ) + 1
    elif unique_ID is None:
        # Compute starting unique ID
        setBrushID_func(useCurrentLab=True)
        unique_ID = posData.brushID+1

    if not return_all and not return_assignments:
        tracked_lab = indexAssignment(
            old_IDs, tracked_IDs, tracked_curr_IDs,
            lab.copy(), rp, unique_ID,
            assign_unique_new_IDs=assign_unique_new_IDs,
            specific_IDs=specific_IDs,
            all_curr_IDs=all_curr_IDs,
        )
    elif dont_return_tracked_lab:
        assignments = indexAssignment(
            old_IDs, tracked_IDs, tracked_curr_IDs,
            lab.copy(), rp, unique_ID,
            assign_unique_new_IDs=assign_unique_new_IDs, 
            return_assignments=True, specific_IDs=specific_IDs,
            dont_return_tracked_lab=True,
            all_curr_IDs=all_curr_IDs,
        )
    else:
        tracked_lab, assignments = indexAssignment(
            old_IDs, tracked_IDs, tracked_curr_IDs,
            lab.copy(), rp, unique_ID,
            assign_unique_new_IDs=assign_unique_new_IDs, 
            return_assignments=True, specific_IDs=specific_IDs,
            all_curr_IDs=all_curr_IDs,
        )

    # old_new_ids = dict(zip(old_IDs, tracked_IDs)) # for now not used, but could be useful in the future
    
    if return_all and dont_return_tracked_lab:
        # special case where we want to only get the assignments but need the rest too!
        return IoA_matrix, assignments, tracked_IDs
    elif return_all:
        return tracked_lab, IoA_matrix, assignments, tracked_IDs # remove tracked_IDs and change code in CellACDC_tracker.py if causing problems
    elif dont_return_tracked_lab:
        return assignments
    elif return_assignments:
        add_info = {
            'assignments': assignments,
        }
        return tracked_lab, add_info
    else:
        return tracked_lab

class tracker:
    def __init__(self, **params):
        self.params = params

    def track(self, segm_video, signals=None, export_to: os.PathLike=None):
        tracked_video = np.zeros_like(segm_video)
        pbar = tqdm(total=len(segm_video), desc='Tracking', ncols=100)
        for frame_i, lab in enumerate(segm_video):
            if frame_i == 0:
                tracked_video[frame_i] = lab
                pbar.update()
                continue

            prev_lab = tracked_video[frame_i-1]

            prev_rp = regionprops(prev_lab)
            rp = regionprops(lab.copy())

            IoA_thresh = self.params.get('IoA_thresh', 0.4)
            tracked_lab = track_frame(
                prev_lab, prev_rp, lab, rp, IoA_thresh=IoA_thresh
            )

            tracked_video[frame_i] = tracked_lab
            self.updateGuiProgressBar(signals)
            pbar.update()
        pbar.close()
        # tracked_video = relabel_sequential(tracked_video)[0]
        return tracked_video
    
    def updateGuiProgressBar(self, signals):
        if signals is None:
            return
        
        if hasattr(signals, 'innerPbar_available'):
            if signals.innerPbar_available:
                # Use inner pbar of the GUI widget (top pbar is for positions)
                signals.innerProgressBar.emit(1)
                return

        if hasattr(signals, 'progressBar'):
            signals.progressBar.emit(1)

    def save_output(self):
        pass