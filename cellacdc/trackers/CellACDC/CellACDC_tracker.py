import os
from typing import List

from tqdm import tqdm

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

from cellacdc import core, printl

DEBUG = False

def calc_IoA_matrix(lab, prev_lab, rp, prev_rp, IDs_curr_untracked=None):
    IDs_prev = []
    if IDs_curr_untracked is None:
        IDs_curr_untracked = [obj.label for obj in rp]

    IoA_matrix = np.zeros((len(rp), len(prev_rp)))

    # For each ID in previous frame get IoA with all current IDs
    # Rows: IDs in current frame, columns: IDs in previous frame
    for j, obj_prev in enumerate(prev_rp):
        ID_prev = obj_prev.label
        A_IDprev = obj_prev.area
        IDs_prev.append(ID_prev)
        mask_ID_prev = prev_lab==ID_prev
        intersect_IDs, intersects = np.unique(
            lab[mask_ID_prev], return_counts=True
        )
        for intersect_ID, I in zip(intersect_IDs, intersects):
            if intersect_ID != 0:
                i = IDs_curr_untracked.index(intersect_ID)
                IoA = I/A_IDprev
                IoA_matrix[i, j] = IoA
    return IoA_matrix, IDs_curr_untracked, IDs_prev

def assign(IoA_matrix, IDs_curr_untracked, IDs_prev, IoA_thresh=0.4, aggr_track=None, IoA_thresh_aggr=0.4, daughters_list=None):
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
        return_assignments=False
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

    Returns
    -------
    tracked_lab : (Y, X) or (Z, Y, X) array of ints
        Segmentation masks with IDs replaced according to input tracking 
        information.
    assignments: dict
        Returned only if `return_assignments` is True.
    """    
    log_debugging(
        'start', 
        IDs_curr_untracked=IDs_curr_untracked,
        old_IDs=old_IDs
    )
    
    # Replace untracked IDs with tracked IDs and new IDs with increasing num
    new_untracked_IDs = [ID for ID in IDs_curr_untracked if ID not in old_IDs]
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
        core.lab_replace_values(
            tracked_lab, rp, old_IDs, tracked_IDs, in_place=True
        )
        assignments.update(dict(zip(old_IDs, tracked_IDs)))
        log_debugging(
            'tracked', 
            tracked_IDs=tracked_IDs,
            old_IDs=old_IDs,
        )

    if not return_assignments:
        return tracked_lab
    else: 
        return tracked_lab, assignments

def track_frame(
        prev_lab, prev_rp, lab, rp, IDs_curr_untracked=None,
        uniqueID=None, setBrushID_func=None, posData=None,
        assign_unique_new_IDs=True, IoA_thresh=0.4, debug=False,
        return_all=False, aggr_track=None, IoA_matrix=None, 
        IoA_thresh_aggr=None, IDs_prev=None, return_prev_IDs=False,
        mother_daughters=None
    ):
    if not np.any(lab):
        # Skip empty frames
        return lab

    if IoA_matrix is None:
        IoA_matrix, IDs_curr_untracked, IDs_prev = calc_IoA_matrix(
            lab, prev_lab, rp, prev_rp, IDs_curr_untracked=IDs_curr_untracked
        )

    daughters_list = []
    if mother_daughters:
        for _, daughters in mother_daughters:
            daughters_list.extend(daughters)

    old_IDs, tracked_IDs = assign(
        IoA_matrix, IDs_curr_untracked, IDs_prev,
        IoA_thresh=IoA_thresh, aggr_track=aggr_track, 
        IoA_thresh_aggr=IoA_thresh_aggr, daughters_list=daughters_list
    )

    if posData is None and uniqueID is None:
        uniqueID = max(
            (max(IDs_prev, default=0), max(IDs_curr_untracked, default=0))
        ) + 1
    elif uniqueID is None:
        # Compute starting unique ID
        setBrushID_func(useCurrentLab=True)
        uniqueID = posData.brushID+1

    if not return_all:
        tracked_lab = indexAssignment(
            old_IDs, tracked_IDs, IDs_curr_untracked,
            lab.copy(), rp, uniqueID,
            assign_unique_new_IDs=assign_unique_new_IDs
        )
    else:
        tracked_lab, assignments = indexAssignment(
            old_IDs, tracked_IDs, IDs_curr_untracked,
            lab.copy(), rp, uniqueID,
            assign_unique_new_IDs=assign_unique_new_IDs, 
            return_assignments=return_all
        )

    # old_new_ids = dict(zip(old_IDs, tracked_IDs)) # for now not used, but could be useful in the future
    
    if return_all:
        return tracked_lab, IoA_matrix, assignments, tracked_IDs # remove tracked_IDs and change code in CellACDC_tracker.py if causing problems
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