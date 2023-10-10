from cellacdc.trackers.CellACDC import CellACDC_tracker
import numpy as np


IoA_thresh_daughter = 0.45
max_children = 2


def find_daughters(
    IoA_matrix, IDs_prev, tracked_IDs, daughter_threshold=0.45, max_children=2):
    
    #Find indexes which did not to track
    column_indexes_in_list = np.in1d(np.arange(IoA_matrix.shape[1]), tracked_IDs)
    columns_not_in_list = ~np.any(IoA_matrix[:, column_indexes_in_list], axis=1)

    if rows_not_in_list == []: #no need to do smth when all was tracked
        return
    
    mother, daughters = CellACDC_tracker.assign(
        IoA_matrix, columns_not_in_list, IDs_prev, IoA_thresh=daughter_threshold, multiple=True, max_children=max_children)
    return mother, daughters


IoA_matrix, IDs_curr_untracked, IDs_prev = CellACDC_tracker.calc_IoA_matrix(lab, prev_lab, rp, prev_rp)

old_IDs, tracked_IDs = CellACDC_tracker.assign(IoA_matrix, IDs_curr_untracked, IDs_prev, IoA_thresh=0.7)

mother, daughters = find_daughters(
    IoA_matrix, IDs_prev, tracked_IDs, daughter_threshold=IoA_thresh_daughter, max_children=max_children)

CellACDC_tracker.indexAssignment(old_IDs, tracked_IDs, IDs_curr_untracked, lab, rp, uniqueID, remove_untracked=False, assign_unique_new_IDs=True)