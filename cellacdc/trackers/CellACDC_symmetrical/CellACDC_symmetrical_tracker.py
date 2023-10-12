import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame, calc_IoA_matrix

import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
from cellacdc import printl

def ident_no_mothers(IoA_matrix, IoA_thresh_daughter=0.4, min_children=2, max_daughter=2):
    # Find cells which dont have several bad overlaps in next frame, implying they have not split  
    aggr_track = []
    daughter_range = range(min_children, max_daughter+1, 1)
    IoA_thresholded = IoA_matrix >= IoA_thresh_daughter

    for i in range(IoA_matrix.shape[0]):
        high_IoA_indices = np.where(IoA_thresholded[i])[0]
        
        if not high_IoA_indices.size:
            continue
        elif not len(high_IoA_indices) in daughter_range:
            aggr_track.append(i)

    return aggr_track



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

            IoA_thresh = self.params.get('IoA_thresh', 0.8)
            IoA_thresh_daughter = self.params.get('IoA_thresh_daughter', 0.45)
            IoA_thresh_aggr = self.params.get('IoA_thresh_aggr', 0.5)
            Min_daughter= self.params.get('Min_daughters', 2)
            Max_daughter = self.params.get('Max_daughters', 2)


            IoA_matrix, IDs_curr_untracked, IDs_prev = calc_IoA_matrix(lab, prev_lab, rp, prev_rp)
            aggr_track = ident_no_mothers(IoA_matrix, IoA_thresh_daughter=IoA_thresh_daughter, min_children=Min_daughter, max_daughter=Max_daughter)
            printl(f'Frame: {frame_i}, No mothers: {len(aggr_track)}, Mothers: {IoA_matrix.shape[0]-len(aggr_track)}')
            tracked_lab = track_frame(
                prev_lab, prev_rp, lab, rp, IoA_thresh=IoA_thresh,IoA_matrix=IoA_matrix, aggr_track=aggr_track, IoA_thresh_aggr=IoA_thresh_aggr, IDs_curr_untracked=IDs_curr_untracked, IDs_prev=IDs_prev
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

        signals.progressBar.emit(1)

    def save_output(self):
        pass



# IoA_thresh_daughter = 0.45
# max_children = 2


# def find_daughters(
#     IoA_matrix, IDs_prev, tracked_IDs, daughter_threshold=0.45, max_children=2):
    
#     #Find indexes which did not to track
#     column_indexes_in_list = np.in1d(np.arange(IoA_matrix.shape[1]), tracked_IDs)
#     columns_not_in_list = ~np.any(IoA_matrix[:, column_indexes_in_list], axis=1)

#     if rows_not_in_list == []: #no need to do smth when all was tracked
#         return
    
#     mother, daughters = CellACDC_tracker.assign(
#         IoA_matrix, columns_not_in_list, IDs_prev, IoA_thresh=daughter_threshold, multiple=True, max_children=max_children)
#     return mother, daughters


# IoA_matrix, IDs_curr_untracked, IDs_prev = CellACDC_tracker.calc_IoA_matrix(lab, prev_lab, rp, prev_rp)

# old_IDs, tracked_IDs = CellACDC_tracker.assign(IoA_matrix, IDs_curr_untracked, IDs_prev, IoA_thresh=0.7)

# mother, daughters = find_daughters(
#     IoA_matrix, IDs_prev, tracked_IDs, daughter_threshold=IoA_thresh_daughter, max_children=max_children)

# CellACDC_tracker.indexAssignment(old_IDs, tracked_IDs, IDs_curr_untracked, lab, rp, uniqueID, remove_untracked=False, assign_unique_new_IDs=True)