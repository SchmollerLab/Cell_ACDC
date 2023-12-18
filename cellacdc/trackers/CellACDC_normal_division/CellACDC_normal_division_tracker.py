import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame, calc_IoA_matrix
from cellacdc.core import printl
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd

def ident_no_mothers(IoA_matrix, IoA_thresh_daughter, min_daughter, max_daughter):
    # Find cells which dont have several bad overlaps in next frame, implying they have not split  
    mother_daughters = []
    aggr_track = []
    daughter_range = range(min_daughter, max_daughter+1, 1)

    IoA_thresholded = IoA_matrix >= IoA_thresh_daughter
    nrows, ncols = IoA_matrix.shape
    for j in range(ncols):
        high_IoA_indices = np.where(IoA_thresholded[:, j])[0]
        
        if not high_IoA_indices.size:
            continue
        elif not len(high_IoA_indices) in daughter_range:
            aggr_track.extend(high_IoA_indices)
        else:
            mother_daughters.append((j, high_IoA_indices))

    # print(f"{IoA_thresholded}, Mothers: {mothers}, AggTrack: {aggr_track}")
    return aggr_track, mother_daughters

class tracker:
    def __init__(self):
        pass

    def track(
            self, segm_video,
            signals=None,
            IoA_thresh = 0.8,
            IoA_thresh_daughter = 0.25,
            IoA_thresh_aggressive = 0.5,
            min_daughter = 2,
            max_daughter = 2,
            record_lineage = True
        ):
        tracked_video = np.zeros_like(segm_video)
        pbar = tqdm(total=len(segm_video), desc='Tracking', ncols=100)

        # if record_lineage == True:
        #     lineage_tree = []
            
        for frame_i, lab in enumerate(segm_video):
            added_lineage_tree = []


            if frame_i == 0:
                tracked_video[frame_i] = lab

                if record_lineage == True:
                    rp = regionprops(lab.copy())
                    families = []
                    for obj in rp:
                        label = obj.label
                        families.append([(label, 0)])
                        added_lineage_tree.append((0, 1, label, -1, -2, label, ""))

                lineage_list = [pd.DataFrame(added_lineage_tree, columns=['Frame', 'Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs'])]
                pbar.update()
                continue

            prev_lab = tracked_video[frame_i-1]

            prev_rp = regionprops(prev_lab)
            rp = regionprops(lab.copy())

            IoA_matrix, IDs_curr_untracked, IDs_prev = calc_IoA_matrix(lab, prev_lab, rp, prev_rp)
            aggr_track, mother_daughters = ident_no_mothers(IoA_matrix, IoA_thresh_daughter=IoA_thresh_daughter, min_daughter=min_daughter, max_daughter=max_daughter)

            # print(f'Frame: {frame_i}, {len(IoA_matrix) - len(aggr_track)}')                  

            tracked_lab, IoA_matrix, assignments = track_frame(
                prev_lab, prev_rp, lab, rp, IoA_thresh=IoA_thresh,IoA_matrix=IoA_matrix, aggr_track=aggr_track, IoA_thresh_aggr=IoA_thresh_aggressive, IDs_curr_untracked=IDs_curr_untracked, IDs_prev=IDs_prev, return_all=True
            )
            # printl(f'Frame: {frame_i}, No mothers: {aggr_track} IoA_matrix: \n{IoA_matrix}')
            tracked_video[frame_i] = tracked_lab

            if record_lineage and mother_daughters:


                for mother, daughters in mother_daughters:
                    mother_ID = IDs_prev[mother]
                    daughter_IDs = []

                    for daughter in daughters:
                        # printl(frame_i, assignments, IDs_curr_untracked)
                        daughter_IDs.append(assignments[IDs_curr_untracked[daughter]])

                    for family in families:
                        for member in family:
                            # printl(families)
                            if mother_ID == member[0]:
                                origin_id = family[0][0]
                                generation = member[1] + 1
                                family.extend([(daughter_ID, generation) for daughter_ID in daughter_IDs])
                                break
                        else: 
                            continue
                        break

                    for i, daughter_ID in enumerate(daughter_IDs):
                        daughter_IDs_copy = daughter_IDs.copy()
                        daughter_IDs_copy.pop(i)
                        added_lineage_tree.append((frame_i, daughter_ID, mother_ID, generation, origin_id, daughter_IDs_copy))

            added_lineage_tree = pd.DataFrame(added_lineage_tree, columns=['Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs'])
            lineage_tree_frame = pd.concat([lineage_list[-1][['Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs']], added_lineage_tree], axis=0)
            lineage_tree_frame['Frame'] = frame_i
            lineage_tree_frame = lineage_tree_frame[['Frame', 'Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs']]
            lineage_list.append(lineage_tree_frame)

            self.updateGuiProgressBar(signals)
            pbar.update()

        if record_lineage:
            df_final = pd.concat(lineage_list)
            df_final.to_csv('lineage_tree.csv', index=False)

        pbar.close()
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