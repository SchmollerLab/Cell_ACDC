import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import calc_IoA_matrix
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame as track_frame_base
from cellacdc.core import getBaseCca_df,printl
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd

# def reorg_daughter_cells(lineage_tree_frame, max_daughter):
#     new_columns = [f'Sister {i+1}' for i in range(max_daughter-1)]
#     sister_columns = lineage_tree_frame['Sister IDs'].apply(pd.Series)
    
#     # Explicitly convert the columns to integers while preserving NaN
#     sister_columns = sister_columns.applymap(lambda x: int(x) if not pd.isna(x) else x)
    
#     lineage_tree_frame[new_columns] = sister_columns
#     lineage_tree_frame = lineage_tree_frame.drop(columns=['Sister IDs'])

#     return lineage_tree_frame

def reorg_daughter_cells(lineage_tree_frame, max_daughter):
    new_columns = [f'sister_ID_tree_{i+1}' for i in range(max_daughter-1)]
    sister_columns = lineage_tree_frame['sister_ID_tree'].apply(pd.Series)
    
    # Explicitly convert the columns to integers while preserving NaN (I think I dont need this anymore as I changed to pd.NA)
    # sister_columns = sister_columns.applymap(lambda x: int(x) if not pd.isna(x) else x)
    
    lineage_tree_frame[new_columns] = sister_columns
    lineage_tree_frame['sister_ID_tree'] = lineage_tree_frame['sister_ID_tree'].apply(lambda x: x[0])

    return lineage_tree_frame


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

    return aggr_track, mother_daughters

def added_lineage_tree_to_cca_df(added_lineage_tree):
    #The input has following columns: 'Division frame', 'Daughter ID'("ID"), 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs'

    cca_df = getBaseCca_df([row[1] for row in added_lineage_tree], with_tree_cols=True)
    cca_df['emerg_frame_i'] = [row[0] for row in added_lineage_tree]
    cca_df['division_frame_i'] = [row[0] for row in added_lineage_tree]
    cca_df['generation_num_tree'] = [row[3] for row in added_lineage_tree]
    cca_df['parent_ID_tree'] = [row[2] for row in added_lineage_tree]
    cca_df['root_ID_tree'] = [row[4] for row in added_lineage_tree]
    cca_df['sister_ID_tree'] = [row[5] for row in added_lineage_tree]
    return cca_df

def track_frame(previous_frame_labels, current_frame_labels, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive):
    segm_video = [previous_frame_labels, current_frame_labels]
    tracker = normal_division_tracker(segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive)
    tracker.track_frame(1)
    tracked_video = tracker.get_tracked_video
    return tracked_video[-1]

class normal_division_tracker:
    def __init__(self, segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive):
        self.IoA_thresh_daughter = IoA_thresh_daughter
        self.min_daughter = min_daughter
        self.max_daughter = max_daughter
        self.IoA_thresh = IoA_thresh
        self.IoA_thresh_aggressive = IoA_thresh_aggressive
        self.segm_video = segm_video

        self.tracked_video = np.zeros_like(segm_video)
        self.tracked_video[0] = segm_video[0]

    def track_frame(self, frame_i, lab=None, prev_lab=None, rp=None, prev_rp=None):
        
        if lab is None:
            lab = self.segm_video[frame_i]

        if prev_lab is None:
            prev_lab = self.tracked_video[frame_i-1]

        if rp is None:
            self.rp = regionprops(lab.copy())
        else:
            self.rp = rp

        if prev_rp is None:
            prev_rp = regionprops(prev_lab)

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_IoA_matrix(
                                                                    lab, 
                                                                    prev_lab, 
                                                                    self.rp, 
                                                                    prev_rp
                                                                    )
        aggr_track, self.mother_daughters = ident_no_mothers(
                                                        IoA_matrix, 
                                                        IoA_thresh_daughter=self.IoA_thresh_daughter, 
                                                        min_daughter=self.min_daughter, 
                                                        max_daughter=self.max_daughter
                                                        )
        self.tracked_lab, IoA_matrix, self.assignments = track_frame_base(
                                                            prev_lab, 
                                                            prev_rp, 
                                                            lab, 
                                                            self.rp, 
                                                            IoA_thresh=self.IoA_thresh,
                                                            IoA_matrix=IoA_matrix, 
                                                            aggr_track=aggr_track, 
                                                            IoA_thresh_aggr=self.IoA_thresh_aggressive, 
                                                            IDs_curr_untracked=self.IDs_curr_untracked, 
                                                            IDs_prev=self.IDs_prev, 
                                                            return_all=True
                                                            )
        self.tracked_video[frame_i] = self.tracked_lab
        

    # def get_tracked_video(self):
    #     return self.tracked_video
        
    # def get_current_rp(self):
    #     return self.rp
    
    # def get_current_mother_daughters(self):
    #     return self.mother_daughters
    
    # def get_current_assignments(self):
    #     return self.assignments
    
    # def get_current_IDs_prev(self):
    #     return self.IDs_prev

    # def get_current_IDs_curr_untracked(self):
    #     return self.IDs_curr_untracked
    
    # def get_current_tracked_lab(self):
    #     return self.tracked_lab

class normal_division_lineage_tree:
    def __init__(self, lab, max_daughter):
        self.max_daughter = max_daughter

        self.families = []
        added_lineage_tree = []

        rp = regionprops(lab.copy())
        for obj in rp:
            label = obj.label
            self.families.append([(label, 0)])
            added_lineage_tree.append((-1, label, -1, -2, label, [-1] * (max_daughter-1)))

        cca_df = added_lineage_tree_to_cca_df(added_lineage_tree)
        self.lineage_list = [cca_df]


    def create_tracked_frame_tree(self, frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments):
        added_lineage_tree = []
        for mother, daughters in mother_daughters:
            mother_ID = IDs_prev[mother]
            daughter_IDs = []

            for daughter in daughters:
                # printl(frame_i, assignments, IDs_curr_untracked)
                daughter_IDs.append(assignments[IDs_curr_untracked[daughter]])

            for family in self.families:
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
                daughter_IDs_copy = daughter_IDs_copy + [-1] * (self.max_daughter - len(daughter_IDs_copy) -1)
                added_lineage_tree.append((frame_i, daughter_ID, mother_ID, generation, origin_id, daughter_IDs_copy))


        cca_df = added_lineage_tree_to_cca_df(added_lineage_tree)
        cca_df = pd.concat([self.lineage_list[-1], cca_df], axis=0)
        self.lineage_list.append(cca_df)

    # def get_lineage_list(self):
    #     return self.lineage_list

class tracker:
    def __init__(self):
        pass

    def track(
            self, 
            segm_video,
            signals=None,
            IoA_thresh = 0.8,
            IoA_thresh_daughter = 0.25,
            IoA_thresh_aggressive = 0.5,
            min_daughter = 2,
            max_daughter = 2,
            record_lineage = True
        ):

        pbar = tqdm(total=len(segm_video), desc='Tracking', ncols=100)
            
        for frame_i, lab in enumerate(segm_video):
            if frame_i == 0:
                tracker = normal_division_tracker(segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive)
                if record_lineage:
                    tree = normal_division_lineage_tree(lab, max_daughter)
                pbar.update()
                continue

            tracker.track_frame(frame_i)

            if record_lineage:
                mother_daughters = tracker.mother_daughters
                IDs_prev = tracker.IDs_prev
                assignments = tracker.assignments
                IDs_curr_untracked = tracker.IDs_curr_untracked
                tree.create_tracked_frame_tree(frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments)

            self.updateGuiProgressBar(signals)
            pbar.update()

        if record_lineage:
            self.cca_dfs = tree.lineage_list

        tracked_video = tracker.tracked_video            
        pbar.close()
        return tracked_video
    
    # def track(
    #         self, segm_video,
    #         signals=None,
    #         IoA_thresh = 0.8,
    #         IoA_thresh_daughter = 0.25,
    #         IoA_thresh_aggressive = 0.5,
    #         min_daughter = 2,
    #         max_daughter = 2,
    #         record_lineage = True
    #     ):

    #     tracked_video = np.zeros_like(segm_video)
    #     pbar = tqdm(total=len(segm_video), desc='Tracking', ncols=100)
            
    #     for frame_i, lab in enumerate(segm_video):
    #         added_lineage_tree = []

    #         if frame_i == 0:
    #             tracked_video[frame_i] = lab

    #             if record_lineage == True:
    #                 rp = regionprops(lab.copy())
    #                 families = []
    #                 for obj in rp:
    #                     label = obj.label
    #                     families.append([(label, 0)])
    #                     added_lineage_tree.append((0, 1, label, -1, -2, label, [-1] * (max_daughter-1)))
    #             lineage_tree_frame = pd.DataFrame(added_lineage_tree, columns=['Frame', 'Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs'])
    #             lineage_list = [lineage_tree_frame]
    #             pbar.update()
    #             continue

    #         prev_lab = tracked_video[frame_i-1]

    #         prev_rp = regionprops(prev_lab)
    #         rp = regionprops(lab.copy())

    #         IoA_matrix, IDs_curr_untracked, IDs_prev = calc_IoA_matrix(
    #                                                                     lab, 
    #                                                                     prev_lab, 
    #                                                                     rp, prev_rp
    #                                                                     )
    #         aggr_track, mother_daughters = ident_no_mothers(
    #                                                         IoA_matrix, 
    #                                                         IoA_thresh_daughter=IoA_thresh_daughter, 
    #                                                         min_daughter=min_daughter, 
    #                                                         max_daughter=max_daughter
    #                                                         )
    #         tracked_lab, IoA_matrix, assignments = track_frame(
    #                                                             prev_lab, 
    #                                                             prev_rp, 
    #                                                             lab, 
    #                                                             rp, 
    #                                                             IoA_thresh=IoA_thresh,
    #                                                             IoA_matrix=IoA_matrix, 
    #                                                             aggr_track=aggr_track, 
    #                                                             IoA_thresh_aggr=IoA_thresh_aggressive, 
    #                                                             IDs_curr_untracked=IDs_curr_untracked, 
    #                                                             IDs_prev=IDs_prev, 
    #                                                             return_all=True
    #                                                             )
    #         tracked_video[frame_i] = tracked_lab

    #         if record_lineage and mother_daughters:

    #             for mother, daughters in mother_daughters:
    #                 mother_ID = IDs_prev[mother]
    #                 daughter_IDs = []

    #                 for daughter in daughters:
    #                     # printl(frame_i, assignments, IDs_curr_untracked)
    #                     daughter_IDs.append(assignments[IDs_curr_untracked[daughter]])

    #                 for family in families:
    #                     for member in family:
    #                         # printl(families)
    #                         if mother_ID == member[0]:
    #                             origin_id = family[0][0]
    #                             generation = member[1] + 1
    #                             family.extend([(daughter_ID, generation) for daughter_ID in daughter_IDs])
    #                             break
    #                     else: 
    #                         continue
    #                     break

    #                 for i, daughter_ID in enumerate(daughter_IDs):
    #                     daughter_IDs_copy = daughter_IDs.copy()
    #                     daughter_IDs_copy.pop(i)
    #                     daughter_IDs_copy = daughter_IDs_copy + [-1] * (max_daughter - len(daughter_IDs_copy) -1)
    #                     added_lineage_tree.append((frame_i, daughter_ID, mother_ID, generation, origin_id, daughter_IDs_copy))
    #         if record_lineage:
    #             added_lineage_tree = pd.DataFrame(added_lineage_tree, columns=['Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs'])
    #             lineage_tree_frame = pd.concat([lineage_list[-1][['Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs']], added_lineage_tree], axis=0)
    #             lineage_tree_frame['Frame'] = frame_i
    #             lineage_tree_frame = lineage_tree_frame[['Frame', 'Division frame', 'Daughter ID', 'Mother ID', 'Generation', 'Origin ID', 'Sister IDs']]            
    #             lineage_list.append(lineage_tree_frame)

    #         self.updateGuiProgressBar(signals)
    #         pbar.update()

    #     if record_lineage:
    #         df_final = pd.concat(lineage_list)
    #         df_final = reorg_daughter_cells(df_final, max_daughter)
    #         df_final.to_csv('lineage_tree.csv', index=False, na_rep='nan', float_format='%.0f') # need to change that

    #     pbar.close()
    #     return tracked_video
    
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