import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import calc_IoA_matrix
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame as track_frame_base
from cellacdc.core import getBaseCca_df,printl
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd

def reorg_daughter_cells(lineage_tree_frame, max_daughter):
    new_columns = [f'sister_ID_tree_{i+1}' for i in range(max_daughter-1)]
    sister_columns = lineage_tree_frame['sister_ID_tree'].apply(pd.Series) 
    lineage_tree_frame[new_columns] = sister_columns
    lineage_tree_frame['sister_ID_tree'] = sister_columns[0]

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

def create_lineage_tree_video(segm_video, IoA_thresh_daughter, min_daughter, max_daughter):
    tree = normal_division_lineage_tree(segm_video[0])
    for i, frame in enumerate(segm_video[1:], start=1):
        rp = regionprops(frame)
        prev_rp = regionprops(segm_video[i-1])
        IoA_matrix, IDs_curr_untracked, IDs_prev = calc_IoA_matrix(frame, segm_video[i-1], rp, prev_rp)
        _, mother_daughters = ident_no_mothers(IoA_matrix, IoA_thresh_daughter, min_daughter, max_daughter)
        assignments = IDs_curr_untracked #bc we dont track the frame
        tree.create_tracked_frame_tree(i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments)
    return tree.lineage_list

class normal_division_tracker:
    def __init__(self, 
                 segm_video, 
                 IoA_thresh_daughter, 
                 min_daughter, 
                 max_daughter, 
                 IoA_thresh, 
                 IoA_thresh_aggressive):
        
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

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_IoA_matrix(lab, 
                                                                             prev_lab, 
                                                                             self.rp, 
                                                                             prev_rp
                                                                             )
        aggr_track, self.mother_daughters = ident_no_mothers(IoA_matrix, 
                                                             IoA_thresh_daughter=self.IoA_thresh_daughter, 
                                                             min_daughter=self.min_daughter, 
                                                             max_daughter=self.max_daughter
                                                             )
        self.tracked_lab, IoA_matrix, self.assignments = track_frame_base(prev_lab, 
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
        cca_df = reorg_daughter_cells(cca_df, max_daughter)
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
        cca_df = reorg_daughter_cells(cca_df, self.max_daughter)
        self.lineage_list.append(cca_df)

class tracker:
    def __init__(self):
        pass

    def track(self, 
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
    
    def track_frame(self, 
                    previous_frame_labels, 
                    current_frame_labels, 
                    IoA_thresh = 0.8,
                    IoA_thresh_daughter = 0.25,
                    IoA_thresh_aggressive = 0.5,
                    min_daughter = 2,
                    max_daughter = 2,
                    ):
        
        if not np.any(current_frame_labels):
            # Skip empty frames
            return current_frame_labels

        segm_video = [previous_frame_labels, current_frame_labels]
        tracker = normal_division_tracker(segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive)
        tracker.track_frame(1)
        tracked_video = tracker.tracked_video
        
        mother_daughters_pairs = tracker.mother_daughters
        IDs_prev = tracker.IDs_prev
        mothers = {IDs_prev[mother[0]] for mother in mother_daughters_pairs}

        return tracked_video[-1], mothers
    
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