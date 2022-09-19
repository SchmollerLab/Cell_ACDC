import pandas as pd
import numpy as np
import trackpy as tp

import skimage.measure
from cellacdc.trackers.CellACDC import CellACDC_tracker

class tracker:
    def __init__(self) -> None:
        pass

    def track(
            self, segm_video,
            search_range=10.0,
            memory=0,
            adaptive_stop: float=None, 
            adaptive_step=0.95,
            neighbor_strategy='KDTree',
            link_strategy = 'recursive',
            signals=None
        ):
        # Handle string input for adaptive_stop
        if isinstance(adaptive_stop, str):
            if adaptive_stop == 'None':
                adaptive_stop = None
            else:
                adaptive_stop = float(adaptive_stop)
        
        # Build tp DataFrame --> https://soft-matter.github.io/trackpy/v0.5.0/generated/trackpy.link.html#trackpy.link
        tp_df = {'x': [], 'y': [], 'frame': [], 'ID': []}
        for frame_i, lab in enumerate(segm_video):
            rp = skimage.measure.regionprops(lab)
            for obj in rp:
                yc, xc = obj.centroid
                tp_df['x'].append(xc)
                tp_df['y'].append(yc)
                tp_df['frame'].append(frame_i)
                tp_df['ID'].append(obj.label)

        tp_df = pd.DataFrame(tp_df)

        # Run tracker
        tp_df = tp.link_df(
            tp_df, search_range,
            memory=int(memory),
            adaptive_stop=adaptive_stop, 
            adaptive_step=adaptive_step,
            neighbor_strategy=neighbor_strategy,
            link_strategy=link_strategy,
        ).set_index('frame')
        tp_df['particle'] += 1 # trackpy starts from 0 with tracked ids

        # Generate tracked video data
        tracked_video = np.zeros_like(segm_video)
        for frame_i, lab in enumerate(segm_video):
            rp = skimage.measure.regionprops(lab)
            tracked_lab = lab.copy()
            tp_df_frame = tp_df.loc[frame_i]

            IDs_curr_untracked = [obj.label for obj in rp]
            if not IDs_curr_untracked:
                # No cells segmented
                continue
            
            try:
                tracked_IDs = tp_df_frame['particle'].astype(int).to_list()
                old_IDs = tp_df_frame['ID'].astype(int).to_list()
            except AttributeError:
                # Single cell
                tracked_IDs = [int(tp_df_frame['particle'])]
                old_IDs = [int(tp_df_frame['ID'])]
            
            if not tracked_IDs:
                # No cells tracked
                continue

            uniqueID = max((max(tracked_IDs), max(IDs_curr_untracked)))+1
            
            tracked_lab = CellACDC_tracker.indexAssignment(
                old_IDs, tracked_IDs, IDs_curr_untracked,
                lab.copy(), rp, uniqueID
            )
            tracked_video[frame_i] = tracked_lab

            # Used to update the progressbar of the gui
            if signals is not None:
                signals.progressBar.emit(1)
        
        return tracked_video
            
def url_help():
    return 'https://soft-matter.github.io/trackpy/v0.5.0/generated/trackpy.link.html#trackpy.link'