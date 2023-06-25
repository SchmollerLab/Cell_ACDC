import pandas as pd
import numpy as np
import trackpy as tp

import skimage.measure
from ..CellACDC import CellACDC_tracker

from cellacdc import apps, printl

DEBUG = False

class tracker:
    def __init__(self) -> None:
        pass

    def _set_frame_features(self, lab, frame_i, tp_df):
        rp = skimage.measure.regionprops(lab)
        for obj in rp:
            yc, xc = obj.centroid
            tp_df['x'].append(xc)
            tp_df['y'].append(yc)
            tp_df['frame'].append(frame_i)
            tp_df['ID'].append(obj.label)

    def track(
            self, segm_video,
            search_range=10.0,
            memory=0,
            adaptive_stop: float=None, 
            adaptive_step=0.95,
            dynamic_predictor=False,
            neighbor_strategy='KDTree',
            link_strategy = 'recursive',
            signals=None, 
            export_to=None,
            export_to_extension='.csv'
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
            self._set_frame_features(lab, frame_i, tp_df)
        tp_df = pd.DataFrame(tp_df)

        # Run tracker
        if dynamic_predictor:
            predictor = tp.predict.NearestVelocityPredict()
        else:
            predictor = tp

        tp_df = predictor.link_df(
            tp_df, search_range,
            memory=int(memory),
            adaptive_stop=adaptive_stop, 
            adaptive_step=adaptive_step,
            neighbor_strategy=neighbor_strategy,
            link_strategy=link_strategy,
        ).set_index('frame')
        
        if export_to is not None:
            tp_df.to_csv(export_to)
        
        tp_df['particle'] += 1 # trackpy starts from 0 with tracked ids

        # Generate tracked video data
        tracked_video = np.zeros_like(segm_video)
        for frame_i, lab in enumerate(segm_video):
            rp = skimage.measure.regionprops(lab)
            tracked_lab = lab.copy()
            tp_df_frame = tp_df.loc[frame_i]

            IDs_curr_untracked = [obj.label for obj in rp]

            if DEBUG:
                printl(f'Current untracked IDs: {IDs_curr_untracked}')

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

            if DEBUG:
                print('-------------------------')
                print(f'Tracking frame n. {frame_i+1}')
                for old_ID, tracked_ID in zip(old_IDs, tracked_IDs):
                    print(f'Tracking ID {old_ID} --> {tracked_ID}')
                print('-------------------------')
            
            tracked_lab = CellACDC_tracker.indexAssignment(
                old_IDs, tracked_IDs, IDs_curr_untracked,
                lab.copy(), rp, uniqueID
            )
            tracked_video[frame_i] = tracked_lab

            self.updateGuiProgressBar(signals)
        
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
            
def url_help():
    return 'https://soft-matter.github.io/trackpy/v0.5.0/generated/trackpy.link.html#trackpy.link'