import os
import sys
import traceback

import numpy as np
import pandas as pd

import btrack
from btrack.constants import BayesianUpdates

from skimage.measure import regionprops
from ..CellACDC import CellACDC_tracker

from cellacdc import printl

from tqdm import tqdm

class tracker:
    def __init__(self, **params):
        self.params = params

    def track(
            self, segm_video, image, signals=None, 
            export_to: os.PathLike=None, verbose=False
        ):
        FEATURES = self.params['features']

        if segm_video.ndim == 3:
            # btrack requires 4D data. Add extra dimension for 3D data
            segm_video = segm_video[:, np.newaxis, :, :]
        
        if image is not None:
            if image.ndim == 3:
                image = image[:, np.newaxis, :, :]

        try:
            obj_from_arr = btrack.utils.segmentation_to_objects(
                segm_video, properties=tuple(FEATURES), intensity_image=image
            )
        except Exception as e:
            # Try ignoring the intensity_image, it might have wrong shape
            obj_from_arr = btrack.utils.segmentation_to_objects(
                segm_video, properties=tuple(FEATURES)
            )

        if signals is not None:
            signals.progress.emit('Running BayesianTracker...')

        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as tracker:

            # configure the tracker using a config file
            tracker.configure_from_file(self.params['model_path'])
            tracker.verbose = self.params['verbose']
            update_method = self.params['update_method']

            if update_method == 'APPROXIMATE':
                tracker.update_method = getattr(BayesianUpdates, update_method)
                tracker.max_search_radius = self.params['max_search_radius']

            # add features
            if FEATURES:
                tracker.features = FEATURES

            # append the objects to be tracked
            tracker.append(obj_from_arr)

            # set the volume
            tracker.volume=self.params['volume']

            # track them (in interactive mode)
            tracker.track(step_size=self.params['step_size'])

            # generate hypotheses and run the global optimizer
            if self.params['optimize']:
                tracker.optimize()

            # save tracks
            if export_to is not None:
                tracker.export(export_to, obj_type="obj_type_1")

            tracks = tracker.tracks

        # Remove the added axis
        segm_video = np.squeeze(segm_video)
        tracked_video = self._from_tracks_to_labels(
            tracks, segm_video, signals=signals, verbose=verbose
        )
        return tracked_video

    def _from_tracks_to_labels(
            self, tracks, segm_video, signals=None, verbose=False
        ):
        if signals is not None:
            signals.progress.emit('Applying BayesianTracker tracks...')

        # Label the segm_video according to tracks
        tracked_video = np.zeros_like(segm_video)
        for frame_i, lab in enumerate(tqdm(segm_video, ncols=100)):
            rp = regionprops(lab.copy())
            IDs_curr_untracked = [obj.label for obj in rp]
            if not IDs_curr_untracked:
                # No cells segmented
                continue

            old_IDs = []
            tracked_IDs = []
            for track in tracks:
                track_dict = track.to_dict()
                if frame_i not in track_dict['t']:
                    continue

                df = pd.DataFrame(track.to_dict()).set_index('t').loc[frame_i]

                yc, xc = df['y'], df['x']
                try:
                    old_ID = lab[int(yc), int(xc)]
                except Exception as e:
                    # btrack sometimes finds cells that are not existing
                    # skip them
                    # traceback.print_exc()
                    continue

                old_IDs.append(old_ID)
                tracked_IDs.append(df['ID'])

            if not tracked_IDs:
                # No cells tracked
                continue

            uniqueID = max((max(tracked_IDs), max(IDs_curr_untracked)))+1

            if verbose:
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

    def save_output(self):
        pass
