import os
import sys

import numpy as np
import pandas as pd

import btrack

from skimage.measure import regionprops
from cellacdc.trackers.CellACDC import CellACDC_tracker

class tracker:
    def __init__(self):
        trackers_path = os.path.dirname(__file__)
        self.model_path = os.path.join(
            trackers_path, 'model', 'cell_config.json'
        )

    def track(self, segm_video, signals=None, export_to: os.PathLike=None):
        if segm_video.ndim == 3:
            # btrack requires 4D data. Add extra dimension for 3D data
            segm_video = segm_video[:, np.newaxis, :, :]

        obj_from_arr = btrack.utils.segmentation_to_objects(
            segm_video, properties=('area',), scale=(1., 1., 1.)
        )
        if signals is not None:
            signals.progress.emit('Running BayesianTracker...')

        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as tracker:

            # configure the tracker using a config file
            tracker.configure_from_file(self.model_path)
            tracker.max_search_radius = 10

            # append the objects to be tracked
            tracker.append(obj_from_arr)

            # set the volume
            tracker.volume=((0, 1200), (0, 1200), (-1e5, 64.))

            # track them (in interactive mode)
            tracker.track_interactive(step_size=100)

            # generate hypotheses and run the global optimizer
            tracker.optimize()

            # save tracks
            tracker.export(export_to, obj_type="obj_type_1")

            tracks = tracker.tracks

        # Remove the added axis
        segm_video = np.squeeze(segm_video)

        return self._from_tracks_to_labels(tracks, segm_video, signals=signals)

    def _from_tracks_to_labels(self, tracks, segm_video, signals=None):
        # Label the segm_video according to tracks
        tracked_video = np.zeros_like(segm_video)
        for frame_i, lab in enumerate(segm_video):
            if frame_i == 0:
                tracked_video[frame_i] = lab
                continue

            rp = regionprops(lab.copy())
            IDs_curr_untracked = [obj.label for obj in rp]

            old_IDs = []
            tracked_IDs = []
            for track in tracks:
                if not track.in_frame(frame_i):
                    continue

                df = pd.DataFrame(track.to_dict()).set_index('t').loc[frame_i]
                yc, xc = df['y'], df['x']
                old_ID = lab[int(yc), int(xc)]
                old_IDs.append(old_ID)
                tracked_IDs.append(df['ID'])

            uniqueID = max((max(tracked_IDs), max(IDs_curr_untracked)))+1

            tracked_lab = CellACDC_tracker.indexAssignment(
                old_IDs, tracked_IDs, IDs_curr_untracked,
                lab.copy(), rp, uniqueID
            )
            tracked_video[frame_i] = tracked_lab
            if signals is not None:
                signals.progressBar.emit(1)

        return tracked_video

    def save_output(self):
        pass
