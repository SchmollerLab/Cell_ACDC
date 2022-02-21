import os
import numpy as np

from cellacdc.models.YeaZ.unet import tracking

class tracker:
    def __init__(self):
        pass

    def track(self, segm_video, signals=None, export_to: os.PathLike=None):
        tracked_stack = tracking.correspondence_stack(
            segm_video, signals=signals
        ).astype(np.uint16)
        return tracked_stack

    def save_output(self):
        pass
