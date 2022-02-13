from cellacdc.models.YeaZ.unet import tracking
import numpy as np

class tracker:
    def __init__(self):
        pass

    def track(self, segm_video, signals=None):
        tracked_stack = tracking.correspondence_stack(
            segm_video, signals=signals
        ).astype(np.uint16)
        return tracked_stack
