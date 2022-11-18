import os
import numpy as np

from . import tracking

class tracker:
    def __init__(self):
        pass

    def track(self, segm_video, signals=None):
        tracked_stack = tracking.correspondence_stack(
            segm_video, signals=signals
        ).astype(np.uint32)
        return tracked_stack

    def save_output(self):
        pass
