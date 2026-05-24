"""Scriptable model rules for object cleanup workflows."""

from __future__ import annotations

import numpy as np


class ObjectCleanupModel:
    """Headless object-cleanup result shaping."""

    def cleared_segmentation_frames(self, cleared_segm_data, *, size_t: int):
        if size_t == 1:
            return cleared_segm_data[np.newaxis]
        return cleared_segm_data

    def frame_labels(self, cleared_segm_data):
        return list(enumerate(cleared_segm_data))
