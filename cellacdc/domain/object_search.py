"""Scriptable object search operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import skimage.measure


def find_frame_with_id(
    segmentation_frames: Sequence,
    frame_records: Sequence[dict],
    searched_id: int,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> int | None:
    """Return the first frame index containing ``searched_id``."""
    for frame_i, segmentation in enumerate(segmentation_frames):
        if frame_i >= len(frame_records):
            break

        frame_record = frame_records[frame_i]
        labels = frame_record['labels']
        if labels is None:
            regionprops = skimage.measure.regionprops(segmentation)
            frame_ids = {obj.label for obj in regionprops}
        else:
            frame_ids = set(frame_record['IDs'])

        if searched_id in frame_ids:
            return frame_i

        if progress_callback is not None:
            progress_callback(1)

    return None
