"""Tracking-related label metadata transforms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class TrackedLostIdsResult:
    """Result of resolving stored lost centroids against a previous label image."""

    lost_ids: set[int]
    remaining_centroids: set[tuple[int, ...]]


@dataclass(frozen=True)
class LostNewIdsResult:
    """Result of comparing previous and current frame IDs."""

    lost_ids: list[int]
    new_ids: list[int]
    ids_with_holes: list[int]


@dataclass(frozen=True)
class FutureIdPropagationScan:
    """Future-frame scan result for an ID-changing segmentation edit."""

    last_tracked_i: int
    has_affected_future_ids: bool


def last_tracked_frame_index(
    frame_labels,
    *,
    first_frame_fallback: int = 0,
    total_frames: int | None = None,
) -> int:
    """Return the last contiguous frame index with stored labels.

    ``first_frame_fallback`` preserves legacy GUI paths that disagree on
    whether an unvisited first frame means frame ``0`` or no frame ``-1``.
    """
    last_seen_i = 0
    saw_frame = False
    for frame_i, labels in enumerate(frame_labels):
        saw_frame = True
        if labels is None:
            return first_frame_fallback if frame_i == 0 else frame_i - 1
        last_seen_i = frame_i

    if total_frames is not None:
        return max(int(total_frames) - 1, 0)
    if not saw_frame:
        return 0
    return last_seen_i


def scan_future_id_propagation(
    target_id: int,
    *,
    current_frame_i: int,
    frame_labels,
    fallback_frame_labels,
    include_unvisited: bool = False,
    total_frames: int | None = None,
) -> FutureIdPropagationScan:
    """Scan future labels for ``target_id`` and report propagation state."""
    frame_labels = list(frame_labels)
    fallback_frame_labels = list(fallback_frame_labels)
    if total_frames is None:
        total_frames = len(fallback_frame_labels)

    last_tracked_i = int(total_frames) - 1
    last_tracked_i_found = False
    has_affected_future_ids = False
    for frame_i in range(current_frame_i + 1, len(fallback_frame_labels)):
        labels = frame_labels[frame_i]
        if labels is None:
            if not last_tracked_i_found:
                last_tracked_i = frame_i - 1
                last_tracked_i_found = True
            if not include_unvisited:
                break
            labels = fallback_frame_labels[frame_i]

        if target_id in labels:
            has_affected_future_ids = True

    return FutureIdPropagationScan(
        last_tracked_i=last_tracked_i,
        has_affected_future_ids=has_affected_future_ids,
    )


def track_labels(
        labels: np.ndarray,
        tracker_name: str = 'CellACDC',
        *,
        init_kwargs: dict | None = None,
        track_params: dict | None = None,
        intensity_img=None,
        logger_func=print,
) -> np.ndarray:
    """Track a label video with a Cell-ACDC tracker plugin."""
    from cellacdc.plugins.registry import import_tracker_module

    init_kwargs = {} if init_kwargs is None else init_kwargs
    track_params = {} if track_params is None else track_params
    tracker_module = import_tracker_module(tracker_name)
    tracker = tracker_module.tracker(**init_kwargs)
    args_to_try = (tuple(), (intensity_img,)) if intensity_img is not None else (tuple(),)

    for args, kwarg_to_remove in product(args_to_try, ('', 'signals')):
        kwargs = track_params.copy()
        kwargs.pop(kwarg_to_remove, None)
        try:
            return tracker.track(labels, *args, **kwargs)
        except Exception as err:
            is_unexpected_kwarg = (
                "got an unexpected keyword argument 'signals'" in str(err)
            )
            is_missing_arg = 'missing 1 required positional argument:' in str(err)
            if is_unexpected_kwarg or is_missing_arg:
                continue
            raise

    raise RuntimeError(f'Unable to run {tracker_name} tracker')


def compute_lost_new_ids(
        previous_ids,
        current_ids,
        *,
        current_deleted_roi_ids=(),
        previous_deleted_roi_ids=(),
        tracked_lost_ids=(),
) -> LostNewIdsResult:
    """Compute ordered lost/new ID lists between adjacent frames."""
    current_id_set = {int(label_id) for label_id in current_ids}
    previous_id_set = {int(label_id) for label_id in previous_ids}
    current_deleted_roi_ids = {
        int(label_id) for label_id in current_deleted_roi_ids
    }
    previous_deleted_roi_ids = {
        int(label_id) for label_id in previous_deleted_roi_ids
    }
    tracked_lost_ids = {int(label_id) for label_id in tracked_lost_ids}

    lost_ids = [
        int(label_id) for label_id in previous_ids
        if (
            int(label_id) not in current_id_set
            and int(label_id) not in previous_deleted_roi_ids
            and int(label_id) not in tracked_lost_ids
        )
    ]
    new_ids = [
        int(label_id) for label_id in current_ids
        if (
            int(label_id) not in previous_id_set
            and int(label_id) not in current_deleted_roi_ids
        )
    ]

    return LostNewIdsResult(
        lost_ids=lost_ids,
        new_ids=new_ids,
        ids_with_holes=[],
    )


def tracked_lost_centroids_from_regionprops(
        regionprops,
        tracked_lost_ids,
) -> set[tuple[int, ...]]:
    """Collect integer centroids for tracker-accepted lost IDs."""
    tracked_lost_ids = {int(label_id) for label_id in tracked_lost_ids}
    return {
        tuple(int(val) for val in obj.centroid)
        for obj in regionprops
        if int(obj.label) in tracked_lost_ids
    }


def tracked_lost_ids_from_centroids(
        prev_labels: np.ndarray,
        tracked_lost_centroids,
        ids_in_frame,
) -> TrackedLostIdsResult:
    """Resolve tracked-lost centroids to IDs and prune re-tracked centroids."""
    tracked_lost_centroids = {
        tuple(int(coord) for coord in centroid)
        for centroid in tracked_lost_centroids
    }
    ids_in_frame = {int(label_id) for label_id in ids_in_frame}
    retracked_centroids = set()
    lost_ids = set()

    for centroid in tracked_lost_centroids:
        if len(centroid) < 3 and prev_labels.ndim == 3:
            # Ignore wrongly stored centroids, preserving the original record.
            continue

        label_id = int(prev_labels[centroid])
        if label_id == 0:
            continue

        if label_id in ids_in_frame:
            retracked_centroids.add(centroid)
            continue

        lost_ids.add(label_id)

    return TrackedLostIdsResult(
        lost_ids=lost_ids,
        remaining_centroids=tracked_lost_centroids - retracked_centroids,
    )
