"""Qt-free model rules for tracking metadata."""

from __future__ import annotations

from cellacdc.domain.tracking import (
    FutureIdPropagationScan,
    LostNewIdsResult,
    TrackedLostIdsResult,
    compute_lost_new_ids,
    last_tracked_frame_index,
    scan_future_id_propagation,
    tracked_lost_centroids_from_regionprops,
    tracked_lost_ids_from_centroids,
)


class TrackingModel:
    """Headless tracking state calculations."""

    def compute_lost_new_ids(
        self,
        previous_ids,
        current_ids,
        *,
        current_deleted_roi_ids=(),
        previous_deleted_roi_ids=(),
        tracked_lost_ids=(),
    ) -> LostNewIdsResult:
        return compute_lost_new_ids(
            previous_ids,
            current_ids,
            current_deleted_roi_ids=current_deleted_roi_ids,
            previous_deleted_roi_ids=previous_deleted_roi_ids,
            tracked_lost_ids=tracked_lost_ids,
        )

    def tracked_lost_centroids_from_regionprops(
        self,
        regionprops,
        tracked_lost_ids,
    ) -> set[tuple[int, ...]]:
        return tracked_lost_centroids_from_regionprops(
            regionprops,
            tracked_lost_ids,
        )

    def tracked_lost_ids_from_centroids(
        self,
        previous_labels,
        tracked_lost_centroids,
        ids_in_frame,
    ) -> TrackedLostIdsResult:
        return tracked_lost_ids_from_centroids(
            previous_labels,
            tracked_lost_centroids,
            ids_in_frame,
        )

    def last_tracked_frame_index(
        self,
        frame_labels,
        *,
        first_frame_fallback: int = 0,
        total_frames: int | None = None,
    ) -> int:
        return last_tracked_frame_index(
            frame_labels,
            first_frame_fallback=first_frame_fallback,
            total_frames=total_frames,
        )

    def scan_future_id_propagation(
        self,
        target_id: int,
        *,
        current_frame_i: int,
        frame_labels,
        fallback_frame_labels,
        include_unvisited: bool = False,
        total_frames: int | None = None,
    ) -> FutureIdPropagationScan:
        return scan_future_id_propagation(
            target_id,
            current_frame_i=current_frame_i,
            frame_labels=frame_labels,
            fallback_frame_labels=fallback_frame_labels,
            include_unvisited=include_unvisited,
            total_frames=total_frames,
        )
