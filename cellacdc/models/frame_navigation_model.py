"""Scriptable model rules for frame and position navigation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationLimit:
    """Maximum navigation frame and optional status label."""

    maximum: int
    last_checked_frame_i: int | None = None
    status_text: str | None = None


class FrameNavigationModel:
    """Headless decisions for frame/position navigation workflows."""

    viewer_mode = 'Viewer'
    segmentation_mode = 'Segmentation and Tracking'
    cell_cycle_mode = 'Cell cycle analysis'
    lineage_mode = 'Normal division: Lineage tree'

    def should_show_next_frame_image(
        self,
        *,
        size_t: int,
        has_right_image_coords: bool,
        action_enabled: bool,
        action_checked: bool,
    ) -> bool:
        return (
            size_t > 1
            and has_right_image_coords
            and action_enabled
            and action_checked
        )

    def next_frame_index(self, *, current_frame_i: int, frames_count: int) -> int:
        next_frame_i = current_frame_i + 1
        if next_frame_i >= frames_count:
            return frames_count - 1
        return next_frame_i

    def navigation_position(
        self,
        *,
        is_snapshot: bool,
        position_i: int,
        frame_i: int,
    ) -> int:
        return position_i + 1 if is_snapshot else frame_i + 1

    def navigation_limit(
        self,
        *,
        mode: str,
        frame_i: int,
        last_tracked_i: int | None,
        last_cca_frame_i: int,
        lineage_tree_frames,
    ) -> NavigationLimit | None:
        if mode == self.segmentation_mode:
            if last_tracked_i is None or frame_i > last_tracked_i:
                maximum = frame_i + 1
            else:
                maximum = last_tracked_i + 1
            return NavigationLimit(
                maximum=maximum,
                last_checked_frame_i=maximum - 1,
            )
        if mode == self.cell_cycle_mode:
            maximum = max(frame_i, last_cca_frame_i) + 1
            return NavigationLimit(
                maximum=maximum,
                status_text=f'Last cc annot. frame n. = {maximum}',
            )
        if mode == self.lineage_mode:
            if lineage_tree_frames:
                maximum = max(lineage_tree_frames) + 1
            else:
                maximum = frame_i + 1
            return NavigationLimit(maximum=maximum)
        return None

    def should_store_when_slider_moves(self, *, mode: str) -> bool:
        return mode != self.viewer_mode

    def should_warn_lost_objects(
        self,
        *,
        requested: bool,
        action_checked: bool,
        mode: str,
        lost_ids,
        already_accepted: bool,
    ) -> bool:
        if not requested:
            return False
        if not action_checked:
            return False
        if mode != self.segmentation_mode:
            return False
        if not lost_ids:
            return False
        return not already_accepted

    def blocks_future_manual_annotation(
        self,
        *,
        manual_annotation_enabled: bool,
        current_frame_i: int,
        frame_to_restore,
    ) -> bool:
        if not manual_annotation_enabled:
            return False
        if frame_to_restore is None:
            return False
        return current_frame_i > frame_to_restore

    def should_apply_new_frame_tools(
        self,
        *,
        mode: str,
        last_tracked_i: int,
        frame_i: int,
        last_frame_ran: int,
    ) -> bool:
        return (
            mode == self.segmentation_mode
            and last_tracked_i is not None
            and last_tracked_i <= frame_i
            and frame_i != last_frame_ran
        )

    def is_single_z_slice_projection(self, how: str) -> bool:
        return how == 'single z-slice'

    def should_disable_overlay_z_slice(self, how: str) -> bool:
        return how.find('max') != -1 or how == 'same as above'

    def projection_frame_indices(
        self,
        *,
        filename,
        frame_i: int,
        size_t: int,
        locked: bool,
    ) -> list[tuple[object, int]]:
        if locked:
            return [(filename, i) for i in range(size_t)]
        return [(filename, frame_i)]

    def z_slice_frame_indices(
        self,
        *,
        filename,
        frame_i: int,
        size_t: int,
        locked: bool,
    ) -> list[tuple[object, int]]:
        if locked:
            return [(filename, i) for i in range(size_t)]
        return [(filename, i) for i in range(frame_i, size_t)]
