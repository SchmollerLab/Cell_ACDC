"""Visited-frame state transitions shared by GUI and scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LastVisitedFrameUpdate:
    """Updated last-visited counters for workflow modes."""

    last_tracked_i: int
    last_cca_frame_i: int
    changed: bool = False


def update_last_visited_frame_state(
    mode: str,
    last_visited_frame_i: int,
    *,
    last_tracked_i: int,
    last_cca_frame_i: int,
) -> LastVisitedFrameUpdate:
    """Return updated last-visited counters for a workflow mode."""
    mode = str(mode)
    last_visited_frame_i = int(last_visited_frame_i)
    last_tracked_i = int(last_tracked_i)
    last_cca_frame_i = int(last_cca_frame_i)

    if mode == 'Segmentation and Tracking':
        if last_tracked_i >= last_visited_frame_i:
            return LastVisitedFrameUpdate(last_tracked_i, last_cca_frame_i)
        return LastVisitedFrameUpdate(
            last_visited_frame_i,
            last_cca_frame_i,
            changed=True,
        )

    if mode == 'Cell cycle analysis':
        if last_cca_frame_i >= last_visited_frame_i:
            return LastVisitedFrameUpdate(last_tracked_i, last_cca_frame_i)
        return LastVisitedFrameUpdate(
            last_tracked_i,
            last_visited_frame_i,
            changed=True,
        )

    return LastVisitedFrameUpdate(last_tracked_i, last_cca_frame_i)
