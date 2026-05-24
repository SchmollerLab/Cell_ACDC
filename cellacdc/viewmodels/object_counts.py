"""View-model commands for object counts and label frame selection."""

from __future__ import annotations

import os

from cellacdc.domain.object_counts import (
    collect_all_ids,
    current_labels,
    snapshot_object_counts,
)


class ObjectCountViewModel:
    """Application-facing object count and label-frame commands."""

    def current_labels(self, pos_data, *, curr_lab=None, frame_i=None):
        return current_labels(pos_data, curr_lab=curr_lab, frame_i=frame_i)

    def collect_all_ids(self, pos_data, *, only_visited: bool = False) -> set[int]:
        return collect_all_ids(pos_data, only_visited=only_visited)

    def snapshot_object_counts(
        self,
        positions,
        current_pos_i: int,
        *,
        current_lab_2d=None,
        include_current_z_slice: bool = False,
        path_exists=os.path.exists,
    ) -> dict[str, int]:
        return snapshot_object_counts(
            positions,
            current_pos_i,
            current_lab_2d=current_lab_2d,
            include_current_z_slice=include_current_z_slice,
            path_exists=path_exists,
        )
