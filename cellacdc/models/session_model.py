"""Scriptable model rules for session workflows."""

from __future__ import annotations

import numpy as np


class SessionModel:
    """Headless decisions for session and frame storage workflows."""

    def should_store_frame_data(
        self,
        *,
        frame_i: int,
        mode: str,
        enforce: bool,
    ) -> bool:
        if frame_i < 0:
            return False
        if mode == 'Viewer' and not enforce:
            return False
        return True

    def should_disable_load_position(self, position_count: int) -> bool:
        return position_count <= 1

    def labels_shape(
        self,
        *,
        is_3d: bool,
        size_z: int,
        size_y: int,
        size_x: int,
    ) -> tuple[int, ...]:
        if is_3d:
            return (size_z, size_y, size_x)
        return (size_y, size_x)

    def empty_labels(
        self,
        *,
        is_3d: bool,
        size_z: int,
        size_y: int,
        size_x: int,
    ) -> np.ndarray:
        shape = self.labels_shape(
            is_3d=is_3d,
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
        )
        return np.zeros(shape, dtype=np.uint32)

    def should_resume_last_session_prompt(self, last_tracked_num: int) -> bool:
        return last_tracked_num > 1
