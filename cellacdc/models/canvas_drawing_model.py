"""Scriptable model rules for canvas drawing interactions."""

from __future__ import annotations

import numpy as np


class CanvasDrawingModel:
    """Headless decisions for canvas drawing workflows."""

    viewer_mode = 'Viewer'

    def should_process_canvas_event(
        self,
        *,
        mode: str,
        in_bounds: bool,
    ) -> bool:
        return mode != self.viewer_mode and in_bounds

    def should_clear_after_out_of_bounds(self, *, image: str) -> bool:
        return image == 'img1'

    def calculate_brush_mask(
        self,
        image_shape: tuple[int, int],
        ymin: int,
        xmin: int,
        ymax: int,
        xmax: int,
        disk_mask: np.ndarray,
        rr_poly: np.ndarray | None = None,
        cc_poly: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes a 2D boolean mask for brush/eraser updates."""
        mask = np.zeros(image_shape, dtype=bool)
        disk_slice = (slice(ymin, ymax), slice(xmin, xmax))
        mask[disk_slice][disk_mask] = True
        if rr_poly is not None and cc_poly is not None:
            mask[rr_poly, cc_poly] = True
        return mask

