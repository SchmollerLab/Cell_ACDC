"""Qt-free model rules for canvas event routing."""

from __future__ import annotations

import numpy as np


class CanvasEventsModel:
    """Headless canvas event routing rules and brush mask computations."""

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

    def map_mouse_coordinates_to_label_id(
        self,
        mouse_pos: tuple[float, float],
        label_matrix: np.ndarray,
    ) -> int:
        """Resolves float pixel coordinate lookup to integer label ID."""
        x, y = mouse_pos
        xdata, ydata = int(x), int(y)
        height, width = label_matrix.shape
        if 0 <= xdata < width and 0 <= ydata < height:
            return int(label_matrix[ydata, xdata])
        return 0
