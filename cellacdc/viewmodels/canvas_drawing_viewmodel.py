"""View-model contracts for canvas drawing interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from cellacdc.models.canvas_drawing_model import CanvasDrawingModel

from .geometry_viewmodel import GeometryViewModel
from .label_edits_viewmodel import LabelEditViewModel


@dataclass(frozen=True)
class CanvasDrawingViewModel:
    """Application-facing canvas drawing decisions and transforms."""

    model: CanvasDrawingModel = field(default_factory=CanvasDrawingModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)

    def is_viewer_mode(self, mode: str) -> bool:
        return mode == self.model.viewer_mode

    def is_in_bounds(self, x: int, y: int, width: int, height: int) -> bool:
        return self.geometry.is_in_bounds(x, y, width, height)

    def should_process_canvas_event(self, *, mode: str, in_bounds: bool) -> bool:
        return self.model.should_process_canvas_event(
            mode=mode,
            in_bounds=in_bounds,
        )

    def should_clear_after_out_of_bounds(self, *, image: str) -> bool:
        return self.model.should_clear_after_out_of_bounds(image=image)

    def binary_fill_holes(self, mask):
        return self.label_edits.binary_fill_holes(mask)

    def convex_hull_mask(self, mask):
        return self.label_edits.convex_hull_mask(mask)

    def nearest_nonzero_2d(self, labels, y, x):
        return self.label_edits.nearest_nonzero_2d(labels, y, x)

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
        return self.model.calculate_brush_mask(
            image_shape, ymin, xmin, ymax, xmax, disk_mask, rr_poly, cc_poly
        )

