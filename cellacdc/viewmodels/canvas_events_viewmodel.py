"""View-model behavior for canvas event routing."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.canvas_events_model import CanvasEventsModel

from .geometry import GeometryViewModel
from .label_edits import LabelEditViewModel


@dataclass(frozen=True)
class CanvasEventsViewModel:
    """GUI-facing helpers for canvas event routing."""

    model: CanvasEventsModel = field(default_factory=CanvasEventsModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)

    def snap_xy_to_closest_angle(self, x0, y0, x1, y1):
        return self.geometry.snap_xy_to_closest_angle(x0, y0, x1, y1)

    def nearest_nonzero_2d(self, labels, y, x):
        return self.label_edits.nearest_nonzero_2d(labels, y, x)

    def binary_fill_holes(self, labels):
        return self.label_edits.binary_fill_holes(labels)

    def convex_hull_mask(self, labels):
        return self.label_edits.convex_hull_mask(labels)

    def calculate_brush_mask(
        self,
        image_shape: tuple[int, int],
        ymin: int,
        xmin: int,
        ymax: int,
        xmax: int,
        disk_mask,
        rr_poly=None,
        cc_poly=None,
    ):
        return self.model.calculate_brush_mask(
            image_shape, ymin, xmin, ymax, xmax, disk_mask, rr_poly, cc_poly
        )

    def map_mouse_coordinates_to_label_id(
        self,
        mouse_pos: tuple[float, float],
        label_matrix,
    ) -> int:
        return self.model.map_mouse_coordinates_to_label_id(
            mouse_pos, label_matrix
        )
