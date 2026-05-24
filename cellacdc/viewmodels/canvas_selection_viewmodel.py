"""View-model behavior for canvas selection interactions."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.canvas_selection_model import CanvasSelectionModel

from .geometry_viewmodel import GeometryViewModel
from .label_edits_viewmodel import LabelEditViewModel


@dataclass(frozen=True)
class CanvasSelectionViewModel:
    """GUI-facing canvas selection decisions and transforms."""

    model: CanvasSelectionModel = field(default_factory=CanvasSelectionModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)

    def is_in_bounds(self, x: int, y: int, width: int, height: int) -> bool:
        return self.geometry.is_in_bounds(x, y, width, height)

    def should_drag_image(self, **kwargs) -> bool:
        return self.model.should_drag_image(**kwargs)

    def should_blink_viewer_mode(self, **kwargs) -> bool:
        return self.model.should_blink_viewer_mode(**kwargs)

    def should_show_labels_menu(self, **kwargs) -> bool:
        return self.model.should_show_labels_menu(**kwargs)

    def can_delete(self, **kwargs) -> bool:
        return self.model.can_delete(**kwargs)

    def is_viewer_mode(self, mode: str) -> bool:
        return self.model.is_viewer_mode(mode)

    def should_process_release(self, **kwargs) -> bool:
        return self.model.should_process_release(**kwargs)

    def nearest_nonzero_2d(self, labels, y, x):
        return self.label_edits.nearest_nonzero_2d(labels, y, x)

    def separate_with_label(self, *args, **kwargs):
        return self.label_edits.separate_with_label(*args, **kwargs)

    def split_along_convexity_defects(self, *args, **kwargs):
        return self.label_edits.split_along_convexity_defects(*args, **kwargs)
