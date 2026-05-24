"""View-model behavior for main-window event handling."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.window_events_model import WindowEventsModel

from .geometry import GeometryViewModel


@dataclass(frozen=True)
class WindowEventsViewModel:
    """GUI-facing helpers for main-window event handling."""

    model: WindowEventsModel = field(default_factory=WindowEventsModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)

    def windows_overlap_from_bounds(self, **kwargs):
        return self.geometry.windows_overlap_from_bounds(**kwargs)

    def should_auto_activate_viewer(self, **kwargs):
        return self.geometry.should_auto_activate_viewer(**kwargs)

    def is_pan_image_click(self, **kwargs):
        return self.geometry.is_pan_image_click(**kwargs)

    def is_default_middle_click(self, **kwargs):
        return self.geometry.is_default_middle_click(**kwargs)

    def is_configured_middle_click(self, **kwargs):
        return self.geometry.is_configured_middle_click(**kwargs)

    def middle_click_text(self, **kwargs):
        return self.geometry.middle_click_text(**kwargs)
