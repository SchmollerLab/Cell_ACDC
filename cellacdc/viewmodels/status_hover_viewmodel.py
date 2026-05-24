"""View-model contracts for hover and status-bar text."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.status_hover_model import StatusHoverModel


@dataclass(frozen=True)
class StatusHoverViewModel:
    """Application-facing status/hover formatting commands."""

    model: StatusHoverModel = field(default_factory=StatusHoverModel)

    def channel_hover_text(self, description, channel, value, format_spec):
        return self.model.channel_hover_text(
            description, channel, value, format_spec
        )

    def object_hover_text(self, *, label_id, max_id, object_count):
        return self.model.object_hover_text(
            label_id=label_id,
            max_id=max_id,
            object_count=object_count,
        )

    def base_hover_text(self, **kwargs):
        return self.model.base_hover_text(**kwargs)

    def replace_view_range_status(self, text, **kwargs):
        return self.model.replace_view_range_status(text, **kwargs)

    def highlight_state(self, **kwargs):
        return self.model.highlight_state(**kwargs)

    def mouse_data_coords_right_image(self, text):
        return self.model.mouse_data_coords_right_image(text)

    def ruler_length_text(self, text):
        return self.model.ruler_length_text(text)

    def ruler_measurement_text(self, *, length_pixels, pixel_to_um):
        return self.model.ruler_measurement_text(
            length_pixels=length_pixels,
            pixel_to_um=pixel_to_um,
        )

    def euclidean_length(self, x_values, y_values):
        return self.model.euclidean_length(x_values, y_values)

    def status_bar_text(self, **kwargs):
        return self.model.status_bar_text(**kwargs)
