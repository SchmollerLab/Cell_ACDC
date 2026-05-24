"""View-model contracts for image control widgets."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.image_controls_model import ImageControlsModel


@dataclass(frozen=True)
class ImageControlsViewModel:
    """Application-facing image-control defaults."""

    model: ImageControlsModel = field(default_factory=ImageControlsModel)

    def draw_ids_cont_combo_items(self) -> tuple[str, ...]:
        return self.model.draw_ids_cont_combo_items

    def z_projection_options(self) -> tuple[str, ...]:
        return self.model.z_projection_options

    def overlay_z_projection_options(self) -> tuple[str, ...]:
        return self.model.overlay_z_projection_options

    def bottom_layout_zoom_values(self) -> tuple[int, ...]:
        return self.model.bottom_layout_zoom_values

    def bottom_layout_zoom_percent(self, df_settings) -> int:
        return self.model.bottom_layout_zoom_percent(df_settings)

    def retain_space_hidden_sliders(self, df_settings) -> bool:
        return self.model.retain_space_hidden_sliders(df_settings)
