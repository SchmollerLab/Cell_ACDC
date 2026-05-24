"""View-model contracts for promptable segmentation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.magic_prompts_model import (
    MagicPromptsModel,
    MagicPromptZoom,
)
from cellacdc.viewmodels.model_registry import ModelRegistryViewModel


@dataclass(frozen=True)
class MagicPromptsViewModel:
    """Application-facing promptable-segmentation commands."""

    model: MagicPromptsModel = field(default_factory=MagicPromptsModel)
    registry: ModelRegistryViewModel = field(
        default_factory=ModelRegistryViewModel
    )

    def store_custom_promptable_model_path(self, model_file_path):
        return self.registry.store_custom_promptable_model_path(
            model_file_path
        )

    def init_prompt_segmentation_model(
        self,
        acdc_prompt_segment,
        position_data,
        init_kwargs,
    ):
        return self.registry.init_prompt_segmentation_model(
            acdc_prompt_segment,
            position_data,
            init_kwargs,
        )

    def set_default_arg_specs_from_kwargs(self, params, kwargs):
        return self.registry.set_default_arg_specs_from_kwargs(params, kwargs)

    def zoom_region(self, view_range, image_shape) -> MagicPromptZoom:
        return self.model.zoom_region(view_range, image_shape)

    def points_in_zoom(self, df_points, zoom: MagicPromptZoom, frame_i):
        return self.model.points_in_zoom(df_points, zoom, frame_i)

    def retained_points_outside_zoom(
        self,
        frame_points_data,
        zoom: MagicPromptZoom,
    ):
        return self.model.retained_points_outside_zoom(
            frame_points_data,
            zoom,
        )
