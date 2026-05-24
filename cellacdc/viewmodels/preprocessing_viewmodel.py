"""View-model contracts for image preprocessing recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cellacdc.models.preprocessing_model import PreprocessingModel


@dataclass(frozen=True)
class PreprocessingViewModel:
    """Application-facing commands for preprocessing recipe execution."""

    model: PreprocessingModel = field(default_factory=PreprocessingModel)

    def validate_multidimensional_recipe(
        self,
        recipe: list[dict[str, Any]],
        *,
        apply_to_all_zslices: bool = False,
        apply_to_all_frames: bool = False,
    ):
        return self.model.validate_multidimensional_recipe(
            recipe,
            apply_to_all_zslices=apply_to_all_zslices,
            apply_to_all_frames=apply_to_all_frames,
        )

    def preprocess_image_from_recipe(self, image, recipe: list[dict[str, Any]]):
        return self.model.preprocess_image_from_recipe(image, recipe)

    def preprocess_zstack_from_recipe(self, image, recipe: list[dict[str, Any]]):
        return self.model.preprocess_zstack_from_recipe(image, recipe)

    def preprocess_video_from_recipe(self, image, recipe: list[dict[str, Any]]):
        return self.model.preprocess_video_from_recipe(image, recipe)

    def preprocess_multi_pos_from_recipe(
        self,
        images,
        recipe: list[dict[str, Any]],
    ):
        return self.model.preprocess_multi_pos_from_recipe(images, recipe)

    def image_to_float(
        self,
        image,
        *,
        force_dtype=None,
        force_missing_dtype=None,
        warn=True,
    ):
        return self.model.image_to_float(
            image,
            force_dtype=force_dtype,
            force_missing_dtype=force_missing_dtype,
            warn=warn,
        )

    def normalize_display_image(self, image, how: str):
        return self.model.normalize_display_image(image, how)

    def create_preprocessed_data(self, image_data=None):
        return self.model.create_preprocessed_data(image_data=image_data)
