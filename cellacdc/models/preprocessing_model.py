"""Scriptable model commands for image preprocessing recipes."""

from __future__ import annotations

from typing import Any

from cellacdc.core import (
    preprocess_image_from_recipe as core_preprocess_image_from_recipe,
    preprocess_multi_pos_from_recipe as core_preprocess_multi_pos_from_recipe,
    preprocess_video_from_recipe as core_preprocess_video_from_recipe,
    preprocess_zstack_from_recipe as core_preprocess_zstack_from_recipe,
    validate_multidimensional_recipe as core_validate_multidimensional_recipe,
)
from cellacdc.domain.display_images import normalize_display_image
from cellacdc.myutils import img_to_float
from cellacdc.preprocess import PreprocessedData


class PreprocessingModel:
    """Headless preprocessing operations used by GUI and scripts."""

    def validate_multidimensional_recipe(
        self,
        recipe: list[dict[str, Any]],
        *,
        apply_to_all_zslices: bool = False,
        apply_to_all_frames: bool = False,
    ):
        return core_validate_multidimensional_recipe(
            recipe,
            apply_to_all_zslices=apply_to_all_zslices,
            apply_to_all_frames=apply_to_all_frames,
        )

    def preprocess_image_from_recipe(self, image, recipe: list[dict[str, Any]]):
        return core_preprocess_image_from_recipe(image, recipe)

    def preprocess_zstack_from_recipe(self, image, recipe: list[dict[str, Any]]):
        return core_preprocess_zstack_from_recipe(image, recipe)

    def preprocess_video_from_recipe(self, image, recipe: list[dict[str, Any]]):
        return core_preprocess_video_from_recipe(image, recipe)

    def preprocess_multi_pos_from_recipe(
        self,
        images,
        recipe: list[dict[str, Any]],
    ):
        return core_preprocess_multi_pos_from_recipe(images, recipe)

    def image_to_float(
        self,
        image,
        *,
        force_dtype=None,
        force_missing_dtype=None,
        warn=True,
    ):
        return img_to_float(
            image,
            force_dtype=force_dtype,
            force_missing_dtype=force_missing_dtype,
            warn=warn,
        )

    def normalize_display_image(self, image, how: str):
        return normalize_display_image(
            image,
            how,
            image_to_float=self.image_to_float,
        )

    def create_preprocessed_data(self, image_data=None):
        return PreprocessedData(image_data=image_data)
