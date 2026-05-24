"""Headless rules and helpers for the Combine Channels feature."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CombineModel:
    """Headless state and helpers for combining channel and image arrays."""

    def initialize_combine_image_data(self, pos_data) -> np.ndarray:
        """Initializes pos_data.combine_img_data if not already present."""
        if not hasattr(pos_data, 'combine_img_data'):
            from cellacdc import preprocess
            pos_data.combine_img_data = preprocess.PreprocessedData(
                image_data=np.zeros(pos_data.img_data.shape)
            )
        return pos_data.combine_img_data

    def validate_dimensions(self, ndim: int) -> bool:
        """Asserts that image data dimensions are valid for combining (3D or 4D)."""
        if ndim not in (3, 4):
            raise ValueError('Invalid number of dimensions in img_data.')
        return True

    def group_processed_data_by_pos(
        self,
        processed_data: list[np.ndarray],
        keys: list[tuple[int, int, int]]
    ) -> dict[int, list[tuple[tuple[int, int, int], np.ndarray]]]:
        """Groups raw processed preview output arrays by position index."""
        unique_pos = {key[0] for key in keys}
        per_pos_data = {pos_i: [] for pos_i in unique_pos}
        for key, img in zip(keys, processed_data):
            pos_i, frame_i, z_slice = key
            per_pos_data[pos_i].append((key, img))
        return per_pos_data

    def update_combine_image_data(
        self,
        pos_data,
        pos_i_data: list[tuple[tuple[int, int, int], np.ndarray]]
    ):
        """Updates preprocessed combined image data container frames and z-slices."""
        n_dim_img = pos_data.img_data.ndim
        self.initialize_combine_image_data(pos_data)
        self.validate_dimensions(n_dim_img)

        if n_dim_img == 4:
            for key, img in pos_i_data:
                _, frame_i, z_slice = key
                pos_data.combine_img_data[frame_i][z_slice] = img
        elif n_dim_img == 3:
            for key, img in pos_i_data:
                _, frame_i, _ = key
                pos_data.combine_img_data[frame_i] = img
