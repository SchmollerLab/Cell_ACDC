"""Scriptable model rules for promptable segmentation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MagicPromptZoom:
    """Computed zoom region for promptable segmentation."""

    bounds: tuple[int, int, int, int]
    image_origin: tuple[int, int, int]
    zoom_slice: tuple[slice, slice]


class MagicPromptsModel:
    """Headless promptable-segmentation geometry and point rules."""

    def zoom_region(self, view_range, image_shape) -> MagicPromptZoom:
        (xmin, xmax), (ymin, ymax) = view_range
        height, width = image_shape[-2:]

        xmin = int(max(0, xmin))
        xmax = int(min(width, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(height, ymax))

        return MagicPromptZoom(
            bounds=(xmin, xmax, ymin, ymax),
            image_origin=(0, ymin, xmin),
            zoom_slice=(slice(ymin, ymax), slice(xmin, xmax)),
        )

    def points_in_zoom(self, df_points, zoom: MagicPromptZoom, frame_i):
        xmin, xmax, ymin, ymax = zoom.bounds
        filtered = df_points[
            (df_points['y'] >= ymin)
            & (df_points['x'] >= xmin)
            & (df_points['y'] < ymax)
            & (df_points['x'] < xmax)
            & (df_points['frame_i'] == frame_i)
        ].copy()
        filtered['y'] -= ymin
        filtered['x'] -= xmin
        return filtered

    def retained_points_outside_zoom(
        self,
        frame_points_data: Mapping,
        zoom: MagicPromptZoom,
    ):
        if 'x' in frame_points_data:
            return self._retained_points_outside_zoom_2d(
                frame_points_data,
                zoom,
            )

        return {
            z: self._retained_points_outside_zoom_2d(z_points, zoom)
            for z, z_points in frame_points_data.items()
        }

    def _retained_points_outside_zoom_2d(self, points_data, zoom):
        xmin, xmax, ymin, ymax = zoom.bounds
        retained = {'x': [], 'y': [], 'id': []}
        for x, y, point_id in zip(
            points_data['x'],
            points_data['y'],
            points_data['id'],
        ):
            if x < xmin or x >= xmax or y < ymin or y >= ymax:
                retained['x'].append(x)
                retained['y'].append(y)
                retained['id'].append(point_id)
        return retained
