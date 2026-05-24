"""Scriptable model rules for curvature and spline editing tools."""

from __future__ import annotations

import numpy as np

from cellacdc.domain.curvature import (
    CurvatureLabelPaintResult,
    closed_spline_coords,
    directional_coords,
    paint_spline_to_labels,
    spline_coords,
    tangent_brush_polygon,
)


class CurvatureModel:
    """Headless spline drawing and label-painting operations."""

    def tangent_brush_polygon(
        self,
        yx_start,
        yx_end,
        radius: int | float,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return tangent_brush_polygon(yx_start, yx_end, radius, shape)

    def directional_coords(
        self,
        alfa_dir: int,
        y: int,
        x: int,
        shape: tuple[int, int],
        *,
        connectivity: int = 1,
    ) -> tuple[list[int], list[int]]:
        return directional_coords(
            alfa_dir,
            y,
            x,
            shape,
            connectivity=connectivity,
        )

    def spline_coords(
        self,
        xx,
        yy,
        *,
        resolution_space=None,
        per: bool = False,
        append_first: bool = False,
    ):
        return spline_coords(
            xx,
            yy,
            resolution_space=resolution_space,
            per=per,
            append_first=append_first,
        )

    def closed_spline_coords(
        self,
        xx_spline,
        yy_spline,
        *,
        anchor_xx=None,
        anchor_yy=None,
        predictor=None,
        max_exec_time: int = 150,
    ):
        return closed_spline_coords(
            xx_spline,
            yy_spline,
            anchor_xx=anchor_xx,
            anchor_yy=anchor_yy,
            predictor=predictor,
            max_exec_time=max_exec_time,
        )

    def paint_spline_to_labels(
        self,
        labels_2d: np.ndarray,
        xx_spline,
        yy_spline,
        label_id: int,
        *,
        empty_only: bool = True,
    ) -> CurvatureLabelPaintResult:
        return paint_spline_to_labels(
            labels_2d,
            xx_spline,
            yy_spline,
            label_id,
            empty_only=empty_only,
        )
