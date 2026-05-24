"""View-model contracts for curvature and spline editing tools."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cellacdc.domain.curvature import CurvatureLabelPaintResult
from cellacdc.models.curvature_model import CurvatureModel


@dataclass(frozen=True)
class CurvatureViewModel:
    """Application-facing commands for spline drawing and label painting."""

    model: CurvatureModel = field(default_factory=CurvatureModel)

    def tangent_brush_polygon(
        self,
        yx_start,
        yx_end,
        radius: int | float,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.model.tangent_brush_polygon(
            yx_start, yx_end, radius, shape
        )

    def directional_coords(
        self,
        alfa_dir: int,
        y: int,
        x: int,
        shape: tuple[int, int],
        *,
        connectivity: int = 1,
    ) -> tuple[list[int], list[int]]:
        return self.model.directional_coords(
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
        return self.model.spline_coords(
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
        return self.model.closed_spline_coords(
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
        return self.model.paint_spline_to_labels(
            labels_2d,
            xx_spline,
            yy_spline,
            label_id,
            empty_only=empty_only,
        )
