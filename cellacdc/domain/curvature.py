"""Pure curvature and spline editing operations (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import skimage.draw


@dataclass(frozen=True)
class CurvatureLabelPaintResult:
    """Result of painting a closed spline into a label image."""

    labels_2d: np.ndarray
    mask: np.ndarray
    painted_pixels: int


def tangent_brush_polygon(
    yx_start,
    yx_end,
    radius: int | float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return polygon coords joining two circular brush centers."""
    y1, x1 = yx_start
    y2, x2 = yx_end
    radius = float(radius)

    arcsin_den = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    arctan_den = x2 - x1
    if arcsin_den == 0 or arctan_den == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    beta = np.arcsin((radius - radius) / arcsin_den)
    gamma = -np.arctan((y2 - y1) / arctan_den)
    alpha = gamma - beta
    x3 = x1 + radius * np.sin(alpha)
    y3 = y1 + radius * np.cos(alpha)
    x4 = x2 + radius * np.sin(alpha)
    y4 = y2 + radius * np.cos(alpha)

    alpha = gamma + beta
    x5 = x1 - radius * np.sin(alpha)
    y5 = y1 - radius * np.cos(alpha)
    x6 = x2 - radius * np.sin(alpha)
    y6 = y2 - radius * np.cos(alpha)

    return skimage.draw.polygon(
        [y3, y4, y6, y5],
        [x3, x4, x6, x5],
        shape=shape,
    )


def directional_coords(
    alfa_dir: int,
    y: int,
    x: int,
    shape: tuple[int, int],
    *,
    connectivity: int = 1,
) -> tuple[list[int], list[int]]:
    height, width = shape
    y_above = y + 1 if y + 1 < height else y
    y_below = y - 1 if y > 0 else y
    x_right = x + 1 if x + 1 < width else x
    x_left = x - 1 if x > 0 else x

    if alfa_dir == 0:
        yy = [y_below, y_below, y, y_above, y_above]
        xx = [x, x_right, x_right, x_right, x]
    elif alfa_dir == 45:
        yy = [y_below, y_below, y_below, y, y_above]
        xx = [x_left, x, x_right, x_right, x_right]
    elif alfa_dir == 90:
        yy = [y, y_below, y_below, y_below, y]
        xx = [x_left, x_left, x, x_right, x_right]
    elif alfa_dir == 135:
        yy = [y_above, y, y_below, y_below, y_below]
        xx = [x_left, x_left, x_left, x, x_right]
    elif alfa_dir == -180 or alfa_dir == 180:
        yy = [y_above, y_above, y, y_below, y_below]
        xx = [x, x_left, x_left, x_left, x]
    elif alfa_dir == -135:
        yy = [y_below, y, y_above, y_above, y_above]
        xx = [x_left, x_left, x_left, x, x_right]
    elif alfa_dir == -90:
        yy = [y, y_above, y_above, y_above, y]
        xx = [x_left, x_left, x, x_right, x_right]
    else:
        yy = [y_above, y_above, y_above, y, y_below]
        xx = [x_left, x, x_right, x_right, x_right]

    if connectivity == 1:
        return yy[1:4], xx[1:4]
    return yy, xx


def spline_coords(
    xx,
    yy,
    *,
    resolution_space=None,
    per: bool = False,
    append_first: bool = False,
):
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    if len(xx) == 0 or len(yy) == 0:
        return [], []

    valid = np.where(np.abs(np.diff(xx)) + np.abs(np.diff(yy)) > 0)
    xx = np.r_[xx[valid], xx[-1]]
    yy = np.r_[yy[valid], yy[-1]]
    if append_first:
        xx = np.r_[xx, xx[0]]
        yy = np.r_[yy, yy[0]]
        per = True

    if resolution_space is None:
        resolution_space = np.linspace(0, 1, 1000)
    k = 2 if len(xx) == 3 else 3

    try:
        tck, _u = scipy.interpolate.splprep([xx, yy], s=0, k=k, per=per)
        return scipy.interpolate.splev(resolution_space, tck)
    except (ValueError, TypeError):
        return [], []


def closed_spline_coords(
    xx_spline,
    yy_spline,
    *,
    anchor_xx=None,
    anchor_yy=None,
    predictor=None,
    max_exec_time: int = 150,
):
    xx_spline = np.asarray(xx_spline)
    yy_spline = np.asarray(yy_spline)
    bbox_area = (
        (xx_spline.max() - xx_spline.min())
        * (yy_spline.max() - yy_spline.min())
    )
    if bbox_area < 26_000:
        return xx_spline, yy_spline

    if predictor is None or anchor_xx is None or anchor_yy is None:
        return xx_spline, yy_spline

    optimal_space_size = predictor.predict(
        bbox_area,
        max_exec_time=max_exec_time,
    )
    if optimal_space_size >= 1000:
        return xx_spline, yy_spline

    if optimal_space_size < 100:
        optimal_space_size = 100

    resolution_space = np.linspace(0, 1, int(optimal_space_size))
    return spline_coords(
        anchor_xx,
        anchor_yy,
        resolution_space=resolution_space,
        per=True,
    )


def paint_spline_to_labels(
    labels_2d: np.ndarray,
    xx_spline,
    yy_spline,
    label_id: int,
    *,
    empty_only: bool = True,
) -> CurvatureLabelPaintResult:
    updated_labels = labels_2d.copy()
    mask = np.zeros(updated_labels.shape, bool)
    rr, cc = skimage.draw.polygon(
        yy_spline,
        xx_spline,
        shape=updated_labels.shape,
    )
    mask[rr, cc] = True
    if empty_only:
        mask[updated_labels != 0] = False

    updated_labels[mask] = int(label_id)
    return CurvatureLabelPaintResult(
        labels_2d=updated_labels,
        mask=mask,
        painted_pixels=int(np.count_nonzero(mask)),
    )
