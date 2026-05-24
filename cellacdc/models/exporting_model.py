"""Scriptable model rules for image and video export workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import skimage.measure
import skimage.segmentation


@dataclass(frozen=True)
class ExportFramePlan:
    """Destination naming for one exported video frame."""

    frame_index_text: str
    png_filename: str
    png_filepath: str


class ExportingModel:
    """Headless export naming, mask, and zoom selection rules."""

    def timestamped_export_filename(self, kind: str, *, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        return f"{timestamp.strftime('%Y%m%d_%H%M%S')}_acdc_exported_{kind}"

    def export_frame_plan(
        self,
        *,
        current_index: int,
        num_digits: int,
        filename: str,
        pngs_folderpath: str,
    ) -> ExportFramePlan:
        frame_index_text = str(current_index).zfill(num_digits)
        png_filename = f'{frame_index_text}_{filename}.png'
        return ExportFramePlan(
            frame_index_text=frame_index_text,
            png_filename=png_filename,
            png_filepath=os.path.join(pngs_folderpath, png_filename),
        )

    def export_mask_image_shape(self, image_shape) -> tuple[int, int, int]:
        height, width = image_shape[-2:]
        return height, width, 4

    def build_export_mask_image(
        self,
        image_shape,
        view_range,
        *,
        invert_bw=False,
    ):
        mask_image = np.zeros(
            self.export_mask_image_shape(image_shape),
            dtype=np.uint8,
        )
        x_range, y_range = view_range
        x0, x1 = map(round, x_range)
        y0, y1 = map(round, y_range)

        if invert_bw:
            mask_image[:, :, :3] = 255

        if x0 > 0:
            mask_image[:, :x0, 3] = 255
        if x1 < mask_image.shape[1]:
            mask_image[:, x1:, 3] = 255
        if y0 > 0:
            mask_image[:y0, :, 3] = 255
        if y1 < mask_image.shape[0]:
            mask_image[y1:, :, 3] = 255

        return mask_image

    def zoom_ids(self, labels_2d, view_range):
        height, width = labels_2d.shape
        ((xmin, xmax), (ymin, ymax)) = view_range
        if xmin <= 0 and ymin <= 0 and xmax >= width and ymax >= height:
            return None

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, width)
        ymax = min(ymax, height)

        zoom_slice = (
            slice(round(ymin), round(ymax)),
            slice(round(xmin), round(xmax)),
        )
        zoom_labels = skimage.segmentation.clear_border(labels_2d[zoom_slice])
        zoom_regionprops = skimage.measure.regionprops(zoom_labels)
        return [obj.label for obj in zoom_regionprops]

    def shifted_view_range(self, previous_range, current_range, window_range):
        prev_x_range, prev_y_range = previous_range
        curr_x_range, curr_y_range = current_range
        win_x_range, win_y_range = window_range

        delta_x = curr_x_range[0] - prev_x_range[0]
        delta_y = curr_y_range[0] - prev_y_range[0]

        return (
            (win_x_range[0] + delta_x, win_x_range[1] + delta_x),
            (win_y_range[0] + delta_y, win_y_range[1] + delta_y),
        )
