"""Scriptable model rules for hover and status-bar text."""

from __future__ import annotations

import math
import os
import re


class StatusHoverModel:
    """Headless status-bar and hover formatting rules."""

    def channel_hover_text(self, description, channel, value, format_spec):
        return f'<b>{description} {channel}</b>: value={value:{format_spec}}'

    def object_hover_text(self, *, label_id, max_id, object_count):
        return (
            f'<b>Objects</b>: ID={label_id}, <i>max ID={max_id}, '
            f'num. of objects={object_count}</i>'
        )

    def base_hover_text(
        self,
        *,
        x,
        y,
        width,
        height,
        x_left,
        y_top,
        x_right,
        y_bottom,
        axis_index,
    ):
        return (
            f'x={x:d}, y={y:d} | '
            f'W={width:d}, H={height:d} | '
            f'x_left={x_left:d}, y_top={y_top:d} | '
            f'x_right={x_right:d}, y_bottom={y_bottom:d} | '
            f'(ax{axis_index})'
        )

    def replace_view_range_status(
        self,
        text,
        *,
        width,
        height,
        x_left,
        y_top,
        x_right,
        y_bottom,
    ):
        pattern = (
            r'W=.*?, H=.*? \| '
            r'x_left=.*?, y_top=.*? \| '
            r'x_right=.*?, y_bottom=.*? \| '
        )
        replacing = (
            f'W={width:d}, H={height:d} | '
            f'x_left={x_left:d}, y_top={y_top:d} | '
            f'x_right={x_right:d}, y_bottom={y_bottom:d} | '
        )
        return re.sub(pattern, replacing, text)

    def highlight_state(
        self,
        *,
        x,
        y,
        bbox,
        enabled,
        active_tool,
        blocked_by_other_highlight=False,
    ):
        if not enabled or active_tool is not None or blocked_by_other_highlight:
            return None
        y_min, x_min, y_max, x_max = bbox
        return x_min <= x <= x_max and y_min <= y <= y_max

    def mouse_data_coords_right_image(self, text):
        if not text:
            return None
        ax_idx = int(re.findall(r'\(ax(\d)\)', text)[0])
        if ax_idx == 0:
            return None
        coords = re.findall(r'x=(\d+), y=(\d+) \|', text)[0]
        return tuple([int(val) for val in coords])

    def ruler_length_text(self, text):
        length_text = re.findall(r'length = (.*)\)', text)[0]
        length_text = length_text.replace('pxl', 'pixels')
        return f'{length_text})'

    def ruler_measurement_text(self, *, length_pixels, pixel_to_um):
        return (
            f'length = {int(length_pixels)} pxl '
            f'({length_pixels*pixel_to_um:.2f} μm)'
        )

    def euclidean_length(self, x_values, y_values):
        return math.sqrt(
            (x_values[0]-x_values[1])**2 + (y_values[0]-y_values[1])**2
        )

    def status_bar_text(
        self,
        *,
        pos_foldername,
        basename,
        filename,
        segm_npz_path,
    ):
        segmented_channel_name = filename[len(basename):]
        segm_filename = os.path.basename(segm_npz_path)
        segm_end_name = segm_filename[len(basename):]
        return (
            f'{pos_foldername} || '
            f'Basename: {basename} || '
            f'Segmented channel: {segmented_channel_name} || '
            f'Segmentation file name: {segm_end_name}'
        )
