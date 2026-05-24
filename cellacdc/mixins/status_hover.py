"""View adapter for hover and status-bar formatting."""

from __future__ import annotations


import math
import os
import re


class StatusHoverMixin:
    """Qt-facing adapter around status/hover view-model contracts."""

    """Headless status-bar and hover formatting rules."""

    def active_tool_button(self):
        for button in self.LeftClickButtons:
            if button.isChecked():
                return button

    def add_overlay_hover_values_formatted(self, txt, xdata, ydata):
        pos_data = self.data[self.pos_i]
        if pos_data.ol_data is None:
            return txt

        for filename in pos_data.ol_data:
            ch_name = self.view_model.formatting.channel_name_from_basename(
                filename, pos_data.basename, remove_ext=False
            )
            if ch_name not in self.checkedOverlayChannels:
                continue

            raw_overlay_img = self.getRawImage(filename=filename)
            raw_overlay_value = raw_overlay_img[ydata, xdata]
            raw_txt = self.channel_hover_values("Raw", ch_name, raw_overlay_value)
            txt = f"{txt} | {raw_txt}"
        return txt

    def add_ruler_measurement_text(self, txt):
        pos_data = self.data[self.pos_i]
        xx, yy = self.ax1_rulerPlotItem.getData()
        if xx is None:
            return txt

        length_pixels = self.euclidean_length(xx, yy)
        depth_axes = self.switchPlaneCombobox.depthAxes()
        if depth_axes != "z":
            pixel_to_um = pos_data.PhysicalSizeZ
        else:
            pixel_to_um = pos_data.PhysicalSizeX

        length_txt = self.ruler_measurement_text(
            length_pixels=length_pixels,
            pixel_to_um=pixel_to_um,
        )
        return f"{txt} | <b>Measurement</b>: {length_txt}"

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
            f"x={x:d}, y={y:d} | "
            f"W={width:d}, H={height:d} | "
            f"x_left={x_left:d}, y_top={y_top:d} | "
            f"x_right={x_right:d}, y_bottom={y_bottom:d} | "
            f"(ax{axis_index})"
        )

    def channel_hover_text(self, description, channel, value, format_spec):
        return f"<b>{description} {channel}</b>: value={value:{format_spec}}"

    def channel_hover_values(self, descr, channel, value, ff=None):
        if ff is None:
            n_digits = len(str(int(value)))
            ff = self.view_model.formatting.number_fstring_formatter(
                type(value), precision=abs(n_digits - 5)
            )
        return self.channel_hover_text(descr, channel, value, ff)

    def check_highlight_scale_bar(self, x, y, active_tool_button):
        if not hasattr(self, "scaleBar"):
            return
        highlighted = self.highlight_state(
            x=x,
            y=y,
            bbox=self.scaleBar.bbox(),
            enabled=self.addScaleBarAction.isChecked(),
            active_tool=active_tool_button,
        )
        if highlighted is None:
            return
        self.scaleBar.setHighlighted(highlighted)

    def check_highlight_timestamp(self, x, y, active_tool_button):
        if not hasattr(self, "timestamp"):
            return
        blocked_by_scale_bar = (
            hasattr(self, "scaleBar") and self.scaleBar.isHighlighted()
        )
        highlighted = self.highlight_state(
            x=x,
            y=y,
            bbox=self.timestamp.bbox(),
            enabled=self.addTimestampAction.isChecked(),
            active_tool=active_tool_button,
            blocked_by_other_highlight=blocked_by_scale_bar,
        )
        if highlighted is None:
            return
        self.timestamp.setHighlighted(highlighted)

    def concat_acdc_df(self):
        pos_data = self.data[self.pos_i]
        return self.view_model.frame_metadata.concat_visited_acdc_frames(
            pos_data.allData_li
        )

    def euclidean_length(self, x_values, y_values):
        return math.sqrt(
            (x_values[0] - x_values[1]) ** 2 + (y_values[0] - y_values[1]) ** 2
        )

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

    def hover_values_formatted(self, xdata, ydata, active_tool_button, is_ax0):
        (xl, xr), (yt, yb) = self.display_decorations_view.ax1_view_range(integers=True)
        width = round(xr - xl)
        height = round(yb - yt)
        axis_index = 0 if is_ax0 else 1
        txt = self.base_hover_text(
            x=xdata,
            y=ydata,
            width=width,
            height=height,
            x_left=xl,
            y_top=yt,
            x_right=xr,
            y_bottom=yb,
            axis_index=axis_index,
        )
        if active_tool_button == self.rulerButton:
            return self.add_ruler_measurement_text(txt)
        if active_tool_button is not None:
            return txt

        pos_data = self.data[self.pos_i]
        raw_img = self.getRawImage()
        raw_value = raw_img[ydata, xdata]
        raw_txt = self.channel_hover_values("Raw", self.user_ch_name, raw_value)
        txt = f"{txt} | {raw_txt}"
        txt = self.add_overlay_hover_values_formatted(txt, xdata, ydata)

        label_id = self.currentLab2D[ydata, xdata]
        label_txt = self.object_hover_text(
            label_id=label_id,
            max_id=max(pos_data.IDs, default=0),
            object_count=len(pos_data.IDs),
        )
        txt = f"{txt} | {label_txt}"
        return self.add_ruler_measurement_text(txt)

    def mouse_data_coords_right_image(self):
        return self.mouse_data_coords_right_image(self.wcLabel.text())

    def object_hover_text(self, *, label_id, max_id, object_count):
        return (
            f"<b>Objects</b>: ID={label_id}, <i>max ID={max_id}, "
            f"num. of objects={object_count}</i>"
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
            r"W=.*?, H=.*? \| "
            r"x_left=.*?, y_top=.*? \| "
            r"x_right=.*?, y_bottom=.*? \| "
        )
        replacing = (
            f"W={width:d}, H={height:d} | "
            f"x_left={x_left:d}, y_top={y_top:d} | "
            f"x_right={x_right:d}, y_bottom={y_bottom:d} | "
        )
        return re.sub(pattern, replacing, text)

    def ruler_length_text(self):
        return self.ruler_length_text(self.wcLabel.text())

    def ruler_measurement_text(self, *, length_pixels, pixel_to_um):
        return (
            f"length = {int(length_pixels)} pxl ({length_pixels * pixel_to_um:.2f} μm)"
        )

    def set_status_bar_label(self, log=True):
        self.statusbar.clearMessage()
        pos_data = self.data[self.pos_i]
        txt = self.status_bar_text(
            pos_foldername=pos_data.pos_foldername,
            basename=pos_data.basename,
            filename=pos_data.filename,
            segm_npz_path=pos_data.segm_npz_path,
        )
        if log:
            self.logger.info(txt)
        self.statusBarLabel.setText(txt)

    def status_bar_text(
        self,
        *,
        pos_foldername,
        basename,
        filename,
        segm_npz_path,
    ):
        segmented_channel_name = filename[len(basename) :]
        segm_filename = os.path.basename(segm_npz_path)
        segm_end_name = segm_filename[len(basename) :]
        return (
            f"{pos_foldername} || "
            f"Basename: {basename} || "
            f"Segmented channel: {segmented_channel_name} || "
            f"Segmentation file name: {segm_end_name}"
        )

    def update_values_status_bar(self):
        (xl, xr), (yt, yb) = self.display_decorations_view.ax1_view_range(integers=True)
        width = round(xr - xl)
        height = round(yb - yt)
        txt = self.replace_view_range_status(
            self.wcLabel.text(),
            width=width,
            height=height,
            x_left=xl,
            y_top=yt,
            x_right=xr,
            y_bottom=yb,
        )
        self.wcLabel.setText(txt)
