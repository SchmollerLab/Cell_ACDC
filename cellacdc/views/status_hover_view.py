"""View adapter for hover and status-bar formatting."""

from __future__ import annotations

from cellacdc.viewmodels.status_hover_viewmodel import StatusHoverViewModel


class StatusHoverView:
    """Qt-facing adapter around status/hover view-model contracts."""

    def __init__(self, host, view_model: StatusHoverViewModel):
        self.host = host
        self.view_model = view_model

    def channel_hover_values(self, descr, channel, value, ff=None):
        if ff is None:
            n_digits = len(str(int(value)))
            ff = self.host.view_model.formatting.number_fstring_formatter(
                type(value), precision=abs(n_digits-5)
            )
        return self.view_model.channel_hover_text(descr, channel, value, ff)

    def add_overlay_hover_values_formatted(self, txt, xdata, ydata):
        pos_data = self.host.data[self.host.pos_i]
        if pos_data.ol_data is None:
            return txt

        for filename in pos_data.ol_data:
            ch_name = (
                self.host.view_model.formatting.channel_name_from_basename(
                    filename, pos_data.basename, remove_ext=False
                )
            )
            if ch_name not in self.host.checkedOverlayChannels:
                continue

            raw_overlay_img = self.host.getRawImage(filename=filename)
            raw_overlay_value = raw_overlay_img[ydata, xdata]
            raw_txt = self.channel_hover_values(
                'Raw', ch_name, raw_overlay_value
            )
            txt = f'{txt} | {raw_txt}'
        return txt

    def active_tool_button(self):
        for button in self.host.LeftClickButtons:
            if button.isChecked():
                return button

    def concat_acdc_df(self):
        pos_data = self.host.data[self.host.pos_i]
        return self.host.view_model.frame_metadata.concat_visited_acdc_frames(
            pos_data.allData_li
        )

    def check_highlight_timestamp(self, x, y, active_tool_button):
        if not hasattr(self.host, 'timestamp'):
            return
        blocked_by_scale_bar = (
            hasattr(self.host, 'scaleBar')
            and self.host.scaleBar.isHighlighted()
        )
        highlighted = self.view_model.highlight_state(
            x=x,
            y=y,
            bbox=self.host.timestamp.bbox(),
            enabled=self.host.addTimestampAction.isChecked(),
            active_tool=active_tool_button,
            blocked_by_other_highlight=blocked_by_scale_bar,
        )
        if highlighted is None:
            return
        self.host.timestamp.setHighlighted(highlighted)

    def check_highlight_scale_bar(self, x, y, active_tool_button):
        if not hasattr(self.host, 'scaleBar'):
            return
        highlighted = self.view_model.highlight_state(
            x=x,
            y=y,
            bbox=self.host.scaleBar.bbox(),
            enabled=self.host.addScaleBarAction.isChecked(),
            active_tool=active_tool_button,
        )
        if highlighted is None:
            return
        self.host.scaleBar.setHighlighted(highlighted)

    def mouse_data_coords_right_image(self):
        return self.view_model.mouse_data_coords_right_image(
            self.host.wcLabel.text()
        )

    def update_values_status_bar(self):
        (xl, xr), (yt, yb) = (
            self.host.display_decorations_view.ax1_view_range(integers=True)
        )
        width = round(xr - xl)
        height = round(yb - yt)
        txt = self.view_model.replace_view_range_status(
            self.host.wcLabel.text(),
            width=width,
            height=height,
            x_left=xl,
            y_top=yt,
            x_right=xr,
            y_bottom=yb,
        )
        self.host.wcLabel.setText(txt)

    def hover_values_formatted(self, xdata, ydata, active_tool_button, is_ax0):
        (xl, xr), (yt, yb) = (
            self.host.display_decorations_view.ax1_view_range(integers=True)
        )
        width = round(xr - xl)
        height = round(yb - yt)
        axis_index = 0 if is_ax0 else 1
        txt = self.view_model.base_hover_text(
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
        if active_tool_button == self.host.rulerButton:
            return self.add_ruler_measurement_text(txt)
        if active_tool_button is not None:
            return txt

        pos_data = self.host.data[self.host.pos_i]
        raw_img = self.host.getRawImage()
        raw_value = raw_img[ydata, xdata]
        raw_txt = self.channel_hover_values(
            'Raw', self.host.user_ch_name, raw_value
        )
        txt = f'{txt} | {raw_txt}'
        txt = self.add_overlay_hover_values_formatted(txt, xdata, ydata)

        label_id = self.host.currentLab2D[ydata, xdata]
        label_txt = self.view_model.object_hover_text(
            label_id=label_id,
            max_id=max(pos_data.IDs, default=0),
            object_count=len(pos_data.IDs),
        )
        txt = f'{txt} | {label_txt}'
        return self.add_ruler_measurement_text(txt)

    def ruler_length_text(self):
        return self.view_model.ruler_length_text(self.host.wcLabel.text())

    def add_ruler_measurement_text(self, txt):
        pos_data = self.host.data[self.host.pos_i]
        xx, yy = self.host.ax1_rulerPlotItem.getData()
        if xx is None:
            return txt

        length_pixels = self.view_model.euclidean_length(xx, yy)
        depth_axes = self.host.switchPlaneCombobox.depthAxes()
        if depth_axes != 'z':
            pixel_to_um = pos_data.PhysicalSizeZ
        else:
            pixel_to_um = pos_data.PhysicalSizeX

        length_txt = self.view_model.ruler_measurement_text(
            length_pixels=length_pixels,
            pixel_to_um=pixel_to_um,
        )
        return f'{txt} | <b>Measurement</b>: {length_txt}'

    def set_status_bar_label(self, log=True):
        self.host.statusbar.clearMessage()
        pos_data = self.host.data[self.host.pos_i]
        txt = self.view_model.status_bar_text(
            pos_foldername=pos_data.pos_foldername,
            basename=pos_data.basename,
            filename=pos_data.filename,
            segm_npz_path=pos_data.segm_npz_path,
        )
        if log:
            self.host.logger.info(txt)
        self.host.statusBarLabel.setText(txt)
