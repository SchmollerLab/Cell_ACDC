"""View adapter for timestamp, scale-bar, and view-range decorations."""

from __future__ import annotations

import numpy as np

from cellacdc import apps, widgets


class DisplayDecorationsView:
    """Qt-facing adapter around display-decoration contracts."""

    """Headless display-decoration decision rules."""

    def clamped_view_range(self, image_shape, view_range):
        y_size, x_size = image_shape[:2]
        x_range, y_range = view_range
        x_min = 0 if x_range[0] < 0 else x_range[0]
        y_min = 0 if y_range[0] < 0 else y_range[0]
        x_max = x_size if x_range[1] >= x_size else x_range[1]
        y_max = y_size if y_range[1] >= y_size else y_range[1]
        return int(y_min), int(y_max), int(x_min), int(x_max)

    def integer_view_range(self, view_range):
        x_range, y_range = view_range
        return (
            [round(x_range[0]), round(x_range[1])],
            [round(y_range[0]), round(y_range[1])],
        )

    def should_move_decoration(
        self,
        *,
        dialog_open: bool,
        move_with_zoom: bool,
    ) -> bool:
        return dialog_open or move_with_zoom

    def should_store_view_range(
        self,
        *,
        has_range_reset_state: bool,
        is_range_reset: bool = False,
    ) -> bool:
        return has_range_reset_state and is_range_reset

    def should_update_timestamp_frame(
        self,
        *,
        has_timestamp: bool,
        timestamp_enabled: bool,
    ) -> bool:
        return has_timestamp and timestamp_enabled


    def __init__(self, host):
        self.host = host
    def get_view_range(self):
        return self.clamped_view_range(
            self.host.img1.image.shape,
            self.host.ax1.viewRange(),
        )

    def ax1_view_range(self, integers=False):
        view_range = self._ax1_raw_view_range()
        if not integers:
            return view_range
        return self.integer_view_range(view_range)

    def view_range_changed(
        self,
        view_box,
        view_range,
        updateExportImageMask=True,
    ):
        self.host.status_hover_view.update_values_status_bar()

        if hasattr(self.host, 'scaleBar'):
            scale_bar_move_with_zoom = (
                self.host.scaleBar.properties()['move_with_zoom']
            )
        else:
            scale_bar_move_with_zoom = False
        if self.should_move_decoration(
            dialog_open=self.host.scaleBarDialog is not None,
            move_with_zoom=scale_bar_move_with_zoom,
        ):
            self.host.scaleBar.updatePosViewRangeChanged(view_range)

        if hasattr(self.host, 'timestamp'):
            timestamp_move_with_zoom = (
                self.host.timestamp.properties()['move_with_zoom']
            )
        else:
            timestamp_move_with_zoom = False
        if self.should_move_decoration(
            dialog_open=self.host.timestampDialog is not None,
            move_with_zoom=timestamp_move_with_zoom,
        ):
            self.host.timestamp.updatePosViewRangeChanged(view_range)

        self.host._viewRange = view_range

    def store_view_range(self):
        if not self.should_store_view_range(
            has_range_reset_state=hasattr(self.host, 'isRangeReset'),
            is_range_reset=getattr(self.host, 'isRangeReset', False),
        ):
            return
        self.host.ax1_viewRange = self.host.ax1.viewRange()
        self.host.isRangeReset = False

    def add_timestamp(self, checked):
        if checked:
            pos_data = self.host.data[self.host.pos_i]
            y_size, x_size = self.host.img1.image.shape[:2]
            view_range = self.ax1_view_range()
            self.host.timestampDialog = apps.TimestampPropertiesDialog(
                parent=self.host
            )
            self.host.timestampDialog.show()
            self.host.timestamp = widgets.TimestampItem(
                y_size,
                x_size,
                view_range,
                secondsPerFrame=pos_data.TimeIncrement,
                start_timedelta=self.host.timestampStartTimedelta,
            )
            self.host.timestamp.sigEditProperties.connect(
                self.edit_timestamp_properties
            )
            self.host.timestamp.sigRemove.connect(
                self.edit_timestamp_remove
            )
            self.host.timestamp.addToAxis(self.host.ax1)
            self.host.timestamp.draw(
                pos_data.frame_i, **self.host.timestampDialog.kwargs()
            )
            self.host.timestampDialog.sigValueChanged.connect(
                self.update_timestamp
            )
            self.host.timestampDialog.exec_()
        else:
            self.host.timestamp.removeFromAxis(self.host.ax1)

        self.host.timestampDialog = None
        self.host.imgGrad.addTimestampAction.setChecked(checked)

    def add_scale_bar(self, checked):
        if checked:
            pos_data = self.host.data[self.host.pos_i]
            y_size, x_size = self.host.img1.image.shape[:2]
            view_range = self.ax1_view_range()
            self.host.scaleBarDialog = apps.ScaleBarPropertiesDialog(
                x_size,
                y_size,
                pos_data.PhysicalSizeX,
                parent=self.host,
            )
            self.host.scaleBarDialog.show()
            self.host.scaleBar = widgets.ScaleBar(
                (y_size, x_size), view_range, parent=self.host.ax1
            )
            self.host.scaleBar.sigEditProperties.connect(
                self.edit_scale_bar_properties
            )
            self.host.scaleBar.sigRemove.connect(self.edit_scale_bar_remove)
            self.host.scaleBar.addToAxis(self.host.ax1)
            self.host.scaleBar.draw(**self.host.scaleBarDialog.kwargs())
            self.host.scaleBarDialog.sigValueChanged.connect(
                self.update_scale_bar
            )
            self.host.scaleBarDialog.exec_()
            if self.host.scaleBarDialog.cancel:
                self.host.addScaleBarAction.setChecked(False)
                return
        else:
            self.host.scaleBar.removeFromAxis(self.host.ax1)

        self.host.scaleBarDialog = None
        self.host.imgGrad.addScaleBarAction.setChecked(checked)

    def update_scale_bar(self, scale_bar_kwargs):
        self.host.scaleBar.draw(**scale_bar_kwargs)

    def update_timestamp(self, timestamp_kwargs):
        pos_data = self.host.data[self.host.pos_i]
        self.host.timestamp.draw(pos_data.frame_i, **timestamp_kwargs)

    def edit_scale_bar_remove(self, timestamp):
        self.host.addScaleBarAction.setChecked(False)

    def edit_scale_bar_properties(self, properties):
        y_size, x_size = self.host.img1.image.shape[:2]
        pos_data = self.host.data[self.host.pos_i]
        self.host.scaleBarDialog = apps.ScaleBarPropertiesDialog(
            x_size,
            y_size,
            pos_data.PhysicalSizeX,
            parent=self.host,
            **properties,
        )
        self.host.scaleBarDialog.sigValueChanged.connect(
            self.update_scale_bar
        )
        self.host.scaleBarDialog.exec_()

    def edit_timestamp_remove(self, timestamp):
        self.host.addTimestampAction.setChecked(False)

    def edit_timestamp_properties(self, properties):
        self.host.timestampDialog = apps.TimestampPropertiesDialog(
            parent=self.host, **properties
        )
        self.host.timestampDialog.sigValueChanged.connect(
            self.update_timestamp
        )
        self.host.timestampDialog.show()

    def update_timestamp_frame(self):
        if not self.should_update_timestamp_frame(
            has_timestamp=hasattr(self.host, 'timestamp'),
            timestamp_enabled=self.host.addTimestampAction.isChecked(),
        ):
            return

        pos_data = self.host.data[self.host.pos_i]
        self.host.timestamp.setText(pos_data.frame_i)

    def _ax1_raw_view_range(self):
        if self.host.exportToImageWindow is None:
            return self.host.ax1.viewRange()
        export_mask = np.all(
            self.host.exportMaskImage == [0, 0, 0, 0], axis=-1
        )
        if np.all(export_mask):
            return self.host.ax1.viewRange()
        return self.host.ax1.viewRange(export_mask)