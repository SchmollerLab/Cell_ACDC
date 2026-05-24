"""View adapter for timestamp, scale-bar, and view-range decorations."""

from __future__ import annotations

import numpy as np

from cellacdc import apps, widgets


class DisplayDecorationsMixin:
    """Qt-facing adapter around display-decoration contracts."""

    """Headless display-decoration decision rules."""

    def _ax1_raw_view_range(self):
        if self.exportToImageWindow is None:
            return self.ax1.viewRange()
        export_mask = np.all(self.exportMaskImage == [0, 0, 0, 0], axis=-1)
        if np.all(export_mask):
            return self.ax1.viewRange()
        return self.ax1.viewRange(export_mask)

    def add_scale_bar(self, checked):
        if checked:
            pos_data = self.data[self.pos_i]
            y_size, x_size = self.img1.image.shape[:2]
            view_range = self.ax1_view_range()
            self.scaleBarDialog = apps.ScaleBarPropertiesDialog(
                x_size,
                y_size,
                pos_data.PhysicalSizeX,
                parent=self,
            )
            self.scaleBarDialog.show()
            self.scaleBar = widgets.ScaleBar(
                (y_size, x_size), view_range, parent=self.ax1
            )
            self.scaleBar.sigEditProperties.connect(self.edit_scale_bar_properties)
            self.scaleBar.sigRemove.connect(self.edit_scale_bar_remove)
            self.scaleBar.addToAxis(self.ax1)
            self.scaleBar.draw(**self.scaleBarDialog.kwargs())
            self.scaleBarDialog.sigValueChanged.connect(self.update_scale_bar)
            self.scaleBarDialog.exec_()
            if self.scaleBarDialog.cancel:
                self.addScaleBarAction.setChecked(False)
                return
        else:
            self.scaleBar.removeFromAxis(self.ax1)

        self.scaleBarDialog = None
        self.imgGrad.addScaleBarAction.setChecked(checked)

    def add_timestamp(self, checked):
        if checked:
            pos_data = self.data[self.pos_i]
            y_size, x_size = self.img1.image.shape[:2]
            view_range = self.ax1_view_range()
            self.timestampDialog = apps.TimestampPropertiesDialog(parent=self)
            self.timestampDialog.show()
            self.timestamp = widgets.TimestampItem(
                y_size,
                x_size,
                view_range,
                secondsPerFrame=pos_data.TimeIncrement,
                start_timedelta=self.timestampStartTimedelta,
            )
            self.timestamp.sigEditProperties.connect(self.edit_timestamp_properties)
            self.timestamp.sigRemove.connect(self.edit_timestamp_remove)
            self.timestamp.addToAxis(self.ax1)
            self.timestamp.draw(pos_data.frame_i, **self.timestampDialog.kwargs())
            self.timestampDialog.sigValueChanged.connect(self.update_timestamp)
            self.timestampDialog.exec_()
        else:
            self.timestamp.removeFromAxis(self.ax1)

        self.timestampDialog = None
        self.imgGrad.addTimestampAction.setChecked(checked)

    def ax1_view_range(self, integers=False):
        view_range = self._ax1_raw_view_range()
        if not integers:
            return view_range
        return self.integer_view_range(view_range)

    def clamped_view_range(self, image_shape, view_range):
        y_size, x_size = image_shape[:2]
        x_range, y_range = view_range
        x_min = 0 if x_range[0] < 0 else x_range[0]
        y_min = 0 if y_range[0] < 0 else y_range[0]
        x_max = x_size if x_range[1] >= x_size else x_range[1]
        y_max = y_size if y_range[1] >= y_size else y_range[1]
        return int(y_min), int(y_max), int(x_min), int(x_max)

    def edit_scale_bar_properties(self, properties):
        y_size, x_size = self.img1.image.shape[:2]
        pos_data = self.data[self.pos_i]
        self.scaleBarDialog = apps.ScaleBarPropertiesDialog(
            x_size,
            y_size,
            pos_data.PhysicalSizeX,
            parent=self,
            **properties,
        )
        self.scaleBarDialog.sigValueChanged.connect(self.update_scale_bar)
        self.scaleBarDialog.exec_()

    def edit_scale_bar_remove(self, timestamp):
        self.addScaleBarAction.setChecked(False)

    def edit_timestamp_properties(self, properties):
        self.timestampDialog = apps.TimestampPropertiesDialog(parent=self, **properties)
        self.timestampDialog.sigValueChanged.connect(self.update_timestamp)
        self.timestampDialog.show()

    def edit_timestamp_remove(self, timestamp):
        self.addTimestampAction.setChecked(False)

    def get_view_range(self):
        return self.clamped_view_range(
            self.img1.image.shape,
            self.ax1.viewRange(),
        )

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

    def store_view_range(self):
        if not self.should_store_view_range(
            has_range_reset_state=hasattr(self, "isRangeReset"),
            is_range_reset=getattr(self, "isRangeReset", False),
        ):
            return
        self.ax1_viewRange = self.ax1.viewRange()
        self.isRangeReset = False

    def update_scale_bar(self, scale_bar_kwargs):
        self.scaleBar.draw(**scale_bar_kwargs)

    def update_timestamp(self, timestamp_kwargs):
        pos_data = self.data[self.pos_i]
        self.timestamp.draw(pos_data.frame_i, **timestamp_kwargs)

    def update_timestamp_frame(self):
        if not self.should_update_timestamp_frame(
            has_timestamp=hasattr(self, "timestamp"),
            timestamp_enabled=self.addTimestampAction.isChecked(),
        ):
            return

        pos_data = self.data[self.pos_i]
        self.timestamp.setText(pos_data.frame_i)

    def view_range_changed(
        self,
        view_box,
        view_range,
        updateExportImageMask=True,
    ):
        self.status_hover_view.update_values_status_bar()

        if hasattr(self, "scaleBar"):
            scale_bar_move_with_zoom = self.scaleBar.properties()["move_with_zoom"]
        else:
            scale_bar_move_with_zoom = False
        if self.should_move_decoration(
            dialog_open=self.scaleBarDialog is not None,
            move_with_zoom=scale_bar_move_with_zoom,
        ):
            self.scaleBar.updatePosViewRangeChanged(view_range)

        if hasattr(self, "timestamp"):
            timestamp_move_with_zoom = self.timestamp.properties()["move_with_zoom"]
        else:
            timestamp_move_with_zoom = False
        if self.should_move_decoration(
            dialog_open=self.timestampDialog is not None,
            move_with_zoom=timestamp_move_with_zoom,
        ):
            self.timestamp.updatePosViewRangeChanged(view_range)

        self._viewRange = view_range
