"""View-model contracts for label-ROI workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.label_roi_model import (
    LabelRoiModel,
    LabelRoiParamsSettings,
)


@dataclass(frozen=True)
class LabelRoiViewModel:
    """Application-facing Magic Labeller ROI decisions."""

    model: LabelRoiModel = field(default_factory=LabelRoiModel)

    def checked_setting_value(self, checked: bool) -> str:
        return self.model.checked_setting_value(checked)

    def checked_from_setting_value(self, value) -> bool:
        return self.model.checked_from_setting_value(value)

    def model_params_ini_path(self, settings_folderpath: str) -> str:
        return self.model.model_params_ini_path(settings_folderpath)

    def params_settings(
        self,
        *,
        checked_roi_type: str,
        circ_roi_radius: int,
        roi_zdepth: int,
        auto_clear_border: bool,
        replace_existing_objects: bool,
    ) -> LabelRoiParamsSettings:
        return self.model.params_settings(
            checked_roi_type=checked_roi_type,
            circ_roi_radius=circ_roi_radius,
            roi_zdepth=roi_zdepth,
            auto_clear_border=auto_clear_border,
            replace_existing_objects=replace_existing_objects,
        )

    def is_frame_range_valid(
        self,
        enabled: bool,
        start_frame_number: int,
        stop_frame_number: int,
    ) -> bool:
        return self.model.is_frame_range_valid(
            enabled,
            start_frame_number,
            stop_frame_number,
        )

    def frame_range_length(
        self,
        enabled: bool,
        start_frame_index: int,
        stop_frame_number: int,
    ) -> int:
        return self.model.frame_range_length(
            enabled,
            start_frame_index,
            stop_frame_number,
        )

    def time_range(
        self,
        enabled: bool,
        start_frame_index: int,
        stop_frame_number: int,
    ):
        return self.model.time_range(
            enabled,
            start_frame_index,
            stop_frame_number,
        )

    def should_enable_range_controls(self, checked: bool) -> bool:
        return self.model.should_enable_range_controls(checked)

    def should_show_circular_cursor(
        self,
        *,
        label_roi_checked: bool,
        circular_roi_checked: bool,
        label_roi_running: bool,
        cursor_checked: bool,
        existing_cursor_empty: bool,
    ) -> bool:
        return self.model.should_show_circular_cursor(
            label_roi_checked=label_roi_checked,
            circular_roi_checked=circular_roi_checked,
            label_roi_running=label_roi_running,
            cursor_checked=cursor_checked,
            existing_cursor_empty=existing_cursor_empty,
        )

    def cursor_points(self, x, y, checked: bool):
        return self.model.cursor_points(x, y, checked)

    def should_uncheck_time_range(
        self,
        *,
        time_range_checked: bool,
        persistent_action_checked: bool,
    ) -> bool:
        return self.model.should_uncheck_time_range(
            time_range_checked=time_range_checked,
            persistent_action_checked=persistent_action_checked,
        )

    def z_range(
        self,
        roi_zdepth: int,
        size_z: int,
        current_z_index: int,
    ) -> tuple[int, int]:
        return self.model.z_range(roi_zdepth, size_z, current_z_index)
