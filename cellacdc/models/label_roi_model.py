"""Scriptable model rules for label-ROI workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelRoiParamsSettings:
    """Settings updates for Magic Labeller ROI options."""

    updates: dict[str, object]


class LabelRoiModel:
    """Headless decisions for Magic Labeller ROI workflows."""

    yes_value = 'Yes'
    no_value = 'No'

    def checked_setting_value(self, checked: bool) -> str:
        return self.yes_value if checked else self.no_value

    def checked_from_setting_value(self, value) -> bool:
        return value == self.yes_value

    def model_params_ini_path(self, settings_folderpath: str) -> str:
        return os.path.join(settings_folderpath, 'last_params_segm_models.ini')

    def params_settings(
        self,
        *,
        checked_roi_type: str,
        circ_roi_radius: int,
        roi_zdepth: int,
        auto_clear_border: bool,
        replace_existing_objects: bool,
    ) -> LabelRoiParamsSettings:
        return LabelRoiParamsSettings(
            updates={
                'labelRoi_checkedRoiType': checked_roi_type,
                'labelRoi_circRoiRadius': circ_roi_radius,
                'labelRoi_roiZdepth': roi_zdepth,
                'labelRoi_autoClearBorder': self.checked_setting_value(
                    auto_clear_border
                ),
                'labelRoi_replaceExistingObjects': (
                    self.checked_setting_value(replace_existing_objects)
                ),
            }
        )

    def is_frame_range_valid(
        self,
        enabled: bool,
        start_frame_number: int,
        stop_frame_number: int,
    ) -> bool:
        return not enabled or start_frame_number <= stop_frame_number

    def frame_range_length(
        self,
        enabled: bool,
        start_frame_index: int,
        stop_frame_number: int,
    ) -> int:
        if not enabled:
            return 1
        return stop_frame_number - start_frame_index

    def time_range(
        self,
        enabled: bool,
        start_frame_index: int,
        stop_frame_number: int,
    ):
        if self.frame_range_length(
            enabled,
            start_frame_index,
            stop_frame_number,
        ) > 1:
            return start_frame_index, stop_frame_number
        return None

    def should_enable_range_controls(self, checked: bool) -> bool:
        return checked

    def should_show_circular_cursor(
        self,
        *,
        label_roi_checked: bool,
        circular_roi_checked: bool,
        label_roi_running: bool,
        cursor_checked: bool,
        existing_cursor_empty: bool,
    ) -> bool:
        return (
            label_roi_checked
            and circular_roi_checked
            and not label_roi_running
            and (cursor_checked or not existing_cursor_empty)
        )

    def cursor_points(self, x, y, checked: bool):
        if not checked:
            return [], []
        return [x], [y]

    def should_uncheck_time_range(
        self,
        *,
        time_range_checked: bool,
        persistent_action_checked: bool,
    ) -> bool:
        return time_range_checked and not persistent_action_checked

    def z_range(
        self,
        roi_zdepth: int,
        size_z: int,
        current_z_index: int,
    ) -> tuple[int, int]:
        if roi_zdepth == size_z:
            return 0, size_z
        if roi_zdepth == 1:
            return current_z_index, current_z_index + 1

        if roi_zdepth % 2 != 0:
            roi_zdepth += 1
        half_zdepth = int(roi_zdepth / 2)
        zc = current_z_index + 1
        z0 = max(zc - half_zdepth, 0)
        z1 = min(zc + half_zdepth, size_z)
        return z0, z1
