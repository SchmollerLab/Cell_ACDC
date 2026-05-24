"""Scriptable model rules for save and autosave workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AutosaveSchedule:
    """Autosave timer decision."""

    use_frame_timer: bool
    interval_ms: int | None = None


@dataclass(frozen=True)
class AutosaveIntervalChange:
    """Settings and UI text for an autosave interval change."""

    value: float
    unit: Literal['minutes', 'frames']
    settings_updates: dict[str, str]
    log_message: str
    tooltip: str
    start_frame_timer: bool


@dataclass(frozen=True)
class ConcatenatePromptPlan:
    """Decision for showing the concatenate-output prompt."""

    should_prompt: bool
    ensure_setting: bool


class SavingModel:
    """Headless decisions for save and autosave workflows."""

    viewer_mode = 'Viewer'
    segmentation_mode = 'Segmentation and Tracking'
    cell_cycle_mode = 'Cell cycle analysis'

    def should_clear_autosave_status(self, *, mode: str) -> bool:
        return mode == self.viewer_mode

    def should_enqueue_autosave(self, *, mode: str, has_active_workers: bool):
        return mode != self.viewer_mode and has_active_workers

    def autosave_schedule(
        self,
        value: float,
        unit: Literal['minutes', 'frames'],
    ) -> AutosaveSchedule | None:
        if value == 0:
            return None
        if unit == 'frames':
            return AutosaveSchedule(use_frame_timer=True)
        return AutosaveSchedule(
            use_frame_timer=False,
            interval_ms=round(value * 60 * 1000),
        )

    def autosave_interval_change(
        self,
        value: float,
        unit: Literal['minutes', 'frames'],
    ) -> AutosaveIntervalChange:
        return AutosaveIntervalChange(
            value=value,
            unit=unit,
            settings_updates={
                'autoSaveIntevalValue': str(value),
                'autoSaveIntervalUnit': unit,
            },
            log_message=f'Autosave interval changed to: {value} {unit}',
            tooltip=(
                'Change autosave interval to every N frames or minutes\n\n'
                f'Current autosave interval: {value} {unit}'
            ),
            start_frame_timer=unit == 'frames',
        )

    def concatenate_prompt_plan(
        self,
        *,
        has_main_window: bool,
        is_quick_save: bool,
        setting_exists: bool,
        show_setting_value: str | None,
    ) -> ConcatenatePromptPlan:
        if not has_main_window or is_quick_save:
            return ConcatenatePromptPlan(
                should_prompt=False,
                ensure_setting=False,
            )

        should_prompt = show_setting_value != 'No'
        return ConcatenatePromptPlan(
            should_prompt=should_prompt,
            ensure_setting=not setting_exists,
        )

    def concatenate_prompt_setting(self, *, do_not_show_again: bool) -> str:
        if do_not_show_again:
            return 'No'
        return 'Yes'

    def autosave_segmentation_enabled(self, *, mode: str, checked: bool) -> bool:
        if mode != self.segmentation_mode:
            return False
        return checked

    def autosave_annotations_enabled(self, *, mode: str, checked: bool) -> bool:
        if mode != self.viewer_mode:
            return False
        return checked

    def save_as_basename(self, basename: str) -> str:
        if basename.endswith('_'):
            return f'{basename}segm'
        return f'{basename}_segm'

    def quick_save_positions(self, position_foldername: str) -> set[str]:
        return {position_foldername}

    def should_ask_positions(
        self,
        *,
        is_snapshot: bool,
        is_quick_save: bool,
        position_count: int,
    ) -> bool:
        return is_snapshot and not is_quick_save and position_count > 1

    def should_compute_volume_metrics(
        self,
        *,
        save_metrics: bool,
        mode: str,
    ) -> bool:
        return save_metrics or mode == self.cell_cycle_mode

    def save_finished_title(
        self,
        *,
        aborted: bool,
        worker_aborted: bool,
        is_quick_save: bool,
    ) -> tuple[str, str | None]:
        if aborted or worker_aborted:
            return 'Saving process cancelled.', 'r'
        if is_quick_save:
            return 'Saved segmentation file and annotations', None
        return 'Saved!', None
