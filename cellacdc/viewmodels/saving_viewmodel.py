"""View-model contracts for save and autosave workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cellacdc.models.saving_model import (
    AutosaveIntervalChange,
    ConcatenatePromptPlan,
    SavingModel,
)

from .cca_workflows import CcaWorkflowViewModel
from .formatting import FormattingViewModel
from .measurements_viewmodel import MeasurementsViewModel
from .tracking_viewmodel import TrackingViewModel
from .workspace import WorkspaceViewModel


@dataclass(frozen=True)
class SavingViewModel:
    """Application-facing save/autosave commands and decisions."""

    model: SavingModel = field(default_factory=SavingModel)
    cca_workflows: CcaWorkflowViewModel = field(
        default_factory=CcaWorkflowViewModel
    )
    formatting: FormattingViewModel = field(default_factory=FormattingViewModel)
    measurements: MeasurementsViewModel = field(
        default_factory=MeasurementsViewModel
    )
    tracking: TrackingViewModel = field(default_factory=TrackingViewModel)
    workspace: WorkspaceViewModel = field(default_factory=WorkspaceViewModel)

    def should_clear_autosave_status(self, *, mode: str) -> bool:
        return self.model.should_clear_autosave_status(mode=mode)

    def should_enqueue_autosave(
        self,
        *,
        mode: str,
        has_active_workers: bool,
    ):
        return self.model.should_enqueue_autosave(
            mode=mode,
            has_active_workers=has_active_workers,
        )

    def autosave_schedule(
        self,
        value: float,
        unit: Literal['minutes', 'frames'],
    ):
        return self.model.autosave_schedule(value, unit)

    def autosave_interval_change(
        self,
        value: float,
        unit: Literal['minutes', 'frames'],
    ) -> AutosaveIntervalChange:
        return self.model.autosave_interval_change(value, unit)

    def concatenate_prompt_plan(
        self,
        *,
        has_main_window: bool,
        is_quick_save: bool,
        setting_exists: bool,
        show_setting_value: str | None,
    ) -> ConcatenatePromptPlan:
        return self.model.concatenate_prompt_plan(
            has_main_window=has_main_window,
            is_quick_save=is_quick_save,
            setting_exists=setting_exists,
            show_setting_value=show_setting_value,
        )

    def concatenate_prompt_setting(self, *, do_not_show_again: bool) -> str:
        return self.model.concatenate_prompt_setting(
            do_not_show_again=do_not_show_again
        )

    def autosave_segmentation_enabled(self, *, mode: str, checked: bool) -> bool:
        return self.model.autosave_segmentation_enabled(
            mode=mode,
            checked=checked,
        )

    def autosave_annotations_enabled(self, *, mode: str, checked: bool) -> bool:
        return self.model.autosave_annotations_enabled(
            mode=mode,
            checked=checked,
        )

    def save_as_basename(self, basename: str) -> str:
        return self.model.save_as_basename(basename)

    def quick_save_positions(self, position_foldername: str) -> set[str]:
        return self.model.quick_save_positions(position_foldername)

    def should_ask_positions(
        self,
        *,
        is_snapshot: bool,
        is_quick_save: bool,
        position_count: int,
    ) -> bool:
        return self.model.should_ask_positions(
            is_snapshot=is_snapshot,
            is_quick_save=is_quick_save,
            position_count=position_count,
        )

    def should_compute_volume_metrics(
        self,
        *,
        save_metrics: bool,
        mode: str,
    ) -> bool:
        return self.model.should_compute_volume_metrics(
            save_metrics=save_metrics,
            mode=mode,
        )

    def save_finished_title(
        self,
        *,
        aborted: bool,
        worker_aborted: bool,
        is_quick_save: bool,
    ) -> tuple[str, str | None]:
        return self.model.save_finished_title(
            aborted=aborted,
            worker_aborted=worker_aborted,
            is_quick_save=is_quick_save,
        )
