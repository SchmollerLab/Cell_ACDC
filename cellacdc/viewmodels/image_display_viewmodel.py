"""View-model behavior for image display workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.image_display_model import (
    ImageDisplayModel,
    LabelsAlphaPlan,
    RightPaneMode,
    RightPaneVisibilityPlan,
)

from .formatting import FormattingViewModel
from .preprocessing_viewmodel import PreprocessingViewModel


@dataclass(frozen=True)
class ImageDisplayViewModel:
    """GUI-facing helpers for image display workflows."""

    model: ImageDisplayModel = field(default_factory=ImageDisplayModel)
    formatting: FormattingViewModel = field(default_factory=FormattingViewModel)
    preprocessing: PreprocessingViewModel = field(
        default_factory=PreprocessingViewModel
    )

    def right_pane_visibility_plan(
        self,
        mode: RightPaneMode,
        checked: bool,
    ) -> RightPaneVisibilityPlan:
        return self.model.right_pane_visibility_plan(mode, checked)

    def invert_bw_setting_value(self, checked: bool) -> str:
        return self.model.invert_bw_setting_value(checked)

    def labels_alpha_plan(
        self,
        value: float,
        *,
        keep_ids_checked: bool,
    ) -> LabelsAlphaPlan:
        return self.model.labels_alpha_plan(
            value,
            keep_ids_checked=keep_ids_checked,
        )

    def intensity_normalization_setting_value(self, how: str) -> str:
        return self.model.intensity_normalization_setting_value(how)

    def rescale_intensity_setting_update(
        self,
        channel: str,
        how: str,
    ) -> tuple[str, str]:
        return self.model.rescale_intensity_setting_update(channel, how)
