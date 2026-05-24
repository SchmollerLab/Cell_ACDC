"""Qt-free model rules for image display workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RightPaneMode = Literal['next_frame', 'right_image', 'labels']


@dataclass(frozen=True)
class RightPaneVisibilityPlan:
    """Settings update for a right-pane visibility toggle."""

    mode: RightPaneMode
    checked: bool
    settings_updates: dict[str, str]


@dataclass(frozen=True)
class LabelsAlphaPlan:
    """Settings and effective opacity for labels overlay alpha."""

    setting_value: float
    opacity: float


class ImageDisplayModel:
    """Headless display settings and image-display rules."""

    def right_pane_visibility_plan(
        self,
        mode: RightPaneMode,
        checked: bool,
    ) -> RightPaneVisibilityPlan:
        settings_updates = {
            'isNextFrameVisible': 'No',
            'isRightImageVisible': 'No',
            'isLabelsVisible': 'No',
        }
        if checked:
            setting_key = {
                'next_frame': 'isNextFrameVisible',
                'right_image': 'isRightImageVisible',
                'labels': 'isLabelsVisible',
            }[mode]
            settings_updates[setting_key] = 'Yes'

        return RightPaneVisibilityPlan(
            mode=mode,
            checked=checked,
            settings_updates=settings_updates,
        )

    def invert_bw_setting_value(self, checked: bool) -> str:
        return 'Yes' if checked else 'No'

    def labels_alpha_plan(
        self,
        value: float,
        *,
        keep_ids_checked: bool,
    ) -> LabelsAlphaPlan:
        opacity = value / 3 if keep_ids_checked else value
        return LabelsAlphaPlan(setting_value=value, opacity=opacity)

    def intensity_normalization_setting_value(self, how: str) -> str:
        return how

    def rescale_intensity_setting_update(
        self,
        channel: str,
        how: str,
    ) -> tuple[str, str]:
        return f'how_rescale_intensities_{channel}', how
