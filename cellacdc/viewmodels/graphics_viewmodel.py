"""View-model composition for graphics workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping
import numpy as np

from cellacdc.models.graphics_model import (
    GraphicsModel,
    OverlayOpacityPlan,
    OverlayVisibilityPlan,
)

from .formatting import FormattingViewModel
from .geometry import GeometryViewModel
from .label_edits import LabelEditViewModel
from .workspace import WorkspaceViewModel


@dataclass(frozen=True)
class GraphicsViewModel:
    """GUI-facing commands for graphics item construction workflows."""

    model: GraphicsModel = field(default_factory=GraphicsModel)
    formatting: FormattingViewModel = field(default_factory=FormattingViewModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)
    workspace: WorkspaceViewModel = field(default_factory=WorkspaceViewModel)

    def overlay_toolbutton_checked(
        self,
        channel: str,
        *,
        checked_channels: Iterable[str],
        is_single_channel: bool,
    ) -> bool:
        return self.model.overlay_toolbutton_checked(
            channel,
            checked_channels=checked_channels,
            is_single_channel=is_single_channel,
        )

    def overlay_toolbutton_click_checked_channels(
        self,
        *,
        clicked_channel: str,
        all_channels: Iterable[str],
        checked_channels: Iterable[str],
        toolbar_single_channel: bool,
    ) -> set[str]:
        return self.model.overlay_toolbutton_click_checked_channels(
            clicked_channel=clicked_channel,
            all_channels=all_channels,
            checked_channels=checked_channels,
            toolbar_single_channel=toolbar_single_channel,
        )

    def overlay_visibility_plan(
        self,
        *,
        all_channels: Iterable[str],
        checked_channels: Iterable[str],
        overlay_enabled: bool,
    ) -> OverlayVisibilityPlan:
        return self.model.overlay_visibility_plan(
            all_channels=all_channels,
            checked_channels=checked_channels,
            overlay_enabled=overlay_enabled,
        )

    def overlay_channel_opacity_map(
        self,
        base_channel: str,
        active_channel_alpha_values: Mapping[str, float],
    ) -> dict[str, float]:
        return self.model.overlay_channel_opacity_map(
            base_channel,
            active_channel_alpha_values,
        )

    def overlay_item_opacity_plan(
        self,
        *,
        all_channels: Iterable[str],
        base_channel: str,
        checked_channels: Iterable[str],
        toolbar_single_channel: bool,
        active_channel_alpha_values: Mapping[str, float],
    ) -> OverlayOpacityPlan:
        return self.model.overlay_item_opacity_plan(
            all_channels=all_channels,
            base_channel=base_channel,
            checked_channels=checked_channels,
            toolbar_single_channel=toolbar_single_channel,
            active_channel_alpha_values=active_channel_alpha_values,
        )

    def generate_labels_image_lut(self, base_lut: np.ndarray) -> np.ndarray:
        """Wrap generate_labels_image_lut model call."""
        return self.model.generate_labels_image_lut(base_lut)

    def extend_labels_lut(self, base_lut: np.ndarray, len_new_lut: int) -> np.ndarray:
        """Wrap extend_labels_lut model call."""
        return self.model.extend_labels_lut(base_lut, len_new_lut)

    def apply_lut_dimming_for_kept_objects(
        self,
        lut: np.ndarray,
        kept_object_ids: list[int],
        keep_ids_enabled: bool,
    ) -> np.ndarray:
        """Wrap apply_lut_dimming_for_kept_objects model call."""
        return self.model.apply_lut_dimming_for_kept_objects(
            lut, kept_object_ids, keep_ids_enabled
        )

