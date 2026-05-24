"""Qt-free model rules for graphics workflows."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import numpy as np


@dataclass(frozen=True)
class OverlayOpacityPlan:
    """Opacity and scrollbar state for overlay image items."""

    opacities: dict[str, float]
    alpha_scrollbar_disabled: dict[str, bool]


@dataclass(frozen=True)
class OverlayVisibilityPlan:
    """Visibility state for overlay controls by channel."""

    channel_visible: dict[str, bool]


class GraphicsModel:
    """Headless graphics workflow rules."""

    def overlay_toolbutton_checked(
        self,
        channel: str,
        *,
        checked_channels: Iterable[str],
        is_single_channel: bool,
    ) -> bool:
        return not is_single_channel and channel in set(checked_channels)

    def overlay_toolbutton_click_checked_channels(
        self,
        *,
        clicked_channel: str,
        all_channels: Iterable[str],
        checked_channels: Iterable[str],
        toolbar_single_channel: bool,
    ) -> set[str]:
        all_channels = set(all_channels)
        checked_channels = set(checked_channels)
        if not checked_channels or toolbar_single_channel:
            checked_channels.add(clicked_channel)

        if toolbar_single_channel:
            return {clicked_channel}

        return checked_channels & all_channels

    def overlay_visibility_plan(
        self,
        *,
        all_channels: Iterable[str],
        checked_channels: Iterable[str],
        overlay_enabled: bool,
    ) -> OverlayVisibilityPlan:
        checked_channels = set(checked_channels)
        return OverlayVisibilityPlan(
            channel_visible={
                channel: overlay_enabled and channel in checked_channels
                for channel in all_channels
            }
        )

    def overlay_channel_opacity_map(
        self,
        base_channel: str,
        active_channel_alpha_values: Mapping[str, float],
    ) -> dict[str, float]:
        channels = list(active_channel_alpha_values)
        alpha_values = list(active_channel_alpha_values.values())
        opacities = self._base_first_hierarchical_opacities(alpha_values)
        channel_opacity_mapper = {
            channel: opacities[i + 1]
            for i, channel in enumerate(channels)
        }
        channel_opacity_mapper[base_channel] = opacities[0]
        return channel_opacity_mapper

    def overlay_item_opacity_plan(
        self,
        *,
        all_channels: Iterable[str],
        base_channel: str,
        checked_channels: Iterable[str],
        toolbar_single_channel: bool,
        active_channel_alpha_values: Mapping[str, float],
    ) -> OverlayOpacityPlan:
        checked_channels = set(checked_channels)
        channel_opacity_mapper = self.overlay_channel_opacity_map(
            base_channel,
            active_channel_alpha_values,
        )
        is_single_channel = toolbar_single_channel or len(checked_channels) == 1

        opacities = {}
        alpha_scrollbar_disabled = {}
        for channel in all_channels:
            if channel in checked_channels and is_single_channel:
                op_val = 1.0
            elif channel in checked_channels:
                op_val = channel_opacity_mapper[channel]
            else:
                op_val = 0.0

            if op_val == 0:
                op_val = 0.01

            opacities[channel] = min(op_val, 0.999)
            if channel != base_channel:
                alpha_scrollbar_disabled[channel] = op_val == 0

        return OverlayOpacityPlan(
            opacities=opacities,
            alpha_scrollbar_disabled=alpha_scrollbar_disabled,
        )

    def _base_first_hierarchical_opacities(
        self,
        alpha_values: Iterable[float],
    ) -> list[float]:
        alphas = [1.0, *alpha_values]
        if not alphas:
            return alphas

        weights = []
        for i, alpha_ref in enumerate(alphas):
            weight = alpha_ref
            for alpha in alphas[i + 1:]:
                weight *= 1 - alpha
            weights.append(weight)

        return weights

    def generate_labels_image_lut(self, base_lut: np.ndarray) -> np.ndarray:
        """Converts a 3-channel base LUT to a 4-channel RGBA LUT with background as transparent."""
        import numpy as np
        lut = np.zeros((len(base_lut), 4), dtype=np.uint8)
        lut[:, -1] = 255
        lut[:, :-1] = base_lut
        lut[0] = [0, 0, 0, 0]
        return lut

    def extend_labels_lut(self, base_lut: np.ndarray, len_new_lut: int) -> np.ndarray:
        """Extends base_lut to include IDs greater than original length of base_lut."""
        import numpy as np
        if len_new_lut <= len(base_lut):
            return base_lut

        num_new_colors = len_new_lut - len(base_lut)
        _lut = np.zeros((len_new_lut, 3), np.uint8)
        _lut[:len(base_lut)] = base_lut
        
        random_idx = np.random.randint(0, len(base_lut), size=num_new_colors)
        for i, idx in enumerate(random_idx):
            rgb = base_lut[idx]
            _lut[len(base_lut) + i] = rgb
        return _lut

    def apply_lut_dimming_for_kept_objects(
        self,
        lut: np.ndarray,
        kept_object_ids: list[int],
        keep_ids_enabled: bool,
    ) -> np.ndarray:
        """Applies dimming to non-kept objects in the LUT if keep_ids is enabled."""
        import numpy as np
        if not keep_ids_enabled:
            return lut

        dimmed_lut = np.round(lut * 0.3).astype(np.uint8)
        valid_ids = [idx for idx in kept_object_ids if idx < len(lut)]
        if valid_ids:
            kept_lut = np.round(dimmed_lut[valid_ids] / 0.3).astype(np.uint8)
            dimmed_lut[valid_ids] = kept_lut
        return dimmed_lut

