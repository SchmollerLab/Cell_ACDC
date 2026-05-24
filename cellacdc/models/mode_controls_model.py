"""Scriptable model rules for GUI mode controls."""

from __future__ import annotations


class ModeControlsModel:
    """Headless decisions for mode toolbar and action state."""

    viewer_mode = 'Viewer'
    segmentation_mode = 'Segmentation and Tracking'
    snapshot_mode = 'Snapshot'
    cca_mode = 'Cell cycle analysis'
    custom_annotations_mode = 'Custom annotations'

    def should_start_blinking(
        self,
        mode: str,
        *,
        ruler_checked: bool = False,
    ) -> bool:
        return mode == self.viewer_mode and not ruler_checked

    def blink_styles(self, flag: bool) -> tuple[str, bool]:
        if flag:
            return 'background-color: orange', False
        return 'background-color: none', True

    def should_store_on_mode_change(self, previous_mode: str) -> bool:
        return previous_mode != self.viewer_mode

    def is_cca_mode(self, mode: str) -> bool:
        return mode == self.cca_mode

    def undo_redo_target(self, mode: str) -> str:
        if mode in {self.segmentation_mode, self.snapshot_mode}:
            return 'labels'
        if mode == self.cca_mode:
            return 'cca'
        if mode == self.custom_annotations_mode:
            return 'custom_annotations'
        return 'disabled'
