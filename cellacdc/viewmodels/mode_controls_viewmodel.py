"""View-model contracts for GUI mode controls."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.mode_controls_model import ModeControlsModel


@dataclass(frozen=True)
class ModeControlsViewModel:
    """Application-facing mode-control decisions."""

    model: ModeControlsModel = field(default_factory=ModeControlsModel)

    def should_start_blinking(
        self,
        mode: str,
        *,
        ruler_checked: bool = False,
    ) -> bool:
        return self.model.should_start_blinking(
            mode, ruler_checked=ruler_checked
        )

    def blink_styles(self, flag: bool) -> tuple[str, bool]:
        return self.model.blink_styles(flag)

    def should_store_on_mode_change(self, previous_mode: str) -> bool:
        return self.model.should_store_on_mode_change(previous_mode)

    def is_cca_mode(self, mode: str) -> bool:
        return self.model.is_cca_mode(mode)

    def undo_redo_target(self, mode: str) -> str:
        return self.model.undo_redo_target(mode)
