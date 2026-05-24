"""View-model contracts for the main GUI toolbars."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.main_toolbar_model import MainToolbarModel


@dataclass(frozen=True)
class MainToolbarViewModel:
    """Application-facing toolbar metadata."""

    model: MainToolbarModel = field(default_factory=MainToolbarModel)

    def mode_items(self) -> tuple[str, ...]:
        return self.model.default_mode_items()
