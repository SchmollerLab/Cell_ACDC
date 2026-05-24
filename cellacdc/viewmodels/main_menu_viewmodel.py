"""View-model contracts for the main menu."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.main_menu_model import MainMenuModel


@dataclass(frozen=True)
class MainMenuViewModel:
    """Application-facing main-menu commands."""

    model: MainMenuModel = field(default_factory=MainMenuModel)

    def default_rescale_intensity_options(self):
        return self.model.default_rescale_intensity_options

    def default_rescale_intensity_how(self, settings):
        return self.model.default_rescale_intensity_how(settings)
