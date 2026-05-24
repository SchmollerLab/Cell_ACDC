"""Scriptable model rules for the main menu."""

from __future__ import annotations


class MainMenuModel:
    """Headless main-menu decision rules."""

    default_rescale_intensity_options = (
        'Rescale each 2D image',
        'Rescale across z-stack',
        'Rescale across time frames',
        'Do no rescale, display raw image',
    )

    def default_rescale_intensity_how(self, settings):
        try:
            return settings.at['default_rescale_intens_how', 'value']
        except Exception:
            return self.default_rescale_intensity_options[0]
