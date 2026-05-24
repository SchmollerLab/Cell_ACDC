"""Scriptable model rules for the main GUI toolbars."""

from __future__ import annotations


class MainToolbarModel:
    """Headless toolbar metadata used by the main toolbar view."""

    mode_items = (
        'Segmentation and Tracking',
        'Cell cycle analysis',
        'Viewer',
        'Custom annotations',
        'Normal division: Lineage tree',
    )

    def default_mode_items(self) -> tuple[str, ...]:
        return self.mode_items
