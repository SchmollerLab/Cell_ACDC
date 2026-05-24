"""Scriptable model rules for label-editing workflows."""

from __future__ import annotations


class LabelEditingModel:
    """Headless decisions for manual label editing."""

    def should_apply_manual_edits(self, edited_labels_by_z) -> bool:
        return bool(edited_labels_by_z)

    def should_store_zslice_regionprops(self, *, is_segm_3d: bool) -> bool:
        return is_segm_3d

    def should_update_zslice_regionprops(
        self,
        *,
        force_update: bool,
        already_stored: bool,
    ) -> bool:
        return force_update or not already_stored

    def should_prompt_for_background_id(self, clicked_id: int) -> bool:
        return clicked_id == 0

    def is_power_button_color(
        self,
        *,
        button_color: str,
        power_color: str,
    ) -> bool:
        return button_color == power_color

    def should_force_new_hover_id(
        self,
        *,
        brush_active: bool,
        shift_pressed: bool,
    ) -> bool:
        return brush_active and shift_pressed

    def should_restore_brush_id_from_hover(
        self,
        *,
        is_hover_z_neighbor: bool,
        shift_pressed: bool,
        last_hover_id: int,
        hover_id: int,
    ) -> bool:
        return (
            is_hover_z_neighbor
            and not shift_pressed
            and last_hover_id != hover_id
        )
