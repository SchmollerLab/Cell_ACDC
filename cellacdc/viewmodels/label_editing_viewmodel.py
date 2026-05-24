"""View-model contracts for label-editing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.label_editing_model import LabelEditingModel

from .cca_edits import CcaEditViewModel
from .edit_id import EditIdViewModel
from .geometry import GeometryViewModel
from .label_edits import LabelEditViewModel


@dataclass(frozen=True)
class LabelEditingViewModel:
    """Application-facing label-editing decisions and commands."""

    model: LabelEditingModel = field(default_factory=LabelEditingModel)
    cca_edits: CcaEditViewModel = field(default_factory=CcaEditViewModel)
    edit_id: EditIdViewModel = field(default_factory=EditIdViewModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)

    def should_apply_manual_edits(self, edited_labels_by_z) -> bool:
        return self.model.should_apply_manual_edits(edited_labels_by_z)

    def should_store_zslice_regionprops(self, *, is_segm_3d: bool) -> bool:
        return self.model.should_store_zslice_regionprops(
            is_segm_3d=is_segm_3d
        )

    def should_update_zslice_regionprops(
        self,
        *,
        force_update: bool,
        already_stored: bool,
    ) -> bool:
        return self.model.should_update_zslice_regionprops(
            force_update=force_update,
            already_stored=already_stored,
        )

    def should_prompt_for_background_id(self, clicked_id: int) -> bool:
        return self.model.should_prompt_for_background_id(clicked_id)

    def is_power_button_color(
        self,
        *,
        button_color: str,
        power_color: str,
    ) -> bool:
        return self.model.is_power_button_color(
            button_color=button_color,
            power_color=power_color,
        )

    def should_force_new_hover_id(
        self,
        *,
        brush_active: bool,
        shift_pressed: bool,
    ) -> bool:
        return self.model.should_force_new_hover_id(
            brush_active=brush_active,
            shift_pressed=shift_pressed,
        )

    def should_restore_brush_id_from_hover(
        self,
        *,
        is_hover_z_neighbor: bool,
        shift_pressed: bool,
        last_hover_id: int,
        hover_id: int,
    ) -> bool:
        return self.model.should_restore_brush_id_from_hover(
            is_hover_z_neighbor=is_hover_z_neighbor,
            shift_pressed=shift_pressed,
            last_hover_id=last_hover_id,
            hover_id=hover_id,
        )
