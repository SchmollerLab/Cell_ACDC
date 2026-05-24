"""View-model contracts for annotation display workflows."""

from __future__ import annotations

from typing import Mapping

try:
    from qtpy.QtCore import QObject, Signal
except ModuleNotFoundError:  # pragma: no cover - exercised without GUI extras
    class QObject:
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class _FallbackBoundSignal:
        def __init__(self) -> None:
            self._slots = []

        def connect(self, slot) -> None:
            self._slots.append(slot)

        def emit(self, *args) -> None:
            for slot in tuple(self._slots):
                slot(*args)

    class Signal:
        def __init__(self, *args) -> None:
            self._name = ''

        def __set_name__(self, owner, name) -> None:
            self._name = f'__signal_{name}'

        def __get__(self, instance, owner):
            if instance is None:
                return self
            return instance.__dict__.setdefault(
                self._name,
                _FallbackBoundSignal(),
            )

from cellacdc.models.annotation_display_model import (
    AnnotationModeChangePlan,
    AnnotationDisplaySettingsRestorePlan,
    AnnotationOption,
    AnnotationOptionChangePlan,
    AnnotationOptionsFromModeTextPlan,
    AnnotationOptionState,
    AnnotationDisplayModel,
    AnnotationSide,
    PixelModeChangePlan,
    TextResolutionChangePlan,
    TreeAnnotationInfoModePlan,
    Visible3DSegmentationWidgetsPlan,
    ZDepthAnnotationOptionsPlan,
    ZNeighborHighlightCheckboxPlan,
)

from .custom_annotations_viewmodel import CustomAnnotationsViewModel
from .edit_id_viewmodel import EditIdViewModel
from .geometry_viewmodel import GeometryViewModel
from .label_edits_viewmodel import LabelEditViewModel
from .lineage_viewmodel import LineageViewModel
from .model_registry_viewmodel import ModelRegistryViewModel


class AnnotationDisplayViewModel(QObject):
    """Application-facing annotation display decisions and commands."""

    settingUpdateRequested = Signal(str, object)
    textAnnotationFlagsChanged = Signal(int, bool, bool)
    imageRefreshRequested = Signal()
    eraserTempResetRequested = Signal()
    annotationOptionStatesChanged = Signal(str, object)
    annotationModeTextUpdateRequested = Signal(str, str, bool)
    textAnnotationPixelModeChanged = Signal(bool)
    logInfoRequested = Signal(str)
    pixelModeActionDisabledChanged = Signal(bool)
    textResolutionChangeRequested = Signal(str)
    treeAnnotationMenuActionRequested = Signal(str, str, bool, bool)
    labelTreeAnnotationsEnabledChanged = Signal(bool)
    genNumTreeAnnotationsEnabledChanged = Signal(bool)
    allTextAnnotationsRefreshRequested = Signal()
    annotationOptionDisabledChanged = Signal(str, str, bool)
    annotationOptionVisibleChanged = Signal(str, str, bool)
    annotationOptionCheckedChanged = Signal(str, str, bool)
    zNeighborHighlightVisibleChanged = Signal(bool)
    zNeighborHighlightCheckedChanged = Signal(bool)
    zNeighborHighlightToggleConnectionRequested = Signal()
    annotationModeComboboxRestoreRequested = Signal(str, str)
    addNewIdsWhitelistToggleChanged = Signal(bool)
    annotationModeRestoreCallbackRequested = Signal(str)

    def __init__(
        self,
        model: AnnotationDisplayModel | None = None,
        custom_annotations: CustomAnnotationsViewModel | None = None,
        edit_id: EditIdViewModel | None = None,
        geometry: GeometryViewModel | None = None,
        label_edits: LabelEditViewModel | None = None,
        lineage: LineageViewModel | None = None,
        model_registry: ModelRegistryViewModel | None = None,
    ) -> None:
        super().__init__()
        self.model = model or AnnotationDisplayModel()
        self.custom_annotations = (
            custom_annotations or CustomAnnotationsViewModel()
        )
        self.edit_id = edit_id or EditIdViewModel()
        self.geometry = geometry or GeometryViewModel()
        self.label_edits = label_edits or LabelEditViewModel()
        self.lineage = lineage or LineageViewModel()
        self.model_registry = model_registry or ModelRegistryViewModel()

    def right_annotation_mode(self, **kwargs) -> str:
        return self.model.right_annotation_mode(**kwargs)

    def text_annotation_flags(self, **kwargs) -> tuple[bool, bool]:
        return self.model.text_annotation_flags(**kwargs)

    def annotation_mode_text(self, **kwargs) -> str:
        return self.model.annotation_mode_text(**kwargs)

    def annotation_flags_from_mode_text(self, text: str) -> dict[str, bool]:
        return self.model.annotation_flags_from_mode_text(text)

    def annotation_option_state_from_mode_text(
        self,
        text: str,
        *,
        num_zslices: bool = False,
    ) -> AnnotationOptionState:
        return self.model.annotation_option_state_from_mode_text(
            text,
            num_zslices=num_zslices,
        )

    def contours_requested(self, **kwargs) -> bool:
        return self.model.contours_requested(**kwargs)

    def moth_bud_lines_requested(self, **kwargs) -> bool:
        return self.model.moth_bud_lines_requested(**kwargs)

    def should_draw_moth_bud_line(self, **kwargs) -> bool:
        return self.model.should_draw_moth_bud_line(**kwargs)

    def should_draw_lineage_tree_lines(self, **kwargs) -> bool:
        return self.model.should_draw_lineage_tree_lines(**kwargs)

    def annotation_mode_setting_update(
        self,
        side: AnnotationSide,
        how: str,
    ) -> tuple[str, str]:
        return self.model.annotation_mode_setting_update(side, how)

    def change_annotation_mode(
        self,
        *,
        side: AnnotationSide,
        how: str,
        save_settings: bool,
        annot_cca_checked: bool,
        annot_ids_checked: bool,
        mode: str,
        is_data_loading: bool,
        eraser_checked: bool = False,
    ) -> AnnotationModeChangePlan:
        plan = self.model.annotation_mode_change_plan(
            side=side,
            how=how,
            save_settings=save_settings,
            annot_cca_checked=annot_cca_checked,
            annot_ids_checked=annot_ids_checked,
            mode=mode,
            is_data_loading=is_data_loading,
            eraser_checked=eraser_checked,
        )
        if plan.setting_update is not None:
            setting, value = plan.setting_update
            self.settingUpdateRequested.emit(setting, value)
        self.textAnnotationFlagsChanged.emit(
            plan.text_annotation_index,
            plan.is_cca_annotation,
            plan.is_id_annotation,
        )
        if plan.should_refresh_images:
            self.imageRefreshRequested.emit()
        if plan.should_reset_eraser_temp:
            self.eraserTempResetRequested.emit()
        return plan

    def change_annotation_options(
        self,
        *,
        side: AnnotationSide,
        clicked_option: AnnotationOption | None,
        save_settings: bool,
        ids: bool,
        cca: bool,
        contours: bool,
        segm_masks: bool,
        mother_bud_lines: bool,
        num_zslices: bool,
        nothing: bool,
    ) -> AnnotationOptionChangePlan:
        plan = self.model.annotation_option_change_plan(
            side=side,
            clicked_option=clicked_option,
            save_settings=save_settings,
            state=AnnotationOptionState(
                ids=ids,
                cca=cca,
                contours=contours,
                segm_masks=segm_masks,
                mother_bud_lines=mother_bud_lines,
                num_zslices=num_zslices,
                nothing=nothing,
            ),
        )
        self.annotationOptionStatesChanged.emit(side, plan.state)
        self.annotationModeTextUpdateRequested.emit(
            side,
            plan.mode_text,
            plan.save_settings,
        )
        return plan

    def refresh_annotation_mode_text(
        self,
        *,
        side: AnnotationSide,
        save_settings: bool,
        ids: bool,
        cca: bool,
        contours: bool,
        segm_masks: bool,
        mother_bud_lines: bool,
        nothing: bool,
    ) -> str:
        mode_text = self.model.annotation_mode_text(
            ids=ids,
            cca=cca,
            contours=contours,
            segm_masks=segm_masks,
            mother_bud_lines=mother_bud_lines,
            nothing=nothing,
        )
        self.annotationModeTextUpdateRequested.emit(
            side,
            mode_text,
            save_settings,
        )
        return mode_text

    def sync_annotation_options_from_mode_text(
        self,
        *,
        left_text: str,
        right_text: str,
        left_num_zslices: bool = False,
        right_num_zslices: bool = False,
    ) -> AnnotationOptionsFromModeTextPlan:
        plan = self.model.annotation_options_from_mode_text_plan(
            left_text=left_text,
            right_text=right_text,
            left_num_zslices=left_num_zslices,
            right_num_zslices=right_num_zslices,
        )
        for side, state in plan.state_updates:
            self.annotationOptionStatesChanged.emit(side, state)
        return plan

    def restore_saved_settings(
        self,
        *,
        settings_values: Mapping[str, object],
        left_num_zslices: bool = False,
        right_num_zslices: bool = False,
    ) -> AnnotationDisplaySettingsRestorePlan:
        plan = self.model.restore_saved_settings_plan(settings_values)
        self.annotationModeComboboxRestoreRequested.emit(
            'left',
            plan.left_mode,
        )
        self.annotationModeComboboxRestoreRequested.emit(
            'right',
            plan.right_mode,
        )
        self.addNewIdsWhitelistToggleChanged.emit(
            plan.add_new_ids_whitelist_toggle
        )
        self.sync_annotation_options_from_mode_text(
            left_text=plan.left_mode,
            right_text=plan.right_mode,
            left_num_zslices=left_num_zslices,
            right_num_zslices=right_num_zslices,
        )
        self.annotationModeRestoreCallbackRequested.emit('left')
        self.annotationModeRestoreCallbackRequested.emit('right')
        return plan

    def pixel_mode_setting_value(self, checked: bool) -> int:
        return self.model.pixel_mode_setting_value(checked)

    def change_pixel_mode(
        self,
        *,
        checked: bool,
        is_data_loaded: bool,
        high_resolution: bool,
    ) -> PixelModeChangePlan:
        plan = self.model.pixel_mode_change_plan(
            checked=checked,
            is_data_loaded=is_data_loaded,
            high_resolution=high_resolution,
        )
        setting, value = plan.setting_update
        self.settingUpdateRequested.emit(setting, value)
        if plan.should_update_text_pixel_mode:
            self.textAnnotationPixelModeChanged.emit(checked)
        if plan.should_refresh_images:
            self.imageRefreshRequested.emit()
        return plan

    def change_text_resolution(
        self,
        *,
        high_resolution: bool,
        is_data_loaded: bool,
    ) -> TextResolutionChangePlan:
        plan = self.model.text_resolution_change_plan(
            high_resolution=high_resolution,
            is_data_loaded=is_data_loaded,
        )
        self.logInfoRequested.emit(plan.log_message)
        self.pixelModeActionDisabledChanged.emit(plan.pixel_mode_disabled)
        if plan.should_update_annotations:
            self.textResolutionChangeRequested.emit(plan.mode)
        if plan.should_refresh_images:
            self.imageRefreshRequested.emit()
        return plan

    def change_label_tree_annotations(self, checked: bool) -> None:
        self.labelTreeAnnotationsEnabledChanged.emit(checked)

    def change_gen_num_tree_annotations(self, checked: bool) -> None:
        self.genNumTreeAnnotationsEnabledChanged.emit(checked)

    def change_tree_annotation_info_mode(
        self,
        checked: bool,
    ) -> TreeAnnotationInfoModePlan:
        plan = self.model.tree_annotation_info_mode_plan(checked)
        self.treeAnnotationMenuActionRequested.emit(
            'id',
            plan.action_text_contains,
            plan.enabled,
            plan.action_checked,
        )
        self.treeAnnotationMenuActionRequested.emit(
            'gen_num',
            plan.action_text_contains,
            plan.enabled,
            plan.action_checked,
        )
        self.labelTreeAnnotationsEnabledChanged.emit(
            plan.label_tree_annotations_enabled
        )
        self.genNumTreeAnnotationsEnabledChanged.emit(
            plan.gen_num_tree_annotations_enabled
        )
        if plan.should_refresh_annotations:
            self.allTextAnnotationsRefreshRequested.emit()
        return plan

    def enable_z_depth_annotation_options(
        self,
        *,
        is_3d: bool,
        ids: bool,
        cca: bool,
        contours: bool,
        segm_masks: bool,
        mother_bud_lines: bool,
        num_zslices: bool,
        nothing: bool,
    ) -> ZDepthAnnotationOptionsPlan:
        plan = self.model.z_depth_annotation_options_plan(
            is_3d=is_3d,
            state=AnnotationOptionState(
                ids=ids,
                cca=cca,
                contours=contours,
                segm_masks=segm_masks,
                mother_bud_lines=mother_bud_lines,
                num_zslices=num_zslices,
                nothing=nothing,
            ),
        )
        if not plan.should_apply:
            return plan

        for option, disabled in plan.disabled_updates:
            self.annotationOptionDisabledChanged.emit(
                'left',
                option,
                disabled,
            )
        self.annotationOptionStatesChanged.emit('left', plan.state)
        option_plan = self.model.annotation_option_change_plan(
            side='left',
            state=plan.state,
            clicked_option=plan.clicked_option,
            save_settings=plan.save_settings,
        )
        self.annotationOptionStatesChanged.emit('left', option_plan.state)
        self.annotationModeTextUpdateRequested.emit(
            'left',
            option_plan.mode_text,
            option_plan.save_settings,
        )
        return plan

    def update_visible_3d_segmentation_widgets(
        self,
        *,
        is_3d: bool,
    ) -> Visible3DSegmentationWidgetsPlan:
        plan = self.model.visible_3d_segmentation_widgets_plan(is_3d=is_3d)
        for side, option, visible in plan.visible_updates:
            self.annotationOptionVisibleChanged.emit(side, option, visible)
        for side, option, checked in plan.checked_updates:
            self.annotationOptionCheckedChanged.emit(side, option, checked)
        return plan

    def update_z_neighbor_highlight_checkbox(
        self,
        *,
        is_3d: bool,
    ) -> ZNeighborHighlightCheckboxPlan:
        plan = self.model.z_neighbor_highlight_checkbox_plan(is_3d=is_3d)
        if not plan.should_apply:
            return plan

        self.zNeighborHighlightVisibleChanged.emit(plan.visible)
        self.zNeighborHighlightCheckedChanged.emit(plan.checked)
        if plan.should_connect_toggle:
            self.zNeighborHighlightToggleConnectionRequested.emit()
        return plan
