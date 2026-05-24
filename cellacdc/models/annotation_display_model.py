"""Qt-free model rules for annotation display workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


AnnotationSide = Literal['left', 'right']
AnnotationOption = Literal[
    'ids',
    'cca',
    'mother_bud_lines',
    'contours',
    'segm_masks',
    'nothing',
    'num_zslices',
]


@dataclass(frozen=True)
class AnnotationOptionState:
    """Checkbox state for one annotation side."""

    ids: bool = False
    cca: bool = False
    contours: bool = False
    segm_masks: bool = False
    mother_bud_lines: bool = False
    num_zslices: bool = False
    nothing: bool = False


@dataclass(frozen=True)
class AnnotationModeChangePlan:
    """Pure outcome of changing the annotation mode for one image side."""

    side: AnnotationSide
    setting_update: tuple[str, str] | None
    text_annotation_index: int
    is_cca_annotation: bool
    is_id_annotation: bool
    should_refresh_images: bool
    should_reset_eraser_temp: bool = False


@dataclass(frozen=True)
class AnnotationOptionChangePlan:
    """Pure outcome of changing annotation option checkboxes."""

    side: AnnotationSide
    state: AnnotationOptionState
    mode_text: str
    save_settings: bool


@dataclass(frozen=True)
class AnnotationOptionsFromModeTextPlan:
    """Pure outcome of syncing option checkboxes from combobox text."""

    state_updates: tuple[tuple[AnnotationSide, AnnotationOptionState], ...]


@dataclass(frozen=True)
class AnnotationDisplaySettingsRestorePlan:
    """Pure outcome of restoring annotation display settings."""

    left_mode: str
    right_mode: str
    add_new_ids_whitelist_toggle: bool


@dataclass(frozen=True)
class PixelModeChangePlan:
    """Pure outcome of toggling annotation pixel mode."""

    setting_update: tuple[str, int]
    should_update_text_pixel_mode: bool
    should_refresh_images: bool


@dataclass(frozen=True)
class TextResolutionChangePlan:
    """Pure outcome of toggling annotation text resolution."""

    mode: str
    log_message: str
    pixel_mode_disabled: bool
    should_update_annotations: bool
    should_refresh_images: bool


@dataclass(frozen=True)
class TreeAnnotationInfoModePlan:
    """Pure outcome of toggling tree annotation info mode."""

    enabled: bool
    action_text_contains: str
    action_checked: bool
    label_tree_annotations_enabled: bool
    gen_num_tree_annotations_enabled: bool
    should_refresh_annotations: bool


@dataclass(frozen=True)
class ZDepthAnnotationOptionsPlan:
    """Pure outcome of enabling left annotation options for z-depth axes."""

    should_apply: bool
    disabled_updates: tuple[tuple[AnnotationOption, bool], ...] = ()
    state: AnnotationOptionState | None = None
    clicked_option: AnnotationOption | None = None
    save_settings: bool = False


@dataclass(frozen=True)
class Visible3DSegmentationWidgetsPlan:
    """Pure outcome of updating 3D-only annotation option widgets."""

    visible_updates: tuple[tuple[AnnotationSide, AnnotationOption, bool], ...]
    checked_updates: tuple[tuple[AnnotationSide, AnnotationOption, bool], ...]


@dataclass(frozen=True)
class ZNeighborHighlightCheckboxPlan:
    """Pure outcome of updating z-neighbor highlight checkbox state."""

    should_apply: bool
    visible: bool = False
    checked: bool = False
    should_connect_toggle: bool = False


class AnnotationDisplayModel:
    """Headless annotation display decisions."""

    def right_annotation_mode(
        self,
        *,
        show_right_image: bool,
        use_right_specific_mode: bool,
        right_mode: str,
        left_mode: str,
    ) -> str:
        if not show_right_image:
            return 'nothing'
        return right_mode if use_right_specific_mode else left_mode

    def text_annotation_flags(
        self,
        *,
        annot_cca_checked: bool,
        annot_ids_checked: bool,
        mode: str,
    ) -> tuple[bool, bool]:
        is_lineage_mode = mode == 'Normal division: Lineage tree'
        is_cca = annot_cca_checked and not is_lineage_mode
        is_id = annot_ids_checked or (annot_cca_checked and is_lineage_mode)
        return is_cca, is_id

    def annotation_mode_text(
        self,
        *,
        ids: bool = False,
        cca: bool = False,
        contours: bool = False,
        segm_masks: bool = False,
        mother_bud_lines: bool = False,
        nothing: bool = False,
    ) -> str:
        if ids:
            if contours:
                return 'Draw IDs and contours'
            if segm_masks:
                return 'Draw IDs and overlay segm. masks'
            return 'Draw only IDs'
        if cca:
            if contours:
                return 'Draw cell cycle info and contours'
            if segm_masks:
                return 'Draw cell cycle info and overlay segm. masks'
            return 'Draw only cell cycle info'
        if segm_masks:
            return 'Draw only overlay segm. masks'
        if contours:
            return 'Draw only contours'
        if mother_bud_lines:
            return 'Draw only mother-bud lines'
        if nothing:
            return 'Draw nothing'
        return 'Draw nothing'

    def annotation_flags_from_mode_text(self, text: str) -> dict[str, bool]:
        return {
            'ids': 'IDs' in text,
            'cca': 'cell cycle info' in text,
            'contours': 'contours' in text,
            'segm_masks': 'segm. masks' in text,
            'mother_bud_lines': 'mother-bud lines' in text,
            'nothing': 'nothing' in text,
        }

    def annotation_option_state_from_mode_text(
        self,
        text: str,
        *,
        num_zslices: bool = False,
    ) -> AnnotationOptionState:
        flags = self.annotation_flags_from_mode_text(text)
        return AnnotationOptionState(
            ids=flags['ids'],
            cca=flags['cca'],
            contours=flags['contours'],
            segm_masks=flags['segm_masks'],
            mother_bud_lines=flags['mother_bud_lines'],
            num_zslices=num_zslices,
            nothing=flags['nothing'],
        )

    def annotation_options_from_mode_text_plan(
        self,
        *,
        left_text: str,
        right_text: str,
        left_num_zslices: bool = False,
        right_num_zslices: bool = False,
    ) -> AnnotationOptionsFromModeTextPlan:
        return AnnotationOptionsFromModeTextPlan(
            state_updates=(
                (
                    'left',
                    self.annotation_option_state_from_mode_text(
                        left_text,
                        num_zslices=left_num_zslices,
                    ),
                ),
                (
                    'right',
                    self.annotation_option_state_from_mode_text(
                        right_text,
                        num_zslices=right_num_zslices,
                    ),
                ),
            )
        )

    def restore_saved_settings_plan(
        self,
        settings_values: Mapping[str, object],
    ) -> AnnotationDisplaySettingsRestorePlan:
        return AnnotationDisplaySettingsRestorePlan(
            left_mode=str(
                settings_values.get(
                    'how_draw_annotations',
                    'Draw IDs and contours',
                )
            ),
            right_mode=str(
                settings_values.get(
                    'how_draw_right_annotations',
                    'Draw IDs and overlay segm. masks',
                )
            ),
            add_new_ids_whitelist_toggle=(
                settings_values.get('addNewIDsWhitelistToggle', 'Yes') == 'Yes'
            ),
        )

    def contours_requested(
        self,
        *,
        ax: int,
        left_contours: bool,
        right_image_visible: bool,
        right_specific_mode: bool,
        right_contours: bool,
    ) -> bool:
        if ax == 0:
            return left_contours
        if not right_image_visible:
            return False
        if right_specific_mode:
            return right_contours
        return left_contours

    def moth_bud_lines_requested(
        self,
        *,
        ax: int,
        left_cca: bool,
        left_mother_bud_lines: bool,
        right_image_visible: bool,
        right_specific_mode: bool,
        right_cca: bool,
        right_mother_bud_lines: bool,
    ) -> bool:
        if ax == 0:
            return left_cca or left_mother_bud_lines
        if not right_image_visible:
            return False
        if right_specific_mode:
            return right_cca or right_mother_bud_lines
        return left_cca or left_mother_bud_lines

    def should_draw_moth_bud_line(
        self,
        *,
        cca_df_available: bool,
        mode: str,
        object_visible: bool,
        cell_cycle_stage: str,
        relationship: str,
    ) -> bool:
        return (
            cca_df_available
            and mode != 'Normal division: Lineage Tree'
            and object_visible
            and cell_cycle_stage != 'G1'
            and relationship == 'bud'
        )

    def should_draw_lineage_tree_lines(
        self,
        *,
        lineage_tree_available: bool,
        frames_count: int,
    ) -> bool:
        return lineage_tree_available and frames_count >= 2

    def annotation_mode_setting_update(
        self,
        side: AnnotationSide,
        how: str,
    ) -> tuple[str, str]:
        setting = (
            'how_draw_right_annotations'
            if side == 'right'
            else 'how_draw_annotations'
        )
        return setting, how

    def annotation_mode_change_plan(
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
        setting_update = None
        if save_settings:
            setting_update = self.annotation_mode_setting_update(side, how)

        is_cca, is_id = self.text_annotation_flags(
            annot_cca_checked=annot_cca_checked,
            annot_ids_checked=annot_ids_checked,
            mode=mode,
        )
        return AnnotationModeChangePlan(
            side=side,
            setting_update=setting_update,
            text_annotation_index=1 if side == 'right' else 0,
            is_cca_annotation=is_cca,
            is_id_annotation=is_id,
            should_refresh_images=not is_data_loading,
            should_reset_eraser_temp=side == 'left' and eraser_checked,
        )

    def annotation_option_change_plan(
        self,
        *,
        side: AnnotationSide,
        state: AnnotationOptionState,
        clicked_option: AnnotationOption | None,
        save_settings: bool,
    ) -> AnnotationOptionChangePlan:
        values = {
            'ids': state.ids,
            'cca': state.cca,
            'contours': state.contours,
            'segm_masks': state.segm_masks,
            'mother_bud_lines': state.mother_bud_lines,
            'num_zslices': state.num_zslices,
            'nothing': state.nothing,
        }

        if values['ids'] and clicked_option == 'ids':
            values['cca'] = False
            values['mother_bud_lines'] = False

        if values['cca'] and clicked_option == 'cca':
            values['ids'] = False
            values['mother_bud_lines'] = False

        if (
            values['mother_bud_lines']
            and clicked_option == 'mother_bud_lines'
        ):
            values['ids'] = False
            values['cca'] = False

        if values['contours'] and clicked_option == 'contours':
            values['segm_masks'] = False

        if values['segm_masks'] and clicked_option == 'segm_masks':
            values['contours'] = False

        if clicked_option == 'nothing':
            values['ids'] = False
            values['cca'] = False
            values['contours'] = False
            values['segm_masks'] = False
            values['mother_bud_lines'] = False
            values['num_zslices'] = False
        else:
            values['nothing'] = False

        if clicked_option == 'num_zslices':
            values['ids'] = True
            values['nothing'] = False

        new_state = AnnotationOptionState(**values)
        return AnnotationOptionChangePlan(
            side=side,
            state=new_state,
            mode_text=self.annotation_mode_text(
                ids=new_state.ids,
                cca=new_state.cca,
                contours=new_state.contours,
                segm_masks=new_state.segm_masks,
                mother_bud_lines=new_state.mother_bud_lines,
                nothing=new_state.nothing,
            ),
            save_settings=save_settings,
        )

    def pixel_mode_setting_value(self, checked: bool) -> int:
        return int(checked)

    def pixel_mode_change_plan(
        self,
        *,
        checked: bool,
        is_data_loaded: bool,
        high_resolution: bool,
    ) -> PixelModeChangePlan:
        return PixelModeChangePlan(
            setting_update=('pxMode', self.pixel_mode_setting_value(checked)),
            should_update_text_pixel_mode=is_data_loaded and high_resolution,
            should_refresh_images=is_data_loaded,
        )

    def text_resolution_change_plan(
        self,
        *,
        high_resolution: bool,
        is_data_loaded: bool,
    ) -> TextResolutionChangePlan:
        mode = 'high' if high_resolution else 'low'
        return TextResolutionChangePlan(
            mode=mode,
            log_message=f'Switching to {mode} for the text annnotations...',
            pixel_mode_disabled=not high_resolution,
            should_update_annotations=is_data_loaded,
            should_refresh_images=is_data_loaded,
        )

    def tree_annotation_info_mode_plan(
        self,
        checked: bool,
    ) -> TreeAnnotationInfoModePlan:
        return TreeAnnotationInfoModePlan(
            enabled=checked,
            action_text_contains='tree',
            action_checked=checked,
            label_tree_annotations_enabled=checked,
            gen_num_tree_annotations_enabled=checked,
            should_refresh_annotations=True,
        )

    def z_depth_annotation_options_plan(
        self,
        *,
        is_3d: bool,
        state: AnnotationOptionState,
    ) -> ZDepthAnnotationOptionsPlan:
        if not is_3d:
            return ZDepthAnnotationOptionsPlan(should_apply=False)

        return ZDepthAnnotationOptionsPlan(
            should_apply=True,
            disabled_updates=(('ids', False), ('contours', False)),
            state=AnnotationOptionState(
                ids=True,
                cca=state.cca,
                contours=True,
                segm_masks=state.segm_masks,
                mother_bud_lines=state.mother_bud_lines,
                num_zslices=state.num_zslices,
                nothing=state.nothing,
            ),
            clicked_option='ids',
            save_settings=False,
        )

    def visible_3d_segmentation_widgets_plan(
        self,
        *,
        is_3d: bool,
    ) -> Visible3DSegmentationWidgetsPlan:
        visible_updates = (
            ('left', 'num_zslices', is_3d),
            ('right', 'num_zslices', is_3d),
        )
        checked_updates = ()
        if not is_3d:
            checked_updates = (
                ('left', 'num_zslices', False),
                ('right', 'num_zslices', False),
            )
        return Visible3DSegmentationWidgetsPlan(
            visible_updates=visible_updates,
            checked_updates=checked_updates,
        )

    def z_neighbor_highlight_checkbox_plan(
        self,
        *,
        is_3d: bool,
    ) -> ZNeighborHighlightCheckboxPlan:
        if not is_3d:
            return ZNeighborHighlightCheckboxPlan(should_apply=False)
        return ZNeighborHighlightCheckboxPlan(
            should_apply=True,
            visible=True,
            checked=True,
            should_connect_toggle=True,
        )
