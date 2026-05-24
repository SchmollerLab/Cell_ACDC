"""Qt view adapter for annotation display workflows."""

from __future__ import annotations

import re

from cellacdc import _palettes, apps, html_utils, widgets

from dataclasses import dataclass
from typing import Literal, Mapping
GREEN_HEX = _palettes.green()


class AnnotationDisplayView:
    """Qt-facing adapter around annotation display and tool state."""

    LEGACY_METHODS = (
        'getAnnotateHowRightImage',
        'activateAnnotations',
        'gui_raiseBottomLayoutContextMenu',
        'annotateRightHowCombobox_cb',
        'drawIDsContComboBox_cb',
        'areContoursRequested',
        'areMothBudLinesRequested',
        'getMothBudLineScatterItem',
        'drawAllMothBudLines',
        'drawObjMothBudLines',
        'clearAllCellToCellLines',
        'drawAllLineageTreeLines',
        'drawObjLin_TreeMothBudLines',
        'getObjCentroid',
        'getObjOptsSegmLabels',
        'update_rp_metadata',
        'annotate_rip_and_bin_IDs',
        'clearAnnotItems',
        'setAllTextAnnotations',
        'labelRoiIsCircularRadioButtonToggled',
        'pxModeActionToggled',
        'changeTextResolution',
        'highLowResToggled',
        'annotGenNumTreeToggled',
        'annotLabelIDtreeToggled',
        'setAnnotInfoMode',
        'annotOptionClicked',
        'setDisabledAnnotCheckBoxesLeft',
        'setEnabledAnnotCheckBoxesLeftZdepthAxes',
        'setDisabledAnnotCheckBoxesRight',
        'annotOptionClickedRight',
        'setAnnotOptionsCcaMode',
        'setAnnotOptionsLin_treeMode',
        'setDrawAnnotComboboxText',
        'setDrawAnnotComboboxTextRight',
        'relabelSequentialCallback',
        'updateAnnotatedIDs',
        'rtTrackerActionToggled',
        'autoPilotToggled',
        'storeCurrentAnnotOptions_ax1',
        'storeCurrentAnnotOptions_ax2',
        'restoreAnnotOptions_ax1',
        'restoreAnnotOptions_ax2',
        'setDrawNothingAnnotations',
        'restoreAnnotationsOptions',
        'onDoubleSpaceBar',
        'zoomRectActionToggled',
        'zoomRectDone',
        'zoomRectCancelled',
        'keepToolActiveActionToggled',
        'applyToolNewFrameActionToggled',
        'keepAllToolsActiveActionToggled',
        'setVisible3DsegmWidgets',
        'showHighlightZneighCheckbox',
        'highlightZneighLabels_cb',
        'restoreSavedSettings',
        'uncheckAnnotOptions',
        'setDisabledAnnotOptions',
        'drawAnnotCombobox_to_options',
    )

    def __init__(self, host):
        object.__setattr__(self, 'host', host)        self._connect_view_model_signals()

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def _connect_view_model_signals(self):
        self.settingUpdateRequested.connect(
            self._apply_view_model_setting_update
        )
        self.textAnnotationFlagsChanged.connect(
            self._apply_text_annotation_flags
        )
        self.imageRefreshRequested.connect(
            self._refresh_images_from_view_model
        )
        self.eraserTempResetRequested.connect(
            self._reset_eraser_temp_from_view_model
        )
        self.annotationOptionStatesChanged.connect(
            self._apply_annotation_option_states
        )
        self.annotationModeTextUpdateRequested.connect(
            self._apply_annotation_mode_text_update
        )
        self.textAnnotationPixelModeChanged.connect(
            self._apply_text_annotation_pixel_mode
        )
        self.logInfoRequested.connect(self.logger.info)
        self.pixelModeActionDisabledChanged.connect(
            self.pxModeAction.setDisabled
        )
        self.textResolutionChangeRequested.connect(
            self._apply_text_resolution_change
        )
        self.treeAnnotationMenuActionRequested.connect(
            self._apply_tree_annotation_menu_action
        )
        self.labelTreeAnnotationsEnabledChanged.connect(
            self._apply_label_tree_annotations_enabled
        )
        self.genNumTreeAnnotationsEnabledChanged.connect(
            self._apply_gen_num_tree_annotations_enabled
        )
        self.allTextAnnotationsRefreshRequested.connect(
            self.setAllTextAnnotations
        )
        self.annotationOptionDisabledChanged.connect(
            self._apply_annotation_option_disabled
        )
        self.annotationOptionVisibleChanged.connect(
            self._apply_annotation_option_visible
        )
        self.annotationOptionCheckedChanged.connect(
            self._apply_annotation_option_checked
        )
        self.zNeighborHighlightVisibleChanged.connect(
            self._apply_z_neighbor_highlight_visible
        )
        self.zNeighborHighlightCheckedChanged.connect(
            self._apply_z_neighbor_highlight_checked
        )
        self.zNeighborHighlightToggleConnectionRequested.connect(
            self._connect_z_neighbor_highlight_toggle
        )
        self.annotationModeComboboxRestoreRequested.connect(
            self._apply_annotation_mode_combobox_restore
        )
        self.addNewIdsWhitelistToggleChanged.connect(
            self._apply_add_new_ids_whitelist_toggle
        )
        self.annotationModeRestoreCallbackRequested.connect(
            self._apply_annotation_mode_restore_callback
        )

    def _apply_view_model_setting_update(self, setting, value):
        self.df_settings.at[setting, 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)

    def _apply_text_annotation_flags(
        self,
        ax,
        is_cca_annotation,
        is_id_annotation,
    ):
        self.textAnnot[ax].setCcaAnnot(is_cca_annotation)
        self.textAnnot[ax].setLabelAnnot(is_id_annotation)

    def _refresh_images_from_view_model(self):
        self.updateAllImages()

    def _reset_eraser_temp_from_view_model(self):
        self.setTempImg1Eraser(None, init=True)

    def _apply_text_annotation_pixel_mode(self, checked):
        for ax in range(2):
            self.textAnnot[ax].setPxMode(checked)

    def _apply_text_resolution_change(self, mode):
        self.setAllIDs()
        posData = self.data[self.pos_i]
        allIDs = posData.allIDs
        img_shape = self.img1.image.shape[:2]
        self.textAnnot[0].changeResolution(mode, allIDs, self.ax1, img_shape)
        self.textAnnot[1].changeResolution(mode, allIDs, self.ax2, img_shape)

    def _apply_label_tree_annotations_enabled(self, checked):
        self.textAnnot[0].setLabelTreeAnnotationsEnabled(checked)

    def _apply_gen_num_tree_annotations_enabled(self, checked):
        self.textAnnot[0].setGenNumTreeAnnotationsEnabled(checked)

    def _apply_tree_annotation_menu_action(
        self,
        menu_name,
        text,
        should_contain_text,
        checked,
    ):
        if menu_name == 'id':
            menu = self.annotSettingsIDmenu
        else:
            menu = self.annotSettingsGenNumMenu

        for action in menu.actions():
            text_found = action.text().find(text) != -1
            if text_found == should_contain_text:
                action.setChecked(checked)
                break

    def _annotation_option_widgets(self, side):
        if side == 'right':
            return {
                'ids': self.annotIDsCheckboxRight,
                'cca': self.annotCcaInfoCheckboxRight,
                'contours': self.annotContourCheckboxRight,
                'segm_masks': self.annotSegmMasksCheckboxRight,
                'mother_bud_lines': self.drawMothBudLinesCheckboxRight,
                'num_zslices': self.annotNumZslicesCheckboxRight,
                'nothing': self.drawNothingCheckboxRight,
            }
        return {
            'ids': self.annotIDsCheckbox,
            'cca': self.annotCcaInfoCheckbox,
            'contours': self.annotContourCheckbox,
            'segm_masks': self.annotSegmMasksCheckbox,
            'mother_bud_lines': self.drawMothBudLinesCheckbox,
            'num_zslices': self.annotNumZslicesCheckbox,
            'nothing': self.drawNothingCheckbox,
        }

    def _annotation_option_state(self, side):
        widgets = self._annotation_option_widgets(side)
        return {
            name: widget.isChecked()
            for name, widget in widgets.items()
        }

    def _annotation_clicked_option(self, side, sender):
        for name, widget in self._annotation_option_widgets(side).items():
            if sender == widget:
                return name
        return None

    def _apply_annotation_option_states(self, side, state):
        widgets = self._annotation_option_widgets(side)
        widgets['ids'].setChecked(state.ids)
        widgets['cca'].setChecked(state.cca)
        widgets['contours'].setChecked(state.contours)
        widgets['segm_masks'].setChecked(state.segm_masks)
        widgets['mother_bud_lines'].setChecked(state.mother_bud_lines)
        widgets['num_zslices'].setChecked(state.num_zslices)
        widgets['nothing'].setChecked(state.nothing)

    def _apply_annotation_option_disabled(self, side, option, disabled):
        widgets = self._annotation_option_widgets(side)
        widgets[option].setDisabled(disabled)

    def _apply_annotation_option_visible(self, side, option, visible):
        widgets = self._annotation_option_widgets(side)
        widgets[option].setVisible(visible)

    def _apply_annotation_option_checked(self, side, option, checked):
        widgets = self._annotation_option_widgets(side)
        widgets[option].setChecked(checked)

    def _apply_z_neighbor_highlight_visible(self, visible):
        self.highlightZneighObjCheckbox.setVisible(visible)

    def _apply_z_neighbor_highlight_checked(self, checked):
        self.highlightZneighObjCheckbox.setChecked(checked)

    def _connect_z_neighbor_highlight_toggle(self):
        self.highlightZneighObjCheckbox.toggled.connect(
            self.highlightZneighLabels_cb
        )

    def _apply_annotation_mode_combobox_restore(self, side, text):
        if side == 'right':
            self.annotateRightHowCombobox.setCurrentText(text)
        else:
            self.drawIDsContComboBox.setCurrentText(text)

    def _apply_add_new_ids_whitelist_toggle(self, checked):
        self.addNewIDsWhitelistToggle = checked

    def _apply_annotation_mode_restore_callback(self, side):
        if side == 'right':
            self.annotateRightHowCombobox_cb(0)
        else:
            self.drawIDsContComboBox_cb(0)

    def _apply_annotation_mode_text_update(
        self,
        side,
        text,
        save_settings,
    ):
        if side == 'right':
            combo = self.annotateRightHowCombobox
            callback = self.annotateRightHowCombobox_cb
        else:
            combo = self.drawIDsContComboBox
            callback = self.drawIDsContComboBox_cb

        if text == combo.currentText():
            callback(0)

        combo.saveSettings = save_settings
        combo.setCurrentText(text)

    def getAnnotateHowRightImage(self):
        return self.right_annotation_mode(
            show_right_image=self.labelsGrad.showRightImgAction.isChecked(),
            use_right_specific_mode=self.rightBottomGroupbox.isChecked(),
            right_mode=self.annotateRightHowCombobox.currentText(),
            left_mode=self.drawIDsContComboBox.currentText(),
        )

    def activateAnnotations(self):
        if self.annotContourCheckbox.isChecked():
            return
        if self.annotSegmMasksCheckbox.isChecked():
            return

        self.annotSegmMasksCheckbox.setChecked(True)
        self.setDrawAnnotComboboxText()

    def gui_raiseBottomLayoutContextMenu(self, event):
        try:
            # Convert QPointF to QPoint
            self.bottomLayoutContextMenu.popup(event.globalPos().toPoint())
        except AttributeError:
            self.bottomLayoutContextMenu.popup(event.globalPos())

    def annotateRightHowCombobox_cb(self, idx):
        how = self.annotateRightHowCombobox.currentText()
        saveSettings = True
        if hasattr(self.annotateRightHowCombobox, 'saveSettings'):
            saveSettings = self.annotateRightHowCombobox.saveSettings

        self.change_annotation_mode(
            side='right',
            how=how,
            save_settings=saveSettings,
            annot_cca_checked=self.annotCcaInfoCheckboxRight.isChecked(),
            annot_ids_checked=self.annotIDsCheckboxRight.isChecked(),
            mode=self.modeComboBox.currentText(),
            is_data_loading=self.isDataLoading,
        )

    def drawIDsContComboBox_cb(self, idx):
        how = self.drawIDsContComboBox.currentText()
        saveSettings = True
        if hasattr(self.drawIDsContComboBox, 'saveSettings'):
            saveSettings = self.drawIDsContComboBox.saveSettings

        self.change_annotation_mode(
            side='left',
            how=how,
            save_settings=saveSettings,
            annot_cca_checked=self.annotCcaInfoCheckbox.isChecked(),
            annot_ids_checked=self.annotIDsCheckbox.isChecked(),
            mode=self.modeComboBox.currentText(),
            is_data_loading=self.isDataLoading,
            eraser_checked=self.eraserButton.isChecked(),
        )

    def areContoursRequested(self, ax):
        return self.contours_requested(
            ax=ax,
            left_contours=self.annotContourCheckbox.isChecked(),
            right_image_visible=self.labelsGrad.showRightImgAction.isChecked(),
            right_specific_mode=self.rightBottomGroupbox.isChecked(),
            right_contours=self.annotContourCheckboxRight.isChecked(),
        )

    def areMothBudLinesRequested(self, ax):
        return self.moth_bud_lines_requested(
            ax=ax,
            left_cca=self.annotCcaInfoCheckbox.isChecked(),
            left_mother_bud_lines=self.drawMothBudLinesCheckbox.isChecked(),
            right_image_visible=self.labelsGrad.showRightImgAction.isChecked(),
            right_specific_mode=self.rightBottomGroupbox.isChecked(),
            right_cca=self.annotCcaInfoCheckboxRight.isChecked(),
            right_mother_bud_lines=(
                self.drawMothBudLinesCheckboxRight.isChecked()
            ),
        )

    def getMothBudLineScatterItem(self, ax, new):
        if ax == 0:
            if new:
                return self.ax1_newMothBudLinesItem
            else:
                return self.ax1_oldMothBudLinesItem
        else:
            if new:
                return self.ax2_newMothBudLinesItem
            else:
                return self.ax2_oldMothBudLinesItem

    def drawAllMothBudLines(self):
        posData = self.data[self.pos_i]
        for obj in posData.rp:
            self.drawObjMothBudLines(obj, posData, ax=0)
            self.drawObjMothBudLines(obj, posData, ax=1)

    def drawObjMothBudLines(self, obj, posData, ax=0):
        areMothBudLinesRequested = self.areMothBudLinesRequested(ax)
        if not areMothBudLinesRequested:
            return

        mode = str(self.modeComboBox.currentText())

        if posData.cca_df is None:
            return

        ID = obj.label
        try:
            cca_df_ID = posData.cca_df.loc[ID]
        except KeyError:
            return

        ccs_ID = cca_df_ID['cell_cycle_stage']
        relationship = cca_df_ID['relationship']
        if not self.should_draw_moth_bud_line(
            cca_df_available=posData.cca_df is not None,
            mode=mode,
            object_visible=self.isObjVisible(obj.bbox),
            cell_cycle_stage=ccs_ID,
            relationship=relationship,
        ):
            return

        emerg_frame_i = cca_df_ID['emerg_frame_i']
        isNew = emerg_frame_i == posData.frame_i
        scatterItem = self.getMothBudLineScatterItem(ax, isNew)
        relative_ID = cca_df_ID['relative_ID']

        try:
            relative_rp_idx = posData.IDs_idxs[relative_ID]
        except KeyError:
            return

        relative_ID_obj = posData.rp[relative_rp_idx]
        y1, x1 = self.getObjCentroid(obj.centroid)
        y2, x2 = self.getObjCentroid(relative_ID_obj.centroid)
        xx, yy = self.geometry.line_coords(
            y1, x1, y2, x2, dashed=True
        )
        scatterItem.addPoints(xx, yy)

    def clearAllCellToCellLines(self):
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax2_newMothBudLinesItem.setData([], [])
        self.ax2_oldMothBudLinesItem.setData([], [])

    def drawAllLineageTreeLines(self):
        """

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

        Draw all lineage tree lines on the GUI.

        This method retrieves the lineage tree data and draws the lineage tree lines
        connecting cells and their respective mothers when the mother has split.
        """
        if not self.should_draw_lineage_tree_lines(
            lineage_tree_available=self.lineage_tree is not None,
            frames_count=(
                0 if self.lineage_tree is None
                else len(self.lineage_tree.frames_for_dfs)
            ),
        ):
            return

        self.clearAllCellToCellLines()
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        lin_tree_df = posData.allData_li[frame_i]['acdc_df']
        lin_tree_df_prev = posData.allData_li[frame_i-1]['acdc_df']
        rp = posData.rp
        prev_rp = posData.allData_li[frame_i-1]['regionprops']

        self.setTitleText()

        new_cells = lin_tree_df.index.difference(lin_tree_df_prev.index) # I could use this for the if already but this is probably faster for frames where nothing changes
        if new_cells.shape[0] == 0:
            return

        for ax in (0, 1):
            if not self.areMothBudLinesRequested(ax):
                continue

            for ID in new_cells:
                curr_obj = self.lineage.object_by_label(rp, ID)
                lin_tree_df_ID = lin_tree_df.loc[ID]

                # lin_tree_df_mother_ID = lin_tree_df_prev.loc[lin_tree_df_ID["parent_ID_tree"]]
                if lin_tree_df_ID["parent_ID_tree"] == -1: # make sure that new obj where the parents are not known get skipped
                    continue

                mother_obj = self.lineage.object_by_label(
                    prev_rp, lin_tree_df_ID["parent_ID_tree"]
                )

                emerg_frame_i = lin_tree_df_ID["emerg_frame_i"]
                isNew = emerg_frame_i == frame_i

                self.drawObjLin_TreeMothBudLines(ax, curr_obj, mother_obj, isNew, ID=ID)

    def drawObjLin_TreeMothBudLines(self, ax, obj, mother_obj, isNew, ID=None):
        """
        Draw moth-bud lines between an object and its mother object.

        Parameters
        ----------
        ax : cellacdc.widgets.MainPlotItem
            The Cell-ACDC GUI axes object to draw on.
        obj : Object
            The object for which to draw the moth-bud lines.
        mother_obj : Object
            The mother object to connect with.
        isNew : bool
            Indicates whether the object is new or not.
        ID : int, optional
            The ID of the object, by default None.
        """
        if not self.areMothBudLinesRequested(ax):
            return

        if not ID:
            ID = obj.label

        isObjVisible = self.isObjVisible(obj.bbox)

        if not isObjVisible:
            return

        scatterItem = self.getMothBudLineScatterItem(ax, isNew)

        y1, x1 = self.getObjCentroid(obj.centroid)
        y2, x2 = self.getObjCentroid(mother_obj.centroid)
        xx, yy = self.geometry.line_coords(
            y1, x1, y2, x2, dashed=True
        )
        scatterItem.addPoints(xx, yy)

    def getObjCentroid(self, obj_centroid):
        depth_axis = (
            self.switchPlaneCombobox.depthAxes() if self.isSegm3D else 'z'
        )
        return self.edit_id.project_centroid(
            obj_centroid,
            is_3d=self.isSegm3D,
            depth_axis=depth_axis,
        )

    def getObjOptsSegmLabels(self, obj):
        if not self.labelsGrad.showLabelsImgAction.isChecked():
            return

        objOpts = self.getObjTextAnnotOpts(obj, 'Draw only IDs', ax=1)
        return objOpts

    # @exec_time
    def update_rp_metadata(self, draw=True):
        posData = self.data[self.pos_i]
        # Add to rp dynamic metadata (e.g. cells annotated as dead)
        for i, obj in enumerate(posData.rp):
            ID = obj.label
            obj.excluded = ID in posData.binnedIDs
            obj.dead = ID in posData.ripIDs

    def annotate_rip_and_bin_IDs(self, updateLabel=False):
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if self.switchPlaneCombobox.isEnabled() and depthAxes != 'z':
            return

        posData = self.data[self.pos_i]
        binnedIDs_xx = []
        binnedIDs_yy = []
        ripIDs_xx = []
        ripIDs_yy = []
        for obj in posData.rp:
            obj.excluded = obj.label in posData.binnedIDs
            obj.dead = obj.label in posData.ripIDs
            if not self.isObjVisible(obj.bbox):
                continue

            if obj.excluded:
                y, x = self.getObjCentroid(obj.centroid)
                binnedIDs_xx.append(x)
                binnedIDs_yy.append(y)
                if updateLabel:
                    self.getObjOptsSegmLabels(obj)
                    how = self.drawIDsContComboBox.currentText()

            if obj.dead:
                y, x = self.getObjCentroid(obj.centroid)
                ripIDs_xx.append(x)
                ripIDs_yy.append(y)
                if updateLabel:
                    self.getObjOptsSegmLabels(obj)
                    how = self.drawIDsContComboBox.currentText()

        self.ax2_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
        self.ax2_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)
        self.ax1_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
        self.ax1_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)

    def clearAnnotItems(self):
        self.textAnnot[0].clear()
        self.textAnnot[1].clear()

    # @exec_time
    def setAllTextAnnotations(self, labelsToSkip=None):
        delROIsIDs = self.setLostNewOldPrevIDs()
        posData = self.data[self.pos_i]
        self.textAnnot[0].setAnnotations(
            posData=posData,
            labelsToSkip=labelsToSkip,
            isVisibleCheckFunc=self.isObjVisible,
            highlightedID=self.highlightedID,
            delROIsIDs=delROIsIDs,
            annotateLost=self.annotLostObjsToggle.isChecked(),
            getCurrentZfunc=self.z_lab,
            getObjCentroidFunc=self.getObjCentroid
        )
        self.textAnnot[1].setAnnotations(
            posData=posData, labelsToSkip=labelsToSkip,
            isVisibleCheckFunc=self.isObjVisible,
            highlightedID=self.highlightedID,
            delROIsIDs=delROIsIDs,
            annotateLost=self.annotLostObjsToggle.isChecked(),
            getCurrentZfunc=self.z_lab,
            getObjCentroidFunc=self.getObjCentroid
        )
        self.textAnnot[0].update()
        self.textAnnot[1].update()
        return delROIsIDs

    def labelRoiIsCircularRadioButtonToggled(self, checked):
        if checked:
            self.labelRoiCircularRadiusSpinbox.setDisabled(False)
        else:
            self.labelRoiCircularRadiusSpinbox.setDisabled(True)

    def pxModeActionToggled(self, checked):
        self.change_pixel_mode(
            checked=checked,
            is_data_loaded=self.isDataLoaded,
            high_resolution=self.highLowResAction.isChecked(),
        )

    def changeTextResolution(self):
        self.change_text_resolution(
            high_resolution=self.highLowResAction.isChecked(),
            is_data_loaded=self.isDataLoaded,
        )

    def highLowResToggled(self, clicked=True):
        self.changeTextResolution()

    def annotGenNumTreeToggled(self, checked):
        self.change_gen_num_tree_annotations(checked)

    def annotLabelIDtreeToggled(self, checked):
        self.change_label_tree_annotations(checked)

    def setAnnotInfoMode(self, checked):
        self.change_tree_annotation_info_mode(checked)

    def annotOptionClicked(self, clicked=True, sender=None, saveSettings=True):
        if sender is None:
            sender = self.sender()
        self.change_annotation_options(
            side='left',
            clicked_option=self._annotation_clicked_option('left', sender),
            save_settings=saveSettings,
            **self._annotation_option_state('left'),
        )

    def setDisabledAnnotCheckBoxesLeft(self, disabled):
        self.annotIDsCheckbox.setDisabled(disabled)
        self.annotCcaInfoCheckbox.setDisabled(disabled)
        self.annotContourCheckbox.setDisabled(disabled)
        self.annotSegmMasksCheckbox.setDisabled(disabled)
        self.drawMothBudLinesCheckbox.setDisabled(disabled)
        self.annotNumZslicesCheckbox.setDisabled(disabled)
        self.drawNothingCheckbox.setDisabled(disabled)

    def setEnabledAnnotCheckBoxesLeftZdepthAxes(self):
        self.enable_z_depth_annotation_options(
            is_3d=self.isSegm3D,
            **self._annotation_option_state('left'),
        )

    def setDisabledAnnotCheckBoxesRight(self, disabled):
        self.annotIDsCheckboxRight.setDisabled(disabled)
        self.annotCcaInfoCheckboxRight.setDisabled(disabled)
        self.annotContourCheckboxRight.setDisabled(disabled)
        self.annotSegmMasksCheckboxRight.setDisabled(disabled)
        self.drawMothBudLinesCheckboxRight.setDisabled(disabled)
        self.annotNumZslicesCheckboxRight.setDisabled(disabled)
        self.drawNothingCheckboxRight.setDisabled(disabled)

    def annotOptionClickedRight(
            self, clicked=True, sender=None, saveSettings=True
        ):
        if sender is None:
            sender = self.sender()
        self.change_annotation_options(
            side='right',
            clicked_option=self._annotation_clicked_option('right', sender),
            save_settings=saveSettings,
            **self._annotation_option_state('right'),
        )

    def setAnnotOptionsCcaMode(self):
        self.prevAnnotOptions = self.storeCurrentAnnotOptions_ax1(
            return_value=True
        )
        self.annotCcaInfoCheckbox.setChecked(True)
        self.annotIDsCheckbox.setChecked(False)
        self.drawMothBudLinesCheckbox.setChecked(False)
        self.setDrawAnnotComboboxText()

    def setAnnotOptionsLin_treeMode(self):
        # self.prevAnnotOptions = self.storeCurrentAnnotOptions_ax1(
        #     return_value=True
        # )
        self.annotCcaInfoCheckbox.setChecked(True)
        self.annotIDsCheckbox.setChecked(False)
        self.drawMothBudLinesCheckbox.setChecked(False)
        self.setDrawAnnotComboboxText()
        self.showTreeInfoCheckbox.setChecked(True)

    def setDrawAnnotComboboxText(self, saveSettings=True):
        state = self._annotation_option_state('left')
        state.pop('num_zslices')
        self.refresh_annotation_mode_text(
            side='left',
            save_settings=saveSettings,
            **state,
        )

    def setDrawAnnotComboboxTextRight(self, saveSettings=True):
        state = self._annotation_option_state('right')
        state.pop('num_zslices')
        self.refresh_annotation_mode_text(
            side='right',
            save_settings=saveSettings,
            **state,
        )

    def relabelSequentialCallback(self):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer' or mode == 'Cell cycle analysis':
            self.mode_controls_view.startBlinkingModeCB()
            return

        posData = self.data[self.pos_i]
        selectedPos = (posData.pos_foldername, )
        if len(self.data) > 1:
            selectedPos = self.askSelectPos(action='to process')
            if selectedPos is None:
                self.logger.info('Re-labelling process stopped.')
                return

        self.store_data()
        # acdc_df_concat = self.status_hover_view.concat_acdc_df()
        # load.store_unsaved_acdc_df(
        #     posData, acdc_df_concat,
        #     log_func=self.logger.info
        # )
        # if posData.SizeT > 1:
        self.progressWin = apps.QDialogWorkerProgress(
            title='Re-labelling sequential', parent=self.host,
            pbarDesc='Relabelling sequential...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)
        self.startRelabellingWorker(selectedPos)

    def updateAnnotatedIDs(self, oldIDs, newIDs, logger=print):
        logger('Updating annotated IDs...')
        posData = self.data[self.pos_i]

        posData.ripIDs = self.label_edits.remap_id_set(
            posData.ripIDs, oldIDs, newIDs
        )
        posData.binnedIDs = self.label_edits.remap_id_set(
            posData.binnedIDs, oldIDs, newIDs
        )
        self.keptObjectsIDs = widgets.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )

        customAnnotButtons = list(self.customAnnotDict.keys())
        for button in customAnnotButtons:
            customAnnotValues = self.customAnnotDict[button]
            annotatedIDs = customAnnotValues['annotatedIDs'][self.pos_i]
            mappedAnnotIDs = self.custom_annotations.remap_ids(
                annotatedIDs,
                oldIDs,
                newIDs,
            )
            customAnnotValues['annotatedIDs'][self.pos_i] = mappedAnnotIDs

    def rtTrackerActionToggled(self, checked):
        if not checked:
            return

        aliases = self.model_registry.real_time_tracker_aliases(
            reverse=True
        )
        if self.sender().text() in aliases:
            trackingAlgo = aliases[self.sender().text()]
        else:
            trackingAlgo = self.sender().text()
        self.df_settings.at['tracking_algorithm', 'value'] = trackingAlgo
        self.df_settings.to_csv(self.settings_csv_path)

        if self.sender().text() == 'YeaZ':
            msg = widgets.myMessageBox(wrapText=False)
            info_txt = html_utils.paragraph(f"""
                Note that YeaZ tracking algorithm tends to be sliglhtly more accurate
                overall, but it is <b>less capable of detecting segmentation
                errors.</b><br><br>
                If you need to correct as many segmentation errors as possible
                we recommend using Cell-ACDC tracking algorithm.
            """)
            msg.information(self.host, 'Info about YeaZ', info_txt)

        self.isRealTimeTrackerInitialized = False
        self.initRealTimeTracker()

    def autoPilotToggled(self, checked):
        self.autoPilotZoomToObjToolbar.setVisible(checked)
        if checked:
            self.autoPilotZoomToObjToggle.setChecked(False)
            self.autoPilotZoomToObjToggle.toggle()

    def storeCurrentAnnotOptions_ax1(self, return_value=False):
        if self.annotOptionsToRestore is not None:
            return

        checkboxes = [
            'annotIDsCheckbox',
            'annotCcaInfoCheckbox',
            'annotContourCheckbox',
            'annotSegmMasksCheckbox',
            'drawMothBudLinesCheckbox',
            'annotNumZslicesCheckbox',
            'drawNothingCheckbox',
        ]
        annotOptions = {}
        for checkboxName in checkboxes:
            checkbox = getattr(self, checkboxName)
            annotOptions[checkboxName] = checkbox.isChecked()
        if return_value:
            return annotOptions
        self.annotOptionsToRestore = annotOptions

    def storeCurrentAnnotOptions_ax2(self):
        if self.annotOptionsToRestoreRight is not None:
            return

        checkboxes = [
            'annotIDsCheckboxRight',
            'annotCcaInfoCheckboxRight',
            'annotContourCheckboxRight',
            'annotSegmMasksCheckboxRight',
            'drawMothBudLinesCheckboxRight',
            'annotNumZslicesCheckboxRight',
            'drawNothingCheckboxRight',
        ]
        self.annotOptionsToRestoreRight = {}
        for checkboxName in checkboxes:
            checkbox = getattr(self, checkboxName)
            self.annotOptionsToRestoreRight[checkboxName] = checkbox.isChecked()

    def restoreAnnotOptions_ax1(self, options=None):
        if options is None and not hasattr(self, 'annotOptionsToRestore'):
            return

        if options is None:
            options = self.annotOptionsToRestore

        if options is None:
            return

        for option, state in options.items():
            checkbox = getattr(self, option)
            checkbox.setChecked(state)

        self.setDrawAnnotComboboxText()
        self.annotOptionsToRestore = None

    def restoreAnnotOptions_ax2(self):
        if not hasattr(self, 'annotOptionsToRestoreRight'):
            return

        if self.annotOptionsToRestoreRight is None:
            return

        for option, state in self.annotOptionsToRestoreRight.items():
            checkbox = getattr(self, option)
            checkbox.setChecked(state)

        self.setDrawAnnotComboboxTextRight()
        self.annotOptionsToRestoreRight = None

    def setDrawNothingAnnotations(self):
        self.storeCurrentAnnotOptions_ax1()
        self.storeCurrentAnnotOptions_ax2()
        self.drawNothingCheckbox.setChecked(True)
        self.annotOptionClicked(
            sender=self.drawNothingCheckbox, saveSettings=False)
        self.drawNothingCheckboxRight.setChecked(True)
        self.annotOptionClickedRight(
            sender=self.drawNothingCheckboxRight, saveSettings=False
        )

    def restoreAnnotationsOptions(self):
        self.restoreAnnotOptions_ax1()
        self.restoreAnnotOptions_ax2()

    def onDoubleSpaceBar(self):
        how = self.drawIDsContComboBox.currentText()
        if how.find('nothing') == -1:
            self.storeCurrentAnnotOptions_ax1()
            self.drawNothingCheckbox.setChecked(True)
            self.annotOptionClicked(
                sender=self.drawNothingCheckbox, saveSettings=False
            )
        else:
            self.restoreAnnotOptions_ax1()

        how = self.annotateRightHowCombobox.currentText()
        if how.find('nothing') == -1:
            self.storeCurrentAnnotOptions_ax2()
            self.drawNothingCheckboxRight.setChecked(True)
            self.annotOptionClickedRight(
                sender=self.drawNothingCheckboxRight, saveSettings=False
            )
        else:
            self.restoreAnnotOptions_ax2()


    def zoomRectActionToggled(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
            self.ax1.addItem(self.zoomRectItem)
        else:
            self.zoomRectItem.setPos((0,0))
            self.zoomRectItem.setSize((0,0))
            self.ax1.removeItem(self.zoomRectItem)

    def zoomRectDone(self):
        xRange, yRange = self.ax1.viewRange()
        self.zoomRectItem.storeLastRange(xRange, yRange)

        ymin, xmin, ymax, xmax = self.zoomRectItem.bbox()

        self.zoomRectItem.setPos((0,0))
        self.zoomRectItem.setSize((0,0))

        self.ax1.setRange(
            xRange=(xmin, xmax),
            yRange=(ymin, ymax),
            padding=0
        )

    def zoomRectCancelled(self):
        self.isMouseDragImg1 = False
        self.zoomRectItem.setPos((0,0))
        self.zoomRectItem.setSize((0,0))

    def keepToolActiveActionToggled(self, checked, toolName=None):
        if toolName is None:
            parentToolButton = self.sender().parent()
            toolName = re.findall(r'Name: (.*)', parentToolButton.toolTip())[0]

        if checked:
            self.df_settings.at[toolName, 'value'] = 'keepActive'
        else:
            self.df_settings = self.df_settings.drop(
                index=toolName, errors='ignore'
            )
        self.df_settings.to_csv(self.settings_csv_path)

    def applyToolNewFrameActionToggled(self, checked, toolName=None):
        if toolName is None:
            parentToolButton = self.sender().parent()
            toolName = re.findall(r'Name: (.*)', parentToolButton.toolTip())[0]
        toolName = toolName.strip()
        button = self.applyToolNewFrameButtons[toolName]
        toolName = toolName.replace(' ', '_')
        settingName = f'{toolName}_applyNewFrame'
        if checked:
            self.df_settings.at[settingName, 'value'] = 'applyNewFrame'
            button.setStyleSheet(f'background-color: {GREEN_HEX}')
        else:
            self.df_settings = self.df_settings.drop(
                index=settingName, errors='ignore'
            )
            button.setStyleSheet('background-color: none')
        self.df_settings.to_csv(self.settings_csv_path)

    def keepAllToolsActiveActionToggled(self, checked):
        for action in self.keepToolActiveActions.values():
            action.setChecked(checked)

        data_loaded = True
        if not hasattr(self, 'data'):
            data_loaded = False
            try:
                self.labelRoiTrangeCheckbox.disconnect()
            except TypeError:
                pass
        self.labelRoiTrangeCheckbox.setChecked(checked) # why this is not wrapped in a QAction?

        if data_loaded:
            self.labelRoiTrangeCheckbox.toggled.connect(
                self.labelRoiTrangeCheckboxToggled
            )

    def setVisible3DsegmWidgets(self):
        self.update_visible_3d_segmentation_widgets(
            is_3d=self.isSegm3D,
        )

    def showHighlightZneighCheckbox(self):
        self.update_z_neighbor_highlight_checkbox(
            is_3d=self.isSegm3D,
        )

    def highlightZneighLabels_cb(self, checked):
        if checked:
            pass
        else:
            pass

    def restoreSavedSettings(self):
        self.restore_saved_settings(
            settings_values=self.df_settings['value'].to_dict(),
            left_num_zslices=self.annotNumZslicesCheckbox.isChecked(),
            right_num_zslices=self.annotNumZslicesCheckboxRight.isChecked(),
        )

    def uncheckAnnotOptions(self, left=True, right=True):
        # Left
        if left:
            self.annotIDsCheckbox.setChecked(False)
            self.annotCcaInfoCheckbox.setChecked(False)
            self.annotContourCheckbox.setChecked(False)
            self.annotSegmMasksCheckbox.setChecked(False)
            self.drawMothBudLinesCheckbox.setChecked(False)
            self.drawNothingCheckbox.setChecked(False)

        # Right
        if right:
            self.annotIDsCheckboxRight.setChecked(False)
            self.annotCcaInfoCheckboxRight.setChecked(False)
            self.annotContourCheckboxRight.setChecked(False)
            self.annotSegmMasksCheckboxRight.setChecked(False)
            self.drawMothBudLinesCheckboxRight.setChecked(False)
            self.drawNothingCheckboxRight.setChecked(False)

    def setDisabledAnnotOptions(self, disabled):
        # Left
        self.annotIDsCheckbox.setDisabled(disabled)
        self.annotCcaInfoCheckbox.setDisabled(disabled)
        self.annotContourCheckbox.setDisabled(disabled)
        # self.annotSegmMasksCheckbox.setDisabled(disabled)
        self.drawMothBudLinesCheckbox.setDisabled(disabled)
        # self.drawNothingCheckbox.setDisabled(disabled)

        # Right
        self.annotIDsCheckboxRight.setDisabled(disabled)
        self.annotCcaInfoCheckboxRight.setDisabled(disabled)
        self.annotContourCheckboxRight.setDisabled(disabled)
        # self.annotSegmMasksCheckboxRight.setDisabled(disabled)
        self.drawMothBudLinesCheckboxRight.setDisabled(disabled)
        # self.drawNothingCheckboxRight.setDisabled(disabled)

    def drawAnnotCombobox_to_options(self):
        self.sync_annotation_options_from_mode_text(
            left_text=self.drawIDsContComboBox.currentText(),
            right_text=self.annotateRightHowCombobox.currentText(),
            left_num_zslices=self.annotNumZslicesCheckbox.isChecked(),
            right_num_zslices=self.annotNumZslicesCheckboxRight.isChecked(),
        )