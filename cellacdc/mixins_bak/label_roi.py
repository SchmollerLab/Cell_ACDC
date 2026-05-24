"""Qt view adapter for label-ROI workflows."""

from __future__ import annotations

import numpy as np
import os
from qtpy.QtCore import QMutex, Qt, QThread, QWaitCondition
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QAction, QMenu

from cellacdc import (
    apps,
    exception_handler,
    html_utils,
    qutils,
    settings_folderpath,
    widgets,
    workers,
)


class LabelRoiMixin:
    """Qt-facing adapter around Magic Labeller ROI workflows."""

    """Headless decisions for Magic Labeller ROI workflows."""

    yes_value = "Yes"
    no_value = "No"

    def checked_from_setting_value(self, value) -> bool:
        return value == self.yes_value

    def checked_setting_value(self, checked: bool) -> str:
        return self.yes_value if checked else self.no_value

    def cursor_points(self, x, y, checked: bool):
        if not checked:
            return [], []
        return [x], [y]

    def frame_range_length(
        self,
        enabled: bool,
        start_frame_index: int,
        stop_frame_number: int,
    ) -> int:
        if not enabled:
            return 1
        return stop_frame_number - start_frame_index

    def getLabelRoiImage(self):
        posData = self.data[self.pos_i]

        if self.labelRoiTrangeCheckbox.isChecked():
            start_frame_i = self.labelRoiStartFrameNoSpinbox.value() - 1
            stop_frame_n = self.labelRoiStopFrameNoSpinbox.value()
            tRangeLen = stop_frame_n - start_frame_i
        else:
            tRangeLen = 1

        if tRangeLen > 1:
            tRange = (start_frame_i, stop_frame_n)
        else:
            tRange = None

        if self.isSegm3D:
            if tRangeLen > 1:
                imgData = posData.img_data
            else:
                # Filtered data not existing
                imgData = posData.img_data[posData.frame_i]

            roi_zdepth = self.labelRoiZdepthSpinbox.value()
            if roi_zdepth == posData.SizeZ:
                z0 = 0
                z1 = posData.SizeZ
            elif roi_zdepth == 1:
                z0 = self.zSliceScrollBar.sliderPosition()
                z1 = z0 + 1
            else:
                if roi_zdepth % 2 != 0:
                    roi_zdepth += 1
                half_zdepth = int(roi_zdepth / 2)
                zc = self.zSliceScrollBar.sliderPosition() + 1
                z0 = zc - half_zdepth
                z0 = z0 if z0 >= 0 else 0
                z1 = zc + half_zdepth
                z1 = z1 if z1 < posData.SizeZ else posData.SizeZ

            if self.labelRoiIsRectRadioButton.isChecked():
                labelRoiSlice = self.labelRoiItem.slice(zRange=(z0, z1), tRange=tRange)
            elif self.labelRoiIsFreeHandRadioButton.isChecked():
                labelRoiSlice = self.freeRoiItem.slice(zRange=(z0, z1), tRange=tRange)
            elif self.labelRoiIsCircularRadioButton.isChecked():
                labelRoiSlice = self.labelRoiCircItemLeft.slice(
                    zRange=(z0, z1), tRange=tRange
                )
        else:
            if self.labelRoiIsRectRadioButton.isChecked():
                labelRoiSlice = self.labelRoiItem.slice(tRange=tRange)
            elif self.labelRoiIsFreeHandRadioButton.isChecked():
                labelRoiSlice = self.freeRoiItem.slice(tRange=tRange)
            elif self.labelRoiIsCircularRadioButton.isChecked():
                labelRoiSlice = self.labelRoiCircItemLeft.slice(tRange=tRange)
            if tRangeLen > 1:
                imgData = posData.img_data
            else:
                imgData = self.img1.image

        roiImg = imgData[labelRoiSlice]
        if self.labelRoiIsFreeHandRadioButton.isChecked():
            mask = self.freeRoiItem.mask()
        elif self.labelRoiIsCircularRadioButton.isChecked():
            mask = self.labelRoiCircItemLeft.mask()
        else:
            mask = None

        if mask is not None:
            # Copy roiImg otherwise we are replacing minimum inside original image
            roiImg = roiImg.copy()
            # Fill outside of freehand roi with minimum of the ROI image
            if tRangeLen > 1:
                for i in range(tRangeLen):
                    ith_roiImg = roiImg[i]
                    if self.isSegm3D:
                        roiImg[i, :, ~mask] = ith_roiImg.min()
                    else:
                        roiImg[i, ~mask] = ith_roiImg.min()
            else:
                if self.isSegm3D:
                    roiImg[:, ~mask] = roiImg.min()
                else:
                    roiImg[~mask] = roiImg.min()

        return roiImg, labelRoiSlice

    def getSecondChannelData(self):
        if self.secondChannelName is None:
            return

        posData = self.data[self.pos_i]

        fluo_ch = self.secondChannelName
        fluo_path, filename = self.getPathFromChName(fluo_ch, posData)
        if filename in posData.fluo_data_dict:
            fluo_data = posData.fluo_data_dict[filename]
        else:
            fluo_data, bkgrData = self.load_fluo_data(fluo_path)
            posData.fluo_data_dict[filename] = fluo_data
            posData.fluo_bkgrData_dict[filename] = bkgrData

        if self.labelRoiTrangeCheckbox.isChecked():
            start_frame_i = self.labelRoiStartFrameNoSpinbox.value() - 1
            stop_frame_n = self.labelRoiStopFrameNoSpinbox.value()
            tRangeLen = stop_frame_n - start_frame_i
        else:
            tRangeLen = 1

        if tRangeLen > 1:
            # fluo_img_data = fluo_data[start_frame_i:stop_frame_n]
            if self.isSegm3D or posData.SizeZ == 1:
                return fluo_data
            else:
                T, Z, Y, X = fluo_data.shape
                secondChannelData = np.zeros((T, Y, X), dtype=fluo_data.dtype)
                for frame_i, fluo_img in enumerate(fluo_data):
                    secondChannelData[frame_i] = self.get_2Dimg_from_3D(
                        fluo_data, frame_i=frame_i
                    )
                return secondChannelData
        else:
            if posData.SizeT > 1:
                fluo_img_data = fluo_data[posData.frame_i]
            else:
                fluo_img_data = fluo_data

            if self.isSegm3D or posData.SizeZ == 1:
                return fluo_img_data
            else:
                return self.get_2Dimg_from_3D(fluo_img_data)

    def indexRoiLab(self, roiLab, roiLabSlice, lab, brushID):
        # Delete only objects touching borders in X and Y not in Z
        if self.labelRoiAutoClearBorderCheckbox.isChecked():
            mask = np.zeros(roiLab.shape, dtype=bool)
            mask[..., 1:-1, 1:-1] = True
            roiLab = skimage.segmentation.clear_border(roiLab, mask=mask)

        roiLabMask = roiLab > 0
        roiLab[roiLabMask] += brushID - 1
        if self.labelRoiReplaceExistingObjectsCheckbox.isChecked():
            IDs_touched_by_new_objects = np.unique(lab[roiLabSlice][roiLabMask])
            for ID in IDs_touched_by_new_objects:
                lab[lab == ID] = 0

        lab[roiLabSlice][roiLabMask] = roiLab[roiLabMask]
        return lab

    def initLabelRoiModel(self):
        self.app.restoreOverrideCursor()
        # Ask which model
        self.initLabelRoiModelDialog = apps.QDialogSelectModel(parent=self)
        self.initLabelRoiModelDialog.exec_()
        if self.initLabelRoiModelDialog.cancel:
            self.logger.info("Magic labeller aborted.")
            self.initLabelRoiModelDialog = None
            return True
        self.app.setOverrideCursor(Qt.WaitCursor)
        model_name = self.initLabelRoiModelDialog.selectedModel
        self.labelRoiModel = self.repeatSegm(
            model_name=model_name, askSegmParams=True, is_label_roi=True
        )
        if self.labelRoiModel is None:
            self.initLabelRoiModelDialog = None
            return True
        self.labelRoiViewCurrentModelAction.setDisabled(False)
        self.initLabelRoiModelDialog = None
        return False

    def is_frame_range_valid(
        self,
        enabled: bool,
        start_frame_number: int,
        stop_frame_number: int,
    ) -> bool:
        return not enabled or start_frame_number <= stop_frame_number

    def labelRoiCancelled(self):
        self.labelRoiRunning = False
        self.app.restoreOverrideCursor()
        self.labelRoiItem.setPos((0, 0))
        self.labelRoiItem.setSize((0, 0))
        self.freeRoiItem.clear()
        self.logger.info("Magic labeller process cancelled.")

    def labelRoiCheckStartStopFrame(self):
        if not self.labelRoiTrangeCheckbox.isChecked():
            return True

        start_n = self.labelRoiStartFrameNoSpinbox.value()
        stop_n = self.labelRoiStopFrameNoSpinbox.value()
        if start_n <= stop_n:
            return True

        self.blinker = qutils.QControlBlink(
            self.labelRoiStopFrameNoSpinbox, qparent=self
        )
        self.blinker.start()
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            Stop frame number is less than start frame number!<br><br>
            What do you want to do?
        """)
        msg.warning(
            self,
            "Stop frame number lower than start",
            txt,
            buttonsTexts=("Cancel", "Segment only current frame"),
        )
        if msg.cancel:
            return False

        posData = self.data[self.pos_i]
        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i + 1)
        self.labelRoiStopFrameNoSpinbox.setValue(posData.frame_i + 1)

    @exception_handler
    def labelRoiDone(self, roiSegmData, isTimeLapse):
        self.setDisabled(False)

        posData = self.data[self.pos_i]
        self.setBrushID()

        if isTimeLapse:
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.mainPbar.setValue(0)
            current_frame_i = posData.frame_i
            start_frame_i = self.labelRoiStartFrameNoSpinbox.value() - 1
            for i, roiLab in enumerate(roiSegmData):
                frame_i = start_frame_i + i
                lab = posData.allData_li[frame_i]["labels"]
                store = True
                if lab is None:
                    if frame_i >= len(posData.segm_data):
                        lab = np.zeros_like(posData.segm_data[0])
                        posData.segm_data = np.append(
                            posData.segm_data, lab[np.newaxis], axis=0
                        )
                    else:
                        lab = posData.segm_data[frame_i]
                    store = False
                roiLabSlice = self.labelRoiSlice[1:]
                lab = self.indexRoiLab(roiLab, roiLabSlice, lab, posData.brushID)
                if store:
                    posData.frame_i = frame_i
                    posData.allData_li[frame_i]["labels"] = lab.copy()
                    self.get_data()
                    self.store_data(autosave=False)

            # Back to current frame
            posData.frame_i = current_frame_i
            self.get_data()
        else:
            roiLab = roiSegmData
            posData.lab = self.indexRoiLab(
                roiLab, self.labelRoiSlice, posData.lab, posData.brushID
            )

        self.update_rp()

        # Repeat tracking
        if self.autoIDcheckbox.isChecked():
            self.tracking(enforce=True, assign_unique_new_IDs=False)

        self.store_data()
        self.updateAllImages()

        self.labelRoiItem.setPos((0, 0))
        self.labelRoiItem.setSize((0, 0))
        self.freeRoiItem.clear()
        self.logger.info("Magic labeller done!")
        self.app.restoreOverrideCursor()

        self.labelRoiRunning = False
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        uncheckLabelRoiTRange = (
            self.labelRoiTrangeCheckbox.isChecked()
            and not self.labelRoiTrangeCheckbox.findChild(QAction).isChecked()
        )
        if uncheckLabelRoiTRange:
            self.labelRoiTrangeCheckbox.setChecked(False)

    def labelRoiFromCurrentFrameTriggered(self):
        posData = self.data[self.pos_i]
        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i + 1)

    def labelRoiToEndFramesTriggered(self):
        posData = self.data[self.pos_i]
        self.labelRoiStopFrameNoSpinbox.setValue(posData.SizeT)

    def labelRoiTrangeCheckboxToggled(self, checked):
        disabled = not checked
        self.labelRoiStartFrameNoSpinbox.setDisabled(disabled)
        self.labelRoiStopFrameNoSpinbox.setDisabled(disabled)
        self.labelRoiStartFrameNoSpinbox.label.setDisabled(disabled)
        self.labelRoiStopFrameNoSpinbox.label.setDisabled(disabled)
        self.labelRoiToEndFramesAction.setDisabled(disabled)
        self.labelRoiFromCurrentFrameAction.setDisabled(disabled)

        if disabled:
            return

        posData = self.data[self.pos_i]

        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i + 1)
        self.labelRoiStopFrameNoSpinbox.setValue(posData.SizeT)

    def labelRoiViewCurrentModel(self):
        from . import config

        ini_path = os.path.join(settings_folderpath, "last_params_segm_models.ini")
        configPars = config.ConfigParser()
        configPars.read(ini_path)
        model_name = self.labelRoiModel.model_name
        txt = f"Model: <b>{model_name}</b>"
        SECTION = f"{model_name}.init"
        txt = f"{txt}<br><br>[Initialization parameters]<br>"
        for option in configPars.options(SECTION):
            value = configPars[SECTION][option]
            param_txt = f"<i>{option}</i> = {value}<br>"
            txt = f"{txt}{param_txt}"

        SECTION = f"{model_name}.segment"
        txt = f"{txt}<br>[Segmentation parameters]<br>"
        for option in configPars.options(SECTION):
            value = configPars[SECTION][option]
            param_txt = f"<i>{option}</i> = {value}<br>"
            txt = f"{txt}{param_txt}"

        win = apps.ViewTextDialog(txt, parent=self)
        win.exec_()

    def labelRoiWorkerFinished(self):
        self.logger.info("Magic labeller closed.")
        self.labelRoiActiveWorkers.pop(-1)

    def labelRoi_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.labelRoiButton)
            self.connectLeftClickButtons()

            self.labelRoiStartFrameNoSpinbox.setMaximum(posData.SizeT)
            self.labelRoiStopFrameNoSpinbox.setMaximum(posData.SizeT)

            if self.labelRoiActiveWorkers:
                lastActiveWorker = self.labelRoiActiveWorkers[-1]
                self.labelRoiGarbageWorkers.append(lastActiveWorker)
                lastActiveWorker.finished.emit()
                self.logger.info("Collected garbage w5orker (magic labeller).")

            self.labelRoiToolbar.setVisible(True)
            if self.isSegm3D:
                self.labelRoiZdepthSpinbox.setDisabled(False)
            else:
                self.labelRoiZdepthSpinbox.setDisabled(True)

            # Start thread and pause it
            self.labelRoiThread = QThread()
            self.labelRoiMutex = QMutex()
            self.labelRoiWaitCond = QWaitCondition()

            labelRoiWorker = workers.LabelRoiWorker(self)

            labelRoiWorker.moveToThread(self.labelRoiThread)
            labelRoiWorker.finished.connect(self.labelRoiThread.quit)
            labelRoiWorker.finished.connect(labelRoiWorker.deleteLater)
            self.labelRoiThread.finished.connect(self.labelRoiThread.deleteLater)

            labelRoiWorker.finished.connect(self.labelRoiWorkerFinished)
            labelRoiWorker.sigLabellingDone.connect(self.labelRoiDone)
            labelRoiWorker.sigProgressBar.connect(self.workerUpdateProgressbar)

            labelRoiWorker.progress.connect(self.workerProgress)
            labelRoiWorker.critical.connect(self.workerCritical)

            self.labelRoiActiveWorkers.append(labelRoiWorker)

            self.labelRoiThread.started.connect(labelRoiWorker.run)
            self.labelRoiThread.start()

            # Add the rectROI to ax1
            self.ax1.addItem(self.labelRoiItem)
        elif self.initLabelRoiModelDialog is not None:
            # User is using other tools while the dialog is still open
            # --> we allow this because it's useful to be able to use
            # the ruler or check things --> do nothing
            pass
        else:
            self.labelRoiToolbar.setVisible(False)

            for worker in self.labelRoiActiveWorkers:
                worker._stop()
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

            self.labelRoiItem.setPos((0, 0))
            self.labelRoiItem.setSize((0, 0))
            self.freeRoiItem.clear()
            self.ax1.removeItem(self.labelRoiItem)
            self.updateLabelRoiCircularCursor(None, None, False)

    def loadLabelRoiLastParams(self):
        idx = "labelRoi_checkedRoiType"
        if idx in self.df_settings.index:
            checkedRoiType = self.df_settings.at[idx, "value"]
            for button in self.labelRoiTypesGroup.buttons():
                if button.text() == checkedRoiType:
                    button.setChecked(True)
                    break

        idx = "labelRoi_circRoiRadius"
        if idx in self.df_settings.index:
            circRoiRadius = self.df_settings.at[idx, "value"]
            self.labelRoiCircularRadiusSpinbox.setValue(int(circRoiRadius))

        idx = "labelRoi_roiZdepth"
        if idx in self.df_settings.index:
            roiZdepth = self.df_settings.at[idx, "value"]
            self.labelRoiZdepthSpinbox.setValue(int(roiZdepth))

        idx = "labelRoi_autoClearBorder"
        if idx in self.df_settings.index:
            clearBorder = self.df_settings.at[idx, "value"]
            checked = clearBorder == "Yes"
            self.labelRoiAutoClearBorderCheckbox.setChecked(checked)

        idx = "labelRoi_replaceExistingObjects"
        if idx in self.df_settings.index:
            val = self.df_settings.at[idx, "value"]
            checked = val == "Yes"
            self.labelRoiReplaceExistingObjectsCheckbox.setChecked(checked)

        if self.labelRoiIsCircularRadioButton.isChecked():
            self.labelRoiCircularRadiusSpinbox.setDisabled(False)

    def model_params_ini_path(self, settings_folderpath: str) -> str:
        return os.path.join(settings_folderpath, "last_params_segm_models.ini")

    def params_settings(
        self,
        *,
        checked_roi_type: str,
        circ_roi_radius: int,
        roi_zdepth: int,
        auto_clear_border: bool,
        replace_existing_objects: bool,
    ) -> LabelRoiParamsSettings:
        return LabelRoiParamsSettings(
            updates={
                "labelRoi_checkedRoiType": checked_roi_type,
                "labelRoi_circRoiRadius": circ_roi_radius,
                "labelRoi_roiZdepth": roi_zdepth,
                "labelRoi_autoClearBorder": self.checked_setting_value(
                    auto_clear_border
                ),
                "labelRoi_replaceExistingObjects": (
                    self.checked_setting_value(replace_existing_objects)
                ),
            }
        )

    def should_enable_range_controls(self, checked: bool) -> bool:
        return checked

    def should_show_circular_cursor(
        self,
        *,
        label_roi_checked: bool,
        circular_roi_checked: bool,
        label_roi_running: bool,
        cursor_checked: bool,
        existing_cursor_empty: bool,
    ) -> bool:
        return (
            label_roi_checked
            and circular_roi_checked
            and not label_roi_running
            and (cursor_checked or not existing_cursor_empty)
        )

    def should_uncheck_time_range(
        self,
        *,
        time_range_checked: bool,
        persistent_action_checked: bool,
    ) -> bool:
        return time_range_checked and not persistent_action_checked

    def showLabelRoiContextMenu(self, event):
        menu = QMenu(self.labelRoiButton)
        action = QAction("Re-initialize magic labeller model...")
        action.triggered.connect(self.initLabelRoiModel)
        menu.addAction(action)
        menu.exec_(QCursor.pos())

    def storeLabelRoiParams(self, value=None, checked=True):
        checkedRoiType = self.labelRoiTypesGroup.checkedButton().text()
        circRoiRadius = self.labelRoiCircularRadiusSpinbox.value()
        roiZdepth = self.labelRoiZdepthSpinbox.value()
        autoClearBorder = self.labelRoiAutoClearBorderCheckbox.isChecked()
        clearBorder = "Yes" if autoClearBorder else "No"
        self.df_settings.at["labelRoi_checkedRoiType", "value"] = checkedRoiType
        self.df_settings.at["labelRoi_circRoiRadius", "value"] = circRoiRadius
        self.df_settings.at["labelRoi_roiZdepth", "value"] = roiZdepth
        self.df_settings.at["labelRoi_autoClearBorder", "value"] = clearBorder
        self.df_settings.at["labelRoi_replaceExistingObjects", "value"] = (
            "Yes" if self.labelRoiReplaceExistingObjectsCheckbox.isChecked() else "No"
        )
        self.df_settings.to_csv(self.settings_csv_path)

    def time_range(
        self,
        enabled: bool,
        start_frame_index: int,
        stop_frame_number: int,
    ):
        if (
            self.frame_range_length(
                enabled,
                start_frame_index,
                stop_frame_number,
            )
            > 1
        ):
            return start_frame_index, stop_frame_number
        return None

    def updateLabelRoiCircularCursor(self, x, y, checked):
        if not self.labelRoiButton.isChecked():
            return
        if not self.labelRoiIsCircularRadioButton.isChecked():
            return
        if self.labelRoiRunning:
            return

        size = self.labelRoiCircularRadiusSpinbox.value()
        if not checked:
            xx, yy = [], []
        else:
            xx, yy = [x], [y]

        if not xx and len(self.labelRoiCircItemLeft.getData()[0]) == 0:
            return

        self.labelRoiCircItemLeft.setData(xx, yy, size=size)
        self.labelRoiCircItemRight.setData(xx, yy, size=size)

    def updateLabelRoiCircularSize(self, value):
        self.labelRoiCircItemLeft.setSize(value)
        self.labelRoiCircItemRight.setSize(value)

    def z_range(
        self,
        roi_zdepth: int,
        size_z: int,
        current_z_index: int,
    ) -> tuple[int, int]:
        if roi_zdepth == size_z:
            return 0, size_z
        if roi_zdepth == 1:
            return current_z_index, current_z_index + 1

        if roi_zdepth % 2 != 0:
            roi_zdepth += 1
        half_zdepth = int(roi_zdepth / 2)
        zc = current_z_index + 1
        z0 = max(zc - half_zdepth, 0)
        z1 = min(zc + half_zdepth, size_z)
        return z0, z1
