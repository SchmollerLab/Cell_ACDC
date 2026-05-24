"""Qt view adapter for label-ROI workflows."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QMutex, Qt, QThread, QWaitCondition
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QAction, QMenu

from cellacdc import (
    apps,
    config,
    exception_handler,
    html_utils,
    qutils,
    settings_folderpath,
    widgets,
    workers,
)
from cellacdc.viewmodels.label_roi_viewmodel import LabelRoiViewModel


class LabelRoiView:
    """Qt-facing adapter around Magic Labeller ROI workflows."""

    LEGACY_METHODS = (
        'labelRoiCancelled',
        'labelRoiCheckStartStopFrame',
        'getSecondChannelData',
        'labelRoiToEndFramesTriggered',
        'labelRoiFromCurrentFrameTriggered',
        'showLabelRoiContextMenu',
        'initLabelRoiModel',
        'labelRoiViewCurrentModel',
        'storeLabelRoiParams',
        'loadLabelRoiLastParams',
        'updateLabelRoiCircularSize',
        'updateLabelRoiCircularCursor',
        'getLabelRoiImage',
        'labelRoiTrangeCheckboxToggled',
        'labelRoi_cb',
        'labelRoiWorkerFinished',
        'indexRoiLab',
        'labelRoiDone',
    )

    def __init__(self, host, view_model: LabelRoiViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def labelRoiCancelled(self):
        self.labelRoiRunning = False
        self.app.restoreOverrideCursor()
        self.labelRoiItem.setPos((0,0))
        self.labelRoiItem.setSize((0,0))
        self.freeRoiItem.clear()
        self.logger.info('Magic labeller process cancelled.')

    def labelRoiCheckStartStopFrame(self):
        enabled = self.labelRoiTrangeCheckbox.isChecked()
        start_n = self.labelRoiStartFrameNoSpinbox.value()
        stop_n = self.labelRoiStopFrameNoSpinbox.value()
        if self.view_model.is_frame_range_valid(enabled, start_n, stop_n):
            return True

        self.blinker = qutils.QControlBlink(
            self.labelRoiStopFrameNoSpinbox,
            qparent=self.host
        )
        self.blinker.start()
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            Stop frame number is less than start frame number!<br><br>
            What do you want to do?
        """)
        msg.warning(
            self.host, 'Stop frame number lower than start', txt,
            buttonsTexts=('Cancel', 'Segment only current frame')
        )
        if msg.cancel:
            return False

        posData = self.data[self.pos_i]
        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i+1)
        self.labelRoiStopFrameNoSpinbox.setValue(posData.frame_i+1)

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

        start_frame_i = self.labelRoiStartFrameNoSpinbox.value()-1
        stop_frame_n = self.labelRoiStopFrameNoSpinbox.value()
        tRangeLen = self.view_model.frame_range_length(
            self.labelRoiTrangeCheckbox.isChecked(),
            start_frame_i,
            stop_frame_n,
        )

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

    def labelRoiToEndFramesTriggered(self):
        posData = self.data[self.pos_i]
        self.labelRoiStopFrameNoSpinbox.setValue(posData.SizeT)

    def labelRoiFromCurrentFrameTriggered(self):
        posData = self.data[self.pos_i]
        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i+1)

    def showLabelRoiContextMenu(self, event):
        menu = QMenu(self.labelRoiButton)
        action = QAction('Re-initialize magic labeller model...')
        action.triggered.connect(self.initLabelRoiModel)
        menu.addAction(action)
        menu.exec_(QCursor.pos())

    def initLabelRoiModel(self):
        self.app.restoreOverrideCursor()
        # Ask which model
        self.initLabelRoiModelDialog = apps.QDialogSelectModel(
            parent=self.host
        )
        self.initLabelRoiModelDialog.exec_()
        if self.initLabelRoiModelDialog.cancel:
            self.logger.info('Magic labeller aborted.')
            self.initLabelRoiModelDialog = None
            return True
        self.app.setOverrideCursor(Qt.WaitCursor)
        model_name = self.initLabelRoiModelDialog.selectedModel
        self.labelRoiModel = self.repeatSegm(
            model_name=model_name, askSegmParams=True,
            is_label_roi=True
        )
        if self.labelRoiModel is None:
            self.initLabelRoiModelDialog = None
            return True
        self.labelRoiViewCurrentModelAction.setDisabled(False)
        self.initLabelRoiModelDialog = None
        return False

    def labelRoiViewCurrentModel(self):
        ini_path = self.view_model.model_params_ini_path(settings_folderpath)
        configPars = config.ConfigParser()
        configPars.read(ini_path)
        model_name = self.labelRoiModel.model_name
        txt = f'Model: <b>{model_name}</b>'
        SECTION = f'{model_name}.init'
        txt = f'{txt}<br><br>[Initialization parameters]<br>'
        for option in configPars.options(SECTION):
            value = configPars[SECTION][option]
            param_txt = f'<i>{option}</i> = {value}<br>'
            txt = f'{txt}{param_txt}'

        SECTION = f'{model_name}.segment'
        txt = f'{txt}<br>[Segmentation parameters]<br>'
        for option in configPars.options(SECTION):
            value = configPars[SECTION][option]
            param_txt = f'<i>{option}</i> = {value}<br>'
            txt = f'{txt}{param_txt}'

        win = apps.ViewTextDialog(txt, parent=self.host)
        win.exec_()

    def storeLabelRoiParams(self, value=None, checked=True):
        checkedRoiType = self.labelRoiTypesGroup.checkedButton().text()
        circRoiRadius = self.labelRoiCircularRadiusSpinbox.value()
        roiZdepth = self.labelRoiZdepthSpinbox.value()
        autoClearBorder = self.labelRoiAutoClearBorderCheckbox.isChecked()
        params = self.view_model.params_settings(
            checked_roi_type=checkedRoiType,
            circ_roi_radius=circRoiRadius,
            roi_zdepth=roiZdepth,
            auto_clear_border=autoClearBorder,
            replace_existing_objects=(
                self.labelRoiReplaceExistingObjectsCheckbox.isChecked()
            ),
        )
        for setting, setting_value in params.updates.items():
            self.df_settings.at[setting, 'value'] = setting_value
        self.df_settings.to_csv(self.settings_csv_path)

    def loadLabelRoiLastParams(self):
        idx = 'labelRoi_checkedRoiType'
        if idx in self.df_settings.index:
            checkedRoiType = self.df_settings.at[idx, 'value']
            for button in self.labelRoiTypesGroup.buttons():
                if button.text() == checkedRoiType:
                    button.setChecked(True)
                    break

        idx = 'labelRoi_circRoiRadius'
        if idx in self.df_settings.index:
            circRoiRadius = self.df_settings.at[idx, 'value']
            self.labelRoiCircularRadiusSpinbox.setValue(int(circRoiRadius))

        idx = 'labelRoi_roiZdepth'
        if idx in self.df_settings.index:
            roiZdepth = self.df_settings.at[idx, 'value']
            self.labelRoiZdepthSpinbox.setValue(int(roiZdepth))

        idx = 'labelRoi_autoClearBorder'
        if idx in self.df_settings.index:
            clearBorder = self.df_settings.at[idx, 'value']
            checked = self.view_model.checked_from_setting_value(clearBorder)
            self.labelRoiAutoClearBorderCheckbox.setChecked(checked)

        idx = 'labelRoi_replaceExistingObjects'
        if idx in self.df_settings.index:
            val = self.df_settings.at[idx, 'value']
            checked = self.view_model.checked_from_setting_value(val)
            self.labelRoiReplaceExistingObjectsCheckbox.setChecked(checked)

        if self.labelRoiIsCircularRadioButton.isChecked():
            self.labelRoiCircularRadiusSpinbox.setDisabled(False)

    def updateLabelRoiCircularSize(self, value):
        self.labelRoiCircItemLeft.setSize(value)
        self.labelRoiCircItemRight.setSize(value)

    def updateLabelRoiCircularCursor(self, x, y, checked):
        size = self.labelRoiCircularRadiusSpinbox.value()
        existing_cursor_empty = len(self.labelRoiCircItemLeft.getData()[0]) == 0
        if not self.view_model.should_show_circular_cursor(
            label_roi_checked=self.labelRoiButton.isChecked(),
            circular_roi_checked=self.labelRoiIsCircularRadioButton.isChecked(),
            label_roi_running=self.labelRoiRunning,
            cursor_checked=checked,
            existing_cursor_empty=existing_cursor_empty,
        ):
            return
        xx, yy = self.view_model.cursor_points(x, y, checked)

        self.labelRoiCircItemLeft.setData(xx, yy, size=size)
        self.labelRoiCircItemRight.setData(xx, yy, size=size)

    def getLabelRoiImage(self):
        posData = self.data[self.pos_i]

        start_frame_i = self.labelRoiStartFrameNoSpinbox.value()-1
        stop_frame_n = self.labelRoiStopFrameNoSpinbox.value()
        frame_range_enabled = self.labelRoiTrangeCheckbox.isChecked()
        tRangeLen = self.view_model.frame_range_length(
            frame_range_enabled,
            start_frame_i,
            stop_frame_n,
        )
        tRange = self.view_model.time_range(
            frame_range_enabled,
            start_frame_i,
            stop_frame_n,
        )

        if self.isSegm3D:
            if tRangeLen > 1:
                imgData = posData.img_data
            else:
                # Filtered data not existing
                imgData = posData.img_data[posData.frame_i]

            roi_zdepth = self.labelRoiZdepthSpinbox.value()
            z0, z1 = self.view_model.z_range(
                roi_zdepth,
                posData.SizeZ,
                self.zSliceScrollBar.sliderPosition(),
            )

            if self.labelRoiIsRectRadioButton.isChecked():
                labelRoiSlice = self.labelRoiItem.slice(
                    zRange=(z0,z1), tRange=tRange
                )
            elif self.labelRoiIsFreeHandRadioButton.isChecked():
                labelRoiSlice = self.freeRoiItem.slice(
                    zRange=(z0,z1), tRange=tRange
                )
            elif self.labelRoiIsCircularRadioButton.isChecked():
                labelRoiSlice = self.labelRoiCircItemLeft.slice(
                    zRange=(z0,z1), tRange=tRange
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

    def labelRoiTrangeCheckboxToggled(self, checked):
        enabled = self.view_model.should_enable_range_controls(checked)
        disabled = not enabled
        self.labelRoiStartFrameNoSpinbox.setDisabled(disabled)
        self.labelRoiStopFrameNoSpinbox.setDisabled(disabled)
        self.labelRoiStartFrameNoSpinbox.label.setDisabled(disabled)
        self.labelRoiStopFrameNoSpinbox.label.setDisabled(disabled)
        self.labelRoiToEndFramesAction.setDisabled(disabled)
        self.labelRoiFromCurrentFrameAction.setDisabled(disabled)

        if not enabled:
            return

        posData = self.data[self.pos_i]

        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i+1)
        self.labelRoiStopFrameNoSpinbox.setValue(posData.SizeT)

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
                self.logger.info('Collected garbage w5orker (magic labeller).')

            self.labelRoiToolbar.setVisible(True)
            if self.isSegm3D:
                self.labelRoiZdepthSpinbox.setDisabled(False)
            else:
                self.labelRoiZdepthSpinbox.setDisabled(True)

            # Start thread and pause it
            self.labelRoiThread = QThread()
            self.labelRoiMutex = QMutex()
            self.labelRoiWaitCond = QWaitCondition()

            labelRoiWorker = workers.LabelRoiWorker(self.host)

            labelRoiWorker.moveToThread(self.labelRoiThread)
            labelRoiWorker.finished.connect(self.labelRoiThread.quit)
            labelRoiWorker.finished.connect(labelRoiWorker.deleteLater)
            self.labelRoiThread.finished.connect(
                self.labelRoiThread.deleteLater
            )

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

            self.labelRoiItem.setPos((0,0))
            self.labelRoiItem.setSize((0,0))
            self.freeRoiItem.clear()
            self.ax1.removeItem(self.labelRoiItem)
            self.updateLabelRoiCircularCursor(None, None, False)

    def labelRoiWorkerFinished(self):
        self.logger.info('Magic labeller closed.')
        worker = self.labelRoiActiveWorkers.pop(-1)

    def indexRoiLab(self, roiLab, roiLabSlice, lab, brushID):
        result = self.host.view_model.label_edits.index_label_roi(
            lab,
            roiLab,
            roiLabSlice,
            brushID,
            clear_border=self.labelRoiAutoClearBorderCheckbox.isChecked(),
            replace_existing=(
                self.labelRoiReplaceExistingObjectsCheckbox.isChecked()
            ),
        )
        return result.labels

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
                lab = posData.allData_li[frame_i]['labels']
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
                lab = self.indexRoiLab(
                    roiLab, roiLabSlice, lab, posData.brushID
                )
                if store:
                    posData.frame_i = frame_i
                    posData.allData_li[frame_i]['labels'] = lab.copy()
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

        self.labelRoiItem.setPos((0,0))
        self.labelRoiItem.setSize((0,0))
        self.freeRoiItem.clear()
        self.logger.info('Magic labeller done!')
        self.app.restoreOverrideCursor()

        self.labelRoiRunning = False
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        uncheckLabelRoiTRange = self.view_model.should_uncheck_time_range(
            time_range_checked=self.labelRoiTrangeCheckbox.isChecked(),
            persistent_action_checked=(
                self.labelRoiTrangeCheckbox.findChild(QAction).isChecked()
            ),
        )
        if uncheckLabelRoiTRange:
            self.labelRoiTrangeCheckbox.setChecked(False)
