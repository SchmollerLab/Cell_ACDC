"""Saving from the main GUI: save data and its prompts, autosave and Set
measurements. Mixins of guiWin (gui/main_window/window.py) only: the
measurements viewer never composes them.
"""
import os
from datetime import datetime
from functools import partial
from tqdm import tqdm
from typing import Literal
import pandas as pd
from qtpy.QtCore import QEventLoop, QTimer, QThread, QMutex, QWaitCondition
from qtpy.QtWidgets import QCheckBox, QMessageBox
from cellacdc import exception_handler, disableWindow
from cellacdc import load, html_utils
from cellacdc.shared.dialogs import (
    combine_measurements as dialogs_combine_measurements,
    file_selection as dialogs_file_selection,
    generic as dialogs_generic,
    set_measurements as dialogs_set_measurements,
)
from cellacdc.shared.workers import saving as workers_saving
from ..dialogs import data_io as gui_dialogs_data_io
from ..workers import loading_saving as gui_workers_loading_saving
from cellacdc.myutils import misc
from cellacdc.shared.widgets import (
    item_views as widgets_item_views,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from cellacdc import measurements
from cellacdc import settings_csv_path
from cellacdc import favourite_func_metrics_csv_path
from cellacdc import cli
from cellacdc.cca_functions import _calc_rot_vol
from . import constants as gui_main_constants


class AutoSaveGuiElements:
    """Autosave worker and its settings."""

    def gui_createAutoSaveWorker(self):        
        if not hasattr(self, 'data'):
            return
        
        if not self.isDataLoaded:
            return 
        
        if self.autoSaveActiveWorkers:
            garbage = self.autoSaveActiveWorkers[-1]
            self.autoSaveGarbageWorkers.append(garbage)
            worker = garbage[0]
            worker._stop()

        posData = self.data[self.pos_i]
        autoSaveThread = QThread()
        self.autoSaveMutex = QMutex()
        self.autoSaveWaitCond = QWaitCondition()

        savedSegmData = posData.segm_data.copy()
        autoSaveWorker = gui_workers_loading_saving.AutoSaveWorker(
            self.autoSaveMutex, self.autoSaveWaitCond, savedSegmData
        )
        autoSaveWorker.isAutoSaveON = self.autoSaveToggle.isChecked()

        autoSaveWorker.moveToThread(autoSaveThread)
        autoSaveWorker.finished.connect(autoSaveThread.quit)
        autoSaveWorker.finished.connect(autoSaveWorker.deleteLater)
        autoSaveThread.finished.connect(autoSaveThread.deleteLater)

        autoSaveWorker.sigDone.connect(self.autoSaveWorkerDone)
        autoSaveWorker.progress.connect(self.workerProgress)
        autoSaveWorker.finished.connect(self.autoSaveWorkerClosed)
        autoSaveWorker.sigAutoSaveCannotProceed.connect(
            self.turnOffAutoSaveWorker
        )
        
        autoSaveThread.started.connect(autoSaveWorker.run)
        autoSaveThread.start()

        self.autoSaveActiveWorkers.append((autoSaveWorker, autoSaveThread))

        self.logger.info('Autosaving worker started.')

    def autoSaveWorkerStartTimer(self, worker, posData):
        self.autoSaveWorkerTimer = QTimer()
        self.autoSaveWorkerTimer.timeout.connect(
            partial(self.autoSaveWorkerTimerCallback, worker, posData)
        )
        self.autoSaveWorkerTimer.start(150)

    def autoSaveWorkerTimerCallback(self, worker, posData):
        if not self.isSaving:
            self.autoSaveWorkerTimer.stop()
            worker._enqueue(posData)

    def autoSaveWorkerDone(self):
        self.logger.info('Autosaving done.')
        self.setStatusBarLabel(log=False)

    def autoSaveWorkerClosed(self, worker):
        if self.autoSaveActiveWorkers:
            self.logger.info('Autosaving worker closed.')
            try:
                self.autoSaveActiveWorkers.remove(worker)
            except Exception as e:
                pass

    def turnOffAutoSaveWorker(self):
        self.autoSaveToggle.setChecked(False)

    def autoSaveTimerTimedOut(self):
        if not hasattr(self, 'data'):
            # This happes when the self.autoSaveTimer times out after 
            # the GUI has been closed -->  we simply ignore it
            self.autoSaveTimer.stop()
            return
        
        self.autoSaveTimer.stop()
        self.flushDirtyPointsLayersAutosave()
        self._enqueueAutoSave()

    def autoSaveTimerCountFrames(self):
        if not hasattr(self, 'data'):
            # This happes when the self.autoSaveTimer times out after 
            # the GUI has been closed -->  we simply ignore it
            return
        
        posData = self.data[self.pos_i]
        autoSaveIntevalValue, autoSaveIntervalUnit = (
            self.autoSaveIntevalValueUnit
        )
        isTimeToAutoSave = (
            abs(posData.frame_i - self.autoSaveTimeStartFrameIdx)
            >= autoSaveIntevalValue
        )
        if not isTimeToAutoSave:
            return
        
        self.autoSaveTimeStartFrameIdx = posData.frame_i
        self.flushDirtyPointsLayersAutosave()
        self._enqueueAutoSave()

    def enqAutosave(self):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            if self.statusBarLabel.text().endswith('Autosaving...'):
                self.statusBarLabel.setText(
                    self.statusBarLabel.text().replace(' | Autosaving...', '')
                )
            return 
        
        if not self.autoSaveActiveWorkers:
            self.gui_createAutoSaveWorker()
        
        if not self.autoSaveActiveWorkers:
            return
        
        if self.autoSaveTimer.isActive():
            return
        
        self._enqueueAutoSave()
        autoSaveIntevalValue, autoSaveIntervalUnit = (
            self.autoSaveIntevalValueUnit
        )
        if autoSaveIntevalValue == 0:
            return
        
        try:
            self.autoSaveTimer.timeout.disconnect()
        except Exception as err:
            pass
            
        
        if autoSaveIntervalUnit == 'minutes':
            autosave_interval_ms = round(autoSaveIntevalValue*60*1000)
            self.autoSaveTimer.timeout.connect(self.autoSaveTimerTimedOut)
            self.autoSaveTimer.start(autosave_interval_ms)
        else:
            self.startAutoSaveEveryNframesTimer()

    def startAutoSaveEveryNframesTimer(self):
        posData = self.data[self.pos_i]
        self.autoSaveTimeStartFrameIdx = posData.frame_i
        self.autoSaveTimer.timeout.connect(
            self.autoSaveTimerCountFrames
        )
        self.autoSaveTimer.start(500)

    def _enqueueAutoSave(self):
        if not self.statusBarLabel.text().endswith('Autosaving...'):
            self.statusBarLabel.setText(
                f'{self.statusBarLabel.text()} | Autosaving...'
            )
            
        timestamp = datetime.now().strftime(r'%H:%M:%S.%f')[:-3]
        self.logger.info(f'Autosaving... - {timestamp}')
        
        posData = self.data[self.pos_i]
        worker, thread = self.autoSaveActiveWorkers[-1]
        worker.enqueue(posData)

    def waitAutoSaveWorker(self, worker):
        if worker.isFinished or worker.isPaused or len(worker.dataQ) == 0:
            self.waitAutoSaveWorkerLoop.exit()
            self.waitAutoSaveWorkerTimer.stop()
            self.setStatusBarLabel(log=False)

    def autoSaveClose(self):
        for worker, thread in self.autoSaveActiveWorkers:
            worker._stop()

    def setAutoSaveSegmentationEnabled(self, enabled):
        if not self.autoSaveActiveWorkers:
            return
        
        worker, thread = self.autoSaveActiveWorkers[-1]
        
        if enabled:
            worker.isAutoSaveON = self.autoSaveToggle.isChecked()
        else:
            worker.isAutoSaveON = False

    def setAutoSaveAnnotationsEnabled(self, enabled):
        if not self.autoSaveActiveWorkers:
            return
        
        worker, thread = self.autoSaveActiveWorkers[-1]
        
        if enabled:
            worker.isAutoSaveAnnotON = self.autoSaveToggle.isChecked()
        else:
            worker.isAutoSaveAnnotON = False

    def autoSaveToggled(self, checked):
        if not self.autoSaveActiveWorkers:
            self.gui_createAutoSaveWorker()
        
        if not self.autoSaveActiveWorkers:
            return
        
        worker, thread = self.autoSaveActiveWorkers[-1]
        
        mode = self.modeComboBox.currentText()
        if mode != 'Segmentation and Tracking':
            # Autosaving segmentation makes sense only in 
            # "Segmentation and Tracking" mode
            checked = False
        
        worker.isAutoSaveON = checked

    def autoSaveAnnotToggled(self, checked):
        if not self.autoSaveActiveWorkers:
            self.gui_createAutoSaveWorker()
        
        if not self.autoSaveActiveWorkers:
            return
        
        worker, thread = self.autoSaveActiveWorkers[-1]
        
        mode = self.modeComboBox.currentText()
        if mode != 'Viewer':
            # No reason to save in viewer mode
            checked = False
        
        worker.isAutoSaveAnnotON = checked

    def autoSaveIntervalEdit(self):
        self.autoSaveIntervalDialog.show()
        self.autoSaveIntervalDialog.raise_()
        self.autoSaveIntervalDialog.activateWindow()

    def autoSaveIntervalValueChanged(
            self, value: float, unit: Literal['minutes', 'frames']
        ):
        self.autoSaveIntevalValueUnit = (value, unit)
        self.autoSaveTimer.stop()
        
        self.df_settings.at['autoSaveIntevalValue', 'value'] = str(value)
        self.df_settings.at['autoSaveIntervalUnit', 'value'] = unit
        self.df_settings.to_csv(settings_csv_path)
        
        self.logger.info(
            f'Autosave interval changed to: {value} {unit}'
        )
        self.autoSaveIntervalSetTooltip()
        
        if unit == 'frames':
            self.startAutoSaveEveryNframesTimer()

    def autoSaveIntervalSetTooltip(self):
        value, unit = self.autoSaveIntevalValueUnit
        autoSaveIntervalEditTooltip = (
            'Change autosave interval to every N frames or minutes\n\n'
            f'Current autosave interval: {value} {unit}'
        )
        self.autoSaveIntervalLabel.setToolTip(autoSaveIntervalEditTooltip)
        self.autoSaveIntervalEditButton.setToolTip(autoSaveIntervalEditTooltip)

    def updateSegmDataAutoSaveWorker(self):
        # Update savedSegmData in autosave worker
        posData = self.data[self.pos_i]
        for worker, thread in self.autoSaveActiveWorkers:
            worker.savedSegmData = posData.segm_data.copy()

    def _waitCloseAutoSaveWorker(self):
        didWorkersFinished = [True]
        for worker, thread in self.autoSaveActiveWorkers:
            if worker.isFinished:
                didWorkersFinished.append(True)
            else:
                didWorkersFinished.append(False)
        if all(didWorkersFinished):
            self.waitCloseAutoSaveWorkerLoop.stop()


class SaveDataGuiElements:
    """Save data and save as, the save prompts and the callbacks of the
    save worker.
    """

    def saveDataWorkerCritical(self, error):
        self.logger.warning(
            'Saving process stopped because of critical error.'
        )
        self.saveWin.aborted = True
        self._saveDataWorker.finished.emit()
        self.workerCritical(error)

    def askSaveOriginalSegm(self, isQuickSave=False):
        if isQuickSave:
            return "", True, True

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return "", True, True
        
        help_txt = html_utils.paragraph(f"""
            You have <b>whitelisted IDs</b> in the current position.<br>
            Do you want to save the <b>not whitelisted</b> segmentation data<br>
            This will allow you to <b>revisit the original segmentation</b>.<br>
            """)

        txt = html_utils.paragraph(f"""
            You have <b>whitelisted IDs</b> in the current position.<br>
            Do you want to save the <b>not whitelisted</b> segmentation data?<br>
            """)

        found_files = load.get_segm_files(posData.images_path)
        existingEndnames = load.get_endnames(
            posData.basename, found_files
        )

        segmFilename = os.path.basename(posData.segm_npz_path)
        segmFilename = f"{segmFilename[:-4]}_not_whitelisted"
        win = dialogs_file_selection.filenameDialog(
            basename=posData.basename,
            hintText=txt,
            defaultEntry=segmFilename,
            existingNames=existingEndnames,
            helpText=help_txt, 
            allowEmpty=False,
            parent=self,
            title='Save not whitelisted segmentation data',
            addDoNotSaveButton=True
        )
        win.exec_()
        if win.cancel:
            return "", False, True
        if win.doNotSave:
            return "", True, True
        return win.entryText, True, False

    def askSaveLastVisitedCcaMode(self, isQuickSave=False):
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        frame_i = 0
        last_tracked_i = 0
        self.save_until_frame_i = 0
        if self.isSnapshot:
            return True
        
        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break
        
        self.save_until_frame_i = frame_i
        self.save_cca_until_frame_i = frame_i
        self.last_tracked_i = frame_i
        
        if isQuickSave:
            return True
        
        last_cca_frame_i = self.navigateScrollBar.maximum()-1
        last_moth_bud_pair_frame_i = (
            self.annotateSingleMothBudPairState.get('last_annot_frame_i', -1)
        )

        if last_moth_bud_pair_frame_i > last_cca_frame_i:
            last_cca_frame_i = last_moth_bud_pair_frame_i
            
        # Ask to save last visited frame or not
        txt = html_utils.paragraph(f"""
            You annotated the cell cycle stages up 
            until frame number {last_cca_frame_i+1}.<br><br>
            Enter <b>up to which frame number</b> you want to save the 
            cell cycle annotations:
        """)
        lastFrameDialog = dialogs_generic.QLineEditDialog(
            title='Last annoated frame number to save', 
            defaultTxt=str(last_cca_frame_i+1),
            msg=txt, parent=self, allowedValues=(1, last_cca_frame_i+1),
            warnLastFrame=True, isInteger=True, stretchEntry=False,
            lastVisitedFrame=last_cca_frame_i+1,
        )
        lastFrameDialog.exec_()
        if lastFrameDialog.cancel:
            return False

        last_save_cca_frame_i = lastFrameDialog.enteredValue - 1
        
        if last_save_cca_frame_i < last_cca_frame_i:
            self.resetCcaFuture(last_cca_frame_i)
        
        self.save_cca_until_frame_i = last_save_cca_frame_i
        
        return True

    def askSaveLastVisitedSegmMode(self, isQuickSave=False):
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        frame_i = 0
        last_tracked_i = 0
        self.save_until_frame_i = 0
        self.save_cca_until_frame_i = 0
        if self.isSnapshot:
            return True

        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break

        if isQuickSave:
            self.save_until_frame_i = frame_i
            self.save_cca_until_frame_i = frame_i
            self.last_tracked_i = frame_i
            return True

        # Ask to save last visited frame or not
        txt = html_utils.paragraph(f"""
            You visualised and corrected segmentation and tracking data up 
            until frame number {frame_i+1}.<br><br>
            Enter <b>up to which frame number</b> you want to save data:
        """)
        lastFrameDialog = dialogs_generic.QLineEditDialog(
            title='Last frame number to save', defaultTxt=str(frame_i+1),
            msg=txt, parent=self, allowedValues=(1, posData.SizeT),
            warnLastFrame=True, isInteger=True, stretchEntry=False,
            lastVisitedFrame=frame_i+1,
        )
        lastFrameDialog.exec_()
        if lastFrameDialog.cancel:
            return False

        self.save_until_frame_i = lastFrameDialog.enteredValue - 1
        self.save_cca_until_frame_i = self.save_until_frame_i
        if self.save_until_frame_i > frame_i:
            self.logger.info(
                f'Storing frames {frame_i+1}-{self.save_until_frame_i+1}...'
            )
            current_frame_i = posData.frame_i
            # User is requesting to save past the last visited frame -->
            # store data as if they were visited
            for i in range(frame_i+1, self.save_until_frame_i+1):
                posData.frame_i = i
                self.get_data()
                self.store_data(autosave=False)
            
            # Go back to current frame
            posData.frame_i = current_frame_i
            self.get_data()
        last_tracked_i = self.save_until_frame_i
        
        self.last_tracked_i = last_tracked_i
        return True

    def askSaveMetrics(self):
        txt = html_utils.paragraph(
        """
            Do you also want to <b>save the measurements</b> 
            (e.g., cell volume, mean, amount etc.)?<br><br>
            
            You can find <b>more information</b> by clicking on the 
            "Set measurements" button below <br>
            where you will be able to select which <b>measurements 
            you want to save</b>.<br><br>
            If you already set the measurements and you want to save them click "Yes".<br><br>
            
            NOTE: Saving metrics might be <b>slow</b>,
            we recommend doing it <b>only when you need it</b>.<br>
        """)
        msg = widgets_message_box.myMessageBox(
            parent=self, resizeButtons=False, wrapText=False
        )
        setMeasurementsButton = widgets_push_buttons.setPushButton('Set measurements...')
        _, yesButton, noButton, _ = msg.question(
            self, 'Save measurements?', txt, 
            buttonsTexts=('Cancel', 'Yes', 'No', setMeasurementsButton),
            showDialog=False
        )
        setMeasurementsButton.disconnect()
        setMeasurementsButton.clicked.connect(
            partial(
                self.showSetMeasurements, 
                qparent=msg,
            )
        )
        msg.exec_()
        save_metrics = msg.clickedButton == yesButton
        return save_metrics, msg.cancel

    def askSelectPos(self, action='to save'):
        last_pos = 1
        for p, posData in enumerate(self.data):
            acdc_df = posData.allData_li[0]['acdc_df']
            if acdc_df is None:
                last_pos = p
                break
        else:
            last_pos = len(self.data)

        items = [posData.pos_foldername for posData in self.data]
        selectPosWin = widgets_item_views.QDialogListbox(
            f'Select Positions {action}', f'Select Positions {action}:\n',
            items, multiSelection=True, parent=self,
            preSelectedItems=items[:last_pos]
        )
        selectPosWin.exec_()
        if selectPosWin.cancel:
            return
        
        return selectPosWin.selectedItemsText

    def askPosToSave(self):
        return self.askSelectPos()

    def saveMetricsCritical(self, traceback_format):
        print('\n====================================')
        self.logger.exception(traceback_format)
        print('====================================\n')
        self.logger.info('Warning: calculating metrics failed see above...')
        print('------------------------------')

        msg = widgets_message_box.myMessageBox(wrapText=False)
        err_msg = html_utils.paragraph(f"""
            Error <b>while saving metrics</b>.<br><br>
            More details below or in the terminal/console.<br><br>
            Note that the error details from this session are also saved
            in the file<br>
            {self.log_path}<br><br>
            Please <b>send the log file</b> when reporting a bug, thanks!
            <b>Please restart Cell-ACDC, we apologise for any inconvenience.</b><br><br>

        """)
        msg.addShowInFileManagerButton(self.logs_path, txt='Show log file...')
        msg.setDetailedText(traceback_format, visible=True)
        msg.critical(self, 'Critical error while saving metrics', err_msg)

        self.is_error_state = True
        self.waitCond.wakeAll()

    def saveAsData(self, checked=True):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        existingFilenames = set()
        for _posData in self.data:
            segm_files = load.get_segm_files(_posData.images_path)
            _existingEndnames = load.get_endnames(
                _posData.basename, segm_files
            )
            existingFilenames.update([
                f'{_posData.basename}{endname}.npz' 
                for endname in _existingEndnames
            ])
        posData = self.data[self.pos_i]
        if posData.basename.endswith('_'):
            basename = f'{posData.basename}segm'
        else:
            basename = f'{posData.basename}_segm'
        win = dialogs_file_selection.filenameDialog(
            basename=basename,
            hintText='Insert a <b>filename</b> for the segmentation file:<br>',
            existingNames=existingFilenames
        )
        win.exec_()
        if win.cancel:
            return

        for posData in self.data:
            posData.setFilePaths(new_endname=win.entryText)

        self.setStatusBarLabel()
        self.saveData()

    def saveDataPermissionError(self, err_msg):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        msg = QMessageBox()
        msg.critical(self, 'Permission denied', err_msg, msg.Ok)
        self.waitCond.wakeAll()

    def saveDataProgress(self, text):
        self.logger.info(text)
        self.saveWin.progressLabel.setText(text)

    def saveDataCustomMetricsCritical(self, traceback_format, func_name):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        self.logger.info('')
        _hl = '===================================='
        self.logger.info(f'{_hl}\n{traceback_format}\n{_hl}')
        self._saveDataWorker.customMetricsErrors[func_name] = traceback_format

    def saveDataCombinedMetricsMissingColumn(self, error_msg, func_name):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        self.logger.info('')
        warning = f'[WARNING]: {error_msg}. Metric {func_name} was skipped.'
        _hl = '===================================='
        self.logger.info(f'{_hl}\n{warning}\n{_hl}')
        self._saveDataWorker.customMetricsErrors[func_name] = warning

    def saveDataAddMetricsCritical(self, traceback_format, error_message):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        self.logger.info('')
        _hl = '===================================='
        self.logger.info(f'{_hl}\n{traceback_format}\n{_hl}')
        self._saveDataWorker.addMetricsErrors[error_message] = traceback_format

    def saveDataRegionPropsCritical(self, traceback_format, error_message):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        self.logger.info('')
        _hl = '===================================='
        self.logger.info(f'{_hl}\n{traceback_format}\n{_hl}')
        self._saveDataWorker.regionPropsErrors[error_message] = traceback_format

    def saveDataUpdateMetricsPbar(self, max, step):
        if max > 0:
            self.saveWin.metricsQPbar.setMaximum(max)
            self.saveWin.metricsQPbar.setValue(0)
        self.saveWin.metricsQPbar.setValue(
            self.saveWin.metricsQPbar.value()+step
        )

    def saveDataUpdatePbar(self, step, max=-1, exec_time=0.0):
        if max >= 0:
            self.saveWin.QPbar.setMaximum(max)
        else:
            self.saveWin.QPbar.setValue(self.saveWin.QPbar.value()+step)
            steps_left = self.saveWin.QPbar.maximum()-self.saveWin.QPbar.value()
            seconds = round(exec_time*steps_left)
            ETA = misc.seconds_to_ETA(seconds)
            self.saveWin.ETA_label.setText(f'ETA: {ETA}')

    def quickSave(self):
        self.saveData(isQuickSave=True)

    def warnDifferentSegmChannel(
            self, loaded_channel, segm_channel_hyperparams, segmEndName
        ):
        txt = html_utils.paragraph(f"""
            You loaded the segmentation file ending with <code>_{segmEndName}.npz</code> 
            which corresponds to the channel 
            <code>{segm_channel_hyperparams}</code>.<br><br>
            However, <b>in this session you loaded the channel</b> 
            <code>{loaded_channel}</code>.<br><br>
            If you proceed with saving, the segmentation file ending with 
            <code>_{segmEndName}.npz</code> <b>will be OVERWRITTEN</b>.<br><br>
            Are you sure you want to proceed?
        """)
        msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
        msg.warning(
            self, 'WARNING: Potential for data loss', txt,
            buttonsTexts=('Cancel', 'Yes')
        )
        return msg.cancel

    @exception_handler
    def saveData(self, checked=False, finishedCallback=None, isQuickSave=False):
        self.setDisabled(True, keepDisabled=True)

        self.askLineageTreeChanges()

        self.store_data(autosave=False)
        self.store_data()
        self._isQuickSave = isQuickSave

        # Wait autosave worker to finish
        for worker, thread in self.autoSaveActiveWorkers:
            self.logger.info('Stopping autosaving process...')
            self.statusBarLabel.setText('Stopping autosaving process...')
            worker.stop()
            self.waitAutoSaveWorkerTimer = QTimer()
            self.waitAutoSaveWorkerTimer.timeout.connect(
                partial(self.waitAutoSaveWorker, worker)
            )
            self.waitAutoSaveWorkerTimer.start(100)
            self.waitAutoSaveWorkerLoop = QEventLoop()
            self.waitAutoSaveWorkerLoop.exec_()

        self.titleLabel.setText(
            'Saving data... (check progress in the terminal)', 
            color=self.titleColor
        )

        # Check channel name correspondence to warn
        posData = self.data[self.pos_i]
        lastSegmChannel, segmEndName = posData.getSegmentedChannelHyperparams()
        if lastSegmChannel != self.user_ch_name and lastSegmChannel:
            cancel = self.warnDifferentSegmChannel(
                self.user_ch_name, lastSegmChannel, segmEndName
            )
            if cancel:
                self.cancelSavingInitialisation()
                self.setDisabled(False, keepDisabled=False)
                self.activateWindow()
                return True
            posData.updateSegmentedChannelHyperparams(self.user_ch_name)

        # Check missing cca annotations in snaphots
        proceed, ignore, self.doNotShowAgainMissingCca = self.checkMissingCca()
        if not proceed and not ignore:
            self.cancelSavingInitialisation()
            self.setDisabled(False, keepDisabled=False)
            self.activateWindow()
            return 
        
        self.save_metrics = False
        if not isQuickSave:
            self.save_metrics, cancel = self.askSaveMetrics()
            if cancel:
                self.cancelSavingInitialisation()
                self.setDisabled(False, keepDisabled=False)
                self.activateWindow()
                return True

        self.posToSave = None
        if self.isSnapshot and not isQuickSave and len(self.data) > 1:
            self.posToSave = self.askPosToSave()
            if self.posToSave is None:
                self.cancelSavingInitialisation()
                self.setDisabled(False, keepDisabled=False)
                self.activateWindow()
                return True

        if isQuickSave:
            # Quick save only current pos
            self.posToSave = {self.data[self.pos_i].pos_foldername}
        
        if self.isSnapshot:
            self.store_data(mainThread=False)

        saveSMBPCcaInfo = False
        mode = self.modeComboBox.currentText()
        if mode == 'Cell cycle analysis':
            proceed, saveSMBPCcaInfo = self.detectSingleMotherBudPairsCcaDf(
                isQuickSave=isQuickSave
            )
            if not proceed:
                self.cancelSavingInitialisation()
                self.setDisabled(False, keepDisabled=False)
                self.activateWindow()
                return True

            proceed = self.askSaveLastVisitedCcaMode(isQuickSave=isQuickSave)
            if not proceed:
                self.cancelSavingInitialisation()
                self.setDisabled(False, keepDisabled=False)
                self.activateWindow()
                return True
        else:
            proceed = self.askSaveLastVisitedSegmMode(isQuickSave=isQuickSave)
            if not proceed:
                self.cancelSavingInitialisation()
                self.setDisabled(False, keepDisabled=False)
                self.activateWindow()
                return True
        
        append_name_og_whitelist, proceed, do_not_save_og_whitelist = (
            self.askSaveOriginalSegm(isQuickSave=isQuickSave)
        )
        if not proceed:
            self.cancelSavingInitialisation()
            self.setDisabled(False, keepDisabled=False)
            self.activateWindow()
            return True

        if self.save_metrics or mode == 'Cell cycle analysis':
            self.computeVolumeRegionprop()

        infoTxt = html_utils.paragraph(
            f'Saving {self.exp_path}...<br>', font_size='14px'
        )

        self.saveWin = gui_dialogs_data_io.QDialogPbar(
            parent=self, title='Saving data', infoTxt=infoTxt
        )
        self.saveWin.setFont(gui_main_constants._font)
        # if not self.save_metrics:
        self.saveWin.metricsQPbar.hide()
        self.saveWin.progressLabel.setText('Preparing data...')
        self.saveWin.show()

        # Set up separate thread for saving and show progress bar widget
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self._saveDataThread = QThread()
        self._saveDataWorker = workers_saving.saveDataWorker(self)
        self._saveDataWorker.mode = mode
        self._saveDataWorker.isQuickSave = isQuickSave
        self._saveDataWorker.append_name_og_whitelist = append_name_og_whitelist
        self._saveDataWorker.do_not_save_og_whitelist = do_not_save_og_whitelist
        self._saveDataWorker.saveSMBPCcaInfo = saveSMBPCcaInfo

        self._saveDataWorker.moveToThread(self._saveDataThread)

        self._saveDataWorker.finished.connect(self._saveDataThread.quit)
        self._saveDataWorker.finished.connect(self._saveDataWorker.deleteLater)
        self._saveDataThread.finished.connect(self._saveDataThread.deleteLater)

        # Custom signals
        self._saveDataWorker.finished.connect(self.saveDataFinished)
        if finishedCallback is not None:
            self._saveDataWorker.finished.connect(finishedCallback)
        self._saveDataWorker.progress.connect(self.saveDataProgress)
        self._saveDataWorker.sigLog.connect(self.workerLog)
        self._saveDataWorker.progressBar.connect(self.saveDataUpdatePbar)
        # self.worker.metricsPbarProgress.connect(self.saveDataUpdateMetricsPbar)
        self._saveDataWorker.critical.connect(self.saveDataWorkerCritical)
        self._saveDataWorker.customMetricsCritical.connect(
            self.saveDataCustomMetricsCritical
        )
        self._saveDataWorker.sigCombinedMetricsMissingColumn.connect(
            self.saveDataCombinedMetricsMissingColumn
        )
        self._saveDataWorker.addMetricsCritical.connect(
            self.saveDataAddMetricsCritical)
        self._saveDataWorker.regionPropsCritical.connect(
            self.saveDataRegionPropsCritical
        )
        self._saveDataWorker.criticalPermissionError.connect(
            self.saveDataPermissionError)
        self._saveDataWorker.askZsliceAbsent.connect(self.zSliceAbsent)
        self._saveDataWorker.sigDebug.connect(self._workerDebug)

        self._saveDataThread.started.connect(self._saveDataWorker.run)

        self._saveDataThread.start()
        
        return False

    def _workerDebug(self, stuff_to_debug):
        pass
        # from acdctools.plot import imshow
        # lab, frame_i, autoBkgr_masks = stuff_to_debug
        # autoBkgr_mask, autoBkgr_mask_proj = autoBkgr_masks
        # imshow(lab, autoBkgr_mask)
        # self.worker.waitCond.wakeAll()

    def warnErrorsCustomMetrics(self):
        win = dialogs_set_measurements.ComputeMetricsErrorsDialog(
            self._saveDataWorker.customMetricsErrors, self.logs_path, 
            log_type='custom_metrics', parent=self
        )
        win.exec_()

    def warnErrorsAddMetrics(self):
        win = dialogs_set_measurements.ComputeMetricsErrorsDialog(
            self._saveDataWorker.addMetricsErrors, self.logs_path, 
            log_type='standard_metrics', parent=self
        )
        win.exec_()

    def warnErrorsRegionProps(self):
        win = dialogs_set_measurements.ComputeMetricsErrorsDialog(
            self._saveDataWorker.regionPropsErrors, self.logs_path, 
            log_type='region_props', parent=self
        )
        win.exec_()

    def askConcatenate(self):
        if self.mainWin is None:
            return
        
        if self._isQuickSave:
            return
        
        if 'showAskConcatenate' not in self.df_settings.index:
            self.df_settings.at['showAskConcatenate', 'value'] = 'Yes'
        
        showAskConcatenate = (
            self.df_settings.at['showAskConcatenate', 'value'] == 'Yes'
        )
        if not showAskConcatenate:
            return
        
        txt = html_utils.paragraph(f"""
            Do you want to <b>concatenate</b> the `acdc_output.csv` tables from 
            multiple Positions into <b>one single CSV file</b>?<br>
        """)
        doNotShowAgainCheckbox = QCheckBox('Do not show again')
        msg = widgets_message_box.myMessageBox(wrapText=False)
        noButton, yesButton = msg.question(
            self, 'Concatenate tables?', txt, 
            buttonsTexts=('No', 'Yes'),
            widgets=doNotShowAgainCheckbox
        )
        showAskConcatenate = (
            'No' if doNotShowAgainCheckbox.isChecked() else 'Yes'
        )
        self.df_settings.at['showAskConcatenate', 'value'] = (
            showAskConcatenate
        )
        self.df_settings.to_csv(settings_csv_path)
        
        if not msg.clickedButton == yesButton:
            return
        
        txt = html_utils.paragraph(f"""
            To <b>concatenate</b> the `acdc_output.csv` tables from 
            multiple Positions and multiple experiments<br>
            launch the concatenation utility from the top menubar of the Cell-ACDC main launcher:<br><br>
            <code>Utilities --> Concatenate --> Concatenate acdc output tables from multiple Positions and experiments...</code>.
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'How to concatenate tables', txt)

    def saveDataFinished(self):
        self.removeSMBPCcaInfoAfterSaving()
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        if self.saveWin.aborted or self._saveDataWorker.abort:
            self.titleLabel.setText('Saving process cancelled.', color='r')
        elif self._isQuickSave:
            self.titleLabel.setText('Saved segmentation file and annotations')
        else:
            self.titleLabel.setText('Saved!')
        self.saveWin.workerFinished = True
        self.saveWin.close()

        if not self.closeGUI:
            # Update savedSegmData in autosave worker
            self.updateSegmDataAutoSaveWorker()

        if self._saveDataWorker.addMetricsErrors:
           self.warnErrorsAddMetrics()    
        if self._saveDataWorker.regionPropsErrors:
            self.warnErrorsRegionProps()
        if self._saveDataWorker.customMetricsErrors:
            self.warnErrorsCustomMetrics()
        
        self.checkManageVersions()
        
        self.askConcatenate()
        
        if self.closeGUI:
            salute_string = misc.get_salute_string()
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph(
                'Data <b>saved!</b>. The GUI will now close.<br><br>'
                f'{salute_string}'
            )
            msg.information(self, 'Data saved', txt)
            self.close()

    def cancelSavingInitialisation(self):
        self.titleLabel.setText(
            'Saving data process cancelled.', color=self.titleColor
        )
        self.closeGUI = False

    @disableWindow
    def askSaveOnClosing(self, event):
        if not self.saveAction.isEnabled():
            return True
        if self.titleLabel.text == 'Saved!':
            return True
        if not self.isDataLoaded:
            return True
        
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph('Do you want to <b>save before closing?</b>')
        _, noButton, yesButton = msg.question(
            self, 'Save?', txt,
            buttonsTexts=('Cancel', 'No', 'Yes')
        )
        if msg.cancel:
            event.ignore()
            return False
        
        if msg.clickedButton == yesButton:
            self.closeGUI = True
            QTimer.singleShot(100, self.saveAction.trigger)
            event.ignore()
            return False
        return True


class SetMeasurementsGuiElements:
    """Set measurements flow of the main GUI: metrics to save, custom and
    combined metrics.
    """

    def initMetricsToSave(self, posData):
        self._measurements_kernel._init_metrics_to_save(posData)

    def initMetrics(self):
        self.logger.info('Initializing measurements...')
        posData = self.data[self.pos_i]
        self._measurements_kernel = cli.ComputeMeasurementsKernel(
            self.logger, self.log_path, False
        )
        self._measurements_kernel.init_args(
            self.getAllPosChNames(), posData.getSegmEndname()
        )
        self._measurements_kernel._init_metrics(posData, self.isSegm3D)

    def showSetMeasurements(self, checked=False, qparent=None):
        qparent = qparent if qparent is not None else self
        if self.measurementsWin is not None:
            self.measurementsWin.show()
            self.measurementsWin.raise_()
            self.measurementsWin.activateWindow()
            return

        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            favourite_funcs = df_favourite_funcs['favourite_func_name'].to_list()
        except Exception as e:
            favourite_funcs = None

        posData = self.data[self.pos_i]
        allPos_acdc_df_cols = set()
        # Columns of each Position (classified with its combined measurements)
        allPos_acdc_df_cols_per_pos = []
        for _posData in self.data:
            pos_acdc_df_cols = set()
            for frame_i, data_dict in enumerate(_posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue

                pos_acdc_df_cols.update(acdc_df.columns)
            allPos_acdc_df_cols.update(pos_acdc_df_cols)
            allPos_acdc_df_cols_per_pos.append(sorted(pos_acdc_df_cols))
        loadedChNames = posData.setLoadedChannelNames(returnList=True)
        posData.fluo_data_dict.pop(self.user_ch_name, None)
        if self.user_ch_name not in loadedChNames:
            loadedChNames.insert(0, self.user_ch_name)
        notLoadedChNames = [
            c for c in self.getAllPosChNames() if c not in loadedChNames
        ]
        self.notLoadedChNames = notLoadedChNames
        self.measurementsWin = dialogs_set_measurements.SetMeasurementsDialog(
            loadedChNames, notLoadedChNames, posData.SizeZ > 1, self.isSegm3D,
            favourite_funcs=favourite_funcs, 
            allPos_acdc_df_cols=list(allPos_acdc_df_cols),
            allPos_acdc_df_cols_per_pos=allPos_acdc_df_cols_per_pos,
            acdc_df_path=posData.images_path, posData=posData,
            addCombineMetricCallback=self.addCombineMetric,
            allPosData=self.data, 
            parent=qparent, 
            state=self.setMeasWinState
        )
        self.measurementsWin.sigCancel.connect(self.setMeasurementsCancelled)
        self.measurementsWin.sigClosed.connect(self.setMeasurements)
        self.measurementsWin.show()

    def setMeasurementsCancelled(self):
        self.measurementsWin = None

    def setMeasurements(self):
        posData = self.data[self.pos_i]
        if self.measurementsWin.delExistingCols:
            self.logger.info('Removing existing unchecked measurements...')
            # Exact column names (also the `<prop>-<i>` regionprops columns,
            # see `measurements.unselected_measurement_columns`). The
            # measurements kernel drops them again when saving
            # (`delete_unselected_existing_measurements`)
            delCols = self.measurementsWin.existingUncheckedColnames
            delCols_format = [f'  *  {colname}' for colname in delCols]
            delCols_format = '\n'.join(delCols_format)
            self.logger.info(delCols_format)
            # Only from the Positions where the column was listed
            locations = self.measurementsWin.existingUncheckedLocations
            for _posData in self.data:
                posDelCols = [
                    col for col in delCols
                    if not locations
                    or _posData.pos_foldername in locations.get(col, [])
                ]
                if not posDelCols:
                    continue
                for frame_i, data_dict in enumerate(_posData.allData_li):
                    acdc_df = data_dict['acdc_df']
                    if acdc_df is None:
                        continue

                    acdc_df = acdc_df.drop(columns=posDelCols, errors='ignore')
                    _posData.allData_li[frame_i]['acdc_df'] = acdc_df
        self.setMeasWinState = self.measurementsWin.state()
        self.logger.info('Setting measurements...')
        self._setMetrics(self.measurementsWin)
        self.logger.info('Metrics successfully set.')
        self.measurementsWin = None

    def _setMetrics(self, measurementsWin):
        self._measurements_kernel.set_metrics_from_set_measurements_dialog(
            measurementsWin
        )
        for ch in self._measurements_kernel.chNamesToProcess:
            if ch not in self.notLoadedChNames:
                continue
            
            success = self.loadFluo_cb(
                fluo_channels=[ch], skipPosWithoutChannel=True
            )
            if not success:
                continue

    def addCustomMetric(self, checked=False):
        txt = measurements.add_metrics_instructions()
        metrics_path = measurements.metrics_path
        msg = widgets_message_box.myMessageBox()
        msg.addShowInFileManagerButton(metrics_path, 'Show example...')
        title = 'Add custom metrics instructions'
        msg.information(self, title, txt, buttonsTexts=('Ok',))

    def addCombineMetric(self):
        posData = self.data[self.pos_i]
        isZstack = posData.SizeZ > 1
        win = dialogs_combine_measurements.CombineMetricsEquationDialog(
            self.getAllPosChNames(), isZstack, self.isSegm3D, parent=self,
            acdc_df=posData.acdc_df
        )
        win.sigOk.connect(self.saveCombineMetricsToPosData)
        win.exec_()
        win.sigOk.disconnect()

    def saveCombineMetricsToPosData(self, window):
        for posData in self.data:
            equationsDict, isMixedChannels = window.getEquationsDict()
            for newColName, equation in equationsDict.items():
                posData.addEquationCombineMetrics(
                    equation, newColName, isMixedChannels
                )
                posData.saveCombineMetrics()
        
        if self.measurementsWin is None:
            return
        
        self.measurementsWinState = self.measurementsWin.state()
        self.measurementsWin.close()
        self.showSetMeasurements()
        self.measurementsWin.restoreState(self.measurementsWinState)

    def setMetricsFunc(self):
        posData = self.data[self.pos_i]
        self._measurements_kernel._set_metrics_func_from_posData(posData)

    def computeVolumeRegionprop(self):
        # Selected or needed by the selected measurements (e.g. input of a
        # concentration or of a combined measurement)
        kernel = self._measurements_kernel
        posToCompute = [
            posData for posData in self.data
            if kernel.needs_rotational_volume(posData)
        ]
        if not posToCompute:
            return

        # We compute the cell volume in the main thread because calling
        # skimage.transform.rotate in a separate thread causes crashes
        # with segmentation fault on macOS. I don't know why yet.
        self.logger.info('Computing cell volume...')
        end_i = self.save_until_frame_i
        pos_iter = tqdm(self.data, ncols=100)
        for p, posData in enumerate(pos_iter):
            if self.posToSave is not None:
                if posData.pos_foldername not in self.posToSave:
                    continue
            
            if not any(posData is _posData for _posData in posToCompute):
                continue
                
            PhysicalSizeY = posData.PhysicalSizeY
            PhysicalSizeX = posData.PhysicalSizeX
            frame_iter = tqdm(
                posData.allData_li[:end_i+1], ncols=100, position=1, leave=False
            )
            for frame_i, data_dict in enumerate(frame_iter):
                lab = data_dict['labels']
                if lab is None:
                    break
                rp = data_dict['regionprops']
                obj_iter = tqdm(rp, ncols=100, position=2, leave=False)
                for i, obj in enumerate(obj_iter):
                    vol_vox, vol_fl = _calc_rot_vol(
                        obj, PhysicalSizeY, PhysicalSizeX
                    )
                    obj.vol_vox = vol_vox
                    obj.vol_fl = vol_fl
                posData.allData_li[frame_i]['regionprops'] = rp
