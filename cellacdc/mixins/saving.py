"""Qt view adapter for save and autosave workflows."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from functools import partial
from typing import Literal

import pandas as pd
from dataclasses import dataclass
from typing import Literal
from qtpy.QtCore import QEventLoop, QMutex, QThread, QTimer, QWaitCondition
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QCheckBox, QMessageBox
from tqdm import tqdm

from cellacdc import _warnings, apps, disableWindow, exception_handler
from cellacdc import cca_df_colnames, html_utils, settings_csv_path, widgets
from cellacdc import load
from cellacdc import workers


_font = QFont()
_font.setPixelSize(11)


class SavingView:
    """Qt-facing adapter for save and autosave workflows."""

    LEGACY_METHODS = (
        'manageVersions',
        'turnOffAutoSaveWorker',
        'autoSaveTimerTimedOut',
        'autoSaveTimerCountFrames',
        'enqAutosave',
        'startAutoSaveEveryNframesTimer',
        '_enqueueAutoSave',
        'computeVolumeRegionprop',
        'askSaveOriginalSegm',
        'askSaveLastVisitedCcaMode',
        'askSaveLastVisitedSegmMode',
        'askSaveMetrics',
        'askSelectPos',
        'askPosToSave',
        'saveMetricsCritical',
        'saveAsData',
        'saveDataPermissionError',
        'saveDataProgress',
        'saveDataCustomMetricsCritical',
        'saveDataCombinedMetricsMissingColumn',
        'saveDataAddMetricsCritical',
        'saveDataRegionPropsCritical',
        'saveDataUpdateMetricsPbar',
        'saveDataUpdatePbar',
        'quickSave',
        'checkMissingCca',
        'warnDifferentSegmChannel',
        'waitAutoSaveWorker',
        'saveData',
        'autoSaveClose',
        'setAutoSaveSegmentationEnabled',
        'setAutoSaveAnnotationsEnabled',
        'autoSaveToggled',
        'autoSaveAnnotToggled',
        'autoSaveIntervalEdit',
        'autoSaveIntervalValueChanged',
        'autoSaveIntervalSetTooltip',
        'warnErrorsCustomMetrics',
        'warnErrorsAddMetrics',
        'warnErrorsRegionProps',
        'askConcatenate',
        'updateSegmDataAutoSaveWorker',
        'saveDataFinished',
        '_waitCloseAutoSaveWorker',
        'cancelSavingInitialisation',
        'askSaveOnClosing',
    )

    def __init__(self, host):
        object.__setattr__(self, 'host', host)
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

    def manageVersions(self):
        posData = self.data[self.pos_i]
        selectVersion = apps.SelectAcdcDfVersionToRestore(
            posData, parent=self.host
        )
        selectVersion.exec_()

        if selectVersion.cancel:
            return

        undoId = uuid.uuid4()
        if posData.cca_df is not None:
            self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        selectedTime = selectVersion.selectedTimestamp

        self.modeComboBox.setCurrentText('Viewer')
        self.logger.info(f'Loading file from {selectedTime}...')

        acdc_df = load.read_acdc_df_from_archive(
            selectVersion.archiveFilePath, selectVersion.selectedKey
        )
        posData.acdc_df = acdc_df
        frames = acdc_df.index.get_level_values(0)
        last_visited_frame_i = frames.max()
        current_frame_i = posData.frame_i
        pbar = tqdm(total=last_visited_frame_i+1, ncols=100)
        for frame_i in range(last_visited_frame_i+1):
            posData.frame_i = frame_i
            self.get_data()
            if posData.cca_df is not None:
                self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)
            if posData.allData_li[frame_i]['labels'] is None:
                pbar.update()
                continue

            if frame_i not in frames:
                acdc_df_i = pd.DataFrame(columns=acdc_df.columns)
                acdc_df_i.drop(self.cca_df_colnames, axis=1, errors='ignore')
                acdc_df_i.index.name = 'Cell_ID'
            else:
                acdc_df_i = acdc_df.loc[frame_i].dropna(axis=1, how='all')

            posData.allData_li[frame_i]['acdc_df'] = acdc_df_i
            pbar.update()
        pbar.close()

        # Back to current frame
        posData.frame_i = current_frame_i
        self.get_data(debug=False)
        self.updateAllImages()
        self.logger.info('Annotations correctly recovered.')

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
        if self.should_clear_autosave_status(mode=mode):
            if self.statusBarLabel.text().endswith('Autosaving...'):
                self.statusBarLabel.setText(
                    self.statusBarLabel.text().replace(' | Autosaving...', '')
                )
            return

        if not self.autoSaveActiveWorkers:
            self.gui_createAutoSaveWorker()

        if not self.should_enqueue_autosave(
            mode=mode,
            has_active_workers=bool(self.autoSaveActiveWorkers),
        ):
            return

        if self.autoSaveTimer.isActive():
            return

        self._enqueueAutoSave()
        autoSaveIntevalValue, autoSaveIntervalUnit = (
            self.autoSaveIntevalValueUnit
        )
        schedule = self.autosave_schedule(
            autoSaveIntevalValue, autoSaveIntervalUnit
        )
        if schedule is None:
            return

        try:
            self.autoSaveTimer.timeout.disconnect()
        except Exception as err:
            pass

        if not schedule.use_frame_timer:
            self.autoSaveTimer.timeout.connect(self.autoSaveTimerTimedOut)
            self.autoSaveTimer.start(schedule.interval_ms)
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

    def computeVolumeRegionprop(self):
        if 'cell_vol_vox' not in self._measurements_kernel.sizeMetricsToSave:
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
                    vol_vox, vol_fl = (
                        self.measurements.rotational_volume(
                            obj, PhysicalSizeY, PhysicalSizeX
                        )
                    )
                    obj.vol_vox = vol_vox
                    obj.vol_fl = vol_fl
                posData.allData_li[frame_i]['regionprops'] = rp

    def askSaveOriginalSegm(self, isQuickSave=False):
        if isQuickSave:
            return "", True, True

        posData = self.data[self.pos_i]
        if not posData.whitelist:
            return "", True, True

        help_txt = html_utils.paragraph(f"""

    """Headless decisions for save and autosave workflows."""

    viewer_mode = 'Viewer'
    segmentation_mode = 'Segmentation and Tracking'
    cell_cycle_mode = 'Cell cycle analysis'

    def should_clear_autosave_status(self, *, mode: str) -> bool:
        return mode == self.viewer_mode

    def should_enqueue_autosave(self, *, mode: str, has_active_workers: bool):
        return mode != self.viewer_mode and has_active_workers

    def autosave_schedule(
        self,
        value: float,
        unit: Literal['minutes', 'frames'],
    ) -> AutosaveSchedule | None:
        if value == 0:
            return None
        if unit == 'frames':
            return AutosaveSchedule(use_frame_timer=True)
        return AutosaveSchedule(
            use_frame_timer=False,
            interval_ms=round(value * 60 * 1000),
        )

    def autosave_interval_change(
        self,
        value: float,
        unit: Literal['minutes', 'frames'],
    ) -> AutosaveIntervalChange:
        return AutosaveIntervalChange(
            value=value,
            unit=unit,
            settings_updates={
                'autoSaveIntevalValue': str(value),
                'autoSaveIntervalUnit': unit,
            },
            log_message=f'Autosave interval changed to: {value} {unit}',
            tooltip=(
                'Change autosave interval to every N frames or minutes\n\n'
                f'Current autosave interval: {value} {unit}'
            ),
            start_frame_timer=unit == 'frames',
        )

    def concatenate_prompt_plan(
        self,
        *,
        has_main_window: bool,
        is_quick_save: bool,
        setting_exists: bool,
        show_setting_value: str | None,
    ) -> ConcatenatePromptPlan:
        if not has_main_window or is_quick_save:
            return ConcatenatePromptPlan(
                should_prompt=False,
                ensure_setting=False,
            )

        should_prompt = show_setting_value != 'No'
        return ConcatenatePromptPlan(
            should_prompt=should_prompt,
            ensure_setting=not setting_exists,
        )

    def concatenate_prompt_setting(self, *, do_not_show_again: bool) -> str:
        if do_not_show_again:
            return 'No'
        return 'Yes'

    def autosave_segmentation_enabled(self, *, mode: str, checked: bool) -> bool:
        if mode != self.segmentation_mode:
            return False
        return checked

    def autosave_annotations_enabled(self, *, mode: str, checked: bool) -> bool:
        if mode != self.viewer_mode:
            return False
        return checked

    def save_as_basename(self, basename: str) -> str:
        if basename.endswith('_'):
            return f'{basename}segm'
        return f'{basename}_segm'

    def quick_save_positions(self, position_foldername: str) -> set[str]:
        return {position_foldername}

    def should_ask_positions(
        self,
        *,
        is_snapshot: bool,
        is_quick_save: bool,
        position_count: int,
    ) -> bool:
        return is_snapshot and not is_quick_save and position_count > 1

    def should_compute_volume_metrics(
        self,
        *,
        save_metrics: bool,
        mode: str,
    ) -> bool:
        return save_metrics or mode == self.cell_cycle_mode

    def save_finished_title(
        self,
        *,
        aborted: bool,
        worker_aborted: bool,
        is_quick_save: bool,
    ) -> tuple[str, str | None]:
        if aborted or worker_aborted:
            return 'Saving process cancelled.', 'r'
        if is_quick_save:
            return 'Saved segmentation file and annotations', None
        return 'Saved!', None

            You have <b>whitelisted IDs</b> in the current position.<br>
            Do you want to save the <b>not whitelisted</b> segmentation data<br>
            This will allow you to <b>revisit the original segmentation</b>.<br>
            """)

        txt = html_utils.paragraph(f"""
            You have <b>whitelisted IDs</b> in the current position.<br>
            Do you want to save the <b>not whitelisted</b> segmentation data?<br>
            """)

        found_files = self.workspace.segmentation_files(
            posData.images_path
        )
        existingEndnames = self.workspace.endnames(
            posData.basename, found_files
        )

        segmFilename = os.path.basename(posData.segm_npz_path)
        segmFilename = f"{segmFilename[:-4]}_not_whitelisted"
        win = apps.filenameDialog(
            basename=posData.basename,
            hintText=txt,
            defaultEntry=segmFilename,
            existingNames=existingEndnames,
            helpText=help_txt,
            allowEmpty=False,
            parent=self.host,
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
        self.save_until_frame_i = 0
        if self.isSnapshot:
            return True

        frame_i = self.tracking.last_tracked_frame_index(
            (data_dict['labels'] for data_dict in posData.allData_li),
            first_frame_fallback=-1,
        )

        self.save_until_frame_i = frame_i
        self.save_cca_until_frame_i = frame_i
        self.last_tracked_i = frame_i

        if isQuickSave:
            return True

        last_cca_frame_i = self.navigateScrollBar.maximum()-1
        # Ask to save last visited frame or not
        txt = html_utils.paragraph(f"""
            You annotated the cell cycle stages up
            until frame number {last_cca_frame_i+1}.<br><br>
            Enter <b>up to which frame number</b> you want to save the
            cell cycle annotations:
        """)
        lastFrameDialog = apps.QLineEditDialog(
            title='Last annoated frame number to save',
            defaultTxt=str(last_cca_frame_i+1),
            msg=txt, parent=self.host, allowedValues=(1, last_cca_frame_i+1),
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
        self.save_until_frame_i = 0
        self.save_cca_until_frame_i = 0
        if self.isSnapshot:
            return True

        frame_i = self.tracking.last_tracked_frame_index(
            (data_dict['labels'] for data_dict in posData.allData_li),
            first_frame_fallback=-1,
        )

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
        lastFrameDialog = apps.QLineEditDialog(
            title='Last frame number to save', defaultTxt=str(frame_i+1),
            msg=txt, parent=self.host, allowedValues=(1, posData.SizeT),
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
        msg = widgets.myMessageBox(
            parent=self.host, resizeButtons=False, wrapText=False
        )
        setMeasurementsButton = widgets.setPushButton('Set measurements...')
        _, yesButton, noButton, _ = msg.question(
            self.host, 'Save measurements?', txt,
            buttonsTexts=('Cancel', 'Yes', 'No', setMeasurementsButton),
            showDialog=False
        )
        setMeasurementsButton.disconnect()
        setMeasurementsButton.clicked.connect(
            partial(
                self.measurements_view.show_set_measurements,
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
        selectPosWin = widgets.QDialogListbox(
            f'Select Positions {action}', f'Select Positions {action}:\n',
            items, multiSelection=True, parent=self.host,
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

        msg = widgets.myMessageBox(wrapText=False)
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
        msg.critical(self.host, 'Critical error while saving metrics', err_msg)

        self.is_error_state = True
        self.waitCond.wakeAll()

    def saveAsData(self, checked=True):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        existingFilenames = set()
        for _posData in self.data:
            segm_files = self.workspace.segmentation_files(
                _posData.images_path
            )
            _existingEndnames = self.workspace.endnames(
                _posData.basename, segm_files
            )
            existingFilenames.update([
                f'{_posData.basename}{endname}.npz'
                for endname in _existingEndnames
            ])
        posData = self.data[self.pos_i]
        basename = self.save_as_basename(posData.basename)
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a <b>filename</b> for the segmentation file:<br>',
            existingNames=existingFilenames
        )
        win.exec_()
        if win.cancel:
            return

        for posData in self.data:
            posData.setFilePaths(new_endname=win.entryText)

        self.status_hover_view.set_status_bar_label()
        self.saveData()

    def saveDataPermissionError(self, err_msg):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        msg = QMessageBox()
        msg.critical(self.host, 'Permission denied', err_msg, msg.Ok)
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
        self.worker.customMetricsErrors[func_name] = traceback_format

    def saveDataCombinedMetricsMissingColumn(self, error_msg, func_name):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        self.logger.info('')
        warning = f'[WARNING]: {error_msg}. Metric {func_name} was skipped.'
        _hl = '===================================='
        self.logger.info(f'{_hl}\n{warning}\n{_hl}')
        self.worker.customMetricsErrors[func_name] = warning

    def saveDataAddMetricsCritical(self, traceback_format, error_message):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        self.logger.info('')
        _hl = '===================================='
        self.logger.info(f'{_hl}\n{traceback_format}\n{_hl}')
        self.worker.addMetricsErrors[error_message] = traceback_format

    def saveDataRegionPropsCritical(self, traceback_format, error_message):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        self.logger.info('')
        _hl = '===================================='
        self.logger.info(f'{_hl}\n{traceback_format}\n{_hl}')
        self.worker.regionPropsErrors[error_message] = traceback_format

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
            ETA = self.formatting.seconds_to_eta(seconds)
            self.saveWin.ETA_label.setText(f'ETA: {ETA}')

    def quickSave(self):
        self.saveData(isQuickSave=True)

    def checkMissingCca(self):
        proceed = True
        ignore = False
        doNotShowAgain = False
        if not self.doNotShowAgainMissingCca:
            return proceed, ignore, doNotShowAgain

        missing_cca_items = [
            (item.cca_df, self.data[item.position_i], item.frame_i)
            for item in self.cca_workflows.missing_annotation_items(
                (posData.allData_li for posData in self.data),
                cca_df_colnames,
                is_snapshot=self.isSnapshot,
            )
        ]

        if not missing_cca_items:
            return proceed, ignore, doNotShowAgain

        proceed = False
        ignore, doNotShowAgain =_warnings.warnMissingCca(
            missing_cca_items, qparent=self.host
        )

        if doNotShowAgain:
            self.df_settings.at['doNotShowAgainMissingCca', 'value'] = 'Yes'
            self.df_settings.to_csv(self.settings_csv_path)

        return proceed, ignore, doNotShowAgain

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
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.warning(
            self.host, 'WARNING: Potential for data loss', txt,
            buttonsTexts=('Cancel', 'Yes')
        )
        return msg.cancel

    def waitAutoSaveWorker(self, worker):
        if worker.isFinished or worker.isPaused or len(worker.dataQ) == 0:
            self.waitAutoSaveWorkerLoop.exit()
            self.waitAutoSaveWorkerTimer.stop()
            self.status_hover_view.set_status_bar_label(log=False)

    @exception_handler
    def saveData(self, checked=False, finishedCallback=None, isQuickSave=False):
        self.setDisabled(True, keepDisabled=True)

        self.askLineageTreeChanges()

        self.store_data(autosave=False)
        self.applyDelROIs()
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
        if self.should_ask_positions(
            is_snapshot=self.isSnapshot,
            is_quick_save=isQuickSave,
            position_count=len(self.data),
        ):
            self.posToSave = self.askPosToSave()
            if self.posToSave is None:
                self.cancelSavingInitialisation()
                self.setDisabled(False, keepDisabled=False)
                self.activateWindow()
                return True

        if isQuickSave:
            # Quick save only current pos
            self.posToSave = self.quick_save_positions(
                self.data[self.pos_i].pos_foldername
            )

        if self.isSnapshot:
            self.store_data(mainThread=False)

        mode = self.modeComboBox.currentText()
        if mode == 'Cell cycle analysis':
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

        append_name_og_whitelist, proceed, do_not_save_og_whitelist = self.askSaveOriginalSegm(isQuickSave=isQuickSave)
        if not proceed:
            self.cancelSavingInitialisation()
            self.setDisabled(False, keepDisabled=False)
            self.activateWindow()
            return True

        if self.should_compute_volume_metrics(
            save_metrics=self.save_metrics,
            mode=mode,
        ):
            self.computeVolumeRegionprop()

        infoTxt = html_utils.paragraph(
            f'Saving {self.exp_path}...<br>', font_size='14px'
        )

        self.saveWin = apps.QDialogPbar(
            parent=self.host, title='Saving data', infoTxt=infoTxt
        )
        self.saveWin.setFont(_font)
        # if not self.save_metrics:
        self.saveWin.metricsQPbar.hide()
        self.saveWin.progressLabel.setText('Preparing data...')
        self.saveWin.show()

        # Set up separate thread for saving and show progress bar widget
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.thread = QThread()
        self.worker = workers.saveDataWorker(self.host)
        self.worker.mode = mode
        self.worker.isQuickSave = isQuickSave
        self.worker.append_name_og_whitelist = append_name_og_whitelist
        self.worker.do_not_save_og_whitelist = do_not_save_og_whitelist

        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.finished.connect(self.saveDataFinished)
        if finishedCallback is not None:
            self.worker.finished.connect(finishedCallback)
        self.worker.progress.connect(self.saveDataProgress)
        self.worker.sigLog.connect(self.workerLog)
        self.worker.progressBar.connect(self.saveDataUpdatePbar)
        # self.worker.metricsPbarProgress.connect(self.saveDataUpdateMetricsPbar)
        self.worker.critical.connect(self.saveDataWorkerCritical)
        self.worker.customMetricsCritical.connect(
            self.saveDataCustomMetricsCritical
        )
        self.worker.sigCombinedMetricsMissingColumn.connect(
            self.saveDataCombinedMetricsMissingColumn
        )
        self.worker.addMetricsCritical.connect(self.saveDataAddMetricsCritical)
        self.worker.regionPropsCritical.connect(
            self.saveDataRegionPropsCritical
        )
        self.worker.criticalPermissionError.connect(self.saveDataPermissionError)
        self.worker.askZsliceAbsent.connect(self.zSliceAbsent)
        self.worker.sigDebug.connect(self._workerDebug)

        self.thread.started.connect(self.worker.run)

        self.thread.start()

        return False

    def autoSaveClose(self):
        for worker, thread in self.autoSaveActiveWorkers:
            worker._stop()

    def setAutoSaveSegmentationEnabled(self, enabled):
        if not self.autoSaveActiveWorkers:
            return

        worker, thread = self.autoSaveActiveWorkers[-1]

        if enabled:
            worker.isAutoSaveON = self.autosave_segmentation_enabled(
                mode=self.modeComboBox.currentText(),
                checked=self.autoSaveToggle.isChecked(),
            )
        else:
            worker.isAutoSaveON = False

    def setAutoSaveAnnotationsEnabled(self, enabled):
        if not self.autoSaveActiveWorkers:
            return

        worker, thread = self.autoSaveActiveWorkers[-1]

        if enabled:
            worker.isAutoSaveAnnotON = (
                self.autosave_annotations_enabled(
                    mode=self.modeComboBox.currentText(),
                    checked=self.autoSaveToggle.isChecked(),
                )
            )
        else:
            worker.isAutoSaveAnnotON = False

    def autoSaveToggled(self, checked):
        if not self.autoSaveActiveWorkers:
            self.gui_createAutoSaveWorker()

        if not self.autoSaveActiveWorkers:
            return

        worker, thread = self.autoSaveActiveWorkers[-1]

        mode = self.modeComboBox.currentText()
        worker.isAutoSaveON = self.autosave_segmentation_enabled(
            mode=mode,
            checked=checked,
        )

    def autoSaveAnnotToggled(self, checked):
        if not self.autoSaveActiveWorkers:
            self.gui_createAutoSaveWorker()

        if not self.autoSaveActiveWorkers:
            return

        worker, thread = self.autoSaveActiveWorkers[-1]

        mode = self.modeComboBox.currentText()
        worker.isAutoSaveAnnotON = (
            self.autosave_annotations_enabled(
                mode=mode,
                checked=checked,
            )
        )

    def autoSaveIntervalEdit(self):
        self.autoSaveIntervalDialog.show()
        self.autoSaveIntervalDialog.raise_()
        self.autoSaveIntervalDialog.activateWindow()

    def autoSaveIntervalValueChanged(
            self, value: float, unit: Literal['minutes', 'frames']
        ):
        interval_change = self.autosave_interval_change(
            value,
            unit,
        )
        self.autoSaveIntevalValueUnit = (
            interval_change.value,
            interval_change.unit,
        )
        self.autoSaveTimer.stop()

        for setting, setting_value in interval_change.settings_updates.items():
            self.df_settings.at[setting, 'value'] = setting_value
        self.df_settings.to_csv(settings_csv_path)

        self.logger.info(interval_change.log_message)
        self.autoSaveIntervalSetTooltip(interval_change.tooltip)

        if interval_change.start_frame_timer:
            self.startAutoSaveEveryNframesTimer()

    def autoSaveIntervalSetTooltip(self, tooltip=None):
        if tooltip is None:
            value, unit = self.autoSaveIntevalValueUnit
            tooltip = self.autosave_interval_change(
                value,
                unit,
            ).tooltip
        self.autoSaveIntervalLabel.setToolTip(tooltip)
        self.autoSaveIntervalEditButton.setToolTip(tooltip)

    def warnErrorsCustomMetrics(self):
        win = apps.ComputeMetricsErrorsDialog(
            self.worker.customMetricsErrors, self.logs_path,
            log_type='custom_metrics', parent=self.host
        )
        win.exec_()

    def warnErrorsAddMetrics(self):
        win = apps.ComputeMetricsErrorsDialog(
            self.worker.addMetricsErrors, self.logs_path,
            log_type='standard_metrics', parent=self.host
        )
        win.exec_()

    def warnErrorsRegionProps(self):
        win = apps.ComputeMetricsErrorsDialog(
            self.worker.regionPropsErrors, self.logs_path,
            log_type='region_props', parent=self.host
        )
        win.exec_()

    def askConcatenate(self):
        setting_exists = 'showAskConcatenate' in self.df_settings.index
        show_setting_value = (
            self.df_settings.at['showAskConcatenate', 'value']
            if setting_exists else None
        )
        prompt_plan = self.concatenate_prompt_plan(
            has_main_window=self.mainWin is not None,
            is_quick_save=self._isQuickSave,
            setting_exists=setting_exists,
            show_setting_value=show_setting_value,
        )
        if prompt_plan.ensure_setting:
            self.df_settings.at['showAskConcatenate', 'value'] = 'Yes'

        if not prompt_plan.should_prompt:
            return

        txt = html_utils.paragraph(f"""
            Do you want to <b>concatenate</b> the `acdc_output.csv` tables from
            multiple Positions into <b>one single CSV file</b>?<br>
        """)
        doNotShowAgainCheckbox = QCheckBox('Do not show again')
        msg = widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.question(
            self.host, 'Concatenate tables?', txt,
            buttonsTexts=('No', 'Yes'),
            widgets=doNotShowAgainCheckbox
        )
        show_ask_concatenate = self.concatenate_prompt_setting(
            do_not_show_again=doNotShowAgainCheckbox.isChecked()
        )
        self.df_settings.at['showAskConcatenate', 'value'] = (
            show_ask_concatenate
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
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self.host, 'How to concatenate tables', txt)

    def updateSegmDataAutoSaveWorker(self):
        # Update savedSegmData in autosave worker
        posData = self.data[self.pos_i]
        for worker, thread in self.autoSaveActiveWorkers:
            worker.savedSegmData = posData.segm_data.copy()

    def saveDataFinished(self):
        self.setDisabled(False, keepDisabled=False)
        self.activateWindow()
        title_text, color = self.save_finished_title(
            aborted=self.saveWin.aborted,
            worker_aborted=self.worker.abort,
            is_quick_save=self._isQuickSave,
        )
        if color is None:
            self.titleLabel.setText(title_text)
        else:
            self.titleLabel.setText(title_text, color=color)
        self.saveWin.workerFinished = True
        self.saveWin.close()

        if not self.closeGUI:
            # Update savedSegmData in autosave worker
            self.updateSegmDataAutoSaveWorker()

        if self.worker.addMetricsErrors:
           self.warnErrorsAddMetrics()
        if self.worker.regionPropsErrors:
            self.warnErrorsRegionProps()
        if self.worker.customMetricsErrors:
            self.warnErrorsCustomMetrics()

        self.checkManageVersions()

        self.askConcatenate()

        if self.closeGUI:
            salute_string = self.formatting.salute_string()
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'Data <b>saved!</b>. The GUI will now close.<br><br>'
                f'{salute_string}'
            )
            msg.information(self.host, 'Data saved', txt)
            self.close()

    def _waitCloseAutoSaveWorker(self):
        didWorkersFinished = [True]
        for worker, thread in self.autoSaveActiveWorkers:
            if worker.isFinished:
                didWorkersFinished.append(True)
            else:
                didWorkersFinished.append(False)
        if all(didWorkersFinished):
            self.waitCloseAutoSaveWorkerLoop.stop()

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

        msg = widgets.myMessageBox()
        txt = html_utils.paragraph('Do you want to <b>save before closing?</b>')
        _, noButton, yesButton = msg.question(
            self.host, 'Save?', txt,
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