import os
import traceback
import logging
from functools import partial
from types import SimpleNamespace
import datetime

import pandas as pd

from tqdm import tqdm

from qtpy.QtCore import Signal, QThread
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QStyle
)

from .base import NewThreadMultipleExpBaseUtil

from .. import (
    widgets, apps, workers, html_utils, myutils,
    gui, cca_functions, load, printl
)

from .. import cellacdc_path, settings_folderpath
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QSpinBox
from cellacdc import prompts
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base
from cellacdc.shared.dialogs import image_tools as dialogs_image_tools

favourite_func_metrics_csv_path = os.path.join(
    settings_folderpath, 'favourite_func_metrics.csv'
)

class computeMeasurmentsUtilWin(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, parent=None, segmEndname='', 
            doRunComputation=True
        ):
        title = 'Compute measurements utility'
        infoText = 'Computing measurements routine running...'
        progressDialogueTitle = 'Computing measurements'
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )

        self.parent = parent
        
        self.cancel = False

        self.endFilenameSegm = segmEndname
        self.doRunComputation = doRunComputation
        self.isWorkerFinished = False
        self.doNotAskAgain = False

    def showEvent(self, event):
        self.runWorker()

    def runWorker(self, showProgress=True, stopFrameNumber=None):
        self.gui = gui.guiWin(self.app, parent=self.parent)
        self.gui.logger = self.logger

        self.progressWin = apps.QDialogWorkerProgress(
            title='Computing measurements', parent=self,
            pbarDesc='Computing measurements...'
        )
        self.progressWin.sigClosed.connect(self.progressWinClosed)
        self.progressWin.show(self.app)

        if not showProgress:
            self.progressWin.hide()
        
        self.thread = QThread()
        self.worker = workers.ComputeMetricsWorker(self)
        self.worker.moveToThread(self.thread)

        self.worker.signals.finished.connect(self.thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.worker.signals.finished.connect(self.workerFinished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.signals.progress.connect(self.workerProgress)
        self.worker.signals.critical.connect(self.workerCritical)
        if not self.endFilenameSegm:
            self.worker.signals.sigSelectSegmFiles.connect(
                self.selectSegmFileLoadData
            )
        else:
            self.worker.signals.sigSelectSegmFiles.connect(
                self.wakeUpWorkerThread
            )
        self.worker.signals.sigInitAddMetrics.connect(self.initAddMetricsWorker)
        self.worker.signals.sigPermissionError.connect(self.warnPermissionError)
        self.worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.signals.sigUpdatePbarDesc.connect(self.workerUpdatePbarDesc)
        self.worker.signals.sigComputeVolume.connect(self.computeVolumeRegionprop)
        self.worker.signals.sigAskRunNow.connect(
            self.askRunNowOrSaveToConfig
        )
        
        if stopFrameNumber is None:
            self.worker.signals.sigAskStopFrame.connect(self.workerAskStopFrame)
        else:
            self.worker.signals.sigAskStopFrame.connect(
                partial(self.setStopFrame, stopFrameNumber=stopFrameNumber)
            )
        self.worker.signals.sigErrorsReport.connect(self.warnErrors)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def askRunNowOrSaveToConfig(self, worker):
        self.worker.savedToWorkflow = False

        txt = html_utils.paragraph("""
            Do you want to <b>compute the measurements now</b><br>
            or save the  workflow to a <b>configuration file</b> and run it 
            <b>later?</b><br><br>
            With the configuration file you can also run the workflow on a<br>
            computing cluster that does not support GUI elements 
            (i.e., headless).<br>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        saveButton = widgets.savePushButton('Save and run later')
        runNowButton = widgets.playPushButton('Run now')
        _, saveButton, runNowButton = msg.question(
            self, 'Run workflow now?', txt, 
            buttonsTexts=(
                'Cancel', saveButton, runNowButton
            )
        )
        self.worker.runNowAlreadyAsked = True

        if not msg.clickedButton == saveButton:
            self.worker.abort = msg.cancel
            self.worker.waitCond.wakeAll()
            return
        
        timestamp = datetime.datetime.now().strftime(
            r'%Y-%m-%d_%H-%M'
        )
        win = apps.filenameDialog(
            parent=self, 
            ext='.ini', 
            title='Insert filename for configuration file',
            hintText='Insert filename for the configuration file',
            allowEmpty=False, 
            defaultEntry=f'{timestamp}_acdc_measurements_workflow'
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        config_filename = win.filename
        mostRecentPath = myutils.getMostRecentPath()
        folder_path = apps.get_existing_directory(
            allow_images_path=False,
            parent=self.progressWin, 
            caption='Select folder where to save configuration file',
            basedir=mostRecentPath,
            # options=QFileDialog.DontUseNativeDialog
        )
        if not folder_path:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        config_filepath = os.path.join(folder_path, config_filename)
        kernel = self.worker.kernel
        self.saveConfigurationFile(config_filepath, kernel)
        
        self.worker.savedToWorkflow = True
        self.worker.waitCond.wakeAll()
        
    def saveConfigurationFile(self, config_filepath, kernel):
        ini_items = {'workflow': {'type': 'measurements'}}
        ini_items['measurements'] = kernel.to_workflow_config_params()
        paths = []            
        stopFrames = []
        for pathInfo in self.worker.allPosDataInputs:
            images_path = os.path.dirname(pathInfo['file_path'])
            paths.append(images_path)
            stopFrames.append(pathInfo['stopFrameNum'])
        
        load.save_workflow_to_config(
            config_filepath, 
            ini_items, 
            paths, 
            stopFrames,
            type='measure'
        )
        self.worker.kernel.setup_done = True
        
        txt = html_utils.paragraph(
            'Compute measurements workflow successfully saved to the following location:<br><br>'
            f'<code>{config_filepath}</code><br><br>'
            'You can run the workflow with the following command:'
        )
        command = f'acdc -p "{config_filepath}"'
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self, 'Workflow save', txt, 
            commands=(command,),
            path_to_browse=os.path.dirname(config_filepath)
        )
    
    def setStopFrame(self, posDatas, stopFrameNumber=1):
        for p, posData in enumerate(posDatas):
            if isinstance(stopFrameNumber, int):
                stop_frame_n = stopFrameNumber
            else:
                stop_frame_n = stopFrameNumber[p]
            posData.stopFrameNum = stop_frame_n

        self.worker.waitCond.wakeAll()
        
    def wakeUpWorkerThread(self, *args, **kwargs):
        self.worker.waitCond.wakeAll()
    
    def warnErrors(
            self, standardMetricsErrors, customMetricsErrors, regionPropsErrors
        ):
        if standardMetricsErrors:
            win = apps.ComputeMetricsErrorsDialog(
                standardMetricsErrors, self.logs_path, 
                log_type='standard_metrics', parent=self
            )
            win.exec_()
        if regionPropsErrors:
            win = apps.ComputeMetricsErrorsDialog(
                regionPropsErrors, self.logs_path, 
                log_type='region_props', parent=self
            )
            win.exec_()
        if customMetricsErrors:
            win = apps.ComputeMetricsErrorsDialog(
                customMetricsErrors, self.logs_path, 
                log_type='custom_metrics', parent=self
            )
            win.exec_()
        self.worker.waitCond.wakeAll()

    def workerAskStopFrame(self, posDatas):
        win = apps.stopFrameDialog(posDatas, parent=self)
        win.exec_()
        self.worker.abort = win.cancel
        self.worker.waitCond.wakeAll()

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)

    def workerUpdatePbarDesc(self, desc):
        self.progressWin.progressLabel.setText(desc)

    def warnPermissionError(self, traceback_str, path):
        err_msg = html_utils.paragraph(
            'The file below is open in another app '
            '(Excel maybe?).<br><br>'
            f'{path}<br><br>'
            'Close file and then press "Ok".'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.setDetailedText(traceback_str)
        msg.warning(self, 'Permission error', err_msg)
        self.worker.waitCond.wakeAll()

    def selectSegmFileLoadData(self, exp_path, pos_foldernames):
        # Get end name of every existing segmentation file
        existingSegmEndNames = set()
        for p, pos in enumerate(pos_foldernames):
            pos_path = os.path.join(exp_path, pos)
            images_path = os.path.join(pos_path, 'Images')
            basename, chNames = myutils.getBasenameAndChNames(images_path)
            # Use first found channel, it doesn't matter for metrics
            for chName in chNames:
                filePath = myutils.getChannelFilePath(images_path, chName)
                if filePath:
                    break
            else:
                raise FileNotFoundError(
                    f'None of the channels "{chNames}" were found in the path '
                    f'"{images_path}".'
                )
            _posData = load.loadData(filePath, chName)
            _posData.getBasenameAndChNames()
            segm_files = load.get_segm_files(_posData.images_path)
            _existingEndnames = load.get_endnames(
                _posData.basename, segm_files
            )
            existingSegmEndNames.update(_existingEndnames)

        if len(existingSegmEndNames) == 1:
            self.endFilenameSegm = list(existingSegmEndNames)[0]
            self.worker.waitCond.wakeAll()
            return

        win = apps.SelectSegmFileDialog(
            existingSegmEndNames, exp_path, parent=self
        )
        win.exec_()
        self.endFilenameSegm = win.selectedItemText
        self.worker.abort = win.cancel
        self.worker.waitCond.wakeAll()
    
    def addCombineMetric(self):
        isZstack = self.posData.SizeZ > 1
        self.combineMetricWindow = apps.CombineMetricsEquationDialog(
            self.allChNames, isZstack, self.posData.isSegm3D,
            parent=self.measurementsWin, closeOnOk=False,
            acdc_df=self.posData.acdc_df
        )
        self.combineMetricWindow.sigOk.connect(self.saveCombineMetricsToPosData)
        self.combineMetricWindow.show()
    
    def saveCombineMetricsToPosData(self, window):
        for p, _posData in enumerate(self.allPosData):
            equationsDict, isMixedChannels = window.getEquationsDict()
            for newColName, equation in equationsDict.items():
                _posData.addEquationCombineMetrics(
                    equation, newColName, isMixedChannels
                )
                _posData.saveCombineMetrics()
        
        self.combineMetricWindow.close()
        self.measurementsWinState = self.measurementsWin.state()
        self.measurementsWin.restart()
        self.initAddMetricsWorker(self.posData, self.allPosDataInputs)
        self.measurementsWin.restoreState(self.measurementsWinState)

    @staticmethod
    def _getAllAcdcDfColumns(allPosDataInputs):
        """Ordered union of the columns of the existing acdc_output tables"""
        columns = []
        seen = set()
        for posDataInputs in allPosDataInputs:
            for col in posDataInputs.get('acdcDfColumns', []):
                if col in seen:
                    continue
                seen.add(col)
                columns.append(col)
        return columns

    @staticmethod
    def _posDataInputsPlaceholder(posDataInputs):
        """Position of `posDataInputs` with what the measurements kernel
        needs to classify its columns (channels and combined measurements)
        """
        file_path = posDataInputs.get('file_path', '')
        pos_path = os.path.dirname(os.path.dirname(file_path))
        return SimpleNamespace(
            chNames=list(posDataInputs.get('chNames', [])),
            combineMetricsConfig=posDataInputs.get('combineMetricsConfig'),
            pos_foldername=os.path.basename(pos_path) if file_path else None,
            exp_path=os.path.dirname(pos_path) if file_path else None,
        )

    @staticmethod
    def unselectedExistingMeasurements(kernel, allPosDataInputs):
        """Existing measurement columns of each Position (classified with
        the combined measurements of the Position) that the selection of
        `kernel` does not save.

        Returns
        -------
        tuple
            `(ordered union of the columns, {colname: [Position folders]})`
        """
        UtilWin = computeMeasurmentsUtilWin
        unselected = []
        locations = {}
        for posDataInputs in allPosDataInputs:
            posData = UtilWin._posDataInputsPlaceholder(posDataInputs)
            colnames = kernel.unselected_existing_measurements(
                posDataInputs.get('acdcDfColumns', []), posData
            )
            for colname in colnames:
                if colname not in locations:
                    locations[colname] = []
                    unselected.append(colname)
                locations[colname].append(posData.pos_foldername)
        return unselected, locations

    def askDeleteUnselectedExistingMeasurements(
            self, posData, allPosDataInputs
        ):
        """With "Use the same selection for all following experiments", ask
        the Delete/Keep question for the existing measurements of this
        experiment that are not selected and were not listed yet (only the
        listed columns are deleted, see
        `cli.ComputeMeasurementsKernel.record_existing_measurements_answer`).
        """
        kernel = self.worker.kernel
        unselected, locations = (
            computeMeasurmentsUtilWin.unselectedExistingMeasurements(
                kernel, allPosDataInputs
            )
        )
        answered = getattr(kernel, 'existing_measurements_answered', None)
        answered = answered or set()
        new_unselected = [col for col in unselected if col not in answered]
        if not new_unselected:
            return

        exp_path = getattr(posData, 'exp_path', None)
        cancel, delete = apps.askDeleteUncheckedExistingMeasurements(
            self, new_unselected, acdc_df_path=posData.images_path,
            locations=locations, num_positions=len(allPosDataInputs),
            exp_foldername=os.path.basename(exp_path) if exp_path else None
        )
        if cancel:
            self.worker.abort = True
            return

        kernel.record_existing_measurements_answer(new_unselected, delete)

    @staticmethod
    def _getAllChNames(posData, allPosDataInputs):
        """Ordered union of the channel names found in all positions
        (order of first appearance, starting from `posData`)."""
        allChNames = list(posData.chNames)
        for posDataInputs in allPosDataInputs:
            for chName in posDataInputs.get('chNames', []):
                if chName not in allChNames:
                    allChNames.append(chName)
        return allChNames

    def initAddMetricsWorker(self, posData, allPosDataInputs):
        allChNames = self._getAllChNames(posData, allPosDataInputs)
        if (
                self.doNotAskAgain
                and set(allChNames) == set(self.worker.kernel.ch_names)
            ):
            self.askDeleteUnselectedExistingMeasurements(
                posData, allPosDataInputs
            )
            self.worker.waitCond.wakeAll()
            return

        # Set measurements
        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            favourite_funcs = df_favourite_funcs['favourite_func_name'].to_list()
        except Exception as e:
            favourite_funcs = None

        self.posData = posData
        self.allPosDataInputs = allPosDataInputs
        self.allChNames = allChNames

        # Positions of this experiment (the worker builds new inputs for
        # each experiment, "Add combined measurement" re-uses them)
        if getattr(self, '_allPosDataInputs', None) is not allPosDataInputs:
            self._allPosDataInputs = allPosDataInputs
            self.allPosData = []
            for p, posDataInputs in enumerate(self.allPosDataInputs):
                combineMetricsConfig = posDataInputs.get('combineMetricsConfig')
                combineMetricsPath = posDataInputs.get('combineMetricsPath')

                # Here we build a placeholder loadData class but we get what is
                # needed to save custom combine metrics from posDataInputs
                _posData = load.loadData(
                    posDataInputs.get('file_path', self.posData.imgPath),
                    posDataInputs.get('chName', self.posData.user_ch_name)
                )
                if combineMetricsConfig is not None:
                    _posData.combineMetricsConfig = combineMetricsConfig
                    _posData.custom_combine_metrics_path = combineMetricsPath
                _posData.chNames = list(
                    posDataInputs.get('chNames', posData.chNames)
                )
                self.allPosData.append(_posData)

        self.measurementsWin = apps.SetMeasurementsDialog(
            allChNames, [], posData.SizeZ > 1, posData.isSegm3D,
            favourite_funcs=favourite_funcs, posData=posData,
            addCombineMetricCallback=self.addCombineMetric,
            allPosData=self.allPosData,
            addDoNotAskAgainCheckbox=True,
            # Existing columns --> on OK the dialog asks whether to delete
            # the existing measurements that are not selected
            allPos_acdc_df_cols=self._getAllAcdcDfColumns(allPosDataInputs),
            allPos_acdc_df_cols_per_pos=[
                posDataInputs.get('acdcDfColumns', [])
                for posDataInputs in allPosDataInputs
            ],
            acdc_df_path=posData.images_path,
        )
        self.measurementsWin.sigClosed.connect(self.askSaveObjectsCount)
        self.measurementsWin.sigCancel.connect(self.cancelWorkerMeasurementsWin)
        self.measurementsWin.show()
    
    def cancelWorkerMeasurementsWin(self):
        self.worker.abort = self.measurementsWin.cancel
        self.worker.waitCond.wakeAll()
        self.cancel = True

    def askSaveObjectsCount(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            Do you also want to save an additional table with the 
            number of objects in each frame?<br><br>
            This table will be saved in each position folder as a CSV file 
            ending with <code>acdc_objects_count</code>.
        """)
        noButton, yesButton = msg.question(
            self, 'Save objects count?', txt, 
            buttonsTexts=('No', 'Yes, save objects count')
        )
        if msg.clickedButton == yesButton:
            self.worker.kernel.set_save_objects_count_table(True)
        
        self.startSaveDataWorker()
    
    def startSaveDataWorker(self):
        self.doNotAskAgain = self.measurementsWin.doNotAskAgain
        if self.doNotAskAgain:
            self.worker.doInitKernel = False
            
        self.worker.kernel.init_args(
            self.allChNames, self.endFilenameSegm
        )
        self.worker.kernel.set_metrics_from_set_measurements_dialog(
            self.measurementsWin
        )
        
        if not self.doRunComputation:
            self.worker.setup_done = True
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        self.gui.mutex = self.worker.mutex
        self.gui.waitCond = self.worker.waitCond
        self.gui.saveWin = self.progressWin
        
        self.gui.saveDataWorker = workers.saveDataWorker(self.gui)

        self.gui.saveDataWorker.criticalPermissionError.connect(self.skipEvent)
        self.gui.saveDataWorker.askZsliceAbsent.connect(self.gui.zSliceAbsent)
        self.gui.saveDataWorker.customMetricsCritical.connect(
            self.addCombinedMetricsError
        )
        self.gui.saveDataWorker.regionPropsCritical.connect(
            self.addRegionPropsErrors
        )
        self.gui.worker = self.gui.saveDataWorker
        self.worker.waitCond.wakeAll()
    
    def addRegionPropsErrors(self, traceback_format, error_message):
        self.logger.info('')
        print('====================================')
        self.logger.info(traceback_format)
        print('====================================')
        self.worker.regionPropsErrors[error_message] = traceback_format
    
    def addCombinedMetricsError(self, traceback_format, func_name):
        self.logger.info('')
        print('====================================')
        self.logger.info(traceback_format)
        print('====================================')
        self.worker.customMetricsErrors[func_name] = traceback_format

    def skipEvent(self, dummy):
        self.worker.waitCond.wakeAll()

    def computeVolumeRegionprop(self, end_frame_i, posData):
        # Selected or needed by the selected measurements (e.g. input of a
        # concentration or of a combined measurement)
        if not self.worker.kernel.needs_rotational_volume(posData):
            self.worker.waitCond.wakeAll()
            return

        # We compute the cell volume in the main thread because calling
        # skimage.transform.rotate in a separate thread causes crashes
        # with segmentation fault on macOS. I don't know why yet.
        self.logger.info('Computing cell volume...')
        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeX = posData.PhysicalSizeX
        iterable = enumerate(tqdm(posData.allData_li[:end_frame_i+1], ncols=100))
        for frame_i, data_dict in iterable:
            lab = data_dict['labels']
            rp = data_dict['regionprops']
            obj_iter = tqdm(rp, ncols=100, position=1, leave=False)
            for i, obj in enumerate(obj_iter):
                vol_vox, vol_fl = cca_functions._calc_rot_vol(
                    obj, PhysicalSizeY, PhysicalSizeX
                )
                obj.vol_vox = vol_vox
                obj.vol_fl = vol_fl
            posData.allData_li[frame_i]['regionprops'] = rp
        self.worker.waitCond.wakeAll()

    def progressWinClosed(self, aborted):
        self.abort = aborted
        if aborted and self.worker is not None:
            self.worker.abort = True
            self.close()

    def cancelCallback(self):
        self.abort = True
        if self.worker is not None:
            self.worker.abort = True
        else:
            self.close()

    def workerFinished(self, worker):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            
        if worker.setup_done:
            txt = 'Measurements set up completed.'
            self.logger.info(txt)
        elif worker.abort:
            txt = 'Computing measurements cancelled.'
            self.logger.info(txt)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.warning(self, 'Process cancelled', html_utils.paragraph(txt))
        
        else:
            txt = 'Computing measurements completed.'
            self.logger.info(txt)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.information(self, 'Process completed', html_utils.paragraph(txt))

        self.isWorkerFinished = True
        self.progressWin = None
        self.close()

    def workerProgress(self, text, loggerLevel='INFO'):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        if loggerLevel.upper() == 'EXCEPTION':
            loggerLevel = 'ERROR'
        self.logger.log(getattr(logging, loggerLevel.upper()), text)


# class CellConstantEquationDialog(_CombineMetricsDialog):
#     """Shared formula editor with candidate and enquiring-cell input trees."""
#     def __init__(
#             self, constant, acdc_df=None, SizeT=None, condition=False,
#             parent=None
#         ):
#         from . import timepoint_predictors
#         self._predictors = timepoint_predictors
#         self.constant = dict(constant)
#         self.SizeT = SizeT
#         self.isCondition = condition
#         self.sourceDf = acdc_df
#         df = (
#             timepoint_predictors._flat_df(acdc_df)
#             if acdc_df is not None else pd.DataFrame(columns=['frame_i', 'Cell_ID'])
#         )
#         super().__init__(
#             [], False, False, parent=parent,
#             acdcDfs={'Candidate cell': df, 'Enquiring cell': df}
#         )
#         self.setWindowTitle('Cell filter formula' if condition else 'Constant formula')
#         self.newColNameLineEdit.setText(constant['name'])
#         self.newColNameLineEdit.setReadOnly(True)
#         self.equation = ''
#         functions = QHBoxLayout()
#         for name, (_, description) in timepoint_predictors.CELL_FUNCTIONS.items():
#             button = QPushButton(name)
#             button.setToolTip(description)
#             button.clicked.connect(partial(self.insertEquationText, f'{name}('))
#             functions.addWidget(button)
#         self.layout().insertLayout(self.layout().count()-3, functions)
#         infoLabel = QLabel(html_utils.paragraph(
#             'Left: candidate measurements. '
#             'Right: enquiring-cell measurements at the same frame, prefixed '
#             '<code>this_</code>. Double-click inputs to insert them.<br>'
#             + ('Return bool or 0/1, either a scalar or one value per overlapping '
#                'frame. Any/All selects cells; Only where true selects frames.'
#                if condition else
#                'Reduce the selected source rows to one value using MAX, MEAN, '
#                'FIRST, etc. Time scope determines whether this is per frame '
#                'over the entire video, or within a custom window around t.')
#         ))
#         infoLabel.setWordWrap(True)
#         self.layout().insertWidget(0, infoLabel)
#         self.formulaEditWidget.setText(
#             constant['condition' if condition else 'equation']
#         )

#     def addTreeItems(self, parentItem, itemsText, **kwargs):
#         start = parentItem.childCount()
#         super().addTreeItems(parentItem, itemsText, **kwargs)
#         for i in range(start, parentItem.childCount()):
#             item = parentItem.child(i)
#             if not getattr(item, 'isCol', False):
#                 continue
#             old = item.variableName
#             name = item.text(0)
#             item.variableName = f'this_{name}' if kwargs.get('index') == 1 else name
#             self.variableNames[self.variableNames.index(old)] = item.variableName

#     def getTestData(self, preview=False):
#         df = self.acdcDfs['Candidate cell'].head(10).reset_index(drop=True).copy()
#         for col in list(df.columns):
#             df[f'this_{col}'] = df[col]
#         return df

#     def validateEquation(self, formula):
#         try:
#             if not formula.strip():
#                 raise ValueError('Formula cannot be empty.')
#             result = self._predictors.eval_cell_equation(
#                 self.getTestData(), formula,
#                 {'SizeT': self.SizeT} if self.SizeT is not None else None
#             )
#             if self.isCondition:
#                 self._predictors.boolean_filter_values(result, len(self.getTestData()))
#         except Exception as err:
#             return False, str(err)
#         return True, formula

#     def test_cb(self):
#         formula = self.formulaEditWidget.text()
#         success, message = self.validateEquation(formula)
#         if success:
#             constant = dict(self.constant)
#             constant['condition' if self.isCondition else 'equation'] = formula
#             if self.isCondition:
#                 constant['condition_type'] = 'equation'
#                 # Show selected source-cell counts independently of the value formula.
#                 constant['equation'] = 'NUNIQUE(Cell_ID)'
#             try:
#                 values = self._predictors.compute_cell_constants(
#                     self.sourceDf, [constant], SizeT=self.SizeT
#                 )
#             except Exception as err:
#                 success, message = False, str(err)
#             else:
#                 if values and self.sourceDf is not None:
#                     preview = self._predictors.add_constants_columns(
#                         self.sourceDf, values
#                     ).reset_index()
#                     preview = preview[['frame_i', 'Cell_ID', constant['name']]]
#                     self.testOutputDisplay.setHtml(preview.head(20).to_html(index=False))
#                 else:
#                     self.testOutputDisplay.setPlainText('Formula is valid; no data to preview.')
#         self.statusLabel.setText('Formula is valid.' if success else message)
#         if not success:
#             self.testOutputDisplay.setPlainText(message)
#         self.testOutputDisplay.setStyleSheet(
#             f'border: 2px solid {"green" if success else "red"}'
#         )
#         self.okButton.setEnabled(success)

#     def ok_cb(self):
#         self.test_cb()
#         if not self.okButton.isEnabled():
#             return
#         self.equation = self.formulaEditWidget.text()
#         self.cancel = False
#         self.close()

#     def showHelp(self):
#         msg = widgets.myMessageBox(wrapText=False)
#         msg.information(self, 'Constant formulas', html_utils.paragraph(
#             'Candidate inputs have their measurement names. Enquiring-cell '
#             'inputs have the prefix <code>this_</code>, for example '
#             '<code>area &gt; this_area</code>. Functions such as MAX and MEAN '
#             'operate on the candidate/enquirer overlapping frames when filtering. '
#             'Custom ranges use inclusive integer bounds relative to t, '
#             'the frame receiving the result (e.g. t - 5 to t + 2). '
#             'SizeT is the total number of video frames.'
#         ))


class stopFrameDialog(dialogs_base.QBaseDialog):
    def __init__(self, posDatas, parent=None):
        super().__init__(parent=parent)

        self.cancel = True

        self.setWindowTitle('Stop frame')

        mainLayout = QVBoxLayout()

        infoTxt = html_utils.paragraph(
            'Enter a <b>stop frame number</b> for each of the loaded Positions',
            center=True
        )
        exp_path = posDatas[0].exp_path
        exp_path = os.path.normpath(exp_path).split(os.sep)
        exp_path = f'...{f"{os.sep}".join(exp_path[-4:])}'
        subInfoTxt = html_utils.paragraph(
            f'Experiment folder: <code>{exp_path}<code>', font_size='12px',
            center=True
        )
        infoLabel = QLabel(f'{infoTxt}{subInfoTxt}')
        infoLabel.setToolTip(posDatas[0].exp_path)
        mainLayout.addWidget(infoLabel)
        mainLayout.addSpacing(20)

        self.posDatas = posDatas
        for posData in posDatas:
            _layout = QHBoxLayout()
            _layout.addStretch(1)
            _label = QLabel(html_utils.paragraph(f'{posData.pos_foldername}'))
            _layout.addWidget(_label)

            _spinBox = QSpinBox()
            _spinBox.setMaximum(214748364)
            _spinBox.setAlignment(Qt.AlignCenter)
            _spinBox.setFont(fonts.font)
            if posData.acdc_df is not None:
                _val = posData.acdc_df.index.get_level_values(0).max()+1
            else:
                _val = posData.readLastUsedStopFrameNumber()
            if _val is None:
                _val = posData.SizeT
            _spinBox.setValue(_val)

            posData.stopFrameSpinbox = _spinBox

            _layout.addWidget(_spinBox)

            viewButton = widgets.viewPushButton('Visualize...')
            viewButton.clicked.connect(
                partial(self.viewChannelData, posData, _spinBox)
            )
            _layout.addWidget(viewButton, alignment=Qt.AlignRight)

            _layout.addStretch(1)

            mainLayout.addLayout(_layout)

        buttonsLayout = QHBoxLayout()

        okButton = widgets.okPushButton(' Ok ')
        cancelButton = widgets.cancelPushButton(' Cancel ')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.setLayout(mainLayout)
    
    def viewChannelData(self, posData, spinBox):
        self.sender().setText('Loading...')
        QTimer.singleShot(
            200, partial(self._viewChannelData, posData, spinBox, self.sender())
        )

    def _viewChannelData(self, posData, spinBox, senderButton):      
        chNames = posData.chNames
        if len(chNames) > 1:
            ch_name_selector = prompts.select_channel_name(
                which_channel='segm', allow_abort=False
            )
            ch_name_selector.QtPrompt(
                self, chNames,'Select channel name to visualize: '
            )
            if ch_name_selector.was_aborted:
                return
            chName = ch_name_selector.channel_name
        else:
            chName = chNames[0]
        
        channel_file_path = load.get_filename_from_channel(
            posData.images_path, chName
        )
        posData.frame_i = 0
        posData.loadImgData(imgPath=channel_file_path)
        self.slideshowWin = dialogs_image_tools.imageViewer(
            posData=posData, spinBox=spinBox
        )
        self.slideshowWin.update_img()
        self.slideshowWin.show()
        senderButton.setText('Visualize...')

    def ok_cb(self):
        self.cancel = False
        for posData in self.posDatas:
            stopFrameNum = posData.stopFrameSpinbox.value()
            posData.stopFrameNum = stopFrameNum
        self.close()
