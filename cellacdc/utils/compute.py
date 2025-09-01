import os
import traceback
import logging
from functools import partial
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
            self.worker.signals.sigSelectSegmFiles.connect(
                partial(self.setStopFrame, stopFrameNumber=stopFrameNumber)
            )
        self.worker.signals.sigErrorsReport.connect(self.warnErrors)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def askRunNowOrSaveToConfig(self, worker):
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
            parent=self, 
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
        
        self.worker.waitCond.wakeAll()
    
    def setStopFrame(self, posDatas, stopFrameNumber=1):
        for posData in self.posDatas:
            posData.stopFrameNum = stopFrameNumber
    
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
        self.combineMetricWindow = apps.combineMetricsEquationDialog(
            self.posData.chNames, isZstack, self.posData.isSegm3D,
            parent=self.measurementsWin, closeOnOk=False
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

    def initAddMetricsWorker(self, posData, allPosDataInputs):
        # Set measurements
        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            favourite_funcs = df_favourite_funcs['favourite_func_name'].to_list()
        except Exception as e:
            favourite_funcs = None

        self.posData = posData
        self.allPosDataInputs = allPosDataInputs

        if not hasattr(self, 'allPosData'):
            self.allPosData = []
            for p, posDataInputs in enumerate(self.allPosDataInputs):
                combineMetricsConfig = posDataInputs['combineMetricsConfig']
                combineMetricsPath = posDataInputs['combineMetricsPath']

                # Here we build a placeholder loadData class but we get what is 
                # needed to save custom combine metrics from posDataInputs
                _posData = load.loadData(
                    self.posData.imgPath, self.posData.user_ch_name
                )
                _posData.combineMetricsConfig = combineMetricsConfig
                _posData.custom_combine_metrics_path = combineMetricsPath
                self.allPosData.append(_posData)

        self.measurementsWin = apps.SetMeasurementsDialog(
            posData.chNames, [], posData.SizeZ > 1, posData.isSegm3D,
            favourite_funcs=favourite_funcs, posData=posData,
            addCombineMetricCallback=self.addCombineMetric,
            allPosData=self.allPosData
        )
        self.measurementsWin.sigClosed.connect(self.startSaveDataWorker)
        self.measurementsWin.sigCancel.connect(self.abortWorkerMeasurementsWin)
        self.measurementsWin.show()
    
    def abortWorkerMeasurementsWin(self):
        self.worker.abort = self.measurementsWin.cancel
        self.worker.waitCond.wakeAll()
        self.cancel = True

    def startSaveDataWorker(self):
        self.worker.kernel.init_args(
            self.posData.chNames, self.endFilenameSegm
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
        if 'cell_vol_vox' not in self.worker.kernel.sizeMetricsToSave:
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

    def abortCallback(self):
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
            txt = 'Computing measurements ABORTED.'
            self.logger.info(txt)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.warning(self, 'Process aborted', html_utils.paragraph(txt))
        
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
        self.logger.log(getattr(logging, loggerLevel.upper()), text)
