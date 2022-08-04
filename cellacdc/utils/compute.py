import os
import traceback
import logging

import pandas as pd

from tqdm import tqdm

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QStyle
)

from .. import (
    widgets, apps, workers, html_utils, myutils,
    gui, measurements,cca_functions, load, printl
)

cellacdc_path = os.path.dirname(os.path.abspath(apps.__file__))
temp_path = os.path.join(cellacdc_path, 'temp')
favourite_func_metrics_csv_path = os.path.join(
    temp_path, 'favourite_func_metrics.csv'
)

class computeMeasurmentsUtilWin(QDialog):
    def __init__(self, expPaths, app, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Compute measurements utility')

        self.parent = parent

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='utils.computeMeasurements'
        )
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        self.expPaths = expPaths
        self.app = app
        self.abort = False
        self.worker = None
        self.progressWin = None

        mainLayout = QVBoxLayout()

        infoLayout = QHBoxLayout()
        infoTxt = html_utils.paragraph(
            'Computing measurements routine running...'
        )

        iconLabel = QLabel(self)
        standardIcon = getattr(QStyle, 'SP_MessageBoxInformation')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        iconLabel.setPixmap(pixmap)

        infoLayout.addWidget(iconLabel)
        infoLayout.addWidget(QLabel(infoTxt))

        buttonsLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton('Abort')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)

        cancelButton.clicked.connect(self.abortCallback)

        mainLayout.addLayout(infoLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def showEvent(self, event):
        self.runWorker()

    def runWorker(self):
        self.gui = gui.guiWin(self.app, parent=self.parent)
        self.gui.logger = self.logger

        self.progressWin = apps.QDialogWorkerProgress(
            title='Computing measurements', parent=self,
            pbarDesc='Computing measurements...'
        )
        self.progressWin.sigClosed.connect(self.progressWinClosed)
        self.progressWin.show(self.app)

        self.thread = QThread()
        self.worker = workers.calcMetricsWorker(self)
        self.worker.moveToThread(self.thread)

        self.worker.signals.finished.connect(self.thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.worker.signals.finished.connect(self.workerFinished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.signals.progress.connect(self.workerProgress)
        self.worker.signals.critical.connect(self.workerCritical)
        self.worker.signals.sigSelectSegmFiles.connect(self.selectSegmFileLoadData)
        self.worker.signals.sigInitAddMetrics.connect(self.initAddMetricsWorker)
        self.worker.signals.sigPermissionError.connect(self.warnPermissionError)
        self.worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.signals.sigUpdatePbarDesc.connect(self.workerUpdatePbarDesc)
        self.worker.signals.sigComputeVolume.connect(self.computeVolumeRegionprop)
        self.worker.signals.sigAskStopFrame.connect(self.workerAskStopFrame)
        self.worker.signals.sigErrorsReport.connect(self.warnErrors)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def warnErrors(self, standardMetricsErrors, customMetricsErrors):
        if standardMetricsErrors:
            win = apps.ComputeMetricsErrorsDialog(
                standardMetricsErrors, self.logs_path, 
                log_type='standard_metrics', parent=self
            )
            win.exec_()
        elif customMetricsErrors:
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
            chName = chNames[0]
            filePath = myutils.getChannelFilePath(images_path, chName)
            _posData = load.loadData(filePath, chName)
            _posData.getBasenameAndChNames()
            segm_files = load.get_segm_files(_posData.images_path)
            _existingEndnames = load.get_existing_segm_endnames(
                _posData.basename, segm_files
            )
            existingSegmEndNames.update(_existingEndnames)

        if len(existingSegmEndNames) == 1:
            self.endFilenameSegm = list(existingSegmEndNames)[0]
            self.worker.waitCond.wakeAll()
            return

        win = apps.QDialogMultiSegmNpz(
            existingSegmEndNames, exp_path, parent=self
        )
        win.exec_()
        self.endFilenameSegm = win.selectedItemText
        self.worker.abort = win.cancel
        self.worker.waitCond.wakeAll()

    def initAddMetricsWorker(self, posData):
        # Set measurements
        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            favourite_funcs = df_favourite_funcs['favourite_func_name'].to_list()
        except Exception as e:
            favourite_funcs = None

        measurementsWin = apps.setMeasurementsDialog(
            posData.chNames, [], posData.SizeZ > 1,
            favourite_funcs=favourite_funcs, posData=posData,
            isSegm3D=posData.isSegm3D
        )
        printl('setMeasurementsDialog executing')
        measurementsWin.exec_()
        if measurementsWin.cancel:
            self.worker.abort = measurementsWin.cancel
            self.worker.waitCond.wakeAll()
            return

        self.gui.ch_names = posData.chNames
        self.gui.notLoadedChNames = []
        self.gui.setMetricsFunc()
        self.gui.setMetricsToSkip(measurementsWin)
        self.gui.mutex = self.worker.mutex
        self.gui.waitCond = self.worker.waitCond
        self.gui.saveWin = self.progressWin

        self.gui.saveDataWorker = gui.saveDataWorker(self.gui)

        self.gui.saveDataWorker.criticalPermissionError.connect(self.skipEvent)
        self.gui.saveDataWorker.askZsliceAbsent.connect(self.gui.zSliceAbsent)
        self.gui.saveDataWorker.customMetricsCritical.connect(
            self.addCombinedMetricsError
        )

        self.worker.waitCond.wakeAll()
    
    def addCombinedMetricsError(self, traceback_format, func_name):
        self.logger.info('')
        print('====================================')
        self.logger.info(traceback_format)
        print('====================================')
        self.worker.customMetricsErrors[func_name] = traceback_format

    def skipEvent(self, dummy):
        self.worker.waitCond.wakeAll()

    def computeVolumeRegionprop(self, end_frame_i, posData):
        if 'cell_vol_vox' not in self.gui.sizeMetricsToSave:
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
            for i, obj in enumerate(rp):
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

    def workerCritical(self, error):
        try:
            raise error
        except:
            traceback_str = traceback.format_exc()
            print('='*20)
            self.worker.logger.log(traceback_str)
            print('='*20)

    def workerFinished(self, worker):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
        if worker.abort:
            txt = 'Computing measurements ABORTED.'
            self.logger.info(txt)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.warning(self, 'Process aborted', html_utils.paragraph(txt))
        else:
            txt = 'Computing measurements completed.'
            self.logger.info(txt)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.information(self, 'Process completed', html_utils.paragraph(txt))

        self.worker = None
        self.progressWin = None
        self.close()

    def workerProgress(self, text, loggerLevel='INFO'):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)
