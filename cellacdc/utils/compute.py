import os
import traceback
import logging

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QStyle
)

from .. import widgets, apps, workers, html_utils, myutils

class computeMeasurmentsUtilWin(QDialog):
    def __init__(self, expPaths, app, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Compute measurements utility')

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='utils.computeMeasurements'
        )
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

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

        self.runWorker()

    def runWorker(self):
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

        self.worker.signals.critical.connect(self.workerCritical)
        self.worker.signals.sigInitLoadData.connect(self.initWorkerLoadData)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def initWorkerLoadData(posData):
        segm_files = posData.detectMultiSegmNpz()
        if len(segm_files)==1:
            segmFilename = segm_files[0]
            self.endFilenameSegm = segmFilename[len(posData.basename):]
            self.worker.waitCond.wakeAll()
            return

        win = apps.QDialogMultiSegmNpz(
            segm_files, posData.images_path, parent=self
        )
        win.exec_()
        self.endFilenameSegm = win.selectedItemText[len(posData.basename):]
        self.worker.abort = win.cancel
        self.worker.waitCond.wakeAll()

    def progressWinClosed(self, aborted):
        self.abort = aborted
        if aborted and self.worker is not None:
            self.worker.abort = True

    def abortCallback(self):
        self.abort = True
        if self.worker is not None:
            self.worker.abort = True
        elif self.progressWin is None:
            self.close()

    def workerCritical(self, error):
        try:
            raise e
        except:
            traceback_str = traceback.format_exc()
            self.worker.log(traceback_str)
            print('='*20)
            self.logger.error(traceback_str)
            print('='*20)

    def workerFinished(self, worker):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
        if worker.abort:
            self.logger.info('Computing measurements ABORTED.')
        else:
            self.logger.info('Computing measurements completed.')

        self.worker = None
        self.progressWin = None
        self.close()

    def workerProgress(self, text, loggerLevel='INFO'):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)
