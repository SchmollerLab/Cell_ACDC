import os
import sys
import traceback

from PyQt5.QtCore import Qt, QThread, QSize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel
)
from PyQt5 import QtGui

from .. import myutils, html_utils, workers, widgets

import os
import traceback
import logging

import pandas as pd

from tqdm import tqdm

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QStyle, QApplication
)

from .. import (
    widgets, apps, workers, html_utils, myutils,
    gui, load, printl
)

class NewThreadMultipleExpBaseUtil(QDialog):
    def __init__(
            self, expPaths, app: QApplication, title: str, module: str, 
            infoText: str, progressDialogueTitle: str, parent=None
        ):
        super().__init__(parent)
        self.setWindowTitle(title)

        self._parent = parent
        self.progressDialogueTitle = progressDialogueTitle 

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module=module
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
        infoTxt = html_utils.paragraph(infoText)

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

    def runWorker(self, worker):
        self.progressWin = apps.QDialogWorkerProgress(
            title=self.progressDialogueTitle, parent=self,
            pbarDesc=f'{self.progressDialogueTitle}...'
        )
        self.progressWin.sigClosed.connect(self.progressWinClosed)
        self.progressWin.show(self.app)

        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.signals.finished.connect(self.thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.worker.signals.finished.connect(self.workerFinished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.signals.progress.connect(self.workerProgress)
        self.worker.signals.critical.connect(self.workerCritical)
        self.worker.signals.sigSelectSegmFiles.connect(self.selectSegmFileLoadData)
        self.worker.signals.sigPermissionError.connect(self.warnPermissionError)
        self.worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.signals.sigUpdatePbarDesc.connect(self.workerUpdatePbarDesc)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

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

    def skipEvent(self, dummy):
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

        self.worker = None
        self.progressWin = None

    def workerProgress(self, text, loggerLevel='INFO'):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

class MainThreadSinglePosUtilBase(QDialog):
    sigClose = pyqtSignal()

    def __init__(
            self, app: QApplication, title: str, module: str, infoText: str, 
            parent=None
        ):
        super().__init__(parent)
        self.setWindowTitle(title)

        self._parent = parent

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module=module
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
        infoTxt = html_utils.paragraph(infoText)

        iconLabel = QLabel(self)
        standardIcon = getattr(QStyle, 'SP_MessageBoxInformation')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        iconLabel.setPixmap(pixmap)

        infoLayout.addWidget(iconLabel)
        infoLayout.addWidget(QLabel(infoTxt))

        buttonsLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton('Close')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)

        cancelButton.clicked.connect(self.closeClicked)

        mainLayout.addLayout(infoLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
    
    def closeClicked(self):
        self.sigClose.emit()