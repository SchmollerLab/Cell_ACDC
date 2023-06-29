import os
import sys
import traceback

from qtpy.QtCore import Qt, QThread, QSize
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel
)
from qtpy import QtGui

from .. import exception_handler, myutils, html_utils, workers, widgets
from .. import _critical_exception_gui

import os
import traceback
import logging

import pandas as pd

from tqdm import tqdm

from qtpy.QtCore import Signal, QThread
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QStyle, QApplication
)

from .. import (
    widgets, apps, workers, html_utils, myutils,
    gui, load, printl, exception_handler
)

class NewThreadMultipleExpBaseUtil(QDialog):
    def __init__(
            self, expPaths, app: QApplication, title: str, module: str, 
            infoText: str, progressDialogueTitle: str, parent=None
        ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._title = title

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
        self.worker.signals.sigSelectAcdcOutputFiles.connect(
            self.selectAcdcOutputTables
        )        
        self.worker.signals.sigPermissionError.connect(self.warnPermissionError)
        self.worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.worker.signals.sigInitInnerPbar.connect(self.workerInitInnerPbar)
        self.worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.signals.sigUpdateInnerPbar.connect(
            self.workerUpdateInnerPbar
        )
        self.worker.signals.sigUpdatePbarDesc.connect(self.workerUpdatePbarDesc)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def workerInitInnerPbar(self, totalIter):
        if totalIter <= 1:
            self.progressWin.innerPbar.hide()
            return
        self.progressWin.innerPbar.show()
        self.progressWin.innerPbar.setValue(0)
        self.progressWin.innerPbar.setMaximum(totalIter)

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)
    
    def workerUpdateInnerPbar(self, step):
        self.progressWin.innerPbar.update(step)
    
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
    
    def selectAcdcOutputTables(
            self, exp_path, pos_foldernames, infoText, allowSingleSelection,
            multiSelection
        ):
        existingAcdcOutputEndnames = set()
        for p, pos in enumerate(pos_foldernames):
            pos_path = os.path.join(exp_path, pos)
            images_path = os.path.join(pos_path, 'Images') 
            basename, chNames = myutils.getBasenameAndChNames(images_path)
            # Use first found channel, it doesn't matter for basename
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
            if p == 0:
                self.basename_pos1 = _posData.basename
            acdc_output_files = load.get_acdc_output_files(_posData.images_path)
            acdc_output_endnames = load.get_endnames_from_basename(
                _posData.basename, acdc_output_files
            )
            existingAcdcOutputEndnames.update(acdc_output_endnames)
        
        self.existingAcdcOutputEndnames = list(existingAcdcOutputEndnames)

        if multiSelection:
            selectWindow = apps.OrderableListWidgetDialog(
            self.existingAcdcOutputEndnames, 
            title='Select acdc_output files',
            infoTxt=(
                'Select acdc_output tables and choose a table number (optional)<br><br>'
                '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
                '<code>Shift+Click</code> <i>to select a range of items</i><br>'
            ),
            helpText=(
                'The table number is useful to ensure that you can load the '
                'same exact equations you used in a previous sessions.<br><br>'
                'Cell-ACDC will automatically save the equations you enter. '
                'They will be saved in a file ending with '
                '<code>_equations_appended_name.ini</code><br> and each table will '
                'be numbered with the number you enter now.<br><br>'
                'When you reopen the equations dialogue you can select to load '
                'equations from a saved .ini file, however, <br><b>only the equations that '
                'used the table ending with the same name you select now<br>'
                'AND same number can be loaded</b>.'
            )
        )
        else:
            selectWindow = widgets.QDialogListbox(
                'Select acdc_output files',
                f'Select acdc_output files{infoText}\n',
                self.existingAcdcOutputEndnames, multiSelection=multiSelection, 
                parent=self, allowSingleSelection=allowSingleSelection
            )
        selectWindow.exec_()
        self.worker.abort = selectWindow.cancel
        self.selectedAcdcOutputEndnames = selectWindow.selectedItemsText
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
            _existingEndnames = load.get_existing_segm_endnames(
                _posData.basename, segm_files
            )
            existingSegmEndNames.update(_existingEndnames)

        self.existingSegmEndNames = list(existingSegmEndNames)

        if len(existingSegmEndNames) == 1:
            self.endFilenameSegm = list(existingSegmEndNames)[0]
            self.worker.waitCond.wakeAll()
            return

        if hasattr(self, 'infoText'):
            infoText = self.infoText
        else:
            infoText = None

        win = apps.QDialogMultiSegmNpz(
            existingSegmEndNames, exp_path, parent=self, infoText=infoText
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

    @exception_handler
    def workerCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True

        try:
            raise error
        except:
            print('='*20)
            self.worker.logger.log(traceback.format_exc())
            print('='*20)
            result = _critical_exception_gui(self, f'{self._title} utility')

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
    
    def closeEvent(self, event):
        self.logger.info('Closing logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

class MainThreadSinglePosUtilBase(QDialog):
    sigClose = Signal()

    def __init__(
            self, app: QApplication, title: str, module: str, infoText: str, 
            parent=None
        ):
        super().__init__(parent)
        self.setWindowTitle(title)

        self.progressDialogueTitle = title 

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

        self.worker = None

        self.setLayout(mainLayout)
    
    def closeClicked(self):
        self.sigClose.emit()
    
    def closeEvent(self, event):
        self.logger.info('Closing logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
    
    def selectSegmFileLoadData(self, posPath):
        # Get end name of every existing segmentation file
        existingSegmEndNames = set()
        pos_path = posPath
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
        _existingEndnames = load.get_existing_segm_endnames(
            _posData.basename, segm_files
        )
        existingSegmEndNames.update(_existingEndnames)

        self.existingSegmEndNames = list(existingSegmEndNames)

        if len(existingSegmEndNames) == 1:
            self.endFilenameSegm = list(existingSegmEndNames)[0]
            if self.worker is not None:
                self.worker.waitCond.wakeAll()
            return self.endFilenameSegm

        if hasattr(self, 'infoText'):
            infoText = self.infoText
        else:
            infoText = None

        win = apps.QDialogMultiSegmNpz(
            existingSegmEndNames, posPath, parent=self, infoText=infoText
        )
        win.exec_()
        self.endFilenameSegm = win.selectedItemText
        if self.worker is not None:
            self.worker.abort = win.cancel
            self.worker.waitCond.wakeAll()
        elif win.cancel:
            return ''
        else:
            return self.endFilenameSegm
    
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
        self.worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.worker.signals.sigInitInnerPbar.connect(self.workerInitInnerPbar)
        self.worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.signals.sigUpdateInnerPbar.connect(
            self.workerUpdateInnerPbar
        )
        self.worker.signals.sigUpdatePbarDesc.connect(self.workerUpdatePbarDesc)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def workerCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True

        try:
            raise error
        except:
            self.traceback_str = traceback.format_exc()
            print('='*20)
            self.worker.logger.log(self.traceback_str)
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
    
    def workerInitInnerPbar(self, totalIter):
        if totalIter <= 1:
            self.progressWin.innerPbar.hide()
            return
        self.progressWin.innerPbar.show()
        self.progressWin.innerPbar.setValue(0)
        self.progressWin.innerPbar.setMaximum(totalIter)

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)
    
    def workerUpdateInnerPbar(self, step):
        self.progressWin.innerPbar.update(step)
    
    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)
    
    def workerUpdatePbarDesc(self, desc):
        self.progressWin.progressLabel.setText(desc)
    
    def progressWinClosed(self, aborted):
        self.abort = aborted
        if aborted and self.worker is not None:
            self.worker.abort = True
            self.close()