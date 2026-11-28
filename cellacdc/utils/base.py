import os
import sys
import traceback

from natsort import natsorted
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
import pprint

import pandas as pd

from tqdm import tqdm

from qtpy.QtCore import Signal, QThread
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QStyle, QApplication
)

from .. import (
    widgets, apps, workers, html_utils, myutils,
    load, printl, exception_handler
)
from qtpy.QtWidgets import (
    QWidget, QComboBox, QAbstractItemView, QListWidgetItem, QLayout
)
from cellacdc.shared.widgets import item_views as widgets_item_views
from cellacdc.shared.widgets import line_edits as widgets_line_edits
from cellacdc.shared.dialogs import base as dialogs_base
from qtpy.QtCore import QObject, QMutex, QWaitCondition
from cellacdc.shared.workers import base as workers_base

def log_init_util(logger, expPaths: dict, util_title, util_module):
    exp_paths_str = pprint.pformat(expPaths, indent=1)
    
    logger.info(f'Utility title: "{util_title}"')
    logger.info(f'Utility module: "{util_module}"')
    logger.info(f'Selected experiments:\n{exp_paths_str}')
    
    
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
        
        log_init_util(logger, expPaths, title, module)
        
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path
        self.is_error_state = False

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
        cancelButton = widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        cancelButton.hide()

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
        self.worker.signals.sigSelectSegmFiles.connect(
            self.selectSegmFileLoadData
        )
        self.worker.signals.sigSelectFilesWithText.connect(
            self.selectFileFromFilesWithText
        )
        self.worker.signals.sigSelectAcdcOutputFiles.connect(
            self.selectAcdcOutputTables
        )     
        self.worker.signals.sigSelectSpotmaxRun.connect(
            self.selectSpotmaxRun
        )  
        self.worker.signals.sigSelectFile.connect(
            self.selectFile
        )  
        self.worker.signals.sigSelectVideoFile.connect(
            self.selectVideoFile
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
    
    def selectVideoFile(self, exp_path, pos_foldernames, multiSelection: bool):
        valid_extensions = {'.tif', '.tiff', '.h5', '.npz'}
        all_video_files_endnames = set()
        for pos_folder in pos_foldernames:
            images_path = os.path.join(exp_path, pos_folder, 'Images')
            basename, chNames = myutils.getBasenameAndChNames(images_path)
            for file in myutils.listdir(images_path):
                filename, ext = os.path.splitext(file)
                if ext not in valid_extensions:
                    continue

                endname = file[len(basename):]
                all_video_files_endnames.add(endname)
        
        all_video_files_endnames = natsorted(all_video_files_endnames)

        win = widgets.QDialogListbox(
            'Select video file',
            'Select video file to load:\n',
            all_video_files_endnames, 
            multiSelection=multiSelection, 
            parent=self,
            allowEmptySelection=False
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        self.worker.selectedVideoEndname = win.selectedItemsText[0]
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

        if len(self.existingAcdcOutputEndnames) == 1:
            self.worker.abort = False
            self.selectedAcdcOutputEndnames = self.existingAcdcOutputEndnames
            self.worker.waitCond.wakeAll()
            return
        
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
        # Closing the window with the "X" button never sets selectedItemsText
        self.selectedAcdcOutputEndnames = (
            getattr(selectWindow, 'selectedItemsText', None) or []
        )
        self.worker.waitCond.wakeAll()

    def selectSpotmaxRun(
            self, exp_path, pos_foldernames, all_runs, infoText, 
            allowSingleSelection, multiSelection
        ):   
        items = natsorted([f'{run}_...{desc}' for run, desc in all_runs])
        if len(items) == 1:
            self.selectedSpotmaxRuns = items
            self.worker.waitCond.wakeAll()
            return
        
        selectWindow = widgets.QDialogListbox(
            'Select spotmax run(s)',
            f'Select one or more spotmax runs{infoText}\n',
            items, multiSelection=multiSelection, 
            parent=self, allowSingleSelection=allowSingleSelection
        )
        selectWindow.exec_()
        if selectWindow.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        self.selectedSpotmaxRuns = selectWindow.selectedItemsText
        self.worker.waitCond.wakeAll()
    
    def selectFile(self, start_dir, caption, filters):
        from qtpy.compat import getopenfilename
        filepath = getopenfilename(
            parent=self, 
            caption=caption,
            basedir=start_dir,
            filters=filters
        )[0]
        if not filepath:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        self.selectedFilepath = filepath
        self.worker.waitCond.wakeAll()
    
    def _selectFileFromFilesWithText(
            self, exp_path, pos_foldernames, with_text, ext
        ):
        # Get end name of every existing segmentation file
        existingEndNames = set()
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

            if with_text == 'segm':
                found_files = load.get_segm_files(_posData.images_path)
            else:
                found_files = load.get_files_with(
                    _posData.images_path, with_text, ext=ext
                )

            _existingEndnames = load.get_endnames(
                _posData.basename, found_files
            )
            existingEndNames.update(_existingEndnames)

        if len(existingEndNames) == 1:
            return existingEndNames, list(existingEndNames)[0], False

        if hasattr(self, 'infoText'):
            infoText = self.infoText
        else:
            infoText = None

        if with_text == 'segm':
            fileType = 'segmentation'
        elif with_text == 'imagej_rois':
            fileType = 'ImageJ ROIs'
        else:
            fileType = with_text.split('_')
    
        win = apps.SelectSegmFileDialog(
            existingEndNames, exp_path, parent=self, infoText=infoText, 
            fileType=fileType
        )
        win.exec_()
        return existingEndNames, win.selectedItemText, win.cancel
    
    def selectFileFromFilesWithText(
            self, exp_path, pos_foldernames, with_text, ext
        ):
        
        out = self._selectFileFromFilesWithText(
            exp_path, pos_foldernames, with_text, ext
        )
        existingEndNamesWithText, endFilenameWithText, cancel = out
        
        self.existingEndNamesWithText = list(existingEndNamesWithText)
        self.endFilenameWithText = endFilenameWithText        
        self.worker.abort = cancel
        self.worker.waitCond.wakeAll()
    
    def selectSegmFileLoadData(self, exp_path, pos_foldernames):
        out = self._selectFileFromFilesWithText(
            exp_path, pos_foldernames, 'segm', None
        )
        existingSegmEndNames, endFilenameSegm, cancel = out
        
        self.existingSegmEndNames = list(existingSegmEndNames)
        self.endFilenameSegm = endFilenameSegm
        
        self.worker.abort = cancel
        self.worker.waitCond.wakeAll()

    def skipEvent(self, dummy):
        self.worker.waitCond.wakeAll()

    def progressWinClosed(self, aborted):
        self.abort = aborted
        if aborted and self.worker is not None:
            self.worker.abort = True
            self.close()
        
    def closeEvent(self, event):
        if self._thread is None or not self._thread.isRunning():
            event.accept()
            return

        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'The process is <b>still running and cannot be safely stopped</b> at this point.'
        ) + html_utils.paragraph(
            'The only way to stop the process now is to close the terminal.<br><br>'
            'However, we do not recommend doing this unless you are happy to<br>'
            'interrupt the process and potentially lose the results that have<br>'
            'not been saved, yet.'
        ) + html_utils.paragraph(
            'Please <b>wait for the process to finish</b>.<br><br>Thank you for your patience!'
        )
        msg.warning(self, 'Process still running', txt)
        event.ignore()

    @exception_handler
    def workerCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True

        try:
            if isinstance(error, tuple):
                worker, error = error
            else:
                worker = None
            raise error
        except:
            print('='*20)
            if hasattr(self, 'worker'):
                self.worker.logger.log(traceback.format_exc())
            elif worker is not None and hasattr(worker, 'logger'):
                worker.logger.log(traceback.format_exc())
            elif hasattr(self, 'logger'):
                self.logger.log(traceback.format_exc())
            else:
                print(traceback.format_exc())
            print('='*20)
            result = _critical_exception_gui(self, f'{self._title} utility')
            # mutex and workerFinished handeling
            try:
                worker.workerAborted()
            except:
                pass

            try:
                worker.waitCond.wakeAll()
            except:
                pass

            try:
                self.worker.workerAborted()
            except:
                pass

            try:
                self.worker.waitCond.wakeAll()
            except:
                pass

            if self.progressWin is not None:
                self.progressWin.close()

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
        logger.info(f'Utility title: "{title}"')
        logger.info(f'Utility module: "{module}"')
        
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
        self._thread = None

        self.setLayout(mainLayout)
    
    def closeClicked(self):
        self.sigClose.emit()
    
    def closeEvent(self, event):
        self.logger.info('Closing logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
    
    def runWorker(self, worker):
        self.progressWin = apps.QDialogWorkerProgress(
            title=self.progressDialogueTitle, parent=self,
            pbarDesc=f'{self.progressDialogueTitle}...'
        )
        self.progressWin.sigClosed.connect(self.progressWinClosed)
        self.progressWin.show(self.app)

        self._thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.signals.finished.connect(self._thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.worker.signals.finished.connect(self.workerFinished)
        self._thread.finished.connect(self._thread.deleteLater)

        self.worker.signals.progress.connect(self.workerProgress)
        self.worker.signals.critical.connect(self.workerCritical)    
        self.worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.worker.signals.sigInitInnerPbar.connect(self.workerInitInnerPbar)
        self.worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.signals.sigUpdateInnerPbar.connect(
            self.workerUpdateInnerPbar
        )
        self.worker.signals.sigUpdatePbarDesc.connect(self.workerUpdatePbarDesc)

        self._thread.started.connect(self.worker.run)
        self._thread.start()
    
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


class OrderableListWidget(QWidget):
    sigEnterEvent = Signal(object)
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labels = []
    
    def setParentItem(self, item):
        self._item = item
    
    def setLabelsColor(self, selected):
        if selected:
            stylesheet = 'color : black'
        else:
            stylesheet = ''
            
        for label in self._labels:
            label.setStyleSheet(stylesheet)
    
    def enterEvent(self, event):
        super().enterEvent(event)
        self.setLabelsColor(True)
        self.sigEnterEvent.emit(self._item)
    
    # def leaveEvent(self, event):
    #     super().leaveEvent(event)
    #     self.setLabelsColor(self._item.isSelected())
    #     printl('leave', self._item.isSelected())
    
    def addLabel(self, label):
        self._labels.append(label)

class OrderableList(widgets_item_views.listWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.itemEntered.connect(self.onItemEntered)
    
    def onItemEntered(self, enteredItem):
        enteredRow = self.row(enteredItem)
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(i == enteredRow or item.isSelected())
    
    def leaveEvent(self, event):
        super().leaveEvent(event)
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(item.isSelected())
    
    def addItems(self, items):
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        nr_items = len(items)
        nn = [str(n) for n in range(1, nr_items+1)]
        for i, item in enumerate(items):
            itemW = QListWidgetItem()
            itemContainer = OrderableListWidget()
            itemContainer.setParentItem(itemW)
            itemText = QLabel(item)
            tableNrLabel = QLabel('| Table nr.')
            itemContainer.addLabel(tableNrLabel)
            itemContainer.addLabel(itemText)
            itemLayout = QHBoxLayout()
            itemNumberWidget = QComboBox()
            itemNumberWidget.addItems(nn)
            itemLayout.addWidget(itemText)
            itemLayout.addWidget(tableNrLabel)
            itemLayout.addWidget(itemNumberWidget)
            itemContainer.setLayout(itemLayout)
            itemLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
            itemW.setSizeHint(itemContainer.sizeHint())
            self.addItem(itemW)
            self.setItemWidget(itemW, itemContainer)
            itemW._text = item
            itemW._nrWidget = itemNumberWidget
            itemW._container = itemContainer
            itemNumberWidget.setDisabled(True)
            itemNumberWidget.textActivated.connect(self.onTextActivated)
            itemNumberWidget._currentNr = 1
            itemNumberWidget.row = i
            itemContainer.sigEnterEvent.connect(self.onItemEntered)
        
        self.itemSelectionChanged.connect(self.onItemSelectionChanged)
        
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.clearSelection()
            event.ignore()
            return
        super().keyPressEvent(event)
    
    def updateNr(self):
        for i in range(self.count()):
            item = self.item(i)
            item._currentNr = int(item._nrWidget.currentText())
    
    def onItemSelectionChanged(self):
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(item.isSelected())
            item._nrWidget.setDisabled(not item.isSelected())
            if item._nrWidget.currentText() != '1':
                item._nrWidget.setCurrentText('1')
                item._currentNr = 1
        
        for i, item in enumerate(self.selectedItems()):
            item._nrWidget.setCurrentText(f'{i+1}')
            item._currentNr = i+1
        
    def onTextActivated(self, text):
        changedNr = self.sender()._currentNr
        for item in self.selectedItems():
            row = self.row(item)
            if self.sender().row == row:
                changedNr = item._currentNr
                continue
        
        for item in self.selectedItems():
            row = self.row(item)
            if self.sender().row == row:
                continue
            nr = int(item._nrWidget.currentText())
            if nr == int(text):
                item._nrWidget.setCurrentText(str(changedNr))
                break
        
        self.updateNr()


class PrefixFilenameLineEdit(QWidget):
    def __init__(self, endNameLabel, *args):      
        super().__init__(*args)

        layout = QHBoxLayout()

        self.le = widgets_line_edits.alphaNumericLineEdit()
        self.endnameLabel = QLabel()
        self.setEndname(endNameLabel)

        layout.addWidget(self.le)
        layout.addWidget(self.endnameLabel)

        layout.setStretch(0, 1)
        layout.setStretch(1, 0)

        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)
    
    def setAlignment(self, alignment):
        self.le.setAlignment(alignment)

    def setEndname(self, text: str):
        self.endnameLabel.setText(text)
    
    def setText(self, text: str):
        self.le.setText(text)
    
    def prefix(self):
        return self.le.text()
    
    def fullFilename(self):
        filename = f'{self.le.text()}{self.endnameLabel.text()}'
        return filename


class OrderableListWidgetDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, items, title='Select items', infoTxt='', helpText='', 
            parent=None
        ):
        super().__init__(parent)

        self.selectedItemsText = []

        self.cancel = True
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        self.helpText = helpText

        if infoTxt:
            mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))

        self.listWidget = widgets.OrderableList()
        self.listWidget.addItems(items)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        if helpText:
            helpButton = widgets.helpPushButton('Help...')
            buttonsLayout.insertWidget(3, helpButton)
            helpButton.clicked.connect(self.showHelp)

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(self.listWidget)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
    
    def showHelp(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(self.helpText)
        msg.information(self, 'Select tables help', txt)
    
    def ok_cb(self):
        self.cancel = False
        self.selectedItemsText = [None]*len(self.listWidget.selectedItems())
        for itemW in self.listWidget.selectedItems():
            idx = int(itemW._nrWidget.currentText()) - 1
            if idx >= len(self.selectedItemsText):
                idx = len(self.selectedItemsText) - 1
            self.selectedItemsText[idx] = itemW._text
        self.close()


class SetupSplitVideoIntoTiffsDialog(dialogs_base.QBaseDialog):
    def __init__(self, video_filepath, logger_func=print, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Split video into single-frame TIFFs setup')

        self.logger_func = logger_func
        self.cancel = True

        mainLayout = QVBoxLayout()
        entriesLayout = widgets.FormLayout()

        headerText = ("""
<b>Split video into single-frame TIFFs setup</b><br><br>
<i>Choose destination folder and conversion settings.</i>                     
""")
        headerLabel = QLabel(html_utils.paragraph(headerText))
        self._headerLabel = headerLabel
        self._headerText = headerText

        row = 0
        self.dstFolderPathControl = widgets.FolderPathControl()
        self.dstFolderPathFormWidget = widgets.formWidget(
            self.dstFolderPathControl, 
            labelTextLeft='Destination folder: ',
        )
        entriesLayout.addFormWidget(self.dstFolderPathFormWidget, row=row)

        row += 1
        self.prefixLineEdit = widgets.PrefixFilenameLineEdit('000.tif')
        self.prefixLineEdit.setAlignment(Qt.AlignCenter)
        self.prefixFormWidget = widgets.formWidget(
            self.prefixLineEdit, 
            labelTextLeft='Prefix for output TIFF files: ',
        )
        entriesLayout.addFormWidget(self.prefixFormWidget, row=row)

        row += 1
        dtypes = (
            'uint8',
            'uint16',
            'uint32',
            'float32',
            'float64'
        )
        _, ext = os.path.splitext(video_filepath)
        image_data = load.load_image_file(video_filepath)
        self._SizeT = len(image_data)
        self.dtypeCombobox = widgets.ComboBox()
        self.dtypeCombobox.addItems(dtypes)
        if ext != '.npz':
            dtype = str(image_data.dtype)
            if dtype not in dtypes:
                self.dtypeCombobox.setCurrentText('float64')
            else:
                self.dtypeCombobox.setCurrentText(dtype)
        else:
            dtype = 'uint16'
            self.dtypeCombobox.setCurrentText(dtype)
        
        self.dtypeFormWidget = widgets.formWidget(
            self.dtypeCombobox, 
            stretchWidget=False,
            labelTextLeft='Data type of output TIFF files: ',
        )
        entriesLayout.addFormWidget(self.dtypeFormWidget, row=row)
        
        row += 1
        infoText = html_utils.paragraph(f"""
Choose whether to stop until the last frame that has been tracked.<br><br>
The last tracked frame is the maximum `frame_i` present in the selected 
`acdc_output.csv` file (see parameter below).
""")
        self.onlyUntilTrackedFormWidget = widgets.formWidget(
            widgets.Toggle(), 
            labelTextLeft='Only until tracked: ',
            stretchWidget=False,
            valueGetterName='isChecked',
            addInfoButton=True,
            infoTxt=infoText
        )
        entriesLayout.addFormWidget(self.onlyUntilTrackedFormWidget, row=row)
        self.onlyUntilTrackedFormWidget.widget.toggled.connect(
            self.onlyUntilTrackedToggled
        )

        row += 1
        infoText = html_utils.paragraph(f"""
Choose whether to stop until the last frame where the cell cycle/lineage has been annotated.<br><br>
The last annotated frame is the maximum `frame_i` present in the "cell_cycle_stage" column in the selected 
`acdc_output.csv` file (see parameter below).
""")
        self.onlyUntilAnnotatedFormWidget = widgets.formWidget(
            widgets.Toggle(), 
            labelTextLeft='Only until lineage annotated: ',
            stretchWidget=False,
            valueGetterName='isChecked',
            addInfoButton=True,
            infoTxt=infoText
        )
        entriesLayout.addFormWidget(self.onlyUntilAnnotatedFormWidget, row=row)
        self.onlyUntilAnnotatedFormWidget.widget.toggled.connect(
            self.onlyUntilTrackedToggled
        )

        row += 1
        images_path = os.path.dirname(video_filepath)
        self._images_path = images_path
        acdcOutputFiles = load.get_acdc_output_files(images_path)
        basename, chNames = myutils.getBasenameAndChNames(images_path)
        acdcOutputEndnames = [f[len(basename):] for f in acdcOutputFiles]
        self.acdcOutputEndnamesCombobox = widgets.ComboBox()
        self.acdcOutputEndnamesCombobox.addItems(acdcOutputEndnames)
        self.acdcOutputEndnamesFormWidget = widgets.formWidget(
            self.acdcOutputEndnamesCombobox, 
            stretchWidget=False,
            labelTextLeft='`acdc_output` file with tracking/lineage info: ',
        )
        self.acdcOutputEndnamesFormWidget.setDisabled(True)
        entriesLayout.addFormWidget(self.acdcOutputEndnamesFormWidget, row=row)
        self.dtypeCombobox.setFixedWidth(
            self.acdcOutputEndnamesCombobox.sizeHint().width()
        )
        self.acdcOutputEndnamesCombobox.currentTextChanged.connect(
            self.acdcOutputEndnameChanged
        )

        row += 1
        infoText = html_utils.paragraph(f"""
            Enter number of frames to split.<br><br>
            This number will overwrite the number of frames calculated with 
            "Only until tracked" and "Only until annotated" options.
        """)
        self.untilFrameNumberFormWidget = widgets.formWidget(
            widgets.SpinBox(), 
            labelTextLeft='Number of frames to save: ',
            addInfoButton=True,
            infoTxt=infoText,
            addActivateCheckbox=True,
        )
        self.untilFrameNumberFormWidget.widget.setMinimum(1)
        self.untilFrameNumberFormWidget.widget.setMaximum(self._SizeT)
        self.untilFrameNumberFormWidget.widget.setValue(self._SizeT)
        entriesLayout.addFormWidget(self.untilFrameNumberFormWidget, row=row)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(headerLabel)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(entriesLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def onlyUntilTrackedToggled(self, checked):
        disabled = (
            not self.onlyUntilTrackedFormWidget.widget.isChecked()
            and not self.onlyUntilAnnotatedFormWidget.widget.isChecked()
        )
        self.acdcOutputEndnamesFormWidget.setDisabled(disabled)
        if disabled:
            self.untilFrameNumberFormWidget.widget.setValue(self._SizeT)
            return

        self.acdcOutputEndnameChanged(
            self.acdcOutputEndnamesCombobox.currentText()
        )

    def acdcOutputEndnameChanged(self, acdcOutputEndname):
        acdcOutputEndname = self.acdcOutputEndnamesCombobox.currentText()
        acdc_df = load.load_acdc_df_file(
            self._images_path, 
            end_name_acdc_df_file=acdcOutputEndname
        )
        if self.onlyUntilTrackedFormWidget.widget.isChecked():
            numFrames = acdc_df['frame_i'].max() + 1
        else:
            try:
                ccs = acdc_df[['cell_cycle_stage']]
            except KeyError:
                ccs = acdc_df[['generation_num_tree']]
            last_index_cca_df = ccs.last_valid_index()
            numFrames = (
                acdc_df.loc[:last_index_cca_df, 'frame_i'].max() + 1
            )

        self.untilFrameNumberFormWidget.widget.setValue(numFrames)

    def sizeHint(self):
        height = super().sizeHint().height()
        width = super().sizeHint().width()
        return QSize(round(width*1.5), height)

    def warnDstFolderPathNotSelected(self):
        txt = html_utils.paragraph(f"""
            The destination folder path is empty.<br><br>
            Please select a destination folder for the output files.<br><br>
            Thank you for your patience!
        """)

        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Invalid destination folder', txt)

    def ok_cb(self):
        if not self.dstFolderPathFormWidget.widget.path():
            self.warnDstFolderPathNotSelected()
            return
        
        self.dstFolderPath = self.dstFolderPathFormWidget.widget.path()
        self.prefixText = self.prefixLineEdit.prefix()
        self.dtypeOut = self.dtypeCombobox.currentText()
        self.onlyUntilTracked = (
            self.onlyUntilTrackedFormWidget.widget.isChecked()
        )
        self.onlyUntilAnnotated = (
            self.onlyUntilAnnotatedFormWidget.widget.isChecked()
        )
        self.acdcOutputEndname = self.acdcOutputEndnamesCombobox.currentText()
        numFramesSpinbox = self.untilFrameNumberFormWidget.widget
        self.numFramesToSplit = (
            numFramesSpinbox.value()
            if self.untilFrameNumberFormWidget.activateCheckbox.isChecked()
            else None
        )
        
        self.cancel = False
        self.close()


class BaseWorkerUtil(QObject):
    progressBar = Signal(int, int, float)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.abort = False
        self.skipExp = False
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.signals.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def emitSelectVideoFile(self, exp_path, pos_foldernames, multiSelection):
        self.mutex.lock()
        self.signals.sigSelectVideoFile.emit(
            exp_path, pos_foldernames, multiSelection
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitSelectFilesWithText(
            self, exp_path, pos_foldernames, with_text, ext=None
        ):
        self.mutex.lock()
        self.signals.sigSelectFilesWithText.emit(
            exp_path, pos_foldernames, with_text, ext
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def emitSelectFile(self, start_dir, caption='', filters='All files (*.)'):
        self.mutex.lock()
        self.signals.sigSelectFile.emit(start_dir, caption, filters)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
        
    def emitSelectAcdcOutputFiles(
            self, exp_path, pos_foldernames, infoText='', 
            allowSingleSelection=False, multiSelection=True
        ):
        self.mutex.lock()
        self.signals.sigSelectAcdcOutputFiles.emit(
            exp_path, pos_foldernames, infoText, allowSingleSelection,
            multiSelection
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitSelectSpotmaxRun(
            self, exp_path, pos_foldernames, all_runs, infoText='', 
            allowSingleSelection=True, multiSelection=True
        ):
        self.mutex.lock()
        self.signals.sigSelectSpotmaxRun.emit(
            exp_path, pos_foldernames, all_runs, infoText, allowSingleSelection,
            multiSelection
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
