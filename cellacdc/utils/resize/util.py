from ... import myutils, workers, widgets, html_utils
from ... import apps

from ..base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QVBoxLayout, QGridLayout, QRadioButton
from cellacdc.shared.dialogs import base as dialogs_base
import os
from qtpy.QtCore import Signal
from .. import resize
from .. import base as utils_base
from cellacdc.shared.workers import base as workers_base

class ResizePositionsUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
        self._parent = parent
    
    def runWorker(self):
        self.worker = workers.ResizeUtilWorker(self)
        self.worker.sigSetResizeProps.connect(self.setResizeProps)
        super().runWorker(self.worker)
    
    def setResizeProps(self, input_path):
        win = apps.ResizeUtilProps(input_path=input_path, parent=self._parent)
        win.exec_()
        self.worker.abort = win.cancel
        if win.cancel:
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.resizeFactor = win.resizeFactor
        self.worker.textToAppend = win.textToAppend
        self.worker.expFolderpathOut = win.expFolderpathOut
        self.worker.waitCond.wakeAll()
    
    def showEvent(self, event):
        self.runWorker()
    
    def workerFinished(self, worker):
        super().workerFinished(worker)
        txt = 'Resizing data process completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Process completed', html_utils.paragraph(txt))
        self.close()


class ResizeUtilProps(dialogs_base.QBaseDialog):
    def __init__(self, input_path='', parent=None):        
        self.cancel = True
        super().__init__(parent)
        
        self.setWindowTitle('Resize Data Properties')
        
        mainLayout = QVBoxLayout()
        
        paramsLayout = QGridLayout()
        
        self._input_path = input_path
        
        row = 0
        paramsLayout.addWidget(QLabel('Overwrite raw data: '), row, 0)
        self.overwriteToggle = widgets.Toggle()
        self.overwriteToggle.setChecked(True)
        paramsLayout.addWidget(
            self.overwriteToggle, row, 1, 1, 2, alignment=Qt.AlignCenter
        )
        
        row += 1
        paramsLayout.addWidget(
            QLabel('Folder path for resized images: '), row, 0
        )
        self.folderPathOutControl = widgets.filePathControl(
            browseFolder=True, 
            fileManagerTitle='Select folder where to save resized data', 
            elide=True, 
            startFolder=myutils.getMostRecentPath()
        )
        self.folderPathOutControl.setDisabled(True)
        paramsLayout.addWidget(self.folderPathOutControl, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(QLabel('Text to append to files: '), row, 0)
        self.textToAppendLineEdit = widgets.alphaNumericLineEdit()
        self.textToAppendLineEdit.setAlignment(Qt.AlignCenter)
        self.textToAppendLineEdit.setDisabled(True)
        paramsLayout.addWidget(self.textToAppendLineEdit, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(QLabel('Resize mode: '), row, 0)
        self.downScaleRadioButton = QRadioButton('Downscale')
        self.upScaleRadioButton = QRadioButton('Upscale')
        self.downScaleRadioButton.setChecked(True)
        paramsLayout.addWidget(
            self.downScaleRadioButton, row, 1, alignment=Qt.AlignCenter
        )
        paramsLayout.addWidget(
            self.upScaleRadioButton, row, 2, alignment=Qt.AlignCenter
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Resize factor: '), row, 0)
        self.factorSpinbox = widgets.FloatLineEdit(allowNegative=False)
        self.factorSpinbox.setMinimum(1.0)
        self.factorSpinbox.setValue(2.0)
        paramsLayout.addWidget(self.factorSpinbox, row, 1, 1, 2)
        
        paramsLayout.setColumnStretch(0, 0)
        paramsLayout.setVerticalSpacing(10)
        
        self.overwriteToggle.toggled.connect(self.overwriteToggled)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(paramsLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)
        
        # self.textToAppendLineEdit.setText(self._getDefaultTextToAppend())
        
        self.setLayout(mainLayout)
    
    def _getDefaultTextToAppend(self):
        rescale_mode = 'up' if self.upScaleRadioButton.isChecked() else 'down'
        factor = self.factorSpinbox.value()
        text = f'{rescale_mode}scaled_factor_{factor}'
        return text
    
    def overwriteToggled(self, checked):
        self.folderPathOutControl.setDisabled(checked)
        self.textToAppendLineEdit.setDisabled(checked)
        if checked:
            text = ''
        else:
            text = self._getDefaultTextToAppend()
        self.textToAppendLineEdit.setText(text)
    
    def warnFolderPathEmpty(self):
        txt = html_utils.paragraph("""
            To prevent overwriting raw data the <code>Folder path for 
            resized images</code> <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty folder path', txt)
    
    def warnTextToAppendEmpty(self):
        txt = html_utils.paragraph("""
            To prevent overwriting raw data the <b>text to append 
            cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty text to append', txt)
    
    def ok_cb(self):
        self.expFolderpathOut = self.folderPathOutControl.path()
        self.textToAppend = self.textToAppendLineEdit.text()
        isAccidentalOverwrite = (
            not self.overwriteToggle.isChecked()
            and self.expFolderpathOut == self._input_path
            and not self.textToAppend
        )
        if isAccidentalOverwrite:
            self.warnTextToAppendEmpty()
            return
        
        if self.textToAppend and not self.textToAppend.startswith('_'):
            self.textToAppend = f'_{self.textToAppend}'
            
        if self.overwriteToggle.isChecked():
            self.expFolderpathOut = None
        
        factor = self.factorSpinbox.value()
        self.resizeFactor = (
            factor if self.upScaleRadioButton.isChecked() else 1/factor
        )
        
        self.cancel = False
        self.close()


class ResizeUtilWorker(utils_base.BaseWorkerUtil):
    sigSetResizeProps = Signal(str)
    
    def emitSetResizeProps(self, input_path):
        self.mutex.lock()
        self.sigSetResizeProps.emit(input_path)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def __init__(self, mainWin):
        super().__init__(mainWin)
    
    def validateOutputPath(self, path):
        if path is None:
            return
        
        images_path = myutils.validate_images_path(path, create_dirs_tree=True)
        return images_path
    
    @workers_base.worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            abort = self.emitSetResizeProps(exp_path)
            if abort:
                self.signals.finished.emit(self)
                return
            
            tot_pos = len(pos_foldernames)
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )
                images_path = os.path.join(exp_path, pos, 'Images')
                
                
                rf = self.resizeFactor
                text_to_append = self.textToAppend
                images_path_out = self.validateOutputPath(self.expFolderpathOut)
                if images_path_out is None:
                    images_path_out = images_path
                resize.run(
                    images_path, rf, 
                    text_to_append=text_to_append, 
                    images_path_out=images_path_out
                )                
                
        self.signals.finished.emit(self)
