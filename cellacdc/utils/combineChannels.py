import os

import pandas as pd

from .. import html_utils, load
from cellacdc.myutils import filesystem
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection
from cellacdc.shared.widgets import (
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from .. import printl

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtWidgets import QLabel, QSpinBox
from cellacdc import qutils
from cellacdc.shared.dialogs import preprocessing as dialogs_preprocessing
from typing import Dict, Any
from qtpy.QtCore import Signal
from cellacdc import core
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class CombineChannelsUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = filesystem.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths

    def runWorker(self):
        self.worker = CombineChannelsWorkerUtil(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAskSetup.connect(self.askSetup)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def askSetup(self, expPaths):
        self.images_paths = []
        chNames = {}
        for j, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            for i, pos in enumerate(pos_foldernames):
                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                self.images_paths.append(images_path)
                basename, chNames_loc = filesystem.getBasenameAndChNames(
                    images_path
                )
                segm_files = load.get_segm_files(images_path)
                segm_endnames = load.get_endnames(
                    basename, segm_files
                )
                if i == 0 and j == 0:
                    chNames = set(chNames_loc)
                    chNames.update(segm_endnames)
                    continue
                
                chNames_loc = set(chNames_loc)
                chNames_loc.update(segm_endnames)
                chNames = chNames.intersection(chNames_loc)

        chNames = sorted(set(chNames))
            
        self.worker.basename = basename
        df_metadata = load.load_metadata_df(images_path)
    
        win = CombineChannelsSetupDialogUtil(
            chNames,
            df_metadata=df_metadata,
            parent=self
        )
        win.exec_()
        
        if win.cancel:
            self.worker.abort = win.cancel
            self.worker.waitCond.wakeAll()
            return 
        
        self.worker.keepInputDataType = win.keepInputDataType
        self.worker.selectedSteps = win.selectedSteps
        self.worker.nThreads = win.nThreadsSpinBox.value()
        self.worker.formula = win.formulaEditWidget.text()
        self.worker.saveAsSegm = win.saveAsSegm()
        self.worker.waitCond.wakeAll()
        
    def showEvent(self, event):
        self.runWorker()
    
    def getBasenameExtAndExtensionOutputImage(self):
        saveAsSegm = self.worker.saveAsSegm
        if saveAsSegm:
            basename_ext = 'segm'
            ext = '.npz'
            return basename_ext, ext
        else:
            basename_ext = ''
            ext = '.tif'
            return basename_ext, ext
    
    def askAppendName(self, basename):
        basename_ext, ext = self.getBasenameExtAndExtensionOutputImage()
        saveAsSegm = self.worker.saveAsSegm
        helpText = (
            f"""
            The {"combined channels" if not saveAsSegm else "combined segmentation"} 
            file will be saved with a different file name.<br><br>
            Insert a name to append to the end of the new file name. The rest of 
            the name will be the same as the original file base.
            """
        )
        win = dialogs_file_selection.filenameDialog(
            basename=f'{basename}{basename_ext}',
            ext=ext,
            hintText=f'Insert a name for the <b>{"combined channels" if not saveAsSegm else "combined segmentation"}</b> file:',
            defaultEntry='combined',
            helpText=helpText,
            allowEmpty=False,
            parent=self
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.appendedName = win.entryText
        self.worker.waitCond.wakeAll()
    
    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Channel combination aborted.'
        else:
            txt = 'Channel combination completed.'
        self.logger.info(txt)
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class CombineChannelsSetupDialogUtil(dialogs_preprocessing.CombineChannelsSetupDialog):
    def __init__(
            self,
            channel_names,
            df_metadata=None,
            parent=None,
        ):

        super().__init__(
            channel_names,
            parent=parent,
            df_metadata=df_metadata
            )
        
        # add int input for number of workers

        
        self.mainLayout.addSpacing(20)

        qutils.hide_and_delete_layout(self.buttonsLayout)
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        self.buttonsLayout = buttonsLayout
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addLayout(buttonsLayout)
        
        self.nThreadsSpinBox = QSpinBox()
        self.nThreadsSpinBox.setMinimum(1)
        self.nThreadsSpinBox.setValue(4)
        self.nThreadsSpinBox.setToolTip("Number of threads to use for processing")
        self.mainLayout.addWidget(QLabel("Number of threads:"))
        self.mainLayout.addWidget(self.nThreadsSpinBox)


class CombineChannelsWorkerUtil(utils_base.BaseWorkerUtil):
    sigAskAppendName = Signal(str)
    sigAskSetup = Signal(object)
    sigAborted = Signal()

    def __init__(self, mainWin, mutex=None, waitCond=None):
        super().__init__(mainWin)
    
    def emitAskSetup(self, expPaths):
        self.mutex.lock()
        self.sigAskSetup.emit(expPaths)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def applyPipeline(
            self, 
            image_paths: os.PathLike,
            steps:  Dict[str, Dict[str, Any]],
            appended_text_filename: str,
            keep_input_data_type: bool,
            n_threads: int = None,
            formula: str = None,
        ):
        save_filepaths = []
        images_path_to_process = []
        if self.saveAsSegm:
            out_ext = '.npz'
            basename_ext = 'segm_'
        else:
            out_ext = '.tif'
            basename_ext = ''
        
        for images_path in image_paths:
            basename, channels = filesystem.getBasenameAndChNames(images_path)
            
            savename = (
                f'{basename}{basename_ext}{appended_text_filename}{out_ext}'
            )

            images_path_to_process.append(images_path)
            save_filepaths.append(os.path.join(images_path, savename))
        
        core.combine_channels_multithread(
            inputs_info=steps,
            images_paths=images_path_to_process,
            keep_input_data_type=keep_input_data_type,
            save_filepaths=save_filepaths,
            signals=self.signals,
            logger_func=self.logger.log,
            n_threads=n_threads,
            output_as_segm=self.saveAsSegm,
            formula=formula,
        )
    
    @workers_base.worker_exception_handler
    def run(self):

        self.signals.initProgressBar.emit(0)

        expPaths = self.mainWin.expPaths
        abort = self.emitAskSetup(expPaths)
        if abort:
            self.sigAborted.emit()
            return
    
        # Ask append name
        self.mutex.lock()
        basename = f'{self.basename}'
        self.sigAskAppendName.emit(basename)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return

        appendedName = self.appendedName

        selectedSteps = self.selectedSteps

        self.logger.log('Applying pipeline...')
        self.logger.log('Selected steps:')
        for step in selectedSteps.values():
            self.logger.log(step)
            
        image_paths = []
        for exp_path, pos_foldernames in expPaths.items():
            image_paths += [os.path.join(exp_path, pos, 'Images') for pos in pos_foldernames]

        self.signals.initProgressBar.emit(len(pos_foldernames))
        formula = self.formula
        self.applyPipeline(
            image_paths,
            selectedSteps,
            appendedName,
            self.keepInputDataType,
            n_threads=self.nThreads,
            formula=formula,
        )

        self.signals.finished.emit(self)
