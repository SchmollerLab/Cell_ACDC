import os

import pandas as pd

from .. import html_utils, load, printl
from cellacdc.myutils import filesystem, image_utils
from cellacdc.shared.dialogs import (
    file_selection as dialogs_file_selection,
    preprocessing as dialogs_preprocessing,
)
from cellacdc.shared.widgets import message_box as widgets_message_box

from .base import NewThreadMultipleExpBaseUtil
from typing import List, Dict, Any, Iterable
from qtpy.QtCore import Signal
from cellacdc import io
from cellacdc import core
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class CustomPreprocessUtil(NewThreadMultipleExpBaseUtil):
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
        self.worker = CustomPreprocessWorkerUtil(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAskSetupRecipe.connect(self.askSetupRecipe)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def askSetupRecipe(self, exp_path, pos_foldernames):
        channel_names = set()
        df_metadata = None
        for p, pos in enumerate(pos_foldernames):
            pos_path = os.path.join(exp_path, pos)
            images_path = os.path.join(pos_path, 'Images')
            basename, chNames = filesystem.getBasenameAndChNames(images_path)
            channel_names.update(chNames)
            if df_metadata is not None:
                continue
            
            self.worker.basename = basename
            df_metadata = load.load_metadata_df(images_path)
        
        win = dialogs_preprocessing.PreProcessRecipeDialogUtil(
            channel_names,
            df_metadata=df_metadata,
            parent=self
        )
        win.exec_()
        
        if win.cancel:
            self.worker.abort = win.cancel
            self.worker.waitCond.wakeAll()
            return 
        
        self.worker.selectedChannels = win.selectedChannels
        self.worker.recipe = win.selectedRecipe
        self.worker.waitCond.wakeAll()
        
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename):
        helpText = (
            """
            The preprocessed image file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new file name. The rest of 
            the name will be the same as the original file.
            """
        )
        win = dialogs_file_selection.filenameDialog(
            basename=basename,
            ext='.tif',
            hintText='Insert a name for the <b>preprocessed image</b> file:',
            defaultEntry='preprocessed',
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
            txt = 'Custom pre-processing aborted.'
        else:
            txt = 'Custom pre-processing completed.'
        self.logger.info(txt)
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class CustomPreprocessWorkerUtil(utils_base.BaseWorkerUtil):
    sigAskAppendName = Signal(str)
    sigAskSetupRecipe = Signal(object, object)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)
    
    def emitAskSetupRecipe(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.sigAskSetupRecipe.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def applyPipeline(
            self, 
            images_path: os.PathLike,
            channel_names: Iterable[str],
            recipe: List[Dict[str, Any]],
            appended_text_filename: str
        ):
        posData = None        
        preprocessed_data = {}
        for channel in channel_names:
            self.logger.log(f'Loading {channel} channel data...')
            ch_filepath = load.get_filename_from_channel(images_path, channel)
            ch_image_data = load.load_image_file(ch_filepath)
            if posData is None:
                posData = load.loadData(ch_filepath, channel)
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_metadata=True,
                )
            if posData.SizeT == 1:
                ch_image_data = (ch_image_data,)

            preprocessed_ch_data = core.preprocess_image_from_recipe_multithread(
                ch_image_data, recipe
            )
            
            keep_input_data_type = recipe[0].get('keep_input_data_type', True)
            if keep_input_data_type:
                preprocessed_ch_data = image_utils.convert_to_dtype(
                    preprocessed_ch_data, ch_image_data.dtype
                )

            _, ext = os.path.splitext(ch_filepath)
            basename = posData.basename
            processed_filename = (
                f'{basename}{channel}_{appended_text_filename}{ext}'
            )
            preprocessed_data[processed_filename] = preprocessed_ch_data
            
        return preprocessed_data
    
    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = 'Setup recipe'
            
            if i == 0:
                abort = self.emitAskSetupRecipe(exp_path, pos_foldernames)
                if abort:
                    self.sigAborted.emit()
                    return
            
                # Ask append name
                self.mutex.lock()
                basename = f'{self.basename}{self.selectedChannels[0]}_'
                self.sigAskAppendName.emit(basename)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                if self.abort:
                    self.sigAborted.emit()
                    return
            
            appendedName = self.appendedName
            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')                
                self.logger.log(
                    'Applying custom pre-processing recipe...\n'
                )
                processed_data = self.applyPipeline(
                    images_path, self.selectedChannels, 
                    self.recipe, appendedName
                )
                
                for filename, preprocessed_ch_data in processed_data.items():
                    preprocessed_filepath = os.path.join(images_path, filename)
                    self.logger.log(
                        f'Saving pre-processed images to '
                        f'"{preprocessed_filepath}"...'
                    )
                    
                    io.save_image_data(
                        preprocessed_filepath, preprocessed_ch_data
                    )                                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)
