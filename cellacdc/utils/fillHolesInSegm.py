from .. import html_utils, load
from cellacdc.myutils import filesystem
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection
from cellacdc.shared.widgets import message_box as widgets_message_box

from .base import NewThreadMultipleExpBaseUtil
import os
import numpy as np
from qtpy.QtCore import Signal
from cellacdc import io
from cellacdc import core
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class fillHolesInSegm(NewThreadMultipleExpBaseUtil):
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
        self.worker = FillHolesInSegWorker(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAborted.connect(self.workerAborted)
        self.worker.sigSelectSegmFiles.connect(self.askInputSegm)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Filling holes in segmentation mask process aborted.'
        else:
            txt = 'Filling holes in segmentation mask process completed.'
        self.logger.info(txt)
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()
    
    def askInputSegm(self, exp_path, pos_foldernames):
        existingSegmEndNames = load.get_segm_endnames_from_exp_path(
        exp_path, pos_foldernames=pos_foldernames
    )
        win = dialogs_file_selection.SelectSegmFileDialog(
            existingSegmEndNames, exp_path, parent=self, allowMultipleSelection=True,
            infoText=f"Select the segmentation files for folder {exp_path}."
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.endFilenameSegmTemp = None
            self.worker.waitCond.wakeAll()
            return
        self.worker.endFilenameSegmTemp = win.selectedItemTexts
        self.worker.waitCond.wakeAll()
    
    def askAppendName(self, basename):
        helpText = (
            """
            The new segmentation file can be saved with a different 
            file name.<br><br> Insert a name if the old segmentation should not 
            be overwritten.
            """
        )
        win = dialogs_file_selection.filenameDialog(
            hintText='Insert a name extension if the old file should not be overwritten.',
            helpText=helpText, basename=basename,
            allowEmpty=True
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        if win.entryText is None:
            self.worker.appendedName = ""
        else:
            self.worker.appendedName = win.entryText
        self.worker.waitCond.wakeAll()


class FillHolesInSegWorker(utils_base.BaseWorkerUtil):
    sigAskAppendName = Signal(str)
    sigAborted = Signal()
    sigSelectSegmFiles = Signal(str, list)


    def __init__(self, mainWin):
        super().__init__(mainWin)
        
    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitAskAppendName(self, basename):
        self.mutex.lock()
        self.sigAskAppendName.emit(basename)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    @workers_base.worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths
        lab_paths_dict = dict()
        unique_segm_files = set()
        tot_segm_files = 0
        for exp_path, pos_foldernames in expPaths.items():
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            for pos_folder in pos_foldernames:
                imgs_path = os.path.join(exp_path, 
                                         pos_folder, 
                                         "Images")
                lab_paths_dict[imgs_path] = self.endFilenameSegmTemp
                tot_segm_files += len(self.endFilenameSegmTemp)
                unique_segm_files.update(self.endFilenameSegmTemp)

        self.logger.info('Filling holes in segmentation masks...')
        abort = self.emitAskAppendName("/".join(unique_segm_files))
        if abort:
            self.sigAborted.emit()
            return
        self.signals.initProgressBar.emit(tot_segm_files)
        for images_path, segm_file_names in lab_paths_dict.items():
            for segm_file_name in segm_file_names:
                segm_data, segm_data_path = load.load_segm_file(
                    images_path, end_name_segm_file=segm_file_name, return_path=True
                )
                segm_data_shape = segm_data.shape
                segm_data_ndim = len(segm_data_shape)
                if segm_data_ndim == 2:
                    segm_data = segm_data[np.newaxis, np.newaxis, ...]
                elif segm_data_ndim == 3:
                    segm_data = segm_data[np.newaxis, ...]
                elif segm_data_ndim == 4:
                    segm_data = segm_data
                else:
                    raise NotImplementedError(
                        "This ndim is not supported!"
                    )
                for i, stack in enumerate(segm_data):
                    for j, lab in enumerate(stack):
                        segm_data[i, j] = core.fill_holes_in_segmentation(lab)
                
                segm_data_save_path = (segm_data_path
                                       .replace(segm_file_name,
                                                f"{segm_file_name}{self.appendedName}"))
                io.savez_compressed(segm_data_save_path, segm_data)
                self.signals.progressBar.emit(1)
        self.signals.finished.emit(self)
