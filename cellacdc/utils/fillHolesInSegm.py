from .. import apps, myutils, workers, widgets, html_utils, load

from .base import NewThreadMultipleExpBaseUtil
import os

class fillHolesInSegm(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = workers.FillHolesInSegWorker(self)
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
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
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
        win = apps.SelectSegmFileDialog(
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
        win = apps.filenameDialog(
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