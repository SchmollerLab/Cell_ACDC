from .. import apps, myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil

class CreateConnected3Dsegm(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.CreateConnected3Dsegm(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename, existingEndnames):
        helpText = (
            """
            The new 3D segmentation file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new name. The rest of 
            the name will be the same as the original file.
            """
        )
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>new 3D segmentation</b> file:',
            existingNames=existingEndnames, 
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
            txt = '3D segmentation mask creation process aborted.'
        else:
            txt = '3D segmentation mask creation process completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()