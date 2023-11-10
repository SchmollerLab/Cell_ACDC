from .. import apps, myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil

class FilterObjsFromCoordsTable(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.FilterObjsFromCoordsTable(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigSetColumnsNames.connect(self.setColumnsNames)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename, existingEndnames):
        helpText = (
            """
            You can choose to save a new file for the filtered segmentation 
            or overwrite the existing one.
            """
        )
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>filtered segmentation</b> file:',
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
    
    def setColumnsNames(self, columns, categories, optionalCategories):
        win = apps.SetColumnNamesDialog(
            columns, categories, optionalCategories=optionalCategories, 
            parent=self
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return 
        
        self.selectedColumnsPerCategory = win.selectedColumns
        self.worker.waitCond.wakeAll()
    
    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Filter segmented objects from coordinates table process aborted.'
        else:
            txt = 'Filter segmented objects from coordinates table process completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()