from qtpy.QtWidgets import QFileDialog

from .. import apps, myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil

class concatWin(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.ConcatAcdcDfsWorker(self)
        self.worker.sigAskFolder.connect(self.askFolderWhereToSaveAllExp)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def askFolderWhereToSaveAllExp(self, allExp_filename):
        txt = html_utils.paragraph(f"""
            After clicking "Ok" you will be asked to <b>select a folder where you
            want to save the file</b><br>
            with the <b>concatenated tables</b> from the multiple experiments selected<br>
            (the filename will be <code>{allExp_filename}</code>)
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Select folder', txt)
        if msg.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
    
        mostRecentPath = myutils.getMostRecentPath()
        save_to_dir = QFileDialog.getExistingDirectory(
            self, f'Select folder where to save {allExp_filename}', 
            mostRecentPath
        )
        if not save_to_dir:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
        self.worker.allExpSaveFolder = save_to_dir

        self.worker.waitCond.wakeAll()

    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Concatenating <code>acdc_output</code> tables aborted.'
        else:
            txt = 'Concatenating <code>acdc_output</code> tables completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()