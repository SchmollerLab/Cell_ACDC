from ... import myutils, workers, widgets, html_utils
from ... import apps

from ..base import NewThreadMultipleExpBaseUtil

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