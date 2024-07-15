from .. import myutils, workers, widgets, html_utils
from .. import apps

from .base import NewThreadMultipleExpBaseUtil

class fromImageJRoiToSegmUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.qparent = parent
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = workers.FromImajeJroiToSegmNpzWorker(self)
        self.worker.sigSelectRoisProps.connect(self.selectRoisProps)
        super().runWorker(self.worker)
    
    def selectRoisProps(self, roi_filepath, TZYX_shape, is_multi_pos):
        win = apps.ImageJRoisToSegmManager(
            roi_filepath, TZYX_shape, 
            addUseSamePropsForNextPosButton=is_multi_pos,
            parent=self.qparent
        )
        win.exec_()
        self.worker.abort = win.cancel
        if win.cancel:
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.IDsToRoisMapper = win.IDsToRoisMapper
        self.worker.rescaleRoisSizes = win.rescaleSizes
        self.worker.repeatRoisZslicesRange = win.repeatRoisZslicesRange
        self.worker.useSamePropsForNextPos = win.useSamePropsForNextPos
        self.worker.areAllRoisSelected = win.areAllRoisSelected
        self.worker.waitCond.wakeAll()
    
    def showEvent(self, event):
        self.runWorker()
    
    def workerFinished(self, worker):
        super().workerFinished(worker)
        txt = 'Converting from ImageJ ROIs to Cell-ACDC segmentation file(s) completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Process completed', html_utils.paragraph(txt))
        self.close()