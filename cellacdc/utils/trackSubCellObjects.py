from .. import apps, myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil

class TrackSubCellFeatures(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, trackSubCellObjParams: dict, 
            parent=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
        self.trackingMode = trackSubCellObjParams['how']
        self.IoAthresh = trackSubCellObjParams['IoA']
        self.relabelSubObjLab = trackSubCellObjParams['relabelSubObjLab']
        self.createThirdSegm = trackSubCellObjParams['createThirdSegm']
        self.thirdSegmAppendedText = trackSubCellObjParams['thirdSegmAppendedText']
    
    def runWorker(self):
        self.worker = workers.TrackSubCellObjectsWorker(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigCriticalNotEnoughSegmFiles.connect(
            self.criticalNotEnoughSegmFiles
        )
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def criticalNotEnoughSegmFiles(self, exp_path):
        text = html_utils.paragraph(f"""
            The following experiment folder<br><br>
            <code>{exp_path}</code><br><br>
            <b>does NOT contain AT LEAST TWO segmentation files</b>.<br><br>
            To track sub-cellular objects you need to generate <b>one segmentation 
            file for the cells and one for the sub-cellular objects</b>.<br><br>
            Note that you can create as many segmentation files as you want, 
            you just need to run the segmentation module again. Thanks! 
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.addShowInFileManagerButton(exp_path)
        msg.critical(
            self, 'Not enough segmentation files!', text
        )
        self.worker.abort = True
        self.worker.waitCond.wakeAll()
    
    def askAppendName(self, basename, existingEndnames):
        helpText = (
            """
            The segmentation file containing the tracked sub-cellular objects 
            will be saved with a different file name.<br><br>
            Insert a name to append to the end of the new name. The rest of 
            the name will be the same as the original file.
            """
        )
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>new, tracked segmentation</b> file:',
            existingNames=existingEndnames, 
            helpText=helpText, 
            allowEmpty=False
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
            txt = 'Tracking sub-cellular objects aborted.'
        else:
            txt = 'Tracking sub-cellular objects completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()