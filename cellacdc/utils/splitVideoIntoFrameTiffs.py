import os

from .. import apps, myutils, workers, widgets, html_utils, printl

from .base import NewThreadMultipleExpBaseUtil

class SplitVideoIntoFrameTiffsUtil(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.SplitVideoIntoFrameTiffs(self)
        self.worker.sigAskSetup.connect(self.askSetupParams)
        self.worker.sigCancelled.connect(self.workerCancelled)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def askSetupParams(self, *args):
        exp_path, pos_foldernames, video_endname = args[0]
        video_filepath = None
        for pos_folder in pos_foldernames:
            images_path = os.path.join(exp_path, pos_folder, 'Images')
            basename, chNames = myutils.getBasenameAndChNames(images_path)
            for file in myutils.listdir(images_path):
                if file == f'{basename}{video_endname}':
                    video_filepath = os.path.join(images_path, file)
                    break
            
            if video_filepath is not None:
                break
        
        win = apps.SetupSplitVideoIntoTiffsDialog(video_filepath)
        win.exec_()

        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
    
        self.worker.dstFolderPath = win.dstFolderPath
        self.worker.prefixText = win.prefixText
        self.worker.dtypeOut = win.dtypeOut
        self.worker.onlyUntilTracked = win.onlyUntilTracked
        self.worker.onlyUntilAnnotated = win.onlyUntilAnnotated
        self.worker.acdcOutputEndname = win.acdcOutputEndname
        self.worker.waitCond.wakeAll()

    def workerCancelled(self):
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