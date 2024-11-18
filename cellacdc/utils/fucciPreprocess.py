import os

import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils, load

from .base import NewThreadMultipleExpBaseUtil

class FucciPreprocessUtil(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.FucciPreprocessWorker(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAskParams.connect(self.askSelectParams)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def askSelectParams(self, exp_path, pos_foldernames):
        channel_names = set()
        df_metadata = None
        for p, pos in enumerate(pos_foldernames):
            pos_path = os.path.join(exp_path, pos)
            images_path = os.path.join(pos_path, 'Images')
            basename, chNames = myutils.getBasenameAndChNames(images_path)
            channel_names.update(chNames)
            if df_metadata is not None:
                continue
            
            self.worker.basename = basename
            df_metadata = load.load_metadata_df(images_path)
        
        win = apps.FucciPreprocessDialog(
            channel_names,
            df_metadata=df_metadata,
            parent=self
        )
        win.exec_()
        
        self.worker.firstChannelName = win.firstChannelName
        self.worker.secondChannelName = win.secondChannelName
        self.worker.fucciFilterKwargs = win.function_kwargs
        self.worker.abort = win.cancel
        self.worker.waitCond.wakeAll()
    
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename):
        helpText = (
            """
            The combined and preprocessed image file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new name. The rest of 
            the name will be the same as the original file.
            """
        )
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>new combined channels</b> file:',
            defaultEntry='fucci_combined',
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