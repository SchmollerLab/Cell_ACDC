import os

import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils, load

from .base import NewThreadMultipleExpBaseUtil

class CombineChannelsUtil(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.CombineChannelsWorker(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAskSetup.connect(self.askSetup)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def askSetup(self, expPaths):
        chNames = {}
        for j, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            for i, pos in enumerate(pos_foldernames):
                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                basename, chNames_loc = myutils.getBasenameAndChNames(images_path)
                if i == 0 and j == 0:
                    chNames = set(chNames_loc)
                    continue
                chNames = chNames.intersection(set(chNames_loc))

        chNames = sorted(set(chNames))
            
        self.worker.basename = basename
        df_metadata = load.load_metadata_df(images_path)
    
        win = apps.CombineChannelsSetupDialogUtil(
            chNames,
            df_metadata=df_metadata,
            parent=self
        )
        win.exec_()
        
        if win.cancel:
            self.worker.abort = win.cancel
            self.worker.waitCond.wakeAll()
            return 
        
        self.worker.keepInputDataType = win.keepInputDataType
        self.worker.selectedSteps = win.selectedSteps
        self.worker.waitCond.wakeAll()
        
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename):
        helpText = (
            """
            The combined channels file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new file name. The rest of 
            the name will be the same as the original file base.
            """
        )
        win = apps.filenameDialog(
            basename=basename,
            ext='.tif',
            hintText='Insert a name for the <b>combined channels</b> file:',
            defaultEntry='combined',
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
            txt = 'Channel combination aborted.'
        else:
            txt = 'Channel combination completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()