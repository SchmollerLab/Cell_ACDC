import os

import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils, load
from .. import printl

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
        self.worker = workers.CombineChannelsWorkerUtil(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAskSetup.connect(self.askSetup)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def askSetup(self, expPaths):
        self.images_paths = []
        chNames = {}
        for j, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            for i, pos in enumerate(pos_foldernames):
                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                self.images_paths.append(images_path)
                basename, chNames_loc = myutils.getBasenameAndChNames(
                    images_path
                )
                segm_files = load.get_segm_files(images_path)
                segm_endnames = load.get_endnames(
                    basename, segm_files
                )
                if i == 0 and j == 0:
                    chNames = set(chNames_loc)
                    chNames.update(segm_endnames)
                    continue
                
                chNames_loc.update(segm_endnames)
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
        self.worker.nThreads = win.nThreadsSpinBox.value()
        self.worker.waitCond.wakeAll()
        
    def showEvent(self, event):
        self.runWorker()
    
    def getBasenameExtAndExtensionOutputImage(self):
        ext = '.npz'
        basename_ext = 'segm_'
        for step_n, step in self.worker.selectedSteps.items():
            channel_name = step['channel']
            if '_segm' not in channel_name:
                basename_ext = ''
                
            for images_path in self.images_paths:
                image_filepath = load.get_filepath_from_endname(
                    images_path, channel_name
                )
                
                _, ext = os.path.splitext(image_filepath)
                if ext != '.npz':
                    return '', '.tif'
        
        return basename_ext, ext
    
    def askAppendName(self, basename):
        basename_ext, ext = self.getBasenameExtAndExtensionOutputImage()
        helpText = (
            """
            The combined channels file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new file name. The rest of 
            the name will be the same as the original file base.
            """
        )
        win = apps.filenameDialog(
            basename=f'{basename}{basename_ext}',
            ext=ext,
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