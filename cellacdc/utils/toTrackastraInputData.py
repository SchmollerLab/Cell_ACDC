import os

from .. import apps, myutils, workers, widgets, html_utils, printl

from .base import NewThreadMultipleExpBaseUtil

class CreateTrackastraInputData(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.CreateTrackastraInputDataWorker(self)
        self.worker.sigAskSetup.connect(self.askSetupParams)
        self.worker.sigCancelled.connect(self.workerCancelled)
        self.worker.sigWarnPartialDstPosFolderFound.connect(
            self.warnPartialDstPosFolderFound
        )
        self.worker.sigAskDstFolderExist.connect(
            self.askDstFolderExist
        )
        super().runWorker(self.worker)
    
    def warnPartialDstPosFolderFound(self, dstFolderPath, pos_num_str):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            The position folder <code>{pos_num_str}</code> already exists but 
            <b>only partially</b>!<br><br>
            Either the <code>{pos_num_str}</code> 
            or <code>{pos_num_str}_GT/TRA</code> sub-folders are missing in the 
            following folder:
            <copiable>{dstFolderPath}</copiable><br>
            Therefore, process cannot continue. We recommend deleting this 
            position folder.<br><br>
            Thank you for your patience!
        """)
        msg.critical(
            self, 'Partial destination folder found!', txt,
            path_to_browse=dstFolderPath
        )
        self.worker.abort = True
        self.worker.waitCond.wakeAll()

    def askDstFolderExist(self, videoDstFolderPath):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            The position folder below already exists!<br><br>
            If you continue, the content will be removed before 
            saving the new TIFF files.<br><br>
            Do you want to continue?
        """)
        msg.warning(
            self, 'Destination folder exists', txt,
            buttonsTexts=(
                'Cancel', 'Yes, overwrite existing content'
            ),
            path_to_browse=videoDstFolderPath
        )
        self.worker.abort = msg.cancel
        self.worker.waitCond.wakeAll()

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
        
        win = apps.SetupCreateTrackastraInputDataDialog(
            video_filepath, parent=self
        )
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
        self.worker.numFramesToSplit = win.numFramesToSplit
        self.worker.waitCond.wakeAll()

    def workerCancelled(self):
        self.workerFinished(None, cancelled=True)
        self.worker.finished.emit(self.worker)
    
    def workerFinished(self, worker, cancelled=False):
        messagebox_type = 'information'
        detailsText = ''
        noteText = ''
        if worker._warnings:
            messagebox_type = 'warning'
            detailsTexts = []
            for images_path, warning_class in worker._warnings.items():
                detailsTexts.append(
                    f'  - {warning_class} in "{images_path}"'
                )
            detailsText = (
                'The following positions were skipped:\n\n'
                f'{"\n\n".join(detailsTexts)}'
            )
            noteText = (
                '<br><br>WARNING: Some positions were skipped. '
                'See below which ones'
            )
        if cancelled:
            txt = f'"{self._title}" process cancelled.'
            path_to_browse = None
        else:
            txt = (
                f'"{self._title}" process completed.<br><br>'
                'Trackastra training data generated in the following folder:'
                f'<copiable>{self.worker.dstFolderPath}</copiable>'
            )
            path_to_browse = self.worker.dstFolderPath
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if cancelled:
            msg.warning(self, 'Process cancelled', html_utils.paragraph(txt))
        else:
            getattr(msg, messagebox_type)(
                self, 'Process completed', 
                html_utils.paragraph(f'{txt}{noteText}'),
                path_to_browse=path_to_browse,
                detailsText=detailsText
            )
        super().workerFinished(worker)
        self.close()