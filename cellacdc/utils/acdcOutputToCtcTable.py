import os

from .. import apps, myutils, workers, widgets, html_utils, printl

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Signal
from cellacdc import load
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class ConvertAcdcOutputToCtcTableUtil(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.CreateCtcTableFromAcdcOutput(self)
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
        self.worker.finished.emit(self.worker)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = f'{self.windowTitle()} process cancelled.'
        else:
            txt = f'{self.windowTitle()} process completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class CreateCtcTableFromAcdcOutput(utils_base.BaseWorkerUtil):
    sigCancelled = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)

    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = (
                'Select <code>acdc_output</code> file(s) '
                '<b>to convert to CTC table(s)</b>'
            )
            cancel = self.emitSelectAcdcOutputFiles(
                exp_path, pos_foldernames, 
                infoText=' to convert to CTC table',
                allowSingleSelection=True
            )
            if cancel:
                self.sigCancelled.emit()
                return
            
            selectedAcdcOutputEndnames = self.mainWin.selectedAcdcOutputEndnames

            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigCancelled.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                basename, chNames = myutils.getBasenameAndChNames(images_path)
                for acdcOutEndname in selectedAcdcOutputEndnames:
                    df_ctc, df_ctc_filepath = (
                        load.create_ctc_table_from_images_path(
                            images_path,
                            acdc_output_endname=acdcOutEndname
                        )
                    )
                    self.logger.log(f'CTC table saved at "{df_ctc_filepath}"')
                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)
