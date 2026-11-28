import os

from .. import html_utils, printl
from cellacdc.myutils import filesystem, image_utils
from cellacdc.shared.widgets import message_box as widgets_message_box

from .base import NewThreadMultipleExpBaseUtil
import skimage.io
import skimage.measure
import skimage.exposure
from qtpy.QtCore import Signal
from cellacdc import load, io
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class SplitVideoIntoFrameTiffsUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = filesystem.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = SplitVideoIntoFrameTiffs(self)
        self.worker.sigAskSetup.connect(self.askSetupParams)
        self.worker.sigCancelled.connect(self.workerCancelled)
        self.worker.sigAskDstFolderExist.connect(
            self.askDstFolderExist
        )
        super().runWorker(self.worker)
    
    def askDstFolderExist(self, videoDstFolderPath):
        msg = widgets_message_box.myMessageBox(wrapText=False)
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
            basename, chNames = filesystem.getBasenameAndChNames(images_path)
            for file in filesystem.listdir(images_path):
                if file == f'{basename}{video_endname}':
                    video_filepath = os.path.join(images_path, file)
                    break
            
            if video_filepath is not None:
                break
        
        win = utils_base.SetupSplitVideoIntoTiffsDialog(
            video_filepath,
            parent=self
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
        self.workerFinished(None, aborted=True)
        self.worker.finished.emit(self.worker)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = '3D segmentation mask creation process aborted.'
        else:
            txt = '3D segmentation mask creation process completed.'
        self.logger.info(txt)
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class SplitVideoIntoFrameTiffs(utils_base.BaseWorkerUtil):
    sigAskSetup = Signal(object)
    sigCancelled = Signal()
    sigAskDstFolderExist = Signal(str)

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitAskSetup(self, exp_path, pos_foldernames, video_endname):
        self.mutex.lock()
        self.sigAskSetup.emit((exp_path, pos_foldernames, video_endname))
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitAskDstFolderExist(self, videoDstFolderPath):
        self.mutex.lock()
        self.sigAskDstFolderExist.emit(videoDstFolderPath)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        tot_num_pos = 0
        for exp_path, pos_foldernames in expPaths.items():
            tot_num_pos += len(pos_foldernames)
        tot_digits = max(2, len(str(tot_num_pos)))
        k = 0
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            if i == 0:
                self.mainWin.infoText = 'Select <b>video file to split</b>'
                cancel = self.emitSelectVideoFile(
                    exp_path, pos_foldernames, False
                )
                if cancel:
                    self.sigCancelled.emit()
                    return
                
                selectedVideoEndname = self.selectedVideoEndname

                # Emit ask setup parameters
                self.mainWin.infoText = 'Setup video splitting process'
                self.emitAskSetup(
                    exp_path, pos_foldernames, selectedVideoEndname
                )
                if self.abort:
                    self.sigCancelled.emit()
                    return
            
            dtype = self.dtypeOut
            prefix = self.prefixText
            onlyUntilTracked = self.onlyUntilTracked
            onlyUntilAnnotated = self.onlyUntilAnnotated
            acdcOutputEndname = self.acdcOutputEndname
            dstFolderPath = self.dstFolderPath

            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigCancelled.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                pos_num_str = f'{k+1}'.zfill(tot_digits)
                videoDstFolderPath = os.path.join(
                    dstFolderPath, pos_num_str
                )

                if os.path.exists(videoDstFolderPath):
                    self.emitAskDstFolderExist(videoDstFolderPath)
                    if self.abort:
                        self.sigCancelled.emit()
                        return
                    
                    io.delete_folder_content(videoDstFolderPath)

                images_path = os.path.join(exp_path, pos, 'Images')
                basename, chNames = filesystem.getBasenameAndChNames(images_path)
                ls = filesystem.listdir(images_path)
                imageFilepath = [
                    os.path.join(images_path, f) for f in ls 
                    if f == f'{basename}{selectedVideoEndname}'
                ][0]
                image_data = load.load_image_file(imageFilepath)

                numFrames = len(image_data)
                if onlyUntilTracked or onlyUntilAnnotated:
                    acdc_df = load.load_acdc_df_file(
                        images_path, 
                        end_name_acdc_df_file=acdcOutputEndname
                    )

                if self.numFramesToSplit is not None:
                    numFrames = self.numFramesToSplit
                elif onlyUntilTracked:
                    numFrames = acdc_df['frame_i'].max() + 1
                elif onlyUntilAnnotated:
                    try:
                        ccs = acdc_df[['cell_cycle_stage']]
                    except KeyError:
                        ccs = acdc_df[['generation_num_tree']]
                    last_index_cca_df = ccs.last_valid_index()
                    numFrames = (
                        acdc_df.loc[:last_index_cca_df, 'frame_i'].max() + 1
                    )

                self.logger.log(
                    f'Splitting video into single-frame TIFF files '
                    f'until frame n. {numFrames}...'
                )
                
                self.signals.sigInitInnerPbar.emit(numFrames)
                for frame_i in range(numFrames):
                    img = image_utils.convert_to_dtype(image_data[frame_i], dtype)
                    t_str = str(frame_i).zfill(3)
                    frame_filename = f'{prefix}{t_str}.tif'
                    frame_filepath = os.path.join(
                        videoDstFolderPath, frame_filename
                    )
                    skimage.io.imsave(frame_filepath, img)
                    self.signals.sigUpdateInnerPbar.emit(1)
                
                self.signals.progressBar.emit(1)

                k += 1

        self.signals.finished.emit(self)
