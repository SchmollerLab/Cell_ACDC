import os

from .. import html_utils, printl
from cellacdc.myutils import filesystem, image_utils
from cellacdc.shared.widgets import message_box as widgets_message_box

from .base import NewThreadMultipleExpBaseUtil
from . import base as utils_base
from collections import defaultdict
import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.exposure
from qtpy.QtCore import Signal
from cellacdc import load, io
from cellacdc import core
from cellacdc.shared.workers import base as workers_base

class CreateTrackastraInputData(NewThreadMultipleExpBaseUtil):
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
        self.worker = CreateTrackastraInputDataWorker(self)
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
        msg = widgets_message_box.myMessageBox(wrapText=False)
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
            commands=(videoDstFolderPath,),
            path_to_browse=videoDstFolderPath,
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
        
        win = SetupCreateTrackastraInputDataDialog(
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
            detailsText = "\n\n".join(detailsTexts)
            detailsText = (
                'The following positions were skipped:\n\n'
                f'{detailsText}'
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
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
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


class SetupCreateTrackastraInputDataDialog(utils_base.SetupSplitVideoIntoTiffsDialog):
    def __init__(self, video_filepath, logger_func=print, parent=None):
        super().__init__(
            video_filepath, 
            logger_func=logger_func, 
            parent=parent
        )

        self.prefixLineEdit.setText('t')
        self.prefixLineEdit.setDisabled(True)

        self.onlyUntilTrackedFormWidget.widget.setChecked(False)
        self.onlyUntilAnnotatedFormWidget.widget.setChecked(True)

        self.onlyUntilAnnotatedFormWidget.setDisabled(True)
        self.onlyUntilTrackedFormWidget.setDisabled(True)

        self._headerLabel.setText(html_utils.paragraph(
            self._headerText.replace(
                'Split video into single-frame TIFFs setup',
                'Generate Trackastra training data setup'
            )
        ))

        self.setWindowTitle('Generate Trackastra training data setup')


class CreateTrackastraInputDataWorker(utils_base.BaseWorkerUtil):
    sigAskSetup = Signal(object)
    sigWarnPartialDstPosFolderFound = Signal(str, str)
    sigAskDstFolderExist = Signal(str)
    sigCancelled = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitAskSetup(self, exp_path, pos_foldernames, video_endname):
        self.mutex.lock()
        self.sigAskSetup.emit((exp_path, pos_foldernames, video_endname))
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitWarnPartialDstPosFolderFound(self, dstFolderPath, pos_num_str):
        self.mutex.lock()
        self.sigWarnPartialDstPosFolderFound.emit(dstFolderPath, pos_num_str)
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
        src_paths_info = {
            'generated_folder': [],
            'source_position_folder': []    
        }
        self._warnings = defaultdict(dict)
        k = 0
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            if i == 0:
                self.mainWin.infoText = 'Select <b>input video file</b>'
                cancel = self.emitSelectVideoFile(
                    exp_path, pos_foldernames, False
                )
                if cancel:
                    self.sigCancelled.emit()
                    return
                
                selectedVideoEndname = self.selectedVideoEndname

                # Emit ask setup parameters
                self.mainWin.infoText = 'Setup Trackastra data creation process'
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

                pos_path = os.path.join(exp_path, pos)
                pos_num_str = f'{k+1}'.zfill(tot_digits)
                videoDstFolderPath = os.path.join(
                    dstFolderPath, pos_num_str
                )
                gtDstFolderPath = os.path.join(
                    dstFolderPath, f'{pos_num_str}_GT', 'TRA'
                )

                partial_dst_pos_found = (
                    os.path.exists(videoDstFolderPath)
                    ^ os.path.exists(gtDstFolderPath)
                ) 
                if partial_dst_pos_found:
                    self.emitWarnPartialDstPosFolderFound(
                        dstFolderPath, pos_num_str
                    )
                    self.sigCancelled.emit()
                    return

                if os.path.exists(videoDstFolderPath):
                    self.emitAskDstFolderExist(videoDstFolderPath)
                    if self.abort:
                        self.sigCancelled.emit()
                        return
                    
                    io.delete_folder_content(videoDstFolderPath)
                    io.delete_folder_content(gtDstFolderPath)

                src_paths_info['generated_folder'].append(videoDstFolderPath)
                src_paths_info['source_position_folder'].append(pos_path)
            
                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(pos_path, 'Images')
                basename, chNames = filesystem.getBasenameAndChNames(images_path)
                ls = filesystem.listdir(images_path)
                imageFilepath = [
                    os.path.join(images_path, f) for f in ls 
                    if f == f'{basename}{selectedVideoEndname}'
                ][0]
                image_data = load.load_image_file(imageFilepath)
                segmEndname = (
                    acdcOutputEndname
                        .replace(
                            'acdc_output', 'segm'
                        )
                        .replace(
                            '.csv', '.npz'
                        )
                )
                segm_data = load.load_segm_file(
                    images_path,
                    end_name_segm_file=segmEndname
                )
                if segm_data is None:
                    warning_class = (
                        f'Segmentation file ending with "{segmEndname}" not found'
                    )
                    self.logger.log(
                        f'---------------------------------\n'
                        f'[WARNING]: {warning_class} in "{images_path}".'
                        f'\n================================='
                    )
                    
                    self._warnings[images_path] = warning_class
                    continue

                numFrames = len(image_data)
                acdc_df = load.load_acdc_df_file(
                    images_path, 
                    end_name_acdc_df_file=acdcOutputEndname
                )

                if acdc_df is None:
                    warning_class = (
                        '`acdc_output` CSV file ending with '
                        f'"{acdcOutputEndname}" not found'
                    )
                    self.logger.log(
                        f'---------------------------------\n'
                        f'[WARNING]: {warning_class} in "{images_path}".\n\n'
                        'Skipping this position.'
                        f'\n================================='
                    )
                    self._warnings[images_path] = warning_class
                    continue

                if 'generation_num_tree' not in acdc_df.columns:
                    warning_class = (
                        '`acdc_output` CSV file ending with '
                        f'"{acdcOutputEndname}" does not have lineage '
                        'tree annotations'
                    )
                    self.logger.log(
                        f'---------------------------------\n'
                        f'[WARNING]: {warning_class} in \n\n'
                        f'"{images_path}".\n\n'
                        'Skipping this position.'
                        f'\n================================='
                    )
                    self._warnings[images_path] = warning_class
                    continue
                
                os.makedirs(videoDstFolderPath, exist_ok=True)
                os.makedirs(gtDstFolderPath, exist_ok=True)

                if self.numFramesToSplit is not None:
                    numFrames = self.numFramesToSplit
                else:
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

                t_digits = max(3, len(str(numFrames)))
                self.signals.sigInitInnerPbar.emit(numFrames*2)
                for frame_i in range(numFrames):
                    img = image_utils.convert_to_dtype(image_data[frame_i], dtype)
                    t_str = str(frame_i).zfill(t_digits)
                    frame_filename = f'{prefix}{t_str}.tif'
                    frame_filepath = os.path.join(
                        videoDstFolderPath, frame_filename
                    )
                    skimage.io.imsave(frame_filepath, img, check_contrast=False)
                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log(
                    f'Splitting segmentation into single-frame GT TIFF files '
                    f'until frame n. {numFrames}...'
                )
                acdc_df = acdc_df.set_index('frame_i')
                for frame_i in range(numFrames):
                    cca_df_frame_i = acdc_df.loc[[frame_i]]
                    lab = segm_data[frame_i]
                    lab = core.replace_Cell_ID_with_Cell_ID_tree(
                        lab, cca_df_frame_i
                    )
                    if np.any(lab > np.iinfo(np.uint16).max):
                        raise ValueError(
                            'Segmentation IDs exceed the uint16 range; '
                            'remap mask and CTC IDs consistently before export.'
                        )
                    
                    lab = lab.astype(np.uint16)
                    t_str = str(frame_i).zfill(t_digits)
                    lab_filename = f'man_track{t_str}.tif'
                    lab_filepath = os.path.join(
                        gtDstFolderPath, lab_filename
                    )
                    skimage.io.imsave(lab_filepath, lab, check_contrast=False)
                    self.signals.sigUpdateInnerPbar.emit(1)

                df_ctc_filepath = os.path.join(gtDstFolderPath, 'man_track.txt')
                self.logger.log(
                    f'Saving CTC table to {df_ctc_filepath}...'
                )
                df_ctc = core.acdc_df_to_ctc(
                    acdc_df,
                    last_training_frame_i=numFrames-1
                )
                df_ctc.to_csv(
                    df_ctc_filepath,
                    sep=" ",
                    header=False,
                    index=False,
                )
                
                self.signals.progressBar.emit(1)

                k += 1

        df_src_path = pd.DataFrame(src_paths_info)
        df_src_path.to_csv(
            os.path.join(dstFolderPath, '_source_paths.txt'),
            index=False
        )

        self.signals.finished.emit(self)
