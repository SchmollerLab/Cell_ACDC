from .. import html_utils
from cellacdc.myutils import filesystem
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection
from cellacdc.shared.widgets import message_box as widgets_message_box

from .base import NewThreadMultipleExpBaseUtil
import os
import numpy as np
from qtpy.QtCore import Signal
from cellacdc import load, io
from cellacdc import core
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class TrackSubCellFeatures(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, trackSubCellObjParams: dict, 
            parent=None
        ):
        module = filesystem.get_module_name(__file__)
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
        self.worker = TrackSubCellObjectsWorker(self)
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
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
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
        win = dialogs_file_selection.filenameDialog(
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
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class TrackSubCellObjectsWorker(utils_base.BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigCriticalNotEnoughSegmFiles = Signal(str)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)
        if mainWin.trackingMode.find('Delete both') != -1:
            self.trackingMode = 'delete_both'
        elif mainWin.trackingMode.find('Delete sub-cellular') != -1:
            self.trackingMode = 'delete_sub'
        elif mainWin.trackingMode.find('Delete cells') != -1:
            self.trackingMode = 'delete_cells'
        elif mainWin.trackingMode.find('Only track') != -1:
            self.trackingMode = 'only_track'
        
        self.relabelSubObjLab = mainWin.relabelSubObjLab
        self.IoAthresh = mainWin.IoAthresh
        self.createThirdSegm = mainWin.createThirdSegm
        self.thirdSegmAppendedText = mainWin.thirdSegmAppendedText

    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            red_text = html_utils.span('OF THE CELLs')
            self.mainWin.infoText = f'Select <b>segmentation file {red_text}</b>'
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            
            # Critical --> there are not enough segm files
            if len(self.mainWin.existingSegmEndNames) < 2:
                self.mutex.lock()
                self.sigCriticalNotEnoughSegmFiles.emit(exp_path)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                self.sigAborted.emit()
                return

            self.cellsSegmEndFilename = self.mainWin.endFilenameSegm
            
            red_text = html_utils.span('OF THE SUB-CELLULAR OBJECTS')
            self.mainWin.infoText = (
                f'Select <b>segmentation file {red_text}</b>'
            )
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            
            # Ask appendend name
            self.mutex.lock()
            self.sigAskAppendName.emit(
                self.mainWin.endFilenameSegm, self.mainWin.existingSegmEndNames
            )
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.abort:
                self.sigAborted.emit()
                return

            appendedName = self.appendedName
            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = filesystem.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ][0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )

                # Load cells segmentation file
                segmDataCells, segmCellsPath = load.load_segm_file(
                    images_path, end_name_segm_file=self.cellsSegmEndFilename,
                    return_path=True
                )
                acdc_df_cells_endname = self.cellsSegmEndFilename.replace(
                    '_segm', '_acdc_output'
                )
                acdc_df_cell, acdc_df_cells_path = load.load_acdc_df_file(
                    images_path, end_name_acdc_df_file=acdc_df_cells_endname,
                    return_path=True
                )

                if posData.SizeT > 1:
                    numFrames = min((len(segmDataCells), len(posData.segm_data)))
                    segmDataCells = segmDataCells[:numFrames]
                    posData.segm_data = posData.segm_data[:numFrames]
                else:
                    numFrames = 1
                
                self.signals.sigInitInnerPbar.emit(numFrames*2)
                
                self.logger.log('Tracking sub-cellular objects...')
                tracked = core.track_sub_cell_objects(
                    segmDataCells, posData.segm_data, self.IoAthresh, 
                    how=self.trackingMode, SizeT=numFrames, 
                    sigProgress=self.signals.sigUpdateInnerPbar,
                    relabel_sub_obj_lab=self.relabelSubObjLab
                )
                (trackedSubSegmData, trackedCellsSegmData, numSubObjPerCell, 
                replacedSubIds) = tracked
       
                self.logger.log('Saving tracked segmentation files...')
                subSegmFilename, ext = os.path.splitext(posData.segm_npz_path)
                trackedSubPath = f'{subSegmFilename}_{appendedName}.npz'
                io.savez_compressed(trackedSubPath, trackedSubSegmData)
                posData.saveIsSegm3Dmetadata(trackedSubPath)

                if trackedCellsSegmData is not None:
                    cellsSegmFilename, ext = os.path.splitext(segmCellsPath)
                    trackedCellsPath = f'{cellsSegmFilename}_{appendedName}.npz'
                    io.savez_compressed(trackedCellsPath, trackedCellsSegmData)
                
                if self.createThirdSegm:
                    self.logger.log(
                        f'Generating segmentation from '
                        f'"{self.cellsSegmEndFilename} - {appendedName}" '
                        'difference...'
                    )
                    if trackedCellsSegmData is not None:
                        parentSegmData = trackedCellsSegmData
                    else:
                        parentSegmData = segmDataCells
                    diffSegmData = parentSegmData.copy()
                    diffSegmData[trackedSubSegmData != 0] = 0

                    self.logger.log('Saving difference segmentation file...')
                    diffSegmPath = (
                        f'{subSegmFilename}_{appendedName}'
                        f'_{self.thirdSegmAppendedText}.npz'
                    )
                    io.savez_compressed(diffSegmPath, diffSegmData)
                    posData.saveIsSegm3Dmetadata(diffSegmPath)
                    del diffSegmData
                
                if self.relabelSubObjLab:
                    # When we relabel the sub-cell objs acdc_df is not valid anymore
                    # because IDs could be different
                    posData.acdc_df = None
                
                self.logger.log('Generating acdc_output tables...')  
                # Update or create acdc_df for sub-cellular objects                
                acdc_dfs_tracked = core.track_sub_cell_objects_acdc_df(
                    trackedSubSegmData, posData.acdc_df, 
                    replacedSubIds, numSubObjPerCell,
                    tracked_cells_segm_data=trackedCellsSegmData,
                    cells_acdc_df=acdc_df_cell, SizeT=posData.SizeT, 
                    sigProgress=self.signals.sigUpdateInnerPbar
                )
                subTrackedAcdcDf, trackedAcdcDf = acdc_dfs_tracked

                self.logger.log('Saving acdc_output tables...')  
                subAcdcDfFilename, _ = os.path.splitext(
                    posData.acdc_output_csv_path
                )
                subTrackedAcdcDfPath = f'{subAcdcDfFilename}_{appendedName}.csv'
                subTrackedAcdcDf.to_csv(subTrackedAcdcDfPath)

                if trackedAcdcDf is not None:
                    basen = posData.basename
                    cellsSegmFilename = os.path.basename(segmCellsPath)
                    cellsSegmFilename, ext = os.path.splitext(cellsSegmFilename)
                    cellsSegmEndname = cellsSegmFilename[len(basen):]
                    trackedAcdcDfEndname = cellsSegmEndname.replace(
                        'segm', 'acdc_output'
                    )
                    trackedAcdcDfFilename = f'{basen}{trackedAcdcDfEndname}'
                    trackedAcdcDfFilename = f'{trackedAcdcDfFilename}_{appendedName}.csv'
                    trackedAcdcDfPath = os.path.join(
                        posData.images_path, trackedAcdcDfFilename
                    )
                    trackedAcdcDf.to_csv(trackedAcdcDfPath)
                    
                    if self.createThirdSegm:
                        if posData.SizeT == 1:
                            parentSegmData = parentSegmData[np.newaxis]
                        subAcdcDfFilename = (
                            subSegmFilename.replace('.npz', '.csv')
                            .replace('segm', 'acdc_output')
                        )
                        diffAcdcDfPath = (
                            f'{subAcdcDfFilename}_{appendedName}'
                            f'_{self.thirdSegmAppendedText}.csv'
                        )
                        third_segm_acdc_df = (
                            core.track_sub_cell_objects_third_segm_acdc_df(
                                parentSegmData, trackedAcdcDf
                            )
                        )
                        third_segm_acdc_df.to_csv(diffAcdcDfPath)

                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)
