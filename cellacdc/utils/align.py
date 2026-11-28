from functools import partial

from qtpy.QtWidgets import QFileDialog

from .. import html_utils
from cellacdc.myutils import filesystem, image_utils
from cellacdc.shared.dialogs import (
    file_selection as dialogs_file_selection,
    generic as dialogs_generic,
)
from cellacdc.shared.widgets import (
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)

from .base import NewThreadMultipleExpBaseUtil
import os
import numpy as np
from qtpy.QtCore import Signal
from cellacdc import load, io
from cellacdc import core
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class alignWin(NewThreadMultipleExpBaseUtil):
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
        self.worker = AlignWorker(self)
        self.worker.sigAskUseSavedShifts.connect(self.askUseSavedShifts)
        self.worker.sigAskSelectChannel.connect(self.askSelectChannel)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def askUseSavedShifts(self, exp_path, basename):
        txt = html_utils.paragraph(f"""
            Some or all the Positions in this experiment folder<br><br>
            <code>{exp_path}</code><br><br>
            contains <b>saved shifts from a previous alignment process</b>.<br><br>
            What do you want to do?
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        _, useShiftsButton, ignoreShiftsButton, revertButton = msg.question(
            self, 'Select how saved shifts', txt,
            buttonsTexts=(
                'Cancel', 'Apply alignment from saved shifts',
                'Ignore saved shifts and compute alignment',
                'Revert alignment using saved shifts'
            )
        )
        if msg.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
        self.worker.revertedAlignEndname = None
        if msg.clickedButton == useShiftsButton:
            savedShiftsHow = 'use_saved_shifts'
        elif msg.clickedButton == ignoreShiftsButton:
            savedShiftsHow = 'ignore_saved_shifts'
        elif msg.clickedButton == revertButton:
            savedShiftsHow = 'rever_alignment'
            txt = html_utils.paragraph(f"""
                How do you want to save the image file with reverted alignment?
            """)
            msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
            overWriteButton = widgets_push_buttons.savePushButton('Overwrite existing file')
            saveAsButton = widgets_push_buttons.newFilePushButton('Save as new file...')
            _, overWriteButton, saveAsButton = msg.question(
                self, 'Select how saved shifts', txt,
                buttonsTexts=('Cancel', overWriteButton, saveAsButton),
                showDialog=False
            )
            saveAsButton.clicked.disconnect()
            saveAsCallback = partial(self.askAppendedName, basename, msg)
            saveAsButton.clicked.connect(saveAsCallback)
            msg.exec_()
            if msg.cancel:
                self.worker.abort = True
                self.worker.waitCond.wakeAll()
        
        self.worker.savedShiftsHow = savedShiftsHow

        self.worker.waitCond.wakeAll()
    
    def askAppendedName(self, basename, parent):
        win = dialogs_file_selection.filenameDialog(
            ext='.tif', title='Reverted alignment data filename',
            hintText='Insert a text to append to the filename', 
            parent=self, basename=basename, allowEmpty=False
        )
        win.exec_()
        if win.cancel:
            return
        self.worker.revertedAlignEndname = win.entryText
        parent.cancel = False
        parent.close()
    
    def askSelectChannel(self, channels):
        selectChannelWin = dialogs_generic.QDialogCombobox(
            'Select channel', channels, 'Select reference channel for the aligment',
            CbLabel='Select channel:  ', parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll() 
        
        self.worker.chName = selectChannelWin.selectedItemText
        self.worker.waitCond.wakeAll()

    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Aligning frames process CANCELLED.'
        else:
            txt = 'Aligning frames process completed.'
        self.logger.info(txt)
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class AlignWorker(utils_base.BaseWorkerUtil):
    sigAborted = Signal()
    sigAskUseSavedShifts = Signal(str, str)
    sigAskSelectChannel = Signal(list)

    def __init__(self, mainWin):
        super().__init__(mainWin)
    
    def emitAskUseSavedShifts(self, expPath, basename):
        self.mutex.lock()
        self.sigAskUseSavedShifts.emit(expPath, basename)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def emitAskSelectChannel(self, channels):
        self.mutex.lock()
        self.sigAskSelectChannel.emit(channels)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    @workers_base.worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            shiftsFound = False
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                ls = filesystem.listdir(images_path)
                for file in ls:
                    if file.endswith('align_shift.npy'):
                        shiftsFound = True
                        basename, chNames = filesystem.getBasenameAndChNames(
                            images_path, useExt=('.tif', '.h5')
                        )
                        break
                if shiftsFound:
                    break
            
            savedShiftsHow = None
            if shiftsFound:
                basename_ch0 = f'{basename}{chNames[0]}_'
                abort = self.emitAskUseSavedShifts(exp_path, basename_ch0)
                if abort:
                    self.sigAborted.emit()
                    return

                savedShiftsHow = self.savedShiftsHow

            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log('*'*40)
                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                basename, chNames = filesystem.getBasenameAndChNames(
                    images_path, useExt=('.tif', '.h5')
                )

                self.signals.sigUpdatePbarDesc.emit(f'Loading {pos_path}...')

                if p == 0:
                    self.logger.log(f'Asking to select reference channel...')
                    abort = self.emitAskSelectChannel(chNames)
                    if abort:
                        self.sigAborted.emit()
                        return
                    chName = self.chName
                
                file_path = filesystem.getChannelFilePath(images_path, chName)

                # Load data
                posData = load.loadData(file_path, chName)
                posData.getBasenameAndChNames(useExt=('.tif', '.h5'))
                posData.buildPaths()
                posData.loadImgData()

                posData.loadOtherFiles(
                    load_segm_data=False, 
                    load_shifts=True,
                    loadSegmInfo=True
                )

                if posData.img_data.ndim == 4:
                    align_func = core.align_frames_3D
                    if posData.segmInfo_df is None:
                        raise FileNotFoundError(
                            'To align 4D data you need to select which z-slice '
                            'you want to use for alignment. Please run the module '
                            '`Launch data prep module...` before aligning the '
                            'frames. (z-slice info MISSING from position '
                            f'"{posData.relPath}")'
                        )
                    df = posData.segmInfo_df.loc[posData.filename]
                    zz = df['z_slice_used_dataPrep'].to_list()
                elif posData.img_data.ndim == 3:
                    align_func = core.align_frames_2D
                    zz = None
                
                useSavedShifts = (
                    savedShiftsHow == 'use_saved_shifts'
                    and posData.loaded_shifts is not None
                )
                if useSavedShifts:
                    user_shifts = posData.loaded_shifts
                else:
                    user_shifts = None
                
                if savedShiftsHow == 'rever_alignment':
                    if posData.loaded_shifts is None:
                        self.logger.log(
                            f'WARNING: Cannot revert alignment in "{posData.relPath}" '
                            'since it is missing previously computed shifts. '
                            'Skipping this positon.'
                        )
                        continue
                    
                    # Revert alignment and save selected channel
                    for chName in chNames:
                        self.logger.log(
                            f'Reverting alignment on "{chName}"...'
                        )
                        if chName == posData.user_ch_name:
                            data = posData.img_data
                        else:
                            file_path = filesystem.getChannelFilePath(
                                images_path, chName
                            )
                            data = load.load_image_file(file_path)
                        
                        self.signals.sigInitInnerPbar.emit(len(data)-1)
                        revertedData = core.revert_alignment(
                            posData.loaded_shifts, data, 
                            sigPyqt=self.signals.sigUpdateInnerPbar
                        )
                        self.logger.log(
                            f'Saving "{chName}"...'
                        )
                        self.signals.sigInitInnerPbar.emit(0)
                        self.saveAlignedData(
                            revertedData, images_path, posData.basename, 
                            chName, self.revertedAlignEndname,
                            ext=posData.ext
                        )
                        del revertedData, data
                else:
                    for chName in chNames:
                        self.logger.log(
                            f'Aligning "{chName}"...'
                        )
                        if chName == posData.user_ch_name:
                            data = posData.img_data
                        else:
                            file_path = filesystem.getChannelFilePath(
                                images_path, chName
                            )
                            data = load.load_image_file(file_path)
                        self.signals.sigInitInnerPbar.emit(len(data)-1)
                        
                        alignedImgData, shifts = align_func(
                            data, slices=zz, user_shifts=user_shifts,
                            sigPyqt=self.signals.sigUpdateInnerPbar
                        )
                        self.logger.log(f'Saving "{chName}"...')
                        np.save(posData.align_shifts_path, shifts)
                        
                        self.signals.sigInitInnerPbar.emit(0)
                        self.saveAlignedData(
                            alignedImgData, images_path, posData.basename, 
                            chName, '', ext=posData.non_aligned_ext
                        )
                        self.saveAlignedData(
                            alignedImgData, images_path, posData.basename, 
                            chName, 'aligned', ext='.npz'
                        )
                        del alignedImgData, data
                
        self.signals.finished.emit(self)
    
    def saveAlignedData(
            self, data, imagesPath, basename, chName, endname, ext='.tif'
        ):
        if endname:
            newFilename = f'{basename}{chName}_{endname}{ext}'
        else:
            newFilename = f'{basename}{chName}{ext}'
        
        filePath = os.path.join(imagesPath, newFilename)

        if ext == '.tif':
            SizeT = data.shape[0]
            SizeZ = 1
            if data.ndim == 4:
                SizeZ = data.shape[1]
            image_utils.to_tiff(filePath, data)
        elif ext == '.npz':
            io.savez_compressed(filePath, data)
        elif ext == '.h5':
            load.save_to_h5(filePath, data)
