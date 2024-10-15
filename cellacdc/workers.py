import re
import os
import shutil
import time
import json
from collections import defaultdict, deque

from typing import Union, List, Dict

from functools import wraps
import numpy as np
import pandas as pd
import h5py
import traceback

import skimage.io
import skimage.measure

import queue

from tqdm import tqdm

from qtpy.QtCore import (
    Signal, QObject, QMutex, QWaitCondition
)

from cellacdc import html_utils

from . import (
    load, myutils, core, prompts, printl, config,
    segm_re_pattern
)
from . import transformation
from .path import copy_or_move_tree
from . import features
from . import core
from . import cca_df_colnames, lineage_tree_cols, default_annot_df
from . import cca_df_colnames_with_tree
from .utils import resize

DEBUG = False

def worker_exception_handler(func):
    @wraps(func)
    def run(self):
        try:
            func(self)
        except Exception as error:
            printl(traceback.format_exc())
            try:
                self.dataQ.clear()
            except Exception as err:
                pass
            
            # Some workers have both self.critical and self.signals.critical 
            # errors but only one of them is connected --> emit both just 
            # in case
            try:
                self.critical.emit(error)
            except Exception as err:
                self.signals.critical.emit(error)
                
            try:
                self.signals.critical.emit(error)
            except Exception as err:
                self.critical.emit(error)
            
            try:
                self.mutex.unlock()
            except Exception as err:
                pass
    return run

class workerLogger:
    def __init__(self, sigProcess):
        self.sigProcess = sigProcess

    def log(self, message, level='INFO'):
        self.sigProcess.emit(str(message), level)

class signals(QObject):
    progress = Signal(str, object)
    finished = Signal(object)
    initProgressBar = Signal(int)
    progressBar = Signal(int)
    critical = Signal(object)
    dataIntegrityWarning = Signal(str)
    dataIntegrityCritical = Signal()
    sigLoadingFinished = Signal()
    sigLoadingNewChunk = Signal(object)
    resetInnerPbar = Signal(int)
    progress_tqdm = Signal(int)
    signal_close_tqdm = Signal()
    create_tqdm = Signal(int)
    innerProgressBar = Signal(int)
    sigPermissionError = Signal(str, object)
    sigSelectSegmFiles = Signal(object, object)
    sigSelectAcdcOutputFiles = Signal(object, object, str, bool, bool)
    sigSelectSpotmaxRun = Signal(object, object, object, str, bool, bool)
    sigSetMeasurements = Signal(object)
    sigInitAddMetrics = Signal(object, object)
    sigUpdatePbarDesc = Signal(str)
    sigComputeVolume = Signal(int, object)
    sigAskStopFrame = Signal(object)
    sigWarnMismatchSegmDataShape = Signal(object)
    sigErrorsReport = Signal(dict, dict, dict)
    sigMissingAcdcAnnot = Signal(dict)
    sigRecovery = Signal(object)
    sigInitInnerPbar = Signal(int)
    sigUpdateInnerPbar = Signal(int)
    sigSelectFile = Signal(str, str, str)
    sigAskCopyCca = Signal(str)
    sigSelectFilesWithText = Signal(str, object, str, object)

class AutoPilotWorker(QObject):
    finished = Signal()
    critical = Signal(object)
    progress = Signal(str, object)
    sigStarted = Signal()
    sigStopTimer = Signal()

    def __init__(self, guiWin):
        QObject.__init__(self)
        self.logger = workerLogger(self.progress)
        self.guiWin = guiWin
        self.app = guiWin.app
        # self.timer = timer
    
    def timerCallback(self):
        pass
    
    def stop(self):
        self.sigStopTimer.emit()
        self.finished.emit()        
    
    def run(self):
        self.sigStarted.emit()

class FindNextNewIdWorker(QObject):
    def __init__(self, posData, guiWin):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.posData = posData
        self.guiWin = guiWin
    
    @worker_exception_handler
    def run(self):
        prev_IDs = None
        next_frame_i = -1
        for frame_i, data_dict in enumerate(self.posData.allData_li):
            lab = data_dict['labels']
            rp = data_dict['regionprops']
            IDs = data_dict['IDs']
            if lab is None:
                lab = self.posData.segm_data[frame_i]
                rp = skimage.measure.regionprops(lab)
                IDs = [obj.label for obj in rp]
            
            if prev_IDs is None:
                prev_IDs = IDs
                continue
            
            newIDs = [ID for ID in IDs if ID not in prev_IDs]
            if newIDs:
                next_frame_i = frame_i
                break            
            prev_IDs = IDs
            
        self.signals.finished.emit(next_frame_i)

class AlignDataWorker(QObject):
    sigWarnTifAligned = Signal(object, object, object)
    sigAskAlignSegmData = Signal()
        
    def __init__(self, posData, dataPrepWin, mutex, waitCond):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.posData = posData
        self.dataPrepWin = dataPrepWin
        self.mutex = mutex
        self.waitCond = waitCond
        self.doNotAlignSegmData = False
        self.doAbort = False
    
    def set_attr(self, align, user_ch_name):
        self.align = align
        self.user_ch_name = user_ch_name
    
    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def restart(self):
        self.waitCond.wakeAll()
    
    def emitWarnTifAligned(self, numFramesWith0s, tif, posData):
        self.sigWarnTifAligned.emit(numFramesWith0s, tif, posData)
        self.pause()
    
    def emitSigAskAlignSegmData(self):
        self.sigAskAlignSegmData.emit()
        self.pause()
    
    def _align_data(self):
        _zip = zip(self.posData.tif_paths, self.posData.npz_paths)
        aligned = False
        self.posData.all_npz_paths = [
            tif.replace('.tif', '_aligned.npz') for tif in self.posData.tif_paths
        ]
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or self.posData.loaded_shifts is None

            filename_tif = os.path.basename(tif)
            user_ch_filename = f'{self.posData.basename}{self.user_ch_name}.tif'

            if not doAlign:
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                if os.path.exists(_npz):
                    self.posData.all_npz_paths[i] = _npz
                continue
            
            if filename_tif != user_ch_filename:
                continue
            
            # Align based on user_ch_name
            aligned = True
            if self.align:
                self.logger.log(f'Aligning: {tif}')
            tif_data = load.imread(tif)
            numFramesWith0s = self.dataPrepWin.detectTifAlignment(
                tif_data, self.posData
            )
            if self.align:
                self.emitWarnTifAligned(
                    numFramesWith0s, tif, self.posData
                )
                if self.doAbort:
                    return

            # Alignment routine
            if self.posData.SizeZ>1:
                align_func = core.align_frames_3D
                df = self.posData.segmInfo_df.loc[self.posData.filename]
                zz = df['z_slice_used_dataPrep'].to_list()
                if not self.posData.filename.endswith('aligned') and self.align:
                    # Add aligned channel to segmInfo
                    df_aligned = self.posData.segmInfo_df.rename(
                        index={self.posData.filename: f'{self.posData.filename}_aligned'}
                    )
                    self.posData.segmInfo_df = pd.concat(
                        [self.posData.segmInfo_df, df_aligned]
                    )
                    self.posData.segmInfo_df.to_csv(self.posData.segmInfo_df_csv_path)
            else:
                align_func = core.align_frames_2D
                zz = None
            
            if self.align:
                self.signals.initProgressBar.emit(len(tif_data))
                aligned_frames, shifts = align_func(
                    tif_data, slices=zz, user_shifts=self.posData.loaded_shifts,
                    sigPyqt=self.signals.progressBar
                )
                self.posData.loaded_shifts = shifts
            else:
                aligned_frames = tif_data
                
            if self.align:
                self.signals.initProgressBar.emit(0)
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                self.logger.log(f'Saving: {_npz}')
                temp_npz = self.dataPrepWin.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_npz, _npz)
                np.save(self.posData.align_shifts_path, self.posData.loaded_shifts)
                self.posData.all_npz_paths[i] = _npz

                self.logger.log(f'Saving: {tif}')
                temp_tif = self.dataPrepWin.getTempfilePath(tif)
                myutils.to_tiff(temp_tif, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_tif, tif)
                self.posData.img_data = load.imread(tif)

        _zip = zip(self.posData.tif_paths, self.posData.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or aligned

            if not doAlign:
                continue
            
            if tif.endswith(f'{self.user_ch_name}.tif'):
                continue
            
            # Align the other channels
            if self.posData.loaded_shifts is None:
                break
            if self.align:
                self.logger.log(f'Aligning: {tif}')
            tif_data = load.imread(tif)

            # Alignment routine
            if self.posData.SizeZ>1:
                align_func = core.align_frames_3D
                df = self.posData.segmInfo_df.loc[self.posData.filename]
                zz = df['z_slice_used_dataPrep'].to_list()
            else:
                align_func = core.align_frames_2D
                zz = None
            if self.align:
                self.signals.initProgressBar.emit(len(tif_data))
                aligned_frames, shifts = align_func(
                    tif_data, slices=zz, user_shifts=self.posData.loaded_shifts,
                    sigPyqt=self.signals.progressBar
                )
            else:
                aligned_frames = tif_data
            
            _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
            
            if self.align:
                self.signals.initProgressBar.emit(0)
                self.logger.log(f'Saving: {_npz}')
                temp_npz = self.dataPrepWin.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_npz, _npz)
                self.posData.all_npz_paths[i] = _npz

                self.logger.log(f'Saving: {tif}')
                temp_tif = self.dataPrepWin.getTempfilePath(tif)
                myutils.to_tiff(temp_tif, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_tif, tif)

        if not aligned:
            return
        
        if not self.posData.segmFound:
            return
        
        # Align segmentation data accordingly
        self.segmAligned = False
        if self.posData.loaded_shifts is None or not self.align:
            return
        
        self.emitSigAskAlignSegmData()
        if self.doNotAlignSegmData:
            return
        
        self.dataPrepWin.segmAligned = True
        self.logger.log(f'Aligning: {self.posData.segm_npz_path}')
        self.posData.segm_data, shifts = core.align_frames_2D(
            self.posData.segm_data, slices=None,
            user_shifts=self.posData.loaded_shifts
        )
        self.logger.log(f'Saving: {self.posData.segm_npz_path}')
        temp_npz = self.dataPrepWin.getTempfilePath(self.posData.segm_npz_path)
        np.savez_compressed(temp_npz, self.posData.segm_data)
        self.dataPrepWin.storeTempFileMove(temp_npz, self.posData.segm_npz_path)

    @worker_exception_handler
    def run(self):     
        self._align_data()     
        self.signals.finished.emit(self)

class LabelRoiWorker(QObject):
    finished = Signal()
    critical = Signal(object)
    progress = Signal(str, object)
    sigProgressBar = Signal(int)
    sigLabellingDone = Signal(object, bool)

    def __init__(self, Gui):
        QObject.__init__(self)
        self.logger = workerLogger(self.progress)
        self.Gui = Gui
        self.mutex = Gui.labelRoiMutex
        self.waitCond = Gui.labelRoiWaitCond
        self.exit = False
        self.started = False
    
    def pause(self):
        self.logger.log('Draw box around object to start magic labeller.')
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def start(self, roiImg, posData, roiSecondChannel=None, isTimelapse=False):
        self.posData = posData
        self.isTimelapse = isTimelapse
        self.imageData = roiImg
        self.roiSecondChannel = roiSecondChannel
        self.restart()
    
    def restart(self, log=True):
        if log:
            self.logger.log('Magic labeller started...')
        self.started = True
        self.waitCond.wakeAll()
    
    def _stop(self):
        self.logger.log('Magic labeller backend process done. Closing it...')
        self.exit = True
        self.waitCond.wakeAll()
    
    def _segment_image(self, img, secondChannelImg):
        if secondChannelImg is not None:
            img = self.Gui.labelRoiModel.to_rgb_stack(
                img, secondChannelImg
            )
        
        lab = core.segm_model_segment(
            self.Gui.labelRoiModel, img, self.Gui.model_kwargs, 
            preproc_recipe=self.Gui.preproc_recipe, 
            posData=self.posData
        )
        if self.Gui.applyPostProcessing:
            lab = core.post_process_segm(
                lab, **self.Gui.standardPostProcessKwargs
            )
            if self.Gui.customPostProcessFeatures:
                lab = features.custom_post_process_segm(
                    self.posData, self.Gui.customPostProcessGroupedFeatures, 
                    lab, img, self.posData.frame_i, self.posData.filename, 
                    self.posData.user_ch_name, 
                    self.Gui.customPostProcessFeatures
                )
        return lab
    
    @worker_exception_handler
    def run(self):
        while not self.exit:
            if self.exit:
                break
            elif self.started:
                if self.isTimelapse:
                    segmData = np.zeros(self.imageData.shape, dtype=np.uint16)
                    for frame_i, img in enumerate(self.imageData):
                        if self.roiSecondChannel is not None:
                            secondChannelImg = self.roiSecondChannel[frame_i]
                        else:
                            secondChannelImg = None
                        lab = self._segment_image(img, secondChannelImg)
                        segmData[frame_i] = lab
                        self.sigProgressBar.emit(1)
                else:
                    img = self.imageData
                    secondChannelImg = self.roiSecondChannel
                    segmData = self._segment_image(img, secondChannelImg)
                
                self.sigLabellingDone.emit(segmData, self.isTimelapse)
                self.started = False
            self.pause()
        self.finished.emit()

class StoreGuiStateWorker(QObject):
    finished = Signal(object)
    sigDone = Signal()
    progress = Signal(str, object)

    def __init__(self, mutex, waitCond):
        QObject.__init__(self)
        self.mutex = mutex
        self.waitCond = waitCond
        self.exit = False
        self.isFinished = False
        self.q = queue.Queue()
        self.logger = workerLogger(self.progress)
    
    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def enqueue(self, posData, img1):
        self.q.put((posData, img1))
        self.waitCond.wakeAll()
    
    def _stop(self):
        self.exit = True
        self.waitCond.wakeAll()
    
    def run(self):
        while True:
            if self.exit:
                self.logger.log('Closing store state worker...')
                break
            elif not self.q.empty():
                posData, img1 = self.q.get()
                # self.logger.log('Storing state...')
                if posData.cca_df is not None:
                    cca_df = posData.cca_df.copy()
                else:
                    cca_df = None

                state = {
                    'image': img1.copy(),
                    'labels': posData.storedLab.copy(),
                    'editID_info': posData.editID_info.copy(),
                    'binnedIDs': posData.binnedIDs.copy(),
                    'ripIDs': posData.ripIDs.copy(),
                    'cca_df': cca_df
                }
                posData.UndoRedoStates[posData.frame_i].insert(0, state)
                if self.q.empty():
                    # self.logger.log('State stored...')
                    self.sigDone.emit()
            else:
                self.pause()
            
        self.isFinished = True
        self.finished.emit(self)

class AutoSaveWorker(QObject):
    finished = Signal(object)
    sigDone = Signal()
    critical = Signal(object)
    progress = Signal(str, object)
    sigStartTimer = Signal(object, object)
    sigStopTimer = Signal()
    sigAutoSaveCannotProceed = Signal()

    def __init__(self, mutex, waitCond, savedSegmData):
        QObject.__init__(self)
        self.savedSegmData = savedSegmData
        self.logger = workerLogger(self.progress)
        self.mutex = mutex
        self.waitCond = waitCond
        self.exit = False
        self.isFinished = False
        self.abortSaving = False
        self.isSaving = False
        self.isPaused = False
        self.dataQ = deque(maxlen=5)
        self.isAutoSaveON = False
        self.debug = False
    
    def pause(self):
        if self.debug:
            self.logger.log('Autosaving is idle.')
        self.mutex.lock()
        self.isPaused = True
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        self.isPaused = False
    
    def enqueue(self, posData):
        # First stop previously saving data
        if self.isSaving:
            self.abortSaving = True
        self._enqueue(posData)
    
    def _enqueue(self, posData):
        if self.debug:
            self.logger.log('Enqueing posData autosave...')
        self.dataQ.append(posData)
        if len(self.dataQ) == 1:
            # Wake worker upon inserting first element
            self.abortSaving = False
            self.waitCond.wakeAll()
    
    def _stop(self):
        self.exit = True
        self.waitCond.wakeAll()
    
    def abort(self):
        self.abortSaving = True
        while not len(self.dataQ) == 0:
            data = self.dataQ.pop()
            del data
        self._stop()
    
    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log('Closing autosaving worker...')
                break
            elif not len(self.dataQ) == 0:
                if self.debug:
                    self.logger.log('Autosaving...')
                data = self.dataQ.pop()
                try:
                    self.saveData(data)
                except Exception as e:
                    error = traceback.format_exc()
                    print('*'*40)
                    self.logger.log(error)
                    print('='*40)
                if len(self.dataQ) == 0:
                    self.sigDone.emit()
            else:
                self.pause()
        self.isFinished = True
        self.finished.emit(self)
        if self.debug:
            self.logger.log('Autosave finished signal emitted')
    
    def getLastTrackedFrame(self, posData):
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break
        if frame_i > 0:
            return frame_i
        else:
            return last_tracked_i
    
    def saveData(self, posData):
        if self.debug:
            self.logger.log('Started autosaving...')
        
        self.isSaving = True
        try:
            posData.setTempPaths()
        except Exception as e:
            self.logger.log(
                '[WARNING]: Cell-ACDC cannot create the recovery folder for '
                'the autosaving process. Autosaving will be turned off.'
            )
            self.sigAutoSaveCannotProceed.emit()
            return
        segm_npz_path = posData.segm_npz_temp_path

        end_i = self.getLastTrackedFrame(posData)
        
        if self.isAutoSaveON:
            if end_i < len(posData.segm_data):
                saved_segm_data = posData.segm_data
            else:
                frame_shape = posData.segm_data.shape[1:]
                segm_shape = (end_i+1, *frame_shape)
                saved_segm_data = np.zeros(segm_shape, dtype=np.uint32)
        
        keys = []
        acdc_df_li = []
        
        for frame_i, data_dict in enumerate(posData.allData_li[:end_i+1]):
            if self.abortSaving:
                break
            
            # Build saved_segm_data
            lab = data_dict['labels']
            if lab is None:
                break
            
            if self.isAutoSaveON:
                if posData.SizeT > 1:
                    saved_segm_data[frame_i] = lab
                else:
                    saved_segm_data = lab

            acdc_df = data_dict['acdc_df']
            
            if acdc_df is None:
                continue

            if not np.any(lab):
                continue

            acdc_df = load.pd_bool_to_int(acdc_df, inplace=False)
            acdc_df_li.append(acdc_df)
            key = (frame_i, posData.TimeIncrement*frame_i)
            keys.append(key)

            if self.abortSaving:
                break
        
        if not self.abortSaving:            
            if self.isAutoSaveON:
                segm_data = np.squeeze(saved_segm_data)
                self._saveSegm(segm_npz_path, segm_data)
            
            if acdc_df_li:
                all_frames_acdc_df = pd.concat(
                    acdc_df_li, keys=keys,
                    names=['frame_i', 'time_seconds', 'Cell_ID']
                )
                self._save_acdc_df(all_frames_acdc_df, posData)

        if self.debug:
            self.logger.log(f'Autosaving done.')
            self.logger.log(f'Aborted autosaving {self.abortSaving}.')

        self.abortSaving = False
        self.isSaving = False
    
    def _saveSegm(self, recovery_path, data):
        try:
            equalToSavedSegm = np.all(self.savedSegmData == data)
        except Exception as err:
            return
        
        if equalToSavedSegm:
            return
        else:
            np.savez_compressed(recovery_path, np.squeeze(data))
    
    def _save_acdc_df(self, recovery_acdc_df, posData):
        recovery_folderpath = posData.recoveryFolderpath()
        if not os.path.exists(posData.acdc_output_csv_path):
            load.store_unsaved_acdc_df(recovery_folderpath, recovery_acdc_df)
            return

        saved_acdc_df_path = posData.acdc_output_csv_path
        saved_acdc_df = (
            pd.read_csv(saved_acdc_df_path, dtype=load.acdc_df_str_cols)
            .set_index(['frame_i', 'Cell_ID'])
        )
        
        recovery_acdc_df = (
            recovery_acdc_df.reset_index().set_index(['frame_i', 'Cell_ID'])
        )
        try:
            # Try to insert into the recovery_acdc_df any column that was saved
            # but is not in the recovered df (e.g., metrics)
            df_left = recovery_acdc_df
            existing_cols = df_left.columns.intersection(saved_acdc_df.columns)
            df_right = saved_acdc_df.drop(columns=existing_cols)
            recovery_acdc_df = df_left.join(df_right, how='left')
        except Exception as error:
            self.logger.log(f'[WARNING]: {error}')
        
        # Check if last saved acdc_df is equal
        last_unsaved_csv_path = load.get_last_stored_unsaved_acdc_df_filepath(
            recovery_folderpath
        )
        if last_unsaved_csv_path is None:
            reference_acdc_df = saved_acdc_df
        else:
            reference_acdc_df = (
                pd.read_csv(last_unsaved_csv_path, dtype=load.acdc_df_str_cols)
                .set_index(['frame_i', 'Cell_ID'])
            )
        
        if myutils.are_acdc_dfs_equal(recovery_acdc_df, reference_acdc_df):
            return
        
        load.store_unsaved_acdc_df(recovery_folderpath, recovery_acdc_df)

class segmWorker(QObject):
    finished = Signal(np.ndarray, float)
    debug = Signal(object)
    critical = Signal(object)

    def __init__(self, mainWin, secondChannelData=None):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.logger = self.mainWin.logger
        self.z_range = None
        self.secondChannelData = secondChannelData

    @worker_exception_handler
    def run(self):
        t0 = time.perf_counter()
        if self.mainWin.segment3D:
            img = self.mainWin.getDisplayedZstack()
            if self.z_range is not None:
                startZ, stopZ = self.z_range
                img = img[startZ:stopZ+1]
        else:
            img = self.mainWin.getDisplayedImg1()
        
        posData = self.mainWin.data[self.mainWin.pos_i]
        lab = np.zeros_like(posData.segm_data[0])

        if self.secondChannelData is not None:
            img = self.mainWin.model.to_rgb_stack(img, self.secondChannelData)

        start_z_slice = 0
        if self.z_range is not None:
            start_z_slice, _ = self.z_range
        elif not self.mainWin.segment3D and posData.isSegm3D:
            idx = (posData.filename, posData.frame_i)
            start_z_slice = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
        
        _lab = core.segm_model_segment(
            self.mainWin.model, img, self.mainWin.model_kwargs, 
            frame_i=posData.frame_i, 
            posData=posData, 
            start_z_slice=start_z_slice
        )
        posData.saveSamEmbeddings(logger_func=self.logger.info)
        if self.mainWin.applyPostProcessing:
            _lab = core.post_process_segm(
                _lab, **self.mainWin.standardPostProcessKwargs
            )
            if self.mainWin.customPostProcessFeatures:
                _lab = features.custom_post_process_segm(
                    posData, self.mainWin.customPostProcessGroupedFeatures, 
                    _lab, img, posData.frame_i, posData.filename, 
                    posData.user_ch_name, self.mainWin.customPostProcessFeatures
                )
        
        if self.z_range is not None:
            # 3D segmentation of a z-slices subset
            startZ, stopZ = self.z_range
            lab[startZ:stopZ+1] = _lab
        elif not self.mainWin.segment3D and posData.isSegm3D:
            # 3D segmentation but segmented current z-slice
            idx = (posData.filename, posData.frame_i)
            z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            lab[z] = _lab
        else:
            # Either whole z-stack or 2D segmentation
            lab = _lab
        
        t1 = time.perf_counter()
        exec_time = t1-t0
        self.finished.emit(lab, exec_time)

class segmVideoWorker(QObject):
    finished = Signal(float)
    debug = Signal(object)
    critical = Signal(object)
    progressBar = Signal(int)
    progress = Signal(str, object)

    def __init__(self, posData, paramWin, model, startFrameNum, stopFrameNum):
        QObject.__init__(self)
        self.standardPostProcessKwargs = paramWin.standardPostProcessKwargs
        self.applyPostProcessing = paramWin.applyPostProcessing
        self.customPostProcessFeatures = paramWin.customPostProcessFeatures
        self.customPostProcessGroupedFeatures = (
            paramWin.customPostProcessGroupedFeatures
        )
        self.model_kwargs = paramWin.model_kwargs
        self.preproc_recipe = paramWin.preproc_recipe
        self.secondChannelName = paramWin.secondChannelName
        self.model = model
        self.posData = posData
        self.startFrameNum = startFrameNum
        self.stopFrameNum = stopFrameNum
        self.logger = workerLogger(self.progress)

    def _check_extend_segm_data(self, segm_data, stop_frame_num):
        if stop_frame_num <= len(segm_data):
            return segm_data
        extended_shape = (stop_frame_num, *segm_data.shape[1:])
        extended_segm_data = np.zeros(extended_shape, dtype=segm_data.dtype)
        extended_segm_data[:len(segm_data)] = segm_data
        if len(extended_shape) == 4:
            return extended_segm_data
        if self.posData.SizeZ == 1:
            return extended_segm_data
        else:
            num_added_frames = len(extended_segm_data) - len(segm_data)
            half_z = int(self.posData.SizeZ/2)
            # 2D segm on 3D over time data --> fix segmInfo
            segmInfo_extended = pd.DataFrame({
                'filename': [self.posData.filename]*num_added_frames,
                'frame_i': list(range(len(segm_data), len(extended_segm_data))),
                'z_slice_used_gui': [half_z]*num_added_frames,
                'which_z_proj_gui': ['single z-slice']*num_added_frames
            }).set_index(['filename', 'frame_i'])
            segmInfo_df = pd.concat([self.posData.segmInfo_df, segmInfo_extended])
            self.posData.segmInfo_df = segmInfo_df
            self.posData.segmInfo_df.to_csv(self.posData.segmInfo_df_csv_path)
        return extended_segm_data

    @worker_exception_handler
    def run(self):
        t0 = time.perf_counter()
        self.posData.segm_data = self._check_extend_segm_data(
            self.posData.segm_data, self.stopFrameNum
        )
        img_data = self.posData.img_data[self.startFrameNum-1:self.stopFrameNum]
        is4D = img_data.ndim == 4
        is2D_segm = self.posData.segm_data.ndim == 3
        if is4D and is2D_segm:
            filename = self.posData.filename
            zz = self.posData.segmInfo_df.loc[filename, 'z_slice_used_gui']
        else:
            zz = None
        for i, img in enumerate(img_data):
            frame_i = i+self.startFrameNum-1
            if self.secondChannelData is not None:
                img = self.model.to_rgb_stack(img, self.secondChannelData)
            if zz is not None:
                z_slice = zz.loc[frame_i]
                img = img[z_slice]
                
            lab = core.segm_model_segment(
                self.model, img, self.model_kwargs, frame_i=frame_i, 
                preproc_recipe=self.preproc_recipe, 
                posData=self.posData
            )
            self.posData.saveSamEmbeddings(logger_func=self.logger.log)
            if self.applyPostProcessing:
                lab = core.post_process_segm(
                    lab, **self.standardPostProcessKwargs
                )
                if self.customPostProcessFeatures:
                    lab = features.custom_post_process_segm(
                        self.posData, 
                        self.customPostProcessGroupedFeatures, 
                        lab, img, self.posData.frame_i, 
                        self.posData.filename, 
                        self.posData.user_ch_name, 
                        self.customPostProcessFeatures
                    )
            self.posData.segm_data[frame_i] = lab
            self.progressBar.emit(1)
        t1 = time.perf_counter()
        exec_time = t1-t0
        self.finished.emit(exec_time)

class calcMetricsWorker(QObject):
    progressBar = Signal(int, int, float)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = signals()
        self.abort = False
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.signals.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            return True
        else:
            return False

    @worker_exception_handler
    def run(self):
        np.seterr(invalid='ignore')
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.standardMetricsErrors = {}
            self.customMetricsErrors = {}
            self.regionPropsErrors = {}
            tot_pos = len(pos_foldernames)
            self.allPosDataInputs = []
            posDatas = []
            self.logger.log('-'*30)
            expFoldername = os.path.basename(exp_path)

            if i == 0:
                abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
                if abort:
                    self.signals.finished.emit(self)
                    return

            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                basename, chNames = myutils.getBasenameAndChNames(
                    images_path, useExt=('.tif', '.h5')
                )

                self.signals.sigUpdatePbarDesc.emit(f'Loading {pos_path}...')

                # Use first found channel, it doesn't matter for metrics
                chName = chNames[0]
                file_path = myutils.getChannelFilePath(images_path, chName)

                # Load data
                posData = load.loadData(file_path, chName)
                posData.getBasenameAndChNames(useExt=('.tif', '.h5'))
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_acdc_df=True,
                    load_metadata=True,
                    loadSegmInfo=True,
                    load_customCombineMetrics=True
                )

                posDatas.append(posData)

                self.allPosDataInputs.append({
                    'file_path': file_path,
                    'chName': chName,
                    'combineMetricsConfig': posData.combineMetricsConfig,
                    'combineMetricsPath': posData.custom_combine_metrics_path
                })

            if any([posData.SizeT > 1 for posData in posDatas]):
                self.mutex.lock()
                self.signals.sigAskStopFrame.emit(posDatas)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                if self.abort:
                    self.signals.finished.emit(self)
                    return
                for p, posData in enumerate(posDatas):
                    self.allPosDataInputs[p]['stopFrameNum'] = posData.stopFrameNum
                # remove posDatas from memory for timelapse data
                # del posDatas
            else:
                for p, posData in enumerate(posDatas):
                    self.allPosDataInputs[p]['stopFrameNum'] = 1
            
            # Iterate pos and calculate metrics
            numPos = len(self.allPosDataInputs)
            for p, posDataInputs in enumerate(self.allPosDataInputs):
                self.logger.log('='*40)
                file_path = posDataInputs['file_path']
                chName = posDataInputs['chName']
                stopFrameNum = posDataInputs['stopFrameNum']

                posData = load.loadData(file_path, chName)

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames(useExt=('.tif', '.h5'))
                posData.buildPaths()
                posData.loadImgData()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_shifts=True,
                    loadSegmInfo=True,
                    load_delROIsInfo=True,
                    load_bkgr_data=True,
                    loadBkgrROIs=True,
                    load_last_tracked_i=True,
                    load_metadata=True,
                    load_customAnnot=True,
                    load_customCombineMetrics=True,
                    end_filename_segm=self.mainWin.endFilenameSegm,
                    load_dataPrep_ROIcoords=True
                )
                posData.labelSegmData()
                if not posData.segmFound:
                    relPath = (
                        f'...{os.sep}{expFoldername}'
                        f'{os.sep}{posData.pos_foldername}'
                    )
                    self.logger.log(
                        f'Skipping "{relPath}" '
                        f'because segm. file was not found.'
                    )
                    continue

                # if posData.SizeT > 1:
                #     self.mainWin.gui.data = [None]*numPos
                # else:
                #     self.mainWin.gui.data = posDatas

                if p == 0:
                    self.mainWin.gui.data = posDatas
                else:
                    self.mainWin.gui.data[p-1] = posDatas[p-1]
                    
                self.mainWin.gui.pos_i = p
                self.mainWin.gui.data[p] = posData
                
                self.mainWin.gui.last_pos = numPos

                self.mainWin.gui.isSegm3D = posData.getIsSegm3D()
                posData.isSegm3D = self.mainWin.gui.isSegm3D

                # Allow single 2D/3D image
                if posData.SizeT == 1:
                    posData.img_data = posData.img_data[np.newaxis]
                    posData.segm_data = posData.segm_data[np.newaxis]

                self.logger.log(
                    'Loaded paths:\n'
                    f'Segmentation file name: {os.path.basename(posData.segm_npz_path)}\n'
                    f'ACDC output file name: {os.path.basename(posData.acdc_output_csv_path)}'
                )

                if p == 0 and i==0:
                    self.mutex.lock()
                    self.signals.sigInitAddMetrics.emit(
                        posData, self.allPosDataInputs
                    )
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    if self.abort:
                        self.signals.finished.emit(self)
                        return
                
                guiWin = self.mainWin.gui
                guiWin.init_segmInfo_df()
                addMetrics_acdc_df = guiWin.saveDataWorker.addMetrics_acdc_df
                addVolumeMetrics = guiWin.saveDataWorker.addVolumeMetrics
                addVelocityMeasurement = (
                    guiWin.saveDataWorker.addVelocityMeasurement
                )

                # Load the other channels
                posData.loadedChNames = []
                for fluoChName in posData.chNames:
                    if fluoChName in self.mainWin.gui.chNamesToSkip:
                        continue

                    if fluoChName == chName:
                        filename = posData.filename
                        posData.fluo_data_dict[filename] = posData.img_data
                        posData.fluo_bkgrData_dict[filename] = posData.bkgrData
                        posData.loadedChNames.append(chName)
                        continue

                    fluo_path, filename = self.mainWin.gui.getPathFromChName(
                        fluoChName, posData
                    )
                    if fluo_path is None:
                        continue

                    self.logger.log(f'Loading {fluoChName} data...')
                    fluo_data, bkgrData = self.mainWin.gui.load_fluo_data(
                        fluo_path, isGuiThread=False
                    )
                    if fluo_data is None:
                        continue

                    if posData.SizeT == 1:
                        # Add single frame for snapshot data
                        fluo_data = fluo_data[np.newaxis]

                    posData.loadedChNames.append(fluoChName)
                    posData.loadedFluoChannels.add(fluoChName)
                    posData.fluo_data_dict[filename] = fluo_data
                    posData.fluo_bkgrData_dict[filename] = bkgrData
                
                # Recreate allData_li attribute of the gui
                posData.allData_li = []
                for frame_i, lab in enumerate(posData.segm_data[:stopFrameNum]):
                    data_dict = {
                        'labels': lab,
                        'regionprops': skimage.measure.regionprops(lab)
                    }
                    posData.allData_li.append(data_dict)

                # Signal to compute volume in the main thread
                self.mutex.lock()
                self.signals.sigComputeVolume.emit(stopFrameNum, posData)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()

                guiWin.initMetricsToSave(posData)

                if not posData.fluo_data_dict:
                    self.logger.log(
                        'None of the signals were loaded from the path: '
                        f'"{posData.pos_path}"'
                    )

                acdc_df_li = []
                keys = []
                self.signals.initProgressBar.emit(stopFrameNum)
                for frame_i, data_dict in enumerate(posData.allData_li[:stopFrameNum]):
                    if self.abort:
                        self.signals.finished.emit(self)
                        return

                    lab = data_dict['labels']
                    if not np.any(lab):
                        # Empty segmentation mask --> skip
                        continue

                    rp = data_dict['regionprops']
                    posData.lab = lab
                    posData.rp = rp

                    if posData.acdc_df is None:
                        acdc_df = myutils.getBaseAcdcDf(rp)
                    else:
                        try:
                            acdc_df = posData.acdc_df.loc[frame_i].copy()
                        except:
                            acdc_df = myutils.getBaseAcdcDf(rp)

                    try:
                        if posData.fluo_data_dict:
                            acdc_df = addMetrics_acdc_df(
                                acdc_df, rp, frame_i, lab, posData
                            )
                            if guiWin.saveDataWorker.abort:
                                self.abort = True
                                self.signals.finished.emit(self)
                                return
                        else:
                            acdc_df = addVolumeMetrics(
                                acdc_df, rp, posData
                            )
                        acdc_df_li.append(acdc_df)
                        key = (frame_i, posData.TimeIncrement*frame_i)
                        keys.append(key)
                    except Exception as error:
                        traceback_format = traceback.format_exc()
                        print('-'*30)      
                        self.logger.log(traceback_format)
                        print('-'*30)
                        self.standardMetricsErrors[str(error)] = traceback_format
                    
                    try:
                        prev_data_dict = posData.allData_li[frame_i-1]
                        prev_lab = prev_data_dict['labels']
                        acdc_df = addVelocityMeasurement(
                            acdc_df, prev_lab, lab, posData
                        )
                    except Exception as error:
                        traceback_format = traceback.format_exc()
                        print('-'*30)      
                        self.logger.log(traceback_format)
                        print('-'*30)
                        self.standardMetricsErrors[str(error)] = traceback_format

                    self.signals.progressBar.emit(1)

                if debugging:
                    continue

                if not acdc_df_li:
                    print('-'*30)
                    self.logger.log(
                        'All selected positions in the experiment folder '
                        f'{expFoldername} have EMPTY segmentation mask. '
                        'Metrics will not be saved.'
                    )
                    print('-'*30)
                    continue

                all_frames_acdc_df = pd.concat(
                    acdc_df_li, keys=keys,
                    names=['frame_i', 'time_seconds', 'Cell_ID']
                )
                self.mainWin.gui.saveDataWorker.addCombineMetrics_acdc_df(
                    posData, all_frames_acdc_df
                )
                self.mainWin.gui.saveDataWorker.addAdditionalMetadata(
                    posData, all_frames_acdc_df
                )
                self.logger.log(
                    f'Saving acdc_output to: "{posData.acdc_output_csv_path}"'
                )
                try:
                    all_frames_acdc_df.to_csv(posData.acdc_output_csv_path)
                except PermissionError:
                    traceback_str = traceback.format_exc()
                    self.mutex.lock()
                    self.signals.sigPermissionError.emit(
                        traceback_str, posData.acdc_output_csv_path
                    )
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    all_frames_acdc_df.to_csv(posData.acdc_output_csv_path)

                if self.abort:
                    self.signals.finished.emit(self)
                    return

            self.logger.log('*'*30)

            self.mutex.lock()
            self.signals.sigErrorsReport.emit(
                self.standardMetricsErrors, self.customMetricsErrors,
                self.regionPropsErrors
            )
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()

        self.signals.finished.emit(self)

class loadDataWorker(QObject):
    def __init__(self, mainWin, user_ch_file_paths, user_ch_name, firstPosData):
        QObject.__init__(self)
        self.signals = signals()
        self.mainWin = mainWin
        self.user_ch_file_paths = user_ch_file_paths
        self.user_ch_name = user_ch_name
        self.logger = workerLogger(self.signals.progress)
        self.mutex = self.mainWin.loadDataMutex
        self.waitCond = self.mainWin.loadDataWaitCond
        self.firstPosData = firstPosData
        self.abort = False
        self.loadUnsaved = False
        self.recoveryAsked = False

    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def checkSelectedDataShape(self, posData, numPos):
        skipPos = False
        abort = False
        emitWarning = (
            not posData.segmFound and posData.SizeT > 1
            and not self.mainWin.isNewFile
        )
        if emitWarning:
            self.signals.dataIntegrityWarning.emit(posData.pos_foldername)
            self.pause()
            abort = self.abort
        return skipPos, abort
    
    def warnMismatchSegmDataShape(self, posData):
        self.skipPos = False
        self.mutex.lock()
        self.signals.sigWarnMismatchSegmDataShape.emit(posData)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.skipPos

    @worker_exception_handler
    def run(self):
        data = []
        user_ch_file_paths = self.user_ch_file_paths
        numPos = len(self.user_ch_file_paths)
        user_ch_name = self.user_ch_name
        self.signals.initProgressBar.emit(len(user_ch_file_paths))
        for i, file_path in enumerate(user_ch_file_paths):
            if i == 0:
                posData = self.firstPosData
                segmFound = self.firstPosData.segmFound
                loadSegm = False
            else:
                posData = load.loadData(file_path, user_ch_name)
                loadSegm = True

            self.logger.log(f'Loading {posData.relPath}...')

            posData.loadSizeS = self.mainWin.loadSizeS
            posData.loadSizeT = self.mainWin.loadSizeT
            posData.loadSizeZ = self.mainWin.loadSizeZ
            posData.SizeT = self.mainWin.SizeT
            posData.SizeZ = self.mainWin.SizeZ
            posData.isSegm3D = self.mainWin.isSegm3D

            if i > 0:
                # First pos was already loaded in the main thread
                # see loadSelectedData function in gui.py
                posData.getBasenameAndChNames()
                posData.buildPaths()
                if not self.firstPosData.onlyEditMetadata:
                    posData.loadImgData()

            if self.firstPosData.onlyEditMetadata:
                loadSegm = False

            posData.loadOtherFiles(
                load_segm_data=loadSegm,
                load_acdc_df=True,
                load_shifts=True,
                loadSegmInfo=True,
                load_delROIsInfo=True,
                load_bkgr_data=True,
                loadBkgrROIs=True,
                load_dataPrep_ROIcoords=True,
                load_last_tracked_i=True,
                load_metadata=True,
                load_customAnnot=True,
                load_customCombineMetrics=True,
                end_filename_segm=self.mainWin.selectedSegmEndName,
                create_new_segm=self.mainWin.isNewFile,
                new_endname=self.mainWin.newSegmEndName,
                labelBoolSegm=self.mainWin.labelBoolSegm
            )
            posData.labelSegmData()

            if i == 0:
                posData.segmFound = segmFound

            isPosSegm3D = posData.getIsSegm3D()
            isMismatch = (
                isPosSegm3D != self.mainWin.isSegm3D 
                and isPosSegm3D is not None
                and not self.mainWin.isNewFile
            )
            if isMismatch:
                skipPos = self.warnMismatchSegmDataShape(posData)
                if skipPos:
                    self.logger.log(
                        f'Skipping "{posData.relPath}" because segmentation '
                        'data shape different from first Position loaded.'
                    )
                    continue
                else:
                    data = 'abort'
                    break

            self.logger.log(
                'Loaded paths:\n'
                f'Segmentation file name: {os.path.basename(posData.segm_npz_path)}\n'
                f'ACDC output file name {os.path.basename(posData.acdc_output_csv_path)}'
            )

            posData.SizeT = self.mainWin.SizeT
            if self.mainWin.SizeZ > 1:
                SizeZ = posData.img_data_shape[-3]
                posData.SizeZ = SizeZ
            else:
                posData.SizeZ = 1
            posData.TimeIncrement = self.mainWin.TimeIncrement
            posData.PhysicalSizeZ = self.mainWin.PhysicalSizeZ
            posData.PhysicalSizeY = self.mainWin.PhysicalSizeY
            posData.PhysicalSizeX = self.mainWin.PhysicalSizeX
            posData.isSegm3D = self.mainWin.isSegm3D
            posData.saveMetadata(
                signals=self.signals, mutex=self.mutex, waitCond=self.waitCond,
                additionalMetadata=self.firstPosData._additionalMetadataValues
            )
            if hasattr(posData, 'img_data_shape'):
                SizeY, SizeX = posData.img_data_shape[-2:]

            if posData.SizeZ > 1 and posData.img_data.ndim < 3:
                posData.SizeZ = 1
                posData.segmInfo_df = None
                try:
                    os.remove(posData.segmInfo_df_csv_path)
                except FileNotFoundError:
                    pass

            posData.setBlankSegmData(
                posData.SizeT, posData.SizeZ, SizeY, SizeX
            )
            if not self.firstPosData.onlyEditMetadata:
                skipPos, abort = self.checkSelectedDataShape(posData, numPos)
            else:
                skipPos, abort = False, False

            if skipPos:
                continue
            elif abort:
                data = 'abort'
                break

            posData.setTempPaths(createFolder=False)
            isRecoveredDataPresent = (
                os.path.exists(posData.segm_npz_temp_path)
                or posData.isRecoveredAcdcDfPresent()
            )
            if isRecoveredDataPresent and not self.mainWin.newSegmEndName:
                if not self.recoveryAsked:
                    self.mutex.lock()
                    self.signals.sigRecovery.emit(posData)
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    self.recoveryAsked = True
                    if self.abort:
                        data = 'abort'
                        break
                if self.loadUnsaved:
                    self.logger.log('Loading unsaved data...')
                    if os.path.exists(posData.segm_npz_temp_path):
                        segm_npz_path = posData.segm_npz_temp_path
                        posData.segm_data = np.load(segm_npz_path)['arr_0']
                        segm_filename = os.path.basename(segm_npz_path)
                        posData.segm_npz_path = os.path.join(
                            posData.images_path, segm_filename
                        )
                    
                    posData.loadMostRecentUnsavedAcdcDf()

            # Allow single 2D/3D image
            if posData.SizeT == 1:
                posData.img_data = posData.img_data[np.newaxis]
                posData.segm_data = posData.segm_data[np.newaxis]
            if hasattr(posData, 'img_data_shape'):
                img_shape = posData.img_data_shape
            img_shape = 'Not Loaded'
            if hasattr(posData, 'img_data_shape'):
                datasetShape = posData.img_data.shape
            else:
                datasetShape = 'Not Loaded'
            if posData.segm_data is not None:
                posData.segmSizeT = len(posData.segm_data)
            SizeT = posData.SizeT
            SizeZ = posData.SizeZ
            self.logger.log(f'Full dataset shape = {img_shape}')
            self.logger.log(f'Loaded dataset shape = {datasetShape}')
            self.logger.log(f'Number of frames = {SizeT}')
            self.logger.log(f'Number of z-slices per frame = {SizeZ}')
            data.append(posData)
            self.signals.progressBar.emit(1)

        if not data:
            data = None
            self.signals.dataIntegrityCritical.emit()

        self.signals.finished.emit(data)

class trackingWorker(QObject):
    finished = Signal()
    critical = Signal(object)
    progress = Signal(str)
    debug = Signal(object)

    def __init__(self, posData, mainWin, video_to_track):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.posData = posData
        self.mutex = QMutex()
        self.signals = signals()
        self.waitCond = QWaitCondition()
        self.tracker = self.mainWin.tracker
        self.track_params = self.mainWin.track_params
        self.video_to_track = video_to_track
    
    def _get_first_untracked_lab(self):
        start_frame_i = self.mainWin.start_n - 1
        frameData = self.posData.allData_li[start_frame_i]
        lab = frameData['labels']
        if lab is not None:
            return lab
        else:
            return self.posData.segm_data[start_frame_i]

    def _relabel_first_frame_labels(self, tracked_video):
        first_untracked_lab = self._get_first_untracked_lab()
        self.mainWin.setAllIDs()
        max_allIDs = max(self.posData.allIDs, default=0)
        max_tracked_video = tracked_video.max()
        overall_max = max(max_allIDs, max_tracked_video)
        uniqueID = overall_max + 1
        
        tracked_video = transformation.retrack_based_on_untracked_first_frame(
            tracked_video, first_untracked_lab, uniqueID=uniqueID        
        )
        return tracked_video

    def _setProgressBarIndefiniteWait(self):
        try:
            if hasattr(self.signals, 'innerPbar_available'):
                if self.signals.innerPbar_available:
                    # Use inner pbar of the GUI widget (top pbar is for positions)
                    self.signals.sigInitInnerPbar.emit(1)
                    return
            else:
                self.signals.initProgressBar.emit(1)
        except Exception as err:
            pass
    
    @worker_exception_handler
    def run(self):
        self.mutex.lock()        
        self.progress.emit(
            'Tracking process started (more details in the terminal)...')
        
        trackerInputImage = None
        self.track_params['signals'] = self.signals
        if 'image' in self.track_params:
            trackerInputImage = self.track_params.pop('image')
            start_frame_i = self.mainWin.start_n-1
            stop_frame_n = self.mainWin.stop_n

            trackerInputImage = trackerInputImage[start_frame_i:stop_frame_n]
        
        tracked_video = core.tracker_track(
            self.video_to_track, self.tracker, self.track_params, 
            intensity_img=trackerInputImage,
            logger_func=self.progress.emit
        )
        
        self._setProgressBarIndefiniteWait()
        
        
        self.progress.emit('Re-tracking first frame to ensure continuity...')
        # Relabel first frame objects back to IDs they had before tracking
        # (to ensure continuity with past untracked frames)
        tracked_video = self._relabel_first_frame_labels(tracked_video)
        
        self.progress.emit('Generating annotations...')
        acdc_df = self.posData.fromTrackerToAcdcDf(
            self.tracker, tracked_video, start_frame_i=self.mainWin.start_n-1
        )
        # Store new tracked video
        current_frame_i = self.posData.frame_i
        self.trackingOnNeverVisitedFrames = False
        print('')
        self.progress.emit(
            'Storing tracked video...')
        pbar = tqdm(total=len(tracked_video), ncols=100)
        for rel_frame_i, lab in enumerate(tracked_video):
            frame_i = rel_frame_i + self.mainWin.start_n - 1

            if acdc_df is not None:
                cca_cols = acdc_df.columns.intersection(
                    cca_df_colnames_with_tree
                )
                # Store cca_df if it is an output of the tracker
                cca_df = acdc_df.loc[frame_i][cca_cols]
                self.mainWin.store_cca_df(
                    frame_i=frame_i, cca_df=cca_df, mainThread=False,
                    autosave=False
                )

            if self.posData.allData_li[frame_i]['labels'] is None:
                # repeating tracking on a never visited frame
                # --> modify only raw data and ask later what to do
                self.posData.segm_data[frame_i] = lab
                self.trackingOnNeverVisitedFrames = True
            else:
                # Get the rest of the stored metadata based on the new lab
                self.posData.allData_li[frame_i]['labels'] = lab
                self.posData.frame_i = frame_i
                self.mainWin.get_data()
                self.mainWin.store_data(autosave=False)
            
            pbar.update()
        pbar.close()

        # Back to current frame
        self.posData.frame_i = current_frame_i
        self.mainWin.get_data()
        self.mainWin.store_data(autosave=True)
        self.mutex.unlock()
        self.finished.emit()

class reapplyDataPrepWorker(QObject):
    finished = Signal()
    debug = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    initPbar = Signal(int)
    updatePbar = Signal()
    sigCriticalNoChannels = Signal(str)
    sigSelectChannels = Signal(object, object, object, str)

    def __init__(self, expPath, posFoldernames):
        super().__init__()
        self.expPath = expPath
        self.posFoldernames = posFoldernames
        self.abort = False
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
    
    def raiseSegmInfoNotFound(self, path):
        raise FileNotFoundError(
            'The following file is required for the alignment of 4D data '
            f'but it was not found: "{path}"'
        )
    
    def saveBkgrData(self, imageData, posData, isAligned=False):
        bkgrROI_data = {}
        for r, roi in enumerate(posData.bkgrROIs):
            xl, yt = [int(round(c)) for c in roi.pos()]
            w, h = [int(round(c)) for c in roi.size()]
            if not yt+h>yt or not xl+w>xl:
                # Prevent 0 height or 0 width roi
                continue
            is4D = posData.SizeT > 1 and posData.SizeZ > 1
            is3Dz = posData.SizeT == 1 and posData.SizeZ > 1
            is3Dt = posData.SizeT > 1 and posData.SizeZ == 1
            is2D = posData.SizeT == 1 and posData.SizeZ == 1
            if is4D:
                bkgr_data = imageData[:, :, yt:yt+h, xl:xl+w]
            elif is3Dz or is3Dt:
                bkgr_data = imageData[:, yt:yt+h, xl:xl+w]
            elif is2D:
                bkgr_data = imageData[yt:yt+h, xl:xl+w]
            bkgrROI_data[f'roi{r}_data'] = bkgr_data

        if not bkgrROI_data:
            return

        if isAligned:
            bkgr_data_fn = f'{posData.filename}_aligned_bkgrRoiData.npz'
        else:
            bkgr_data_fn = f'{posData.filename}_bkgrRoiData.npz'
        bkgr_data_path = os.path.join(posData.images_path, bkgr_data_fn)
        self.progress.emit('Saving background data to:')
        self.progress.emit(bkgr_data_path)
        np.savez_compressed(bkgr_data_path, **bkgrROI_data)

    def run(self):
        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )
        for p, pos in enumerate(self.posFoldernames):
            if self.abort:
                break
            
            self.progress.emit(f'Processing {pos}...')
                
            posPath = os.path.join(self.expPath, pos)
            imagesPath = os.path.join(posPath, 'Images')

            ls = myutils.listdir(imagesPath)
            if p == 0:
                ch_names, basenameNotFound = (
                    ch_name_selector.get_available_channels(ls, imagesPath)
                )
                if not ch_names:
                    self.sigCriticalNoChannels.emit(imagesPath)
                    break
                self.mutex.lock()
                if len(self.posFoldernames) == 1:
                    # User selected only one pos --> allow selecting and adding
                    # and external .tif file that will be renamed with the basename
                    basename = ch_name_selector.basename
                else:
                    basename = None
                self.sigSelectChannels.emit(
                    ch_name_selector, ch_names, imagesPath, basename
                )
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                if self.abort:
                    break
            
                self.progress.emit(
                    f'Selected channels: {self.selectedChannels}'
                )
            
            for chName in self.selectedChannels:
                filePath = load.get_filename_from_channel(imagesPath, chName)
                posData = load.loadData(filePath, chName)
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadImgData()
                posData.loadOtherFiles(
                    load_segm_data=False, 
                    getTifPath=True,
                    load_metadata=True,
                    load_shifts=True,
                    load_dataPrep_ROIcoords=True,
                    loadBkgrROIs=True
                )

                imageData = posData.img_data

                prepped = False
                isAligned = False
                # Align
                if posData.loaded_shifts is not None:
                    self.progress.emit('Aligning frames...')
                    shifts = posData.loaded_shifts
                    if imageData.ndim == 4:
                        align_func = core.align_frames_3D
                    else:
                        align_func = core.align_frames_2D 
                    imageData, _ = align_func(imageData, user_shifts=shifts)
                    prepped = True
                    isAligned = True
                
                # Crop and save background
                if posData.dataPrep_ROIcoords is not None:
                    df = posData.dataPrep_ROIcoords
                    isCropped = int(df.at['cropped', 'value']) == 1
                    if isCropped:
                        self.saveBkgrData(imageData, posData, isAligned)
                        self.progress.emit('Cropping...')
                        x0 = int(df.at['x_left', 'value']) 
                        y0 = int(df.at['y_top', 'value']) 
                        x1 = int(df.at['x_right', 'value']) 
                        y1 = int(df.at['y_bottom', 'value']) 
                        if imageData.ndim == 4:
                            imageData = imageData[:, :, y0:y1, x0:x1]
                        elif imageData.ndim == 3:
                            imageData = imageData[:, y0:y1, x0:x1]
                        elif imageData.ndim == 2:
                            imageData = imageData[y0:y1, x0:x1]
                        prepped = True
                    else:
                        filename = os.path.basename(posData.dataPrepBkgrROis_path)
                        self.progress.emit(
                            f'WARNING: the file "{filename}" was not found. '
                            'I cannot crop the data.'
                        )
                    
                if prepped:              
                    self.progress.emit('Saving prepped data...')
                    np.savez_compressed(posData.align_npz_path, imageData)
                    if hasattr(posData, 'tif_path'):
                        myutils.to_tiff(
                            posData.tif_path, imageData
                        )

            self.updatePbar.emit()
            if self.abort:
                break
        self.finished.emit()

class LazyLoader(QObject):
    sigLoadingFinished = Signal()

    def __init__(self, mutex, waitCond, readH5mutex, waitReadH5cond):
        QObject.__init__(self)
        self.signals = signals()
        self.mutex = mutex
        self.waitCond = waitCond
        self.exit = False
        self.salute = True
        self.sender = None
        self.H5readWait = False
        self.waitReadH5cond = waitReadH5cond
        self.readH5mutex = readH5mutex

    def setArgs(self, posData, current_idx, axis, updateImgOnFinished):
        self.wait = False
        self.updateImgOnFinished = updateImgOnFinished
        self.posData = posData
        self.current_idx = current_idx
        self.axis = axis

    def pauseH5read(self):
        self.readH5mutex.lock()
        self.waitReadH5cond.wait(self.mutex)
        self.readH5mutex.unlock()

    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.signals.progress.emit(
                    'Closing lazy loader...', 'INFO'
                )
                break
            elif self.wait:
                self.signals.progress.emit(
                    'Lazy loader paused.', 'INFO'
                )
                self.pause()
            else:
                self.signals.progress.emit(
                    'Lazy loader resumed.', 'INFO'
                )
                self.posData.loadChannelDataChunk(
                    self.current_idx, axis=self.axis, worker=self
                )
                self.sigLoadingFinished.emit()
                self.wait = True

        self.signals.finished.emit(None)


class ImagesToPositionsWorker(QObject):
    finished = Signal()
    debug = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    initPbar = Signal(int)
    updatePbar = Signal()

    def __init__(self, folderPath, targetFolderPath, appendText):
        super().__init__()
        self.abort = False
        self.folderPath = folderPath
        self.targetFolderPath = targetFolderPath
        self.appendText = appendText
    
    @worker_exception_handler
    def run(self):
        self.progress.emit(f'Selected folder: "{self.folderPath}"')
        self.progress.emit(f'Target folder: "{self.targetFolderPath}"')
        self.progress.emit(' ')
        ls = myutils.listdir(self.folderPath)
        numFiles = len(ls)
        self.initPbar.emit(numFiles)
        numPosDigits = len(str(numFiles))
        if numPosDigits == 1:
            numPosDigits = 2
        pos = 1
        for file in ls:
            if self.abort:
                break
            
            filePath = os.path.join(self.folderPath, file)
            if os.path.isdir(filePath):
                # Skip directories
                self.updatePbar.emit()
                continue
            
            self.progress.emit(f'Loading file: {file}')
            filename, ext = os.path.splitext(file)
            s0p = str(pos).zfill(numPosDigits)
            try:
                data = load.imread(filePath)
                if data.ndim == 3 and (data.shape[-1] == 3 or data.shape[-1] == 4):
                    self.progress.emit('Converting RGB image to grayscale...')
                    data = skimage.color.rgb2gray(data)
                    data = skimage.img_as_ubyte(data)
                
                posName = f'Position_{pos}'
                posPath = os.path.join(self.targetFolderPath, posName)
                imagesPath = os.path.join(posPath, 'Images')
                if not os.path.exists(imagesPath):
                    os.makedirs(imagesPath, exist_ok=True)
                newFilename = f's{s0p}_{filename}_{self.appendText}.tif'
                relPath = os.path.join(posName, 'Images', newFilename)
                tifFilePath = os.path.join(imagesPath, newFilename)
                self.progress.emit(f'Saving to file: ...{os.sep}{relPath}')
                myutils.to_tiff(
                    tifFilePath, data
                )
                pos += 1
            except Exception as e:
                self.progress.emit(
                    f'WARNING: {file} is not a valid image file. Skipping it.'
                )
            
            self.progress.emit(' ')
            self.updatePbar.emit()

            if self.abort:
                break
        self.finished.emit()

class BaseWorkerUtil(QObject):
    progressBar = Signal(int, int, float)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = signals()
        self.abort = False
        self.skipExp = False
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.signals.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def emitSelectFilesWithText(
            self, exp_path, pos_foldernames, with_text, ext=None
        ):
        self.mutex.lock()
        self.signals.sigSelectFilesWithText.emit(
            exp_path, pos_foldernames, with_text, ext
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def emitSelectFile(self, start_dir, caption='', filters='All files (*.)'):
        self.mutex.lock()
        self.signals.sigSelectFile.emit(start_dir, caption, filters)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
        
    def emitSelectAcdcOutputFiles(
            self, exp_path, pos_foldernames, infoText='', 
            allowSingleSelection=False, multiSelection=True
        ):
        self.mutex.lock()
        self.signals.sigSelectAcdcOutputFiles.emit(
            exp_path, pos_foldernames, infoText, allowSingleSelection,
            multiSelection
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitSelectSpotmaxRun(
            self, exp_path, pos_foldernames, all_runs, infoText='', 
            allowSingleSelection=True, multiSelection=True
        ):
        self.mutex.lock()
        self.signals.sigSelectSpotmaxRun.emit(
            exp_path, pos_foldernames, all_runs, infoText, allowSingleSelection,
            multiSelection
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

class DataPrepSaveBkgrDataWorker(QObject):
    def __init__(self, posData, dataPrepWin):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.posData = posData
        self.dataPrepWin = dataPrepWin
    
    @worker_exception_handler
    def run(self):
        self.dataPrepWin.saveBkgrData(self.posData)
        self.signals.finished.emit(self)

class DataPrepCropWorker(QObject):
    def __init__(self, posData, dataPrepWin, dstPath):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.posData = posData
        self.dataPrepWin = dataPrepWin
        self.dstPath = dstPath
    
    @worker_exception_handler
    def run(self):
        self.dataPrepWin.saveSingleCrop(
            self.posData, self.posData.cropROIs[0], self.dstPath
        )
        self.signals.finished.emit(self)

class TrackSubCellObjectsWorker(BaseWorkerUtil):
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

    @worker_exception_handler
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
                ls = myutils.listdir(images_path)
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
                np.savez_compressed(trackedSubPath, trackedSubSegmData)
                posData.saveIsSegm3Dmetadata(trackedSubPath)

                if trackedCellsSegmData is not None:
                    cellsSegmFilename, ext = os.path.splitext(segmCellsPath)
                    trackedCellsPath = f'{cellsSegmFilename}_{appendedName}.npz'
                    np.savez_compressed(trackedCellsPath, trackedCellsSegmData)
                
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
                    np.savez_compressed(diffSegmPath, diffSegmData)
                    posData.saveIsSegm3Dmetadata(diffSegmPath)
                    del diffSegmData

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

class PostProcessSegmWorker(QObject):
    def __init__(
            self, 
            postProcessKwargs, 
            customPostProcessGroupedFeatures, 
            customPostProcessFeatures,
            mainWin
        ):
        super().__init__()
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.kwargs = postProcessKwargs
        self.customPostProcessGroupedFeatures = customPostProcessGroupedFeatures
        self.customPostProcessFeatures = customPostProcessFeatures
        self.mainWin = mainWin
    
    @worker_exception_handler
    def run(self):
        mainWin = self.mainWin
        data = mainWin.data
        posData = data[mainWin.pos_i]
        if len(data) > 1:
            self.signals.initProgressBar.emit(len(data))
        else:
            current_frame_i = posData.frame_i
            self.signals.initProgressBar.emit(posData.SizeT - current_frame_i)
        
        self.logger.log('Post-process segmentation process started.')
        self._run()
        self.signals.finished.emit(None)
    
    def _run(self):
        kwargs = self.kwargs
        mainWin = self.mainWin
        data = mainWin.data

        for posData in data:
            current_frame_i = posData.frame_i
            data_li = posData.allData_li[current_frame_i:]
            for i, data_dict in enumerate(data_li):
                frame_i = current_frame_i + i
                visited = True
                lab = data_dict['labels']
                if lab is None:
                    visited = False
                    try:
                        lab = posData.segm_data[frame_i]
                    except Exception as e:
                        return

                image = posData.img_data[frame_i]
                
                processed_lab = core.post_process_segm(
                    lab, return_delIDs=False, **kwargs
                )
                if self.customPostProcessFeatures:
                    processed_lab = features.custom_post_process_segm(
                        posData, 
                        self.customPostProcessGroupedFeatures, 
                        processed_lab, 
                        image, 
                        posData.frame_i, 
                        posData.filename, 
                        posData.user_ch_name, 
                        self.customPostProcessFeatures
                    )
                if visited:
                    posData.allData_li[frame_i]['labels'] = processed_lab
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = frame_i
                    mainWin.get_data()
                    mainWin.store_data(autosave=False)
                else:
                    posData.segm_data[frame_i] = lab

                self.signals.progressBar.emit(1)
            
            posData.frame_i = current_frame_i

class CreateConnected3Dsegm(BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def criticalSegmIsNot3D(self):
        raise TypeError(
            'Input segmentation masks are not 3D. You can use this utility '
            'only on 3D z-stack data or 4D z-stack over time data.'
        )
    
    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f'Select <b>3D segmentation file to connect</b>'
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
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ][0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(
                    f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )
                if posData.segm_data.ndim == 3:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                self.logger.log('Connecting 3D objects...')
                
                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                connectedSegmData = np.zeros_like(posData.segm_data)
                for frame_i, lab in enumerate(posData.segm_data):
                    if lab.ndim != 3:
                        self.criticalSegmIsNot3D()
                        
                    connected_lab = core.connect_3Dlab_zboundaries(lab)
                    connectedSegmData[frame_i] = connected_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log('Saving connected 3D segmentation file...')
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f'{segmFilename}_{appendedName}.npz'
                connectedSegmData = np.squeeze(connectedSegmData)
                np.savez_compressed(newSegmFilepath, connectedSegmData)
                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)

class ApplyTrackInfoWorker(BaseWorkerUtil):
    def __init__(
            self, parentWin, endFilenameSegm, trackInfoCsvPath, 
            trackedSegmFilename, trackColsInfo, posPath
        ):
        super().__init__(parentWin)
        self.endFilenameSegm = endFilenameSegm
        self.trackInfoCsvPath = trackInfoCsvPath
        self.trackedSegmFilename = trackedSegmFilename
        self.trackColsInfo = trackColsInfo
        self.posPath = posPath
    
    @worker_exception_handler
    def run(self):
        self.logger.log('Loading segmentation file...')  
        self.signals.initProgressBar.emit(0)
        imagesPath = os.path.join(self.posPath, 'Images')
        segmFilename = [
            f for f in myutils.listdir(imagesPath) 
            if f.endswith(f'{self.endFilenameSegm}.npz')
        ][0]
        segmFilePath = os.path.join(imagesPath, segmFilename)
        segmData = np.load(segmFilePath)['arr_0']

        self.logger.log('Loading table containing tracking info...') 
        df = pd.read_csv(self.trackInfoCsvPath)

        frameIndexCol = self.trackColsInfo['frameIndexCol']

        parentIDcol = self.trackColsInfo['parentIDcol']
        pbarMax = len(df[frameIndexCol].unique())
        self.signals.initProgressBar.emit(pbarMax)

        # Apply tracking info
        result = core.apply_tracking_from_table(
            segmData, self.trackColsInfo, df, signal=self.signals.progressBar,
            logger=self.logger.log, pbarMax=pbarMax
        )
        trackedData, trackedIDsMapper, deleteIDsMapper = result

        if self.trackedSegmFilename:
            trackedSegmFilepath = os.path.join(
                imagesPath, self.trackedSegmFilename
            )
        else:
            trackedSegmFilepath = os.path.join(segmFilePath)
        
        self.signals.initProgressBar.emit(0)
        self.logger.log('Saving tracked segmentation file...') 
        np.savez_compressed(trackedSegmFilepath, trackedData)

        
        mapperPath = os.path.splitext(trackedSegmFilepath)[0]
        mapperJsonPath = f'{mapperPath}_deletedIDs_mapper.json'
        mapperJsonName = os.path.basename(mapperJsonPath)
        self.logger.log(f'Saving deleted IDs to {mapperJsonName}...')
        with open(mapperJsonPath, 'w') as file:
            file.write(json.dumps(deleteIDsMapper))

        mapperPath = os.path.splitext(trackedSegmFilepath)[0]
        mapperJsonPath = f'{mapperPath}_replacedIDs_mapper.json'
        mapperJsonName = os.path.basename(mapperJsonPath)
        self.logger.log(f'Saving IDs replacements to {mapperJsonName}...')
        with open(mapperJsonPath, 'w') as file:
            file.write(json.dumps(trackedIDsMapper))

        self.logger.log('Generating acdc_output table...')
        acdc_df = None
        if not self.trackedSegmFilename:
            # Fix existing acdc_df
            acdcEndname = self.endFilenameSegm.replace('_segm', '_acdc_output')
            acdcFilename = [
                f for f in myutils.listdir(imagesPath) 
                if f.endswith(f'{acdcEndname}.csv')
            ]
            if acdcFilename:
                acdcFilePath = os.path.join(imagesPath, acdcFilename[0])
                acdc_df = pd.read_csv(
                    acdcFilePath, index_col=['frame_i', 'Cell_ID']
                )

        if acdc_df is not None:
            acdc_df = core.apply_trackedIDs_mapper_to_acdc_df(
                trackedIDsMapper, deleteIDsMapper, acdc_df
            )
        else:
            acdc_dfs = []
            keys = []
            for frame_i, lab in enumerate(trackedData):            
                rp = skimage.measure.regionprops(lab)
                acdc_df_frame_i = myutils.getBaseAcdcDf(rp)
                acdc_dfs.append(acdc_df_frame_i)
                keys.append(frame_i)
            
            acdc_df = pd.concat(acdc_dfs, keys=keys, names=['frame_i', 'Cell_ID'])
            segmFilename = os.path.basename(trackedSegmFilepath)
            acdcFilename = re.sub(segm_re_pattern, '_acdc_output', segmFilename)
            acdcFilePath = os.path.join(imagesPath, acdcFilename)
        
        self.signals.initProgressBar.emit(pbarMax)
        parentIDcol = self.trackColsInfo['parentIDcol']
        trackIDsCol = self.trackColsInfo['trackIDsCol']
        if parentIDcol != 'None':
            self.logger.log(f'Adding lineage info from "{parentIDcol}" column...')
            acdc_df = core.add_cca_info_from_parentID_col(
                df, acdc_df, frameIndexCol, trackIDsCol, parentIDcol, 
                len(segmData), signal=self.signals.progressBar,
                maskID_colname=self.trackColsInfo['maskIDsCol'], 
                x_colname=self.trackColsInfo['xCentroidCol'], 
                y_colname=self.trackColsInfo['yCentroidCol']
            )     
        
        self.logger.log('Saving acdc_output table...')
        acdc_df.to_csv(acdcFilePath)

        self.signals.finished.emit(self)

class RestructMultiPosWorker(BaseWorkerUtil):
    sigSaveTiff = Signal(str, object, object)

    def __init__(self, rootFolderPath, dstFolderPath, action='copy'):
        super().__init__(None)
        self.rootFolderPath = rootFolderPath
        self.dstFolderPath = dstFolderPath
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.action = action

    @worker_exception_handler
    def run(self):
        load._restructure_multi_files_multi_pos(
            self.rootFolderPath, self.dstFolderPath, signals=self.signals, 
            logger=self.logger.log, action=self.action
        )
        self.signals.finished.emit(self)


class RestructMultiTimepointsWorker(BaseWorkerUtil):
    sigSaveTiff = Signal(str, object, object)

    def __init__(
            self, allChannels, frame_name_pattern, basename, validFilenames,
            rootFolderPath, dstFolderPath, segmFolderPath=''
        ):
        super().__init__(None)
        self.allChannels = allChannels
        self.frame_name_pattern = frame_name_pattern
        self.basename = basename
        self.validFilenames = validFilenames
        self.rootFolderPath = rootFolderPath
        self.dstFolderPath = dstFolderPath
        self.segmFolderPath = segmFolderPath
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()

    @worker_exception_handler
    def run(self):
        allChannels = self.allChannels
        frame_name_pattern = self.frame_name_pattern
        rootFolderPath = self.rootFolderPath
        dstFolderPath = self.dstFolderPath
        segmFolderPath = self.segmFolderPath
        filesInfo = {}
        self.signals.initProgressBar.emit(len(self.validFilenames)+1)
        for file in self.validFilenames:
            try:
                # Determine which channel is this file
                for ch in allChannels:
                    m = re.findall(rf'(.*)_{ch}{frame_name_pattern}', file)
                    if m:
                        break
                else:
                    raise FileNotFoundError(
                        f'The file name "{file}" does not contain any channel name'
                    )
                posName, _, frameName = m[0]
                frameNumber = int(frameName)
                if posName not in filesInfo:
                    filesInfo[posName] = {ch: [(file, frameNumber)]}
                elif ch not in filesInfo[posName]:
                    filesInfo[posName][ch] = [(file, frameNumber)]
                else:
                    filesInfo[posName][ch].append((file, frameNumber))
            except Exception as e:
                self.logger.log(traceback.format_exc())
                self.logger.log(
                    f'WARNING: File "{file}" does not contain valid pattern. '
                    'Skipping it.'
                )
                continue
        
        self.signals.progressBar.emit(1)

        df_metadata = None
        partial_basename = self.basename
        allPosDataInfo = []
        for p, (posName, channelInfo) in enumerate(filesInfo.items()):
            self.logger.log(f'='*40)
            self.logger.log(f'Processing position "{posName}"...')

            for _, filesList in channelInfo.items():
                # Get info from first file
                filePath = os.path.join(rootFolderPath, filesList[0][0])
                try:
                    img = load.imread(filePath)
                    break
                except Exception as e:
                    self.logger.log(traceback.format_exc())
                    continue
            else:
                self.logger.log(
                    f'WARNING: No valid image files found for position {posName}'
                )
                continue

            # Get basename
            if partial_basename:
                basename = f'{partial_basename}_{posName}_'
            else:
                basename = f'{posName}_'

            # Get SizeT from first file
            SizeT = len(filesList)
            
            # Save metadata.csv      
            df_metadata = pd.DataFrame({
                'SizeT': SizeT,
                'basename': basename
            }, index=['values'])

            # Iterate channels
            for c, (channelName, filesList) in enumerate(channelInfo.items()):
                self.logger.log(
                    f'  Processing channel "{channelName}"...'
                )
                # Sort by frame number
                sortedFilesList = sorted(filesList, key=lambda t:t[1])

                df_metadata[f'channel_{c}_name'] = [channelName]

                imagesPath = os.path.join(dstFolderPath, f'Position_{p+1}', 'Images')
                if not os.path.exists(imagesPath):
                    os.makedirs(imagesPath, exist_ok=True)

                # Iterate frames
                videoData = None
                srcSegmPaths = ['']*SizeT
                frameNumbers = []
                for frame_i, fileInfo in enumerate(sortedFilesList):
                    file, _ = fileInfo
                    ext = os.path.splitext(file)[1]
                    srcImgFilePath = os.path.join(rootFolderPath, file)
                    try:
                        img = load.imread(srcImgFilePath)
                        if videoData is None:
                            shape = (SizeT, *img.shape)
                            videoData = np.zeros(shape, dtype=img.dtype)
                        videoData[frame_i] = img
                        pattern = self.frame_name_pattern
                        frameNumberMatch = re.findall(pattern, file)[0][1]
                        frameNumber = int(frameNumberMatch)
                        frameNumbers.append(frameNumber)
                    except Exception as e:
                        self.logger.log(traceback.format_exc())
                        continue

                    if segmFolderPath and c==0:
                        srcSegmFilePath = os.path.join(segmFolderPath, file)
                        srcSegmPaths[frame_i] = srcSegmFilePath

                    SizeZ = 1
                    if img.ndim == 3:
                        SizeZ = len(img)
                    
                    df_metadata['SizeZ'] = [SizeZ]                 

                    self.signals.progressBar.emit(1)
                
                if videoData is None:
                    self.logger.log(
                        f'WARNING: No valid image files found for position '
                        f'"{posName}", channel "{channelName}"'
                    )
                    continue
                else:
                    imgFileName = f'{basename}{channelName}.tif'
                    dstImgFilePath = os.path.join(imagesPath, imgFileName)
                    dstSegmFileName = f'{basename}segm_{channelName}.npz'
                    dstSegmPath = os.path.join(imagesPath, dstSegmFileName)
                    imgDataInfo = {
                        'path': dstImgFilePath, 'SizeT': SizeT, 'SizeZ': SizeZ,
                        'data': videoData, 'frameNumbers': frameNumbers,
                        'dst_segm_path': dstSegmPath, 
                        'src_segm_paths': srcSegmPaths
                    }
                    allPosDataInfo.append(imgDataInfo)

            if df_metadata is not None:
                metadata_csv_path = os.path.join(
                    imagesPath, f'{basename}metadata.csv'
                )
                df_metadata = df_metadata.T
                df_metadata.index.name = 'Description'
                df_metadata.to_csv(metadata_csv_path)

            self.logger.log(f'*'*40)
        
        if not allPosDataInfo:
            self.signals.finished.emit(self)
            return
        
        self.signals.initProgressBar.emit(len(allPosDataInfo))
        self.logger.log('Saving image files...')
        maxSizeT = max([d['SizeT'] for d in allPosDataInfo])
        minFrameNumber = min([d['frameNumbers'][0] for d in allPosDataInfo])
        # Pad missing frames in video files according to frame number
        for p, imgDataInfo in enumerate(allPosDataInfo):
            SizeT = imgDataInfo['SizeT']
            SizeZ = imgDataInfo['SizeZ']
            dstImgFilePath = imgDataInfo['path']
            videoData = imgDataInfo['data']
            frameNumbers = imgDataInfo['frameNumbers']
            paddedShape = (maxSizeT, *videoData.shape[1:])
            imgDataInfo['paddedShape'] = paddedShape
            dtype = videoData.dtype
            paddedVideoData = np.zeros(paddedShape, dtype=dtype)
            for n, img in zip(frameNumbers, videoData):
                frame_i = n - minFrameNumber
                paddedVideoData[frame_i] = img

            del videoData
            imgDataInfo['data'] = None

            self.mutex.lock()        
            self.sigSaveTiff.emit(dstImgFilePath, paddedVideoData, self.waitCond)
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()        

            self.signals.progressBar.emit(1)      

        if not segmFolderPath:
            self.signals.finished.emit(self)
            return

        self.signals.initProgressBar.emit(len(allPosDataInfo))
        self.logger.log('Saving segmentation files...')
        for p, imgDataInfo in enumerate(allPosDataInfo):
            SizeT = imgDataInfo['SizeT']
            frameNumbers = imgDataInfo['frameNumbers']
            SizeT = imgDataInfo['SizeT']
            SizeZ = imgDataInfo['SizeZ']
            frameNumbers = imgDataInfo['frameNumbers']
            paddedShape = imgDataInfo['paddedShape']
            segmData = np.zeros(paddedShape, dtype=np.uint32)
            for n, segmFilePath in zip(frameNumbers, imgDataInfo['src_segm_paths']):
                frame_i = n - minFrameNumber
                try:
                    lab = load.imread(segmFilePath).astype(np.uint32)
                    segmData[frame_i] = lab
                except Exception as e:
                    self.logger.log(traceback.format_exc())
                    self.logger.log(
                        'WARNING: The following segmentation file does not '
                        f'exist, saving empty masks: "{srcSegmFilePath}"'
                    ) 

            np.savez_compressed(imgDataInfo['dst_segm_path'], segmData)
            del segmData 

        self.signals.finished.emit(self)

class ComputeMetricsMultiChannelWorker(BaseWorkerUtil):
    sigAskAppendName = Signal(str, list, list)
    sigCriticalNotEnoughSegmFiles = Signal(str)
    sigAborted = Signal()
    sigHowCombineMetrics = Signal(str, list, list, list)

    def __init__(self, mainWin):
        super().__init__(mainWin)
    
    def emitHowCombineMetrics(
            self, imagesPath, selectedAcdcOutputEndnames, 
            existingAcdcOutputEndnames, allChNames
        ):
        self.mutex.lock()
        self.sigHowCombineMetrics.emit(
            imagesPath, selectedAcdcOutputEndnames, 
            existingAcdcOutputEndnames, allChNames
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def loadAcdcDfs(self, imagesPath, selectedAcdcOutputEndnames):
        for end in selectedAcdcOutputEndnames:
            filePath, _ = load.get_path_from_endname(end, imagesPath)
            acdc_df = pd.read_csv(filePath)
            yield acdc_df
    
    def run_iter_exp(self, exp_path, pos_foldernames, i, tot_exp):
        tot_pos = len(pos_foldernames)
        
        abort = self.emitSelectAcdcOutputFiles(
            exp_path, pos_foldernames, infoText=' to combine',
            allowSingleSelection=False
        )
        if abort:
            self.sigAborted.emit()
            return
        
        # Ask appendend name
        self.mutex.lock()
        self.sigAskAppendName.emit(
            f'{self.mainWin.basename_pos1}acdc_output', 
            self.mainWin.existingAcdcOutputEndnames,
            self.mainWin.selectedAcdcOutputEndnames
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return

        selectedAcdcOutputEndnames = self.mainWin.selectedAcdcOutputEndnames
        existingAcdcOutputEndnames = self.mainWin.existingAcdcOutputEndnames
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

            imagesPath = os.path.join(exp_path, pos, 'Images')
            basename, chNames = myutils.getBasenameAndChNames(
                imagesPath, useExt=('.tif', '.h5')
            )

            if p == 0:
                abort = self.emitHowCombineMetrics(
                    imagesPath, selectedAcdcOutputEndnames, 
                    existingAcdcOutputEndnames, chNames
                )
                if abort:
                    self.sigAborted.emit()
                    return
                acdcDfs = self.acdcDfs.values()
                # Update selected acdc_dfs since the user could have 
                # loaded additional ones inside the emitHowCombineMetrics
                # dialog
                selectedAcdcOutputEndnames = self.acdcDfs.keys()
            else:
                acdcDfs = self.loadAcdcDfs(
                    imagesPath, selectedAcdcOutputEndnames
                )

            dfs = []
            for i, acdc_df in enumerate(acdcDfs):
                dfs.append(acdc_df.add_suffix(f'_table{i+1}'))
            combined_df = pd.concat(dfs, axis=1)

            newAcdcDf = pd.DataFrame(index=combined_df.index)
            for newColname, equation in self.equations.items():
                newAcdcDf[newColname] = combined_df.eval(equation)
            
            newAcdcDfPath = os.path.join(
                imagesPath, f'{basename}acdc_output_{appendedName}.csv'
            )
            newAcdcDf.to_csv(newAcdcDfPath)

            equationsIniPath = os.path.join(
                imagesPath, f'{basename}equations_{appendedName}.ini'
            )
            equationsConfig = config.ConfigParser()
            if os.path.exists(equationsIniPath):
                equationsConfig.read(equationsIniPath)
            equationsConfig = self.addEquationsToConfigPars(
                equationsConfig, selectedAcdcOutputEndnames, self.equations
            )
            with open(equationsIniPath, 'w') as configfile:
                equationsConfig.write(configfile)

            self.signals.progressBar.emit(1)
        
        return True
    
    def addEquationsToConfigPars(self, cp, selectedAcdcOutputEndnames, equations):
        section = [
            f'df{i+1}:{end}' for i, end in enumerate(selectedAcdcOutputEndnames)
        ]
        section = ';'.join(section)
        if section not in cp:
            cp[section] = {}
        
        for metricName, expression in equations.items():
            cp[section][metricName] = expression
        
        return cp
         
    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        self.errors = {}
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            try:
                result = self.run_iter_exp(exp_path, pos_foldernames, i, tot_exp)
                if result is None:
                    return
            except Exception as e:
                traceback_str = traceback.format_exc()
                self.errors[e] = traceback_str
                self.logger.log(traceback_str)

        self.signals.finished.emit(self)

class ConcatAcdcDfsWorker(BaseWorkerUtil):
    sigAborted = Signal()
    sigAskFolder = Signal(str)
    sigSetMeasurements = Signal(object)
    sigAskAppendName = Signal(str, list)

    def __init__(self, mainWin, format='CSV'):
        super().__init__(mainWin)
        if format.startswith('CSV'):
            self._to_format = 'to_csv'
        elif format.startswith('XLS'):
            self._to_format = 'to_excel'
    
    def emitSetMeasurements(self, kwargs):
        self.mutex.lock()
        self.sigSetMeasurements.emit(kwargs)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def emitAskAppendName(self, allPos_acdc_df_basename):
        # Ask appendend name
        self.mutex.lock()
        self.sigAskAppendName.emit(allPos_acdc_df_basename, [])
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        acdc_dfs_allexp = []
        keys_exp = []
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)
            
            if i == 0:
                abort = self.emitSelectAcdcOutputFiles(
                    exp_path, pos_foldernames, infoText=' to combine',
                    allowSingleSelection=True, multiSelection=False
                )
                if abort:
                    self.sigAborted.emit()
                    return

            selectedAcdcOutputEndname = self.mainWin.selectedAcdcOutputEndnames[0]

            self.signals.initProgressBar.emit(len(pos_foldernames))
            acdc_dfs = []
            keys = []
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')

                ls = myutils.listdir(images_path)

                acdc_output_file = [
                    f for f in ls 
                    if f.endswith(f'{selectedAcdcOutputEndname}.csv')
                ]
                if not acdc_output_file:
                    self.logger.log(
                        f'{pos} does not contain any '
                        f'{selectedAcdcOutputEndname}.csv file. '
                        'Skipping it.'
                    )
                    self.signals.progressBar.emit(1)
                    continue
                
                acdc_df_filepath = os.path.join(images_path, acdc_output_file[0])
                acdc_df = pd.read_csv(acdc_df_filepath).set_index('Cell_ID')
                acdc_dfs.append(acdc_df)
                keys.append(pos)

                self.signals.progressBar.emit(1)

            self.signals.initProgressBar.emit(0)
            acdc_df_allpos = pd.concat(
                acdc_dfs, keys=keys, names=['Position_n', 'Cell_ID']
            )
            acdc_df_allpos['experiment_folderpath'] = exp_path
            
            basename, chNames = myutils.getBasenameAndChNames(
                images_path, useExt=('.tif', '.h5')
            )
            df_metadata = load.load_metadata_df(images_path)
            SizeZ = df_metadata.at['SizeZ', 'values']
            SizeZ = int(float(SizeZ))
            existing_colnames = acdc_df_allpos.columns
            isSegm3D = any([col.endswith('3D') for col in existing_colnames])
            
            if i == 0:
                kwargs = {
                    'loadedChNames': chNames, 
                    'notLoadedChNames': [],
                    'isZstack': SizeZ > 1,
                    'isSegm3D': isSegm3D,
                    'existing_colnames': existing_colnames
                }
                self.emitSetMeasurements(kwargs)
                if self.abort:
                    self.sigAborted.emit()
                    return
            
            selected_cols = [
                col for col in self.selectedColumns 
                if col in acdc_df_allpos.columns
            ]
            acdc_df_allpos = acdc_df_allpos[selected_cols]
            acdc_dfs_allexp.append(acdc_df_allpos)
            exp_name = os.path.basename(exp_path)
            keys_exp.append((exp_path, exp_name))

            allpos_dir = os.path.join(exp_path, 'AllPos_acdc_output')
            if not os.path.exists(allpos_dir):
                os.mkdir(allpos_dir)
            
            allPos_acdc_df_basename = f'AllPos_{selectedAcdcOutputEndname}'
            if i == 0:
                self.emitAskAppendName(allPos_acdc_df_basename)
                if self.abort:
                    self.sigAborted.emit()
                    return
            
            acdc_dfs_allpos_filepath = os.path.join(
                allpos_dir, self.concat_df_filename
            )

            self.logger.log(
                'Saving all positions concatenated file to '
                f'"{acdc_dfs_allpos_filepath}"'
            )
            to_format_func = getattr(acdc_df_allpos, self._to_format)
            to_format_func(acdc_dfs_allpos_filepath)
            self.acdc_dfs_allpos_filepath = acdc_dfs_allpos_filepath

        if len(keys_exp) > 1:
            allExp_filename = f'multiExp_{self.concat_df_filename}'
            self.mutex.lock()
            self.sigAskFolder.emit(allExp_filename)
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.abort:
                self.sigAborted.emit()
                return
            
            acdc_df_allexp = pd.concat(
                acdc_dfs_allexp, keys=keys_exp, 
                names=['experiment_folderpath', 'experiment_foldername']
            )
            acdc_dfs_allexp_filepath = os.path.join(
                self.allExpSaveFolder, allExp_filename
            )
            self.logger.log(
                'Saving multiple experiments concatenated file to '
                f'"{acdc_dfs_allexp_filepath}"'
            )
            to_format_func = getattr(acdc_df_allexp, self._to_format)
            to_format_func(acdc_dfs_allexp_filepath)

        self.signals.finished.emit(self)

class FromImajeJroiToSegmNpzWorker(BaseWorkerUtil):
    sigSelectRoisProps = Signal(str, object, bool)
    
    def __init__(self, mainWin):
        super().__init__(mainWin)
    
    def emitSelectRoisProps(self, roi_filepath, TZYX_shape, is_multi_pos):
        self.mutex.lock()
        self.sigSelectRoisProps.emit(roi_filepath, TZYX_shape, is_multi_pos)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    @worker_exception_handler
    def run(self):
        import roifile
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            abort = self.emitSelectFilesWithText(
                exp_path, pos_foldernames, 'imagej_rois', ext='.zip'
            )
            if abort:
                self.signals.finished.emit(self)
                return
            
            self.askRoiPreferences = True
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameRoi = self.mainWin.endFilenameWithText
                ls = myutils.listdir(images_path)
                rois_filepaths = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameRoi}.zip')
                ]
                
                if not rois_filepaths:
                    self.logger.log(
                        '[WARNING]: The following Position folder does not '
                        f'contain any file ending with {endFilenameRoi}. '
                        f'Skipping it. "{os.path.join(exp_path, pos)}")'
                    )
                    continue
                    
                rois_filepath = rois_filepaths[0]
                
                if self.askRoiPreferences:
                    is_multi_pos = len(pos_foldernames) > 1
                    self.logger.log('Loading image data to get image shape...')
                    TZYX_shape = load.get_tzyx_shape(images_path)
                    abort = self.emitSelectRoisProps(
                        rois_filepath, TZYX_shape, is_multi_pos
                    )
                    if abort:
                        self.signals.finished.emit(self)
                        return
                    
                    self.askRoiPreferences = not self.useSamePropsForNextPos
                elif self.areAllRoisSelected:
                    rois = roifile.roiread(rois_filepath)
                    self.IDsToRoisMapper = {i+i: roi for roi in enumerate(rois)}
                else:
                    # Use same ID of previous position
                    rois = roifile.roiread(rois_filepath)
                    IDsToRoisMapper = {i+i: roi for i, roi in enumerate(rois)}
                    self.IDsToRoisMapper = {
                        ID: IDsToRoisMapper[ID] 
                        for ID in self.IDsToRoisMapper.keys()
                    }
                
                self.logger.log('Generating segm mask from ROIs...')
                segm_data = myutils.from_imagej_rois_to_segm_data(
                    TZYX_shape, self.IDsToRoisMapper, self.rescaleRoisSizes, 
                    self.repeatRoisZslicesRange
                )
                
                
                segm_filepath = (rois_filepath
                    .replace('imagej_rois', 'segm')
                    .replace('.zip', '.npz')
                )
                self.logger.log(f'Saving segm mask to "{segm_filepath}"...')
                np.savez_compressed(segm_filepath, segm_data)
        
        self.signals.finished.emit(self)
                
        
class ToImajeJroiWorker(BaseWorkerUtil):
    def __init__(self, mainWin):
        super().__init__(mainWin)
        
    @worker_exception_handler
    def run(self):
        from roifile import ImagejRoi, roiwrite

        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.signals.finished.emit(self)
                return
            
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
                
                files_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ]
                
                if not files_path:
                    self.logger.log(
                        '[WARNING]: The following Position folder does not '
                        f'contain any file ending with {endFilenameSegm}. '
                        f'Skipping it. "{os.path.join(exp_path, pos)}")'
                    )
                    continue
                
                file_path = files_path[0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )
    
                if posData.SizeT > 1:
                    rois = []
                    max_ID = posData.segm_data.max()
                    for t, lab in enumerate(posData.segm_data):
                        rois_t = myutils.from_lab_to_imagej_rois(
                            lab, ImagejRoi, t=t, SizeT=posData.SizeT, 
                            max_ID=max_ID
                        )
                        rois.extend(rois_t)
                else:
                    rois = myutils.from_lab_to_imagej_rois(
                        posData.segm_data, ImagejRoi
                    )

                roi_filepath = posData.segm_npz_path.replace('.npz', '.zip')
                roi_filepath = roi_filepath.replace('_segm', '_imagej_rois')

                try:
                    os.remove(roi_filepath)
                except Exception as e:
                    pass

                roiwrite(roi_filepath, rois)
                        
        self.signals.finished.emit(self)


class ToSymDivWorker(QObject):
    progressBar = Signal(int, int, float)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = signals()
        self.abort = False
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.signals.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            self.missingAnnotErrors = {}
            tot_pos = len(pos_foldernames)
            self.allPosDataInputs = []
            posDatas = []
            self.logger.log('-'*30)
            expFoldername = os.path.basename(exp_path)

            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.signals.finished.emit(self)
                return

            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                basename, chNames = myutils.getBasenameAndChNames(
                    images_path, useExt=('.tif', '.h5')
                )

                self.signals.sigUpdatePbarDesc.emit(f'Loading {pos_path}...')

                # Use first found channel, it doesn't matter for metrics
                for chName in chNames:
                    file_path = myutils.getChannelFilePath(images_path, chName)
                    if file_path:
                        break
                else:
                    raise FileNotFoundError(
                        f'None of the channels "{chNames}" were found in the path '
                        f'"{images_path}".'
                    )

                # Load data
                posData = load.loadData(file_path, chName)
                posData.getBasenameAndChNames(useExt=('.tif', '.h5'))

                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_acdc_df=True,
                    load_metadata=True,
                    loadSegmInfo=True
                )

                posDatas.append(posData)

                self.allPosDataInputs.append({
                    'file_path': file_path,
                    'chName': chName
                })
            
            # Iterate pos and calculate metrics
            numPos = len(self.allPosDataInputs)
            for p, posDataInputs in enumerate(self.allPosDataInputs):
                file_path = posDataInputs['file_path']
                chName = posDataInputs['chName']

                posData = load.loadData(file_path, chName)

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames(useExt=('.tif', '.h5'))
                posData.buildPaths()
                posData.loadImgData()

                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_acdc_df=True,
                    end_filename_segm=self.mainWin.endFilenameSegm
                )
                if not posData.acdc_df_found:
                    relPath = (
                        f'...{os.sep}{expFoldername}'
                        f'{os.sep}{posData.pos_foldername}'
                    )
                    self.logger.log(
                        f'WARNING: Skipping "{relPath}" '
                        f'because acdc_output.csv file was not found.'
                    )
                    self.missingAnnotErrors[relPath] = (
                        f'<br>FileNotFoundError: the Positon "{relPath}" '
                        'does not have the <code>acdc_output.csv</code> file.<br>')
                    
                    continue
                
                acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
                self.logger.log(
                    'Loaded path:\n'
                    f'ACDC output file name: "{acdc_df_filename}"'
                )

                self.logger.log('Building tree...')
                try:
                    tree = core.LineageTree(posData.acdc_df)
                    error = tree.build()
                    if isinstance(error, KeyError):
                        self.logger.log(str(error))
                        
                        self.logger.log(
                            'WARNING: Annotations missing in '
                            f'"{posData.acdc_output_csv_path}"'
                        )
                        self.missingAnnotErrors[acdc_df_filename] = str(error)
                        continue
                    elif error is not None:
                        raise error
                    posData.acdc_df = tree.df
                except Exception as error:
                    traceback_format = traceback.format_exc()
                    self.logger.log(traceback_format)
                    self.errors[error] = traceback_format
                
                try:
                    posData.acdc_df.to_csv(posData.acdc_output_csv_path)
                except PermissionError:
                    traceback_str = traceback.format_exc()
                    self.mutex.lock()
                    self.signals.sigPermissionError.emit(
                        traceback_str, posData.acdc_output_csv_path
                    )
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    posData.acdc_df.to_csv(posData.acdc_output_csv_path)
                
                self.signals.progressBar.emit(1)
                
        self.signals.finished.emit(self)

class AlignWorker(BaseWorkerUtil):
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

    @worker_exception_handler
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
                ls = myutils.listdir(images_path)
                for file in ls:
                    if file.endswith('align_shift.npy'):
                        shiftsFound = True
                        basename, chNames = myutils.getBasenameAndChNames(
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
                basename, chNames = myutils.getBasenameAndChNames(
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
                
                file_path = myutils.getChannelFilePath(images_path, chName)

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
                            '`1. Launch data prep module...` before aligning the '
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
                            file_path = myutils.getChannelFilePath(
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
                            file_path = myutils.getChannelFilePath(
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
            myutils.to_tiff(filePath, data)
        elif ext == '.npz':
            np.savez_compressed(filePath, data)
        elif ext == '.h5':
            load.save_to_h5(filePath, data)

class ToObjCoordsWorker(BaseWorkerUtil):
    def __init__(self, mainWin):
        super().__init__(mainWin)
        
    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.signals.finished.emit(self)
                return
            
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
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
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )

                if posData.SizeT == 1:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                dfs = []
                n_frames = len(posData.segm_data)
                self.signals.initProgressBar.emit(n_frames)
                for frame_i, lab in enumerate(posData.segm_data):
                    df_coords_i = myutils.from_lab_to_obj_coords(lab)
                    dfs.append(df_coords_i)
                    self.signals.progressBar.emit(1)
                df_filepath = posData.segm_npz_path.replace('.npz', '.csv')
                df_filepath = df_filepath.replace('_segm', '_objects_coordinates')

                keys = list(range(len(posData.segm_data)))
                df = pd.concat(dfs, keys=keys, names=['frame_i'])
                
                self.signals.initProgressBar.emit(0)
                df.to_csv(df_filepath)
                        
        self.signals.finished.emit(self)

class Stack2DsegmTo3Dsegm(BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigAborted = Signal()

    def __init__(self, mainWin, SizeZ):
        super().__init__(mainWin)
        self.SizeZ = SizeZ

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f'Select <b>2D segmentation file to stack</b>'
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
                ls = myutils.listdir(images_path)
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
                if posData.segm_data.ndim == 2:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                self.logger.log('Stacking 2D into 3D objects...')
                
                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                T, Y, X = posData.segm_data.shape
                newShape = (T, self.SizeZ, Y, X)
                segmData2D = np.zeros(newShape, dtype=np.uint32)
                for frame_i, lab in enumerate(posData.segm_data):
                    stacked_lab = core.stack_2Dlab_to_3D(lab, self.SizeZ)
                    segmData2D[frame_i] = stacked_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log('Saving stacked 3D segmentation file...')
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f'{segmFilename}_{appendedName}.npz'
                segmData2D = np.squeeze(segmData2D)
                np.savez_compressed(newSegmFilepath, segmData2D)
                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)

class MigrateUserProfileWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    debug = Signal(object)

    def __init__(self, src_path, dst_path, acdc_folders):
        QObject.__init__(self)
        self.signals = signals()
        self.src_path = src_path
        self.dst_path = dst_path
        self.acdc_folders = acdc_folders
    
    @worker_exception_handler
    def run(self):
        import shutil
        from . import models_path

        self.progress.emit(
            'Migrating user profile data from '
            f'"{self.src_path}" to "{self.dst_path}"...'
        )
        acdc_folders = self.acdc_folders
        self.signals.initProgressBar.emit(2*len(acdc_folders))
        dst_folder = os.path.basename(self.dst_path)
        folders_to_remove = []
        for acdc_folder in acdc_folders:
            if acdc_folder == dst_folder:
                # Skip the destination folder that would be picked up if the 
                # user called it with acdc at the start of the name
                self.signals.progressBar.emit(2)
                continue
            src = os.path.join(self.src_path, acdc_folder)
            dst = os.path.join(self.dst_path, acdc_folder)
            self.progress.emit(f'Copying {src} to {dst}...')
            files_failed_move = copy_or_move_tree(
                src, dst, copy=False,
                sigInitPbar=self.signals.sigInitInnerPbar, 
                sigUpdatePbar=self.signals.sigUpdateInnerPbar
            )
            folders_to_remove.append(src)
            self.signals.progressBar.emit(1)
        
        for to_remove in folders_to_remove:
            try:
                self.progress.emit(f'Removing "{to_remove}"...')
                shutil.rmtree(to_remove)
            except Exception as err:
                self.progress.emit(
                    '--------------------------------------------------------\n'
                    f'[WARNING]: Removal of the folder "{to_remove}" failed. '
                    'Please remove manually.\n'
                    '--------------------------------------------------------'
                )
            finally:
                self.signals.progressBar.emit(1)
        
        # Update model's paths
        load.migrate_models_paths(self.dst_path)        
        
        # Store user profile data folder path
        from . import user_profile_path_txt
        os.makedirs(os.path.dirname(user_profile_path_txt), exist_ok=True)
        with open(user_profile_path_txt, 'w') as txt:
            txt.write(self.dst_path)
        
        self.finished.emit(self)

class DelObjectsOutsideSegmROIWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    debug = Signal(object)

    def __init__(
            self, 
            segm_roi_endname: os.PathLike, 
            segm_data: np.ndarray,
            images_path: os.PathLike
        ):
        QObject.__init__(self)
        self.signals = signals()
        self.segm_roi_endname = segm_roi_endname
        self.segm_data = segm_data
        self.images_path = images_path
        
    @worker_exception_handler
    def run(self):
        segm_roi_endname = self.segm_roi_endname
        segm_roi_filepath, _ = load.get_path_from_endname(
            segm_roi_endname, self.images_path
        )
        self.progress.emit(f'Loading segmentation file "{segm_roi_filepath}"...')
        segm_roi_data = load.load_image_file(segm_roi_filepath)
        
        self.progress.emit(f'Deleting objects outside of selected ROIs...')
        cleared_segm_data, delIDs = transformation.del_objs_outside_segm_roi(
            segm_roi_data, self.segm_data
        )
        
        self.finished.emit((self, cleared_segm_data, delIDs))

class ConcatSpotmaxDfsWorker(BaseWorkerUtil):
    sigAborted = Signal()
    sigAskFolder = Signal(str)
    sigSetMeasurements = Signal(object)
    sigAskAppendName = Signal(str, list)

    def __init__(self, mainWin, format='CSV'):
        super().__init__(mainWin)
        if format.startswith('CSV'):
            self._final_ext = '.csv'
        elif format.startswith('XLS'):
            self._final_ext = '.xlsx'
        self.acdcOutputEndname = None
    
    def emitSetMeasurements(self, kwargs):
        self.mutex.lock()
        self.sigSetMeasurements.emit(kwargs)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def emitAskAppendName(self, allPos_spotmax_df_basename):
        # Ask appendend name
        self.mutex.lock()
        self.sigAskAppendName.emit(allPos_spotmax_df_basename, [])
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitAskCopyCca(self, images_path):
        self.mutex.lock()
        self.signals.sigAskCopyCca.emit(images_path)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def setAcdcOutputEndname(self, acdcOutputEndname):
        self.acdcOutputEndname = acdcOutputEndname
    
    def getAcdcDf(self, images_path):
        if self.acdcOutputEndname is None:
            return
        
        for file in myutils.listdir(images_path):
            if not file.endswith(self.acdcOutputEndname):
                continue
            
            filepath = os.path.join(images_path, file)
            acdc_df = pd.read_csv(filepath, index_col=['frame_i', 'Cell_ID'])
            return acdc_df
    
    def copyCcaColsFromAcdcDf(self, df, acdc_df, debug=False):
        if acdc_df is None:
            return df
        
        if debug:
            printl(acdc_df.columns.to_list(), pretty=True)
        
        idx = df.index.intersection(acdc_df.index)
        for col in cca_df_colnames:
            if col not in acdc_df.columns:
                continue
        
            if col not in self.selectedColumns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
        
        for col in lineage_tree_cols:
            if col not in acdc_df.columns:
                continue
            
            if col not in self.selectedColumns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
        
        for col in default_annot_df.keys():
            if col not in acdc_df.columns:
                continue
            
            if col not in self.selectedColumns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
        
        for col in self.selectedColumns:
            if col not in acdc_df.columns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
            
            if debug and col == 'cell_vol_fl':
                printl(df[[col]])
        
        return df
    
    def emitAskFolderWhereToSaveMultiExp(self):
        self.mutex.lock()
        self.sigAskFolder.emit('')
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return

        return self.allExpSaveFolder
    
    def askSelectMeasurements(self, exp_path, posFoldernames):
        acdc_dfs = []
        keys = []
        for p, pos in enumerate(posFoldernames):
            if self.abort:
                self.sigAborted.emit()
                return False

            images_path = os.path.join(exp_path, pos, 'Images')
            acdc_df = self.getAcdcDf(images_path)
            if acdc_df is None:
                continue
            
            acdc_dfs.append(acdc_df)
            keys.append(pos)
        
        if not acdc_dfs:
            return True
        
        acdc_df_allpos = pd.concat(
            acdc_dfs, keys=keys, names=['Position_n', 'frame_i', 'Cell_ID']
        )
        acdc_df_allpos['experiment_folderpath'] = exp_path
        basename, chNames = myutils.getBasenameAndChNames(
            images_path, useExt=('.tif', '.h5')
        )
        df_metadata = load.load_metadata_df(images_path)
        SizeZ = df_metadata.at['SizeZ', 'values']
        SizeZ = int(float(SizeZ))
        existing_colnames = acdc_df_allpos.columns
        isSegm3D = any([col.endswith('3D') for col in existing_colnames])
        
        kwargs = {
            'loadedChNames': chNames, 
            'notLoadedChNames': [],
            'isZstack': SizeZ > 1,
            'isSegm3D': isSegm3D,
            'existing_colnames': existing_colnames
        }
        self.emitSetMeasurements(kwargs)
        if self.abort:
            self.sigAborted.emit()
            return False
        
        return True
    
    @worker_exception_handler
    def run(self):
        from spotmax import DFs_FILENAMES, DF_REF_CH_FILENAME
        from spotmax.utils import get_runs_num_and_desc
        import spotmax.io
        
        self.selectedColumns = None
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        spotmax_dfs_spots_allexp = defaultdict(lambda: defaultdict(list))
        spotmax_dfs_aggr_allexp = defaultdict(lambda: defaultdict(list))
        ref_ch_dfs_allexp = defaultdict(lambda: defaultdict(list))
        runNumberAlreadyAsked = False
        copyFromCcaAlreadyAsked = False
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)
            
            all_runs = get_runs_num_and_desc(
                exp_path, pos_foldernames=pos_foldernames
            )
            if not all_runs:
                self.logger.log(
                    '[WARNING] The following experiment does not contain '
                    f'valid spotMAX output files. Skipping it. "{exp_path}"'
                )
                continue
            
            if not runNumberAlreadyAsked:
                abort = self.emitSelectSpotmaxRun(
                    exp_path, pos_foldernames, all_runs, 
                    infoText=' to combine',
                    allowSingleSelection=True, 
                    multiSelection=False
                )
                if abort:
                    self.sigAborted.emit()
                    return
                runNumberAlreadyAsked = True
            
            selectedSpotmaxRuns = self.mainWin.selectedSpotmaxRuns

            self.signals.initProgressBar.emit(len(pos_foldernames))
            dfs_spots = defaultdict(list)
            dfs_aggr = defaultdict(list)
            dfs_ref_ch = defaultdict(list)
            pos_runs = defaultdict(list)
            pos_runs_ref_ch = defaultdict(list)
            pos_ini_filepaths = {}
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return
                
                pos_path = os.path.join(exp_path, pos)
                spotmax_output_path = os.path.join(pos_path, 'spotMAX_output')
                
                if not os.path.exists(spotmax_output_path):
                    self.logger.log(
                        '[WARNING] The following Position folder does not contain '
                        f'valid spotMAX output files. Skipping it. "{pos_path}"'
                    )
                    continue
                
                images_path = os.path.join(exp_path, pos, 'Images')
                
                if not copyFromCcaAlreadyAsked:
                    self.emitAskCopyCca(images_path)
                    if self.abort:
                        self.sigAborted.emit()
                        return
                    
                    self.askSelectMeasurements(exp_path, pos_foldernames)
                    if self.abort:
                        return  
                    copyFromCcaAlreadyAsked = True            
                    
                acdc_df = self.getAcdcDf(images_path)
                
                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                
                for run_desc in selectedSpotmaxRuns:
                    run, desc = run_desc.split('_...')
                    ini_filename = f'{run}_analysis_parameters{desc}.ini'
                    ini_filepath = os.path.join(
                        spotmax_output_path, ini_filename
                    )
                    if not os.path.exists(ini_filepath):
                        self.logger.log(
                            '[WARNING] The following Position folder does not contain '
                            f'the spotMAX output file for run number {run}. '
                            f'Skipping it. "{pos_path}"'
                        )
                        continue
                        
                    pos_ini_filepaths[(run, desc)] = ini_filepath
                    for _, pattern_filename in DFs_FILENAMES.items():
                        run_filename = pattern_filename.replace('*rn*', run)
                        run_filename = run_filename.replace('*desc*', desc)
                        aggr_filename = f'{run_filename}_aggregated.csv'
                        aggr_filepath = os.path.join(
                            spotmax_output_path, aggr_filename
                        )
                        if not os.path.exists(aggr_filepath):
                            continue                        
                        
                        df_spots_filename = f'{run_filename}.h5'
                        spots_filepath = os.path.join(
                            spotmax_output_path, df_spots_filename
                        )
                        ext_spots = '.h5'
                        if not os.path.exists(spots_filepath):
                            df_spots_filename = f'{run_filename}.csv'
                            spots_filepath = os.path.join(
                                spotmax_output_path, df_spots_filename
                            )
                            ext_spots = '.csv'
                        
                        if not os.path.exists(spots_filepath):
                            continue
                        
                        analysis_step = re.findall(
                            r'\*rn\*(.*)\*desc\*', pattern_filename
                        )[0]
                        key = (run, analysis_step, desc, ext_spots)
                        try:
                            df_spots = spotmax.io.load_spots_table(
                                spotmax_output_path, df_spots_filename
                            ).reset_index().set_index(['frame_i', 'Cell_ID'])
                            df_spots = self.copyCcaColsFromAcdcDf(
                                df_spots, acdc_df, debug=False
                            )
                            df_spots = (
                                df_spots.reset_index()
                                .set_index(['frame_i', 'Cell_ID', 'spot_id'])
                            )
                            dfs_spots[key].append(df_spots)
                        except Exception as err:
                            self.logger.log(str(err), level='ERROR')
                            self.logger.log(
                                'WARNING: Error when reading single-spots '
                                'tables (possibly because there are no spots). '
                                'Skipping this Position.', 
                                level='WARNING'
                            )
                            pass
                        
                        df_aggregated = pd.read_csv(
                            aggr_filepath, index_col=['frame_i', 'Cell_ID']
                        )
                        df_aggregated = self.copyCcaColsFromAcdcDf(
                            df_aggregated, acdc_df
                        )
                        dfs_aggr[key].append(df_aggregated)
                        pos_runs[key].append(pos)
                    
                    ref_ch_id_text = re.findall(
                        r'\*rn\*(.*)\*desc\*', DF_REF_CH_FILENAME
                    )[0]
                    ref_ch_filename = (
                        DF_REF_CH_FILENAME.replace('*rn*', run)
                    )
                    ref_ch_filename = (
                        ref_ch_filename.replace('*desc*', desc)
                    )
                    ref_ch_filepath = os.path.join(
                        spotmax_output_path, ref_ch_filename
                    )
                    if not os.path.exists(ref_ch_filepath):
                        continue
                    
                    df_ref_ch = pd.read_csv(
                        ref_ch_filepath, index_col=['frame_i', 'Cell_ID']
                    )
                    df_ref_ch = self.copyCcaColsFromAcdcDf(df_ref_ch, acdc_df)
                    ref_ch_key = (run, ref_ch_id_text, desc)
                    dfs_ref_ch[ref_ch_key].append(df_ref_ch)
                    pos_runs_ref_ch[ref_ch_key].append(pos)

                self.signals.progressBar.emit(1)        
            
            self.signals.initProgressBar.emit(0)
            
            self.logger.log('Saving concantenated files...')
            
            allpos_folderpath = os.path.join(exp_path, 'spotMAX_multipos_output')
            os.makedirs(allpos_folderpath, exist_ok=True)
            
            exp_name = os.path.basename(exp_path)
            for key, dfs in dfs_spots.items():
                pos_keys = pos_runs[key]
                run, analysis_step, desc, ext_spots = key
                
                if ext_spots == '.csv':
                    ext_spots = self._final_ext
                filename = f'multipos_{run}{analysis_step}{desc}{ext_spots}'
                df_spots_concat = spotmax.io.save_concat_dfs(
                    dfs, pos_keys, allpos_folderpath, filename, ext_spots, 
                    names=['Position_n'], return_concat_df=True
                )
                df_spots_concat['exp_foldername'] = exp_name
                spotmax_dfs_spots_allexp[filename]['dfs'].append(df_spots_concat)
                spotmax_dfs_spots_allexp[filename]['keys'].append(
                    exp_path
                )
                ini_filepath = pos_ini_filepaths[(run, desc)]
                ini_filename = os.path.basename(ini_filepath)
                dst_ini_filepath = os.path.join(allpos_folderpath, ini_filename)
                if not os.path.exists(dst_ini_filepath):
                    shutil.copy2(ini_filepath, dst_ini_filepath)
                    
                spotmax_dfs_spots_allexp[filename]['ini_filepath'].append(
                    dst_ini_filepath
                )
            
            for key, dfs in dfs_aggr.items():
                pos_keys = pos_runs[key]
                run, analysis_step, desc, _ = key
                filename = (
                    f'multipos_{run}{analysis_step}{desc}'
                    f'_aggregated{self._final_ext}'
                )
                df_aggr_concat = spotmax.io.save_concat_dfs(
                    dfs, pos_keys, allpos_folderpath, filename, self._final_ext, 
                    names=['Position_n'], return_concat_df=True
                )
                spotmax_dfs_aggr_allexp[filename]['dfs'].append(df_aggr_concat)
                spotmax_dfs_aggr_allexp[filename]['keys'].append(
                    (exp_path, exp_name)
                )
            
            for key, dfs in dfs_ref_ch.items():
                run, ref_ch_id_text, desc = key
                pos_keys = pos_runs_ref_ch[key]
                filename = (
                    f'multipos_{run}{ref_ch_id_text}{desc}{self._final_ext}'
                )
                df_ref_ch_concat = spotmax.io.save_concat_dfs(
                    dfs, pos_keys, allpos_folderpath, filename, self._final_ext,
                    names=['Position_n'], return_concat_df=True
                )
                ref_ch_dfs_allexp[filename]['dfs'].append(df_ref_ch_concat)
                ref_ch_dfs_allexp[filename]['keys'].append(
                    (exp_path, exp_name)
                )
        
        multiexp_dst_folderpath = ''
        if len(expPaths) == 1:
            self.signals.finished.emit(self)
            return
        
        multiexp_dst_folderpath = self.emitAskFolderWhereToSaveMultiExp()
        if multiexp_dst_folderpath is None:
            return
        
        names = ['experiment_folderpath', 'experiment_foldername']
        for filename, items in spotmax_dfs_spots_allexp.items():
            keys = items['keys']
            dfs = items['dfs']
            multiexp_filename = f'multiexp_{filename}'
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs, keys, multiexp_dst_folderpath, 
                multiexp_filename, 
                extension,
                names=['experiment_folderpath']
            )
            ini_filepath = items['ini_filepath'][0]
            ini_filename = os.path.basename(ini_filepath)
            dst_ini_filepath = os.path.join(
                multiexp_dst_folderpath, ini_filename
            )
            if not os.path.exists(dst_ini_filepath):
                shutil.copy2(ini_filepath, dst_ini_filepath)
            
        for filename, items in spotmax_dfs_aggr_allexp.items():
            keys = items['keys']
            dfs = items['dfs']
            multiexp_filename = f'multiexp_{filename}'
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs, keys, multiexp_dst_folderpath, 
                multiexp_filename, 
                extension,
                names=names
            )
        
        for filename, items in spotmax_dfs_aggr_allexp.items():
            keys = items['keys']
            dfs = items['dfs']
            multiexp_filename = f'multiexp_{filename}'
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs, keys, multiexp_dst_folderpath, 
                multiexp_filename, 
                extension,
                names=names
            )
        
        self.signals.finished.emit(self)

class FilterObjsFromCoordsTable(BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigAborted = Signal()
    sigSetColumnsNames = Signal(object, object, object)

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitSetColumnsNames(self, columns, categories, optionalCategories):
        self.mutex.lock()
        self.sigSetColumnsNames.emit(columns, categories, optionalCategories)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort 
    
    def getColumnsCategories(
            self, df_coords, exp_path, pos_foldernames, endFilenameSegm
        ):
        columns = df_coords.columns.to_list()
        categories = ['X coord. column', 'Y coord. column']
        optionalCategories = []
        
        images_path = os.path.join(exp_path, pos_foldernames[0], 'Images')
        metadata_df = load.load_metadata_df(images_path)
        SizeT = float(metadata_df.at['SizeT', 'values'])
        SizeZ = float(metadata_df.at['SizeZ', 'values'])
        
        segmData = load.load_segm_file(
            images_path, end_name_segm_file=endFilenameSegm
        )
        
        if segmData.ndim == 4:
            categories.append('Z coord. column')
            categories.append('Frame index column')
        elif segmData.ndim == 3:
            if SizeZ > 1 and SizeT == 1:
                # 3D z-stack data
                categories.append('Z coord. column')
            else:
                optionalCategories.append('Z coord. column')
                
            if SizeT > 1:
                # 3D time-lapse
                categories.append('Frame index column')
            else:
                optionalCategories.append('Frame index column')
        else:
            optionalCategories.append('Z coord. column')
            optionalCategories.append('Frame index column')
        
        if len(pos_foldernames) > 1:
            categories.append('Position_n')
        else:
            optionalCategories.append('Position_n')
        
        return columns, categories, optionalCategories
        
    def getDfCoords(
            self, df_coords, selectedColumnsPerCategory, pos_foldername, frame_i
        ):
        pos_col = selectedColumnsPerCategory.get('Position_n', 'None')
        frame_i_col = selectedColumnsPerCategory.get(
            'Frame index column', 'None'
        )
        x_col = selectedColumnsPerCategory['X coord. column']
        y_col = selectedColumnsPerCategory['Y coord. column']
        if pos_col != 'None':
            df_coords = df_coords[df_coords[pos_col] == pos_foldername]
        if frame_i_col != 'None':
            df_coords = df_coords[df_coords[frame_i_col] == frame_i]
        
        xy_cols = [x_col, y_col]
        
        df_out = pd.DataFrame(
            index=df_coords.index, 
            data=df_coords[xy_cols].values,
            columns=['x', 'y']
        )
        z_col = selectedColumnsPerCategory.get('Z coord. column', 'None')
        if z_col != 'None':
            df_out['z'] =  df_coords[z_col]
        
        return df_out
    
    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f'Select <b>segmentation file to filter</b>'
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            endFilenameSegm = self.mainWin.endFilenameSegm
            
            self.logger.log('Asking to select the CSV table file...')
            
            abort = self.emitSelectFile(
                exp_path, 'Select CSV table file with coordinates to filter',
                'CSV (*.csv)'
            )
            if abort:
                self.sigAborted.emit()
                return
            
            self.logger.log(
                f'Loading table file `{self.mainWin.selectedFilepath}`..'
            )
            df_coords = pd.read_csv(self.mainWin.selectedFilepath)
            
            columns, categories, optionalCategories = self.getColumnsCategories(
                df_coords, exp_path, pos_foldernames, endFilenameSegm
            )            
            
            abort = self.emitSetColumnsNames(
                columns, categories, optionalCategories
            )
            if abort:
                self.sigAborted.emit()
                return
            
            selectedColumnsPerCategory = self.mainWin.selectedColumnsPerCategory
            
            # Ask appendend name
            self.mutex.lock()
            self.sigAskAppendName.emit(
                self.mainWin.endFilenameSegm, 
                self.mainWin.existingSegmEndNames
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
                ls = myutils.listdir(images_path)
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
                if posData.SizeT == 1:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                self.logger.log('Filtering objects...')
                
                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                filteredSegmData = np.zeros_like(posData.segm_data)
                for frame_i, lab in enumerate(posData.segm_data):
                    df_coords_frame_i = self.getDfCoords(
                        df_coords, selectedColumnsPerCategory, pos, frame_i
                    )
                    if df_coords_frame_i.empty:
                        num_frames_missing = len(posData.segm_data[frame_i:])
                        self.signals.sigUpdateInnerPbar.emit(num_frames_missing)
                        filteredSegmData = filteredSegmData[:frame_i]
                        break
                    
                    filtered_lab = core.filter_segm_objs_from_table_coords(
                        lab, df_coords_frame_i
                    )
                    filteredSegmData[frame_i] = filtered_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log('Saving filtered segmentation file...')
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f'{segmFilename}_{appendedName}.npz'
                filteredSegmData = np.squeeze(filteredSegmData)
                np.savez_compressed(newSegmFilepath, filteredSegmData)
                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)

class ScreenRecorderWorker(QObject):
    sigGrabScreen = Signal()
    finished = Signal()

    def __init__(self, screenRecorderWin, folder_path):
        QObject.__init__(self)
        self.screenRecorderWin = screenRecorderWin
        self.folder_path = folder_path
    
    def run(self):
        for i in range(4):
            fn = f'shot_{i:03}.jpg'
            grab_path = os.path.join(self.folder_path, fn)
            screen = self.screenRecorderWin.screen()
            screenshot = screen.grabWindow(self.screenRecorderWin.winId())
            screenshot.save(grab_path, 'jpg')
            print(grab_path)
            time.sleep(0.2)

        self.finished.emit()

class CcaIntegrityCheckerWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str, object)
    sigDone = Signal()
    sigWarning = Signal(str, str)
    sigFixWillDivide = Signal(str, list)
    
    def __init__(self, mutex, waitCond):
        QObject.__init__(self)
        self.logger = workerLogger(self.progress)
        self.mutex = mutex
        self.waitCond = waitCond
        self.exit = False
        self.isFinished = False
        self.abortChecking = False
        self.isChecking = False
        self.isPaused = False
        self.debug = False
        self.dataQ = deque(maxlen=10)
    
    def pause(self):
        if self.debug:
            self.logger.log('Cell cycle annotations checker is idle.')
        self.mutex.lock()
        self.isPaused = True
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        self.isPaused = False
    
    def enqueue(self, posData):
        # First stop previous checking
        if self.isChecking:
            self.abortChecking = True
        self._enqueue(posData)
    
    def _enqueue(self, posData):
        if self.debug:
            self.logger.log('Enqueing posData...')
        self.dataQ.append(posData)
        if len(self.dataQ) == 1:
            # Wake worker upon inserting first element
            self.abortChecking = False
            self.waitCond.wakeAll()
    
    def clearQueue(self):
        self.dataQ.clear()
    
    def _stop(self):
        self.exit = True
        self.waitCond.wakeAll()
    
    def abort(self):
        self.abortChecking = True
        while not len(self.dataQ) == 0:
            data = self.dataQ.pop()
            del data
        self._stop()
    
    def _check_equality_num_mothers_buds_in_S(self, checker, frame_i):
        num_moth_S, num_buds = checker.get_num_mothers_and_buds_in_S()
        
        if num_moth_S == num_buds:
            return True
        
        category = 'number of buds different from number of mothers in S phase'
        ul_items = [
            f'Number of buds = {num_buds}', 
            f'Number of mothers in S phase = {num_moth_S}'
        ]
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} the number of buds and number of '
            'mother cells in S phase are different!'
            f'{html_utils.to_list(ul_items)}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_mothers_multiple_buds(self, checker, frame_i):
        mother_IDs_with_multiple_buds = (
            checker.get_mother_IDs_with_multiple_buds()
        )
        if len(mother_IDs_with_multiple_buds) == 0:
            return True

        category = 'mother cells with multiple buds'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following mother cells have <b>multiple buds</b> assigned to it'
            f'<br><br>{mother_IDs_with_multiple_buds}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_cells_without_G1(self, checker, global_cca_df):
        IDs_cycles_without_G1 = (
            checker.get_IDs_cycles_without_G1(global_cca_df)
        )
        if len(IDs_cycles_without_G1) == 0:
            return True

        category = 'cell cycles without G1'
        txt = html_utils.paragraph(
            'Cell-ACDC requires that every cell cycle has at least '
            'one frame in G1.<br>'
            'The following pairs of <code>(ID, generation number)</code> '
            'do not satisfy this condition:<br><br>'
            f'{IDs_cycles_without_G1}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_will_divide_is_true(self, checker, global_cca_df):
        # NOTE: unfortunately this function performs pandas manipulations 
        # that are either not thread-safe or in any case are freezing the 
        # GUI. For now we don't run this until we find a solution
        return True
    
        IDs_will_divide_wrong = (
            checker.get_IDs_gen_num_will_divide_wrong(global_cca_df)
        )
        if len(IDs_will_divide_wrong) == 0:
            return True

        txt = html_utils.paragraph(
            'Cell-ACDC found that `will_divide` is annotated as True on the '
            'following <code>(ID, generation number)</code> cell<br>'
            'despite the fact that division is still not annotated on '
            'these cells <br><br>:'
            f'{IDs_will_divide_wrong}'
        )
        self.sigFixWillDivide.emit(txt, IDs_will_divide_wrong)
        return False
    
    def _check_buds_gen_num_zero(self, checker, frame_i):
        bud_IDs_gen_num_nonzero = (
            checker.get_bud_IDs_gen_num_nonzero()
        )
        if len(bud_IDs_gen_num_nonzero) == 0:
            return True

        category = 'buds whose generation number is not zero'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following bud IDs have generation number different from 0:'
            f'<br><br>{bud_IDs_gen_num_nonzero}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_mothers_gen_num_greater_one(self, checker, frame_i):
        moth_IDs_gen_num_non_greater_one = (
            checker.get_moth_IDs_gen_num_non_greater_one()
        )
        if len(moth_IDs_gen_num_non_greater_one) == 0:
            return True

        category = 'mothers whose generation number is < 1'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following mother cells have generation number &lt; 1:'
            f'<br><br>{moth_IDs_gen_num_non_greater_one}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_buds_G1(self, checker, frame_i):
        buds_G1 = (
            checker.get_buds_G1()
        )
        if len(buds_G1) == 0:
            return True

        category = 'buds in G1'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following bud IDs are in G1 (buds must be in S):'
            f'<br><br>{buds_G1}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_cell_S_rel_ID_zero(self, checker, frame_i):
        cell_S_rel_ID_zero = (
            checker.get_cell_S_rel_ID_zero()
        )
        if len(cell_S_rel_ID_zero) == 0:
            return True

        category = 'buds in G1'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following cell IDs in S phase do not have '
            '<code>relative_ID > 0</code>:'
            f'<br><br>{cell_S_rel_ID_zero}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_ID_rel_ID_mismatches(self, checker, frame_i):
        ID_rel_ID_mismatches = checker.get_ID_rel_ID_mismatches()
        if len(ID_rel_ID_mismatches) == 0:
            return True

        items = [
            f'Cell ID {ID} has relative ID = {relID}, '
            f'while cell ID {relID} has relative ID = {relID_of_relID}'
            for ID, relID, relID_of_relID in ID_rel_ID_mismatches
        ]
        category = '`ID-relative_ID` mismatches'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'there are the following `ID-relative_ID` mismatches:'
            f'{html_utils.to_list(items)}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_lonely_cells_in_S(self, checker, frame_i):
        lonely_cells_in_S = checker.get_lonely_cells_in_S()
        if len(lonely_cells_in_S) == 0:
            return True

        category = 'Lovely cells in S phase'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following cell IDs are in `S` phase but their `relative_ID` '
            f'does not exist:<br><br>'
            f'{lonely_cells_in_S}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _get_cca_df_copy(self, acdc_df):
        try:
            cca_df = pd.DataFrame(
                data=acdc_df[cca_df_colnames].values,
                columns=cca_df_colnames,
                index=acdc_df.index
            )
            return cca_df
        except KeyError as error:
            return 
        
    def check(self, posData):    
        self.isChecking = True
        checkpoints = (
            '_check_lonely_cells_in_S',
            '_check_equality_num_mothers_buds_in_S',
            '_check_mothers_multiple_buds',
            '_check_buds_gen_num_zero',
            '_check_mothers_gen_num_greater_one',
            '_check_buds_G1',
            '_check_cell_S_rel_ID_zero',
            '_check_ID_rel_ID_mismatches'
        )
        cca_dfs = []
        keys = []
        check_integrity_globally = True
        for frame_i, data_dict in enumerate(posData.allData_li):
            if self.abortChecking:
                check_integrity_globally = False
                break
            
            lab = data_dict['labels']
            if lab is None:
                break
            
            cca_df = data_dict.get('cca_df_checker')
            if cca_df is None:
                # There are no annotations at frame_i --> stop
                break
            
            IDs = data_dict['IDs']
            checker = core.CcaIntegrityChecker(cca_df, lab, IDs)
            
            for checkpoint in checkpoints:
                proceed = getattr(self, checkpoint)(checker, frame_i)
                if not proceed:
                    break
            
            if not proceed:
                check_integrity_globally = False
                break
            
            cca_dfs.append(cca_df)
            keys.append(frame_i)
        
        if check_integrity_globally and len(cca_dfs)>1:
            global_checkpoints = [
                '_check_cells_without_G1',
                # '_check_will_divide_is_true'
            ]
            # Check integrity globally
            global_cca_df = pd.concat(cca_dfs, keys=keys, names=['frame_i'])
            for checkpoint in global_checkpoints:
                proceed = getattr(self, checkpoint)(checker, global_cca_df)
                if not proceed:
                    break
        
        self.abortChecking = False
        self.isChecking = False
        time.sleep(1)
    
    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log('Closing cell cycle integrity checker worker...')
                break
            elif not len(self.dataQ) == 0:
                if self.debug:
                    self.logger.log(
                        'Checking integrity of cell cycle annotations '
                        f'({len(self.dataQ)})...'
                    )
                data = self.dataQ.pop()
                self.check(data)
                if len(self.dataQ) == 0:
                    self.sigDone.emit()
            else:
                self.pause()
        self.isFinished = True
        self.finished.emit(self)
    
class ApplyImageFilterWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    
    def __init__(self, filter_func, input_data):
        QObject.__init__(self)
        self.filter_func = filter_func
        self.input_data = input_data
    
    @worker_exception_handler
    def run(self):
        self.progress.emit('Filtering image...')
        filtered_data = self.filter_func(self.input_data)
        self.finished.emit(filtered_data)

class MoveTempFilesWorker(QObject):
    def __init__(self, temp_files_to_move: Dict[os.PathLike, os.PathLike]):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.temp_files_to_move = temp_files_to_move
    
    @worker_exception_handler
    def run(self):
        for src, dst in self.temp_files_to_move.items():
            self.logger.log(f'Saving channel data to: {dst}...')
            shutil.move(src, dst)
            tempDir = os.path.dirname(src)
            shutil.rmtree(tempDir)
            self.signals.progressBar.emit(1)
        self.signals.finished.emit(self)

class ResizeUtilWorker(BaseWorkerUtil):
    sigSetResizeProps = Signal(str)
    
    def emitSetResizeProps(self, input_path):
        self.mutex.lock()
        self.sigSetResizeProps.emit(input_path)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    def __init__(self, mainWin):
        super().__init__(mainWin)
    
    def validateOutputPath(self, path):
        images_path = myutils.validate_images_path(path, create_dirs_tree=True)
        return images_path
    
    @worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            abort = self.emitSetResizeProps(exp_path)
            if abort:
                self.signals.finished.emit(self)
                return
            
            tot_pos = len(pos_foldernames)
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )
                images_path = os.path.join(exp_path, pos, 'Images')
                
                rf = self.resizeFactor
                text_to_append = self.textToAppend
                images_path_out = self.validateOutputPath(self.expFolderpathOut)
                resize.run(
                    images_path, rf, text_to_append=text_to_append, 
                    images_path_out=images_path_out
                )                
                
        self.signals.finished.emit(self)