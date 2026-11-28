"""Main GUI (guiWin-only) workers that load data and store state:
loadDataWorker (loads Positions), LazyLoader (loads frames in the
background), StoreGuiStateWorker (undo states) and AutoSaveWorker
(periodic autosave of the annotations).

The save worker of the main GUI, saveDataWorker, is in
shared/workers/saving.py.
"""
import os
from collections import deque
import numpy as np
import pandas as pd
import traceback
import queue
from qtpy.QtCore import Signal, QObject
from cellacdc import load, myutils, io
from cellacdc import cca_df_colnames, lineage_tree_cols
from cellacdc.shared.workers import base as workers_base


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
        self.logger = workers_base.workerLogger(self.progress)
    
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
        self.logger = workers_base.workerLogger(self.progress)
        self.mutex = mutex
        self.waitCond = waitCond
        self.exit = False
        self.isFinished = False
        self.stopSaving = False
        self.isSaving = False
        self.isPaused = False
        self.dataQ = deque(maxlen=2)
        self.isAutoSaveON = False
        self.isAutoSaveAnnotON = True
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
            self.stopSaving = True
        self._enqueue(posData)
    
    def _enqueue(self, posData):
        if self.debug:
            self.logger.log('Enqueing posData autosave...')
        self.dataQ.append(posData)
        if len(self.dataQ) == 1:
            # Wake up worker upon inserting first element
            self.stopSaving = False
            self.waitCond.wakeAll()
    
    def _stop(self):
        self.exit = True
        self.waitCond.wakeAll()
    
    def stop(self):
        self.stopSaving = True
        while not len(self.dataQ) == 0:
            data = self.dataQ.pop()
            del data
        self._stop()
    
    def cancelSaving(self):
        ...
    
    @workers_base.worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log('Closing autosaving worker...')
                break
            elif not len(self.dataQ) == 0:
                if self.debug:
                    self.logger.log('Autosaving...')
                data = self.dataQ.pop()
                self.isSaving = True
                try:
                    self.saveData(data)
                except Exception as e:
                    error = traceback.format_exc()
                    print('*'*40)
                    self.logger.log(error)
                    print('='*40)
                self.isSaving = False
                
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
        
        if not self.isAutoSaveON and not self.isAutoSaveAnnotON:
            return
        
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
        
        saved_segm_data = None
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
            if self.stopSaving:
                break
            
            # Build saved_segm_data
            lab = data_dict['labels']
            if lab is None:
                break
            
            if self.isAutoSaveON and saved_segm_data is not None:
                if posData.SizeT > 1:
                    saved_segm_data[frame_i] = lab
                else:
                    saved_segm_data = lab

            isAutoSaveAnnotON = self.isAutoSaveAnnotON # make sure that it doesn't change during execution
            if isAutoSaveAnnotON:
                acdc_df = data_dict['acdc_df']
                
                if acdc_df is None:
                    continue

            if not np.any(lab):
                continue
            
            if isAutoSaveAnnotON:
                acdc_df = load.pd_bool_and_float_to_int_to_str(
                    acdc_df, inplace=False, colsToCastInt=[]
                )
                
                acdc_df_li.append(acdc_df)
                key = (frame_i, posData.TimeIncrement*frame_i)
                keys.append(key)

            if self.stopSaving:
                break
        
        if not self.stopSaving:            
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
            self.logger.log(f'Stopped autosaving {self.stopSaving}.')

        self.stopSaving = False
    
    def _saveSegm(self, recovery_path, data):
        try:
            equalToSavedSegm = np.all(self.savedSegmData == data)
        except Exception as err:
            return
        
        if equalToSavedSegm:
            return
        else:
            io.savez_compressed(recovery_path, np.squeeze(data))
    
    def _save_acdc_df(self, recovery_acdc_df: pd.DataFrame, posData):
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
            recovery_acdc_df.reset_index(allow_duplicates=True)
            .set_index(['frame_i', 'Cell_ID'])
        )
        recovery_acdc_df = recovery_acdc_df.loc[
            :, ~recovery_acdc_df.columns.duplicated()
        ]
        try:
            # Try to insert into the recovery_acdc_df any column that was saved
            # but is not in the recovered df (e.g., metrics). Not annotation
            # columns: they are missing because the annotations were removed
            # (e.g. after a segmentation edit from the first frame)
            df_left = recovery_acdc_df
            existing_cols = df_left.columns.intersection(saved_acdc_df.columns)
            annot_cols = [
                col for col in saved_acdc_df.columns
                if col in cca_df_colnames or col in lineage_tree_cols
                or str(col).startswith('sister_ID_tree')
            ]
            df_right = saved_acdc_df.drop(
                columns=existing_cols.union(annot_cols)
            )
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
            try:
                reference_acdc_df = (
                    pd.read_csv(last_unsaved_csv_path, dtype=load.acdc_df_str_cols)
                    .set_index(['frame_i', 'Cell_ID'])
                )
            except Exception as e:
                self.logger.log(f'[WARNING]: {e}')
                reference_acdc_df = saved_acdc_df
        
        if myutils.are_acdc_dfs_equal(recovery_acdc_df, reference_acdc_df):
            return
        
        load.store_unsaved_acdc_df(recovery_folderpath, recovery_acdc_df)


class loadDataWorker(QObject):
    def __init__(self, mainWin, user_ch_file_paths, user_ch_name, firstPosData):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.mainWin = mainWin
        self.user_ch_file_paths = user_ch_file_paths
        self.user_ch_name = user_ch_name
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.mutex = self.mainWin.loadDataMutex
        self.waitCond = self.mainWin.loadDataWaitCond
        self.firstPosData = firstPosData
        self.abort = False
        self.loadUnsaved = False
        self.recoveryAsked = False
        self.loadSafeOverwriteNpz = False

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

    @workers_base.worker_exception_handler
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
                labelBoolSegm=self.mainWin.labelBoolSegm,
            )
            posData.labelSegmData()

            if i == 0:
                posData.segmFound = segmFound

            posData.addYXcentroidColsIfMissing(show_progress=True)
            
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
                or posData.isSafeNpzOverwritePresent()
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
                elif self.loadSafeOverwriteNpz:
                    self.logger.log('Loading safe npz overwrite...')
                    segm_safe_npz_path = posData.getSafeNpzOverwritePath()
                    posData.segm_data = np.load(segm_safe_npz_path)['arr_0']

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


class LazyLoader(QObject):
    sigLoadingFinished = Signal()

    def __init__(self, mutex, waitCond, readH5mutex, waitReadH5cond):
        QObject.__init__(self)
        self.signals = workers_base.signals()
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

    @workers_base.worker_exception_handler
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


def are_acdc_dfs_equal(df_left, df_right):
    if df_left.shape != df_right.shape:
        return False
    
    try:
        for col in df_left.columns:
            if col not in df_right.columns:
                return False
            
            try:
                eq_mask = np.isclose(df_left[col], df_right[col], equal_nan=True)
            except Exception as err:
                # Data type is string
                eq_mask = df_left[col] == df_right[col]
            
            nan_mask = ((df_left[col].isna()) & (df_right[col].isna()))
            equality_mask = (eq_mask) | (nan_mask)
            if not equality_mask.all():
                return False
    except Exception as err:
        return False
    
    return True
