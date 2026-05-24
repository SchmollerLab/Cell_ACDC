"""Background Qt workers: io."""

import re
import os
import shutil
import time
import json
import concurrent.futures
from functools import partial
from collections import defaultdict, deque
import itertools

from typing import Union, List, Dict, Callable, Any, Tuple, Iterable

from functools import wraps
import numpy as np
import pandas as pd
import h5py
import traceback

import skimage.io
import skimage.measure
import skimage.exposure

import queue

from tqdm import tqdm

from qtpy.QtCore import Signal, QObject, QMutex, QWaitCondition

from cellacdc import html_utils

from .. import load, utils, core, prompts, printl, config, segm_re_pattern, io
from .. import transformation, measurements, cca_functions
from ..path import copy_or_move_tree
from .. import features, plot
from .. import core
from .. import cca_df_colnames, lineage_tree_cols, default_annot_df
from .. import cca_df_colnames_with_tree
from .. import cli
from ..tools import resize
from .. import segm_utils

DEBUG = False

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
                self.logger.log("Closing store state worker...")
                break
            elif not self.q.empty():
                posData, img1 = self.q.get()
                # self.logger.log('Storing state...')
                if posData.cca_df is not None:
                    cca_df = posData.cca_df.copy()
                else:
                    cca_df = None

                state = {
                    "image": img1.copy(),
                    "labels": posData.storedLab.copy(),
                    "editID_info": posData.editID_info.copy(),
                    "binnedIDs": posData.binnedIDs.copy(),
                    "ripIDs": posData.ripIDs.copy(),
                    "cca_df": cca_df,
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
        self.stopSaving = False
        self.isSaving = False
        self.isPaused = False
        self.dataQ = deque(maxlen=5)
        self.isAutoSaveON = False
        self.isAutoSaveAnnotON = True
        self.debug = False

    def pause(self):
        if self.debug:
            self.logger.log("Autosaving is idle.")
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
            self.logger.log("Enqueing posData autosave...")
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

    def cancelSaving(self): ...

    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log("Closing autosaving worker...")
                break
            elif not len(self.dataQ) == 0:
                if self.debug:
                    self.logger.log("Autosaving...")
                data = self.dataQ.pop()
                self.isSaving = True
                try:
                    self.saveData(data)
                except Exception as e:
                    error = traceback.format_exc()
                    print("*" * 40)
                    self.logger.log(error)
                    print("=" * 40)
                self.isSaving = False

                if len(self.dataQ) == 0:
                    self.sigDone.emit()
            else:
                self.pause()
        self.isFinished = True
        self.finished.emit(self)
        if self.debug:
            self.logger.log("Autosave finished signal emitted")

    def getLastTrackedFrame(self, posData):
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict["labels"]
            if lab is None:
                frame_i -= 1
                break
        if frame_i > 0:
            return frame_i
        else:
            return last_tracked_i

    def saveData(self, posData):
        if self.debug:
            self.logger.log("Started autosaving...")

        if not self.isAutoSaveON and not self.isAutoSaveAnnotON:
            return

        try:
            posData.setTempPaths()
        except Exception as e:
            self.logger.log(
                "[WARNING]: Cell-ACDC cannot create the recovery folder for "
                "the autosaving process. Autosaving will be turned off."
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
                segm_shape = (end_i + 1, *frame_shape)
                saved_segm_data = np.zeros(segm_shape, dtype=np.uint32)

        keys = []
        acdc_df_li = []

        for frame_i, data_dict in enumerate(posData.allData_li[: end_i + 1]):
            if self.stopSaving:
                break

            # Build saved_segm_data
            lab = data_dict["labels"]
            if lab is None:
                break

            if self.isAutoSaveON and saved_segm_data is not None:
                if posData.SizeT > 1:
                    saved_segm_data[frame_i] = lab
                else:
                    saved_segm_data = lab

            if self.isAutoSaveAnnotON:
                acdc_df = data_dict["acdc_df"]

                if acdc_df is None:
                    continue

            if not np.any(lab):
                continue

            if self.isAutoSaveAnnotON:
                acdc_df = load.pd_bool_and_float_to_int_to_str(
                    acdc_df, inplace=False, colsToCastInt=[]
                )

                acdc_df_li.append(acdc_df)
                key = (frame_i, posData.TimeIncrement * frame_i)
                keys.append(key)

            if self.stopSaving:
                break

        if not self.stopSaving:
            if self.isAutoSaveON:
                segm_data = np.squeeze(saved_segm_data)
                self._saveSegm(segm_npz_path, segm_data)

            if acdc_df_li:
                all_frames_acdc_df = pd.concat(
                    acdc_df_li, keys=keys, names=["frame_i", "time_seconds", "Cell_ID"]
                )
                self._save_acdc_df(all_frames_acdc_df, posData)

        if self.debug:
            self.logger.log(f"Autosaving done.")
            self.logger.log(f"Stopped autosaving {self.stopSaving}.")

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
        saved_acdc_df = pd.read_csv(
            saved_acdc_df_path, dtype=load.acdc_df_str_cols
        ).set_index(["frame_i", "Cell_ID"])

        recovery_acdc_df = recovery_acdc_df.reset_index(
            allow_duplicates=True
        ).set_index(["frame_i", "Cell_ID"])
        recovery_acdc_df = recovery_acdc_df.loc[
            :, ~recovery_acdc_df.columns.duplicated()
        ]
        try:
            # Try to insert into the recovery_acdc_df any column that was saved
            # but is not in the recovered df (e.g., metrics)
            df_left = recovery_acdc_df
            existing_cols = df_left.columns.intersection(saved_acdc_df.columns)
            df_right = saved_acdc_df.drop(columns=existing_cols)
            recovery_acdc_df = df_left.join(df_right, how="left")
        except Exception as error:
            self.logger.log(f"[WARNING]: {error}")

        # Check if last saved acdc_df is equal
        last_unsaved_csv_path = load.get_last_stored_unsaved_acdc_df_filepath(
            recovery_folderpath
        )
        if last_unsaved_csv_path is None:
            reference_acdc_df = saved_acdc_df
        else:
            try:
                reference_acdc_df = pd.read_csv(
                    last_unsaved_csv_path, dtype=load.acdc_df_str_cols
                ).set_index(["frame_i", "Cell_ID"])
            except Exception as e:
                self.logger.log(f"[WARNING]: {e}")
                reference_acdc_df = saved_acdc_df

        if utils.are_acdc_dfs_equal(recovery_acdc_df, reference_acdc_df):
            return

        load.store_unsaved_acdc_df(recovery_folderpath, recovery_acdc_df)


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
        self.loadSafeOverwriteNpz = False

    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def checkSelectedDataShape(self, posData, numPos):
        skipPos = False
        abort = False
        emitWarning = (
            not posData.segmFound and posData.SizeT > 1 and not self.mainWin.isNewFile
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

            self.logger.log(f"Loading {posData.relPath}...")

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
                        "data shape different from first Position loaded."
                    )
                    continue
                else:
                    data = "abort"
                    break

            self.logger.log(
                "Loaded paths:\n"
                f"Segmentation file name: {os.path.basename(posData.segm_npz_path)}\n"
                f"ACDC output file name {os.path.basename(posData.acdc_output_csv_path)}"
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
                signals=self.signals,
                mutex=self.mutex,
                waitCond=self.waitCond,
                additionalMetadata=self.firstPosData._additionalMetadataValues,
            )
            if hasattr(posData, "img_data_shape"):
                SizeY, SizeX = posData.img_data_shape[-2:]

            if posData.SizeZ > 1 and posData.img_data.ndim < 3:
                posData.SizeZ = 1
                posData.segmInfo_df = None
                try:
                    os.remove(posData.segmInfo_df_csv_path)
                except FileNotFoundError:
                    pass

            posData.setBlankSegmData(posData.SizeT, posData.SizeZ, SizeY, SizeX)
            if not self.firstPosData.onlyEditMetadata:
                skipPos, abort = self.checkSelectedDataShape(posData, numPos)
            else:
                skipPos, abort = False, False

            if skipPos:
                continue
            elif abort:
                data = "abort"
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
                        data = "abort"
                        break
                if self.loadUnsaved:
                    self.logger.log("Loading unsaved data...")
                    if os.path.exists(posData.segm_npz_temp_path):
                        segm_npz_path = posData.segm_npz_temp_path
                        posData.segm_data = np.load(segm_npz_path)["arr_0"]
                        segm_filename = os.path.basename(segm_npz_path)
                        posData.segm_npz_path = os.path.join(
                            posData.images_path, segm_filename
                        )

                    posData.loadMostRecentUnsavedAcdcDf()
                elif self.loadSafeOverwriteNpz:
                    self.logger.log("Loading safe npz overwrite...")
                    segm_safe_npz_path = posData.getSafeNpzOverwritePath()
                    posData.segm_data = np.load(segm_safe_npz_path)["arr_0"]

            # Allow single 2D/3D image
            if posData.SizeT == 1:
                posData.img_data = posData.img_data[np.newaxis]
                posData.segm_data = posData.segm_data[np.newaxis]
            if hasattr(posData, "img_data_shape"):
                img_shape = posData.img_data_shape
            img_shape = "Not Loaded"
            if hasattr(posData, "img_data_shape"):
                datasetShape = posData.img_data.shape
            else:
                datasetShape = "Not Loaded"
            if posData.segm_data is not None:
                posData.segmSizeT = len(posData.segm_data)
            SizeT = posData.SizeT
            SizeZ = posData.SizeZ
            self.logger.log(f"Full dataset shape = {img_shape}")
            self.logger.log(f"Loaded dataset shape = {datasetShape}")
            self.logger.log(f"Number of frames = {SizeT}")
            self.logger.log(f"Number of z-slices per frame = {SizeZ}")
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
                self.signals.progress.emit("Closing lazy loader...", "INFO")
                break
            elif self.wait:
                self.signals.progress.emit("Lazy loader paused.", "INFO")
                self.pause()
            else:
                self.signals.progress.emit("Lazy loader resumed.", "INFO")
                self.posData.loadChannelDataChunk(
                    self.current_idx, axis=self.axis, worker=self
                )
                self.sigLoadingFinished.emit()
                self.wait = True

        self.signals.finished.emit(None)


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
            "Migrating user profile data from "
            f'"{self.src_path}" to "{self.dst_path}"...'
        )
        acdc_folders = self.acdc_folders
        self.signals.initProgressBar.emit(2 * len(acdc_folders))
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
            self.progress.emit(f"Copying {src} to {dst}...")
            files_failed_move = copy_or_move_tree(
                src,
                dst,
                copy=False,
                sigInitPbar=self.signals.sigInitInnerPbar,
                sigUpdatePbar=self.signals.sigUpdateInnerPbar,
            )
            folders_to_remove.append(src)
            self.signals.progressBar.emit(1)

        for to_remove in folders_to_remove:
            try:
                self.progress.emit(f'Removing "{to_remove}"...')
                shutil.rmtree(to_remove)
            except Exception as err:
                self.progress.emit(
                    "--------------------------------------------------------\n"
                    f'[WARNING]: Removal of the folder "{to_remove}" failed. '
                    "Please remove manually.\n"
                    "--------------------------------------------------------"
                )
            finally:
                self.signals.progressBar.emit(1)

        # Update model's paths
        load.migrate_models_paths(self.dst_path)

        # Store user profile data folder path
        from . import user_profile_path_txt

        os.makedirs(os.path.dirname(user_profile_path_txt), exist_ok=True)
        with open(user_profile_path_txt, "w") as txt:
            txt.write(self.dst_path)

        self.finished.emit(self)


class MoveTempFilesWorker(QObject):
    def __init__(self, temp_files_to_move: Dict[os.PathLike, os.PathLike]):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.temp_files_to_move = temp_files_to_move

    @worker_exception_handler
    def run(self):
        for src, dst in self.temp_files_to_move.items():
            self.logger.log(f"Saving channel data to: {dst}...")
            shutil.move(src, dst)
            tempDir = os.path.dirname(src)
            shutil.rmtree(tempDir)
            self.signals.progressBar.emit(1)
        self.signals.finished.emit(self)


class saveDataWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    sigLog = Signal(str)
    progressBar = Signal(int, int, float)
    critical = Signal(object)
    addMetricsCritical = Signal(str, str)
    regionPropsCritical = Signal(str, str)
    criticalPermissionError = Signal(str)
    metricsPbarProgress = Signal(int, int)
    askZsliceAbsent = Signal(str, object)
    customMetricsCritical = Signal(str, str)
    sigCombinedMetricsMissingColumn = Signal(str, str)
    sigDebug = Signal(object)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.saveWin = mainWin.saveWin
        self.mutex = mainWin.mutex
        self.waitCond = mainWin.waitCond
        self.customMetricsErrors = {}
        self.addMetricsErrors = {}
        self.regionPropsErrors = {}
        self.abort = False

    def checkAbort(self):
        if self.saveWin.aborted:
            self.finished.emit()
            return True
        return False

    def saveManualBackgroundData(self, posData, frame_i):
        data_dict = posData.allData_li[frame_i]
        if "manualBackgroundLab" not in data_dict:
            return

        manualBackgrData = data_dict["manualBackgroundLab"]
        posData.saveManualBackgroundData(manualBackgrData)

    def emitSigPermissionErrorAndSave(
        self, all_frames_acdc_df, acdc_output_csv_path, custom_annot_columns
    ):
        err_msg = (
            "The below file is open in another app "
            "(Excel maybe?).\n\n"
            f"{acdc_output_csv_path}\n\n"
            'Close file and then press "Ok".'
        )
        self.mutex.lock()
        self.criticalPermissionError.emit(err_msg)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

        # Save segmentation metadata
        load.save_acdc_df_file(
            all_frames_acdc_df,
            acdc_output_csv_path,
            custom_annot_columns=custom_annot_columns,
            last_cca_frame_i=self.mainWin.save_cca_until_frame_i,
        )

    def _emitSigDebug(self, stuff_to_debug):
        self.mutex.lock()
        self.sigDebug.emit(stuff_to_debug)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitUpdateProgressBar(self):
        t = time.perf_counter()
        exec_time = t - self.time_last_pbar_update
        self.progressBar.emit(1, -1, exec_time)
        self.time_last_pbar_update = t

    def saveAcdcDf(self, posData: load.loadData, end_i):
        acdc_dfs_li = []
        keys = []
        self.progress.emit(f"Saving annotations for {posData.relPath}...")
        for frame_i, data_dict in enumerate(posData.allData_li[: end_i + 1]):
            if self.saveWin.aborted:
                self.finished.emit()
                return

            # Build saved_segm_data
            lab = data_dict["labels"]
            if lab is None:
                break

            acdc_df = posData.allData_li[frame_i]["acdc_df"]
            if acdc_df is None:
                continue

            acdc_dfs_li.append(acdc_df)
            keys.append((frame_i, posData.TimeIncrement * frame_i))

        if not acdc_dfs_li:
            return

        self.mainWin._measurements_kernel._concat_and_save_acdc_df(
            acdc_dfs_li,
            keys,
            posData,
            self.mainWin.save_metrics,
            saveDataWorker=self,
            last_cca_frame_i=self.mainWin.save_cca_until_frame_i,
        )

    def saveSegmData(self, posData, end_i, saved_segm_data):
        self.progress.emit(f"Saving segmentation data for {posData.relPath}...")

        # extend saved_segm_data if needed
        if posData.SizeT > 1:
            missing_frames_number = end_i + 1 - len(saved_segm_data)
            if missing_frames_number > 0:
                saved_segm_data = np.concatenate(
                    (
                        saved_segm_data,
                        np.zeros(
                            (missing_frames_number, *saved_segm_data.shape[1:]),
                            dtype=saved_segm_data.dtype,
                        ),
                    ),
                )

        for frame_i, data_dict in enumerate(posData.allData_li[: end_i + 1]):
            if self.saveWin.aborted:
                self.finished.emit()
                return

            # Build saved_segm_data
            lab = data_dict["labels"]
            if lab is None:
                break

            posData.lab = lab

            if posData.SizeT > 1:
                saved_segm_data[frame_i] = lab
            else:
                saved_segm_data = lab
                if "manualBackgroundLab" in data_dict:
                    manualBackgrData = data_dict["manualBackgroundLab"]
                    posData.saveManualBackgroundData(manualBackgrData)

        # Save segmentation file
        io.savez_compressed(posData.segm_npz_path, np.squeeze(saved_segm_data))
        posData.segm_data = saved_segm_data
        # Allow single 2D/3D image
        if posData.SizeT == 1:
            posData.segm_data = posData.segm_data[np.newaxis]

        try:
            os.remove(posData.segm_npz_temp_path)
        except Exception as e:
            pass

    @worker_exception_handler
    def run(self):
        posToSave = self.mainWin.posToSave
        if posToSave is None:
            numPosToSave = 1
        else:
            numPosToSave = len(posToSave)
        save_metrics = self.mainWin.save_metrics
        if self.isQuickSave:
            save_metrics = False
        self.time_last_pbar_update = time.perf_counter()
        mode = self.mode
        for p, posData in enumerate(self.mainWin.data):
            if self.saveWin.aborted:
                self.finished.emit()
                return

            if posToSave is not None:
                if posData.pos_foldername not in posToSave:
                    self.progress.emit(f"Skipping {posData.relPath}")
                    continue

            last_tracked_i_path = posData.last_tracked_i_path
            end_i = self.mainWin.save_until_frame_i
            self.saveSegmData(posData, end_i, posData.segm_data)

            posData.saveCustomAnnotationParams()
            current_frame_i = posData.frame_i

            posData.saveTrackedLostCentroids()

            if not self.mainWin.isSnapshot:
                last_tracked_i = self.mainWin.last_tracked_i
                if last_tracked_i is None:
                    self.mainWin.saveWin.aborted = True
                    self.finished.emit()
                    return
            elif self.mainWin.isSnapshot:
                last_tracked_i = 0

            if p == 0:
                self.progressBar.emit(0, numPosToSave * (last_tracked_i + 1), 0)

            acdc_output_csv_path = posData.acdc_output_csv_path
            delROIs_info_path = posData.delROIs_info_path

            # Add segmented channel data for calc metrics if requested
            add_user_channel_data = True
            for chName in self.mainWin._measurements_kernel.chNamesToSkip:
                skipUserChannel = posData.filename.endswith(
                    chName
                ) or posData.filename.endswith(f"{chName}_aligned")
                if skipUserChannel:
                    add_user_channel_data = False

            if add_user_channel_data and not self.isQuickSave:
                posData.fluo_data_dict[posData.filename] = posData.img_data

            if not self.isQuickSave:
                posData.fluo_bkgrData_dict[posData.filename] = posData.bkgrData

            posData.setLoadedChannelNames()

            if not self.isQuickSave:
                self.mainWin.initMetricsToSave(posData)
                self.mainWin._measurements_kernel.run(
                    posData=posData,
                    stop_frame_n=end_i + 1,
                    saveDataWorker=self,
                    save_metrics=self.mainWin.save_metrics,
                    last_cca_frame_i=self.mainWin.save_cca_until_frame_i,
                )
            else:
                self.saveAcdcDf(posData, end_i)

            self.progress.emit(f"Saving {posData.relPath}")

            if not self.do_not_save_og_whitelist:
                og_save_path = os.path.join(
                    posData.images_path, self.append_name_og_whitelist
                )
                posData.whitelist.saveOGLabs(og_save_path)

            if posData.whitelist:
                whitelistIDs_path = posData.segm_npz_path.replace(
                    ".npz", "_whitelistIDs.json"
                )
                new_centroids_path = posData.segm_npz_path.replace(
                    ".npz", "_new_centroids.json"
                )
                posData.whitelist.save(
                    whitelistIDs_path, new_centroids_path=new_centroids_path
                )

            if posData.segmInfo_df is not None:
                try:
                    posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
                except PermissionError:
                    err_msg = (
                        "The below file is open in another app "
                        "(Excel maybe?).\n\n"
                        f"{posData.segmInfo_df_csv_path}\n\n"
                        'Close file and then press "Ok".'
                    )
                    self.mutex.lock()
                    self.criticalPermissionError.emit(err_msg)
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

            posData.fluo_data_dict.pop(posData.filename, None)

            if not self.isQuickSave:
                posData.fluo_bkgrData_dict.pop(posData.filename)

            if posData.SizeT > 1:
                self.progress.emit("Almost done...")
                self.progressBar.emit(0, 0, 0)

            if self.isQuickSave:
                # Go back to current frame
                posData.frame_i = current_frame_i
                self.mainWin.get_data()
                continue

            with open(last_tracked_i_path, "w+") as txt:
                txt.write(str(end_i))

            # Save combined metrics equations
            posData.saveCombineMetrics()
            self.mainWin.pointsLayerDataToDf(posData)
            posData.saveClickEntryPointsDfs()

            posData.last_tracked_i = last_tracked_i

            # Go back to current frame
            posData.frame_i = current_frame_i
            self.mainWin.get_data()

            if mode == "Segmentation and Tracking" or mode == "Viewer":
                self.progress.emit(f"Saved data until frame number {end_i + 1}")
            elif mode == "Cell cycle analysis":
                self.progress.emit(
                    "Saved cell cycle annotations until frame "
                    f"number {self.mainWin.last_cca_frame_i + 1}"
                )
            # self.progressBar.emit(1)
        if self.mainWin.isSnapshot:
            self.progress.emit(f"Saved all {p + 1} Positions!")

        self.finished.emit()


class relabelSequentialWorker(QObject):
    finished = Signal()
    critical = Signal(object)
    progress = Signal(str)
    sigRemoveItemsGUI = Signal(int)
    debug = Signal(object)

    def __init__(self, mainWin, posFoldernames):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.data = mainWin.data
        self.posFoldernames = posFoldernames
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()

    def progressNewIDs(self, oldIDs, newIDs):
        li = list(zip(oldIDs, newIDs))
        s = "\n".join([str(pair).replace(",", " -->") for pair in li])
        s = f"IDs relabelled as follows:\n{s}"
        self.progress.emit(s)

    @worker_exception_handler
    def run(self):
        self.mutex.lock()

        self.progress.emit("Relabelling process started...")
        mainWin = self.mainWin

        current_pos_i = mainWin.pos_i

        for p, posData in enumerate(self.data):
            if posData.pos_foldername not in self.posFoldernames:
                continue

            mainWin.pos_i = p
            current_lab = mainWin.get_2Dlab(posData.lab).copy()
            current_frame_i = posData.frame_i
            segm_data = []
            for frame_i, data_dict in enumerate(posData.allData_li):
                lab = data_dict["labels"]
                if lab is None:
                    break
                segm_data.append(lab)
                # if frame_i == current_frame_i:
                #     break

            if not segm_data:
                segm_data = np.array([current_lab])

            segm_data = np.array(segm_data)
            segm_data, oldIDs, newIDs = core.relabel_sequential(
                segm_data, is_timelapse=posData.SizeT > 1
            )
            self.progressNewIDs(oldIDs, newIDs)
            self.sigRemoveItemsGUI.emit(np.max(segm_data))

            self.progress.emit(
                "Updating stored data and cell cycle annotations (if present)..."
            )

            mainWin.updateAnnotatedIDs(oldIDs, newIDs, logger=self.progress.emit)
            mainWin.store_data(mainThread=False)

            for frame_i, lab in enumerate(segm_data):
                posData.frame_i = frame_i
                posData.lab = lab
                mainWin.get_cca_df()
                if posData.cca_df is not None:
                    mainWin.update_cca_df_relabelling(posData, oldIDs, newIDs)
                mainWin.update_rp(draw=False)
                mainWin.store_data(mainThread=False)

        # Go back to current frame
        mainWin.pos_i = current_pos_i
        posData = self.data[mainWin.pos_i]
        posData.frame_i = current_frame_i
        mainWin.get_data()

        self.mutex.unlock()
        self.finished.emit()

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    signals,
    workerLogger,
    worker_exception_handler,
)

