"""Background Qt workers: alignment."""

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

from ._base import (
    BaseWorkerUtil,
)

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
            tif.replace(".tif", "_aligned.npz") for tif in self.posData.tif_paths
        ]
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or self.posData.loaded_shifts is None

            filename_tif = os.path.basename(tif)
            user_ch_filename = f"{self.posData.basename}{self.user_ch_name}.tif"

            if not doAlign:
                _npz = f"{os.path.splitext(tif)[0]}_aligned.npz"
                if os.path.exists(_npz):
                    self.posData.all_npz_paths[i] = _npz
                continue

            if filename_tif != user_ch_filename:
                continue

            if not self.align:
                continue

            # Align based on user_ch_name
            aligned = True
            self.logger.log(f"Aligning: {tif}")

            tif_data = load.imread(tif)
            numFramesWith0s = self.dataPrepWin.detectTifAlignment(
                tif_data, self.posData
            )
            if self.align:
                self.emitWarnTifAligned(numFramesWith0s, tif, self.posData)
                if self.doAbort:
                    return

            # Alignment routine
            if self.posData.SizeZ > 1:
                align_func = core.align_frames_3D
                df = self.posData.segmInfo_df.loc[self.posData.filename]
                zz = df["z_slice_used_dataPrep"].to_list()
                if not self.posData.filename.endswith("aligned") and self.align:
                    # Add aligned channel to segmInfo
                    df_aligned = self.posData.segmInfo_df.rename(
                        index={
                            self.posData.filename: f"{self.posData.filename}_aligned"
                        }
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
                    tif_data,
                    slices=zz,
                    user_shifts=self.posData.loaded_shifts,
                    sigPyqt=self.signals.progressBar,
                )
                self.posData.loaded_shifts = shifts
            else:
                aligned_frames = tif_data

            if self.align:
                self.signals.initProgressBar.emit(0)
                _npz = f"{os.path.splitext(tif)[0]}_aligned.npz"
                self.logger.log(f"Storing temporary file: {_npz}")
                temp_npz = self.dataPrepWin.getTempfilePath(_npz)
                io.savez_compressed(temp_npz, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_npz, _npz)
                np.save(self.posData.align_shifts_path, self.posData.loaded_shifts)
                self.posData.all_npz_paths[i] = _npz

                self.logger.log(f"Storing temporary file: {tif}")
                temp_tif = self.dataPrepWin.getTempfilePath(tif)
                utils.to_tiff(temp_tif, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_tif, tif)
                self.posData.img_data = load.imread(temp_tif)

        _zip = zip(self.posData.tif_paths, self.posData.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or aligned

            if not doAlign:
                continue

            if tif.endswith(f"{self.user_ch_name}.tif"):
                continue

            if not self.align:
                continue

            # Align the other channels
            if self.posData.loaded_shifts is None:
                break

            if self.align:
                self.logger.log(f"Aligning: {tif}")
                tif_data = load.imread(tif)

            # Alignment routine
            if self.posData.SizeZ > 1:
                align_func = core.align_frames_3D
                df = self.posData.segmInfo_df.loc[self.posData.filename]
                zz = df["z_slice_used_dataPrep"].to_list()
            else:
                align_func = core.align_frames_2D
                zz = None
            if self.align:
                self.signals.initProgressBar.emit(len(tif_data))
                aligned_frames, shifts = align_func(
                    tif_data,
                    slices=zz,
                    user_shifts=self.posData.loaded_shifts,
                    sigPyqt=self.signals.progressBar,
                )
            else:
                aligned_frames = tif_data

            _npz = f"{os.path.splitext(tif)[0]}_aligned.npz"

            if self.align:
                self.signals.initProgressBar.emit(0)
                self.logger.log(f"Saving: {_npz}")
                temp_npz = self.dataPrepWin.getTempfilePath(_npz)
                io.savez_compressed(temp_npz, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_npz, _npz)
                self.posData.all_npz_paths[i] = _npz

                self.logger.log(f"Saving: {tif}")
                temp_tif = self.dataPrepWin.getTempfilePath(tif)
                utils.to_tiff(temp_tif, aligned_frames)
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
        self.logger.log(f"Aligning: {self.posData.segm_npz_path}")
        self.posData.segm_data, shifts = core.align_frames_2D(
            self.posData.segm_data, slices=None, user_shifts=self.posData.loaded_shifts
        )
        self.logger.log(f"Saving: {self.posData.segm_npz_path}")
        temp_npz = self.dataPrepWin.getTempfilePath(self.posData.segm_npz_path)
        io.savez_compressed(temp_npz, self.posData.segm_data)
        self.dataPrepWin.storeTempFileMove(temp_npz, self.posData.segm_npz_path)

    @worker_exception_handler
    def run(self):
        self._align_data()
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
                images_path = os.path.join(exp_path, pos, "Images")
                ls = utils.listdir(images_path)
                for file in ls:
                    if file.endswith("align_shift.npy"):
                        shiftsFound = True
                        basename, chNames = utils.getBasenameAndChNames(
                            images_path, useExt=(".tif", ".h5")
                        )
                        break
                if shiftsFound:
                    break

            savedShiftsHow = None
            if shiftsFound:
                basename_ch0 = f"{basename}{chNames[0]}_"
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

                self.logger.log("*" * 40)
                self.logger.log(
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, "Images")
                basename, chNames = utils.getBasenameAndChNames(
                    images_path, useExt=(".tif", ".h5")
                )

                self.signals.sigUpdatePbarDesc.emit(f"Loading {pos_path}...")

                if p == 0:
                    self.logger.log(f"Asking to select reference channel...")
                    abort = self.emitAskSelectChannel(chNames)
                    if abort:
                        self.sigAborted.emit()
                        return
                    chName = self.chName

                file_path = utils.getChannelFilePath(images_path, chName)

                # Load data
                posData = load.loadData(file_path, chName)
                posData.getBasenameAndChNames(useExt=(".tif", ".h5"))
                posData.buildPaths()
                posData.loadImgData()

                posData.loadOtherFiles(
                    load_segm_data=False, load_shifts=True, loadSegmInfo=True
                )

                if posData.img_data.ndim == 4:
                    align_func = core.align_frames_3D
                    if posData.segmInfo_df is None:
                        raise FileNotFoundError(
                            "To align 4D data you need to select which z-slice "
                            "you want to use for alignment. Please run the module "
                            "`1. Launch data prep module...` before aligning the "
                            "frames. (z-slice info MISSING from position "
                            f'"{posData.relPath}")'
                        )
                    df = posData.segmInfo_df.loc[posData.filename]
                    zz = df["z_slice_used_dataPrep"].to_list()
                elif posData.img_data.ndim == 3:
                    align_func = core.align_frames_2D
                    zz = None

                useSavedShifts = (
                    savedShiftsHow == "use_saved_shifts"
                    and posData.loaded_shifts is not None
                )
                if useSavedShifts:
                    user_shifts = posData.loaded_shifts
                else:
                    user_shifts = None

                if savedShiftsHow == "rever_alignment":
                    if posData.loaded_shifts is None:
                        self.logger.log(
                            f'WARNING: Cannot revert alignment in "{posData.relPath}" '
                            "since it is missing previously computed shifts. "
                            "Skipping this positon."
                        )
                        continue

                    # Revert alignment and save selected channel
                    for chName in chNames:
                        self.logger.log(f'Reverting alignment on "{chName}"...')
                        if chName == posData.user_ch_name:
                            data = posData.img_data
                        else:
                            file_path = utils.getChannelFilePath(images_path, chName)
                            data = load.load_image_file(file_path)

                        self.signals.sigInitInnerPbar.emit(len(data) - 1)
                        revertedData = core.revert_alignment(
                            posData.loaded_shifts,
                            data,
                            sigPyqt=self.signals.sigUpdateInnerPbar,
                        )
                        self.logger.log(f'Saving "{chName}"...')
                        self.signals.sigInitInnerPbar.emit(0)
                        self.saveAlignedData(
                            revertedData,
                            images_path,
                            posData.basename,
                            chName,
                            self.revertedAlignEndname,
                            ext=posData.ext,
                        )
                        del revertedData, data
                else:
                    for chName in chNames:
                        self.logger.log(f'Aligning "{chName}"...')
                        if chName == posData.user_ch_name:
                            data = posData.img_data
                        else:
                            file_path = utils.getChannelFilePath(images_path, chName)
                            data = load.load_image_file(file_path)
                        self.signals.sigInitInnerPbar.emit(len(data) - 1)

                        alignedImgData, shifts = align_func(
                            data,
                            slices=zz,
                            user_shifts=user_shifts,
                            sigPyqt=self.signals.sigUpdateInnerPbar,
                        )
                        self.logger.log(f'Saving "{chName}"...')
                        np.save(posData.align_shifts_path, shifts)

                        self.signals.sigInitInnerPbar.emit(0)
                        self.saveAlignedData(
                            alignedImgData,
                            images_path,
                            posData.basename,
                            chName,
                            "",
                            ext=posData.non_aligned_ext,
                        )
                        self.saveAlignedData(
                            alignedImgData,
                            images_path,
                            posData.basename,
                            chName,
                            "aligned",
                            ext=".npz",
                        )
                        del alignedImgData, data

        self.signals.finished.emit(self)

    def saveAlignedData(self, data, imagesPath, basename, chName, endname, ext=".tif"):
        if endname:
            newFilename = f"{basename}{chName}_{endname}{ext}"
        else:
            newFilename = f"{basename}{chName}{ext}"

        filePath = os.path.join(imagesPath, newFilename)

        if ext == ".tif":
            SizeT = data.shape[0]
            SizeZ = 1
            if data.ndim == 4:
                SizeZ = data.shape[1]
            utils.to_tiff(filePath, data)
        elif ext == ".npz":
            io.savez_compressed(filePath, data)
        elif ext == ".h5":
            load.save_to_h5(filePath, data)

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    signals,
    workerLogger,
    worker_exception_handler,
)

