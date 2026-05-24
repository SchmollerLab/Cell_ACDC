"""Background Qt workers: util."""

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

from .. import load, myutils, core, prompts, printl, config, segm_re_pattern, io
from .. import transformation, measurements, cca_functions
from ..path import copy_or_move_tree
from .. import features, plot
from .. import core
from .. import cca_df_colnames, lineage_tree_cols, default_annot_df
from .. import cca_df_colnames_with_tree
from .. import cli
from ..utils import resize
from .. import segm_utils

DEBUG = False

from ._base import (
    BaseWorkerUtil,
)

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
                exp_path, pos_foldernames, "imagej_rois", ext=".zip"
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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                images_path = os.path.join(exp_path, pos, "Images")
                endFilenameRoi = self.mainWin.endFilenameWithText
                ls = myutils.listdir(images_path)
                rois_filepaths = [
                    os.path.join(images_path, f)
                    for f in ls
                    if f.endswith(f"{endFilenameRoi}.zip")
                ]

                if not rois_filepaths:
                    self.logger.log(
                        "[WARNING]: The following Position folder does not "
                        f"contain any file ending with {endFilenameRoi}. "
                        f'Skipping it. "{os.path.join(exp_path, pos)}")'
                    )
                    continue

                rois_filepath = rois_filepaths[0]

                if self.askRoiPreferences:
                    is_multi_pos = len(pos_foldernames) > 1
                    self.logger.log("Loading image data to get image shape...")
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
                    self.IDsToRoisMapper = {i + i: roi for roi in enumerate(rois)}
                else:
                    # Use same ID of previous position
                    rois = roifile.roiread(rois_filepath)
                    IDsToRoisMapper = {i + i: roi for i, roi in enumerate(rois)}
                    self.IDsToRoisMapper = {
                        ID: IDsToRoisMapper[ID] for ID in self.IDsToRoisMapper.keys()
                    }

                self.logger.log("Generating segm mask from ROIs...")
                segm_data = myutils.from_imagej_rois_to_segm_data(
                    TZYX_shape,
                    self.IDsToRoisMapper,
                    self.rescaleRoisSizes,
                    self.repeatRoisZslicesRange,
                )

                segm_filepath = rois_filepath.replace("imagej_rois", "segm").replace(
                    ".zip", ".npz"
                )
                self.logger.log(f'Saving segm mask to "{segm_filepath}"...')
                io.savez_compressed(segm_filepath, segm_data)

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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                images_path = os.path.join(exp_path, pos, "Images")
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)

                files_path = [
                    os.path.join(images_path, f)
                    for f in ls
                    if f.endswith(f"{endFilenameSegm}.npz")
                ]

                if not files_path:
                    self.logger.log(
                        "[WARNING]: The following Position folder does not "
                        f"contain any file ending with {endFilenameSegm}. "
                        f'Skipping it. "{os.path.join(exp_path, pos)}")'
                    )
                    continue

                file_path = files_path[0]

                posData = load.loadData(file_path, "")

                self.signals.sigUpdatePbarDesc.emit(f"Processing {posData.pos_path}")

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm,
                )

                if posData.SizeT > 1:
                    rois = []
                    max_ID = posData.segm_data.max()
                    for t, lab in enumerate(posData.segm_data):
                        rois_t = myutils.from_lab_to_imagej_rois(
                            lab, ImagejRoi, t=t, SizeT=posData.SizeT, max_ID=max_ID
                        )
                        rois.extend(rois_t)
                else:
                    rois = myutils.from_lab_to_imagej_rois(posData.segm_data, ImagejRoi)

                roi_filepath = posData.segm_npz_path.replace(".npz", ".zip")
                roi_filepath = roi_filepath.replace("_segm", "_imagej_rois")

                try:
                    os.remove(roi_filepath)
                except Exception as e:
                    pass

                roiwrite(roi_filepath, rois)

        self.signals.finished.emit(self)


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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                images_path = os.path.join(exp_path, pos, "Images")
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f)
                    for f in ls
                    if f.endswith(f"{endFilenameSegm}.npz")
                ][0]

                posData = load.loadData(file_path, "")

                self.signals.sigUpdatePbarDesc.emit(f"Processing {posData.pos_path}")

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm,
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
                df_filepath = posData.segm_npz_path.replace(".npz", ".csv")
                df_filepath = df_filepath.replace("_segm", "_objects_coordinates")

                keys = list(range(len(posData.segm_data)))
                df = pd.concat(dfs, keys=keys, names=["frame_i"])

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

            self.mainWin.infoText = f"Select <b>2D segmentation file to stack</b>"
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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                images_path = os.path.join(exp_path, pos, "Images")
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f)
                    for f in ls
                    if f.endswith(f"{endFilenameSegm}.npz")
                ][0]

                posData = load.loadData(file_path, "")

                self.signals.sigUpdatePbarDesc.emit(f"Processing {posData.pos_path}")

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm,
                )
                if posData.segm_data.ndim == 2:
                    posData.segm_data = posData.segm_data[np.newaxis]

                self.logger.log("Stacking 2D into 3D objects...")

                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                T, Y, X = posData.segm_data.shape
                newShape = (T, self.SizeZ, Y, X)
                segmData2D = np.zeros(newShape, dtype=np.uint32)
                for frame_i, lab in enumerate(posData.segm_data):
                    stacked_lab = core.stack_2Dlab_to_3D(lab, self.SizeZ)
                    segmData2D[frame_i] = stacked_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log("Saving stacked 3D segmentation file...")
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f"{segmFilename}_{appendedName}.npz"
                segmData2D = np.squeeze(segmData2D)
                io.savez_compressed(newSegmFilepath, segmData2D)

                self.signals.progressBar.emit(1)

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
        categories = ["X coord. column", "Y coord. column"]
        optionalCategories = []

        images_path = os.path.join(exp_path, pos_foldernames[0], "Images")
        metadata_df = load.load_metadata_df(images_path)
        SizeT = float(metadata_df.at["SizeT", "values"])
        SizeZ = float(metadata_df.at["SizeZ", "values"])

        segmData = load.load_segm_file(images_path, end_name_segm_file=endFilenameSegm)

        if segmData.ndim == 4:
            categories.append("Z coord. column")
            categories.append("Frame index column")
        elif segmData.ndim == 3:
            if SizeZ > 1 and SizeT == 1:
                # 3D z-stack data
                categories.append("Z coord. column")
            else:
                optionalCategories.append("Z coord. column")

            if SizeT > 1:
                # 3D time-lapse
                categories.append("Frame index column")
            else:
                optionalCategories.append("Frame index column")
        else:
            optionalCategories.append("Z coord. column")
            optionalCategories.append("Frame index column")

        if len(pos_foldernames) > 1:
            categories.append("Position_n")
        else:
            optionalCategories.append("Position_n")

        return columns, categories, optionalCategories

    def getDfCoords(
        self, df_coords, selectedColumnsPerCategory, pos_foldername, frame_i
    ):
        pos_col = selectedColumnsPerCategory.get("Position_n", "None")
        frame_i_col = selectedColumnsPerCategory.get("Frame index column", "None")
        x_col = selectedColumnsPerCategory["X coord. column"]
        y_col = selectedColumnsPerCategory["Y coord. column"]
        if pos_col != "None":
            df_coords = df_coords[df_coords[pos_col] == pos_foldername]
        if frame_i_col != "None":
            df_coords = df_coords[df_coords[frame_i_col] == frame_i]

        xy_cols = [x_col, y_col]

        df_out = pd.DataFrame(
            index=df_coords.index, data=df_coords[xy_cols].values, columns=["x", "y"]
        )
        z_col = selectedColumnsPerCategory.get("Z coord. column", "None")
        if z_col != "None":
            df_out["z"] = df_coords[z_col]

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

            self.mainWin.infoText = f"Select <b>segmentation file to filter</b>"
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            endFilenameSegm = self.mainWin.endFilenameSegm

            self.logger.log("Asking to select the CSV table file...")

            abort = self.emitSelectFile(
                exp_path,
                "Select CSV table file with coordinates to filter",
                "CSV (*.csv)",
            )
            if abort:
                self.sigAborted.emit()
                return

            self.logger.log(f"Loading table file `{self.mainWin.selectedFilepath}`..")
            df_coords = pd.read_csv(self.mainWin.selectedFilepath)

            columns, categories, optionalCategories = self.getColumnsCategories(
                df_coords, exp_path, pos_foldernames, endFilenameSegm
            )

            abort = self.emitSetColumnsNames(columns, categories, optionalCategories)
            if abort:
                self.sigAborted.emit()
                return

            selectedColumnsPerCategory = self.mainWin.selectedColumnsPerCategory

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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                images_path = os.path.join(exp_path, pos, "Images")
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f)
                    for f in ls
                    if f.endswith(f"{endFilenameSegm}.npz")
                ][0]

                posData = load.loadData(file_path, "")

                self.signals.sigUpdatePbarDesc.emit(f"Processing {posData.pos_path}")

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm,
                )
                if posData.SizeT == 1:
                    posData.segm_data = posData.segm_data[np.newaxis]

                self.logger.log("Filtering objects...")

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

                self.logger.log("Saving filtered segmentation file...")
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f"{segmFilename}_{appendedName}.npz"
                filteredSegmData = np.squeeze(filteredSegmData)
                io.savez_compressed(newSegmFilepath, filteredSegmData)

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
            fn = f"shot_{i:03}.jpg"
            grab_path = os.path.join(self.folder_path, fn)
            screen = self.screenRecorderWin.screen()
            screenshot = screen.grabWindow(self.screenRecorderWin.winId())
            screenshot.save(grab_path, "jpg")
            print(grab_path)
            time.sleep(0.2)

        self.finished.emit()


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
        self.progress.emit("Filtering image...")
        filtered_data = self.filter_func(self.input_data)
        self.finished.emit(filtered_data)


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
        if path is None:
            return

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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )
                images_path = os.path.join(exp_path, pos, "Images")

                rf = self.resizeFactor
                text_to_append = self.textToAppend
                images_path_out = self.validateOutputPath(self.expFolderpathOut)
                if images_path_out is None:
                    images_path_out = images_path
                resize.run(
                    images_path,
                    rf,
                    text_to_append=text_to_append,
                    images_path_out=images_path_out,
                )

        self.signals.finished.emit(self)

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    worker_exception_handler,
)

