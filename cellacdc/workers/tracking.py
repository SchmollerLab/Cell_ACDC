"""Background Qt workers: tracking."""

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
        lab = frameData["labels"]
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
            if hasattr(self.signals, "innerPbar_available"):
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
        self.progress.emit("Tracking process started (more details in the terminal)...")

        trackerInputImage = None
        self.track_params["signals"] = self.signals
        if "image" in self.track_params:
            trackerInputImage = self.track_params.pop("image")
            start_frame_i = self.mainWin.start_n - 1
            stop_frame_n = self.mainWin.stop_n

            trackerInputImage = trackerInputImage[start_frame_i:stop_frame_n]

        tracked_video = core.tracker_track(
            self.video_to_track,
            self.tracker,
            self.track_params,
            intensity_img=trackerInputImage,
            logger_func=self.progress.emit,
        )

        self._setProgressBarIndefiniteWait()

        # self.debug.emit((tracked_video, self))
        # self.waitCond.wait(self.mutex)

        self.progress.emit("Re-tracking first frame to ensure continuity...")
        # Relabel first frame objects back to IDs they had before tracking
        # (to ensure continuity with past untracked frames)
        tracked_video = self._relabel_first_frame_labels(tracked_video)

        print("")
        self.progress.emit("Generating annotations...")
        acdc_df = self.posData.fromTrackerToAcdcDf(
            self.tracker, tracked_video, start_frame_i=self.mainWin.start_n - 1
        )
        # Store new tracked video
        current_frame_i = self.posData.frame_i
        self.trackingOnNeverVisitedFrames = False
        print("")
        self.progress.emit("Storing tracked video...")
        pbar = tqdm(total=len(tracked_video), ncols=100)
        for rel_frame_i, lab in enumerate(tracked_video):
            frame_i = rel_frame_i + self.mainWin.start_n - 1

            if acdc_df is not None:
                cca_cols = acdc_df.columns.intersection(cca_df_colnames_with_tree)
                # Store cca_df if it is an output of the tracker
                cca_df = acdc_df.loc[frame_i][cca_cols]
                self.mainWin.store_cca_df(
                    frame_i=frame_i, cca_df=cca_df, mainThread=False, autosave=False
                )

            if self.posData.allData_li[frame_i]["labels"] is None:
                # repeating tracking on a never visited frame
                # --> modify only raw data and ask later what to do
                self.posData.segm_data[frame_i] = lab
                self.trackingOnNeverVisitedFrames = True
            else:
                # Get the rest of the stored metadata based on the new lab
                self.posData.allData_li[frame_i]["labels"] = lab
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


class TrackSubCellObjectsWorker(BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigCriticalNotEnoughSegmFiles = Signal(str)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)
        if mainWin.trackingMode.find("Delete both") != -1:
            self.trackingMode = "delete_both"
        elif mainWin.trackingMode.find("Delete sub-cellular") != -1:
            self.trackingMode = "delete_sub"
        elif mainWin.trackingMode.find("Delete cells") != -1:
            self.trackingMode = "delete_cells"
        elif mainWin.trackingMode.find("Only track") != -1:
            self.trackingMode = "only_track"

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

            red_text = html_utils.span("OF THE CELLs")
            self.mainWin.infoText = f"Select <b>segmentation file {red_text}</b>"
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

            red_text = html_utils.span("OF THE SUB-CELLULAR OBJECTS")
            self.mainWin.infoText = f"Select <b>segmentation file {red_text}</b>"
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

                # Load cells segmentation file
                segmDataCells, segmCellsPath = load.load_segm_file(
                    images_path,
                    end_name_segm_file=self.cellsSegmEndFilename,
                    return_path=True,
                )
                acdc_df_cells_endname = self.cellsSegmEndFilename.replace(
                    "_segm", "_acdc_output"
                )
                acdc_df_cell, acdc_df_cells_path = load.load_acdc_df_file(
                    images_path,
                    end_name_acdc_df_file=acdc_df_cells_endname,
                    return_path=True,
                )

                if posData.SizeT > 1:
                    numFrames = min((len(segmDataCells), len(posData.segm_data)))
                    segmDataCells = segmDataCells[:numFrames]
                    posData.segm_data = posData.segm_data[:numFrames]
                else:
                    numFrames = 1

                self.signals.sigInitInnerPbar.emit(numFrames * 2)

                self.logger.log("Tracking sub-cellular objects...")
                tracked = core.track_sub_cell_objects(
                    segmDataCells,
                    posData.segm_data,
                    self.IoAthresh,
                    how=self.trackingMode,
                    SizeT=numFrames,
                    sigProgress=self.signals.sigUpdateInnerPbar,
                    relabel_sub_obj_lab=self.relabelSubObjLab,
                )
                (
                    trackedSubSegmData,
                    trackedCellsSegmData,
                    numSubObjPerCell,
                    replacedSubIds,
                ) = tracked

                self.logger.log("Saving tracked segmentation files...")
                subSegmFilename, ext = os.path.splitext(posData.segm_npz_path)
                trackedSubPath = f"{subSegmFilename}_{appendedName}.npz"
                io.savez_compressed(trackedSubPath, trackedSubSegmData)
                posData.saveIsSegm3Dmetadata(trackedSubPath)

                if trackedCellsSegmData is not None:
                    cellsSegmFilename, ext = os.path.splitext(segmCellsPath)
                    trackedCellsPath = f"{cellsSegmFilename}_{appendedName}.npz"
                    io.savez_compressed(trackedCellsPath, trackedCellsSegmData)

                if self.createThirdSegm:
                    self.logger.log(
                        f"Generating segmentation from "
                        f'"{self.cellsSegmEndFilename} - {appendedName}" '
                        "difference..."
                    )
                    if trackedCellsSegmData is not None:
                        parentSegmData = trackedCellsSegmData
                    else:
                        parentSegmData = segmDataCells
                    diffSegmData = parentSegmData.copy()
                    diffSegmData[trackedSubSegmData != 0] = 0

                    self.logger.log("Saving difference segmentation file...")
                    diffSegmPath = (
                        f"{subSegmFilename}_{appendedName}"
                        f"_{self.thirdSegmAppendedText}.npz"
                    )
                    io.savez_compressed(diffSegmPath, diffSegmData)
                    posData.saveIsSegm3Dmetadata(diffSegmPath)
                    del diffSegmData

                if self.relabelSubObjLab:
                    # When we relabel the sub-cell objs acdc_df is not valid anymore
                    # because IDs could be different
                    posData.acdc_df = None

                self.logger.log("Generating acdc_output tables...")
                # Update or create acdc_df for sub-cellular objects
                acdc_dfs_tracked = core.track_sub_cell_objects_acdc_df(
                    trackedSubSegmData,
                    posData.acdc_df,
                    replacedSubIds,
                    numSubObjPerCell,
                    tracked_cells_segm_data=trackedCellsSegmData,
                    cells_acdc_df=acdc_df_cell,
                    SizeT=posData.SizeT,
                    sigProgress=self.signals.sigUpdateInnerPbar,
                )
                subTrackedAcdcDf, trackedAcdcDf = acdc_dfs_tracked

                self.logger.log("Saving acdc_output tables...")
                subAcdcDfFilename, _ = os.path.splitext(posData.acdc_output_csv_path)
                subTrackedAcdcDfPath = f"{subAcdcDfFilename}_{appendedName}.csv"
                subTrackedAcdcDf.to_csv(subTrackedAcdcDfPath)

                if trackedAcdcDf is not None:
                    basen = posData.basename
                    cellsSegmFilename = os.path.basename(segmCellsPath)
                    cellsSegmFilename, ext = os.path.splitext(cellsSegmFilename)
                    cellsSegmEndname = cellsSegmFilename[len(basen) :]
                    trackedAcdcDfEndname = cellsSegmEndname.replace(
                        "segm", "acdc_output"
                    )
                    trackedAcdcDfFilename = f"{basen}{trackedAcdcDfEndname}"
                    trackedAcdcDfFilename = (
                        f"{trackedAcdcDfFilename}_{appendedName}.csv"
                    )
                    trackedAcdcDfPath = os.path.join(
                        posData.images_path, trackedAcdcDfFilename
                    )
                    trackedAcdcDf.to_csv(trackedAcdcDfPath)

                    if self.createThirdSegm:
                        if posData.SizeT == 1:
                            parentSegmData = parentSegmData[np.newaxis]
                        subAcdcDfFilename = subSegmFilename.replace(
                            ".npz", ".csv"
                        ).replace("segm", "acdc_output")
                        diffAcdcDfPath = (
                            f"{subAcdcDfFilename}_{appendedName}"
                            f"_{self.thirdSegmAppendedText}.csv"
                        )
                        third_segm_acdc_df = (
                            core.track_sub_cell_objects_third_segm_acdc_df(
                                parentSegmData, trackedAcdcDf
                            )
                        )
                        third_segm_acdc_df.to_csv(diffAcdcDfPath)

                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)


class ApplyTrackInfoWorker(BaseWorkerUtil):
    def __init__(
        self,
        parentWin,
        endFilenameSegm,
        trackInfoCsvPath,
        trackedSegmFilename,
        trackColsInfo,
        posPath,
    ):
        super().__init__(parentWin)
        self.endFilenameSegm = endFilenameSegm
        self.trackInfoCsvPath = trackInfoCsvPath
        self.trackedSegmFilename = trackedSegmFilename
        self.trackColsInfo = trackColsInfo
        self.posPath = posPath

    @worker_exception_handler
    def run(self):
        self.logger.log("Loading segmentation file...")
        self.signals.initProgressBar.emit(0)
        imagesPath = os.path.join(self.posPath, "Images")
        segmFilename = [
            f
            for f in myutils.listdir(imagesPath)
            if f.endswith(f"{self.endFilenameSegm}.npz")
        ][0]
        segmFilePath = os.path.join(imagesPath, segmFilename)
        segmData = np.load(segmFilePath)["arr_0"]

        self.logger.log("Loading table containing tracking info...")
        df = pd.read_csv(self.trackInfoCsvPath)

        frameIndexCol = self.trackColsInfo["frameIndexCol"]

        parentIDcol = self.trackColsInfo["parentIDcol"]
        pbarMax = len(df[frameIndexCol].unique())
        self.signals.initProgressBar.emit(pbarMax)

        # Apply tracking info
        result = core.apply_tracking_from_table(
            segmData,
            self.trackColsInfo,
            df,
            signal=self.signals.progressBar,
            logger=self.logger.log,
            pbarMax=pbarMax,
        )
        trackedData, trackedIDsMapper, deleteIDsMapper = result

        if self.trackedSegmFilename:
            trackedSegmFilepath = os.path.join(imagesPath, self.trackedSegmFilename)
        else:
            trackedSegmFilepath = os.path.join(segmFilePath)

        self.signals.initProgressBar.emit(0)
        self.logger.log("Saving tracked segmentation file...")
        io.savez_compressed(trackedSegmFilepath, trackedData)

        mapperPath = os.path.splitext(trackedSegmFilepath)[0]
        mapperJsonPath = f"{mapperPath}_deletedIDs_mapper.json"
        mapperJsonName = os.path.basename(mapperJsonPath)
        self.logger.log(f"Saving deleted IDs to {mapperJsonName}...")
        with open(mapperJsonPath, "w") as file:
            file.write(json.dumps(deleteIDsMapper))

        mapperPath = os.path.splitext(trackedSegmFilepath)[0]
        mapperJsonPath = f"{mapperPath}_replacedIDs_mapper.json"
        mapperJsonName = os.path.basename(mapperJsonPath)
        self.logger.log(f"Saving IDs replacements to {mapperJsonName}...")
        with open(mapperJsonPath, "w") as file:
            file.write(json.dumps(trackedIDsMapper))

        self.logger.log("Generating acdc_output table...")
        acdc_df = None
        if not self.trackedSegmFilename:
            # Fix existing acdc_df
            acdcEndname = self.endFilenameSegm.replace("_segm", "_acdc_output")
            acdcFilename = [
                f
                for f in myutils.listdir(imagesPath)
                if f.endswith(f"{acdcEndname}.csv")
            ]
            if acdcFilename:
                acdcFilePath = os.path.join(imagesPath, acdcFilename[0])
                acdc_df = pd.read_csv(acdcFilePath, index_col=["frame_i", "Cell_ID"])

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

            acdc_df = pd.concat(acdc_dfs, keys=keys, names=["frame_i", "Cell_ID"])
            segmFilename = os.path.basename(trackedSegmFilepath)
            acdcFilename = re.sub(segm_re_pattern, "_acdc_output", segmFilename)
            acdcFilePath = os.path.join(imagesPath, acdcFilename)

        self.signals.initProgressBar.emit(pbarMax)
        parentIDcol = self.trackColsInfo["parentIDcol"]
        trackIDsCol = self.trackColsInfo["trackIDsCol"]
        if parentIDcol != "None":
            self.logger.log(f'Adding lineage info from "{parentIDcol}" column...')
            acdc_df = core.add_cca_info_from_parentID_col(
                df,
                acdc_df,
                frameIndexCol,
                trackIDsCol,
                parentIDcol,
                len(segmData),
                signal=self.signals.progressBar,
                maskID_colname=self.trackColsInfo["maskIDsCol"],
                x_colname=self.trackColsInfo["xCentroidCol"],
                y_colname=self.trackColsInfo["yCentroidCol"],
            )

        self.logger.log("Saving acdc_output table...")
        acdc_df.to_csv(acdcFilePath)

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
            self.logger.log("-" * 30)
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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, "Images")
                basename, chNames = myutils.getBasenameAndChNames(
                    images_path, useExt=(".tif", ".h5")
                )

                self.signals.sigUpdatePbarDesc.emit(f"Loading {pos_path}...")

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
                posData.getBasenameAndChNames(useExt=(".tif", ".h5"))

                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_acdc_df=True,
                    load_metadata=True,
                    loadSegmInfo=True,
                )

                posDatas.append(posData)

                self.allPosDataInputs.append({"file_path": file_path, "chName": chName})

            # Iterate pos and calculate metrics
            numPos = len(self.allPosDataInputs)
            for p, posDataInputs in enumerate(self.allPosDataInputs):
                file_path = posDataInputs["file_path"]
                chName = posDataInputs["chName"]

                posData = load.loadData(file_path, chName)

                self.signals.sigUpdatePbarDesc.emit(f"Processing {posData.pos_path}")

                posData.getBasenameAndChNames(useExt=(".tif", ".h5"))
                posData.buildPaths()
                posData.loadImgData()

                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_acdc_df=True,
                    end_filename_segm=self.mainWin.endFilenameSegm,
                )
                if not posData.acdc_df_found:
                    relPath = (
                        f"...{os.sep}{expFoldername}{os.sep}{posData.pos_foldername}"
                    )
                    self.logger.log(
                        f'WARNING: Skipping "{relPath}" '
                        f"because acdc_output.csv file was not found."
                    )
                    self.missingAnnotErrors[relPath] = (
                        f'<br>FileNotFoundError: the Positon "{relPath}" '
                        "does not have the <code>acdc_output.csv</code> file.<br>"
                    )

                    continue

                acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
                self.logger.log(
                    f'Loaded path:\nACDC output file name: "{acdc_df_filename}"'
                )

                self.logger.log("Building tree...")
                try:
                    tree = core.LineageTree(posData.acdc_df)
                    error = tree.build()
                    if isinstance(error, KeyError):
                        self.logger.log(str(error))

                        self.logger.log(
                            "WARNING: Annotations missing in "
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


class CopyAllLostObjectsWorker(QObject):
    navigateToFrame = Signal(int)
    returnToFrame = Signal(int)
    copyLostObjectMask = Signal(int)
    refreshRp = Signal()
    progressBar = Signal(int)
    finished = Signal(object)
    critical = Signal(object)

    def __init__(self, gui, posData, for_future_frame_n, max_overlap_perc):
        super().__init__()
        self.gui = gui
        self.posData = posData
        self.for_future_frame_n = for_future_frame_n
        self.max_overlap_perc = max_overlap_perc

    @worker_exception_handler
    def run(self):
        current_frame_i = self.posData.frame_i
        last_visited_frame_i = self.gui.get_last_tracked_i()
        last_copied_frame_i = current_frame_i + self.for_future_frame_n + 1
        frames_range = (current_frame_i, last_copied_frame_i)
        overlap_warning = False
        output = {}

        for frame_i in range(*frames_range):
            if frame_i == self.posData.SizeT:
                break

            if frame_i > self.posData.frame_i:
                # Main thread navigates, runs tracking, updates rp/IDs, etc
                self.navigateToFrame.emit(frame_i)

            for lostObj in skimage.measure.regionprops(self.gui.lostObjImage):
                overlap = np.count_nonzero(
                    self.gui.currentLab2D[lostObj.slice][lostObj.image]
                )
                overlap_perc = overlap / lostObj.area * 100
                if overlap_perc > self.max_overlap_perc:
                    overlap_warning = True
                    continue

                self.copyLostObjectMask.emit(lostObj.label)

            # Refresh rp so the next frame's updateLostNewCurrentIDs sees the
            # copied IDs as belonging to this frame and marks them lost there.
            self.refreshRp.emit()

            self.progressBar.emit(1)

        if self.for_future_frame_n == 0:
            output["overlap_warning"] = overlap_warning
            self.finished.emit(output)
            return

        # Back to current frame
        self.returnToFrame.emit(current_frame_i)

        if last_visited_frame_i < last_copied_frame_i:
            output["doReinitLastSegmFrame"] = True
            output["last_visited_frame_i"] = last_visited_frame_i

        output["overlap_warning"] = overlap_warning
        self.finished.emit(output)

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    signals,
    workerLogger,
    worker_exception_handler,
)

