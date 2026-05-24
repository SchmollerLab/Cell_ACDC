"""Background Qt workers: data_prep."""

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
            "The following file is required for the alignment of 4D data "
            f'but it was not found: "{path}"'
        )

    def saveBkgrData(self, imageData, posData, isAligned=False):
        bkgrROI_data = {}
        for r, roi in enumerate(posData.bkgrROIs):
            xl, yt = [int(round(c)) for c in roi.pos()]
            w, h = [int(round(c)) for c in roi.size()]
            if not yt + h > yt or not xl + w > xl:
                # Prevent 0 height or 0 width roi
                continue
            is4D = posData.SizeT > 1 and posData.SizeZ > 1
            is3Dz = posData.SizeT == 1 and posData.SizeZ > 1
            is3Dt = posData.SizeT > 1 and posData.SizeZ == 1
            is2D = posData.SizeT == 1 and posData.SizeZ == 1
            if is4D:
                bkgr_data = imageData[:, :, yt : yt + h, xl : xl + w]
            elif is3Dz or is3Dt:
                bkgr_data = imageData[:, yt : yt + h, xl : xl + w]
            elif is2D:
                bkgr_data = imageData[yt : yt + h, xl : xl + w]
            bkgrROI_data[f"roi{r}_data"] = bkgr_data

        if not bkgrROI_data:
            return

        if isAligned:
            bkgr_data_fn = f"{posData.filename}_aligned_bkgrRoiData.npz"
        else:
            bkgr_data_fn = f"{posData.filename}_bkgrRoiData.npz"
        bkgr_data_path = os.path.join(posData.images_path, bkgr_data_fn)
        self.progress.emit("Saving background data to:")
        self.progress.emit(bkgr_data_path)
        io.savez_compressed(bkgr_data_path, **bkgrROI_data)

    def run(self):
        ch_name_selector = prompts.select_channel_name(
            which_channel="segm", allow_abort=False
        )
        for p, pos in enumerate(self.posFoldernames):
            if self.abort:
                break

            self.progress.emit(f"Processing {pos}...")

            posPath = os.path.join(self.expPath, pos)
            imagesPath = os.path.join(posPath, "Images")

            ls = myutils.listdir(imagesPath)
            if p == 0:
                ch_names, basenameNotFound = ch_name_selector.get_available_channels(
                    ls, imagesPath
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

                self.progress.emit(f"Selected channels: {self.selectedChannels}")

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
                    loadBkgrROIs=True,
                )

                imageData = posData.img_data

                prepped = False
                isAligned = False
                # Align
                if posData.loaded_shifts is not None:
                    self.progress.emit("Aligning frames...")
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
                    isCropped = int(df.at["cropped", "value"]) == 1
                    if isCropped:
                        self.saveBkgrData(imageData, posData, isAligned)
                        self.progress.emit("Cropping...")
                        x0 = int(df.at["x_left", "value"])
                        y0 = int(df.at["y_top", "value"])
                        x1 = int(df.at["x_right", "value"])
                        y1 = int(df.at["y_bottom", "value"])
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
                            "I cannot crop the data."
                        )

                if prepped:
                    self.progress.emit("Saving prepped data...")
                    io.savez_compressed(posData.align_npz_path, imageData)
                    if hasattr(posData, "tif_path"):
                        myutils.to_tiff(posData.tif_path, imageData)

            self.updatePbar.emit()
            if self.abort:
                break
        self.finished.emit()


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
        self.progress.emit(" ")
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

            self.progress.emit(f"Loading file: {file}")
            filename, ext = os.path.splitext(file)
            s0p = str(pos).zfill(numPosDigits)
            try:
                data = load.imread(filePath)
                if data.ndim == 3 and (data.shape[-1] == 3 or data.shape[-1] == 4):
                    self.progress.emit("Converting RGB image to grayscale...")
                    data = skimage.color.rgb2gray(data)
                    data = skimage.img_as_ubyte(data)

                posName = f"Position_{pos}"
                posPath = os.path.join(self.targetFolderPath, posName)
                imagesPath = os.path.join(posPath, "Images")
                if not os.path.exists(imagesPath):
                    os.makedirs(imagesPath, exist_ok=True)
                newFilename = f"s{s0p}_{filename}_{self.appendText}.tif"
                relPath = os.path.join(posName, "Images", newFilename)
                tifFilePath = os.path.join(imagesPath, newFilename)
                self.progress.emit(f"Saving to file: ...{os.sep}{relPath}")
                myutils.to_tiff(tifFilePath, data)
                pos += 1
            except Exception as e:
                self.progress.emit(
                    f"WARNING: {file} is not a valid image file. Skipping it."
                )

            self.progress.emit(" ")
            self.updatePbar.emit()

            if self.abort:
                break
        self.finished.emit()


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


class RestructMultiPosWorker(BaseWorkerUtil):
    sigSaveTiff = Signal(str, object, object)

    def __init__(self, rootFolderPath, dstFolderPath, action="copy"):
        super().__init__(None)
        self.rootFolderPath = rootFolderPath
        self.dstFolderPath = dstFolderPath
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.action = action

    @worker_exception_handler
    def run(self):
        load._restructure_multi_files_multi_pos(
            self.rootFolderPath,
            self.dstFolderPath,
            signals=self.signals,
            logger=self.logger.log,
            action=self.action,
        )
        self.signals.finished.emit(self)


class RestructMultiTimepointsWorker(BaseWorkerUtil):
    sigSaveTiff = Signal(str, object, object)

    def __init__(
        self,
        allChannels,
        frame_name_pattern,
        basename,
        validFilenames,
        rootFolderPath,
        dstFolderPath,
        segmFolderPath="",
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
        self.signals.initProgressBar.emit(len(self.validFilenames) + 1)
        for file in self.validFilenames:
            try:
                # Determine which channel is this file
                for ch in allChannels:
                    m = re.findall(rf"(.*)_{ch}{frame_name_pattern}", file)
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
                    "Skipping it."
                )
                continue

        self.signals.progressBar.emit(1)

        df_metadata = None
        partial_basename = self.basename
        allPosDataInfo = []
        for p, (posName, channelInfo) in enumerate(filesInfo.items()):
            self.logger.log(f"=" * 40)
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
                    f"WARNING: No valid image files found for position {posName}"
                )
                continue

            # Get basename
            if partial_basename:
                basename = f"{partial_basename}_{posName}_"
            else:
                basename = f"{posName}_"

            # Get SizeT from first file
            SizeT = len(filesList)

            # Save metadata.csv
            df_metadata = pd.DataFrame(
                {"SizeT": SizeT, "basename": basename}, index=["values"]
            )

            # Iterate channels
            for c, (channelName, filesList) in enumerate(channelInfo.items()):
                self.logger.log(f'  Processing channel "{channelName}"...')
                # Sort by frame number
                sortedFilesList = sorted(filesList, key=lambda t: t[1])

                df_metadata[f"channel_{c}_name"] = [channelName]

                imagesPath = os.path.join(dstFolderPath, f"Position_{p + 1}", "Images")
                if not os.path.exists(imagesPath):
                    os.makedirs(imagesPath, exist_ok=True)

                # Iterate frames
                videoData = None
                srcSegmPaths = [""] * SizeT
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

                    if segmFolderPath and c == 0:
                        srcSegmFilePath = os.path.join(segmFolderPath, file)
                        srcSegmPaths[frame_i] = srcSegmFilePath

                    SizeZ = 1
                    if img.ndim == 3:
                        SizeZ = len(img)

                    df_metadata["SizeZ"] = [SizeZ]

                    self.signals.progressBar.emit(1)

                if videoData is None:
                    self.logger.log(
                        f"WARNING: No valid image files found for position "
                        f'"{posName}", channel "{channelName}"'
                    )
                    continue
                else:
                    imgFileName = f"{basename}{channelName}.tif"
                    dstImgFilePath = os.path.join(imagesPath, imgFileName)
                    dstSegmFileName = f"{basename}segm_{channelName}.npz"
                    dstSegmPath = os.path.join(imagesPath, dstSegmFileName)
                    imgDataInfo = {
                        "path": dstImgFilePath,
                        "SizeT": SizeT,
                        "SizeZ": SizeZ,
                        "data": videoData,
                        "frameNumbers": frameNumbers,
                        "dst_segm_path": dstSegmPath,
                        "src_segm_paths": srcSegmPaths,
                    }
                    allPosDataInfo.append(imgDataInfo)

            if df_metadata is not None:
                metadata_csv_path = os.path.join(imagesPath, f"{basename}metadata.csv")
                df_metadata = df_metadata.T
                df_metadata.index.name = "Description"
                df_metadata.to_csv(metadata_csv_path)

            self.logger.log(f"*" * 40)

        if not allPosDataInfo:
            self.signals.finished.emit(self)
            return

        self.signals.initProgressBar.emit(len(allPosDataInfo))
        self.logger.log("Saving image files...")
        maxSizeT = max([d["SizeT"] for d in allPosDataInfo])
        minFrameNumber = min([d["frameNumbers"][0] for d in allPosDataInfo])
        # Pad missing frames in video files according to frame number
        for p, imgDataInfo in enumerate(allPosDataInfo):
            SizeT = imgDataInfo["SizeT"]
            SizeZ = imgDataInfo["SizeZ"]
            dstImgFilePath = imgDataInfo["path"]
            videoData = imgDataInfo["data"]
            frameNumbers = imgDataInfo["frameNumbers"]
            paddedShape = (maxSizeT, *videoData.shape[1:])
            imgDataInfo["paddedShape"] = paddedShape
            dtype = videoData.dtype
            paddedVideoData = np.zeros(paddedShape, dtype=dtype)
            for n, img in zip(frameNumbers, videoData):
                frame_i = n - minFrameNumber
                paddedVideoData[frame_i] = img

            del videoData
            imgDataInfo["data"] = None

            self.mutex.lock()
            self.sigSaveTiff.emit(dstImgFilePath, paddedVideoData, self.waitCond)
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()

            self.signals.progressBar.emit(1)

        if not segmFolderPath:
            self.signals.finished.emit(self)
            return

        self.signals.initProgressBar.emit(len(allPosDataInfo))
        self.logger.log("Saving segmentation files...")
        for p, imgDataInfo in enumerate(allPosDataInfo):
            SizeT = imgDataInfo["SizeT"]
            frameNumbers = imgDataInfo["frameNumbers"]
            SizeT = imgDataInfo["SizeT"]
            SizeZ = imgDataInfo["SizeZ"]
            frameNumbers = imgDataInfo["frameNumbers"]
            paddedShape = imgDataInfo["paddedShape"]
            segmData = np.zeros(paddedShape, dtype=np.uint32)
            for n, segmFilePath in zip(frameNumbers, imgDataInfo["src_segm_paths"]):
                frame_i = n - minFrameNumber
                try:
                    lab = load.imread(segmFilePath).astype(np.uint32)
                    segmData[frame_i] = lab
                except Exception as e:
                    self.logger.log(traceback.format_exc())
                    self.logger.log(
                        "WARNING: The following segmentation file does not "
                        f'exist, saving empty masks: "{srcSegmFilePath}"'
                    )

            io.savez_compressed(imgDataInfo["dst_segm_path"], segmData)
            del segmData

        self.signals.finished.emit(self)


class FucciPreprocessWorker(BaseWorkerUtil):
    sigAskAppendName = Signal(str)
    sigAskParams = Signal(object, object)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitAskParams(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.sigAskParams.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def applyPipeline(self, first_ch_data, second_ch_data, filter_kwargs):
        processed_data = np.zeros(first_ch_data.shape, dtype=np.uint8)
        pbar = tqdm(total=len(processed_data), ncols=100)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            iterable = enumerate(zip(first_ch_data, second_ch_data))
            func = partial(core.fucci_pipeline_executor_map, **filter_kwargs)
            result = executor.map(func, iterable)
            for frame_i, processed_img in result:
                processed_img = skimage.exposure.rescale_intensity(
                    processed_img, out_range=(0, 255)
                )
                processed_img = processed_img.astype(np.uint8)
                processed_data[frame_i] = processed_img
                pbar.update()
        pbar.close()

        return processed_data

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f"Setup parameters"

            if i == 0:
                abort = self.emitAskParams(exp_path, pos_foldernames)
                if abort:
                    self.sigAborted.emit()
                    return

            # Ask appendend name
            self.mutex.lock()
            self.sigAskAppendName.emit(self.basename)
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

                self.logger.log(f"Loading {self.firstChannelName} channel data...")
                first_ch_filepath = load.get_filename_from_channel(
                    images_path, self.firstChannelName
                )
                first_ch_data = load.load_image_file(first_ch_filepath)

                self.logger.log(f"Loading {self.secondChannelName} channel data...")
                second_ch_filepath = load.get_filename_from_channel(
                    images_path, self.secondChannelName
                )
                second_ch_data = load.load_image_file(second_ch_filepath)

                self.logger.log("Applying FUCCI pre-processing pipeline...\n")
                processed_data = self.applyPipeline(
                    first_ch_data, second_ch_data, self.fucciFilterKwargs
                )

                basename, chNames = myutils.getBasenameAndChNames(images_path)
                _, ext = os.path.splitext(first_ch_filepath)
                processed_filename = f"{basename}{appendedName}{ext}"
                processed_filepath = os.path.join(images_path, processed_filename)
                self.logger.log(
                    f'Saving pre-processed images to "{processed_filepath}"...'
                )
                io.save_image_data(processed_filepath, processed_data)

                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)


class SaveProcessedDataWorker(QObject):
    def __init__(
        self,
        allPosData: Iterable["load.loadData"],
        appended_text_filename: str,
        ext: str = None,
    ):
        QObject.__init__(self)
        self.allPosData = allPosData
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.appended_text_filename = appended_text_filename
        self.ext = ext

    @worker_exception_handler
    def run(self):
        self.signals.initProgressBar.emit(0)
        for posData in self.allPosData:
            ext_loc = self.ext if self.ext is not None else posData.ext
            processed_filename = (
                f"{posData.basename}{posData.user_ch_name}_"
                f"{self.appended_text_filename}{ext_loc}"
            )
            processed_filepath = os.path.join(posData.images_path, processed_filename)
            self.logger.log(f"Saving {processed_filepath}...")
            processed_data = posData.preprocessedDataArray()
            if processed_data is None:
                self.logger.log(
                    f"[WARNING]: {posData.pos_foldername} does not have "
                    "preprocessed data. Skipping it."
                )
                continue

            io.save_image_data(processed_filepath, processed_data)

        self.signals.finished.emit(self)


class SaveCombinedChannelsWorker(QObject):
    sigDebugShowImg = Signal(object)

    def __init__(
        self, allPosData: Iterable["load.loadData"], filename: str, debug: bool = False
    ):
        QObject.__init__(self)
        self.allPosData = allPosData
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.filename = filename
        self.debug = debug

    @worker_exception_handler
    def run(self):
        self.signals.initProgressBar.emit(0)
        for posData in self.allPosData:
            processed_filepath = os.path.join(posData.images_path, self.filename)
            self.logger.log(f"Saving {processed_filepath}...")
            processed_data = posData.combinedChannelsDataArray()
            if processed_data is None:
                self.logger.log(
                    f"[WARNING]: {posData.pos_foldername} does not have "
                    "combined channels data. Skipping it."
                )
                continue
            if self.debug:
                printl(processed_data.shape)
                printl(processed_data.dtype)
                printl(processed_data.min())
                printl(processed_data.max())
                printl(processed_filepath)
                self.sigDebugShowImg.emit(processed_data)
            # cellacdc.plot.imshow(processed_data)
            io.save_image_data(processed_filepath, processed_data)

        self.signals.finished.emit(self)


class CustomPreprocessWorkerGUI(QObject):
    sigDone = Signal(object, str)
    sigPreviewDone = Signal(object, tuple)
    sigIsQueueEmpty = Signal(bool)

    def __init__(self, mutex, waitCond):
        QObject.__init__(self)
        self.signals = signals()
        self.mutex = mutex
        self.waitCond = waitCond
        self.logger = workerLogger(self.signals.progress)
        self.dataQ = deque(maxlen=2)
        self.exit = False
        self.wait = True
        self._abort = False

    def enqueue(
        self,
        func: Callable,
        image: np.ndarray,
        recipe: Dict[str, Any],
        key: Tuple[int, int, Union[int, str]],
    ):
        self.dataQ.append((func, image, recipe, key))
        if len(self.dataQ) == 1:
            self.sigIsQueueEmpty.emit(False)
            # Wake up worker upon inserting first element
            self.wakeUp()

    def wakeUp(self):
        self.wait = False
        self.waitCond.wakeAll()

    def pause(self):
        self.wait = True
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def abort(self):
        self._abort = True

    def stop(self):
        self.abort()
        self.exit = True
        self.waitCond.wakeAll()
        self.signals.finished.emit(self)

    def setupJob(
        self, func: Callable, image_data: np.ndarray, recipe: Dict[str, Any], how: str
    ):
        self._func = func
        self._image_data = image_data
        self._recipe = recipe
        self._how = how

    def runJob(self, image=None, recipe=None):
        if image is None:
            image = self._image_data.copy()
        if recipe is None:
            recipe = self._recipe

        return self.applyRecipe(self._func, image, recipe)

    def applyRecipe(
        self, func: Callable, image: np.ndarray, recipe: List[Dict[str, Any]]
    ):
        preprocessed_data = func(image, recipe)

        keep_input_data_type = recipe[0].get("keep_input_data_type", True)
        if not keep_input_data_type:
            return preprocessed_data

        try:
            preprocessed_data = myutils.convert_to_dtype(preprocessed_data, image.dtype)
        except Exception as err:
            preprocessed_data = preprocessed_data.astype(image.dtype)
        return preprocessed_data

    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log("Closing pre-processing worker...")
                break
            elif self.wait:
                self.logger.log("Pre-processing worker paused.")
                self.pause()
            elif len(self.dataQ) > 0:
                func, image, recipe, key = self.dataQ.pop()
                processed_data = self.applyRecipe(func, image, recipe)
                self.sigPreviewDone.emit(processed_data, key)
                if len(self.dataQ) == 0:
                    self.wait = True
                    self.sigIsQueueEmpty.emit(True)
            else:
                self.logger.log("Pre-processing worker resumed.")
                processed_data = self.runJob()
                self.sigDone.emit(processed_data, self._how)
                self.wait = True

        self.signals.finished.emit(self)


class CombineChannelsWorkerGUI(CustomPreprocessWorkerGUI):
    sigDone = Signal(object, list)
    sigPreviewDone = Signal(object, list)
    sigAskLoadChannels = Signal(set, object)

    def __init__(
        self,
        mutex,
        waitCond,
        logger_func: Callable,
    ):
        #                 signals_parent=None):
        super().__init__(mutex, waitCond)

        self.waitCondLoadFluoChannels = QWaitCondition()
        self.logger_func = logger_func

        # if not signals_parent:
        #     signals_parent = signals()

        # self.signals = signals_parent

    def enqueue(
        self,
        data,
        steps: Dict[str, Any],
        key: Tuple[int, int, Union[int, str]],
        keep_input_data_type: bool,
        output_as_segm: bool,
        formula: str,
    ):
        self.dataQ.append(
            (data, steps, key, keep_input_data_type, output_as_segm, formula)
        )
        if len(self.dataQ) == 1:
            self.sigIsQueueEmpty.emit(False)
            # Wake up worker upon inserting first element
            self.wakeUp()

    def setupJob(
        self,
        data: Dict[str, np.ndarray],
        steps: Dict[str, Any],
        keep_input_data_type: bool,
        key: Tuple[Union[int, None], Union[int, None], Union[int, None]],
        output_as_segm: bool,
        formula: str,
    ):
        self._key = key
        self._steps = steps
        self._data = data
        self._keep_input_data_type = keep_input_data_type
        self._output_as_segm = output_as_segm
        self._formula = formula

    def runJob(
        self,
        data=None,
        steps=None,
        keep_input_data_type=None,
        key=None,
        output_as_segm=None,
        formula=None,
    ):
        if data is None:
            data = self._data
        if steps is None:
            steps = self._steps
        if keep_input_data_type is None:
            keep_input_data_type = self._keep_input_data_type
        if key is None:
            key = self._key
        if output_as_segm is None:
            output_as_segm = self._output_as_segm
        if formula is None:
            formula = self._formula

        if not steps and formula is None:
            return

        return self.applySteps(
            data, steps, keep_input_data_type, key, output_as_segm, formula=formula
        )

    def applySteps(
        self,
        data: Dict[str, np.ndarray],
        steps: List[Dict[str, Any]],
        keep_input_data_type: bool,
        key: Tuple[Union[int, None], Union[int, None], Union[int, None]],
        output_as_segm: bool,
        formula: str,
    ):

        new_keys = []
        key = list(key)
        if key[0] is None:
            pos_number = len(data)
            key[0] = list(range(pos_number))
        else:
            key[0] = [key[0]]

        for pos_i in key[0]:
            new_keys_per_pos = [[pos_i]]
            if key[1] is None:
                frames = data[pos_i].SizeT
                new_keys_per_pos.append(list(range(frames)))
            else:
                new_keys_per_pos.append([key[1]])

            if key[2] is None:
                z_slices = data[pos_i].SizeZ
                if not z_slices:
                    z_slices = 1
                new_keys_per_pos.append(list(range(z_slices)))
            else:
                new_keys_per_pos.append([key[2]])

            new_keys_per_pos = list(itertools.product(*new_keys_per_pos))
            new_keys.extend(new_keys_per_pos)

        output_imgs, out_keys = core.combine_channels_multithread_return_imgs(
            steps=steps,
            data=data,
            keep_input_data_type=keep_input_data_type,
            keys=new_keys,
            logger_func=self.logger,
            signals=self.signals,
            output_as_segm=output_as_segm,
            formula=formula,
        )
        return output_imgs, out_keys

    def requiredChannels(self, steps=None, pos_i=None):
        if steps is None:
            steps = self._steps

        required_channels = core.get_selected_channels(steps)
        if pos_i is None:
            pos_i = self._key[0]

        return required_channels, pos_i

    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log("Closing combining channels worker...")
                break
            elif self.wait:
                self.logger.log("Combining channels worker paused.")
                self.pause()
            elif len(self.dataQ) > 0:
                data, steps, key, keep_input_data_type, output_as_segm, formula = (
                    self.dataQ.pop()
                )
                requ_steps, pos_i = self.requiredChannels(steps, key[0])
                self.emitsigAskLoadChannels(requ_steps, pos_i)
                output_imgs, out_keys = self.applySteps(
                    data,
                    steps,
                    keep_input_data_type,
                    key,
                    output_as_segm=output_as_segm,
                    formula=formula,
                )
                self.sigPreviewDone.emit(output_imgs, out_keys)
                if len(self.dataQ) == 0:
                    self.wait = True
                    self.sigIsQueueEmpty.emit(True)
            else:
                self.logger.log("Combining channels worker resumed.")
                requ_steps, pos_i = self.requiredChannels()
                self.emitsigAskLoadChannels(requ_steps, pos_i)
                output_imgs, out_keys = self.runJob()
                self.sigDone.emit(output_imgs, out_keys)
                self.wait = True

        self.signals.finished.emit(self)

    def emitsigAskLoadChannels(self, requChannels, pos_i):
        self.mutex.lock()
        self.sigAskLoadChannels.emit(requChannels, pos_i)
        self.waitCondLoadFluoChannels.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def wake_waitCondLoadFluoChannels(self):
        self.mutex.lock()
        self.waitCondLoadFluoChannels.wakeAll()
        self.mutex.unlock()


class CustomPreprocessWorkerUtil(BaseWorkerUtil):
    sigAskAppendName = Signal(str)
    sigAskSetupRecipe = Signal(object, object)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitAskSetupRecipe(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.sigAskSetupRecipe.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def applyPipeline(
        self,
        images_path: os.PathLike,
        channel_names: Iterable[str],
        recipe: List[Dict[str, Any]],
        appended_text_filename: str,
    ):
        posData = None
        preprocessed_data = {}
        for channel in channel_names:
            self.logger.log(f"Loading {channel} channel data...")
            ch_filepath = load.get_filename_from_channel(images_path, channel)
            ch_image_data = load.load_image_file(ch_filepath)
            if posData is None:
                posData = load.loadData(ch_filepath, channel)
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_metadata=True,
                )
            if posData.SizeT == 1:
                ch_image_data = (ch_image_data,)

            preprocessed_ch_data = core.preprocess_image_from_recipe_multithread(
                ch_image_data, recipe
            )

            keep_input_data_type = recipe[0].get("keep_input_data_type", True)
            if keep_input_data_type:
                preprocessed_ch_data = myutils.convert_to_dtype(
                    preprocessed_ch_data, ch_image_data.dtype
                )

            _, ext = os.path.splitext(ch_filepath)
            basename = posData.basename
            processed_filename = f"{basename}{channel}_{appended_text_filename}{ext}"
            preprocessed_data[processed_filename] = preprocessed_ch_data

        return preprocessed_data

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = "Setup recipe"

            if i == 0:
                abort = self.emitAskSetupRecipe(exp_path, pos_foldernames)
                if abort:
                    self.sigAborted.emit()
                    return

                # Ask append name
                self.mutex.lock()
                basename = f"{self.basename}{self.selectedChannels[0]}_"
                self.sigAskAppendName.emit(basename)
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
                self.logger.log("Applying custom pre-processing recipe...\n")
                processed_data = self.applyPipeline(
                    images_path, self.selectedChannels, self.recipe, appendedName
                )

                for filename, preprocessed_ch_data in processed_data.items():
                    preprocessed_filepath = os.path.join(images_path, filename)
                    self.logger.log(
                        f'Saving pre-processed images to "{preprocessed_filepath}"...'
                    )

                    io.save_image_data(preprocessed_filepath, preprocessed_ch_data)
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)


class CombineChannelsWorkerUtil(BaseWorkerUtil):
    sigAskAppendName = Signal(str)
    sigAskSetup = Signal(object)
    sigAborted = Signal()

    def __init__(self, mainWin, mutex=None, waitCond=None):
        super().__init__(mainWin)

    def emitAskSetup(self, expPaths):
        self.mutex.lock()
        self.sigAskSetup.emit(expPaths)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def applyPipeline(
        self,
        image_paths: os.PathLike,
        steps: Dict[str, Dict[str, Any]],
        appended_text_filename: str,
        keep_input_data_type: bool,
        n_threads: int = None,
        formula: str = None,
    ):
        save_filepaths = []
        images_path_to_process = []
        if self.saveAsSegm:
            out_ext = ".npz"
            basename_ext = "segm_"
        else:
            out_ext = ".tif"
            basename_ext = ""
        for images_path in image_paths:
            basename, channels = myutils.getBasenameAndChNames(images_path)

            savename = f"{basename}{basename_ext}{appended_text_filename}{out_ext}"

            images_path_to_process.append(images_path)
            save_filepaths.append(os.path.join(images_path, savename))

        core.combine_channels_multithread(
            steps=steps,
            images_paths=images_path_to_process,
            keep_input_data_type=keep_input_data_type,
            save_filepaths=save_filepaths,
            signals=self.signals,
            logger_func=self.logger.log,
            n_threads=n_threads,
            output_as_segm=self.saveAsSegm,
            formula=formula,
        )

    @worker_exception_handler
    def run(self):

        self.signals.initProgressBar.emit(0)

        expPaths = self.mainWin.expPaths
        abort = self.emitAskSetup(expPaths)
        if abort:
            self.sigAborted.emit()
            return

        # Ask append name
        self.mutex.lock()
        basename = f"{self.basename}"
        self.sigAskAppendName.emit(basename)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return

        appendedName = self.appendedName

        selectedSteps = self.selectedSteps

        self.logger.log("Applying pipeline...")
        self.logger.log("Selected steps:")
        for step in selectedSteps.values():
            self.logger.log(step)

        image_paths = []
        for exp_path, pos_foldernames in expPaths.items():
            image_paths += [
                os.path.join(exp_path, pos, "Images") for pos in pos_foldernames
            ]

        self.signals.initProgressBar.emit(len(pos_foldernames))
        formula = self.formula
        self.applyPipeline(
            image_paths,
            selectedSteps,
            appendedName,
            self.keepInputDataType,
            n_threads=self.nThreads,
            formula=formula,
        )

        self.signals.finished.emit(self)

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    signals,
    workerLogger,
    worker_exception_handler,
)

