"""Background Qt workers: segm."""

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

class SegForLostIDsWorker(QObject):
    sigAskInit = Signal()
    sigAskInstallModel = Signal(str)
    sigshowImageDebug = Signal(object)
    sigStoreData = Signal(bool)
    sigUpdateRP = Signal(bool, bool)
    # sigGetData = Signal()
    # sigGet2Dlab = Signal()
    # sigGetTrackedLostIDs = Signal()
    # sigGetBrushID = Signal()
    sigSegForLostIDsWorkerAskInstallGPU = Signal(str, bool)
    sigTrackManuallyAddedObject = Signal(object, object, bool, bool)

    def __init__(self, guiWin, mutex, waitCond, debug=False):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.guiWin = guiWin
        self.mutex = mutex
        self.waitCond = waitCond
        self._debug = debug

    def emitSigAskInit(self):
        self.mutex.lock()
        self.sigAskInit.emit()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitSigShowImageDebug(self, img):
        # self.mutex.lock()
        self.sigshowImageDebug.emit(img)
        # self.waitCond.wait(self.mutex)
        # self.mutex.unlock()

    def emitSigStoreData(self, autosave):
        self.mutex.lock()
        self.sigStoreData.emit(autosave)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitSigUpdateRP(self, wl_track_og_curr, wl_update):
        self.mutex.lock()
        self.sigUpdateRP.emit(wl_track_og_curr, wl_update)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    # def emitSigGetData(self):
    #     self.mutex.lock()
    #     self.sigGetData.emit()
    #     self.waitCond.wait(self.mutex)
    #     self.mutex.unlock()

    def emitSigAskInstallModel(self, model_name):
        self.mutex.lock()
        self.sigAskInstallModel.emit(model_name)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitSigAskInstallGPU(self, base_model_name, use_gpu):
        self.mutex.lock()
        self.sigSegForLostIDsWorkerAskInstallGPU.emit(base_model_name, use_gpu)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    # def emitGet2Dlab(self):
    #     self.mutex.lock()
    #     self.sigGet2Dlab.emit()
    #     self.waitCond.wait(self.mutex)
    #     self.mutex.unlock()

    # def emitGetTrackedLostIDs(self):
    #     self.mutex.lock()
    #     self.sigGetTrackedLostIDs.emit()
    #     self.waitCond.wait(self.mutex)
    #     self.mutex.unlock()

    # def emitGetBrushID(self):
    #     self.mutex.lock()
    #     self.sigGetBrushID.emit()
    #     self.waitCond.wait(self.mutex)
    #     self.mutex.unlock()

    def emitTrackManuallyAddedObject(self, IDs, isLost, wl_update, wl_track_og_curr):
        self.mutex.lock()
        self.sigTrackManuallyAddedObject.emit(IDs, isLost, wl_update, wl_track_og_curr)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    @worker_exception_handler
    def run(self):
        posData = self.guiWin.data[self.guiWin.pos_i]
        frame_i = posData.frame_i

        if not self.guiWin.SegForLostIDsSettings:
            self.emitSigAskInit()

        if not self.guiWin.SegForLostIDsSettings:
            self.signals.finished.emit(self)
            return

        self.logger.info("Segmentation for lost IDs started.")
        model_name = "local_seg"
        base_model_name = self.guiWin.SegForLostIDsSettings["base_model_name"]
        idx = self.guiWin.modelNames.index(model_name)
        acdcSegment = self.guiWin.acdcSegment_li[idx]

        init_kwargs = self.guiWin.SegForLostIDsSettings["win"].init_kwargs

        use_gpu = init_kwargs.get("device_type", "cpu") != "cpu"
        use_gpu = use_gpu or init_kwargs.get("use_gpu", False)

        self.emitSigAskInstallGPU(base_model_name, use_gpu)

        if not self.gpu_go:
            self.signals.finished.emit(self)
            return

        if not self.dont_force_cpu:
            if "device" in init_kwargs:
                init_kwargs["device"] = "cpu"
            if "use_gpu" in init_kwargs:
                init_kwargs["use_gpu"] = False

        if (
            acdcSegment is None
            or base_model_name != self.guiWin.local_seg_base_model_name
        ):
            try:
                self.logger.info(f"Importing {base_model_name}...")
                self.emitSigAskInstallModel(base_model_name)
                acdcSegment = myutils.import_segment_module(base_model_name)
                self.guiWin.acdcSegment_li[idx] = acdcSegment
                self.guiWin.local_seg_base_model_name = base_model_name
            except (IndexError, ImportError, KeyError) as e:
                self.logger.warning(
                    f"Cannot import {base_model_name} model. Please install it first."
                )
                self.signals.critical.emit(
                    (
                        self,
                        f"Cannot import {base_model_name} model. "
                        "Please install it first.",
                    )
                )
                self.signals.finished.emit(self)
                return

        win = self.guiWin.SegForLostIDsSettings["win"]
        init_kwargs_new = self.guiWin.SegForLostIDsSettings["init_kwargs_new"]
        args_new = self.guiWin.SegForLostIDsSettings["args_new"]

        model = myutils.init_segm_model(acdcSegment, posData, init_kwargs_new)
        if model is None:
            self.logger.info("Segmentation model was not initialized correctly!")
            self.signals.critical.emit(
                (self, "Segmentation model was not initialized correctly!")
            )
            self.signals.finished.emit(self)
            return
        if self._debug:
            try:
                model.setupLogger(self.guiwin.logger)
            except Exception as e:
                pass

        assigned_IDs = []
        missing_IDs_global = set()
        original_lab = posData.lab.copy()
        IDs_bboxs_list = []
        bboxs_list = []

        curr_img = self.guiWin.getDisplayedImg1()
        prev_lab = self.guiWin.get_2Dlab(posData.allData_li[frame_i - 1]["labels"])
        prev_IDs = set(posData.allData_li[frame_i - 1]["IDs"])

        # should probably not paly so much with posData.lab, instead handle stuff myself
        self.signals.initProgressBar.emit(2 * args_new["max_iterations"])
        new_labs = np.zeros(
            [args_new["max_iterations"], *posData.lab.shape], dtype=np.uint32
        )
        for i in range(args_new["max_iterations"]):
            curr_lab = self.guiWin.get_2Dlab(posData.lab)
            tracked_lost_IDs = self.guiWin.getTrackedLostIDs()
            new_unique_ID = self.guiWin.setBrushID(useCurrentLab=True, return_val=True)

            missing_IDs = prev_IDs - set(posData.IDs) - set(tracked_lost_IDs)
            missing_IDs_global.update(missing_IDs)

            assigned_IDs_prev = assigned_IDs.copy()
            out = segm_utils.single_cell_seg(
                model,
                prev_lab,
                curr_lab,
                curr_img,
                missing_IDs,
                new_unique_ID,
                win,
                posData,
                distance_filler_growth=args_new["distance_filler_growth"],
                overlap_threshold=args_new["overlap_threshold"],
                padding=args_new["padding"],
            )
            new_lab, assigned_IDs, IDs_bboxs, bboxs = out

            IDs_bboxs_list.append(IDs_bboxs)
            bboxs_list.append(bboxs)
            posData.lab = new_lab
            self.emitSigUpdateRP(wl_update=True, wl_track_og_curr=False)
            newly_assigned_IDs = set(assigned_IDs) - set(assigned_IDs_prev)
            self.emitTrackManuallyAddedObject(newly_assigned_IDs, True, False, False)
            new_labs[i] = posData.lab.copy()
            self.signals.progressBar.emit(1)

        if self._debug:
            originals = []
            models = []

        posData.lab = original_lab.copy()

        global_area_mean = np.mean([obj.area for obj in posData.rp])
        for IDs_bboxs, bboxs in zip(IDs_bboxs_list, bboxs_list):
            model_lab = new_labs[i]
            if self._debug:
                originals.append(original_lab.copy())
                models.append(posData.lab.copy())

            for IDs, bbox in zip(IDs_bboxs, bboxs):
                box_x_min, box_x_max, box_y_min, box_y_max = bbox
                original_bbox_lab = original_lab[
                    box_x_min:box_x_max, box_y_min:box_y_max
                ]
                original_bbox_lab_cleared_borders = skimage.segmentation.clear_border(
                    original_bbox_lab
                )
                box_model_lab = model_lab[box_x_min:box_x_max, box_y_min:box_y_max]

                # original_bbox_lab[np.isin(original_bbox_lab, IDs)] = 0 should be a given. If not seg for lost IDs this recommended

                box_model_lab = skimage.segmentation.clear_border(
                    box_model_lab, buffer_size=1
                )

                rp_model_lab = skimage.measure.regionprops(box_model_lab)
                rp_original_lab = skimage.measure.regionprops(original_bbox_lab)
                rp_original_lab_cleared = skimage.measure.regionprops(
                    original_bbox_lab_cleared_borders
                )

                original_IDs = [obj.label for obj in rp_original_lab]
                areas = [obj.area for obj in rp_original_lab_cleared]
                if len(areas) > 0:
                    area_mean = np.mean(areas)
                else:
                    area_mean = global_area_mean
                if args_new["allow_only_tracked_cells"]:
                    filtered_IDs = [
                        obj.label
                        for obj in rp_model_lab
                        if obj.area > (1 - args_new["size_perc_diff"]) * area_mean
                        and obj.area < (1 + args_new["size_perc_diff"]) * area_mean
                        and obj.label not in original_IDs
                        and obj.label in missing_IDs_global
                    ]
                else:
                    filtered_IDs = [
                        obj.label
                        for obj in rp_model_lab
                        if obj.area > (1 - args_new["size_perc_diff"]) * area_mean
                        and obj.area < (1 + args_new["size_perc_diff"]) * area_mean
                        and obj.label not in original_IDs
                    ]

                if self._debug or DEBUG:
                    filtered_sizes = [
                        (obj.label, obj.area)
                        for obj in rp_model_lab
                        if obj.label in filtered_IDs
                    ]
                    self.logger.info(f"Filtered sizes: {filtered_sizes}")
                for label in filtered_IDs:
                    original_bbox_lab[box_model_lab == label] = (
                        label  # here the stuff should be tracked, so we keep the ID!
                    )

                # original_lab[box_x_min:box_x_max, box_y_min:box_y_max] = original_bbox_lab

            self.signals.progressBar.emit(1)

        posData.lab = original_lab

        # if self._debug:
        #     originals = np.concatenate(originals, axis=0)
        #     models = np.concatenate(models, axis=0)
        #     self.emitSigShowImageDebug(originals)
        #     self.emitSigShowImageDebug(models)

        self.emitSigUpdateRP(wl_track_og_curr=True, wl_update=True)
        self.emitSigStoreData(autosave=True)

        self.logger.info("Segmentation for lost IDs done.")

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
        self.logger.log("Draw box around object to start magic labeller.")
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
            self.logger.log("Magic labeller started...")
        self.started = True
        self.waitCond.wakeAll()

    def _stop(self):
        self.logger.log("Magic labeller backend process done. Closing it...")
        self.exit = True
        self.waitCond.wakeAll()

    def _segment_image(self, img, secondChannelImg):
        if secondChannelImg is not None:
            img = self.Gui.labelRoiModel.second_ch_img_to_stack(img, secondChannelImg)

        lab = core.segm_model_segment(
            self.Gui.labelRoiModel,
            img,
            self.Gui.model_kwargs,
            preproc_recipe=self.Gui.preproc_recipe,
            posData=self.posData,
        )
        if self.Gui.applyPostProcessing:
            from cellacdc.workflow.pipelines.postprocess_nodes import apply_postprocess

            lab = apply_postprocess(
                lab,
                img,
                self.posData,
                self.posData.frame_i,
                apply_postprocessing=True,
                standard_postprocess_kwargs=self.Gui.standardPostProcessKwargs,
                custom_postprocess_features=self.Gui.customPostProcessFeatures,
                custom_postprocess_grouped_features=self.Gui.customPostProcessGroupedFeatures,
            )
        return lab

    @worker_exception_handler
    def run(self):
        while not self.exit:
            if self.exit:
                break
            elif self.started:
                self.logger.log("Magic labeller is doing its magic...")
                if self.isTimelapse:
                    segmData = np.zeros(self.imageData.shape, dtype=np.uint32)
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


class segmWorker(QObject):
    finished = Signal(np.ndarray, float)
    debug = Signal(object)
    critical = Signal(object)

    def __init__(
        self,
        mainWin,
        secondChannelData=None,
        mutex: QWaitCondition = None,
        waitCond: QMutex = None,
    ):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.logger = self.mainWin.logger
        self.z_range = None
        self.secondChannelData = secondChannelData
        self.mutex = mutex
        self.waitCond = waitCond

    def emitDebug(self, to_debug):
        if self.mutex is None:
            return

        self.mutex.lock()
        self.debug.emit(to_debug)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    @worker_exception_handler
    def run(self):
        from cellacdc.workflow.adapters import (
            interactive_segm_context_from_main_win,
            runnable_config_from_main_win,
        )
        from cellacdc.workflow.pipelines.interactive_segm import (
            build_interactive_segm_graph,
        )
        from cellacdc.workflow.state import InteractiveSegmState

        t0 = time.perf_counter()
        ctx = interactive_segm_context_from_main_win(
            self.mainWin,
            second_channel_data=self.secondChannelData,
            z_range=self.z_range,
        )
        graph = build_interactive_segm_graph(ctx).compile()
        state = graph.invoke(
            InteractiveSegmState(main_win=self.mainWin),
            runnable_config_from_main_win(self.mainWin),
        )
        t1 = time.perf_counter()
        self.finished.emit(state.lab, t1 - t0)


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

    @worker_exception_handler
    def run(self):
        from cellacdc.workflow.adapters import interactive_video_segm_context_from_worker
        from cellacdc.workflow.pipelines.interactive_video_segm import (
            build_interactive_video_segm_graph,
        )
        from cellacdc.workflow.state import InteractiveVideoSegmState

        t0 = time.perf_counter()
        ctx = interactive_video_segm_context_from_worker(self)
        graph = build_interactive_video_segm_graph(ctx).compile()
        graph.invoke(
            InteractiveVideoSegmState(pos_data=self.posData),
        )
        t1 = time.perf_counter()
        self.finished.emit(t1 - t0)


class PostProcessSegmWorker(QObject):
    def __init__(
        self,
        postProcessKwargs,
        customPostProcessGroupedFeatures,
        customPostProcessFeatures,
        mainWin,
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

        self.logger.log("Post-process segmentation process started.")
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
                lab = data_dict["labels"]
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
                        self.customPostProcessFeatures,
                    )
                if visited:
                    posData.allData_li[frame_i]["labels"] = processed_lab
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
            "Input segmentation masks are not 3D. You can use this utility "
            "only on 3D z-stack data or 4D z-stack over time data."
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

            self.mainWin.infoText = f"Select <b>3D segmentation file to connect</b>"
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
                if posData.segm_data.ndim == 3:
                    posData.segm_data = posData.segm_data[np.newaxis]

                self.logger.log("Connecting 3D objects...")

                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                connectedSegmData = np.zeros_like(posData.segm_data)
                for frame_i, lab in enumerate(posData.segm_data):
                    if lab.ndim != 3:
                        self.criticalSegmIsNot3D()

                    connected_lab = core.connect_3Dlab_zboundaries(lab)
                    connectedSegmData[frame_i] = connected_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log("Saving connected 3D segmentation file...")
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f"{segmFilename}_{appendedName}.npz"
                connectedSegmData = np.squeeze(connectedSegmData)
                io.savez_compressed(newSegmFilepath, connectedSegmData)

                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)


class DelObjectsOutsideSegmROIWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    debug = Signal(object)

    def __init__(
        self,
        segm_roi_endname: os.PathLike,
        segm_data: np.ndarray,
        images_path: os.PathLike,
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

        self.progress.emit(f"Deleting objects outside of selected ROIs...")
        cleared_segm_data, delIDs = transformation.del_objs_outside_segm_roi(
            segm_roi_data, self.segm_data
        )

        self.finished.emit((self, cleared_segm_data, delIDs))


class MagicPromptsWorker(QObject):
    def __init__(
        self,
        posData,
        image,
        df_points,
        model,
        model_segment_kwargs,
        image_origin=(0, 0, 0),
        global_image=None,
    ):
        QObject.__init__(self)

        self.signals = signals()
        self.posData = posData
        self.image = image
        if global_image is not None:
            self.global_image = global_image
        else:
            self.global_image = image
        self.df_points = df_points
        self.image_origin = image_origin
        self.model = model
        self.model_segment_kwargs = model_segment_kwargs

    @worker_exception_handler
    def run(self):
        from cellacdc.segmenters_promptable import utils

        for row in self.df_points.itertuples():
            prompt_id = row.id
            point = (row.z, row.y, row.x)
            print(f"Adding point prompt {point} with id = {prompt_id}...")
            parent_obj_id = row.Cell_ID if row.Cell_ID == prompt_id else 0
            self.model.add_prompt(
                prompt=point,
                prompt_id=prompt_id,
                parent_obj_id=parent_obj_id,
                image=self.image,
                image_origin=self.image_origin,
                prompt_type="point",
            )

        lab_out = self.model.segment(
            self.global_image, lab=self.posData.lab, **self.model_segment_kwargs
        )
        edited_IDs = self.df_points["Cell_ID"].unique()

        lab_new, lab_union, lab_interesection = utils.insert_model_output_into_labels(
            self.posData.lab, lab_out, edited_IDs=edited_IDs
        )

        self.signals.finished.emit((lab_new, lab_union, lab_interesection))


class FillHolesInSegWorker(BaseWorkerUtil):
    sigAskAppendName = Signal(str)
    sigAborted = Signal()
    sigSelectSegmFiles = Signal(str, list)

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitAskAppendName(self, basename):
        self.mutex.lock()
        self.sigAskAppendName.emit(basename)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    @worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths
        lab_paths_dict = dict()
        unique_segm_files = set()
        tot_segm_files = 0
        for exp_path, pos_foldernames in expPaths.items():
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            for pos_folder in pos_foldernames:
                imgs_path = os.path.join(exp_path, pos_folder, "Images")
                lab_paths_dict[imgs_path] = self.endFilenameSegmTemp
                tot_segm_files += len(self.endFilenameSegmTemp)
                unique_segm_files.update(self.endFilenameSegmTemp)

        self.logger.info("Filling holes in segmentation masks...")
        abort = self.emitAskAppendName("/".join(unique_segm_files))
        if abort:
            self.sigAborted.emit()
            return
        self.signals.initProgressBar.emit(tot_segm_files)
        for images_path, segm_file_names in lab_paths_dict.items():
            for segm_file_name in segm_file_names:
                segm_data, segm_data_path = load.load_segm_file(
                    images_path, end_name_segm_file=segm_file_name, return_path=True
                )
                segm_data_shape = segm_data.shape
                segm_data_ndim = len(segm_data_shape)
                if segm_data_ndim == 2:
                    segm_data = segm_data[np.newaxis, np.newaxis, ...]
                elif segm_data_ndim == 3:
                    segm_data = segm_data[np.newaxis, ...]
                elif segm_data_ndim == 4:
                    segm_data = segm_data
                else:
                    raise NotImplementedError("This ndim is not supported!")
                for i, stack in enumerate(segm_data):
                    for j, lab in enumerate(stack):
                        segm_data[i, j] = core.fill_holes_in_segmentation(lab)

                segm_data_save_path = segm_data_path.replace(
                    segm_file_name, f"{segm_file_name}{self.appendedName}"
                )
                io.savez_compressed(segm_data_save_path, segm_data)
                self.signals.progressBar.emit(1)
        self.signals.finished.emit(self)

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    signals,
    workerLogger,
    worker_exception_handler,
)

