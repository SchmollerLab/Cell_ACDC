"""Qt view adapter for segmenting lost IDs."""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QMutex, QThread, QWaitCondition

from cellacdc import apps, workers
from cellacdc.plot import imshow

from .segmentation import Segmentation
from .frame_navigation import FrameNavigation


class SegForLostIds(Segmentation, FrameNavigation):
    """Extracted from guiWin."""

    def SegForLostIDsSetSettings(self):

        try:
            prev_model = str(self.df_settings.at["SegForLostIDsModel", "value"])
        except KeyError:
            prev_model = None
        win = apps.QDialogSelectModel(parent=self, customFirst=prev_model)
        win.exec_()
        if win.cancel:
            self.logger.info("Seg for lost IDs cancelled.")
            return
        base_model_name = win.selectedModel

        if base_model_name:
            self.df_settings.at["SegForLostIDsModel", "value"] = base_model_name
            self.df_settings.to_csv(self.settings_csv_path)

        model_name = "local_seg"

        idx = self.modelNames.index(model_name)
        acdcSegment = self.acdcSegment_li[idx]

        try:
            if acdcSegment is None or base_model_name != self.local_seg_base_model_name:
                self.logger.info(f"Importing {base_model_name}...")
                acdcSegment = utils.import_segment_module(base_model_name)
                self.acdcSegment_li[idx] = acdcSegment
                self.local_seg_base_model_name = base_model_name
        except (IndexError, ImportError, KeyError) as e:
            self.logger.error(f"Error importing {base_model_name}: {e}")
            return

        extra_params = [
            "overlap_threshold",
            "padding",
            "size_perc_diff",
            "distance_filler_growth",
            "max_iterations",
            "allow_only_tracked_cells",
        ]

        extra_types = [float, float, float, float, int, bool]

        extra_defaults = [0.5, 0.8, 0.3, 1.0, 2, False]

        extra_desc = [
            "Overlap threshold with other already segemented cells over which newly segmented cells are discarded",
            "Padding of the box used for new segmentation around the segmentation from the previous frame",
            "Relative size difference acceptable compared to previous frames",
            """Cells which are already segmented are filled with random noise sampled from background 
                    to ensure that they don't get segmented again. 
                    This parameter controls the additional padding around the already segmented cells.""",
            """The algorithm will try and segment the maximum amount 
                    of cells in the image by running the model several 
                    times and filling new found cells with background noise. 
                    How many of these iterations should be run?""",
            "If no new cell IDs should be permitted (based on real time tracking)",
        ]

        extra_ArgSpec = []
        for i, param in enumerate(extra_params):
            param = ArgSpec(
                name=param,
                default=extra_defaults[i],
                type=extra_types[i],
                desc=extra_desc[i],
                docstring="",
            )

            extra_ArgSpec.append(param)

        init_params, segment_params = utils.getModelArgSpec(acdcSegment)
        segment_params = [arg for arg in segment_params if arg[0] != "diameter"]

        extraParamsTitle = "Settings for local segmentation"
        win = self.initSegmModelParams(
            base_model_name,
            acdcSegment,
            init_params,
            segment_params,
            extraParams=extra_ArgSpec,
            extraParamsTitle=extraParamsTitle,
            initLastParams=True,
            ini_filename="segmentation_for_lostIDs.ini",
        )

        if win is None:
            self.logger.info("Segmentation for lost IDs cancelled.")
            return

        init_kwargs_new = {}
        args_new = {}
        for key, val in win.init_kwargs.items():
            if key in extra_params:
                args_new[key] = val
            else:
                init_kwargs_new[key] = val

        for key, val in win.extra_kwargs.items():
            if key in extra_params:
                args_new[key] = val

        self.SegForLostIDsSettings = {
            "win": win,
            "init_kwargs_new": init_kwargs_new,
            "args_new": args_new,
            "base_model_name": base_model_name,
        }

    def SegForLostIDsWorkerAskInstallGPU(self, model_name, use_gpu):
        result = utils.check_gpu_available(model_name, use_gpu, qparent=self)
        self.SegForLostIDsWorker.gpu_go = result
        dont_force_cpu = utils.check_gpu_available(
            model_name, use_gpu, do_not_warn=True
        )
        self.SegForLostIDsWorker.dont_force_cpu = dont_force_cpu
        self.SegForLostIDsWaitCond.wakeAll()

    def SegForLostIDsWorkerAskInstallModel(self, model_name):
        utils.check_install_package(model_name)
        self.SegForLostIDsWaitCond.wakeAll()

    def SegForLostIDsWorkerFinished(self):
        self.updateAllImages()
        self.update_rp()
        self.store_data(autosave=True)
        self.setFrameNavigationDisabled(disable=False, why="Segmentation for lost IDs")

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

    def onSegForLostInit(self):
        self.logger.info("Settings for segmentation for lost IDs not set.")
        self.SegForLostIDsSetSettings()
        self.SegForLostIDsWaitCond.wakeAll()

    def onSigGetData(self, waitcond, debug=False):
        self.get_data(debug=debug)
        waitcond.wakeAll()

    def onSigStoreData(
        self,
        waitcond,
        pos_i=None,
        enforce=True,
        debug=False,
        mainThread=True,
        autosave=True,
        store_cca_df_copy=False,
    ):
        self.store_data(
            pos_i=pos_i,
            enforce=enforce,
            debug=debug,
            mainThread=mainThread,
            autosave=autosave,
            store_cca_df_copy=store_cca_df_copy,
        )
        waitcond.wakeAll()

    def onSigStoreDataSegForLostIDsWorker(self, autosave):
        self.onSigStoreData(self.SegForLostIDsWaitCond, autosave=autosave)

    def onSigTrackManuallyAddedObjectSegForLostIDsWorker(
        self, added_IDs, isNewID, wl_update, wl_track_og_curr
    ):
        self.trackManuallyAddedObject(
            added_IDs, isNewID, wl_update=wl_update, wl_track_og_curr=wl_track_og_curr
        )
        self.SegForLostIDsWaitCond.wakeAll()

    def onSigUpdateRP(
        self,
        waitcond,
        draw=True,
        debug=False,
        update_IDs=True,
        wl_update=True,
        wl_track_og_curr=False,
    ):
        self.update_rp(
            draw=draw,
            debug=debug,
            update_IDs=update_IDs,
            wl_update=wl_update,
            wl_track_og_curr=wl_track_og_curr,
        )
        waitcond.wakeAll()

    def onSigUpdateRPSegForLostIDsWorker(self, wl_update, wl_track_og_curr):
        self.onSigUpdateRP(
            self.SegForLostIDsWaitCond,
            wl_update=wl_update,
            wl_track_og_curr=wl_track_og_curr,
        )

    def segForLostIDsButtonClicked(self):

        self.setFrameNavigationDisabled(disable=True, why="Segmentation for lost IDs")
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            self.logger.info("Segmentation for lost IDs not available on first frame.")
            self.setFrameNavigationDisabled(
                disable=False, why="Segmentation for lost IDs"
            )
            return
        self.storeUndoRedoStates(False)
        self.progressWin = apps.QDialogWorkerProgress(
            title="Segmenting for lost IDs",
            parent=self,
            pbarDesc=f"Segmenting for lost IDs...",
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        self.startSegForLostIDsWorker()

    def showImageDebug(self, img):
        imshow(img)

    def startSegForLostIDsWorker(self):
        self.SegForLostIDsMutex = QMutex()
        self.SegForLostIDsWaitCond = QWaitCondition()
        self._thread = QThread()

        # Initialize the worker with mutex and wait condition
        self.SegForLostIDsWorker = workers.SegForLostIDsWorker(
            self, self.SegForLostIDsMutex, self.SegForLostIDsWaitCond
        )

        # Connect the worker's signal to the main thread's slot
        self.SegForLostIDsWorker.sigAskInit.connect(self.onSegForLostInit)
        self.SegForLostIDsWorker.sigAskInstallModel.connect(
            self.SegForLostIDsWorkerAskInstallModel
        )
        self.SegForLostIDsWorker.sigshowImageDebug.connect(self.showImageDebug)

        self.SegForLostIDsWorker.sigSegForLostIDsWorkerAskInstallGPU.connect(
            self.SegForLostIDsWorkerAskInstallGPU
        )

        self.SegForLostIDsWorker.sigStoreData.connect(
            self.onSigStoreDataSegForLostIDsWorker
        )
        self.SegForLostIDsWorker.sigUpdateRP.connect(
            self.onSigUpdateRPSegForLostIDsWorker
        )
        # self.SegForLostIDsWorker.sigGetData.connect(self.onSigGetDataSegForLostIDsWorker)
        # self.SegForLostIDsWorker.sigGet2Dlab.connect(self.onSigGet2DlabSegForLostIDsWorker)
        # self.SegForLostIDsWorker.sigGetTrackedLostIDs.connect(self.onSigGetTrackedSegForLostIDsWorker)
        # self.SegForLostIDsWorker.sigGetBrushID.connect(self.onSigGetBrushIDSegForLostIDsWorker)
        self.SegForLostIDsWorker.sigTrackManuallyAddedObject.connect(
            self.onSigTrackManuallyAddedObjectSegForLostIDsWorker
        )

        # Move the worker to the thread
        self.SegForLostIDsWorker.moveToThread(self._thread)

        # Manage thread lifecycle
        self.SegForLostIDsWorker.signals.finished.connect(self._thread.quit)
        self.SegForLostIDsWorker.signals.finished.connect(
            self.SegForLostIDsWorker.deleteLater
        )
        self._thread.finished.connect(self._thread.deleteLater)

        # Connect other worker signals to the appropriate slots
        self.SegForLostIDsWorker.signals.finished.connect(
            self.SegForLostIDsWorkerFinished
        )
        self.SegForLostIDsWorker.signals.progress.connect(self.workerProgress)
        self.SegForLostIDsWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.SegForLostIDsWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.SegForLostIDsWorker.signals.critical.connect(self.workerCritical)

        # Start the thread and worker
        self._thread.started.connect(self.SegForLostIDsWorker.run)
        self._thread.start()
