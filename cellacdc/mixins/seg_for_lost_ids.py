"""Qt view adapter for segmenting lost IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from qtpy.QtCore import QMutex, QThread, QWaitCondition

from cellacdc import apps, workers
from cellacdc.plot import imshow


class SegForLostIdsView:
    """Qt-facing adapter around lost-ID segmentation commands."""

    """Headless settings and launch rules for lost-ID segmentation."""

    settings_key = 'SegForLostIDsModel'
    worker_model_name = 'local_seg'

    def previous_model_name(self, df_settings) -> str | None:
        try:
            return str(df_settings.at[self.settings_key, 'value'])
        except KeyError:
            return None

    def should_persist_model_choice(self, base_model_name: str | None) -> bool:
        return bool(base_model_name)

    def extra_arg_specs(self) -> list[ArgSpec]:
        extra_params = (
            'overlap_threshold',
            'padding',
            'size_perc_diff',
            'distance_filler_growth',
            'max_iterations',
            'allow_only_tracked_cells',
        )
        extra_types = (float, float, float, float, int, bool)
        extra_defaults = (0.5, 0.8, 0.3, 1.0, 2, False)
        extra_desc = (
            (
                'Overlap threshold with other already segemented cells over '
                'which newly segmented cells are discarded'
            ),
            (
                'Padding of the box used for new segmentation around the '
                'segmentation from the previous frame'
            ),
            (
                'Relative size difference acceptable compared to previous '
                'frames'
            ),
            (
                'Cells which are already segmented are filled with random '
                'noise sampled from background to ensure that they do not get '
                'segmented again. This parameter controls the additional '
                'padding around the already segmented cells.'
            ),
            (
                'The algorithm will try and segment the maximum amount of '
                'cells in the image by running the model several times and '
                'filling new found cells with background noise. How many of '
                'these iterations should be run?'
            ),
            (
                'If no new cell IDs should be permitted '
                '(based on real time tracking)'
            ),
        )

        return [
            ArgSpec(
                name=name,
                default=default,
                type=arg_type,
                desc=desc,
                docstring='',
            )
            for name, default, arg_type, desc in zip(
                extra_params, extra_defaults, extra_types, extra_desc
            )
        ]

    def split_model_kwargs(
        self,
        init_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_param_names = {arg.name for arg in self.extra_arg_specs()}
        init_kwargs_new = {}
        args_new = {}

        for key, val in init_kwargs.items():
            if key in extra_param_names:
                args_new[key] = val
            else:
                init_kwargs_new[key] = val

        for key, val in extra_kwargs.items():
            if key in extra_param_names:
                args_new[key] = val

        return init_kwargs_new, args_new

    def settings_from_dialog(self, win, base_model_name: str):
        init_kwargs_new, args_new = self.split_model_kwargs(
            win.init_kwargs,
            win.extra_kwargs,
        )
        return SegForLostIdsSettings(
            win=win,
            init_kwargs_new=init_kwargs_new,
            args_new=args_new,
            base_model_name=base_model_name,
        )

    def can_start_from_frame(self, frame_i: int) -> bool:
        return frame_i > 0


    def __init__(self, host):
        object.__setattr__(self, 'host', host)
    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def SegForLostIDsSetSettings(self):

        prev_model = self.previous_model_name(self.df_settings)
        win = apps.QDialogSelectModel(parent=self.host, customFirst=prev_model)
        win.exec_()
        if win.cancel:
            self.logger.info('Seg for lost IDs cancelled.')
            return
        base_model_name = win.selectedModel

        if self.should_persist_model_choice(base_model_name):
            self.df_settings.at[
                self.settings_key, 'value'
            ] = base_model_name
            self.df_settings.to_csv(self.settings_csv_path)

        model_name = self.worker_model_name

        idx = self.modelNames.index(model_name)
        acdcSegment = self.acdcSegment_li[idx]

        try:
            if (
                acdcSegment is None
                or base_model_name != self.local_seg_base_model_name
            ):
                self.logger.info(f'Importing {base_model_name}...')
                acdcSegment = (
                    self.host.view_model.model_registry
                    .import_segmentation_module(base_model_name)
                )
                self.acdcSegment_li[idx] = acdcSegment
                self.local_seg_base_model_name = base_model_name
        except (IndexError, ImportError, KeyError) as e:
            self.logger.error(f'Error importing {base_model_name}: {e}')
            return

        extra_ArgSpec = self.extra_arg_specs()

        init_params, segment_params = (
            self.host.view_model.model_registry.model_arg_specs(acdcSegment)
        )
        segment_params = [arg for arg in segment_params if arg[0] != 'diameter']

        extraParamsTitle = 'Settings for local segmentation'
        win = self.initSegmModelParams(
            base_model_name, acdcSegment, init_params, segment_params,
            extraParams=extra_ArgSpec, extraParamsTitle=extraParamsTitle,
            initLastParams=True, ini_filename='segmentation_for_lostIDs.ini',
        )

        if win is None:
            self.logger.info('Segmentation for lost IDs cancelled.')
            return

        settings = self.settings_from_dialog(win, base_model_name)
        self.SegForLostIDsSettings = {
            'win': settings.win,
            'init_kwargs_new': settings.init_kwargs_new,
            'args_new': settings.args_new,
            'base_model_name': settings.base_model_name,
        }

    def segForLostIDsButtonClicked(self):

        why = 'Segmentation for lost IDs'
        self.setFrameNavigationDisabled(disable=True, why=why)
        posData = self.data[self.pos_i]
        if not self.can_start_from_frame(posData.frame_i):
            self.logger.info(
                'Segmentation for lost IDs not available on first frame.'
            )
            self.setFrameNavigationDisabled(disable=False, why=why)
            return
        self.storeUndoRedoStates(False)
        self.progressWin = apps.QDialogWorkerProgress(
            title='Segmenting for lost IDs',
            parent=self.host,
            pbarDesc='Segmenting for lost IDs...',
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        self.startSegForLostIDsWorker()

    def onSegForLostInit(self):
        self.logger.info('Settings for segmentation for lost IDs not set.')
        self.SegForLostIDsSetSettings()
        self.SegForLostIDsWaitCond.wakeAll()

    def SegForLostIDsWorkerAskInstallModel(self, model_name):
        self.host.view_model.model_registry.check_install_package(model_name)
        self.SegForLostIDsWaitCond.wakeAll()

    def startSegForLostIDsWorker(self):
        self.SegForLostIDsMutex = QMutex()
        self.SegForLostIDsWaitCond = QWaitCondition()
        self._thread = QThread()

        # Initialize the worker with mutex and wait condition
        self.SegForLostIDsWorker = workers.SegForLostIDsWorker(
            self.host, self.SegForLostIDsMutex, self.SegForLostIDsWaitCond
        )

        # Connect the worker's signal to the main thread's slot
        self.SegForLostIDsWorker.sigAskInit.connect(self.onSegForLostInit)
        self.SegForLostIDsWorker.sigAskInstallModel.connect(
            self.SegForLostIDsWorkerAskInstallModel
        )
        self.SegForLostIDsWorker.sigshowImageDebug.connect(
            self.showImageDebug
        )

        self.SegForLostIDsWorker.sigSegForLostIDsWorkerAskInstallGPU.connect(
            self.SegForLostIDsWorkerAskInstallGPU
        )

        self.SegForLostIDsWorker.sigStoreData.connect(
            self.onSigStoreDataSegForLostIDsWorker
        )
        self.SegForLostIDsWorker.sigUpdateRP.connect(
            self.onSigUpdateRPSegForLostIDsWorker
        )
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

    def SegForLostIDsWorkerAskInstallGPU(self, model_name, use_gpu):
        result = self.host.view_model.model_registry.check_gpu_available(
            model_name, use_gpu, qparent=self.host
        )
        self.SegForLostIDsWorker.gpu_go = result
        dont_force_cpu = self.host.view_model.model_registry.check_gpu_available(
            model_name, use_gpu, do_not_warn=True
        )
        self.SegForLostIDsWorker.dont_force_cpu = dont_force_cpu
        self.SegForLostIDsWaitCond.wakeAll()

    def onSigStoreDataSegForLostIDsWorker(self, autosave):
        self.onSigStoreData(
            self.SegForLostIDsWaitCond, autosave=autosave
        )

    def onSigUpdateRPSegForLostIDsWorker(self, wl_update, wl_track_og_curr):
        self.onSigUpdateRP(
            self.SegForLostIDsWaitCond,
            wl_update=wl_update,
            wl_track_og_curr=wl_track_og_curr,
        )

    def onSigTrackManuallyAddedObjectSegForLostIDsWorker(
        self,
        added_IDs,
        isNewID,
        wl_update,
        wl_track_og_curr,
    ):
        self.trackManuallyAddedObject(
            added_IDs,
            isNewID,
            wl_update=wl_update,
            wl_track_og_curr=wl_track_og_curr,
        )
        self.SegForLostIDsWaitCond.wakeAll()

    def onSigStoreData(
            self, waitcond, pos_i=None, enforce=True, debug=False,
            mainThread=True, autosave=True, store_cca_df_copy=False
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

    def onSigUpdateRP(self, waitcond, draw=True, debug=False, update_IDs=True,
                  wl_update=True, wl_track_og_curr=False):
        self.update_rp(draw=draw, debug=debug, update_IDs=update_IDs,
                        wl_update=wl_update, wl_track_og_curr=wl_track_og_curr)
        waitcond.wakeAll()

    def onSigGetData(self, waitcond, debug=False):
        self.get_data(debug=debug)
        waitcond.wakeAll()

    def SegForLostIDsWorkerFinished(self):
        self.updateAllImages()
        self.update_rp()
        self.store_data(autosave=True)
        self.setFrameNavigationDisabled(
            disable=False, why='Segmentation for lost IDs'
        )

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

    def showImageDebug(self, img):
        imshow(img)