"""Segmentation from the main GUI: repeat segmentation, model execution,
post-processing and the segment lost IDs tool. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
import os
import traceback
from contextlib import contextmanager
import json
import numpy as np
from qtpy.QtCore import QTimer, QThread, QMutex, QWaitCondition
from qtpy.QtWidgets import QAction
from cellacdc import exception_handler
from cellacdc import prompts, apps, workers, html_utils
from cellacdc import myutils
from cellacdc import widgets
from cellacdc import printl
from cellacdc import seg_for_lost_IDs_settings_path
from cellacdc.myutils import ArgSpec
from cellacdc.plot import imshow


class SegForLostIDsGuiElements:
    """Segment lost IDs tool and the handshake with its worker."""

    def SegForLostIDsSetSettings(self):
        posData = self.data[self.pos_i]
        displayed_input_label = 'Displayed image'

        last_recipe_json_path = os.path.join(
            seg_for_lost_IDs_settings_path, '.last_segmentation_for_lostIDs_recipe.json'
        )

        try:
            prev_models = [
                model.strip() for model in str(
                    self.df_settings.at['SegForLostIDsModel', 'value']
                ).split(',') if model.strip()
            ]
        except KeyError:
            prev_models = []

        has_last_recipe = bool(prev_models) and os.path.exists(last_recipe_json_path)
        seg_for_lost_ids_info = (
            '<b>Segmentation for lost IDs settings</b><br><br>'
            'Use this dialog to define the segmentation workflow used for '
            'resegmenting local neighborhood lost IDs. Other already segmented cells are filled '
            'with background, which makes even dimm cells seem bright after '
            'rescaling before resegmentation. This is especially usefull for '
            'cells which have varying intensities over time, like FUCCI cells. <br><br>'
            '<b>How model selection works</b><br>'
            '- You can select one model or multiple models.<br>'
            '- In multi-selection mode you can include the same model multiple '
            'times (for example, model A, then model B, then model A again).<br>'
            '- After confirming, you can reorder the selected models. The order '
            'is the execution order. '
            '- You then will be asked to set model parameters in the order selected. <br><br>'
            '    - Pay special attention to the additional "Settings for local '
            'segmentation" section, here you can for example select any image as input.<br><br>'
            '<b>Load last selection...</b><br>'
            'Restores only the list of selected model names (the recipe order '
            'selection), then lets you continue configuring parameters.<br><br>'
            '<b>Load last recipe...</b><br>'
            'Loads the complete saved recipe from disk, including model order and '
            'all model-specific settings (when available).<br><br>'
            '<b>Choose from saved recipes...</b><br>'
            'Lets you pick any saved recipe JSON file from disk and load it.<br><br>'
            '<b>Add custom model...</b><br>'
            'Lets you register an additional local custom model and include it in '
            'the sequence.<br><br>'
            'Tip: if you want to run the same model twice with different '
            'parameters, add it twice and configure each step independently.'
        )
        win = apps.QDialogSelectModel(
            parent=self,
            allowMultiSelection=True,
            lastSelection=prev_models,
            addSelectLastSelectionButton=bool(prev_models),
            addSelectLastRecipeButton=has_last_recipe,
            addSelectSavedRecipeButton=True,
            add_save_func=True,
            recipes_path=seg_for_lost_IDs_settings_path,
            recipe_prefix='segmentation_for_lostIDs_recipe',
            recipe_ext_label='JSON',
            recipe_ext='json',
            custom_title='Select model(s) for segmentation of lost IDs',
            info_label=seg_for_lost_ids_info,
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Seg for lost IDs cancelled.')
            return

        save_name = None
        if getattr(win, 'selectionSaveNameFormatted', False):
            save_name = win.selectionSaveNameFormatted

        recipe_to_load_path = None
        if getattr(win, 'loadSavedRecipe', False):
            recipe_to_load_path = win.selectedRecipeFilepath
            self.logger.info(
                'Loading selected segmentation recipe for lost IDs...'
            )
        elif getattr(win, 'loadLastRecipe', False):
            recipe_to_load_path = last_recipe_json_path
            self.logger.info('Loading last segmentation recipe for lost IDs...')

        if recipe_to_load_path is not None:
            try:
                with open(recipe_to_load_path, 'r') as f:
                    recipe_data = json.load(f)
                model_settings = []
                for entry in recipe_data['models']:
                    model_settings.append({
                        'win': None,
                        'init_kwargs_new': entry['init_kwargs_new'],
                        'args_new': entry['args_new'],
                        'base_model_name': entry['base_model_name'],
                        'init_kwargs': entry.get('init_kwargs', {}),
                        'model_kwargs': entry.get('model_kwargs', {}),
                        'preproc_recipe': entry.get('preproc_recipe', None),
                        'applyPostProcessing': entry.get('applyPostProcessing', False),
                        'standardPostProcessKwargs': entry.get('standardPostProcessKwargs', {}),
                        'customPostProcessFeatures': entry.get('customPostProcessFeatures', None),
                        'customPostProcessGroupedFeatures': entry.get('customPostProcessGroupedFeatures', None),
                    })
                for model_name in dict.fromkeys(
                    settings['base_model_name'] for settings in model_settings
                ):
                    if self.prepareModelExecution(model_name) is None:
                        self.logger.info(
                            'Loading segmentation recipe cancelled.'
                        )
                        return
                self.SegForLostIDsSettings = {'models_settings': model_settings}
                self.releaseSegForLostIDsModels()
                # Restore model names in settings
                restored_models = [
                    'Automatic thresholding'
                    if m['base_model_name'] == 'thresholding'
                    else m['base_model_name']
                    for m in model_settings
                ]
                self.df_settings.at['SegForLostIDsModel', 'value'] = (
                    ', '.join(restored_models)
                )
                self.df_settings.to_csv(self.settings_csv_path)
                self.logger.info('Segmentation recipe loaded successfully.')
            except Exception as e:
                self.logger.error(
                    f'Failed to load segmentation recipe: {e}'
                )
            return

        selected_models = win.selectedModel
        if isinstance(selected_models, str):
            selected_models = [selected_models]

        if not selected_models:
            self.logger.info('Seg for lost IDs cancelled.')
            return

        if selected_models:
            self.df_settings.at['SegForLostIDsModel', 'value'] = (
                ', '.join(selected_models)
            )
            self.df_settings.to_csv(self.settings_csv_path)

        all_extra_params = [
            'image_channel_name',
            'overlap_threshold',
            'fill_other_cells_with_background',
            'padding',
            'size_perc_diff',
            'distance_filler_growth',
            'allow_only_tracked_cells',
        ]
        extra_types = {
            'overlap_threshold': float,
            'fill_other_cells_with_background': bool,
            'padding': float,
            'size_perc_diff': float,
            'distance_filler_growth': float,
            'allow_only_tracked_cells': bool,
            'image_channel_name': str,
        }
        extra_defaults = {
            'overlap_threshold': 0.5,
            'fill_other_cells_with_background': True,
            'padding': 0.8,
            'size_perc_diff': 0.3,
            'distance_filler_growth': 1.,
            'allow_only_tracked_cells': False,
            'image_channel_name': displayed_input_label,
        }
        extra_desc = {
            'overlap_threshold': (
                'Overlap threshold with other already segemented cells '
                'over which newly segmented cells are discarded'
            ),
            'fill_other_cells_with_background' : (
                'Fill already segmented cells with background'
            ),
            'padding': (
                'Padding of the box used for new segmentation around the '
                'segmentation from the previous frame'
            ),
            'size_perc_diff': (
                'Relative size difference acceptable compared to previous '
                'frames'
            ),
            'distance_filler_growth': (
                'Cells which are already segmented are filled with random '
                'noise sampled from background to ensure that they do not '
                'get segmented again. This parameter controls the additional '
                'padding around the already segmented cells.'
            ),
            'allow_only_tracked_cells': (
                'If no new cell IDs should be permitted '
                '(based on real time tracking)'
            ),
            'image_channel_name': (
                'Image channel used as model input. '
                'Select "Displayed image" to use exactly what is currently '
                'shown in the viewer, or select a specific fluorescence '
                'channel.'
            ),
        }

        model_settings = []
        remembered_extra_args = {}
        for model_idx, selected_model_name in enumerate(selected_models):
            model_name = selected_model_name
            if model_name == 'Automatic thresholding':
                model_name = 'thresholding'
            # Read the parameters (from the model's isolated env if used)
            model_spec = self.prepareModelExecution(model_name)
            if model_spec is None:
                self.logger.info('Segmentation for lost IDs cancelled.')
                return
            init_params, segment_params, url = model_spec

            extra_params = all_extra_params

            available_fluo_channels = [
                ch for ch in posData.chNames if ch != self.user_ch_name
            ]
            channel_options = [displayed_input_label, *available_fluo_channels]

            class _SegForLostIDsInputChannelType:
                values = channel_options

            extra_types['image_channel_name'] = _SegForLostIDsInputChannelType

            extra_ArgSpec = []
            for param in extra_params:
                param_arg = ArgSpec(
                    name=param,
                    default=extra_defaults[param],
                    type=extra_types[param],
                    desc=extra_desc[param],
                    docstring=''
                )
                extra_ArgSpec.append(param_arg)

            segment_params = [
                arg for arg in segment_params
            ] 

            initLastParams = True
            if model_name == 'thresholding':
                win_thresh = apps.QDialogAutomaticThresholding(
                    parent=self, isSegm3D=self.isSegm3D
                )
                win_thresh.exec_()
                if win_thresh.cancel:
                    for selected in selected_models:
                        name = 'thresholding' if selected == 'Automatic thresholding' else selected
                        self.getModelExecution(name).cancel_preparation()
                    self.logger.info('Segmentation for lost IDs cancelled.')
                    return
                self.model_kwargs = win_thresh.segment_kwargs
                thresh_method = self.model_kwargs['threshold_method']
                gauss_sigma = self.model_kwargs['gauss_sigma']
                segment_params = myutils.insertModelArgSpec(
                    segment_params, 'threshold_method', thresh_method
                )
                segment_params = myutils.insertModelArgSpec(
                    segment_params, 'gauss_sigma', gauss_sigma
                )
                initLastParams = False

            extraParamsTitle = (
                f'Settings for local segmentation '
                f'({model_idx + 1}/{len(selected_models)})'
            )
            win = self.initSegmModelParams(
                model_name, url, init_params, segment_params,
                extraParams=extra_ArgSpec,
                extraParamsTitle=extraParamsTitle,
                initLastParams=initLastParams,
                ini_filename='segmentation_for_lostIDs.ini',
            )

            if win is None:
                for selected in selected_models:
                    name = 'thresholding' if selected == 'Automatic thresholding' else selected
                    self.getModelExecution(name).cancel_preparation()
                self.logger.info('Segmentation for lost IDs cancelled.')
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
                    if key == 'image_channel_name':
                        init_kwargs_new[key] = val
                    else:
                        args_new[key] = val

            for key, val in remembered_extra_args.items():
                if key == 'image_channel_name':
                    init_kwargs_new.setdefault(key, val)
                    continue
                args_new.setdefault(key, val)

            if model_idx == 0:
                remembered_extra_args = args_new.copy()
                remembered_extra_args['image_channel_name'] = (
                    init_kwargs_new.get('image_channel_name', displayed_input_label)
                )

            model_settings.append({
                'win': win,
                'init_kwargs_new': init_kwargs_new,
                'args_new': args_new,
                'base_model_name': model_name,
                'init_kwargs': dict(win.init_kwargs),
                'model_kwargs': dict(win.model_kwargs),
                'preproc_recipe': win.preproc_recipe,
                'applyPostProcessing': win.applyPostProcessing,
                'standardPostProcessKwargs': win.standardPostProcessKwargs,
                'customPostProcessFeatures': win.customPostProcessFeatures,
                'customPostProcessGroupedFeatures': win.customPostProcessGroupedFeatures,
            })

        self.SegForLostIDsSettings = {
            'models_settings': model_settings,
        }
        self.releaseSegForLostIDsModels()

        # Persist recipe to disk so it survives across sessions
        try:
            recipe_data = {
                'models': [
                    {
                        'base_model_name': ms['base_model_name'],
                        'init_kwargs_new': ms['init_kwargs_new'],
                        'args_new': ms['args_new'],
                        'init_kwargs': ms['init_kwargs'],
                        'model_kwargs': ms['model_kwargs'],
                        'preproc_recipe': ms['preproc_recipe'],
                        'applyPostProcessing': ms['applyPostProcessing'],
                        'standardPostProcessKwargs': ms['standardPostProcessKwargs'],
                        'customPostProcessFeatures': ms['customPostProcessFeatures'],
                        'customPostProcessGroupedFeatures': ms['customPostProcessGroupedFeatures'],
                    }
                    for ms in model_settings
                ]
            }
            os.makedirs(seg_for_lost_IDs_settings_path, exist_ok=True)
            with open(last_recipe_json_path, 'w') as f:
                json.dump(recipe_data, f, indent=2, default=str)
            
            # save if requested
            if save_name:
                save_path = os.path.join(
                    seg_for_lost_IDs_settings_path, save_name
                )
                with open(save_path, 'w') as f:
                    json.dump(recipe_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f'Could not save recipe to disk: {e}')

    def segForLostIDsButtonClicked(self):

        self.setFrameNavigationDisabled(disable=True, why='Segmentation for lost IDs',warn=False)
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        if frame_i == 0:
            self.logger.info('Segmentation for lost IDs not available on first frame.')
            self.setFrameNavigationDisabled(disable=False, why='Segmentation for lost IDs')
                        
            if hasattr(self, "wait_worker_loop"):
                self.wait_worker_loop = None
            return

        prev_IDs = posData.allData_li[frame_i-1]['regionprops'].IDs_set
        tracked_lost_IDs = self.getTrackedLostIDs()

        missing_IDs = prev_IDs - posData.rp.IDs_set - set(tracked_lost_IDs)
        if not missing_IDs:
            # self.logger.info('No missing IDs, skipping Segmentation for lost IDs')
            self.setFrameNavigationDisabled(disable=False, why='Segmentation for lost IDs')

            if hasattr(self, "wait_worker_loop"):
                self.wait_worker_loop = None
            return

        self.storeUndoRedoStates(False)
        self.progressWin = apps.QDialogWorkerProgress(
            title='Segmenting for lost IDs', parent=self,
            pbarDesc=f'Segmenting for lost IDs...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)
        
        self.startSegForLostIDsWorker()

    def onSegForLostInit(self):
        self.logger.info('Settings for segmentation for lost IDs not set.')
        self.SegForLostIDsSetSettings()
        self._ackSegForLostIDsWorker('ask_init')

    def _ackSegForLostIDsWorker(self, ack_key):
        try:
            self.SegForLostIDsMutex.lock()
            self.SegForLostIDsWorker.ack(ack_key)
            self.SegForLostIDsWaitCond.wakeAll()
        except Exception:
            pass
        finally:
            try:
                self.SegForLostIDsMutex.unlock()
            except Exception:
                pass

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
        self.SegForLostIDsWorker.sigInitModel.connect(
            self.onSegForLostIDsInitModel
        )
        self.SegForLostIDsWorker.sigShowImageDebug.connect(
            self.showImageDebug
        )
        
        self.SegForLostIDsWorker.sigSegForLostIDsWorkerAskInstallGPU.connect(
            self.SegForLostIDsWorkerAskInstallGPU
        )

        self.SegForLostIDsWorker.sigStoreData.connect(
            self.onSigStoreDataSegForLostIDsWorker)
        self.SegForLostIDsWorker.sigUpdateRP.connect(
            self.onSigUpdateRPSegForLostIDsWorker)
        self.SegForLostIDsWorker.sigGetSegForLostIDsInputImg.connect(
            self.onSigGetInputImgSegForLostIDsWorker
        )
        self.SegForLostIDsWorker.sigTrackManuallyAddedObject.connect(
            self.onSigTrackManuallyAddedObjectSegForLostIDsWorker
            )
        # Move the worker to the thread
        self.SegForLostIDsWorker.moveToThread(self._thread)

        # Manage thread lifecycle
        self.SegForLostIDsWorker.signals.finished.connect(self._thread.quit)
        self.SegForLostIDsWorker.signals.finished.connect(self.SegForLostIDsWorker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        # Connect other worker signals to the appropriate slots
        self.SegForLostIDsWorker.signals.finished.connect(self.SegForLostIDsWorkerFinished)
        self.SegForLostIDsWorker.signals.progress.connect(self.workerProgress)
        self.SegForLostIDsWorker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.SegForLostIDsWorker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.SegForLostIDsWorker.signals.critical.connect(self.workerCritical)

        # Start the thread and worker
        self._thread.started.connect(self.SegForLostIDsWorker.run)
        self._thread.start()

    def onSegForLostIDsInitModel(self, model_idx, model_name, init_kwargs):
        """Provide the SegForLostIDsWorker with an initialized model.

        Models are cached per model slot and reused as long as the model,
        its init parameters and the position stay the same, so isolated
        workers are not restarted at every frame.
        """
        worker = self.SegForLostIDsWorker
        try:
            posData = self.data[self.pos_i]
            key = (
                model_name,
                json.dumps(init_kwargs, sort_keys=True, default=str),
                id(posData),
            )
            cached = self.segForLostIDsModels.get(model_idx)

            self.logger.info(f'Initializing {model_name} model...')
            replace = None
            if cached is not None and cached[1] == model_name:
                replace = cached[2]
            model = self.initSegmModel(
                model_name, init_kwargs, posData, replace=replace
            )
            if model is None:
                return
            self.segForLostIDsModels[model_idx] = (key, model_name, model)
            if cached is not None and replace is None:
                self.releaseSegmModel(cached[1], cached[2])
            worker.initModelResult = model
        except Exception as error:
            self.logger.exception(traceback.format_exc())
            worker.initModelError = error
        finally:
            self._ackSegForLostIDsWorker('init_model')

    def SegForLostIDsWorkerAskInstallGPU(self, model_name, use_gpu):
        worker = self.SegForLostIDsWorker
        worker.gpu_go = False
        worker.dont_force_cpu = False
        try:
            python_exec = self.getModelExecution(model_name).env_python
            cached = getattr(self, 'segForLostIDsGPUSettings', {})
            if (
                cached.get('model_name') == model_name
                and cached.get('use_gpu') == use_gpu
                and cached.get('python_exec') == python_exec
            ):
                worker.gpu_go = cached['gpu_go']
                worker.dont_force_cpu = cached['dont_force_cpu']
                return

            result = myutils.check_gpu_available(
                model_name, use_gpu, qparent=self, python_exec=python_exec
            )
            dont_force_cpu = myutils.check_gpu_available(
                model_name, use_gpu, do_not_warn=True, python_exec=python_exec
            )
            worker.gpu_go = result
            worker.dont_force_cpu = dont_force_cpu
            self.segForLostIDsGPUSettings = {
                'gpu_go': result,
                'dont_force_cpu': dont_force_cpu,
                'model_name': model_name,
                'use_gpu': use_gpu,
                'python_exec': python_exec,
            }
        except Exception:
            self.logger.exception('Could not check GPU availability for %s', model_name)
        finally:
            self._ackSegForLostIDsWorker('ask_install_gpu')

    def onSigStoreDataSegForLostIDsWorker(self, autosave):
        self.onSigStoreData(
            self.SegForLostIDsWaitCond, autosave=autosave)
        self._ackSegForLostIDsWorker('store_data')

    def onSigUpdateRPSegForLostIDsWorker(self, wl_update, wl_track_og_curr):
        self.onSigUpdateRP(self.SegForLostIDsWaitCond,
                           wl_update=wl_update, 
                           wl_track_og_curr=wl_track_og_curr)
        self._ackSegForLostIDsWorker('update_rp')

    def onSigTrackManuallyAddedObjectSegForLostIDsWorker(
        self, added_IDs, isNewID, wl_update, wl_track_og_curr
        ):
        assignments = self.trackManuallyAddedObject(
            added_IDs, isNewID, wl_update=wl_update, 
            wl_track_og_curr=wl_track_og_curr,
            clearAssignedObjsSecondStep=False
            )
        self.SegForLostIDsWorker.assignments = assignments
        self._ackSegForLostIDsWorker('track_manually_added')

    def onSigGetInputImgSegForLostIDsWorker(self, image_channel_name):
        displayed_input_label = 'Displayed image'
        posData = self.data[self.pos_i]

        if (
                not image_channel_name
                or image_channel_name == displayed_input_label
            ):
            img = self.getDisplayedImg1()
            self.SegForLostIDsWorker.inputImgForSegForLostIDs = img
            self._ackSegForLostIDsWorker('get_input_img')
            return

        self.getChData(requ_ch={image_channel_name})

        _, filename = self.getPathAndFilenameNoExtFromChName(image_channel_name, posData)
        fluo_data = posData.fluo_data_dict.get(filename)
        if posData.SizeT > 1:
            fluo_img_data = fluo_data[posData.frame_i]
        else:
            fluo_img_data = fluo_data

        self.SegForLostIDsWorker.inputImgForSegForLostIDs = fluo_img_data
        self._ackSegForLostIDsWorker('get_input_img')


    def onSigStoreData(
            self, waitcond, pos_i=None, enforce=True, debug=False,
            mainThread=True, autosave=True, store_cca_df_copy=False
        ):
        self.store_data(pos_i=pos_i, enforce=enforce, debug=debug, mainThread=mainThread,
            autosave=autosave, store_cca_df_copy=store_cca_df_copy)
        waitcond.wakeAll()

    def onSigUpdateRP(self, waitcond, 
            draw=True, debug=False, # og stuff
            assignments=None, deletionIDs=None, # very quick upates, rp labels are changed but rest is same
            specific_IDs=None, use_curr_view=False, use_bbox=False, preloaded_bbox=None, # for local updates to PR
            wl_update=True, wl_track_og_curr=False,wl_update_lab=False, # wl stuff
        ):
        self.update_rp(draw=True, debug=False, # og stuff
                       assignments=None, deletionIDs=None, # very quick upates, rp labels are changed but rest is same
                       specific_IDs=None, use_curr_view=False, use_bbox=False, preloaded_bbox=None, # for local updates to PR
                       wl_update=True, wl_track_og_curr=False,wl_update_lab=False, # wl stuff
                       )
        waitcond.wakeAll()

    def onSigGetData(self, waitcond, debug=False):
        self.get_data(debug=debug)
        waitcond.wakeAll()

    def SegForLostIDsWorkerFinished(self):
        self.updateAllImages()
        self.update_rp() # will update when updating segoforlostIDs
        self.store_data(autosave=True)
        self.setFrameNavigationDisabled(disable=False, why='Segmentation for lost IDs')

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        if hasattr(self, "wait_worker_loop") and self.wait_worker_loop is not None:
            self.wait_worker_loop.exit()

    def releaseSegForLostIDsModels(self):
        if hasattr(self, 'segForLostIDsModels'):
            models, self.segForLostIDsModels = self.segForLostIDsModels, {}
            for _, model_name, model in models.values():
                self.releaseSegmModel(model_name, model)


class SegmentationGuiElements:
    """Segmentation from the GUI: repeat segmentation of a frame or video,
    model initialization and execution bookkeeping, post-processing and
    the last segmented frame.
    """

    def startPostProcessSegmWorker(
            self, postProcessKwargs, customPostProcessGroupedFeatures, 
            customPostProcessFeatures
        ):
        self.thread = QThread()
        self.postProcessWorker = workers.PostProcessSegmWorker(
            postProcessKwargs, customPostProcessGroupedFeatures, 
            customPostProcessFeatures, self
        )
        
        self.postProcessWorker.moveToThread(self.thread)
        self.postProcessWorker.signals.finished.connect(self.thread.quit)
        self.postProcessWorker.signals.finished.connect(
            self.postProcessWorker.deleteLater
        )
        self.thread.finished.connect(self.thread.deleteLater)

        self.postProcessWorker.signals.finished.connect(
            self.postProcessSegmWorkerFinished
        )
        self.postProcessWorker.signals.progress.connect(self.workerProgress)
        self.postProcessWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.postProcessWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.postProcessWorker.signals.critical.connect(
            self.workerCritical
        )

        self.thread.started.connect(self.postProcessWorker.run)
        self.thread.start()

    def autoSegm_cb(self, checked):
        if checked:
            self.askSegmParam = True
            # Ask which model
            models = myutils.get_list_of_models()
            win = widgets.QDialogListbox(
                'Select model',
                'Select model to use for segmentation: ',
                models,
                multiSelection=False,
                parent=self,
                allowEmptySelection=False
            )
            win.exec_()
            if win.cancel:
                return
            model_name = win.selectedItemsText[0]
            self.segmModelName = model_name
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.updateAllImages()
            self.computeSegm()
            self.askSegmParam = False
        else:
            self.segmModelName = None

    def postProcessSegm(self, checked):
        if self.isSegm3D:
            SizeZ = max([posData.SizeZ for posData in self.data])
        else:
            SizeZ = None
        if checked:
            posData = self.data[self.pos_i]
            self.postProcessSegmWin = apps.PostProcessSegmDialog(
                posData, mainWin=self
            )
            self.postProcessSegmWin.sigClosed.connect(
                self.postProcessSegmWinClosed
            )
            self.postProcessSegmWin.sigValueChanged.connect(
                self.postProcessSegmValueChanged
            )
            self.postProcessSegmWin.sigEditingFinished.connect(
                self.postProcessSegmEditingFinished
            )
            self.postProcessSegmWin.sigApplyToAllFutureFrames.connect(
                self.postProcessSegmApplyToAllFutureFrames
            )
            self.postProcessSegmWin.show()
            self.postProcessSegmWin.valueChanged(None)
        else:
            self.postProcessSegmWin.close()
            self.postProcessSegmWin = None

    def postProcessSegmApplyToAllFutureFrames(
            self, postProcessKwargs, 
            customPostProcessGroupedFeatures, 
            customPostProcessFeatures
        ):
        proceed = self.warnEditingWithCca_df(
            'post-processing segmentation', update_images=False
        )
        if not proceed:
            self.logger.info('Post-processing segmentation cancelled.')
            return

        self.progressWin = apps.QDialogWorkerProgress(
            title='Post-processing segmentation', parent=self,
            pbarDesc=f'Post-processing segmentation masks...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        self.startPostProcessSegmWorker(
            postProcessKwargs, customPostProcessGroupedFeatures, 
            customPostProcessFeatures
        )

    def postProcessSegmEditingFinished(self):
        self.update_rp()
        self.store_data()
        self.updateAllImages()

    def postProcessSegmWorkerFinished(self):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None
        self.get_data()
        self.updateAllImages()
        self.titleLabel.setText('Post-processing segmentation done!', color='w')
        self.logger.info('Post-processing segmentation done!')

    def postProcessSegmWinClosed(self):
        self.postProcessSegmWin = None
        self.postProcessSegmAction.toggled.disconnect()
        self.postProcessSegmAction.setChecked(False)
        self.postProcessSegmAction.toggled.connect(self.postProcessSegm)

    def closePostProcessSegmWin(self):
        """Close the post-processing dialog (`postProcessSegm`) when leaving
        the modes that edit the segmentation (`changeMode`).

        Left open, its `apply` would run `warnEditingWithCca_df` on every
        parameter and frame change (`postProcessing`) in the other modes,
        where removing the annotations resets e.g. the live lineage tree.

        The masks are kept as they are shown (the annotations warning of
        `apply` was answered for them) and nothing is restored (the
        dialog's Cancel restores the masks of the current frame without
        asking, `PostProcessSegmDialog.undoChanges`). A preview that was
        not applied yet (e.g. the one shown when the dialog opens) changed
        only `posData.lab`: the regionprops are updated, so that the
        `store_data` of `changeMode` stores labels, regionprops and table
        of the same objects. The dialog's `sigClosed` unchecks the menu
        action (`postProcessSegmWinClosed`).
        """
        win = self.postProcessSegmWin
        if win is None:
            return

        self.update_rp()
        win.cancel = False
        win.close()

    def postProcessSegmValueChanged(self, lab, delObjs: dict):
        for delObj in delObjs.values():
            self.clearObjContour(obj=delObj, ax=0)
            self.clearObjContour(obj=delObj, ax=1)
            
        posData = self.data[self.pos_i]
        
        labelsToSkip = {}
        for ID in posData.IDs:
            if ID in delObjs:
                labelsToSkip[ID] = True
                continue
            
            restoreObj = self.postProcessSegmWin.origObjs[ID]
            self.addObjContourToContoursImage(obj=restoreObj, ax=0)
            self.addObjContourToContoursImage(obj=restoreObj, ax=1)
 
        # self.setAllTextAnnotations(labelsToSkip=labelsToSkip)

        posData.lab = lab
        self.setImageImg2()
        if self.annotSegmMasksCheckbox.isChecked():
            self.labelsLayerImg1.setImage(self.currentLab2D, autoLevels=False)
        if self.annotSegmMasksCheckboxRight.isChecked():
            self.labelsLayerRightImg.setImage(self.currentLab2D, autoLevels=False)

    def segmFrameCallback(self, action):
        if action == self.addCustomModelFrameAction:
            return
        
        idx = self.segmActions.index(action)
        model_name = self.modelNames[idx]
        self.repeatSegm(model_name=model_name, askSegmParams=True)

    def segmVideoCallback(self, action):
        if action == self.addCustomModelVideoAction:
            return

        posData = self.data[self.pos_i]
        win = apps.startStopFramesDialog(
            posData.SizeT, currentFrameNum=posData.frame_i+1
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Segmentation on multiple frames aborted.')
            return

        idx = self.segmActionsVideo.index(action)
        model_name = self.modelNames[idx]
        self.repeatSegmVideo(model_name, win.startFrame, win.stopFrame)

    def segmentToolActionTriggered(self):
        if self.segmModelName is None:
            win = apps.QDialogSelectModel(parent=self)
            win.exec_()
            if win.cancel:
                self.logger.info('Repeat segmentation cancelled.')
                return
            model_name = win.selectedModel
            self.repeatSegm(
                model_name=model_name, askSegmParams=True
            )
        else:
            self.repeatSegm(model_name=self.segmModelName)        

    def getModelExecution(self, model_name):
        """Return the `ModelExecution` of `model_name` (cached per session).

        It decides whether the model runs in its isolated environment (as a
        worker process) or in-process; see `service/model_execution.py`.
        Progress messages of `ModelExecution` are shown in the title, while 
        output of background steps (e.g., installations) goes to the log.
        """
        execution = self.modelExecutions.get(model_name)
        if execution is None:
            from cellacdc.service import model_execution, prompts as service_prompts
            
            def run_in_background(func, *args, **kwargs):
                return model_execution.run_in_background(
                    self, func, *args, log_func=self.logger.info, **kwargs
                )
            
            def ask(prompt):
                return service_prompts.QtHandler(parent=self).ask(prompt)

            execution = model_execution.ModelExecution(
                model_name, 
                run_in_background=run_in_background,
                prompt_func=ask, 
                log_func=self.setModelExecutionStatus,
            )
            self.modelExecutions[model_name] = execution
        return execution

    def getPromptModelExecution(self, model_name):
        execution = self.promptModelExecutions.get(model_name)
        if execution is None:
            from cellacdc.service import model_execution, prompts as service_prompts

            def run_in_background(func, *args, **kwargs):
                return model_execution.run_in_background(
                    self, func, *args, log_func=self.logger.info, **kwargs
                )

            execution = model_execution.ModelExecution(
                model_name,
                run_in_background=run_in_background,
                prompt_func=lambda prompt: service_prompts.QtHandler(
                    parent=self
                ).ask(prompt),
                log_func=self.setModelExecutionStatus,
                model_kind='promptable',
            )
            self.promptModelExecutions[model_name] = execution
        return execution

    def setModelExecutionStatus(self, text):
        self.logger.info(text)
        self.titleLabel.setText(text, color=self.titleColor)
        # Repaint now, the next step may block the event loop (e.g., import)
        # self.titleLabel.repaint()

    @contextmanager
    def modelExecutionBusy(self, text):
        """Disable the GUI and show `text` in the title while busy.

        The previous enabled state and cursor are restored afterwards, so 
        callers that already disabled the GUI (e.g., lost IDs) are not 
        re-enabled.
        """
        was_enabled = self.isEnabled()
        self.setDisabled(True)
        self.setModelExecutionStatus(text)
        try:
            yield
        except Exception:
            self.titleLabel.setText(f'{text} failed.', color='r')
            raise
        finally:
            if was_enabled:
                self.setDisabled(False)

    def prepareModelExecution(self, model_name):
        """Returns `(init_params, segment_params, url)` or None if cancelled."""
        execution = self.getModelExecution(model_name)
        if execution.spec is not None:
            return execution.spec
        with self.modelExecutionBusy(f'Preparing {model_name}...'):
            spec = execution.prepare()
        if spec is None:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
        return spec

    def initSegmModel(
            self, model_name, init_kwargs, posData=None, replace=None,
            *, is_rgb=False,
        ):
        """Return a model instance (a worker in isolated mode) or None.

        Running workers are reused (see `ModelExecution.init_model`). Pass 
        the instance that the new one replaces as `replace`, so that its 
        worker can be re-initialized instead of starting a new one.
        Pass `is_rgb=True` when preparing a second-channel RGB stack.
        """
        if posData is None:
            posData = self.data[self.pos_i]
        if replace is self.labelRoiModel:
            # Still used by the magic labeller
            replace = None
        execution = self.getModelExecution(model_name)
        with self.modelExecutionBusy(f'Initializing {model_name}...'):
            if execution.prepare() is None:
                return None
            return execution.init_model(
                init_kwargs, posData, replace=replace, is_rgb=is_rgb
            )

    def releaseSegmModel(self, model_name, model):
        """Give back a model from `initSegmModel` that is not used anymore.

        The worker keeps running for reuse; all workers stop on close.
        """
        if model is None or model is self.labelRoiModel:
            return
        execution = self.modelExecutions.get(model_name)
        if execution is not None:
            execution.release(model)

    def stopModelExecutions(self):
        if not hasattr(self, 'modelExecutions'):
            return
        for execution in self.modelExecutions.values():
            execution.stop()
        for execution in self.promptModelExecutions.values():
            execution.stop()
        self.segForLostIDsModels = {}
        self.modelExecutions = {}
        self.promptModelExecutions = {}

    def initSegmModelParams(
            self, model_name, url, init_params, segment_params, 
            is_label_roi=False, initLastParams=False,
            extraParams=None, extraParamsTitle=None,ini_filename=None

        ):
        posData = self.data[self.pos_i]        
        text_if_cancelled = 'Segmentation process cancelled.'
        out = prompts.init_segm_model_params(
            posData, model_name, init_params, segment_params, 
            help_url=url, qparent=self, init_last_params=initLastParams, 
            check_sam_embeddings=not is_label_roi, is_gui_caller=True,
            extraParams=extraParams,extraParamsTitle=extraParamsTitle,
            ini_filename=ini_filename,
        )
        if out.get('load_sam_embeddings', False):
            self.logger.info('Loading Segment Anything image embeddings...')
            for _posData in self.data:
                _posData.loadSamEmbeddings(logger_func=None)
            text_if_cancelled = 'SAM embeddings loaded.'
            
        win = out.get('win')
        if win is None:
            self.getModelExecution(model_name).cancel_preparation()
            self.logger.info(text_if_cancelled)
            self.titleLabel.setText(text_if_cancelled)
            return
        
        if win.cancel:
            self.getModelExecution(model_name).cancel_preparation()
            self.logger.info(text_if_cancelled)
            self.titleLabel.setText(text_if_cancelled)
            return
        
        if model_name != 'thresholding':
            self.model_kwargs = win.model_kwargs
        
        return win

    @exception_handler
    def repeatSegm(
            self, model_name='', askSegmParams=False, is_label_roi=False
        ):
        if model_name == 'thresholding':
            # thresholding model is stored as 'Automatic thresholding'
            # at line of code `models.append('Automatic thresholding')`
            model_name = 'Automatic thresholding'
        
        idx = self.modelNames.index(model_name)
        # Ask segm parameters if not already set
        # and not called by segmSingleFrameMenu (askSegmParams=False)
        if not askSegmParams:
            askSegmParams = self.model_kwargs is None

        self.downloadWin = apps.downloadModel(model_name, parent=self)
        self.downloadWin.download()

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        if model_name == 'Automatic thresholding':
            # Automatic thresholding is the name of the models as stored 
            # in self.modelNames, but the actual model is called thresholding
            # (see cellacdc/models/thresholding)
            model_name = 'thresholding'

        posData = self.data[self.pos_i]

        # Ask parameters if the user clicked on the action
        # Otherwise this function is called by "computeSegm" function and
        # we use loaded parameters
        if askSegmParams:
            self.segmModelName = model_name
            # Read all models parameters (from its isolated env if used)
            model_spec = self.prepareModelExecution(model_name)
            if model_spec is None:
                return
            init_params, segment_params, url = model_spec
            
            self.preproc_recipe = None
            initLastParams = True
            if model_name == 'thresholding':
                win = apps.QDialogAutomaticThresholding(
                    parent=self, isSegm3D=self.isSegm3D
                )
                win.exec_()
                if win.cancel:
                    self.getModelExecution(model_name).cancel_preparation()
                    return
                self.model_kwargs = win.segment_kwargs
                thresh_method = self.model_kwargs['threshold_method']
                gauss_sigma = self.model_kwargs['gauss_sigma']
                segment_params = myutils.insertModelArgSpec(
                    segment_params, 'threshold_method', thresh_method
                )
                segment_params = myutils.insertModelArgSpec(
                    segment_params, 'gauss_sigma', gauss_sigma
                )
                initLastParams = False
            
            win = self.initSegmModelParams(
                model_name, url, init_params, segment_params, 
                is_label_roi=is_label_roi, 
                initLastParams=initLastParams
            )
            if win is None:
                return
            
            self.standardPostProcessKwargs = win.standardPostProcessKwargs
            self.customPostProcessFeatures = win.customPostProcessFeatures
            self.customPostProcessGroupedFeatures = (
                win.customPostProcessGroupedFeatures
            )
            self.applyPostProcessing = win.applyPostProcessing
            self.secondChannelName = win.secondChannelName
            self.preproc_recipe = win.preproc_recipe
            
            myutils.log_segm_params(
                model_name, win.init_kwargs, win.model_kwargs, 
                logger_func=self.logger.info, 
                preproc_recipe=win.preproc_recipe, 
                apply_post_process=self.applyPostProcessing, 
                standard_postprocess_kwargs=self.standardPostProcessKwargs, 
                custom_postprocess_features=self.customPostProcessFeatures
            )

            use_gpu = myutils.check_gpu_requested_segm_model(win.init_kwargs)
            proceed = myutils.check_gpu_available(
                model_name, use_gpu, qparent=self,
                python_exec=self.getModelExecution(model_name).env_python,
            )
            if not proceed:
                self.getModelExecution(model_name).cancel_preparation()
                self.logger.info('Segmentation process cancelled.')
                self.titleLabel.setText('Segmentation process cancelled.')
                return
            
            previous_model = self.models[idx]
            model = self.initSegmModel(
                model_name, win.init_kwargs, posData, replace=previous_model,
                is_rgb=self.secondChannelName is not None,
            )
            if model is None:
                self.logger.info('Segmentation process cancelled.')
                self.titleLabel.setText('Segmentation process cancelled.')
                return            
            try:
                model.setupLogger(self.logger)
            except Exception as e:
                pass
            self.models[idx] = model
            model.model_name = model_name
        else:
            model = self.models[idx]
        
        if is_label_roi:
            return model

        self.titleLabel.setText(
            f'Segmenting with {model_name}... '
            '(check progress in terminal/console)', color=self.titleColor
        )
        
        post_process_params = {
            'applied_postprocessing': self.applyPostProcessing
        }
        post_process_params = {
            **post_process_params, 
            **self.standardPostProcessKwargs,
            **self.customPostProcessFeatures
        }
        if askSegmParams:
            posData.saveSegmHyperparams(
                model_name, win.init_kwargs, win.model_kwargs,
                post_process_params=post_process_params,
                preproc_recipe=self.preproc_recipe
            )

        if self.askRepeatSegment3D:
            self.segment3D = False
        if self.isSegm3D and self.askRepeatSegment3D:
            msg = widgets.myMessageBox(showCentered=False)
            msg.addDoNotShowAgainCheckbox(text='Do not ask again')
            txt = html_utils.paragraph(
                'Do you want to segment the <b>entire z-stack</b> or only the '
                '<b>current z-slice</b>?'
            )
            _, segment3DButton, _ = msg.question(
                self, '3D segmentation?', txt,
                buttonsTexts=(
                    'Cancel', 'Segment 3D z-stack', 'Segment 2D z-slice'
                )
            )
            if msg.cancel:
                self.titleLabel.setText('Segmentation process aborted.')
                self.logger.info('Segmentation process aborted.')
                return
            self.segment3D = msg.clickedButton == segment3DButton
            if msg.doNotShowAgainCheckbox.isChecked():
                self.askRepeatSegment3D = False
        
        if self.askZrangeSegm3D:
            self.z_range = None
        if self.isSegm3D and self.segment3D and self.askZrangeSegm3D:
            idx = (posData.filename, posData.frame_i)
            try:
                orignal_z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            except ValueError as e:
                orignal_z = posData.segmInfo_df.loc[idx, 'z_slice_used_gui'].iloc[0] 
            selectZtool = apps.QCropZtool(
                posData.SizeZ, parent=self, cropButtonText='Ok',
                addDoNotShowAgain=True, title='Select z-slice range to segment'
            )
            selectZtool.sigZvalueChanged.connect(self.selectZtoolZvalueChanged)
            selectZtool.sigCrop.connect(selectZtool.close)
            selectZtool.exec_()
            self.update_z_slice(orignal_z)
            if selectZtool.cancel:
                self.titleLabel.setText('Segmentation process aborted.')
                self.logger.info('Segmentation process aborted.')
                return
            startZ = selectZtool.lowerZscrollbar.value()
            stopZ = selectZtool.upperZscrollbar.value()
            self.z_range = (startZ, stopZ)
            if selectZtool.doNotShowAgainCheckbox.isChecked():
                self.askZrangeSegm3D = False
        
        secondChannelData = None
        if self.secondChannelName is not None:
            secondChannelData = self.getSecondChannelData()
        
        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        self.model = model
        # Re-enabled in segmWorkerFinished or workerCritical
        self.setDisabled(True)
        
        self.segmWorkerMutex = QMutex()
        self.segmWorkerWaitCond = QWaitCondition()
        self.thread = QThread()
        self.worker = workers.segmWorker(
            self, 
            secondChannelData=secondChannelData,
            mutex=self.segmWorkerMutex, 
            waitCond=self.segmWorkerWaitCond
        )
        self.worker.z_range = self.z_range
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        if self.debug:
            self.worker.debug.connect(self.debugSegmWorker)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.segmWorkerFinished)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def debugSegmWorker(self, to_debug):
        img, _lab, lab = to_debug
        printl(img.shape, _lab.shape, lab.shape)
        imshow(img, _lab, lab)
        self.segmWorkerWaitCond.wakeAll()

    def selectZtoolZvalueChanged(self, whichZ, z):
        self.update_z_slice(z)

    @exception_handler
    def repeatSegmVideo(self, model_name, startFrameNum, stopFrameNum):
        if model_name == 'thresholding':
            # thresholding model is stored as 'Automatic thresholding'
            # at line of code `models.append('Automatic thresholding')`
            model_name = 'Automatic thresholding'

        idx = self.modelNames.index(model_name)

        self.downloadWin = apps.downloadModel(model_name, parent=self)
        self.downloadWin.download()

        if model_name == 'Automatic thresholding':
            # Automatic thresholding is the name of the models as stored 
            # in self.modelNames, but the actual model is called thresholding
            # (see cellacdc/models/thresholding)
            model_name = 'thresholding'

        posData = self.data[self.pos_i]
        # Read all models parameters (from its isolated env if used)
        model_spec = self.prepareModelExecution(model_name)
        if model_spec is None:
            return
        init_params, segment_params, url = model_spec
        
        if model_name == 'thresholding':
            autoThreshWin = apps.QDialogAutomaticThresholding(
                parent=self, isSegm3D=self.isSegm3D
            )
            autoThreshWin.exec_()
            if autoThreshWin.cancel:
                self.getModelExecution(model_name).cancel_preparation()
                return
        
        win = self.initSegmModelParams(
            model_name, url, init_params, segment_params
        )
        if win is None:
            return

        self.standardPostProcessKwargs = win.standardPostProcessKwargs
        self.customPostProcessFeatures = win.customPostProcessFeatures
        self.customPostProcessGroupedFeatures = (
            win.customPostProcessGroupedFeatures
        )
        self.applyPostProcessing = win.applyPostProcessing
        self.preproc_recipe = win.preproc_recipe
        
        myutils.log_segm_params(
            model_name, win.init_kwargs, win.model_kwargs, 
            logger_func=self.logger.info, 
            preproc_recipe=win.preproc_recipe, 
            apply_post_process=self.applyPostProcessing, 
            standard_postprocess_kwargs=self.standardPostProcessKwargs, 
            custom_postprocess_features=self.customPostProcessFeatures
        )
        
        secondChannelData = None
        if win.secondChannelName is not None:
            secondChannelData = self.getSecondChannelData()

        use_gpu = myutils.check_gpu_requested_segm_model(win.init_kwargs)
        proceed = myutils.check_gpu_available(
            model_name, use_gpu, qparent=self,
            python_exec=self.getModelExecution(model_name).env_python,
        )
        if not proceed:
            self.getModelExecution(model_name).cancel_preparation()
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return

        model = self.initSegmModel(
            model_name, win.init_kwargs, posData,
            is_rgb=win.secondChannelName is not None,
        )
        if model is None:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return
        try:
            model.setupLogger(self.logger)
        except Exception as e:
            pass
        # The video model is used only by this worker: stop it when done
        previous = getattr(self, 'segmVideoModel', None)
        if previous is not None:
            self.releaseSegmModel(*previous)
        self.segmVideoModel = (model_name, model)
        
        self.extendSegmDataIfNeeded(stopFrameNum)
        self.reInitLastSegmFrame(
            from_frame_i=startFrameNum-1, updateImages=False
        )

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        self.progressWin = apps.QDialogWorkerProgress(
            title='Segmenting video', parent=self,
            pbarDesc=f'Segmenting from frame n. {startFrameNum} to {stopFrameNum}...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(stopFrameNum-startFrameNum)

        self.thread = QThread()
        self.worker = workers.segmVideoWorker(
            posData, win, model, startFrameNum, stopFrameNum
        )
        self.worker.secondChannelData = secondChannelData
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.segmVideoWorkerFinished)
        self.worker.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.progress.connect(self.workerProgress)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def segmVideoWorkerFinished(self, exec_time, startFrameNum, stopFrameNum):
        segmVideoModel = getattr(self, 'segmVideoModel', None)
        if segmVideoModel is not None:
            self.segmVideoModel = None
            self.releaseSegmModel(*segmVideoModel)
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        # update RP and allData_li
        posData = self.data[self.pos_i]
        for frame_i in range(startFrameNum-1, stopFrameNum):
            lab = posData.segm_data[frame_i]
            allData_li_frame = posData.allData_li[frame_i]
            allData_li_frame['regionprops'] = self._acdcRegionProps(
                    lab, precache_centroids=False
                )
            if allData_li_frame['labels'] is not None:
                allData_li_frame['labels'] = lab
            
        self.activateAnnotations()

        self.get_data()
        self.tracking(enforce=True)
        self.updateAllImages()

        txt = f'Done. Segmentation computed in {exec_time:.3f} s'
        self.logger.info('-----------------')
        self.logger.info(txt)
        self.logger.info('=================')
        self.titleLabel.setText(txt, color='g')

    def segmWorkerFinished(self, lab, exec_time):
        self.setDisabled(False)
        posData = self.data[self.pos_i]

        if posData.segmInfo_df is not None and posData.SizeZ>1:
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, 'resegmented_in_gui'] = True

        if lab.ndim == 2 and self.isSegm3D:
            self.set_2Dlab(lab)
        else:
            posData.lab = lab.copy()

        self.activateAnnotations()
        
        self.update_rp(wl_update=False)
        self.tracking(enforce=True, against_next=posData.frame_i==0)
        
        proceed = self.checkHandleTooManyNewItems()
        if proceed:
            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Repeat segmentation')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Repeat segmentation')

        txt = f'Done. Segmentation computed in {exec_time:.3f} s'
        self.logger.info('-----------------')
        self.logger.info(txt)
        self.logger.info('=================')
        self.titleLabel.setText(txt, color='g')
        self.checkIfAutoSegm()
        
        QTimer.singleShot(200, self.resizeGui)

    @exception_handler
    def postProcessing(self):
        if self.postProcessSegmWin is None:
            return
        
        self.postProcessSegmWin.setPosData()
        posData = self.data[self.pos_i]
        lab, delIDs = self.postProcessSegmWin.apply()
        if posData.allData_li[posData.frame_i]['labels'] is None:
            posData.lab = lab.copy()
            self.update_rp()
        else:
            posData.allData_li[posData.frame_i]['labels'] = lab
            self.get_data()
            self.update_rp()

    def checkIfAutoSegm(self):
        """
        If there are any frame or position with empty segmentation mask
        ask whether automatic segmentation should be turned ON
        """
        if self.autoSegmAction.isChecked():
            return
        if self.autoSegmDoNotAskAgain:
            return

        ask = False
        for posData in self.data:
            if posData.SizeT > 1:
                for lab in posData.segm_data:
                    if not np.any(lab):
                        ask = True
                        txt = 'frames'
                        break
            else:
                if not np.any(posData.segm_data):
                    ask = True
                    txt = 'positions'
                    break

        if not ask:
            return

        questionTxt = html_utils.paragraph(
            f'Some or all loaded {txt} contain <b>empty segmentation masks</b>.<br><br>'
            'Do you want to <b>activate automatic segmentation</b><sup>*</sup> '
            f'when visiting these {txt}?<br><br>'
            '<i>* Automatic segmentation can always be turned ON/OFF from the menu<br>'
            '  <code>Edit --> Segmentation --> Enable automatic segmentation</code><br><br></i>'
            f'NOTE: you can automatically segment all {txt} using the<br>'
            '    segmentation module.'
        )
        msg = widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.question(
            self, 'Automatic segmentation?', questionTxt,
            buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton == yesButton:
            self.autoSegmAction.setChecked(True)
        else:
            self.autoSegmDoNotAskAgain = True
            self.autoSegmAction.setChecked(False)

    def computeSegm(self, force=False):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer' or mode == 'Cell cycle analysis':
            return

        if np.any(posData.lab) and not force:
            # Do not compute segm if there is already a mask
            return

        if not self.autoSegmAction.isChecked():
            return

        self.repeatSegm(model_name=self.segmModelName)

    def showInstructionsCustomModel(self):
        modelFilePath = apps.addCustomModelMessages(self)
        if modelFilePath is None:
            self.logger.info('Adding custom model process stopped.')
            return
        
        myutils.store_custom_model_path(modelFilePath)
        modelName = os.path.basename(os.path.dirname(modelFilePath))
        customModelAction = QAction(modelName)
        self.segmSingleFrameMenu.addAction(customModelAction)
        self.segmActions.append(customModelAction)
        self.segmActionsVideo.append(customModelAction)
        self.modelNames.append(modelName)
        self.models.append(None)
        self.sender().callback(customModelAction)

    def warnReinitLastSegmFrame(self, frame_n=None):
        if frame_n is None:
            current_frame_n = self.navigateScrollBar.value()
        else:
            current_frame_n = frame_n
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            Are you sure you want to <b>re-initialize the last visited and 
            validated</b> frame to number {current_frame_n}?<br><br>
            WARNING: If you save, <b>all annotations after frame number 
            {current_frame_n} will be lost!</b> 
        """)
        msg.warning(
            self, 'WARNING: Potential loss of data', txt,
            buttonsTexts=('Cancel', 'Yes, I am sure')
        )
        return msg.cancel

    def extendSegmDataIfNeeded(self, stopFrameNum):
        posData = self.data[self.pos_i]
        segmSizeT = len(posData.segm_data)
        if stopFrameNum <= segmSizeT:
            return
        numFramesToAdd = stopFrameNum - segmSizeT
        posData.allData_li.extend(
            [myutils.get_empty_stored_data_dict() for i in range(numFramesToAdd)]
        )
        lab_shape = posData.segm_data[0].shape
        shapeToAdd = (numFramesToAdd, *lab_shape)
        additionalSegmData = np.zeros(shapeToAdd, dtype=posData.segm_data.dtype)
        extendedSegmData = np.concatenate((posData.segm_data, additionalSegmData))
        posData.segm_data = extendedSegmData

    def reInitLastSegmFrame(
            self, checked=True, from_frame_i=None, updateImages=True,
            force=False
        ):
        if not force:
            cancel = self.warnReinitLastSegmFrame()
            if cancel:
                self.logger.info(
                    'Re-initialization of last validated frame cancelled.'
                )
                return

        posData = self.data[self.pos_i]
        if from_frame_i is None:
            from_frame_i = posData.frame_i
        
        self.lastFrameRanOnFirstVisitTools = posData.frame_i
        
        self.updateLastCheckedFrameWidgets(from_frame_i)
        posData.last_tracked_i = from_frame_i
        self.navigateScrollBar.setMaximum(from_frame_i+1)
        self.navSpinBox.setMaximum(from_frame_i+1)
        # self.navigateScrollBar.setMinimum(1)
        
        # posData.tracked_lost_centroids[from_frame_i-1] = set()
        for i in range(from_frame_i, posData.SizeT):
            if posData.allData_li[i]['labels'] is None:
                break
            
            posData.segm_data[i] = posData.allData_li[i]['labels']
            rp_old = posData.allData_li[i]['regionprops']
            posData.allData_li[i] = myutils.get_empty_stored_data_dict()
            posData.allData_li[i]['regionprops'] = rp_old
            
            posData.tracked_lost_centroids[i] = set()
            posData.acdcTracker2stepsAnnotInfo.pop(i, None)            
        
        if posData.acdc_df is not None:
            frames = posData.acdc_df.index.get_level_values(0)
            if from_frame_i in frames:
                posData.acdc_df = posData.acdc_df.loc[:from_frame_i]
        
        self.removeAlldelROIsCurrentFrame()
        
        if not updateImages:
            return
        
        self.updateAllImages()

    def reinitStoredSegmModels(self):
        old_models = [
            (getattr(model, 'model_name', None), model) 
            for model in self.models if model is not None
        ]
        self.models = [None]*len(self.models)
        for model_name, model in old_models:
            self.releaseSegmModel(model_name, model)
        self.releaseSegForLostIDsModels()
