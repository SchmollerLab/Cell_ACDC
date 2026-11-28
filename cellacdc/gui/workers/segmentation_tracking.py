"""Main GUI (guiWin-only) workers that segment, track and edit masks:
segmWorker and segmVideoWorker (repeat segmentation of a frame or a
video), MagicPromptsWorker and MagicPromptsSegmStackWorker (promptable
models; the latter subclasses segmVideoWorker), LabelRoiWorker,
SegForLostIDsWorker, trackingWorker, relabelSequentialWorker,
PostProcessSegmWorker, and the segmentation edit helpers
FindNextNewIdWorker, CopyAllLostObjectsWorker and
DelObjectsOutsideSegmROIWorker.
"""
import os
import inspect
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.exposure
from tqdm import tqdm
from qtpy.QtCore import (
    Signal, QObject, QMutex, QWaitCondition
)
from cellacdc import load
from cellacdc.myutils import model_installs
from cellacdc import transformation
from cellacdc import features
from cellacdc import core
from cellacdc import cca_df_colnames_with_tree
from cellacdc import segm_utils
from cellacdc.shared.workers import base as workers_base


class FindNextNewIdWorker(QObject):
    def __init__(self, posData, guiWin):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.posData = posData
        self.guiWin = guiWin
    
    @workers_base.worker_exception_handler
    def run(self):
        prev_IDs = None
        next_frame_i = -1
        for frame_i, data_dict in enumerate(self.posData.allData_li):
            lab = data_dict['labels']
            rp = data_dict['regionprops']
            IDs = data_dict['regionprops'].IDs
            if lab is None:
                lab = self.posData.segm_data[frame_i]
                rp = skimage.measure.regionprops(lab)
                IDs = [obj.label for obj in rp]
            
            if prev_IDs is None:
                prev_IDs = IDs
                continue
            
            newIDs = [ID for ID in IDs if ID not in prev_IDs]
            if newIDs:
                next_frame_i = frame_i
                break            
            prev_IDs = IDs
            
        self.signals.finished.emit(next_frame_i)

class SegForLostIDsWorker(QObject):
    sigAskInit = Signal()
    sigShowImageDebug = Signal(object)
    sigStoreData = Signal(bool)
    sigUpdateRP = Signal(bool, bool)
    sigGetSegForLostIDsInputImg = Signal(str)
    sigSegForLostIDsWorkerAskInstallGPU = Signal(str, bool)
    sigInitModel = Signal(int, str, object)
    sigTrackManuallyAddedObject = Signal(object, object, bool, bool)

    def __init__(self, guiWin, mutex, waitCond, debug=False):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.guiWin = guiWin
        self.mutex = mutex
        self.waitCond = waitCond
        self._debug = debug
        self.inputImgForSegForLostIDs = None
        self.initModelResult = None
        self.initModelError = None
        self.assignments = {}
        self._acks = defaultdict(int)

    def _emit_and_wait(self, ack_key, signal, *args):
        """Emit a signal and wait until the GUI slot acknowledges it.

        This prevents missed wake-ups when wakeAll() happens before wait().
        """
        self.mutex.lock()
        prev_ack = self._acks[ack_key]
        try:
            signal.emit(*args)
            while self._acks[ack_key] == prev_ack:
                self.waitCond.wait(self.mutex)
        finally:
            self.mutex.unlock()

    def ack(self, ack_key):
        self._acks[ack_key] += 1

    def emitSigAskInit(self):
        self._emit_and_wait('ask_init', self.sigAskInit)
    
    def emitSigStoreData(self, autosave):
        self._emit_and_wait('store_data', self.sigStoreData, autosave)
    
    def emitSigUpdateRP(self, wl_update, wl_track_og_curr):
        self._emit_and_wait(
            'update_rp', self.sigUpdateRP, wl_update, wl_track_og_curr
        )
        
    def emitSigInitModel(self, model_idx, base_model_name, init_kwargs):
        """Ask the GUI thread for an initialized model.

        Models may run in an isolated environment, whose startup can show
        dialogs and run a Qt event loop, so it must happen on the GUI thread.
        Returns ``(model, error)``.
        """
        self.initModelResult = None
        self.initModelError = None
        self._emit_and_wait(
            'init_model',
            self.sigInitModel,
            model_idx,
            base_model_name,
            init_kwargs,
        )
        model, error = self.initModelResult, self.initModelError
        self.initModelResult = None
        self.initModelError = None
        return model, error

    def emitGetSegForLostIDsInputImg(self, image_channel_name):
        self._emit_and_wait( # emit and waiting is handled here
            'get_input_img', self.sigGetSegForLostIDsInputImg, image_channel_name
        )
        img = self.inputImgForSegForLostIDs
        self.inputImgForSegForLostIDs = None
        return img
        
    def emitSigAskInstallGPU(self, base_model_name, use_gpu):
        self._emit_and_wait(
            'ask_install_gpu',
            self.sigSegForLostIDsWorkerAskInstallGPU,
            base_model_name,
            use_gpu,
        )
    
    def emitTrackManuallyAddedObject(self, IDs, isLost, wl_update, wl_track_og_curr):
        self.assignments = {}
        self._emit_and_wait(
            'track_manually_added',
            self.sigTrackManuallyAddedObject,
            IDs,
            isLost,
            wl_update,
            wl_track_og_curr,
        )
        return self.assignments
              
    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        
    def emitSigImageDebug(self, display_info):
        self._emit_and_wait('image_debug', self.sigShowImageDebug, display_info)
    
    @workers_base.worker_exception_handler
    def run(self):
        from cellacdc import regionprops
        
        posData = self.guiWin.data[self.guiWin.pos_i]
        frame_i = posData.frame_i

        if not self.guiWin.SegForLostIDsSettings:
            self.emitSigAskInit()
        
        if not self.guiWin.SegForLostIDsSettings:
            self.signals.finished.emit(self)
            return

        self.logger.info('Segmentation for lost IDs started.')

        model_settings = self.guiWin.SegForLostIDsSettings['models_settings']

        n_models = len(model_settings)
        total_steps = 2 * n_models
        self.signals.initProgressBar.emit(total_steps)
        
        assigned_IDs = []
        original_lab = posData.lab.copy()
        IDs_bboxs_list = []
        bboxs_list = []
        new_labs = []
        newly_assigned_IDs_list = []
        
        prev_lab = self.guiWin.get_2Dlab(posData.allData_li[frame_i-1]['labels'])
        prev_IDs = posData.allData_li[frame_i-1]['regionprops'].IDs_set
        
        # iteratively go through models, keeping found labs and only feeding remaining lost IDs to the next model
        for model_idx, model_settings_i in enumerate(model_settings):
            base_model_name = model_settings_i['base_model_name']
            # self.logger.info(
            #     f'[SegForLostIDs] Model {model_idx+1}/{n_models}: {base_model_name}'
            # )
            init_kwargs_new = dict(model_settings_i['init_kwargs_new'])
            effective_init_kwargs = dict(init_kwargs_new)
            image_channel_name = init_kwargs_new.pop(
                'image_channel_name', 'Displayed image'
            )
            args_new = model_settings_i['args_new']
            init_kwargs = model_settings_i.get('init_kwargs', {})
            effective_init_kwargs.update(init_kwargs)
            model_kwargs = model_settings_i.get('model_kwargs', {})
            preproc_recipe = model_settings_i.get('preproc_recipe', None)
            applyPostProcessing = model_settings_i.get('applyPostProcessing', False)
            standardPostProcessKwargs = model_settings_i.get('standardPostProcessKwargs', {})
            customPostProcessFeatures = model_settings_i.get('customPostProcessFeatures', None)
            customPostProcessGroupedFeatures = model_settings_i.get('customPostProcessGroupedFeatures', None)

            # Fall back to reading from the live win object when available
            win = model_settings_i.get('win')
            if win is not None:
                init_kwargs = win.init_kwargs
                effective_init_kwargs.update(init_kwargs)
                model_kwargs = win.model_kwargs
                preproc_recipe = win.preproc_recipe
                applyPostProcessing = win.applyPostProcessing
                standardPostProcessKwargs = win.standardPostProcessKwargs
                customPostProcessFeatures = win.customPostProcessFeatures
                customPostProcessGroupedFeatures = win.customPostProcessGroupedFeatures

            use_gpu = model_installs.check_gpu_requested_segm_model(effective_init_kwargs)
            use_gpu = use_gpu or effective_init_kwargs.get('device_type', 'cpu').lower() != 'cpu'

            # self.logger.info('[SegForLostIDs] Checking GPU availability...')
            self.emitSigAskInstallGPU(base_model_name, use_gpu)

            if not self.gpu_go:
                self.signals.finished.emit(self)
                return

            if not self.dont_force_cpu:
                changed_smth = False
                if 'device' in effective_init_kwargs:
                    effective_init_kwargs = dict(effective_init_kwargs, device=None)
                    changed_smth = True
                if 'use_gpu' in effective_init_kwargs:
                    effective_init_kwargs = dict(effective_init_kwargs, use_gpu=False)
                    changed_smth = True
                if 'gpu' in effective_init_kwargs:
                    effective_init_kwargs = dict(effective_init_kwargs, gpu=False)
                    changed_smth = True
                if 'device_type' in effective_init_kwargs:
                    effective_init_kwargs = dict(effective_init_kwargs, device_type='cpu')
                    changed_smth = True
                if not changed_smth:
                    self.logger.warning(
                        f'Could not force {base_model_name} model to use CPU. '
                        'Please check the model settings.'
                    )

            effective_init_kwargs.pop('image_channel_name', None)
            model, init_error = self.emitSigInitModel(
                model_idx, base_model_name, effective_init_kwargs
            )
            if init_error is not None:
                self.logger.warning(
                    f'Could not initialize {base_model_name} model.'
                )
                self.signals.critical.emit((self, init_error))
                self.signals.finished.emit(self)
                return
            if model is None:
                self.logger.info('Segmentation model was not initialized correctly!')
                self.signals.critical.emit(
                    (self, 'Segmentation model was not initialized correctly!')
                )
                self.signals.finished.emit(self)
                return
            if self._debug:
                try:
                    model.setupLogger(self.guiWin.logger)
                except Exception:
                    pass

            # self.logger.info('[SegForLostIDs] Fetching input image...')
            curr_img = self.emitGetSegForLostIDsInputImg(image_channel_name)
            # self.logger.info('[SegForLostIDs] Input image ready.')
            if curr_img is None:
                self.signals.critical.emit(
                    (self, 'Could not get input image for SegForLostIDsWorker')
                )
                self.signals.finished.emit(self)
                return

            tracked_lost_IDs = self.guiWin.getTrackedLostIDs()
            new_unique_ID = self.guiWin.setBrushID(useCurrentLab=True, return_val=True)

            missing_IDs = prev_IDs - posData.rp.IDs_set - set(tracked_lost_IDs)

            assigned_IDs_prev = assigned_IDs.copy()
            
            # self.sigShowImageDebug.emit({
            # 'title': f'posData.lab  ({model_idx}) right before running single_cell_seg',
            # 'images': [posData.lab.copy()],
            # 'img_titles': [f'posData.lab ({model_idx}) right before running single_cell_seg']
            #     })
            
            # self.logger.info('[SegForLostIDs] Running single-cell segmentation...')
            
            out = segm_utils.single_cell_seg(
                model, prev_lab, posData.lab, curr_img,
                missing_IDs, new_unique_ID,
                posData,
                distance_filler_growth=args_new['distance_filler_growth'],
                overlap_threshold=args_new['overlap_threshold'],
                padding=args_new['padding'],
                fill_other_cells_with_background=args_new['fill_other_cells_with_background'],
                model_kwargs=model_kwargs,
                preproc_recipe=preproc_recipe,
                applyPostProcessing=applyPostProcessing,
                standardPostProcessKwargs=standardPostProcessKwargs,
                customPostProcessFeatures=customPostProcessFeatures,
                customPostProcessGroupedFeatures=customPostProcessGroupedFeatures,
                debug=self._debug,
                sigShowImageFunc = self.emitSigImageDebug
            )
            if self._debug:
                new_lab, assigned_IDs, IDs_bboxs, bboxs, imgs_to_show = out
            else:
                new_lab, assigned_IDs, IDs_bboxs, bboxs = out
            # self.logger.info('[SegForLostIDs] single-cell segmentation done.')

            bboxs_list.append(bboxs)
            posData.lab = new_lab
            self.emitSigUpdateRP(wl_update=True, wl_track_og_curr=False)
            newly_assigned_IDs = set(assigned_IDs) - set(assigned_IDs_prev)
            assignments = self.emitTrackManuallyAddedObject(
                newly_assigned_IDs, True, True, False
            )
            if assignments is None:
                assignments = {} 

            # update assigned_IDs based on assignments made during tracking
            for i, ID in enumerate(assigned_IDs):
                if ID in assignments:
                    assigned_IDs[i] = assignments[ID]


            IDs_bboxs_list.append(IDs_bboxs)
            newly_assigned_IDs = set(assigned_IDs) - set(assigned_IDs_prev)
            newly_assigned_IDs_list.append(newly_assigned_IDs)

            new_labs.append(posData.lab.copy())
            self.signals.progressBar.emit(1)
            if self._debug:
                print(f'Model {model_idx+1}: Assigned IDs: {assigned_IDs}, assignments: {assignments}, newly assigned IDs: {newly_assigned_IDs}')
            
        global_areas = [obj.area for obj in posData.allData_li[frame_i-1]['regionprops']]
        global_area_mean = np.mean(global_areas)
        
        for i, (IDs_bboxs, bboxs) in enumerate(zip(IDs_bboxs_list, bboxs_list)):
            args_new = model_settings[i]['args_new']
            model_lab = new_labs[i]
            current_model_new_IDs = newly_assigned_IDs_list[i]

            if not current_model_new_IDs:
                self.signals.progressBar.emit(1)
                continue
            
            for j, (IDs_prev_bbox, bbox) in enumerate(zip(IDs_bboxs, bboxs)):
                box_x_min, box_x_max, box_y_min, box_y_max = bbox
    
                
                model_bbox_lab = model_lab[box_x_min:box_x_max, box_y_min:box_y_max]
                model_bbox_lab_cleared = skimage.segmentation.clear_border(model_bbox_lab, buffer_size=1)
                model_lab_rp = regionprops.acdcRegionprops(model_bbox_lab_cleared, precache_centroids=False)

                original_bbox_lab = original_lab[box_x_min:box_x_max, box_y_min:box_y_max]
                area_mean = global_area_mean
                
                skip_size_filter = area_mean is None
                if not skip_size_filter:
                    min_area = (1 - args_new['size_perc_diff']) * area_mean
                    max_area = (1 + args_new['size_perc_diff']) * area_mean

                filtered_IDs = []
                if self._debug:
                    filtered_not_found_curr_model = []
                    filtered_size = []
                    filtered_not_tracked = []

                for obj in model_lab_rp:
                    if obj.label not in current_model_new_IDs: # make sure its an ID actually found by model and not prev model
                        if self._debug:
                            filtered_not_found_curr_model.append(obj.label)
                        continue
                    if not skip_size_filter and not (obj.area > min_area and obj.area < max_area):
                        if self._debug:
                            filtered_size.append(obj.label)
                        continue
                    if args_new['allow_only_tracked_cells'] and obj.label not in prev_IDs:
                        if self._debug:
                            filtered_not_tracked.append(obj.label)
                        continue

                    filtered_IDs.append(obj.label)

                if self._debug:
                    print(f'Filtered IDs for model {i+1}, bbox {j+1}: {filtered_IDs}')
                    print(f'current_model_new_IDs: {current_model_new_IDs}')
                    print(f'Filtered not found in current model: {filtered_not_found_curr_model}')
                    print(f'Filtered by size: {filtered_size}')
                    print(f'Filtered not tracked: {filtered_not_tracked}')
                    
                if filtered_IDs:
                    mask = np.isin(model_bbox_lab, filtered_IDs)
                    # make sure to not overwrite any existing IDs in the original lab
                    mask = np.logical_and(mask, original_bbox_lab == 0)
                    original_bbox_lab[mask] = model_bbox_lab[mask]

            self.signals.progressBar.emit(1)

        posData.lab = original_lab
        
        self.emitSigUpdateRP(wl_update=True, wl_track_og_curr=False)
        self.emitSigStoreData(autosave=True)

        self.logger.info('Segmentation for lost IDs done.')

        self.signals.finished.emit(self)


class LabelRoiWorker(QObject):
    finished = Signal()
    critical = Signal(object)
    progress = Signal(str, object)
    sigProgressBar = Signal(int)
    sigLabellingDone = Signal(object, bool)

    def __init__(self, Gui):
        QObject.__init__(self)
        self.logger = workers_base.workerLogger(self.progress)
        self.Gui = Gui
        self.mutex = Gui.labelRoiMutex
        self.waitCond = Gui.labelRoiWaitCond
        self.exit = False
        self.started = False
    
    def pause(self):
        self.logger.log('Draw box around object to start magic labeller.')
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
            self.logger.log('Magic labeller started...')
        self.started = True
        self.waitCond.wakeAll()
    
    def _stop(self):
        self.logger.log('Magic labeller backend process done. Closing it...')
        self.exit = True
        self.waitCond.wakeAll()
    
    def _segment_image(self, img, secondChannelImg):
        if secondChannelImg is not None:
            img = self.Gui.labelRoiModel.second_ch_img_to_stack(
                img, secondChannelImg
            )
        
        lab = core.segm_model_segment(
            self.Gui.labelRoiModel, img, self.Gui.model_kwargs, 
            preproc_recipe=self.Gui.preproc_recipe, 
            posData=self.posData
        )
        if self.Gui.applyPostProcessing:
            lab = core.post_process_segm(
                lab, **self.Gui.standardPostProcessKwargs
            )
            if self.Gui.customPostProcessFeatures:
                lab = features.custom_post_process_segm(
                    self.posData, self.Gui.customPostProcessGroupedFeatures, 
                    lab, img, self.posData.frame_i, self.posData.filename, 
                    self.posData.user_ch_name, 
                    self.Gui.customPostProcessFeatures
                )
        return lab
    
    @workers_base.worker_exception_handler
    def run(self):
        while not self.exit:
            if self.exit:
                break
            elif self.started:
                self.logger.log('Magic labeller is doing its magic...')
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
            self, mainWin, 
            secondChannelData=None, 
            mutex: QWaitCondition=None, 
            waitCond: QMutex=None
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
    
    @workers_base.worker_exception_handler
    def run(self):
        t0 = time.perf_counter()
        if self.mainWin.segment3D:
            img = self.mainWin.getDisplayedZstack()
            if self.z_range is not None:
                startZ, stopZ = self.z_range
                img = img[startZ:stopZ+1]
        else:
            img = self.mainWin.getDisplayedImg1()
        
        posData = self.mainWin.data[self.mainWin.pos_i]
        lab = np.zeros_like(posData.segm_data[0])
        
        # self.emitDebug((img, self.secondChannelData))
        
        if self.secondChannelData is not None:
            img = self.mainWin.model.second_ch_img_to_stack(
                img, self.secondChannelData
            )

        start_z_slice = 0
        if self.z_range is not None:
            start_z_slice, _ = self.z_range
        elif not self.mainWin.segment3D and posData.isSegm3D:
            idx = (posData.filename, posData.frame_i)
            start_z_slice = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
        
        _lab = core.segm_model_segment(
            self.mainWin.model, img, 
            self.mainWin.model_kwargs, 
            frame_i=posData.frame_i, 
            posData=posData, 
            start_z_slice=start_z_slice
        )
        posData.saveSamEmbeddings(logger_func=self.logger.info)

        if self.mainWin.applyPostProcessing:
            _lab = core.post_process_segm(
                _lab, **self.mainWin.standardPostProcessKwargs
            )
            if self.mainWin.customPostProcessFeatures:
                _lab = features.custom_post_process_segm(
                    posData, self.mainWin.customPostProcessGroupedFeatures, 
                    _lab, img, posData.frame_i, posData.filename, 
                    posData.user_ch_name, self.mainWin.customPostProcessFeatures
                )
        
        if self.z_range is not None:
            # 3D segmentation of a z-slices subset
            startZ, stopZ = self.z_range
            lab[startZ:stopZ+1] = _lab
        elif not self.mainWin.segment3D and posData.isSegm3D:
            # 3D segmentation but segmented current z-slice
            idx = (posData.filename, posData.frame_i)
            z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            lab[z] = _lab
        else:
            # Either whole z-stack or 2D segmentation
            lab = _lab
        
        t1 = time.perf_counter()
        exec_time = t1-t0
        self.finished.emit(lab, exec_time)

class segmVideoWorker(QObject):
    finished = Signal(float, int, int)
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
        self.logger = workers_base.workerLogger(self.progress)

    def _check_extend_segm_data(self, segm_data, stop_frame_num):
        if stop_frame_num <= len(segm_data):
            return segm_data
        extended_shape = (stop_frame_num, *segm_data.shape[1:])
        extended_segm_data = np.zeros(extended_shape, dtype=segm_data.dtype)
        extended_segm_data[:len(segm_data)] = segm_data
        if len(extended_shape) == 4:
            return extended_segm_data
        if self.posData.SizeZ == 1:
            return extended_segm_data
        else:
            num_added_frames = len(extended_segm_data) - len(segm_data)
            half_z = int(self.posData.SizeZ/2)
            # 2D segm on 3D over time data --> fix segmInfo
            segmInfo_extended = pd.DataFrame({
                'filename': [self.posData.filename]*num_added_frames,
                'frame_i': list(range(len(segm_data), len(extended_segm_data))),
                'z_slice_used_gui': [half_z]*num_added_frames,
                'which_z_proj_gui': ['single z-slice']*num_added_frames
            }).set_index(['filename', 'frame_i'])
            segmInfo_df = pd.concat([self.posData.segmInfo_df, segmInfo_extended])
            self.posData.segmInfo_df = segmInfo_df
            self.posData.segmInfo_df.to_csv(self.posData.segmInfo_df_csv_path)
        return extended_segm_data

    @workers_base.worker_exception_handler
    def run(self):
        t0 = time.perf_counter()
        self.posData.segm_data = self._check_extend_segm_data(
            self.posData.segm_data, self.stopFrameNum
        )
        img_data = self.posData.img_data[self.startFrameNum-1:self.stopFrameNum]
        is4D = img_data.ndim == 4
        is2D_segm = self.posData.segm_data.ndim == 3
        if is4D and is2D_segm:
            filename = self.posData.filename
            zz = self.posData.segmInfo_df.loc[filename, 'z_slice_used_gui']
        else:
            zz = None
        for i, img in enumerate(img_data):
            frame_i = i+self.startFrameNum-1
            if self.secondChannelData is not None:
                img = self.model.second_ch_img_to_stack(
                    img, self.secondChannelData
                )
            if zz is not None:
                z_slice = zz.loc[frame_i]
                img = img[z_slice]
                
            lab = core.segm_model_segment(
                self.model, img, self.model_kwargs, frame_i=frame_i, 
                preproc_recipe=self.preproc_recipe, 
                posData=self.posData
            )
            self.posData.saveSamEmbeddings(logger_func=self.logger.log)
            if self.applyPostProcessing:
                lab = core.post_process_segm(
                    lab, **self.standardPostProcessKwargs
                )
                if self.customPostProcessFeatures:
                    lab = features.custom_post_process_segm(
                        self.posData, 
                        self.customPostProcessGroupedFeatures, 
                        lab, img, self.posData.frame_i, 
                        self.posData.filename, 
                        self.posData.user_ch_name, 
                        self.customPostProcessFeatures
                    )
            self.posData.segm_data[frame_i] = lab
            self.progressBar.emit(1)
        t1 = time.perf_counter()
        exec_time = t1-t0
        self.finished.emit(exec_time, self.startFrameNum, self.stopFrameNum)


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
        self.signals = workers_base.signals()
        self.waitCond = QWaitCondition()
        self.tracker = self.mainWin.tracker
        self.track_params = self.mainWin.track_params
        self.video_to_track = video_to_track
    
    def _get_first_untracked_lab(self):
        start_frame_i = self.mainWin.start_n - 1
        frameData = self.posData.allData_li[start_frame_i]
        lab = frameData['labels']
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
            if hasattr(self.signals, 'innerPbar_available'):
                if self.signals.innerPbar_available:
                    # Use inner pbar of the GUI widget (top pbar is for positions)
                    self.signals.sigInitInnerPbar.emit(1)
                    return
            else:
                self.signals.initProgressBar.emit(1)
        except Exception as err:
            pass
    
    @workers_base.worker_exception_handler
    def run(self):
        self.mutex.lock()        
        self.progress.emit(
            'Tracking process started (more details in the terminal)...')
        
        trackerInputImage = None
        self.track_params['signals'] = self.signals
        if 'image' in self.track_params:
            trackerInputImage = self.track_params.pop('image')
            start_frame_i = self.mainWin.start_n-1
            stop_frame_n = self.mainWin.stop_n

            trackerInputImage = trackerInputImage[start_frame_i:stop_frame_n]
        
        tracked_video = core.tracker_track(
            self.video_to_track, self.tracker, self.track_params, 
            intensity_img=trackerInputImage,
            logger_func=self.progress.emit
        )
        
        self._setProgressBarIndefiniteWait()
        
        # self.debug.emit((tracked_video, self))
        # self.waitCond.wait(self.mutex)
        
        self.progress.emit('Re-tracking first frame to ensure continuity...')
        # Relabel first frame objects back to IDs they had before tracking
        # (to ensure continuity with past untracked frames)
        tracked_video = self._relabel_first_frame_labels(tracked_video)
        
        print('')
        self.progress.emit('Generating annotations...')
        acdc_df = self.posData.fromTrackerToAcdcDf(
            self.tracker, tracked_video, start_frame_i=self.mainWin.start_n-1
        )
        # Store new tracked video
        current_frame_i = self.posData.frame_i
        self.trackingOnNeverVisitedFrames = False
        print('')
        self.progress.emit('Storing tracked video...')
        pbar = tqdm(total=len(tracked_video), ncols=100)
        for rel_frame_i, lab in enumerate(tracked_video):
            frame_i = rel_frame_i + self.mainWin.start_n - 1

            if acdc_df is not None:
                cca_cols = acdc_df.columns.intersection(
                    cca_df_colnames_with_tree
                )
                # Store cca_df if it is an output of the tracker
                cca_df = acdc_df.loc[frame_i][cca_cols]
                self.mainWin.store_cca_df(
                    frame_i=frame_i, cca_df=cca_df, mainThread=False,
                    autosave=False
                )

            if self.posData.allData_li[frame_i]['labels'] is None:
                # repeating tracking on a never visited frame
                # --> modify only raw data and ask later what to do
                self.posData.segm_data[frame_i] = lab
                if frame_i == current_frame_i:
                    self.posData.lab = lab

                self.mainWin.update_rp(
                    frame_i=frame_i,
                    wl_update=False,
                    is_worker_thread=True
                )
                if frame_i == current_frame_i:
                    self.posData.allData_li[frame_i]['regionprops'] = (
                        self.posData.rp
                    )
                self.trackingOnNeverVisitedFrames = True
            else:
                # Get the rest of the stored metadata based on the new lab
                self.posData.allData_li[frame_i]['labels'] = lab
                self.posData.frame_i = frame_i
                self.mainWin.get_data()
                self.mainWin.update_rp(is_worker_thread=True)
                self.mainWin.store_data(autosave=False)
            
            pbar.update()
        pbar.close()

        # Back to current frame
        self.posData.frame_i = current_frame_i
        self.mainWin.get_data()
        self.mainWin.store_data(autosave=True)
        self.mutex.unlock()
        self.finished.emit()


class PostProcessSegmWorker(QObject):
    def __init__(
            self, 
            postProcessKwargs, 
            customPostProcessGroupedFeatures, 
            customPostProcessFeatures,
            mainWin
        ):
        super().__init__()
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.kwargs = postProcessKwargs
        self.customPostProcessGroupedFeatures = customPostProcessGroupedFeatures
        self.customPostProcessFeatures = customPostProcessFeatures
        self.mainWin = mainWin
    
    @workers_base.worker_exception_handler
    def run(self):
        mainWin = self.mainWin
        data = mainWin.data
        posData = data[mainWin.pos_i]
        if len(data) > 1:
            self.signals.initProgressBar.emit(len(data))
        else:
            current_frame_i = posData.frame_i
            self.signals.initProgressBar.emit(posData.SizeT - current_frame_i)
        
        self.logger.log('Post-process segmentation process started.')
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
                lab = data_dict['labels']
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
                        self.customPostProcessFeatures
                    )
                if visited:
                    posData.allData_li[frame_i]['labels'] = processed_lab
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = frame_i
                    mainWin.get_data()
                    mainWin.update_rp(is_worker_thread=True)
                    mainWin.store_data(autosave=False)
                else:
                    posData.segm_data[frame_i] = processed_lab
                    mainWin.update_rp(
                        frame_i=frame_i, 
                        is_worker_thread=True,
                        wl_update=False
                    )

                self.signals.progressBar.emit(1)
            
            posData.frame_i = current_frame_i


class DelObjectsOutsideSegmROIWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    debug = Signal(object)

    def __init__(
            self, 
            segm_roi_endname: os.PathLike, 
            segm_data: np.ndarray,
            images_path: os.PathLike
        ):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.segm_roi_endname = segm_roi_endname
        self.segm_data = segm_data
        self.images_path = images_path
        
    @workers_base.worker_exception_handler
    def run(self):
        segm_roi_endname = self.segm_roi_endname
        segm_roi_filepath, _ = load.get_path_from_endname(
            segm_roi_endname, self.images_path
        )
        self.progress.emit(f'Loading segmentation file "{segm_roi_filepath}"...')
        segm_roi_data = load.load_image_file(segm_roi_filepath)
        
        self.progress.emit(f'Deleting objects outside of selected ROIs...')
        cleared_segm_data, delIDs = transformation.del_objs_outside_segm_roi(
            segm_roi_data, self.segm_data
        )
        
        self.finished.emit((self, cleared_segm_data, delIDs))


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

    @workers_base.worker_exception_handler
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
            output['overlap_warning'] = overlap_warning
            self.finished.emit(output)
            return

        # Back to current frame
        self.returnToFrame.emit(current_frame_i)

        if last_visited_frame_i < last_copied_frame_i:
            output['doReinitLastSegmFrame'] = True
            output['last_visited_frame_i'] = last_visited_frame_i

        output['overlap_warning'] = overlap_warning
        self.finished.emit(output)


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
        s = '\n'.join([str(pair).replace(',', ' -->') for pair in li])
        s = f'IDs relabelled as follows:\n{s}'
        self.progress.emit(s)

    @workers_base.worker_exception_handler
    def run(self):
        self.mutex.lock()

        self.progress.emit('Relabelling process started...')
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
                lab = data_dict['labels']
                if lab is None:
                    break
                segm_data.append(lab)
                # if frame_i == current_frame_i:
                #     break

            if not segm_data:
                segm_data = np.array([current_lab])

            segm_data = np.array(segm_data)
            segm_data, oldIDs, newIDs = core.relabel_sequential(
                segm_data, is_timelapse=posData.SizeT>1
            )
            self.progressNewIDs(oldIDs, newIDs)
            self.sigRemoveItemsGUI.emit(np.max(segm_data))

            self.progress.emit(
                'Updating stored data and cell cycle annotations '
                '(if present)...'
            )

            mainWin.updateAnnotatedIDs(oldIDs, newIDs, logger=self.progress.emit)
            mainWin.store_data(mainThread=False)

            for frame_i, lab in enumerate(segm_data):
                posData.frame_i = frame_i
                posData.lab = lab
                mainWin.get_cca_df()
                if posData.cca_df is not None:
                    mainWin.update_cca_df_relabelling(
                        posData, oldIDs, newIDs
                    )
                mainWin.update_rp(
                    is_worker_thread=True
                )
                mainWin.store_data(mainThread=False)

        # Go back to current frame
        mainWin.pos_i = current_pos_i
        posData = self.data[mainWin.pos_i]
        posData.frame_i = current_frame_i
        mainWin.get_data()

        self.mutex.unlock()
        self.finished.emit()       

class MagicPromptsWorker(QObject):
    def __init__(
            self, posData, image, df_points, model, model_segment_kwargs,
            image_origin=(0, 0, 0), global_image=None, id_offset=0,
            z_slice=None, df_boxes=None
        ):
        QObject.__init__(self)

        self.signals = workers_base.signals()
        self.posData = posData
        self.image = image
        if global_image is not None:
            self.global_image = global_image
        else:
            self.global_image = image
        # `df_points` is None for prompts that are not points
        # (e.g., brush strokes already used to train the model)
        self.df_points = df_points
        # Box prompts (columns 'id', 'z', 'ymin', 'xmin', 'ymax', 'xmax'),
        # None if there are no boxes
        self.df_boxes = df_boxes
        self.image_origin = image_origin
        self.model = model
        self.model_segment_kwargs = model_segment_kwargs
        # Added to the IDs of model output (e.g., to avoid collisions of
        # new objects with existing IDs)
        self.id_offset = id_offset
        # If not None, only this z-slice of the z-stack `global_image` is
        # segmented (passed to the model as `z_slice`) and combined with the
        # same z-slice of the 3D labels
        self.z_slice = z_slice

    @staticmethod
    def _accepts_kwarg(func, name):
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return True
        if name in params:
            return True
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _canOmitPromptImage(self):
        """True if `add_prompt` can receive `image=None` instead of the
        image (the prompts then refer to the image passed to `segment`).

        Only for the built-in promptable models running in their isolated
        environment (`ModelClient`), where it avoids sending the image to
        the model worker with every prompt. In-process models (e.g., custom
        models) receive the image as before (no copy is involved).
        """
        from cellacdc import promptable_models_path
        from cellacdc.service.client import ModelClient

        if not isinstance(self.model, ModelClient):
            return False
        model_name = getattr(self.model, 'model_name', None)
        if not model_name:
            return False
        # Exact (case-sensitive) folder name, like the import of the model
        # by the worker (e.g., a custom model "SAM2" is not "sam2")
        try:
            builtin_names = os.listdir(promptable_models_path)
        except OSError:
            return False
        if model_name not in builtin_names:
            return False
        module_path = os.path.join(
            promptable_models_path, model_name, 'acdcPromptSegment.py'
        )
        if not os.path.isfile(module_path):
            return False
        return self._isPromptImageSegmentedImage()

    def _isPromptImageSegmentedImage(self):
        """True if the prompts refer to the image passed to `segment` (i.e.,
        not to a crop of it, see "Compute on zoomed area")."""
        if tuple(self.image_origin) != (0, 0, 0):
            return False
        if self.image is self.global_image:
            return True
        image = np.asarray(self.image)
        global_image = np.asarray(self.global_image)
        return (
            image.shape == global_image.shape
            and np.may_share_memory(image, global_image)
        )

    @workers_base.worker_exception_handler
    def run(self):
        from cellacdc.promptable_models import utils

        # With `image=None` the prompts refer to the image passed to
        # `segment`. This avoids sending the image with every prompt to the
        # model worker process (see `_canOmitPromptImage`)
        prompt_image = self.image
        if self._canOmitPromptImage():
            prompt_image = None

        edited_IDs = []
        if self.df_points is not None:
            for row in self.df_points.itertuples():
                prompt_id = row.id
                point = (row.z, row.y, row.x)
                print(f'Adding point prompt {point} with id = {prompt_id}...')
                parent_obj_id = row.Cell_ID if row.Cell_ID == prompt_id else 0
                self.model.add_prompt(
                    prompt=point,
                    prompt_id=prompt_id,
                    parent_obj_id=parent_obj_id,
                    image=prompt_image,
                    image_origin=self.image_origin,
                    prompt_type='point'
                )
            edited_IDs = list(self.df_points['Cell_ID'].unique())

        if self.df_boxes is not None:
            for row in self.df_boxes.itertuples():
                z = None if pd.isna(row.z) else int(row.z)
                box = (
                    z, float(row.ymin), float(row.xmin), float(row.ymax),
                    float(row.xmax)
                )
                prompt_id = int(row.id)
                print(f'Adding box prompt {box} with id = {prompt_id}...')
                self.model.add_prompt(
                    prompt=box,
                    prompt_id=prompt_id,
                    image=prompt_image,
                    image_origin=self.image_origin,
                    prompt_type='box'
                )
            edited_IDs.extend(self.df_boxes['id'].unique())

        lab = self.posData.lab
        image = self.global_image
        segment_kwargs = dict(self.model_segment_kwargs)
        if self.z_slice is not None:
            lab = lab[self.z_slice]
            if self._accepts_kwarg(self.model.segment, 'z_slice'):
                segment_kwargs['z_slice'] = self.z_slice
            else:
                # The model cannot segment one z-slice of a z-stack (e.g.,
                # a custom model) --> give it only the z-slice
                image = image[self.z_slice]

        lab_out = self.model.segment(
            image,
            lab=lab,
            **segment_kwargs
        )

        lab_out = np.asarray(lab_out)
        is_whole_zstack = (
            self.z_slice is not None
            and lab_out.ndim == lab.ndim + 1
            and lab_out.shape[1:] == lab.shape
        )
        if is_whole_zstack:
            # The model ignored `z_slice` (e.g., dropped by the model server
            # because the model does not accept it) and segmented the
            # entire z-stack
            lab_out = lab_out[self.z_slice]
        if lab_out.shape != lab.shape:
            raise ValueError(
                f'The model output has shape {lab_out.shape} which is '
                f'different from the segmentation masks shape {lab.shape}.'
            )

        if self.id_offset:
            max_output_id = int(lab_out.max()) if lab_out.size else 0
            if int(self.id_offset) + max_output_id > np.iinfo(np.uint32).max:
                raise ValueError(
                    'Cannot add new objects: their IDs would exceed the uint32 '
                    'range. Relabel the existing objects before segmenting.'
                )
            lab_out = lab_out.astype(np.uint32)
            lab_out[lab_out > 0] += np.uint32(self.id_offset)

        lab_new, lab_union, lab_interesection = (
            utils.insert_model_output_into_labels(
                lab,
                lab_out,
                edited_IDs=edited_IDs
            )
        )

        self.signals.finished.emit((lab_new, lab_union, lab_interesection))

class MagicPromptsSegmStackWorker(segmVideoWorker):
    """Segment a range of frames with a trained promptable model (e.g.,
    ConvPaint trained on brush strokes). Reuses the signals of
    `segmVideoWorker` so that `guiWin.segmVideoWorkerFinished` can be used.
    """
    def __init__(
            self, posData, model, model_segment_kwargs, startFrameNum,
            stopFrameNum, annot_ndim
        ):
        QObject.__init__(self)
        self.posData = posData
        self.model = model
        self.model_kwargs = model_segment_kwargs
        self.startFrameNum = startFrameNum
        self.stopFrameNum = stopFrameNum
        self.annot_ndim = annot_ndim
        self.logger = workers_base.workerLogger(self.progress)

    @workers_base.worker_exception_handler
    def run(self):
        from cellacdc.promptable_models import utils

        t0 = time.perf_counter()
        self.posData.segm_data = self._check_extend_segm_data(
            self.posData.segm_data, self.stopFrameNum
        )
        lab_shape = self.posData.segm_data.shape[1:]
        for frame_i in range(self.startFrameNum-1, self.stopFrameNum):
            img = utils.get_brush_strokes_image(
                self.posData, frame_i, self.annot_ndim
            )
            segment_kwargs = {**self.model_kwargs, 'frame_i': frame_i}
            lab = self.model.segment(img, **segment_kwargs)
            lab = np.asarray(lab)
            if lab.shape != lab_shape:
                raise ValueError(
                    f'The model output of frame n. {frame_i+1} has shape '
                    f'{lab.shape} which is different from the segmentation '
                    f'masks shape {lab_shape}.'
                )
            self.posData.segm_data[frame_i] = lab
            self.progressBar.emit(1)
        t1 = time.perf_counter()
        exec_time = t1-t0
        self.finished.emit(exec_time, self.startFrameNum, self.stopFrameNum)
