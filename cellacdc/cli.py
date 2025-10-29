import os
import traceback
import time
from tqdm import tqdm

import re

import numpy as np
import pandas as pd

import skimage.measure

from . import exception_handler_cli
from . import printl
from . import load
from . import error_up_str
from . import issues_url
from . import myutils
from . import config
from . import core
from . import features
from . import measurements
from . import io
from . import favourite_func_metrics_csv_path
from . import cca_functions

class HeadlessSignal:
    def __init__(self, *args):
        pass
    
    def emit(self, *args, **kwargs):
        pass

class ProgressCliSignal:
    def __init__(self, logger_func):
        self.logger_func = logger_func
    
    def emit(self, text):
        self.logger_func(text)

class KernelCliSignals:
    def __init__(self, logger_func):
        self.finished = HeadlessSignal(float)
        self.progress = ProgressCliSignal(logger_func)
        self.progressBar = HeadlessSignal(int)
        self.innerProgressBar = HeadlessSignal(int)
        self.resetInnerPbar = HeadlessSignal(int)
        self.progress_tqdm = HeadlessSignal(int)
        self.signal_close_tqdm = HeadlessSignal()
        self.create_tqdm = HeadlessSignal(int)
        self.debug = HeadlessSignal(object)
        self.critical = HeadlessSignal(object)

class _WorkflowKernel:
    def __init__(self, logger, log_path, is_cli=False):
        self.logger = logger
        self.log_path = log_path
        self.is_cli = is_cli
    
    @exception_handler_cli
    def parse_paths(self, workflow_params):
        paths_to_segm = workflow_params['paths_info']['paths']
        if 'initialization' in workflow_params:
            ch_name = workflow_params['initialization']['user_ch_name']
        elif 'measurements' in workflow_params:
            ch_name = workflow_params['measurements']['channels'][0]
        else:
            printl(workflow_params, pretty=True)
            raise KeyError(
                'Cannot find channel name in workflow parameters. '
                'See above.'
            )
        parsed_paths = []
        for path in paths_to_segm:
            if os.path.isfile(path):
                parsed_paths.append(path)
                continue
            
            images_paths = load.get_images_paths(path)
            ch_filepaths = load.get_user_ch_paths(images_paths, ch_name)
            parsed_paths.extend(ch_filepaths)
        return parsed_paths

    @exception_handler_cli
    def parse_stop_frame_numbers(self, workflow_params):
        stop_frames_param = (
            workflow_params['paths_info']['stop_frame_numbers']
        )
        return [int(n) for n in stop_frames_param]
    
    def quit(self, error=None):
        if not self.is_cli and error is not None:
            raise error
        
        self.logger.info('='*50)
        if error is not None:
            self.logger.exception(traceback.format_exc())
            print('-'*60)
            self.logger.info(f'[ERROR]: {error}{error_up_str}')
            err_msg = (
                'Cell-ACDC aborted due to **error**. '
                'More details above or in the following log file:\n\n'
                f'{self.log_path}\n\n'
                'If you cannot solve it, you can report this error by opening '
                'an issue on our '
                'GitHub page at the following link:\n\n'
                f'{issues_url}\n\n'
                'Please **send the log file** when reporting a bug, thanks!'
            )
            self.logger.info(err_msg)
        else:
            self.logger.info(
                'Cell-ACDC command-line interface closed. '
                f'{myutils.get_salute_string()}'
            )
        self.logger.info('='*50)
        exit()

class SegmKernel(_WorkflowKernel):
    def __init__(self, logger, log_path, is_cli):
        super().__init__(logger, log_path, is_cli=is_cli)

    @exception_handler_cli
    def parse_custom_postproc_features_grouped(self, workflow_params):
        custom_postproc_grouped_features = {}
        for section, options in workflow_params.items():
            if not section.startswith('postprocess_features.'):
                continue
            category = section.split('.')[-1]
            for option, value in options.items():
                if option == 'names':
                    values = value.strip('\n').strip().split('\n')
                    custom_postproc_grouped_features[category] = values
                    continue
                channel = option
                if category not in custom_postproc_grouped_features:
                    custom_postproc_grouped_features[category] = {
                        channel: [value]
                    }
                elif channel not in custom_postproc_grouped_features[category]:
                    custom_postproc_grouped_features[category][channel] = (
                        [value]
                    )
                else:
                    custom_postproc_grouped_features[category][channel].append(value)
        return custom_postproc_grouped_features
    
    @exception_handler_cli      
    def init_args_from_params(self, workflow_params, logger_func):
        args = workflow_params['initialization'].copy()
        args['use3DdataFor2Dsegm'] = workflow_params.get(
            'use3DdataFor2Dsegm', False
        )
        args['model_kwargs'] = workflow_params['segmentation_model_params']
        args['track_params'] = workflow_params.get('tracker_params', {})
        args['standard_postrocess_kwargs'] = (
            workflow_params.get('standard_postprocess_features', {})
        )
        args['custom_postproc_features'] = (
            workflow_params.get('custom_postprocess_features', {})
        )
        args['custom_postproc_grouped_features'] = (
            self.parse_custom_postproc_features_grouped(workflow_params)
        )
        
        args['SizeT'] = workflow_params['metadata']['SizeT']
        args['SizeZ'] = workflow_params['metadata']['SizeZ']
        args['logger_func'] = logger_func
        args['init_model_kwargs'] = (
            workflow_params.get('init_segmentation_model_params', {})
        )
        args['init_tracker_kwargs'] = (
            workflow_params.get('init_tracker_params', {})
        )
        
        args['preproc_recipe'] = config.preprocess_ini_items_to_recipe(
            workflow_params
        )
        
        self.init_args(**args)
    
    @exception_handler_cli
    def init_args(
            self, 
            user_ch_name, 
            segm_endname,
            model_name, 
            do_tracking,
            do_postprocess, 
            do_save,
            image_channel_tracker,
            standard_postrocess_kwargs,
            custom_postproc_grouped_features,
            custom_postproc_features,
            isSegm3D,
            use_ROI,
            second_channel_name,
            use3DdataFor2Dsegm,
            model_kwargs, 
            track_params,
            SizeT, 
            SizeZ,
            tracker_name='',
            model=None,
            preproc_recipe=None,
            init_model_kwargs=None,
            init_tracker_kwargs=None,
            tracker=None,
            signals=None,
            logger_func=print,
            innerPbar_available=False,
            is_segment3DT_available=False, 
            reduce_memory_usage=False,
            use_freehand_ROI=True
        ):
        self.user_ch_name = user_ch_name
        self.segm_endname = segm_endname
        self.model_name = model_name
        self.do_postprocess = do_postprocess
        self.standard_postrocess_kwargs = standard_postrocess_kwargs
        self.custom_postproc_grouped_features = custom_postproc_grouped_features
        self.custom_postproc_features = custom_postproc_features
        self.do_tracking = do_tracking
        self.do_save = do_save
        self.image_channel_tracker = image_channel_tracker
        self.isSegm3D = isSegm3D
        self.use3DdataFor2Dsegm = use3DdataFor2Dsegm
        self.use_ROI = use_ROI
        self.second_channel_name = second_channel_name
        self.logger_func = logger_func
        self.innerPbar_available = innerPbar_available
        self.SizeT = SizeT
        self.SizeZ = SizeZ
        self.init_model_kwargs = init_model_kwargs
        self.init_tracker_kwargs = init_tracker_kwargs
        self.is_segment3DT_available = (
            is_segment3DT_available and not reduce_memory_usage
        )
        self.preproc_recipe = preproc_recipe
        self.use_freehand_ROI = use_freehand_ROI
        if signals is None:
            self.signals = KernelCliSignals(logger_func)
        else:
            self.signals = signals
        self.model = model
        self.model_kwargs = model_kwargs
        self.tracker_name = tracker_name
        self.init_tracker(
            self.do_tracking, track_params, tracker_name=tracker_name, 
            tracker=tracker
        )
    
    @exception_handler_cli
    def init_segm_model(self, posData):
        self.signals.progress.emit(
            f'\nInitializing {self.model_name} segmentation model...'
        )
        acdcSegment = myutils.import_segment_module(self.model_name)
        init_argspecs, segment_argspecs = myutils.getModelArgSpec(acdcSegment)
        self.init_model_kwargs = myutils.parse_model_params(
            init_argspecs, self.init_model_kwargs
        )
        self.model_kwargs = myutils.parse_model_params(
            segment_argspecs, self.model_kwargs
        )
        if self.second_channel_name is not None:
            self.init_model_kwargs['is_rgb'] = True

        self.model = myutils.init_segm_model(
            acdcSegment, posData, self.init_model_kwargs
        )
        if self.model is None:
            # The model was not initialized correctly
            return
        self.is_segment3DT_available = any(
            [name=='segment3DT' for name in dir(self.model)]
        )
    
    @exception_handler_cli
    def init_tracker(
            self, do_tracking, track_params, tracker_name='', tracker=None
        ):
        if not do_tracking:
            self.tracker = None
            return
        
        if tracker is None:
            self.signals.progress.emit(f'Initializing {tracker_name} tracker...')
            tracker_module = myutils.import_tracker_module(tracker_name)
            init_argspecs, track_argspecs = myutils.getTrackerArgSpec(
                tracker_module, realTime=False
            )
            self.init_tracker_kwargs = myutils.parse_model_params(
                init_argspecs, self.init_tracker_kwargs
            )
            self.init_tracker_kwargs = myutils.parse_model_params(
                init_argspecs, self.init_tracker_kwargs
            )
            track_params = myutils.parse_model_params(
                track_argspecs, track_params
            )
            tracker = tracker_module.tracker(**self.init_tracker_kwargs)
            
        self.track_params = track_params
        self.tracker = tracker
    
    def _tracker_track(self, lab, tracker_input_img=None):
        tracked_lab = core.tracker_track(
            lab, self.tracker, self.track_params, 
            intensity_img=tracker_input_img, 
            logger_func=self.logger_func
        )
        return tracked_lab
        
    @exception_handler_cli
    def run(
            self,
            img_path,  
            stop_frame_n
        ):    
        posData = load.loadData(img_path, self.user_ch_name)

        self.logger_func(f'Loading {posData.relPath}...')

        posData.getBasenameAndChNames()
        posData.buildPaths()
        posData.loadImgData()
        posData.loadOtherFiles(
            load_segm_data=False,
            load_acdc_df=False,
            load_shifts=True,
            loadSegmInfo=True,
            load_delROIsInfo=False,
            load_dataPrep_ROIcoords=True,
            load_bkgr_data=True,
            load_last_tracked_i=False,
            load_metadata=True,
            load_dataprep_free_roi=True,
            end_filename_segm=self.segm_endname
        )
        # Get only name from the string 'segm_<name>.npz'
        endName = (
            self.segm_endname.replace('segm', '', 1)
            .replace('_', '', 1)
            .split('.')[0]
        )
        if endName:
            # Create a new file that is not the default 'segm.npz'
            posData.setFilePaths(endName)

        segmFilename = os.path.basename(posData.segm_npz_path)
        if self.do_save:
            self.logger_func(f'\nSegmentation file {segmFilename}...')

        posData.SizeT = self.SizeT
        if self.SizeZ > 1:
            SizeZ = posData.img_data.shape[-3]
            posData.SizeZ = SizeZ
        else:
            posData.SizeZ = 1

        posData.isSegm3D = self.isSegm3D
        posData.saveMetadata()
        
        isROIactive = False
        if posData.dataPrep_ROIcoords is not None and self.use_ROI:
            df_roi = posData.dataPrep_ROIcoords.loc[0]
            isROIactive = df_roi.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = df_roi['value'].astype(int)[:4]
            Y, X = posData.img_data.shape[-2:]
            x0 = x0 if x0>0 else 0
            y0 = y0 if y0>0 else 0
            x1 = x1 if x1<X else X
            y1 = y1 if y1<Y else Y

        # Note that stop_i is not used when SizeT == 1 so it does not matter
        # which value it has in that case
        stop_i = stop_frame_n

        if self.second_channel_name is not None:
            self.logger_func(
                f'Loading second channel "{self.second_channel_name}"...'
            )
            secondChFilePath = load.get_filename_from_channel(
                posData.images_path, self.second_channel_name
            )
            secondChImgData = load.load_image_file(secondChFilePath)

        if posData.SizeT > 1:
            self.t0 = 0
            if posData.SizeZ > 1 and not self.isSegm3D and not self.use3DdataFor2Dsegm:
                # 2D segmentation on 3D data over time
                img_data = posData.img_data

                if self.second_channel_name is not None:
                    second_ch_data_slice = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, :, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data_slice = second_ch_data_slice[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))

                img_data_slice = img_data[self.t0:stop_i]
                postprocess_img = img_data
                
                Y, X = img_data.shape[-2:]
                newShape = (stop_i, Y, X)
                img_data = np.zeros(newShape, img_data.dtype)
                
                if self.second_channel_name is not None:
                    second_ch_data = np.zeros(newShape, secondChImgData.dtype)
                df = posData.segmInfo_df.loc[posData.filename]
                for z_info in df[:stop_i].itertuples():
                    i = z_info.Index
                    z = z_info.z_slice_used_dataPrep
                    zProjHow = z_info.which_z_proj
                    img = img_data_slice[i]
                    if self.second_channel_name is not None:
                        second_ch_img = second_ch_data_slice[i]
                    if zProjHow == 'single z-slice':
                        img_data[i] = img[z]
                        if self.second_channel_name is not None:
                            second_ch_data[i] = second_ch_img[z]
                    elif zProjHow == 'max z-projection':
                        img_data[i] = img.max(axis=0)
                        if self.second_channel_name is not None:
                            second_ch_data[i] = second_ch_img.max(axis=0)
                    elif zProjHow == 'mean z-projection':
                        img_data[i] = img.mean(axis=0)
                        if self.second_channel_name is not None:
                            second_ch_data[i] = second_ch_img.mean(axis=0)
                    elif zProjHow == 'median z-proj.':
                        img_data[i] = np.median(img, axis=0)
                        if self.second_channel_name is not None:
                            second_ch_data[i] = np.median(second_ch_img, axis=0)
            elif posData.SizeZ > 1 and (self.isSegm3D or self.use3DdataFor2Dsegm):
                # 3D segmentation on 3D data over time
                img_data = posData.img_data[self.t0:stop_i]
                postprocess_img = img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, :, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (0, 0), (y0, Y-y1), (x0, X-x1))
            else:
                # 2D data over time
                img_data = posData.img_data[self.t0:stop_i]
                postprocess_img = img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
        else:
            if posData.SizeZ > 1 and not self.isSegm3D and not self.use3DdataFor2Dsegm:
                img_data = posData.img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]

                postprocess_img = img_data
                # 2D segmentation on single 3D image
                z_info = posData.segmInfo_df.loc[posData.filename].iloc[0]
                z = z_info.z_slice_used_dataPrep
                zProjHow = z_info.which_z_proj
                if zProjHow == 'single z-slice':
                    img_data = img_data[z]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[z]
                elif zProjHow == 'max z-projection':
                    img_data = img_data.max(axis=0)
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data.max(axis=0)
                elif zProjHow == 'mean z-projection':
                    img_data = img_data.mean(axis=0)
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data.mean(axis=0)
                elif zProjHow == 'median z-proj.':
                    img_data = np.median(img_data, axis=0)
                    if self.second_channel_name is not None:
                        second_ch_data[i] = np.median(second_ch_data, axis=0)
            elif posData.SizeZ > 1 and (self.isSegm3D or self.use3DdataFor2Dsegm):
                # 3D segmentation on 3D z-stack
                img_data = posData.img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[:, y0:y1, x0:x1]
                postprocess_img = img_data
            else:
                # Single 2D image
                img_data = posData.img_data
                if self.second_channel_name is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[y0:y1, x0:x1]
                    if self.second_channel_name is not None:
                        second_ch_data = second_ch_data[y0:y1, x0:x1]
                postprocess_img = img_data

        self.logger_func(f'\nImage shape = {img_data.shape}')

        if self.model is None:
            self.init_segm_model(posData)
        
        if self.model is None:
            self.logger_func(
                f'\nSegmentation model {self.model_name} was not initialized!'
            )
            return
        
        """Segmentation routine"""
        self.logger_func(f'\nSegmenting with {self.model_name}...')
        t0 = time.perf_counter()
        if posData.SizeT > 1:
            if self.innerPbar_available and self.signals is not None:
                self.signals.resetInnerPbar.emit(len(img_data))
            
            if self.is_segment3DT_available:
                self.model_kwargs['signals'] = (
                    self.signals, self.innerPbar_available
                )
                if self.second_channel_name is not None:
                    img_data = self.model.second_ch_img_to_stack(
                        img_data, second_ch_data
                    )
                lab_stack = core.segm_model_segment(
                    self.model, img_data, self.model_kwargs, 
                    is_timelapse_model_and_data=True, 
                    preproc_recipe=self.preproc_recipe, 
                    posData=posData
                )
                if self.innerPbar_available:
                    # emit one pos done
                    self.signals.progressBar.emit(1)
            else:
                lab_stack = []
                pbar = tqdm(total=len(img_data), ncols=100)
                for t, img in enumerate(img_data):
                    if self.second_channel_name is not None:
                        img = self.model.second_ch_img_to_stack(
                            img, second_ch_data[t]
                        )
                        
                    lab = core.segm_model_segment(
                        self.model, img, self.model_kwargs, frame_i=t, 
                        preproc_recipe=self.preproc_recipe, 
                        posData=posData
                    )
                    lab_stack.append(lab)
                    if self.innerPbar_available:
                        self.signals.innerProgressBar.emit(1)
                    else:
                        self.signals.progressBar.emit(1)
                    pbar.update()
                pbar.close()
                lab_stack = np.array(lab_stack, dtype=np.uint32)
                if self.innerPbar_available:
                    # emit one pos done
                    self.signals.progressBar.emit(1)
        else:
            if self.second_channel_name is not None:
                img_data = self.model.second_ch_img_to_stack(
                    img_data, second_ch_data
                )

            lab_stack = core.segm_model_segment(
                self.model, img_data, self.model_kwargs, frame_i=0, 
                preproc_recipe=self.preproc_recipe, 
                posData=posData
            )
            self.signals.progressBar.emit(1)
            # lab_stack = smooth_contours(lab_stack, radius=2)

        posData.saveSamEmbeddings(logger_func=self.logger_func)
        
        if len(posData.dataPrepFreeRoiPoints) > 0 and self.use_freehand_ROI:
            self.logger_func(
                'Removing objects outside the dataprep free-hand ROI...'
            )
            lab_stack = posData.clearSegmObjsDataPrepFreeRoi(
                lab_stack, is_timelapse=posData.SizeT > 1
            )
        
        if self.do_postprocess:
            if posData.SizeT > 1:
                pbar = tqdm(total=len(lab_stack), ncols=100)
                for t, lab in enumerate(lab_stack):
                    lab_cleaned = core.post_process_segm(
                        lab, **self.standard_postrocess_kwargs
                    )
                    lab_stack[t] = lab_cleaned
                    if self.custom_postproc_features:
                        lab_filtered = features.custom_post_process_segm(
                            posData, self.custom_postproc_grouped_features, 
                            lab_cleaned, postprocess_img, t, posData.filename, 
                            posData.user_ch_name, self.custom_postproc_features
                        )
                        lab_stack[t] = lab_filtered
                    pbar.update()
                pbar.close()
            else:
                lab_stack = core.post_process_segm(
                    lab_stack, **self.standard_postrocess_kwargs
                )
                if self.custom_postproc_features:
                    lab_stack = features.custom_post_process_segm(
                        posData, self.custom_postproc_grouped_features, 
                        lab_stack, postprocess_img, 0, posData.filename, 
                        posData.user_ch_name, self.custom_postproc_features
                    )

        if posData.SizeT > 1 and self.do_tracking:     
            self.logger_func(f'\nTracking with {self.tracker_name} tracker...')       
            if self.do_save:
                # Since tracker could raise errors we save the not-tracked 
                # version which will eventually be overwritten
                self.logger_func(f'Saving NON-tracked masks of {posData.relPath}...')
                io.savez_compressed(posData.segm_npz_path, lab_stack)

            self.signals.innerPbar_available = self.innerPbar_available
            self.track_params['signals'] = self.signals
            if self.image_channel_tracker is not None:
                # Check if loading the image for the tracker is required
                if 'image' in self.track_params:
                    trackerInputImage = self.track_params.pop('image')
                else:
                    self.logger_func(
                        'Loading image data of channel '
                        f'"{self.image_channel_tracker}"')
                    trackerInputImage = posData.loadChannelData(
                        self.image_channel_tracker)
                tracked_stack = self._tracker_track(
                    lab_stack, tracker_input_img=trackerInputImage
                )
            else:
                tracked_stack = self._tracker_track(lab_stack)
            posData.fromTrackerToAcdcDf(self.tracker, tracked_stack, save=True)
        else:
            tracked_stack = lab_stack
            try:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(stop_frame_n)
                else:
                    self.signals.progressBar.emit(stop_frame_n)
            except AttributeError:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(1)
                else:
                    self.signals.progressBar.emit(1)

        if isROIactive:
            self.logger_func(f'Padding with zeros {pad_info}...')
            tracked_stack = np.pad(tracked_stack, pad_info, mode='constant')

        if self.do_save:
            self.logger_func(f'Saving {posData.relPath}...')
            io.savez_compressed(posData.segm_npz_path, tracked_stack)

        t_end = time.perf_counter()

        self.logger_func(f'\n{posData.relPath} done.')

class ComputeMeasurementsKernel(_WorkflowKernel):
    def __init__(self, logger, log_path, is_cli):
        super().__init__(logger, log_path, is_cli=is_cli)
        self.setup_done = False
    
    def init_args(self, channel_names, end_filename_segm):
        self.ch_names = channel_names
        self.end_filename_segm = end_filename_segm
        self.notLoadedChNames = []
    
    def log(self, message, level='INFO'):
        try:
            self.logger.log(message, level=level)
            return
        except Exception as err:
            pass
        
        try:
            self.logger.log(message)
            return
        except Exception as err:
            pass
        
        try:
            log_func = getattr(self.logger, level.lower())
            log_func(message)
            return
        except Exception as err:
            pass
    
    def _set_metrics_func_from_posData(self, posData):
        (metrics_func, all_metrics_names, custom_func_dict, total_metrics,
        ch_indipend_custom_func_dict) = measurements.getMetricsFunc(posData)
        self.metrics_func = metrics_func
        self.all_metrics_names = all_metrics_names
        self.total_metrics = total_metrics
        self.custom_func_dict = custom_func_dict
        self.ch_indipend_custom_func_dict = ch_indipend_custom_func_dict
        self.mixed_channel_combine_metrics = []
        self.channel_names = posData.chNames
        self.not_loaded_channel_names = []
    
    def to_workflow_config_params(self):
        params = {
            'channels': '\n'.join(self.ch_names), 
            'end_filename_segm': self.end_filename_segm
        }
        params['channel_names_to_skip'] = '\n'.join(self.chNamesToSkip)
        params['channel_names_to_process'] = '\n'.join(self.chNamesToProcess)
        calc_for_each_zslice = [
            f'{channel},{value}' 
            for channel, value in self.calc_for_each_zslice_mapper.items()
        ]
        params['calc_for_each_zslice_channels'] = '\n'.join(calc_for_each_zslice)
        
        for channel, colnames in self.metricsToSkip.items():
            params[f'metrics_to_skip_{channel}'] = '\n'.join(colnames)
        
        for channel, colnames in self.metricsToSave.items():
            params[f'metrics_to_save_{channel}'] = '\n'.join(colnames)
        
        params['calc_for_each_zslice_size'] = str(
            self.calc_size_for_each_zslice
        )
        
        params['size_metrics_to_save'] = '\n'.join(self.sizeMetricsToSave)
        params['regionprops_to_save'] = '\n'.join(self.regionPropsToSave)
        if hasattr(self, 'chIndipendCustomMetricsToSave'):
            params['channel_indipendent_custom_metrics_to_save'] = (
                '\n'.join(self.chIndipendCustomMetricsToSave)
            )
        if hasattr(self, 'mixedChCombineMetricsToSkip'):
            params['mixed_combine_metrics_to_skip'] = (
                '\n'.join(self.mixedChCombineMetricsToSkip)
            )
        
        return params
        
    def set_metrics_from_workflow_config_params(self, config_params):
        self.init_args(
            config_params['channels'],
            config_params['end_filename_segm']
        )
        
        self.chNamesToSkip = config_params['channel_names_to_skip']
        self.chNamesToProcess = config_params.get(
            'channel_names_to_process', config_params['channels']
        )
        self.metricsToSkip = {chName:[] for chName in self.ch_names}
        self.metricsToSave = {chName:[] for chName in self.ch_names}
        self.mixedChCombineMetricsToSkip = []
        self.calc_for_each_zslice_mapper = {}
        self.calc_size_for_each_zslice = (
            config_params['calc_for_each_zslice_size']
        )
        self.sizeMetricsToSave = config_params['size_metrics_to_save']
        self.regionPropsToSave = config_params['regionprops_to_save']
        if 'channel_indipendent_custom_metrics_to_save' in config_params:
            self.chIndipendCustomMetricsToSave = (
                config_params['channel_indipendent_custom_metrics_to_save']
            )
        
        if 'mixed_combine_metrics_to_skip' in config_params:
            self.mixedChCombineMetricsToSkip = (
                config_params['mixed_combine_metrics_to_skip']
            )
        
        for channel_value in config_params['calc_for_each_zslice_channels']:
            channel, value = channel_value.split(',')
            value = value.lower() == 'true'
            self.calc_for_each_zslice_mapper[channel] = value
        
        for channel in self.ch_names:
            metrics_to_skip = config_params.get(
                f'metrics_to_skip_{channel}', ''
            )
            if metrics_to_skip:
                self.metricsToSkip[channel] = metrics_to_skip
            
            metrics_to_save = config_params.get(
                f'metrics_to_save_{channel}', ''
            )
            if metrics_to_save:
                self.metricsToSave[channel] = metrics_to_save
        
    def set_metrics_from_set_measurements_dialog(self, setMeasurementsDialog):
        self.chNamesToSkip = []
        self.chNamesToProcess = []
        self.metricsToSkip = {chName:[] for chName in self.ch_names}
        self.metricsToSave = {chName:[] for chName in self.ch_names}
        self.calc_for_each_zslice_mapper = {}
        self.calc_size_for_each_zslice = False
        
        favourite_funcs = set()
        last_selected_groupboxes_measurements = load.read_last_selected_gb_meas(
            logger_func=self.log
        )
        refChannel = setMeasurementsDialog.chNameGroupboxes[0].chName
        if refChannel not in last_selected_groupboxes_measurements:
            last_selected_groupboxes_measurements[refChannel] = []
        # Remove unchecked metrics and load checked not loaded channels
        for chNameGroupbox in setMeasurementsDialog.chNameGroupboxes:
            chName = chNameGroupbox.chName
            if not chNameGroupbox.isChecked():
                # Skip entire channel
                self.chNamesToSkip.append(chName)
                continue
            
            self.chNamesToProcess.append(chName)
            self.calc_for_each_zslice_mapper[chName] = (
                chNameGroupbox.calcForEachZsliceRequested
            )
            last_selected_groupboxes_measurements[refChannel].append(
                chNameGroupbox.title()
            )
            for checkBox in chNameGroupbox.checkBoxes:
                colname = checkBox.text()
                if not checkBox.isChecked():
                    self.metricsToSkip[chName].append(colname)
                else:
                    self.metricsToSave[chName].append(colname)
                    func_name = colname[len(chName):]
                    favourite_funcs.add(func_name)

        self.calc_size_for_each_zslice = (
            setMeasurementsDialog.sizeMetricsQGBox.calcForEachZsliceRequested
        )
        if not setMeasurementsDialog.sizeMetricsQGBox.isChecked():
            self.sizeMetricsToSave = []
        else:
            self.sizeMetricsToSave = []
            title = setMeasurementsDialog.sizeMetricsQGBox.title()
            last_selected_groupboxes_measurements[refChannel].append(title)
            for checkBox in setMeasurementsDialog.sizeMetricsQGBox.checkBoxes:
                if checkBox.isChecked():
                    self.sizeMetricsToSave.append(checkBox.text())
                    favourite_funcs.add(checkBox.text())

        if not setMeasurementsDialog.regionPropsQGBox.isChecked():
            self.regionPropsToSave = ()
        else:
            self.regionPropsToSave = []
            title = setMeasurementsDialog.regionPropsQGBox.title()
            last_selected_groupboxes_measurements[refChannel].append(title)
            for checkBox in setMeasurementsDialog.regionPropsQGBox.checkBoxes:
                if checkBox.isChecked():
                    self.regionPropsToSave.append(checkBox.text())
                    favourite_funcs.add(checkBox.text())
            self.regionPropsToSave = tuple(self.regionPropsToSave)

        if setMeasurementsDialog.chIndipendCustomeMetricsQGBox is not None:
            skipAll = (
                not setMeasurementsDialog.chIndipendCustomeMetricsQGBox.isChecked()
            )
            if not skipAll:
                title = setMeasurementsDialog.chIndipendCustomeMetricsQGBox.title()
                last_selected_groupboxes_measurements[refChannel].append(title)
            chIndipendCustomMetricsToSave = []
            win = setMeasurementsDialog
            checkBoxes = win.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if skipAll:
                    continue
    
                if checkBox.isChecked():
                    chIndipendCustomMetricsToSave.append(checkBox.text())           
                    favourite_funcs.add(checkBox.text())
            self.chIndipendCustomMetricsToSave = tuple(
                chIndipendCustomMetricsToSave
            )
        
        self.mixedChCombineMetricsToSkip = []
        if setMeasurementsDialog.mixedChannelsCombineMetricsQGBox is not None:
            skipAll = (
                not setMeasurementsDialog.mixedChannelsCombineMetricsQGBox.isChecked()
            )
            if not skipAll:
                title = setMeasurementsDialog.mixedChannelsCombineMetricsQGBox.title()
                last_selected_groupboxes_measurements[refChannel].append(title)
            mixedChCombineMetricsToSkip = []
            win = setMeasurementsDialog
            checkBoxes = win.mixedChannelsCombineMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if skipAll:
                    mixedChCombineMetricsToSkip.append(checkBox.text())
                elif not checkBox.isChecked():
                    mixedChCombineMetricsToSkip.append(checkBox.text())
                else:             
                    favourite_funcs.add(checkBox.text())
            self.mixedChCombineMetricsToSkip = tuple(mixedChCombineMetricsToSkip)

        df_favourite_funcs = pd.DataFrame(
            {'favourite_func_name': list(favourite_funcs)}
        )
        df_favourite_funcs.to_csv(favourite_func_metrics_csv_path)

        load.save_last_selected_gb_meas(last_selected_groupboxes_measurements)
    
    def _init_metrics_to_save(self, posData):
        posData.setLoadedChannelNames()
        self.isSegm3D = posData.getIsSegm3D()

        if self.metricsToSave is None:
            # self.metricsToSave means that the user did not set 
            # through setMeasurements dialog --> save all measurements
            self.metricsToSave = {chName:[] for chName in posData.loadedChNames}
            isManualBackgrPresent = posData.manualBackgroundLab is not None
            for chName in posData.loadedChNames:
                metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
                    posData.SizeZ>1, chName, isSegm3D=self.isSegm3D,
                    isManualBackgrPresent=isManualBackgrPresent
                )
                self.metricsToSave[chName].extend(metrics_desc.keys())
                self.metricsToSave[chName].extend(bkgr_val_desc.keys())

                custom_metrics_desc = measurements.custom_metrics_desc(
                    posData.SizeZ>1, chName, posData=posData, 
                    isSegm3D=self.isSegm3D, return_combine=False
                )
                self.metricsToSave[chName].extend(
                    custom_metrics_desc.keys()
                )
        
        # Get metrics parameters --> function name, how etc
        self.metrics_func, _ = measurements.standard_metrics_func()
        self.custom_func_dict = measurements.get_custom_metrics_func()
        params = measurements.get_metrics_params(
            self.metricsToSave, self.metrics_func, self.custom_func_dict
        )
        (bkgr_metrics_params, foregr_metrics_params, 
        concentration_metrics_params, custom_metrics_params) = params
        self.bkgr_metrics_params = bkgr_metrics_params
        self.foregr_metrics_params = foregr_metrics_params
        self.concentration_metrics_params = concentration_metrics_params
        self.custom_metrics_params = custom_metrics_params
        
        self.ch_indipend_custom_func_dict = (
            measurements.get_channel_indipendent_custom_metrics_func()
        )
        if not hasattr(self, 'chIndipendCustomMetricsToSave'):
            self.chIndipendCustomMetricsToSave = list(
                measurements.ch_indipend_custom_metrics_desc(
                    posData.SizeZ>1, isSegm3D=self.isSegm3D,
                ).keys()
            )
            
        self.ch_indipend_custom_func_params = (
            measurements.get_channel_indipend_custom_metrics_params(
                self.ch_indipend_custom_func_dict,
                self.chIndipendCustomMetricsToSave
            )
        )
    
    def _load_posData(self, img_path, end_filename_segm):
        images_path = os.path.dirname(img_path)
        exp_foldername = os.path.basename(
            os.path.dirname(os.path.dirname(images_path))
        )
        basename, channel_names = myutils.getBasenameAndChNames(
            images_path, useExt=('.tif', '.h5')
        )
        posData = load.loadData(img_path, channel_names[0])
        
        posData.getBasenameAndChNames(useExt=('.tif', '.h5'))
        posData.buildPaths()
        posData.loadImgData()

        posData.loadOtherFiles(
            load_segm_data=True,
            load_acdc_df=True,
            load_shifts=True,
            loadSegmInfo=True,
            load_delROIsInfo=True,
            load_bkgr_data=True,
            loadBkgrROIs=True,
            load_last_tracked_i=True,
            load_metadata=True,
            load_customAnnot=True,
            load_customCombineMetrics=True,
            end_filename_segm=end_filename_segm,
            load_dataPrep_ROIcoords=True
        )
        posData.labelSegmData()
    
        self.isSegm3D = posData.getIsSegm3D()
        
        # Allow single 2D/3D image
        if posData.SizeT == 1:
            posData.img_data = posData.img_data[np.newaxis]
            
            if posData.segm_data is not None:
                posData.segm_data = posData.segm_data[np.newaxis]
        
        return posData
    
    def _load_image_data(self, posData, channel_names):
        if posData.fluo_data_dict:
            return 
        
        posData.loadedChNames = []
        for c, channel in enumerate(channel_names):
            if channel in self.chNamesToSkip:
                continue 
            
            if c == 0:
                img_data = posData.img_data
                filename = posData.filename
                bkgrData = posData.bkgrData
            else:
                # Delay loading image data
                filepath = load.get_filename_from_channel(
                    posData.images_path, channel
                )
                img_data, bkgrData = self._load_channel_data(filepath)
                if posData.SizeT == 1:
                    img_data = img_data[np.newaxis]
                    
                filename_ext = os.path.basename(filepath)
                filename, _ = os.path.splitext(filename_ext)
            
            posData.loadedChNames.append(channel)
            posData.loadedFluoChannels.add(channel)
            posData.fluo_data_dict[filename] = img_data
            posData.fluo_bkgrData_dict[filename] = bkgrData
    
    def init_signals(self, computeMetricsWorker, saveDataWorker):
        self.customMetricsCritical = HeadlessSignal()
        self.regionPropsCritical = HeadlessSignal()
        
        if saveDataWorker is not None:
            self.customMetricsCritical = saveDataWorker.customMetricsCritical
            self.regionPropsCritical = saveDataWorker.regionPropsCritical
        
        elif computeMetricsWorker is not None:
            saveDataWorker = computeMetricsWorker.mainWin.gui.saveDataWorker
            self.customMetricsCritical = saveDataWorker.customMetricsCritical
            self.regionPropsCritical = saveDataWorker.regionPropsCritical
    
    @exception_handler_cli
    def run(
            self,
            img_path: os.PathLike='',  
            stop_frame_n: int=1,
            end_filename_segm: str='',
            computeMetricsWorker=None, 
            saveDataWorker=None,
            posData=None,
            save_metrics=True,
            do_init_metrics=True
        ):      
        if posData is None:
            posData = self._load_posData(img_path, end_filename_segm)
        
        channel_names = posData.chNames
        images_path = posData.images_path
        exp_foldername = os.path.basename(posData.exp_path)
        
        self._set_metrics_func_from_posData(posData)

        if computeMetricsWorker is not None and do_init_metrics:
            computeMetricsWorker.emitSigInitMetricsDialog(posData)
            if computeMetricsWorker.abort:
                computeMetricsWorker.signals.finished.emit(computeMetricsWorker)
                return
            
            if self.setup_done:
                computeMetricsWorker.signals.finished.emit(computeMetricsWorker)
                return

            computeMetricsWorker.emitSigAskRunNow()
            if computeMetricsWorker.abort or computeMetricsWorker.savedToWorkflow:
                computeMetricsWorker.signals.finished.emit(computeMetricsWorker)
                return 
        
        if not posData.segmFound:
            rel_path = (
                f'...{os.sep}{exp_foldername}'
                f'{os.sep}{posData.pos_foldername}'
            )
            self.log(
                f'Skipping "{rel_path}" '
                f'because segm. file was not found.'
            )
            return
        
        self.init_signals(computeMetricsWorker, saveDataWorker)
        
        self.log(
            'Loaded paths:\n'
            f'Segmentation file name: {os.path.basename(posData.segm_npz_path)}\n'
            f'ACDC output file name: {os.path.basename(posData.acdc_output_csv_path)}'
        )
        
        posData.init_segmInfo_df()
        
        if computeMetricsWorker is not None:
            computeMetricsWorker.emitSigComputeVolume(posData, stop_frame_n)
            
        self._init_metrics_to_save(posData)
    
        if computeMetricsWorker is not None:
            computeMetricsWorker.signals.initProgressBar.emit(stop_frame_n)
        
        channels_to_load = [
            ch for ch in channel_names if not ch in self.chNamesToSkip 
            and ch in self.chNamesToProcess
        ]
        
        self._load_image_data(posData, channels_to_load)
        
        acdc_df_li = []
        keys = []
        for frame_i in range(stop_frame_n):
            if saveDataWorker is not None:
                stop = saveDataWorker.checkAbort()
                if stop:
                    break
                
            lab = posData.segm_data[frame_i]            
            if not np.any(lab):
                # Empty segmentation mask --> skip
                continue
            
            acdc_df = None
            if computeMetricsWorker is not None:
                rp = posData.allData_li[frame_i]['regionprops']
            elif saveDataWorker is not None:
                rp = posData.allData_li[frame_i]['regionprops']
                acdc_df = posData.allData_li[frame_i]['acdc_df']
                if acdc_df is None:
                    continue
            else:
                if frame_i == 0:
                    self.log('\nComputing cell volume...')
                rp = skimage.measure.regionprops(lab)
                rp = self._calc_volume_metrics(rp, posData)
            
            posData.lab = lab
            posData.rp = rp
            
            if acdc_df is None:
                if posData.acdc_df is None:
                    acdc_df = myutils.getBaseAcdcDf(rp)
                else:
                    try:
                        acdc_df = posData.acdc_df.loc[frame_i].copy()
                    except:
                        acdc_df = myutils.getBaseAcdcDf(rp)
            
            key = (frame_i, posData.TimeIncrement*frame_i)
            acdc_df = load.pd_bool_and_float_to_int_to_str(
                acdc_df, inplace=False, colsToCastInt=[]
            )
            
            if not save_metrics:
                if saveDataWorker is not None:
                    saveDataWorker.emitUpdateProgressBar()
                acdc_df_li.append(acdc_df)
                keys.append(key)
                continue
            
            try:
                acdc_df = self._add_volume_metrics(acdc_df, rp, posData)
                calc_metrics_addtional_args = self._init_calc_metrics(
                    acdc_df, rp, frame_i, lab, posData, 
                    saveDataWorker=saveDataWorker
                )
                acdc_df = self._calc_metrics_iter_channels(
                    acdc_df, rp, frame_i, lab, posData, 
                    *calc_metrics_addtional_args
                )
            except Exception as error:
                traceback_format = traceback.format_exc()    
                self.log(f'\n{traceback_format}')
                if computeMetricsWorker is not None:
                    computeMetricsWorker.standardMetricsErrors[str(error)] = (
                        traceback_format
                    )
                if saveDataWorker is not None:
                    saveDataWorker.addMetricsCritical.emit(
                        traceback_format, str(error)
                    )
            
            if frame_i == 0:
                if saveDataWorker is not None:
                    saveDataWorker.emitUpdateProgressBar()
                acdc_df_li.append(acdc_df)
                keys.append(key)
                continue

            try:
                prev_lab = posData.segm_data[frame_i-1]
                acdc_df = self._add_velocity_measurement(
                    acdc_df, prev_lab, lab, posData
                )
            except Exception as error:
                traceback_format = traceback.format_exc()
                self.log(f'\n{traceback_format}')
                if computeMetricsWorker is not None:
                    e = str(error)
                    computeMetricsWorker.standardMetricsErrors[e] = (
                        traceback_format
                    )
            
            acdc_df_li.append(acdc_df)
            keys.append(key)

            if computeMetricsWorker is not None:
                computeMetricsWorker.signals.progressBar.emit(1)
            
            if saveDataWorker is not None:
                saveDataWorker.emitUpdateProgressBar()
        
        if not acdc_df_li:
            print('-'*30)
            self.log(
                'All selected positions in the experiment folder '
                f'{exp_foldername} have EMPTY segmentation mask. '
                'Metrics will not be saved.'
            )
            print('-'*30)
            return
        
        all_frames_acdc_df = pd.concat(
            acdc_df_li, keys=keys, names=['frame_i', 'time_seconds', 'Cell_ID']
        )
        
        if save_metrics:
            self._add_combined_metrics(
                posData, all_frames_acdc_df, saveDataWorker=saveDataWorker
            )
        
        all_frames_acdc_df = self._add_additional_metadata(
            posData, all_frames_acdc_df, posData.segm_data
        )
        all_frames_acdc_df = self._remove_deprecated_rows(
            all_frames_acdc_df
        )
        all_frames_acdc_df = self._add_derived_cell_cycle_columns(
            all_frames_acdc_df
        )
        all_frames_acdc_df = load._fix_will_divide(all_frames_acdc_df)
        custom_annot_columns = posData.getCustomAnnotColumnNames()
        self.log(
            f'Saving acdc_output to: "{posData.acdc_output_csv_path}"'
        )
        
        self._save_acdc_df(
            all_frames_acdc_df, posData, custom_annot_columns, 
            computeMetricsWorker=computeMetricsWorker, 
            saveDataWorker=saveDataWorker
        )
    
    def _remove_deprecated_rows(self, df):
        v1_2_4_rc25_deprecated_cols = [
            'editIDclicked_x', 'editIDclicked_y',
            'editIDnewID', 'editIDnewIDs'
        ]
        df = df.drop(columns=v1_2_4_rc25_deprecated_cols, errors='ignore')

        # Remove old gui_ columns from version < v1.2.4.rc-7
        gui_columns = df.filter(regex='gui_*').columns
        df = df.drop(columns=gui_columns, errors='ignore')
        cell_id_cols = df.filter(regex='Cell_ID.*').columns
        df = df.drop(columns=cell_id_cols, errors='ignore')
        time_seconds_cols = df.filter(regex='time_seconds.*').columns
        df = df.drop(columns=time_seconds_cols, errors='ignore')
        df = df.drop(columns='relative_ID_tree', errors='ignore')
        df = df.drop(columns=['level_0', 'index'], errors='ignore')

        return df
    
    def _save_acdc_df(
            self, all_frames_acdc_df, posData, custom_annot_columns, 
            computeMetricsWorker=None, saveDataWorker=None
        ):
        try:
            if saveDataWorker is not None:
                load.store_copy_acdc_df(
                    posData, posData.acdc_output_csv_path, 
                    log_func=saveDataWorker.progress.emit
                )
            load.save_acdc_df_file(
                all_frames_acdc_df, posData.acdc_output_csv_path, 
                custom_annot_columns=custom_annot_columns
            )
            posData.acdc_df = all_frames_acdc_df
        except PermissionError as error:
            traceback_str = traceback.format_exc()
            if computeMetricsWorker is not None:
                computeMetricsWorker.emitSigPermissionErrorAndSave(
                    posData, traceback_str, all_frames_acdc_df, 
                    custom_annot_columns
                )
            
            if saveDataWorker is not None:
                saveDataWorker.emitSigPermissionErrorAndSave(
                    all_frames_acdc_df, posData.acdc_output_csv_path,
                    custom_annot_columns
                )
        except Exception as error:
            if saveDataWorker is not None:
                saveDataWorker.mutex.lock()
                saveDataWorker.critical.emit(error)
                saveDataWorker.waitCond.wait(saveDataWorker.mutex)
                saveDataWorker.mutex.unlock()
        
    def _load_channel_data(self, channel_path):
        self.log(f'Loading fluorescence image data from "{channel_path}"...')
        images_path = os.path.dirname(channel_path)
        bkgrData = None
        # Load overlay frames and align if needed
        filename = os.path.basename(channel_path)
        filename_noEXT, ext = os.path.splitext(filename)
        if ext == '.npy' or ext == '.npz':
            img_data = np.load(channel_path)
            try:
                img_data = np.squeeze(img_data['arr_0'])
            except Exception as e:
                img_data = np.squeeze(img_data)

            # Load background data
            bkgrData_path = os.path.join(
                images_path, f'{filename_noEXT}_bkgrRoiData.npz'
            )
            if os.path.exists(bkgrData_path):
                bkgrData = np.load(bkgrData_path)
        elif ext == '.tif' or ext == '.tiff':
            aligned_filename = f'{filename_noEXT}_aligned.npz'
            aligned_path = os.path.join(images_path, aligned_filename)
            if os.path.exists(aligned_path):
                img_data = np.load(aligned_path)['arr_0']

                # Load background data
                bkgrData_path = os.path.join(
                    images_path, f'{aligned_filename}_bkgrRoiData.npz'
                )
                if os.path.exists(bkgrData_path):
                    bkgrData = np.load(bkgrData_path)
            else:
                img_data = np.squeeze(skimage.io.imread(channel_path))

                # Load background data
                bkgrData_path = os.path.join(
                    images_path, f'{filename_noEXT}_bkgrRoiData.npz'
                )
                if os.path.exists(bkgrData_path):
                    bkgrData = np.load(bkgrData_path)
        else:
            return None, None

        return img_data, bkgrData
    
    def _calc_volume_metrics(self, rp, posData):
        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeX = posData.PhysicalSizeX
        obj_iter = tqdm(rp, ncols=100, position=1, leave=False)
        for i, obj in enumerate(obj_iter):
            vol_vox, vol_fl = cca_functions._calc_rot_vol(
                obj, PhysicalSizeY, PhysicalSizeX
            )
            obj.vol_vox = vol_vox
            obj.vol_fl = vol_fl
        return rp
    
    def _add_volume_metrics(self, df, rp, posData):
        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeX = posData.PhysicalSizeX
        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX
        vox_to_fl_3D = PhysicalSizeY*PhysicalSizeX*posData.PhysicalSizeZ
        
        init_list = [-2]*len(rp)
        IDs = init_list.copy()
        IDs_vol_vox = init_list.copy()
        IDs_area_pxl = init_list.copy()
        IDs_vol_fl = init_list.copy()
        IDs_area_um2 = init_list.copy()
        if self.isSegm3D:
            IDs_vol_vox_3D = init_list.copy()
            IDs_vol_fl_3D = init_list.copy()

        for i, obj in enumerate(rp):
            IDs[i] = obj.label
            IDs_vol_vox[i] = obj.vol_vox
            IDs_vol_fl[i] = obj.vol_fl
            IDs_area_pxl[i] = obj.area
            IDs_area_um2[i] = obj.area*yx_pxl_to_um2
            if self.isSegm3D:
                IDs_vol_vox_3D[i] = obj.area
                IDs_vol_fl_3D[i] = obj.area*vox_to_fl_3D
            
        df['cell_area_pxl'] = pd.Series(data=IDs_area_pxl, index=IDs, dtype=float)
        df['cell_vol_vox'] = pd.Series(data=IDs_vol_vox, index=IDs, dtype=float)
        df['cell_area_um2'] = pd.Series(data=IDs_area_um2, index=IDs, dtype=float)
        df['cell_vol_fl'] = pd.Series(data=IDs_vol_fl, index=IDs, dtype=float)
        if self.isSegm3D:
            df['cell_vol_vox_3D'] = pd.Series(
                data=IDs_vol_vox_3D, index=IDs, dtype=float
            )
            df['cell_vol_fl_3D'] = pd.Series(
                data=IDs_vol_fl_3D, index=IDs, dtype=float
            )
        return df
    
    def _check_zSlice(self, posData, frame_i, saveDataWorker=None):
        if posData.SizeZ == 1:
            return True
        
        # Iteare fluo channels and get 2D data from 3D if needed
        filenames = posData.fluo_data_dict.keys()
        for chName, filename in zip(posData.loadedChNames, filenames):
            idx = (filename, frame_i)
            try:
                if posData.segmInfo_df.at[idx, 'resegmented_in_gui']:
                    col = 'z_slice_used_gui'
                else:
                    col = 'z_slice_used_dataPrep'
                z_slice = posData.segmInfo_df.at[idx, col]
            except KeyError:
                try:
                    # Try to see if the user already selected z-slice in prev pos
                    segmInfo_df = pd.read_csv(posData.segmInfo_df_csv_path)
                    index_col = ['filename', 'frame_i']
                    posData.segmInfo_df = segmInfo_df.set_index(index_col)
                    col = 'z_slice_used_dataPrep'
                    z_slice = posData.segmInfo_df.at[idx, col]
                except KeyError as e:
                    if saveDataWorker is not None:
                        saveDataWorker.progress.emit(
                            f'z-slice for channel "{chName}" absent. '
                            'Follow instructions on pop-up dialogs.'
                        )
                        saveDataWorker.mutex.lock()
                        saveDataWorker.askZsliceAbsent.emit(filename, posData)
                        saveDataWorker.waitCond.wait(saveDataWorker.mutex)
                        saveDataWorker.mutex.unlock()
                        if saveDataWorker.abort:
                            return False
                        saveDataWorker.progress.emit(
                            f'Saving (check terminal for additional progress info)...'
                        )
                        segmInfo_df = pd.read_csv(posData.segmInfo_df_csv_path)
                        index_col = ['filename', 'frame_i']
                        posData.segmInfo_df = segmInfo_df.set_index(index_col)
                        col = 'z_slice_used_dataPrep'
                        z_slice = posData.segmInfo_df.at[idx, col]
                    else:
                        raise e
        return True
    
    def _init_calc_metrics(
            self, acdc_df, rp, frame_i, lab, posData, saveDataWorker=None
        ):
        yx_pxl_to_um2 = posData.PhysicalSizeY*posData.PhysicalSizeX
        vox_to_fl_3D = (
            posData.PhysicalSizeY*posData.PhysicalSizeX*posData.PhysicalSizeZ
        )

        manualBackgrLab = posData.manualBackgroundLab
        manualBackgrRp = None
        if manualBackgrLab is not None:
            manualBackgrRp = skimage.measure.regionprops(manualBackgrLab)
        isZstack = posData.SizeZ > 1
        isSegm3D = self.isSegm3D
        all_channels_metrics = self.metricsToSave
        size_metrics_to_save = self.sizeMetricsToSave
        regionprops_to_save = self.regionPropsToSave
        custom_func_dict = self.custom_func_dict
        
        calc_size_for_each_zslice = self.calc_size_for_each_zslice

        # Pre-populate columns with zeros
        all_columns = list(size_metrics_to_save)
        for channel, metrics in all_channels_metrics.items():
            all_columns.extend(metrics)
        all_columns.extend(regionprops_to_save)

        df_shape = (len(acdc_df), len(all_columns))
        data = np.zeros(df_shape)
        df = pd.DataFrame(data=data, index=acdc_df.index, columns=all_columns)
        # df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.combine_first(acdc_df)

        # Check if z-slice is present for 3D z-stack data
        proceed = self._check_zSlice(
            posData, frame_i, saveDataWorker=saveDataWorker
        )
        if not proceed:
            return []
        
        df = measurements.add_size_metrics(
            df, rp, size_metrics_to_save, isSegm3D, yx_pxl_to_um2, 
            vox_to_fl_3D, calc_size_for_each_zslice=calc_size_for_each_zslice
        )
        
        # Get background masks
        autoBkgr_masks = measurements.get_autoBkgr_mask(
            lab, isSegm3D, posData, frame_i
        )
        # self._emitSigDebug((lab, frame_i, autoBkgr_masks))
        
        autoBkgr_mask, autoBkgr_mask_proj = autoBkgr_masks
        dataPrepBkgrROI_mask = measurements.get_bkgrROI_mask(posData, isSegm3D)
        
        out = (
            autoBkgr_mask, 
            autoBkgr_mask_proj, 
            dataPrepBkgrROI_mask,
            manualBackgrRp
        )
    
        return out
    
    def _init_metrics(self, posData, isSegm3D):
        self.chNamesToSkip = []
        loadedChannels = posData.setLoadedChannelNames(returnList=True)
        self.chNamesToProcess = [posData.user_ch_name, *loadedChannels]
        self.metricsToSkip = {}
        self.calc_for_each_zslice_mapper = {}
        self.calc_size_for_each_zslice = False
        # At the moment we don't know how many channels the user will load -->
        # we set the measurements to save either at setMeasurements dialog
        # or at initMetricsToSave
        self.metricsToSave = None
        self.regionPropsToSave = measurements.get_props_names()
        if isSegm3D:
            self.regionPropsToSave = measurements.get_props_names_3D()
        else:
            self.regionPropsToSave = measurements.get_props_names()  

        self.mixedChCombineMetricsToSkip = []
        self.chIndipendCustomMetricsToSave = list(
            measurements.ch_indipend_custom_metrics_desc(
                posData.SizeZ>1, isSegm3D=isSegm3D,
            ).keys()
        )
        self.sizeMetricsToSave = list(
            measurements.get_size_metrics_desc(
                isSegm3D, posData.SizeT>1
            ).keys()
        )
        
        exp_path = posData.exp_path
        posFoldernames = myutils.get_pos_foldernames(exp_path)
        for pos in posFoldernames:
            images_path = os.path.join(exp_path, pos, 'Images')
            for file in myutils.listdir(images_path):
                if not file.endswith('custom_combine_metrics.ini'):
                    continue
                filePath = os.path.join(images_path, file)
                configPars = load.read_config_metrics(filePath)

                posData.combineMetricsConfig = load.add_configPars_metrics(
                    configPars, posData.combineMetricsConfig
                )
    
    def _add_custom_metrics(
            self, posData, frame_i, isSegm3D, df, rp, custom_metrics_params, 
            lab, calc_for_each_zslice_mapper
        ):
        iter_channels = zip(
            posData.loadedChNames, 
            posData.fluo_data_dict.items()
        )
        # Add custom measurements
        for channel, (filename, channel_data) in iter_channels:
            foregr_img = channel_data[frame_i]
            
            iter_other_channels = zip(
                posData.loadedChNames, 
                posData.fluo_data_dict.items()
            )
            other_channels_foregr_imgs = {
                ch:ch_data[frame_i] for ch, (_, ch_data) in iter_other_channels
                if ch != channel
            }
            
            # Get the z-slice if we have z-stacks
            z = posData.zSliceSegmentation(filename, frame_i)
            
            foregr_data = measurements.get_foregr_data(foregr_img, isSegm3D, z)
            
            df = measurements.add_custom_metrics(
                df, rp, channel, foregr_data, 
                custom_metrics_params[channel], 
                isSegm3D, lab, foregr_img, 
                other_channels_foregr_imgs,
                z_slice=z,
                customMetricsCritical=self.customMetricsCritical,
            )
            
            if not calc_for_each_zslice_mapper.get(channel, False):
                continue
            
            # Repeat measureemnts for each z-slice
            pbar_z = tqdm(
                total=posData.SizeZ, desc='Computing for z-slices: ', 
                ncols=100, leave=False, unit='z-slice'
            )
            for z in range(posData.SizeZ):
                foregr_data = measurements.get_foregr_data(
                    foregr_img, isSegm3D, z
                )
                foregr_data = {'zSlice': foregr_data['zSlice']}
                
                df = measurements.add_custom_metrics(
                    df, rp, channel, foregr_data, 
                    custom_metrics_params[channel], 
                    isSegm3D, lab, foregr_img, 
                    other_channels_foregr_imgs,
                    z_slice=z,
                    text_to_append_to_col=str(z),
                    customMetricsCritical=self.customMetricsCritical, 
                )
        
        return df
    
    def _calc_metrics_iter_channels(
            self, acdc_df, rp, frame_i, lab, posData, autoBkgr_mask, 
            autoBkgr_mask_proj, dataPrepBkgrROI_mask, manualBackgrRp
        ):
        all_channels_foregr_data = {}
        all_channels_foregr_imgs = {}
        all_channels_z_slices = {}
        isSegm3D = self.isSegm3D
        bkgr_metrics_params = self.bkgr_metrics_params
        metrics_func = self.metrics_func
        foregr_metrics_params = self.foregr_metrics_params
        calc_for_each_zslice_mapper = self.calc_for_each_zslice_mapper
        concentration_metrics_params = self.concentration_metrics_params
        regionprops_to_save = self.regionPropsToSave
        custom_metrics_params = self.custom_metrics_params
        ch_indipend_custom_func_params = (
            self.ch_indipend_custom_func_params
        )
        images_path = posData.images_path

        # Iterate channels
        iter_channels = zip(
            posData.loadedChNames, 
            posData.fluo_data_dict.items()
        )
        for channel, (filename, channel_data) in iter_channels:
            foregr_img = channel_data[frame_i]

            # Get the z-slice if we have z-stacks
            z = posData.zSliceSegmentation(filename, frame_i)
            
            # Get the background data
            bkgr_data = measurements.get_bkgr_data(
                foregr_img, posData, filename, frame_i, autoBkgr_mask, z,
                autoBkgr_mask_proj, dataPrepBkgrROI_mask, isSegm3D, lab
            )
            
            foregr_data = measurements.get_foregr_data(foregr_img, isSegm3D, z)
            
            all_channels_foregr_data[channel] = foregr_data
            all_channels_foregr_imgs[channel] = foregr_img
            all_channels_z_slices[channel] = z

            # Compute background values
            acdc_df = measurements.add_bkgr_values(
                acdc_df, bkgr_data, bkgr_metrics_params[channel], metrics_func,
                manualBackgrRp=manualBackgrRp, foregr_data=foregr_data
            )

            # Iterate objects and compute foreground metrics
            acdc_df = measurements.add_foregr_standard_metrics(
                acdc_df, rp, channel, foregr_data, 
                foregr_metrics_params[channel], 
                metrics_func, isSegm3D, 
                lab, foregr_img, 
                manualBackgrRp=manualBackgrRp,
                z_slice=z
            )

            if not calc_for_each_zslice_mapper.get(channel, False):
                continue
            
            # Repeat measureemnts for each z-slice
            pbar_z = tqdm(
                total=posData.SizeZ, desc='Computing for z-slices: ', 
                ncols=100, leave=False, unit='z-slice'
            )
            for z in range(posData.SizeZ):
                # Get the background data
                bkgr_data = measurements.get_bkgr_data(
                    foregr_img, posData, filename, frame_i, autoBkgr_mask, z,
                    autoBkgr_mask_proj, dataPrepBkgrROI_mask, isSegm3D, lab
                )
                bkgr_data = {
                    'autoBkgr': {'zSlice': bkgr_data['autoBkgr']['zSlice']},
                    'dataPrepBkgr': {'zSlice': bkgr_data['dataPrepBkgr']['zSlice']}
                }
                
                foregr_data = measurements.get_foregr_data(
                    foregr_img, isSegm3D, z
                )
                foregr_data = {'zSlice': foregr_data['zSlice']}

                # Compute background values
                acdc_df = measurements.add_bkgr_values(
                    acdc_df, bkgr_data, bkgr_metrics_params[channel], 
                    metrics_func,
                    manualBackgrRp=manualBackgrRp, 
                    foregr_data=foregr_data,
                    text_to_append_to_col=str(z)
                )

                # Iterate objects and compute foreground metrics
                acdc_df = measurements.add_foregr_standard_metrics(
                    acdc_df, rp, channel, foregr_data, 
                    foregr_metrics_params[channel], 
                    metrics_func, isSegm3D, 
                    lab, foregr_img, 
                    manualBackgrRp=manualBackgrRp,
                    z_slice=z, text_to_append_to_col=str(z)
                )
                pbar_z.update()
            pbar_z.close()

        acdc_df = measurements.add_concentration_metrics(
            acdc_df, concentration_metrics_params
        )
        
        # Add region properties
        try:
            acdc_df, rp_errors = measurements.add_regionprops_metrics(
                acdc_df, lab, regionprops_to_save, 
                logger_func=self.logger.log
            )
            if rp_errors:
                print('')
                self.logger.log(
                    'WARNING: Some objects had the following errors:\n'
                    f'{rp_errors}\n'
                    'Region properties with errors were saved as `Not A Number`.'
                )
        except Exception as error:
            traceback_format = traceback.format_exc()
            self.regionPropsCritical.emit(traceback_format, str(error))

        acdc_df = self._add_custom_metrics(
            posData, frame_i, isSegm3D, acdc_df, rp, custom_metrics_params, 
            lab, calc_for_each_zslice_mapper
        )
        
        acdc_df = measurements.add_ch_indipend_custom_metrics(
            acdc_df, rp, all_channels_foregr_data, 
            ch_indipend_custom_func_params, 
            isSegm3D, lab, all_channels_foregr_imgs, 
            all_channels_z_slices=all_channels_z_slices,
            customMetricsCritical=self.customMetricsCritical, 
        )
        
        # Remove 0s columns
        acdc_df = acdc_df.loc[:, (acdc_df != -2).any(axis=0)]

        return acdc_df
    
    def _add_velocity_measurement(self, acdc_df, prev_lab, lab, posData):
        if 'velocity_pixel' not in self.sizeMetricsToSave:
            return acdc_df
        
        if 'velocity_um' not in self.sizeMetricsToSave:
            spacing = None 
        elif self.isSegm3D:
            spacing = np.array([
                posData.PhysicalSizeZ, 
                posData.PhysicalSizeY, 
                posData.PhysicalSizeX
            ])
        else:
            spacing = np.array([
                posData.PhysicalSizeY, 
                posData.PhysicalSizeX
            ])
        velocities_pxl, velocities_um = core.compute_twoframes_velocity(
            prev_lab, lab, spacing=spacing
        )
        acdc_df['velocity_pixel'] = velocities_pxl
        acdc_df['velocity_um'] = velocities_um
        return acdc_df
    
    def _add_combined_metrics(self, posData, df, saveDataWorker=None):
        # Add channel specifc combined metrics (from equations and 
        # from user_path_equations sections)
        config = posData.combineMetricsConfig
        for chName in posData.loadedChNames:
            metricsToSkipChannel = self.metricsToSkip.get(chName, [])
            posDataEquations = config['equations']
            userPathChEquations = config['user_path_equations']
            for newColName, equation in posDataEquations.items():
                if not newColName.startswith(chName):
                    continue
                if newColName in metricsToSkipChannel:
                    continue
                self._df_eval_equation(
                    df, newColName, equation, saveDataWorker=saveDataWorker
                )
            for newColName, equation in userPathChEquations.items():
                if not newColName.startswith(chName):
                    continue
                if newColName in metricsToSkipChannel:
                    continue
                self._df_eval_equation(
                    df, newColName, equation, saveDataWorker=saveDataWorker
                )

        # Add mixed channels combined metrics
        mixedChannelsEquations = config['mixed_channels_equations']
        for newColName, equation in mixedChannelsEquations.items():
            if newColName in self.mixedChCombineMetricsToSkip:
                continue
            cols = re.findall(r'[A-Za-z0-9]+_[A-Za-z0-9_]+', equation)
            if all([col in df.columns for col in cols]):
                self._df_eval_equation(
                    df, newColName, equation, saveDataWorker=saveDataWorker
                )
    
    def _df_eval_equation(self, df, newColName, expr, saveDataWorker=None):
        try:
            df[newColName] = df.eval(expr)
        except pd.errors.UndefinedVariableError as error:
            if saveDataWorker is not None:
                saveDataWorker.sigCombinedMetricsMissingColumn.emit(
                    str(error), newColName
                )
        
        try:
            df[newColName] = df.eval(expr)
        except Exception as error:
            if saveDataWorker is not None:
                saveDataWorker.customMetricsCritical.emit(
                    traceback.format_exc(), newColName
                )
    
    def _add_additional_metadata(
            self, posData: load.loadData, df: pd.DataFrame, saved_segm_data
        ):
        for col, val in posData.additionalMetadataValues().items():
            if col in df.columns:
                df.pop(col)
            df.insert(0, col, val)
        
        try:
            df.pop('time_minutes')
        except Exception as e:
            pass
        try:
            df.pop('time_hours')
        except Exception as e:
            pass
        try:
            time_seconds = df.index.get_level_values('time_seconds')
            df.insert(0, 'time_minutes', time_seconds/60)
            df.insert(1, 'time_hours', time_seconds/3600)
        except Exception as e:
            pass
        
        df = self._add_disappears_before_end(df, saved_segm_data)
        return df
    
    def _add_disappears_before_end(
            self, acdc_df: pd.DataFrame, saved_segm_data
        ):
        acdc_df = acdc_df.drop('time_seconds', axis=1, errors='ignore')
        acdc_df = (
            acdc_df.reset_index()
            .set_index(['frame_i', 'Cell_ID'])
            .sort_index()
        )
        acdc_df['disappears_before_end'] = 0
        for frame_i, lab in enumerate(saved_segm_data):
            if frame_i == 0:
                continue
            
            try:
                df_frame = acdc_df.loc[frame_i]
            except KeyError:
                break
                
            prev_lab = saved_segm_data[frame_i-1]
            prev_rp = skimage.measure.regionprops(prev_lab)
            
            curr_rp = skimage.measure.regionprops(lab)
            curr_rp_mapper = {obj.label: obj for obj in curr_rp}
            lost_IDs = []
            for prev_obj in prev_rp:
                if curr_rp_mapper.get(prev_obj.label) is None:
                    lost_IDs.append(prev_obj.label)
            
            if 'parent_ID_tree' in df_frame.columns:
                parent_IDs = set(df_frame['parent_ID_tree'].values)
                lost_IDs = [ID for ID in lost_IDs if ID not in parent_IDs]
            
            if not lost_IDs:
                continue
            
            idx = pd.IndexSlice[frame_i-1, lost_IDs]
            try:
                acdc_df.loc[idx, 'disappears_before_end'] = 1
            except Exception as err:
                printl(frame_i, lost_IDs)
                
        return acdc_df
    
    def _add_derived_cell_cycle_columns(self, all_frames_acdc_df):
        try:
            all_frames_acdc_df = cca_functions.add_derived_cell_cycle_columns(
                all_frames_acdc_df.copy()
            )
        except Exception as err:
            self.sigLog.emit(traceback.format_exc())
        
        return all_frames_acdc_df
