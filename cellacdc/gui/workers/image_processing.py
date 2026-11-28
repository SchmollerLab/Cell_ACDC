"""Main GUI (guiWin-only) workers for preprocessing and combining channels:
CombineChannelsWorkerGUI (a subclass of CustomPreprocessWorkerGUI in
shared/workers/base.py), and SaveProcessedDataWorker and
SaveCombinedChannelsWorker, which save the preprocessed and the
combined-channel images.
"""
import os
import itertools
from typing import Union, List, Dict, Callable, Any, Tuple, Iterable
import numpy as np
from qtpy.QtCore import Signal, QObject, QWaitCondition
from cellacdc import load, printl, io
from cellacdc import core
from cellacdc.shared.workers import base as workers_base


class SaveProcessedDataWorker(QObject):
    def __init__(
            self, 
            allPosData: Iterable['load.loadData'], 
            appended_text_filename: str,
            ext: str = None
        ):
        QObject.__init__(self)
        self.allPosData = allPosData
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.appended_text_filename = appended_text_filename
        self.ext = ext

    @workers_base.worker_exception_handler
    def run(self):
        self.signals.initProgressBar.emit(0)
        for posData in self.allPosData:
            ext_loc = self.ext if self.ext is not None else posData.ext
            processed_filename = (
                f'{posData.basename}{posData.user_ch_name}_'
                f'{self.appended_text_filename}{ext_loc}'
            )
            processed_filepath = os.path.join(
                posData.images_path, processed_filename
            )
            self.logger.log(f'Saving {processed_filepath}...')
            processed_data = posData.preprocessedDataArray()
            if processed_data is None:
                self.logger.log(
                    f'[WARNING]: {posData.pos_foldername} does not have '
                    'preprocessed data. Skipping it.'
                )
                continue
                
            io.save_image_data(processed_filepath, processed_data)
        
        self.signals.finished.emit(self)

class SaveCombinedChannelsWorker(QObject):
    sigDebugShowImg = Signal(object)
    def __init__(
            self, 
            allPosData: Iterable['load.loadData'], 
            filename: str,
            debug: bool = False
        ):
        QObject.__init__(self)
        self.allPosData = allPosData
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.filename = filename
        self.debug = debug
    
    @workers_base.worker_exception_handler
    def run(self):
        self.signals.initProgressBar.emit(0)
        for posData in self.allPosData:
            processed_filepath = os.path.join(
                posData.images_path, self.filename
            )
            self.logger.log(f'Saving {processed_filepath}...')
            processed_data = posData.combinedChannelsDataArray()
            if processed_data is None:
                self.logger.log(
                    f'[WARNING]: {posData.pos_foldername} does not have '
                    'combined channels data. Skipping it.'
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


class CombineChannelsWorkerGUI(workers_base.CustomPreprocessWorkerGUI):
    sigDone = Signal(object, list)
    sigPreviewDone = Signal(object, list)
    sigAskLoadChannels = Signal(set, object)

    def __init__(self, mutex, waitCond, logger_func: Callable,):
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
        self.dataQ.append((data, steps, key, keep_input_data_type,output_as_segm, formula))
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

    def runJob(self, data=None, steps=None, keep_input_data_type=None, key=None,
               output_as_segm=None, formula=None):
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

        return self.applySteps(data, steps, keep_input_data_type, key, output_as_segm, formula=formula)
    
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

    @workers_base.worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log('Closing combining channels worker...')
                break
            elif self.wait:
                self.logger.log('Combining channels worker paused.')
                self.pause()
            elif len(self.dataQ) > 0:
                data, steps, key, keep_input_data_type, output_as_segm, formula = self.dataQ.pop()
                requ_steps, pos_i = self.requiredChannels(steps, key[0])
                self.emitsigAskLoadChannels(requ_steps, pos_i)
                output_imgs, out_keys = self.applySteps(
                    data, steps, keep_input_data_type, key,
                    output_as_segm=output_as_segm, formula=formula
                )
                self.sigPreviewDone.emit(output_imgs, out_keys)
                if len(self.dataQ) == 0:
                    self.wait = True
                    self.sigIsQueueEmpty.emit(True)
            else:
                self.logger.log('Combining channels worker resumed.')
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
