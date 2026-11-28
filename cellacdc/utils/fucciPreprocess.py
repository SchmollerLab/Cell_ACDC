import os

import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils, load

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox, QGridLayout
from cellacdc.shared.dialogs import preprocessing as dialogs_preprocessing
import concurrent.futures
from functools import partial
import numpy as np
import skimage.io
import skimage.measure
import skimage.exposure
from tqdm import tqdm
from qtpy.QtCore import Signal
from cellacdc import io
from cellacdc import core
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class FucciPreprocessUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = workers.FucciPreprocessWorker(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigAskParams.connect(self.askSelectParams)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def askSelectParams(self, exp_path, pos_foldernames):
        channel_names = set()
        df_metadata = None
        for p, pos in enumerate(pos_foldernames):
            pos_path = os.path.join(exp_path, pos)
            images_path = os.path.join(pos_path, 'Images')
            basename, chNames = myutils.getBasenameAndChNames(images_path)
            channel_names.update(chNames)
            if df_metadata is not None:
                continue
            
            self.worker.basename = basename
            df_metadata = load.load_metadata_df(images_path)
        
        if len(channel_names) < 2:
            txt = (
                'At least two channels are needed to run the FUCCI '
                'pre-processing.'
            )
            self.logger.error(txt)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.critical(self, 'Error', html_utils.paragraph(txt))
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        win = apps.FucciPreprocessDialog(
            channel_names,
            df_metadata=df_metadata,
            parent=self
        )
        win.exec_()
        
        self.worker.firstChannelName = win.firstChannelName
        self.worker.secondChannelName = win.secondChannelName
        fucciFilterKwargs = win.function_kwargs
        self.worker.fucciFilterKwargs = fucciFilterKwargs
        
        if fucciFilterKwargs['do_basicpy_background_correction']:
            from cellacdc import preprocess
            preprocess._init_basicpy_background_correction(parent=self)
        
        self.worker.abort = win.cancel
        self.worker.waitCond.wakeAll()
    
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename):
        helpText = (
            """
            The combined and preprocessed image file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new name. The rest of 
            the name will be the same as the original file.
            """
        )
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>new combined channels</b> file:',
            defaultEntry='fucci_combined',
            helpText=helpText, 
            allowEmpty=False,
            parent=self
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.appendedName = win.entryText
        self.worker.waitCond.wakeAll()
    
    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'FUCCI pre-processing aborted.'
        else:
            txt = 'FUCCI pre-processing completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class FucciPreprocessDialog(dialogs_preprocessing.FunctionParamsDialog):
    def __init__(
            self, channel_names,
            df_metadata=None,
            parent=None,
        ):
        
        from cellacdc.preprocess import fucci_filter
        params_argspecs = myutils.get_function_argspec(fucci_filter)
        
        super().__init__(
            params_argspecs, 
            function_name='FUCCI pre-processing', 
            df_metadata=df_metadata,
            parent=parent,
        )
        
        channelNamesLayout = QGridLayout()
        
        row = 0
        label = QLabel('First channel name:  ')
        channelNamesLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)
        self.firstChNameWidget = QComboBox()
        self.firstChNameWidget.addItems(channel_names)
        channelNamesLayout.addWidget(self.firstChNameWidget, row, 1)
        
        row += 1
        label = QLabel('Second channel name:  ')
        channelNamesLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)
        self.secondChNameWidget = QComboBox()
        self.secondChNameWidget.addItems(channel_names)
        self.secondChNameWidget.setCurrentText(list(channel_names)[1])
        channelNamesLayout.addWidget(self.secondChNameWidget, row, 1)
        
        channelNamesLayout.setColumnStretch(0, 0)
        channelNamesLayout.setColumnStretch(1, 1)
        
        self.mainLayout.insertLayout(0, channelNamesLayout)
        self.mainLayout.insertWidget(1, widgets.QHLine())
    
    def ok_cb(self):
        self.firstChannelName = self.firstChNameWidget.currentText()
        self.secondChannelName = self.secondChNameWidget.currentText()
        super().ok_cb()


class FucciPreprocessWorker(utils_base.BaseWorkerUtil):
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
            func = partial(
                core.fucci_pipeline_executor_map, **filter_kwargs
            )
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
    
    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f'Setup parameters'
            
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
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                
                self.logger.log(
                    f'Loading {self.firstChannelName} channel data...'
                )
                first_ch_filepath = load.get_filename_from_channel(
                    images_path, self.firstChannelName
                )
                first_ch_data = load.load_image_file(first_ch_filepath)
                
                self.logger.log(
                    f'Loading {self.secondChannelName} channel data...'
                )
                second_ch_filepath = load.get_filename_from_channel(
                    images_path, self.secondChannelName
                )
                second_ch_data = load.load_image_file(second_ch_filepath)
                
                self.logger.log(
                    'Applying FUCCI pre-processing pipeline...\n'
                )
                processed_data = self.applyPipeline(
                    first_ch_data, second_ch_data, self.fucciFilterKwargs
                )
                
                basename, chNames = myutils.getBasenameAndChNames(images_path)
                _, ext = os.path.splitext(first_ch_filepath)
                processed_filename = f'{basename}{appendedName}{ext}'
                processed_filepath = os.path.join(
                    images_path, processed_filename
                )
                self.logger.log(
                    f'Saving pre-processed images to "{processed_filepath}"...'
                )
                io.save_image_data(processed_filepath, processed_data)
                                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)
