import sys
import os
import time
from functools import wraps

import numpy as np
import pandas as pd
import h5py
import traceback

import skimage.io

from PyQt5.QtCore import (
    pyqtSignal, QObject, QRunnable, QMutex, QWaitCondition
)

from . import load, myutils, core, measurements, gui

def worker_exception_handler(func):
    @wraps(func)
    def run(self):
        try:
            func(self)
        except Exception as error:
            try:
                self.signals.critical.emit(error)
            except AttributeError:
                self.critical.emit(error)
    return run

class workerLogger:
    def __init__(self, sigProcess):
        self.sigProcess = sigProcess

    def log(self, message):
        self.sigProcess.emit(message, 'INFO')

class signals(QObject):
    progress = pyqtSignal(str, object)
    finished = pyqtSignal(object)
    initProgressBar = pyqtSignal(int)
    progressBar = pyqtSignal(int)
    critical = pyqtSignal(object)
    dataIntegrityWarning = pyqtSignal(str)
    dataIntegrityCritical = pyqtSignal()
    sigLoadingFinished = pyqtSignal()
    sigLoadingNewChunk = pyqtSignal(object)
    resetInnerPbar = pyqtSignal(int)
    progress_tqdm = pyqtSignal(int)
    signal_close_tqdm = pyqtSignal()
    create_tqdm = pyqtSignal(int)
    innerProgressBar = pyqtSignal(int)
    sigPermissionError = pyqtSignal(str, object)
    sigSelectSegmFiles = pyqtSignal(object)
    sigSetMeasurements = pyqtSignal(object)
    sigInitAddMetrics = pyqtSignal(object)

class segmVideoWorker(QObject):
    finished = pyqtSignal(float)
    debug = pyqtSignal(object)
    critical = pyqtSignal(object)
    progressBar = pyqtSignal(int)

    def __init__(self, posData, paramWin, model, startFrameNum, stopFrameNum):
        QObject.__init__(self)
        self.minSize = paramWin.minSize
        self.minSolidity = paramWin.minSolidity
        self.maxElongation = paramWin.maxElongation
        self.applyPostProcessing = paramWin.applyPostProcessing
        self.segment2D_kwargs = paramWin.segment2D_kwargs
        self.model = model
        self.posData = posData
        self.startFrameNum = startFrameNum
        self.stopFrameNum = stopFrameNum

    @worker_exception_handler
    def run(self):
        t0 = time.time()
        img_data = self.posData.img_data[self.startFrameNum-1:self.stopFrameNum]
        for i, img in enumerate(img_data):
            frame_i = i+self.startFrameNum-1
            img = myutils.uint_to_float(img)
            lab = self.model.segment(img, **self.segment2D_kwargs)
            if self.applyPostProcessing:
                lab = core.remove_artefacts(
                    lab,
                    min_solidity=self.minSolidity,
                    min_area=self.minSize,
                    max_elongation=self.maxElongation
                )
            self.posData.segm_data[frame_i] = lab
            self.progressBar.emit(1)
        t1 = time.time()
        exec_time = t1-t0
        self.finished.emit(exec_time)

class calcMetricsWorker(QObject):
    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = signals()
        self.abort = False
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def getImgFilePath(images_path, chName):
        for file in myutils.listdir(images_path):
            pass

    @worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            tot_pos = len(pos_foldernames)
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'Position n. {p+1}/{tot_pos}'
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                basename, chNames = myutils.getBasenameAndChNames(images_path)

                # Use first found channel, it doesn't matter for metrics
                chName = chNames[0]
                file_path = myutils.getChannelFilePath(images_path, chName)

                # Load data
                posData = load.loadData(file_path, chName)
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadImgData()

                if p == 0:
                    self.mutex.lock()
                    self.signals.sigSelectSegmFiles.emit(posData)
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    if self.abort:
                        self.signals.finished.emit(self)
                        return

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_shifts=False,
                    loadSegmInfo=True,
                    load_delROIsInfo=True,
                    loadBkgrData=True,
                    loadBkgrROIs=True,
                    load_last_tracked_i=True,
                    load_metadata=True,
                    load_customAnnot=True,
                    load_customCombineMetrics=True,
                    endFilenameSegm=self.mainWin.endFilenameSegm
                )
                posData.labelSegmData()

                self.logger.log(
                    'Loaded paths:\n'
                    f'Segmentation file name: {os.path.basename(posData.segm_npz_path)}\n'
                    f'ACDC output file name {os.path.basename(posData.acdc_output_csv_path)}'
                )

                if p == 0:
                    self.mutex.lock()
                    self.signals.sigInitAddMetrics.emit(posData)
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    if self.abort:
                        self.signals.finished.emit(self)
                        return

                if self.abort:
                    self.signals.finished.emit(self)
                    return

                # (metrics_func, all_metrics_names, custom_func_dict,
                # total_metrics) = measurements.getMetricsFunc(posData)

        self.signals.finished.emit(self)

class loadDataWorker(QObject):
    def __init__(self, mainWin, user_ch_file_paths, user_ch_name, firstPosData):
        QObject.__init__(self)
        self.signals = signals()
        self.mainWin = mainWin
        self.user_ch_file_paths = user_ch_file_paths
        self.user_ch_name = user_ch_name
        self.logger = workerLogger(self.signals.progress)
        self.mutex = self.mainWin.loadDataMutex
        self.waitCond = self.mainWin.loadDataWaitCond
        self.firstPosData = firstPosData

    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def checkSelectedDataShape(self, posData, numPos):
        skipPos = False
        abort = False
        if numPos > 1:
            if posData.SizeT > 1:
                err_msg = (f'{posData.pos_foldername} contains frames over time. '
                           f'Skipping it.')
                self.logger.log(err_msg)
                self.titleLabel.setText(err_msg, color='r')
                skipPos = True
        else:
            if not posData.segmFound and posData.SizeT > 1:
                self.signals.dataIntegrityWarning.emit(posData.pos_foldername)
                self.pause()
                abort = self.abort
        return skipPos, abort

    @worker_exception_handler
    def run(self):
        data = []
        user_ch_file_paths = self.user_ch_file_paths
        numPos = len(self.user_ch_file_paths)
        user_ch_name = self.user_ch_name
        self.signals.initProgressBar.emit(len(user_ch_file_paths))
        for i, file_path in enumerate(user_ch_file_paths):
            if i == 0:
                posData = self.firstPosData
                segmFound = self.firstPosData.segmFound
                loadSegm = False
            else:
                posData = load.loadData(file_path, user_ch_name)
                loadSegm = True

            self.logger.log(f'Loading {posData.relPath}...')

            posData.loadSizeS = self.mainWin.loadSizeS
            posData.loadSizeT = self.mainWin.loadSizeT
            posData.loadSizeZ = self.mainWin.loadSizeZ
            posData.SizeT = self.mainWin.SizeT
            posData.SizeZ = self.mainWin.SizeZ
            posData.isSegm3D = self.mainWin.isSegm3D

            if i > 0:
                # First pos was already loaded in the main thread
                # see loadSelectedData function in gui.py
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadImgData()

            posData.loadOtherFiles(
                load_segm_data=loadSegm,
                load_acdc_df=True,
                load_shifts=False,
                loadSegmInfo=True,
                load_delROIsInfo=True,
                loadBkgrData=True,
                loadBkgrROIs=True,
                load_last_tracked_i=True,
                load_metadata=True,
                load_customAnnot=True,
                load_customCombineMetrics=True,
                endFilenameSegm=self.mainWin.endFilenameSegm,
                create_new_segm=self.mainWin.isNewFile,
                new_segm_filename=self.mainWin.newSegmFilename,
                labelBoolSegm=self.mainWin.labelBoolSegm
            )
            posData.labelSegmData()

            if i == 0:
                posData.segmFound = segmFound

            self.logger.log(
                'Loaded paths:\n'
                f'Segmentation file name: {os.path.basename(posData.segm_npz_path)}\n'
                f'ACDC output file name {os.path.basename(posData.acdc_output_csv_path)}'
            )

            posData.SizeT = self.mainWin.SizeT
            if self.mainWin.SizeZ > 1:
                SizeZ = posData.img_data_shape[-3]
                posData.SizeZ = SizeZ
            else:
                posData.SizeZ = 1
            posData.TimeIncrement = self.mainWin.TimeIncrement
            posData.PhysicalSizeZ = self.mainWin.PhysicalSizeZ
            posData.PhysicalSizeY = self.mainWin.PhysicalSizeY
            posData.PhysicalSizeX = self.mainWin.PhysicalSizeX
            posData.saveMetadata(
                signals=self.signals, mutex=self.mutex, waitCond=self.waitCond
            )
            SizeY, SizeX = posData.img_data_shape[-2:]

            if posData.SizeZ > 1 and posData.img_data.ndim < 3:
                posData.SizeZ = 1
                posData.segmInfo_df = None
                try:
                    os.remove(posData.segmInfo_df_csv_path)
                except FileNotFoundError:
                    pass

            posData.setBlankSegmData(
                posData.SizeT, posData.SizeZ, SizeY, SizeX
            )
            skipPos, abort = self.checkSelectedDataShape(posData, numPos)

            if skipPos:
                continue
            elif abort:
                data = 'abort'
                break

            # Allow single 2D/3D image
            if posData.SizeT < 2:
                posData.img_data = np.array([posData.img_data])
                posData.segm_data = np.array([posData.segm_data])
            img_shape = posData.img_data_shape
            posData.segmSizeT = len(posData.segm_data)
            SizeT = posData.SizeT
            SizeZ = posData.SizeZ
            self.logger.log(f'Full dataset shape = {img_shape}')
            self.logger.log(f'Loaded dataset shape = {posData.img_data.shape}')
            self.logger.log(f'Number of frames = {SizeT}')
            self.logger.log(f'Number of z-slices per frame = {SizeZ}')
            data.append(posData)
            self.signals.progressBar.emit(1)

        if not data:
            data = None
            self.signals.dataIntegrityCritical.emit()

        self.signals.finished.emit(data)
