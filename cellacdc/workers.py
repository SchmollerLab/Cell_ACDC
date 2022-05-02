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

from . import load, myutils, core

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
    sigMultiSegm = pyqtSignal(object, object, bool, object)

class loadDataWorker(QObject):
    def __init__(self, mainWin, user_ch_file_paths, user_ch_name):
        QObject.__init__(self)
        self.signals = signals()
        self.mainWin = mainWin
        self.user_ch_file_paths = user_ch_file_paths
        self.user_ch_name = user_ch_name
        self.logger = workerLogger(self.signals.progress)
        self.mutex = self.mainWin.loadDataMutex
        self.waitCond = self.mainWin.loadDataWaitCond

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
        for file_path in user_ch_file_paths:
            posData = load.loadData(file_path, user_ch_name)

            self.logger.log(f'Loading {posData.relPath}...')

            posData.loadSizeS = self.mainWin.loadSizeS
            posData.loadSizeT = self.mainWin.loadSizeT
            posData.loadSizeZ = self.mainWin.loadSizeZ
            posData.SizeT = self.mainWin.SizeT
            posData.SizeZ = self.mainWin.SizeZ
            posData.isSegm3D = self.mainWin.isSegm3D

            posData.getBasenameAndChNames()
            posData.buildPaths()
            posData.loadImgData()
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
                selectedSegmNpz=self.mainWin.selectedSegmNpz,
                create_new_segm=self.mainWin.isNewFile,
                new_segm_filename=self.mainWin.newSegmFilename,
            )

            self.logger.log(
                'Loaded paths:\n'
                f'Segmentation path: {posData.segm_npz_path}\n'
                f'ACDC output path {posData.acdc_output_csv_path}'
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
