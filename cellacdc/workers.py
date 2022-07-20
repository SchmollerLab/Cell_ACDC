import sys
import os
import time

from pprint import pprint
from functools import wraps

import numpy as np
import pandas as pd
import h5py
import traceback

import skimage.io
import skimage.measure

from PyQt5.QtCore import (
    pyqtSignal, QObject, QRunnable, QMutex, QWaitCondition
)

from . import load, myutils, core
from . import printl

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
    sigSelectSegmFiles = pyqtSignal(object, object)
    sigSetMeasurements = pyqtSignal(object)
    sigInitAddMetrics = pyqtSignal(object)
    sigUpdatePbarDesc = pyqtSignal(str)
    sigComputeVolume = pyqtSignal(int, object)
    sigAskStopFrame = pyqtSignal(object)
    sigWarnMismatchSegmDataShape = pyqtSignal(object)

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
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        exec_time = t1-t0
        self.finished.emit(exec_time)

class calcMetricsWorker(QObject):
    progressBar = pyqtSignal(int, int, float)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = signals()
        self.abort = False
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.signals.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            return True
        else:
            return False

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            tot_pos = len(pos_foldernames)
            self.allPosDataInputs = []
            posDatas = []
            self.logger.log('-'*30)
            expFoldername = os.path.basename(exp_path)

            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.signals.finished.emit(self)
                return

            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                basename, chNames = myutils.getBasenameAndChNames(
                    images_path, useExt=('.tif',)
                )

                self.signals.sigUpdatePbarDesc.emit(f'Loading {pos_path}...')

                # Use first found channel, it doesn't matter for metrics
                chName = chNames[0]
                file_path = myutils.getChannelFilePath(images_path, chName)

                # Load data
                posData = load.loadData(file_path, chName)
                posData.getBasenameAndChNames(useExt=('.tif',))

                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_acdc_df=True,
                    load_metadata=True,
                    loadSegmInfo=True
                )

                posDatas.append(posData)

                self.allPosDataInputs.append({
                    'file_path': file_path,
                    'chName': chName
                })

            if any([posData.SizeT > 1 for posData in posDatas]):
                self.mutex.lock()
                self.signals.sigAskStopFrame.emit(posDatas)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                if self.abort:
                    self.signals.finished.emit(self)
                    return
                for p, posData in enumerate(posDatas):
                    self.allPosDataInputs[p]['stopFrameNum'] = posData.stopFrameNum
                # remove posDatas from memory for timelapse data
                del posDatas
            else:
                for p, posData in enumerate(posDatas):
                    self.allPosDataInputs[p]['stopFrameNum'] = 1
            
            # Iterate pos and calculate metrics
            numPos = len(self.allPosDataInputs)
            for p, posDataInputs in enumerate(self.allPosDataInputs):
                file_path = posDataInputs['file_path']
                chName = posDataInputs['chName']
                stopFrameNum = posDataInputs['stopFrameNum']

                posData = load.loadData(file_path, chName)

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames(useExt=('.tif',))
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
                    load_customCombineMetrics=True,
                    end_filename_segm=self.mainWin.endFilenameSegm
                )
                posData.labelSegmData()
                if not posData.segmFound:
                    relPath = (
                        f'...{os.sep}{expFoldername}'
                        f'{os.sep}{posData.pos_foldername}'
                    )
                    self.logger.log(
                        f'Skipping "{relPath}" '
                        f'because segm. file was not found.'
                    )
                    continue

                if posData.SizeT > 1:
                    self.mainWin.gui.data = [None]*numPos
                else:
                    self.mainWin.gui.data = posDatas

                self.mainWin.gui.pos_i = p
                self.mainWin.gui.data[p] = posData
                self.mainWin.gui.last_pos = numPos

                self.mainWin.gui.isSegm3D = posData.getIsSegm3D()

                # Allow single 2D/3D image
                if posData.SizeT == 1:
                    posData.img_data = posData.img_data[np.newaxis]
                    posData.segm_data = posData.segm_data[np.newaxis]

                self.logger.log(
                    'Loaded paths:\n'
                    f'Segmentation file name: {os.path.basename(posData.segm_npz_path)}\n'
                    f'ACDC output file name: {os.path.basename(posData.acdc_output_csv_path)}'
                )

                if p == 0:
                    self.mutex.lock()
                    self.signals.sigInitAddMetrics.emit(posData)
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    if self.abort:
                        self.signals.finished.emit(self)
                        return

                # Load the other channels
                posData.loadedChNames = []
                for fluoChName in posData.chNames:
                    if fluoChName in self.mainWin.gui.chNamesToSkip:
                        continue

                    if fluoChName == chName:
                        filename = posData.filename
                        posData.fluo_data_dict[filename] = posData.img_data
                        posData.fluo_bkgrData_dict[filename] = posData.bkgrData
                        posData.loadedChNames.append(chName)
                        continue

                    fluo_path, filename = self.mainWin.gui.getPathFromChName(
                        fluoChName, posData
                    )
                    if fluo_path is None:
                        continue

                    self.logger.log(f'Loading {fluoChName} data...')
                    fluo_data, bkgrData = self.mainWin.gui.load_fluo_data(
                        fluo_path
                    )
                    if fluo_data is None:
                        continue

                    if posData.SizeT == 1:
                        # Add single frame for snapshot data
                        fluo_data = fluo_data[np.newaxis]

                    posData.loadedChNames.append(fluoChName)
                    posData.loadedFluoChannels.add(fluoChName)
                    posData.fluo_data_dict[filename] = fluo_data
                    posData.fluo_bkgrData_dict[filename] = bkgrData

                if not posData.fluo_data_dict:
                    self.logger.log(
                        f'The following path does not contain '
                        f'any valid image data: "{posData.pos_path}"'
                    )
                    continue

                # Recreate allData_li attribute of the gui
                posData.allData_li = []
                for frame_i, lab in enumerate(posData.segm_data[:stopFrameNum]):
                    data_dict = {
                        'labels': lab,
                        'regionprops': skimage.measure.regionprops(lab)
                    }
                    posData.allData_li.append(data_dict)

                # Signal to compute volume in the main thread
                self.mutex.lock()
                self.signals.sigComputeVolume.emit(stopFrameNum, posData)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()

                acdc_df_li = []
                keys = []
                self.signals.initProgressBar.emit(stopFrameNum)
                for frame_i, data_dict in enumerate(posData.allData_li[:stopFrameNum]):
                    if self.abort:
                        self.signals.finished.emit(self)
                        return

                    lab = data_dict['labels']
                    if not np.any(lab):
                        # Empty segmentation mask --> skip
                        continue

                    rp = data_dict['regionprops']
                    posData.lab = lab
                    posData.rp = rp

                    if posData.acdc_df is None:
                        acdc_df = myutils.getBaseAcdcDf(rp)
                    else:
                        try:
                            acdc_df = posData.acdc_df.loc[frame_i].copy()
                        except:
                            acdc_df = myutils.getBaseAcdcDf(rp)

                    acdc_df = self.mainWin.gui.saveDataWorker.addMetrics_acdc_df(
                        acdc_df, rp, frame_i, lab, posData
                    )
                    acdc_df_li.append(acdc_df)
                    key = (frame_i, posData.TimeIncrement*frame_i)
                    keys.append(key)

                    self.signals.progressBar.emit(1)

                if debugging:
                    continue

                if not acdc_df_li:
                    print('-'*30)
                    self.logger.log(
                        'All selected positions in the experiment folder '
                        f'{expFoldername} have EMPTY segmentation mask. '
                        'Metrics will not be saved.'
                    )
                    print('-'*30)
                    continue

                all_frames_acdc_df = pd.concat(
                    acdc_df_li, keys=keys,
                    names=['frame_i', 'time_seconds', 'Cell_ID']
                )
                self.mainWin.gui.saveDataWorker.addCombineMetrics_acdc_df(
                    posData, all_frames_acdc_df
                )
                self.logger.log(
                    f'Saving acdc_output to: "{posData.acdc_output_csv_path}"'
                )
                try:
                    all_frames_acdc_df.to_csv(posData.acdc_output_csv_path)
                except PermissionError:
                    traceback_str = traceback.format_exc()
                    self.mutex.lock()
                    self.signals.sigPermissionError.emit(
                        traceback_str, posData.acdc_output_csv_path
                    )
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    all_frames_acdc_df.to_csv(posData.acdc_output_csv_path)

                if self.abort:
                    self.signals.finished.emit(self)
                    return

            self.logger.log('*'*30)

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
            if not posData.segmFound and posData.SizeT > 1 and not self.mainWin.isNewFile:
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
                end_filename_segm=self.mainWin.selectedSegmEndName,
                create_new_segm=self.mainWin.isNewFile,
                new_endname=self.mainWin.newSegmEndName,
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
                signals=self.signals, mutex=self.mutex, waitCond=self.waitCond,
                additionalMetadata=self.firstPosData._additionalMetadataValues
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
            if posData.SizeT == 1:
                posData.img_data = posData.img_data[np.newaxis]
                posData.segm_data = posData.segm_data[np.newaxis]
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
