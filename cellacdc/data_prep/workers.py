"""Data prep module workers: AlignDataWorker (aligns the frames),
DataPrepSaveBkgrDataWorker (saves the background data),
DataPrepCropWorker and MoveTempFilesWorker. Used by
data_prep/main_window.py.
"""
import os
import shutil
from typing import Dict
import numpy as np
import pandas as pd
from qtpy.QtCore import Signal, QObject
from cellacdc import load, io
from cellacdc.myutils import image_utils
from cellacdc import core
from cellacdc.shared.workers import base as workers_base


class AlignDataWorker(QObject):
    sigWarnTifAligned = Signal(object, object, object)
    sigAskAlignSegmData = Signal()
        
    def __init__(self, posData, dataPrepWin, mutex, waitCond):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.posData = posData
        self.dataPrepWin = dataPrepWin
        self.mutex = mutex
        self.waitCond = waitCond
        self.doNotAlignSegmData = False
        self.doAbort = False
    
    def set_attr(self, align, user_ch_name):
        self.align = align
        self.user_ch_name = user_ch_name
    
    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def restart(self):
        self.waitCond.wakeAll()
    
    def emitWarnTifAligned(self, numFramesWith0s, tif, posData):
        self.sigWarnTifAligned.emit(numFramesWith0s, tif, posData)
        self.pause()
    
    def emitSigAskAlignSegmData(self):
        self.sigAskAlignSegmData.emit()
        self.pause()
    
    def _align_data(self):
        _zip = zip(self.posData.tif_paths, self.posData.npz_paths)
        aligned = False
        self.posData.all_npz_paths = [
            tif.replace('.tif', '_aligned.npz') for tif in self.posData.tif_paths
        ]
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or self.posData.loaded_shifts is None

            filename_tif = os.path.basename(tif)
            user_ch_filename = f'{self.posData.basename}{self.user_ch_name}.tif'

            if not doAlign:
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                if os.path.exists(_npz):
                    self.posData.all_npz_paths[i] = _npz
                continue
            
            if filename_tif != user_ch_filename:
                continue
            
            if not self.align:
                continue

            # Align based on user_ch_name
            aligned = True
            self.logger.log(f'Aligning: {tif}')

            tif_data = load.imread(tif)
            numFramesWith0s = self.dataPrepWin.detectTifAlignment(
                tif_data, self.posData
            )
            if self.align:
                self.emitWarnTifAligned(
                    numFramesWith0s, tif, self.posData
                )
                if self.doAbort:
                    return

            # Alignment routine
            if self.posData.SizeZ>1:
                align_func = core.align_frames_3D
                df = self.posData.segmInfo_df.loc[self.posData.filename]
                zz = df['z_slice_used_dataPrep'].to_list()
                if not self.posData.filename.endswith('aligned') and self.align:
                    # Add aligned channel to segmInfo
                    df_aligned = self.posData.segmInfo_df.rename(
                        index={self.posData.filename: f'{self.posData.filename}_aligned'}
                    )
                    self.posData.segmInfo_df = pd.concat(
                        [self.posData.segmInfo_df, df_aligned]
                    )
                    self.posData.segmInfo_df.to_csv(self.posData.segmInfo_df_csv_path)
            else:
                align_func = core.align_frames_2D
                zz = None
            
            if self.align:
                self.signals.initProgressBar.emit(len(tif_data))
                aligned_frames, shifts = align_func(
                    tif_data, slices=zz, user_shifts=self.posData.loaded_shifts,
                    sigPyqt=self.signals.progressBar
                )
                self.posData.loaded_shifts = shifts
            else:
                aligned_frames = tif_data
                
            if self.align:
                self.signals.initProgressBar.emit(0)
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                self.logger.log(f'Storing temporary file: {_npz}')
                temp_npz = self.dataPrepWin.getTempfilePath(_npz)
                io.savez_compressed(temp_npz, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_npz, _npz)
                np.save(
                    self.posData.align_shifts_path, self.posData.loaded_shifts
                )
                self.posData.all_npz_paths[i] = _npz

                self.logger.log(f'Storing temporary file: {tif}')
                temp_tif = self.dataPrepWin.getTempfilePath(tif)
                image_utils.to_tiff(temp_tif, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_tif, tif)
                self.posData.img_data = load.imread(temp_tif)

        _zip = zip(self.posData.tif_paths, self.posData.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or aligned

            if not doAlign:
                continue
            
            if tif.endswith(f'{self.user_ch_name}.tif'):
                continue
            
            if not self.align:
                continue

            # Align the other channels
            if self.posData.loaded_shifts is None:
                break

            if self.align:
                self.logger.log(f'Aligning: {tif}')
                tif_data = load.imread(tif)

            # Alignment routine
            if self.posData.SizeZ>1:
                align_func = core.align_frames_3D
                df = self.posData.segmInfo_df.loc[self.posData.filename]
                zz = df['z_slice_used_dataPrep'].to_list()
            else:
                align_func = core.align_frames_2D
                zz = None
            if self.align:
                self.signals.initProgressBar.emit(len(tif_data))
                aligned_frames, shifts = align_func(
                    tif_data, slices=zz, user_shifts=self.posData.loaded_shifts,
                    sigPyqt=self.signals.progressBar
                )
            else:
                aligned_frames = tif_data
            
            _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
            
            if self.align:
                self.signals.initProgressBar.emit(0)
                self.logger.log(f'Saving: {_npz}')
                temp_npz = self.dataPrepWin.getTempfilePath(_npz)
                io.savez_compressed(temp_npz, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_npz, _npz)
                self.posData.all_npz_paths[i] = _npz

                self.logger.log(f'Saving: {tif}')
                temp_tif = self.dataPrepWin.getTempfilePath(tif)
                image_utils.to_tiff(temp_tif, aligned_frames)
                self.dataPrepWin.storeTempFileMove(temp_tif, tif)

        if not aligned:
            return
        
        if not self.posData.segmFound:
            return
        
        # Align segmentation data accordingly
        self.segmAligned = False
        if self.posData.loaded_shifts is None or not self.align:
            return
        
        self.emitSigAskAlignSegmData()
        if self.doNotAlignSegmData:
            return
        
        self.dataPrepWin.segmAligned = True
        self.logger.log(f'Aligning: {self.posData.segm_npz_path}')
        self.posData.segm_data, shifts = core.align_frames_2D(
            self.posData.segm_data, slices=None,
            user_shifts=self.posData.loaded_shifts
        )
        self.logger.log(f'Saving: {self.posData.segm_npz_path}')
        temp_npz = self.dataPrepWin.getTempfilePath(self.posData.segm_npz_path)
        io.savez_compressed(temp_npz, self.posData.segm_data)
        self.dataPrepWin.storeTempFileMove(temp_npz, self.posData.segm_npz_path)

    @workers_base.worker_exception_handler
    def run(self):     
        self._align_data()     
        self.signals.finished.emit(self)


class DataPrepSaveBkgrDataWorker(QObject):
    def __init__(self, posData, dataPrepWin):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.posData = posData
        self.dataPrepWin = dataPrepWin
    
    @workers_base.worker_exception_handler
    def run(self):
        self.dataPrepWin.saveBkgrData(self.posData)
        self.signals.finished.emit(self)

class DataPrepCropWorker(QObject):
    def __init__(self, posData, dataPrepWin, dstPath):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.posData = posData
        self.dataPrepWin = dataPrepWin
        self.dstPath = dstPath
    
    @workers_base.worker_exception_handler
    def run(self):
        self.dataPrepWin.saveSingleCrop(
            self.posData, self.posData.cropROIs[0], self.dstPath
        )
        self.signals.finished.emit(self)


class MoveTempFilesWorker(QObject):
    def __init__(self, temp_files_to_move: Dict[os.PathLike, os.PathLike]):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        self.temp_files_to_move = temp_files_to_move
    
    @workers_base.worker_exception_handler
    def run(self):
        for src, dst in self.temp_files_to_move.items():
            self.logger.log(f'Saving channel data to: {dst}...')
            shutil.move(src, dst)
            tempDir = os.path.dirname(src)
            shutil.rmtree(tempDir)
            self.signals.progressBar.emit(1)
        self.signals.finished.emit(self)
