import os
import sys
import traceback
import re
import cv2
import json
from math import isnan
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd
import tkinter as tk
from tkinter import ttk
from skimage import io
import skimage.filters
from datetime import datetime
from tifffile import TiffFile
from natsort import natsorted
import skimage
import skimage.measure
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QRectF
from PyQt5.QtWidgets import (
    QApplication
)
import pyqtgraph as pg

from . import prompts, apps, myutils
from . import base_cca_df, base_acdc_df # defined in __init__.py

cca_df_colnames = list(base_cca_df.keys())

def get_user_ch_paths(images_paths, user_ch_name):
    user_ch_file_paths = []
    for images_path in images_paths:
        img_aligned_found = False
        for filename in myutils.listdir(images_path):
            if filename.find(f'{user_ch_name}_aligned.np') != -1:
                img_path_aligned = f'{images_path}/{filename}'
                img_aligned_found = True
            elif filename.find(f'{user_ch_name}.tif') != -1:
                img_path_tif = f'{images_path}/{filename}'

        if img_aligned_found:
            img_path = img_path_aligned
        else:
            img_path = img_path_tif
        user_ch_file_paths.append(img_path)
        print(f'Loading {img_path}...')
    return user_ch_file_paths

class loadData:
    def __init__(self, imgPath, user_ch_name, QParent=None):
        self.fluo_data_dict = {}
        self.fluo_bkgrData_dict = {}
        self.bkgrROIs = []
        self.loadedFluoChannels = set()
        self.parent = QParent
        self.imgPath = imgPath
        self.user_ch_name = user_ch_name
        self.images_path = os.path.dirname(imgPath)
        self.pos_path = os.path.dirname(self.images_path)
        self.exp_path = os.path.dirname(self.pos_path)
        self.pos_foldername = os.path.basename(self.pos_path)
        self.cropROI = None
        self.loadSizeT = None
        self.loadSizeZ = None
        self.multiSegmAllPos = False
        path_li = os.path.normpath(imgPath).split(os.sep)
        self.relPath = f'{f"{os.sep}".join(path_li[-3:])}'
        filename_ext = os.path.basename(imgPath)
        self.filename_ext = filename_ext
        self.filename, self.ext = os.path.splitext(filename_ext)
        self.loadLastEntriesMetadata()

    def loadLastEntriesMetadata(self):
        cellacdc_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(cellacdc_path, 'temp')
        if not os.path.exists(temp_path):
            self.last_md_df = None
            return
        csv_path = os.path.join(temp_path, 'last_entries_metadata.csv')
        if not os.path.exists(csv_path):
            self.last_md_df = None
        else:
            self.last_md_df = pd.read_csv(csv_path).set_index('Description')

    def saveLastEntriesMetadata(self):
        cellacdc_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(cellacdc_path, 'temp')
        if not os.path.exists:
            return
        csv_path = os.path.join(temp_path, 'last_entries_metadata.csv')
        self.metadata_df.to_csv(csv_path)

    def getBasenameAndChNames(self, useExt='.tif'):
        ls = myutils.listdir(self.images_path)
        selector = prompts.select_channel_name()
        self.chNames, _ = selector.get_available_channels(
            ls, self.images_path, useExt=useExt
        )
        self.basename = selector.basename

    def loadImgData(self):
        self.z0_window = 0
        self.t0_window = 0
        if self.ext == '.h5':
            self.h5f = h5py.File(self.imgPath, 'r')
            self.dset = self.h5f['data']
            self.img_data_shape = self.dset.shape
            readH5 = self.loadSizeT is not None and self.loadSizeZ is not None
            if not readH5:
                return

            is4D = self.SizeZ > 1 and self.SizeT > 1
            is3Dz = self.SizeZ > 1 and self.SizeT == 1
            is3Dt = self.SizeZ == 1 and self.SizeT > 1
            is2D = self.SizeZ == 1 and self.SizeT == 1
            if is4D:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.t0_window = 0
                self.img_data = self.dset[:self.loadSizeT, z0:z1]
            elif is3Dz:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.img_data = np.squeeze(self.dset[z0:z1])
            elif is3Dt:
                self.t0_window = 0
                self.img_data = np.squeeze(self.dset[:self.loadSizeT])
            elif is2D:
                self.img_data = np.squeeze(self.dset[:])

        elif self.ext == '.npz':
            self.img_data = np.load(self.imgPath)['arr_0']
            self.dset = self.img_data
            self.img_data_shape = self.img_data.shape
        elif self.ext == '.npy':
            self.img_data = np.load(self.imgPath)
            self.dset = self.img_data
            self.img_data_shape = self.img_data.shape
        else:
            try:
                self.img_data = skimage.io.imread(self.imgPath)
                self.dset = self.img_data
                self.img_data_shape = self.img_data.shape
            except ValueError:
                self.img_data = self._loadVideo(self.imgPath)
                self.dset = self.img_data
                self.img_data_shape = self.img_data.shape
            except Exception as e:
                traceback.print_exc()
                self.criticalExtNotValid()

    def _loadVideo(self, path):
        video = cv2.VideoCapture(path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            _, frame = video.read()
            if frame.shape[-1] == 3:
                frame = skimage.color.rgb2gray(frame)
            if i == 0:
                img_data = np.zeros((num_frames, *frame.shape), frame.dtype)
            img_data[i] = frame
        return img_data

    def detectMultiSegmNpz(self, _endswith='', multiPos=False):
        ls = myutils.listdir(self.images_path)
        if _endswith:
            self.multiSegmAllPos = True
            selectedSegmNpz_found = [
                f for f in ls if f.endswith(_endswith)
            ]
            if selectedSegmNpz_found:
                return selectedSegmNpz_found[0], False

        segm_files = [
            file for file in ls if file.endswith('segm.npz')
            or file.find('segm_raw_postproc') != -1
            or file.endswith('segm_raw.npz')
            or (file.endswith('.npz') and file.find('segm') != -1)
        ]
        is_multi_npz = len(segm_files)>1
        if is_multi_npz:
            win = apps.QDialogMultiSegmNpz(
                segm_files, self.pos_path, parent=self.parent,
                multiPos=multiPos
            )
            win.exec_()
            self.multiSegmAllPos = win.okAllPos
            if win.removeOthers:
                for file in segm_files:
                    if file == win.selectedItemText:
                        continue
                    os.remove(os.path.join(self.images_path, file))
            return win.selectedItemText, win.cancel
        else:
            return '', False

    def loadOtherFiles(
            self,
            load_segm_data=True,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=False,
            load_delROIsInfo=False,
            loadBkgrData=False,
            loadBkgrROIs=False,
            load_last_tracked_i=False,
            load_metadata=False,
            load_dataPrep_ROIcoords=False,
            getTifPath=False,
            selectedSegmNpz=''
        ):

        self.segmFound = False if load_segm_data else None
        self.acdc_df_found = False if load_acdc_df else None
        self.shiftsFound = False if load_shifts else None
        self.segmInfoFound = False if loadSegmInfo else None
        self.delROIsInfoFound = False if load_delROIsInfo else None
        self.bkgrDataFound = False if loadBkgrData else None
        self.bkgrROisFound = False if loadBkgrROIs else None
        self.last_tracked_i_found = False if load_last_tracked_i else None
        self.metadataFound = False if load_metadata else None
        self.dataPrep_ROIcoordsFound = False if load_dataPrep_ROIcoords else None
        self.TifPathFound = False if getTifPath else None
        ls = myutils.listdir(self.images_path)

        for file in ls:
            filePath = os.path.join(self.images_path, file)
            if selectedSegmNpz:
                is_segm_file = file == selectedSegmNpz
            else:
                is_segm_file = file.endswith('segm.npz')

            if load_segm_data and is_segm_file:
                self.segmFound = True
                self.segm_npz_path = filePath
                self.segm_data = np.load(filePath)['arr_0']
                squeezed_arr = np.squeeze(self.segm_data)
                if squeezed_arr.shape != self.segm_data.shape:
                    self.segm_data = squeezed_arr
                    np.savez_compressed(filePath, squeezed_arr)
            elif getTifPath and file.find(f'{self.user_ch_name}.tif')!=-1:
                self.tif_path = filePath
                self.TifPathFound = True
            elif load_acdc_df and file.endswith('acdc_output.csv'):
                self.acdc_df_found = True
                acdc_df = pd.read_csv(
                      filePath, index_col=['frame_i', 'Cell_ID']
                )
                acdc_df = self.BooleansTo0s1s(acdc_df, inplace=True)
                acdc_df = self.intToBoolean(acdc_df)
                self.acdc_df = acdc_df
            elif load_shifts and file.endswith('align_shift.npy'):
                self.shiftsFound = True
                self.loaded_shifts = np.load(filePath)
            elif loadSegmInfo and file.endswith('segmInfo.csv'):
                self.segmInfoFound = True
                df = pd.read_csv(filePath)
                if 'filename' not in df.columns:
                    df['filename'] = self.filename
                self.segmInfo_df = df.set_index(['filename', 'frame_i'])
            elif load_delROIsInfo and file.endswith('delROIsInfo.npz'):
                self.delROIsInfoFound = True
                self.delROIsInfo_npz = np.load(filePath)
            elif loadBkgrData and file.endswith(f'{self.filename}_bkgrRoiData.npz'):
                self.bkgrDataFound = True
                self.bkgrData = np.load(filePath)
            elif loadBkgrROIs and file.endswith('dataPrep_bkgrROIs.json'):
                self.bkgrROisFound = True
                with open(filePath) as json_fp:
                    bkgROIs_states = json.load(json_fp)

                for roi_state in bkgROIs_states:
                    Y, X = self.img_data.shape[-2:]
                    roi = pg.ROI(
                        [0, 0], [1, 1],
                        rotatable=False,
                        removable=False,
                        pen=pg.mkPen(color=(150,150,150)),
                        maxBounds=QRectF(QRect(0,0,X,Y))
                    )
                    roi.setState(roi_state)
                    self.bkgrROIs.append(roi)
            elif load_dataPrep_ROIcoords and file.endswith('dataPrepROIs_coords.csv'):
                df = pd.read_csv(filePath)
                if 'description' in df.columns:
                    df = df.set_index('description')
                    if 'value' in df.columns:
                        self.dataPrep_ROIcoordsFound = True
                        self.dataPrep_ROIcoords = df
            elif (load_metadata and file.endswith('metadata.csv')
                and not file.endswith('segm_metadata.csv')
                ):
                self.metadataFound = True
                self.metadata_df = pd.read_csv(filePath).set_index('Description')
                self.extractMetadata()

        if load_last_tracked_i:
            self.last_tracked_i_found = True
            try:
                self.last_tracked_i = max(self.acdc_df.index.get_level_values(0))
            except AttributeError as e:
                # traceback.print_exc()
                self.last_tracked_i = None
        self.setNotFoundData()


    def extractMetadata(self):
        if 'SizeT' in self.metadata_df.index:
            self.SizeT = int(self.metadata_df.at['SizeT', 'values'])
        elif self.last_md_df is not None and 'SizeT' in self.last_md_df.index:
            self.SizeT = int(self.last_md_df.at['SizeT', 'values'])
        else:
            self.SizeT = 1

        if 'SizeZ' in self.metadata_df.index:
            self.SizeZ = int(self.metadata_df.at['SizeZ', 'values'])
        elif self.last_md_df is not None and 'SizeZ' in self.last_md_df.index:
            self.SizeZ = int(self.last_md_df.at['SizeZ', 'values'])
        else:
            self.SizeZ = 1

        if 'TimeIncrement' in self.metadata_df.index:
            self.TimeIncrement = float(
                self.metadata_df.at['TimeIncrement', 'values']
            )
        elif self.last_md_df is not None and 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(self.last_md_df.at['TimeIncrement', 'values'])
        else:
            self.TimeIncrement = 1

        if 'PhysicalSizeX' in self.metadata_df.index:
            self.PhysicalSizeX = float(
                self.metadata_df.at['PhysicalSizeX', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(self.last_md_df.at['PhysicalSizeX', 'values'])
        else:
            self.PhysicalSizeX = 1

        if 'PhysicalSizeY' in self.metadata_df.index:
            self.PhysicalSizeY = float(
                self.metadata_df.at['PhysicalSizeY', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(self.last_md_df.at['PhysicalSizeY', 'values'])
        else:
            self.PhysicalSizeY = 1

        if 'PhysicalSizeZ' in self.metadata_df.index:
            self.PhysicalSizeZ = float(
                self.metadata_df.at['PhysicalSizeZ', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(self.last_md_df.at['PhysicalSizeZ', 'values'])
        else:
            self.PhysicalSizeZ = 1

        load_last_segmSizeT = (
            self.last_md_df is not None
            and 'segmSizeT' in self.last_md_df.index
            and self.SizeT > 1
        )
        if 'segmSizeT' in self.metadata_df.index:
             self.segmSizeT = int(
                 self.metadata_df.at['segmSizeT', 'values']
             )
        elif load_last_segmSizeT:
            self.segmSizeT = int(self.last_md_df.at['segmSizeT', 'values'])
        else:
            self.segmSizeT = self.SizeT

    def setNotFoundData(self):
        if self.segmFound is not None and not self.segmFound:
            self.segm_data = None
        if self.acdc_df_found is not None and not self.acdc_df_found:
            self.acdc_df = None
        if self.shiftsFound is not None and not self.shiftsFound:
            self.loaded_shifts = None
        if self.segmInfoFound is not None and not self.segmInfoFound:
            self.segmInfo_df = None
        if self.delROIsInfoFound is not None and not self.delROIsInfoFound:
            self.delROIsInfo_npz = None
        if self.bkgrDataFound is not None and not self.bkgrDataFound:
            self.bkgrData = None
        if self.dataPrep_ROIcoordsFound is not None and not self.dataPrep_ROIcoordsFound:
            self.dataPrep_ROIcoords = None
        if self.last_tracked_i_found is not None and not self.last_tracked_i_found:
            self.last_tracked_i = None
        if self.TifPathFound is not None and not self.TifPathFound:
            self.tif_path = None

        if self.metadataFound is None:
            # Loading metadata was not requested
            return

        if self.metadataFound:
            return

        if self.img_data.ndim == 3:
            if len(self.img_data) > 49:
                self.SizeT, self.SizeZ = len(self.img_data), 1
            else:
                self.SizeT, self.SizeZ = 1, len(self.img_data)
        elif self.img_data.ndim == 4:
            self.SizeT, self.SizeZ = self.img_data.shape[:2]
        else:
            self.SizeT, self.SizeZ = 1, 1

        self.TimeIncrement = 1.0
        self.PhysicalSizeX = 1.0
        self.PhysicalSizeY = 1.0
        self.PhysicalSizeZ = 1.0
        self.segmSizeT = self.SizeT
        self.metadata_df = None

        if self.last_md_df is None:
            # Last entered values do not exists
            return

        # Since metadata was not found use the last entries saved in temp folder
        # if 'SizeT' in self.last_md_df.index and self.SizeT == 1:
        #     self.SizeT = int(self.last_md_df.at['SizeT', 'values'])
        # if 'SizeZ' in self.last_md_df.index and self.SizeZ == 1:
        #     self.SizeZ = int(self.last_md_df.at['SizeZ', 'values'])
        if 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(self.last_md_df.at['TimeIncrement', 'values'])
        if 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(self.last_md_df.at['PhysicalSizeX', 'values'])
        if 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(self.last_md_df.at['PhysicalSizeY', 'values'])
        if 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(self.last_md_df.at['PhysicalSizeZ', 'values'])
        if 'segmSizeT' in self.last_md_df.index:
            self.segmSizeT = int(self.last_md_df.at['segmSizeT', 'values'])

    def check_acdc_df_integrity(self):
        check = (
            self.acdc_df_found is not None # acdc_df was laoded if present
            and self.acdc_df is not None # acdc_df was present
            and self.segmFound is not None # segm data was loaded if present
            and self.segm_data is not None # segm data was present
        )
        if check:
            if self.SizeT > 1:
                annotates_frames = self.acdc_df.index.get_level_values(0)
                for frame_i, lab in enumerate(self.segm_data):
                    if frame_i not in annotates_frames:
                        break
                    self._fix_acdc_df(lab, frame_i=frame_i)
            else:
                lab = self.segm_data
                self._fix_acdc_df(lab)

    def _fix_acdc_df(self, lab, frame_i=0):
        rp = skimage.measure.regionprops(lab)
        segm_IDs = [obj.label for obj in rp]
        acdc_df_IDs = self.acdc_df.loc[frame_i].index
        try:
            cca_df = self.acdc_df[cca_df_colnames]
        except KeyError:
            # Columns not present because not annotated --> no need to fix
            return

        for obj in rp:
            ID = obj.label
            if ID in acdc_df_IDs:
                continue
            idx = (frame_i, ID)
            self.acdc_df.loc[idx, cca_df_colnames] = base_cca_df.values()
            for col, val in base_acdc_df.items():
                if not isnan(self.acdc_df.at[idx, col]):
                    continue
                self.acdc_df.at[idx, col] = val
            y, x = obj.centroid
            self.acdc_df.at[idx, 'x_centroid'] = x
            self.acdc_df.at[idx, 'y_centroid'] = y

    def saveSegmHyperparams(self, hyperparams):
        df = pd.DataFrame(hyperparams, index=['value'])
        df.to_csv(self.segm_hyperparams_csv_path)

    def buildPaths(self):
        if self.basename.endswith('_'):
            basename = self.basename
        else:
            basename = f'{self.basename}_'
        base_path = os.path.join(self.images_path, basename)
        self.slice_used_align_path = f'{base_path}slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}slice_segm.csv'
        self.align_npz_path = f'{base_path}{self.user_ch_name}_aligned.npz'
        self.align_old_path = f'{base_path}phc_aligned.npy'
        self.align_shifts_path = f'{base_path}align_shift.npy'
        self.segm_npz_path = f'{base_path}segm.npz'
        self.last_tracked_i_path = f'{base_path}last_tracked_i.txt'
        self.acdc_output_csv_path = f'{base_path}acdc_output.csv'
        self.segmInfo_df_csv_path = f'{base_path}segmInfo.csv'
        self.delROIs_info_path = f'{base_path}delROIsInfo.npz'
        self.dataPrepROI_coords_path = f'{base_path}dataPrepROIs_coords.csv'
        # self.dataPrepBkgrValues_path = f'{base_path}dataPrep_bkgrValues.csv'
        self.dataPrepBkgrROis_path = f'{base_path}dataPrep_bkgrROIs.json'
        self.metadata_csv_path = f'{base_path}metadata.csv'
        self.mot_events_path = f'{base_path}mot_events'
        self.mot_metrics_csv_path = f'{base_path}mot_metrics'
        self.raw_segm_npz_path = f'{base_path}segm_raw.npz'
        self.raw_postproc_segm_path = f'{base_path}segm_raw_postproc'
        self.post_proc_mot_metrics = f'{base_path}post_proc_mot_metrics'
        self.segm_hyperparams_csv_path = f'{base_path}segm_hyperparams.csv'
        self.btrack_tracks_h5_path = f'{base_path}btrack_tracks.h5'

    def setBlankSegmData(self, SizeT, SizeZ, SizeY, SizeX):
        Y, X = self.img_data.shape[-2:]
        if self.segmFound is not None and not self.segmFound:
            if SizeT > 1:
                self.segm_data = np.zeros((SizeT, Y, X), int)
            else:
                self.segm_data = np.zeros((Y, X), int)

    def loadAllImgPaths(self):
        tif_paths = []
        npy_paths = []
        npz_paths = []
        basename = self.basename[0:-1]
        for filename in myutils.listdir(self.images_path):
            file_path = os.path.join(self.images_path, filename)
            f, ext = os.path.splitext(filename)
            m = re.match(fr'{basename}.*\.tif', filename)
            if m is not None:
                tif_paths.append(file_path)
                # Search for npy fluo data
                npy = f'{f}_aligned.npy'
                npz = f'{f}_aligned.npz'
                npy_found = False
                npz_found = False
                for name in myutils.listdir(self.images_path):
                    _path = os.path.join(self.images_path, name)
                    if name == npy:
                        npy_paths.append(_path)
                        npy_found = True
                    if name == npz:
                        npz_paths.append(_path)
                        npz_found = True
                if not npy_found:
                    npy_paths.append(None)
                if not npz_found:
                    npz_paths.append(None)
        self.tif_paths = tif_paths
        self.npy_paths = npy_paths
        self.npz_paths = npz_paths

    def checkH5memoryFootprint(self):
        if self.ext != '.h5':
            return 0
        else:
            Y, X = self.dset.shape[-2:]
            size = self.loadSizeT*self.loadSizeZ*Y*X
            itemsize = self.dset.dtype.itemsize
            required_memory = size*itemsize
            return required_memory

    def askInputMetadata(
            self, numPos,
            ask_SizeT=False,
            ask_TimeIncrement=False,
            ask_PhysicalSizes=False,
            singlePos=False,
            save=False
        ):
        font = QtGui.QFont()
        font.setPointSize(10)
        metadataWin = apps.QDialogMetadata(
            self.SizeT, self.SizeZ, self.TimeIncrement,
            self.PhysicalSizeZ, self.PhysicalSizeY, self.PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes,
            parent=self.parent, font=font, imgDataShape=self.img_data_shape,
            posData=self, singlePos=singlePos
        )
        metadataWin.setFont(font)
        metadataWin.exec_()
        if metadataWin.cancel:
            return False

        self.SizeT = metadataWin.SizeT
        self.SizeZ = metadataWin.SizeZ

        self.loadSizeS = numPos
        self.loadSizeT = metadataWin.SizeT
        self.loadSizeZ = metadataWin.SizeZ

        source = metadataWin if ask_TimeIncrement else self
        self.TimeIncrement = source.TimeIncrement

        source = metadataWin if ask_PhysicalSizes else self
        self.PhysicalSizeZ = source.PhysicalSizeZ
        self.PhysicalSizeY = source.PhysicalSizeY
        self.PhysicalSizeX = source.PhysicalSizeX
        if save:
            self.saveMetadata()
        return True

    def transferMetadata(self, from_posData):
        self.SizeT = from_posData.SizeT
        self.SizeZ = from_posData.SizeZ
        self.PhysicalSizeZ = from_posData.PhysicalSizeZ
        self.PhysicalSizeY = from_posData.PhysicalSizeY
        self.PhysicalSizeX = from_posData.PhysicalSizeX

    def saveMetadata(self):
        if self.metadata_df is None:
            self.metadata_df = pd.DataFrame({
                'SizeT': self.SizeT,
                'SizeZ': self.SizeZ,
                'TimeIncrement': self.TimeIncrement,
                'PhysicalSizeZ': self.PhysicalSizeZ,
                'PhysicalSizeY': self.PhysicalSizeY,
                'PhysicalSizeX': self.PhysicalSizeX,
                'segmSizeT': self.segmSizeT
            }, index=['values']).T
            self.metadata_df.index.name = 'Description'
        else:
            self.metadata_df.at['SizeT', 'values'] = self.SizeT
            self.metadata_df.at['SizeZ', 'values'] = self.SizeZ
            self.metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
            self.metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
            self.metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
            self.metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
            self.metadata_df.at['segmSizeT', 'values'] = self.segmSizeT
        try:
            self.metadata_df.to_csv(self.metadata_csv_path)
        except PermissionError:
            msg = QtGui.QMessageBox()
            warn_cca = msg.critical(
                self.parent, 'Permission denied',
                f'The below file is open in another app (Excel maybe?).\n\n'
                f'{self.metadata_csv_path}\n\n'
                'Close file and then press "Ok".',
                msg.Ok
            )
            self.metadata_df.to_csv(self.metadata_csv_path)

        self.saveLastEntriesMetadata()

    @staticmethod
    def BooleansTo0s1s(acdc_df, csv_path=None, inplace=True):
        """
        Function used to convert "FALSE" strings and booleans to 0s and 1s
        to avoid pandas interpreting as strings or numbers
        """
        if not inplace:
            acdc_df = acdc_df.copy()
        colsToCast = ['is_cell_dead', 'is_cell_excluded']
        for col in colsToCast:
            isInt = pd.api.types.is_integer_dtype(acdc_df[col])
            isFloat = pd.api.types.is_float_dtype(acdc_df[col])
            isObject = pd.api.types.is_object_dtype(acdc_df[col])
            isString = pd.api.types.is_string_dtype(acdc_df[col])
            isBool = pd.api.types.is_bool_dtype(acdc_df[col])
            if isFloat or isBool:
                acdc_df[col] = acdc_df[col].astype(int)
            elif isString or isObject:
                acdc_df[col] = (acdc_df[col].str.lower() == 'true').astype(int)
        if csv_path is not None:
            acdc_df.to_csv(csv_path)
        return acdc_df

    def intToBoolean(self, acdc_df):
        colsToCast = ['is_cell_dead', 'is_cell_excluded']
        for col in colsToCast:
            acdc_df[col] = acdc_df[col] > 0
        return acdc_df


    def criticalExtNotValid(self):
        err_title = f'File extension {self.ext} not valid.'
        err_msg = (
            f'The requested file {self.relPath}\n'
            'has an invalid extension.\n\n'
            'Valid extensions are .tif, .tiff, .npy or .npz'
        )
        if self.parent is None:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            raise FileNotFoundError(err_title)
        else:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            msg = QtGui.QMessageBox()
            msg.critical(self.parent, err_title, err_msg, msg.Ok)
            return None

class select_exp_folder:
    def QtPrompt(self, parentQWidget, values,
                       current=0,
                       title='Select Position folder',
                       CbLabel="Select \'Position_n\' folder to analyze:",
                       showinexplorer_button=False,
                       full_paths=None, allow_abort=True,
                       show=False, toggleMulti=False):
        font = QtGui.QFont()
        font.setPointSize(10)
        win = apps.QtSelectItems(
            title, values, '', CbLabel=CbLabel, parent=parentQWidget
        )
        win.setFont(font)
        toFront = win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
        win.setWindowState(toFront)
        win.activateWindow()
        if toggleMulti:
            win.multiPosButton.setChecked(True)
        win.exec_()
        self.was_aborted = win.cancel
        if not win.cancel:
            self.selected_pos = [
                self.pos_foldernames[idx]
                for idx in win.selectedItemsIdx
            ]

    def get_values_segmGUI(self, exp_path):
        pos_foldernames = natsorted(myutils.listdir(exp_path))
        pos_foldernames = [f for f in pos_foldernames
                           if f.find('Position_')!=-1
                           and os.path.isdir(f'{exp_path}/{f}')]
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            last_tracked_i_found = False
            pos_path = f'{exp_path}/{pos}'
            if os.path.isdir(pos_path):
                images_path = f'{exp_path}/{pos}/Images'
                filenames = myutils.listdir(images_path)
                for filename in filenames:
                    if filename.find('acdc_output.csv') != -1:
                        last_tracked_i_found = True
                        acdc_df_path = f'{images_path}/{filename}'
                        acd_df = pd.read_csv(acdc_df_path)
                        last_tracked_i = max(acd_df['frame_i'])
                if last_tracked_i_found:
                    values.append(f'{pos} (Last tracked frame: {last_tracked_i+1})')
                else:
                    values.append(pos)
        self.values = values
        return values

    def get_values_cca(self, exp_path):
        pos_foldernames = natsorted(myutils.listdir(exp_path))
        pos_foldernames = [pos for pos in pos_foldernames
                               if re.match(r'Position_(\d+)', pos)]
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            cc_stage_found = False
            pos_path = f'{exp_path}/{pos}'
            if os.path.isdir(pos_path):
                images_path = f'{exp_path}/{pos}/Images'
                filenames = myutils.listdir(images_path)
                for filename in filenames:
                    if filename.find('cc_stage.csv') != -1:
                        cc_stage_found = True
                        cc_stage_path = f'{images_path}/{filename}'
                        cca_df = pd.read_csv(
                            cc_stage_path, index_col=['frame_i', 'Cell_ID']
                        )
                        last_analyzed_frame_i = (
                            cca_df.index.get_level_values(0).max()
                        )
                if cc_stage_found:
                    values.append(f'{pos} (Last analyzed frame: '
                                  f'{last_analyzed_frame_i})')
                else:
                    values.append(pos)
        self.values = values
        return values

    def _close(self):
        val = self.pos_n_sv.get()
        idx = list(self.values).index(val)
        if self.full_paths is None:
            self.selected_pos = [self.pos_foldernames[idx]]
        else:
            self.TIFFs_path = self.full_paths[idx]
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        self.selected_pos = [None]
        self.was_aborted = True
        self.root.quit()
        self.root.destroy()
        if self.allow_abort:
            exit('Execution aborted by the user')


def load_shifts(parent_path, basename=None):
    shifts_found = False
    shifts = None
    if basename is None:
        for filename in myutils.listdir(parent_path):
            if filename.find('align_shift.npy')>0:
                shifts_found = True
                shifts_path = os.path.join(parent_path, filename)
                shifts = np.load(shifts_path)
    else:
        align_shift_fn = f'{basename}_align_shift.npy'
        if align_shift_fn in myutils.listdir(parent_path):
            shifts_found = True
            shifts_path = os.path.join(parent_path, align_shift_fn)
            shifts = np.load(shifts_path)
        else:
            shifts = None
    return shifts, shifts_found
