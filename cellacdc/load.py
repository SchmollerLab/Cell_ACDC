import os
import sys
import traceback
import re
import cv2
import json
import h5py
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
    QApplication, QMessageBox
)
import pyqtgraph as pg

from . import prompts, apps, myutils, widgets, measurements, config
from . import base_cca_df, base_acdc_df, html_utils, temp_path, printl

cca_df_colnames = list(base_cca_df.keys())
acdc_df_bool_cols = [
    'is_cell_dead',
    'is_cell_excluded',
    'is_history_known',
    'corrected_assignment'
]

additional_metadata_path = os.path.join(temp_path, 'additional_metadata.json')
last_entries_metadata_path = os.path.join(temp_path, 'last_entries_metadata.csv')

def read_json(json_path, logger_func=print, desc='custom annotations'):
    json_data = {}
    try:
        with open(json_path) as file:
            json_data = json.load(file)
    except Exception as e:
        print('****************************')
        logger_func(traceback.format_exc())
        print('****************************')
        logger_func(f'json path: {json_path}')
        print('----------------------------')
        logger_func(f'Error while reading saved {desc}. See above')
        print('============================')
    return json_data

def read_config_metrics(ini_path):
    configPars = config.ConfigParser()
    configPars.read(ini_path)
    if 'equations' not in configPars:
        configPars['equations'] = {}

    if 'mixed_channels_equations' not in configPars:
        configPars['mixed_channels_equations'] = {}

    if 'user_path_equations' not in configPars:
        configPars['user_path_equations'] = {}
    
    return configPars

def add_configPars_metrics(configPars_ref, configPars2_to_add):
    configPars_ref['equations'] = {
        **configPars2_to_add['equations'], **configPars_ref['equations']
    }
    configPars_ref['mixed_channels_equations'] = {
        **configPars2_to_add['mixed_channels_equations'], 
        **configPars_ref['mixed_channels_equations']
    }
    configPars_ref['user_path_equations'] = {
        **configPars2_to_add['user_path_equations'], 
        **configPars_ref['user_path_equations']
    }
    keep_user_path_equations = {
        key:val for key, val in configPars_ref['user_path_equations'].items()
        if key not in configPars_ref['equations']
    } 
    configPars_ref['user_path_equations'] = keep_user_path_equations
    return configPars_ref

def h5py_iter(g, prefix=''):
    for key, item in g.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_iter(item, path)

def h5dump_to_arr(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as f:
        for (path, dset) in h5py_iter(f):
            data_dict[dset.name] = dset[()]
    sorted_keys = natsorted(data_dict.keys())
    arr = np.array([data_dict[key] for key in sorted_keys])
    return arr

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

def get_segm_files(images_path):
    ls = myutils.listdir(images_path)

    segm_files = [
        file for file in ls if file.endswith('segm.npz')
        or file.find('segm_raw_postproc') != -1
        or file.endswith('segm_raw.npz')
        or (file.endswith('.npz') and file.find('segm') != -1)
    ]
    return segm_files

def get_filename_from_channel(images_path, channel_name):
    h5_aligned_path = ''
    h5_path = ''
    npz_aligned_path = ''
    tif_path = ''
    for file in myutils.listdir(images_path):
        channelDataPath = os.path.join(images_path, file)
        if file.endswith(f'{channel_name}_aligned.h5'):
            h5_aligned_path = channelDataPath
        elif file.endswith(f'{channel_name}.h5'):
            h5_path = channelDataPath
        elif file.endswith(f'{channel_name}_aligned.npz'):
            npz_aligned_path = channelDataPath
        elif file.endswith(f'{channel_name}.tif'):
            tif_path = channelDataPath
    
    if h5_aligned_path:
        return h5_aligned_path
    elif h5_path:
        return h5_path
    elif npz_aligned_path:
        return npz_aligned_path
    elif tif_path:
        return tif_path
    else:
        return ''

def get_existing_segm_endnames(basename, segm_files):
    existing_endnames = []
    for f in segm_files:
        filename, _ = os.path.splitext(f)
        endname = filename[len(basename):]
        # Remove the 'segm_' part
        # endname = endname.replace('segm', '', 1).replace('_', '', 1)
        # endname = endname.replace('_', '', 1)
        existing_endnames.append(endname)
    return existing_endnames

def get_endname_from_channels(filename, channels):
    endname = None
    for ch in channels:
        ch_aligned = f'{ch}_aligned'
        m = re.search(fr'{ch}(.\w+)*$', filename)
        m_aligned = re.search(fr'{ch_aligned}(.\w+)*$', filename)
        if m_aligned is not None:
            return endname
        elif m is not None:
            return endname

def pd_int_to_bool(acdc_df, colsToCast=None):
    if colsToCast is None:
        colsToCast = acdc_df_bool_cols
    for col in colsToCast:
        try:
            acdc_df[col] = acdc_df[col] > 0
        except KeyError:
            continue
    return acdc_df

def pd_bool_to_int(acdc_df, colsToCast=None, csv_path=None, inplace=True):
    """
    Function used to convert "FALSE" strings and booleans to 0s and 1s
    to avoid pandas interpreting as strings or numbers
    """
    if not inplace:
        acdc_df = acdc_df.copy()
    if colsToCast is None:
        colsToCast = acdc_df_bool_cols
    for col in colsToCast:   
        try:
            series = acdc_df[col]
            notna_idx = series.notna()
            notna_series = series.dropna()
            isInt = pd.api.types.is_integer_dtype(notna_series)
            isFloat = pd.api.types.is_float_dtype(notna_series)
            isObject = pd.api.types.is_object_dtype(notna_series)
            isString = pd.api.types.is_string_dtype(notna_series)
            isBool = pd.api.types.is_bool_dtype(notna_series)
            if isFloat or isBool:
                acdc_df.loc[notna_idx, col] = acdc_df.loc[notna_idx, col].astype(int)
            elif isString or isObject:
                # Object data type can have mixed data types so we first convert
                # to strings
                acdc_df.loc[notna_idx, col] = acdc_df.loc[notna_idx, col].astype(str)
                acdc_df.loc[notna_idx, col] = (
                    acdc_df.loc[notna_idx, col].str.lower() == 'true'
                ).astype(int)
        except KeyError:
            continue
        except Exception as e:
            printl(col)
            traceback.print_exc()
    if csv_path is not None:
        acdc_df.to_csv(csv_path)
    return acdc_df

class loadData:
    def __init__(self, imgPath, user_ch_name, relPathDepth=3, QParent=None):
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
        self.pos_num = self.getPosNum()
        self.cropROI = None
        self.loadSizeT = None
        self.loadSizeZ = None
        self.multiSegmAllPos = False
        path_li = os.path.normpath(imgPath).split(os.sep)
        self.relPath = f'{f"{os.sep}".join(path_li[-relPathDepth:])}'
        filename_ext = os.path.basename(imgPath)
        self.filename_ext = filename_ext
        self.filename, self.ext = os.path.splitext(filename_ext)
        self._additionalMetadataValues = None
        self.loadLastEntriesMetadata()

    def getPosNum(self):
        try:
            pos_num = int(re.findall('Position_(\d+)', self.pos_foldername))[0]
        except Exception:
            pos_num = 0
        return pos_num

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
        if not os.path.exists(temp_path):
            return
        self.metadata_df.to_csv(last_entries_metadata_path)

    def getBasenameAndChNames(self, useExt=None):
        ls = myutils.listdir(self.images_path)
        selector = prompts.select_channel_name()
        self.chNames, _ = selector.get_available_channels(
            ls, self.images_path, useExt=useExt
        )
        self.basename = selector.basename

    def loadImgData(self, signals=None):
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
                self.criticalExtNotValid(signals=signals)

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

    def detectMultiSegmNpz(
            self, multiPos=False, signals=None,
            mutex=None, waitCond=None, askMultiSegmFunc=None,
            newEndFilenameSegm=''
        ):
        if newEndFilenameSegm:
            return '', newEndFilenameSegm, False

        segm_files = get_segm_files(self.images_path)

        if askMultiSegmFunc is None:
            return segm_files

        is_multi_npz = len(segm_files)>1
        if is_multi_npz and askMultiSegmFunc is not None:
            askMultiSegmFunc(segm_files, self, waitCond)
            endFilename = self.selectedItemText[len(self.basename):]
            return self.selectedItemText, endFilename, self.cancel
        elif len(segm_files)==1:
            segmFilename = segm_files[0]
            endFilename = segmFilename[len(self.basename):]
            return segm_files[0], endFilename, False
        else:
            return '', '', False

    def loadOtherFiles(
            self,
            load_segm_data=True,
            create_new_segm=False,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=False,
            load_delROIsInfo=False,
            loadBkgrData=False,
            loadBkgrROIs=False,
            load_last_tracked_i=False,
            load_metadata=False,
            load_dataPrep_ROIcoords=False,
            load_customAnnot=False,
            load_customCombineMetrics=False,
            getTifPath=False,
            end_filename_segm='',
            new_endname='',
            labelBoolSegm=None
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
        self.customAnnotFound = False if load_customAnnot else None
        self.combineMetricsFound = False if load_customCombineMetrics else None
        self.labelBoolSegm = labelBoolSegm
        self.bkgrDataExists = False
        ls = myutils.listdir(self.images_path)

        linked_acdc_filename = None
        if end_filename_segm and load_acdc_df:
            # Check if there is an acdc_output file linked to selected .npz
            _acdc_df_end_fn = end_filename_segm.replace('segm', 'acdc_output')
            _acdc_df_end_fn = f'{_acdc_df_end_fn}.csv'
            self._acdc_df_end_fn = _acdc_df_end_fn
            _linked_acdc_fn = f'{self.basename}{_acdc_df_end_fn}'
            acdc_df_path = os.path.join(self.images_path, _linked_acdc_fn)
            self.acdc_output_csv_path = acdc_df_path
            linked_acdc_filename = _linked_acdc_fn

        for file in ls:
            filePath = os.path.join(self.images_path, file)
            filename, segmExt = os.path.splitext(file)
            if not hasattr(self, 'basename'):
                self.getBasenameAndChNames()
            endName = filename[len(self.basename):]

            loadMetadata = (
                load_metadata and file.endswith('metadata.csv')
                and not file.endswith('segm_metadata.csv')
            )

            if new_endname:
                # Do not load any segmentation file since user asked for new one
                # This is redundant since we alse have create_new_segm=True
                # but we keep it for code readability
                is_segm_file = False
            elif end_filename_segm:
                # Load the segmentation file selected by the user
                self._segm_end_fn = end_filename_segm
                is_segm_file = endName == end_filename_segm and segmExt == '.npz'
            else:
                # Load default segmentation file
                is_segm_file = file.endswith('segm.npz')

            if linked_acdc_filename is not None:
                is_acdc_df_file = file == linked_acdc_filename
            elif end_filename_segm:
                # Requested a specific file but it is not present
                # do not load acdc_df file
                is_acdc_df_file = False
            else:
                is_acdc_df_file = file.endswith('acdc_output.csv')

            if load_segm_data and is_segm_file and not create_new_segm:
                self.segmFound = True
                self.segm_npz_path = filePath
                self.segm_data = np.load(filePath)['arr_0']
                if self.segm_data.dtype == bool:
                    if self.labelBoolSegm is None:
                        self.askBooleanSegm()
                squeezed_arr = np.squeeze(self.segm_data)
                if squeezed_arr.shape != self.segm_data.shape:
                    self.segm_data = squeezed_arr
                    np.savez_compressed(filePath, squeezed_arr)
            elif getTifPath and file.find(f'{self.user_ch_name}.tif')!=-1:
                self.tif_path = filePath
                self.TifPathFound = True
            elif load_acdc_df and is_acdc_df_file and not create_new_segm:
                self.acdc_df_found = True
                acdc_df = pd.read_csv(filePath)
                try:
                    acdc_df_drop_cca = acdc_df.drop(columns=cca_df_colnames).fillna(0)
                    acdc_df[acdc_df_drop_cca.columns] = acdc_df_drop_cca
                except KeyError:
                    pass
                acdc_df = acdc_df.set_index(['frame_i', 'Cell_ID'])
                acdc_df = pd_bool_to_int(acdc_df, acdc_df_bool_cols, inplace=True)
                acdc_df = pd_int_to_bool(acdc_df, acdc_df_bool_cols)
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
            elif file.endswith(f'{self.filename}_bkgrRoiData.npz'):
                self.bkgrDataExists = True
                if loadBkgrData:
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
                        maxBounds=QRectF(QRect(0,0,X,Y)),
                        scaleSnap=True,
                        translateSnap=True
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
            elif loadMetadata:
                self.metadataFound = True
                self.metadata_df = pd.read_csv(filePath).set_index('Description')
            elif load_customAnnot and file.endswith('custom_annot_params.json'):
                self.customAnnotFound = True
                self.customAnnot = read_json(filePath)
            elif load_customCombineMetrics and file.endswith('custom_combine_metrics.ini'):
                self.combineMetricsFound = True
                self.setCombineMetricsConfig(ini_path=filePath)

        if self.metadataFound is not None and self.metadataFound:
            self.extractMetadata()

        # Check if there is the old segm.npy
        if not self.segmFound and not create_new_segm:
            for file in ls:
                is_segm_npy = file.endswith('segm.npy')
                filePath = os.path.join(self.images_path, file)
                if load_segm_data and is_segm_npy and not self.segmFound:
                    self.segmFound = True
                    self.segm_data = np.load(filePath)

        if load_last_tracked_i:
            self.last_tracked_i_found = True
            try:
                self.last_tracked_i = max(self.acdc_df.index.get_level_values(0))
            except AttributeError as e:
                # traceback.print_exc()
                self.last_tracked_i = None

        if create_new_segm:
            self.setFilePaths(new_endname)

        self.getCustomAnnotatedIDs()
        self.setNotFoundData()

    def askBooleanSegm(self):
        segmFilename = os.path.basename(self.segm_npz_path)
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            f'The loaded segmentation file<br><br>'
            f'"{segmFilename}"<br><br> '
            'has <b>boolean data type</b>.<br><br>'
            'To correctly load it, Cell-ACDC needs to <b>convert</b> it '
            'to <b>integer data type</b>.<br><br>'
            'Do you want to <b>label the mask</b> to separate the objects '
            '(recommended) or do you want to keep one single object?<br>'
        )
        LabelButton, _  = msg.question(
            self.parent, 'Boolean segmentation mask?', txt,
            buttonsTexts=('Label (recommended)', 'Keep single object')
        )
        if msg.clickedButton == LabelButton:
            self.labelBoolSegm = True
        else:
            self.labelBoolSegm = False

    def labelSegmData(self):
        if self.labelBoolSegm is None:
            return

        if self.segm_data.dtype != bool:
            return

        if self.labelBoolSegm:
            if self.SizeT > 1:
                segm_data = np.zeros(self.segm_data.shape, dtype=np.uint16)
                for i, lab in enumerate(self.segm_data):
                    segm_data[i] = skimage.measure.label(lab)
                self.segm_data = segm_data
            else:
                self.segm_data = skimage.measure.label(self.segm_data)
        else:
            self.segm_data = self.segm_data.astype(np.uint16)

    def setFilePaths(self, new_endname):
        if self.basename.endswith('_'):
            basename = self.basename
        else:
            basename = f'{self.basename}_'

        segm_new_filename = f'{basename}segm_{new_endname}.npz'
        filePath = os.path.join(self.images_path, segm_new_filename)
        self.segm_npz_path = filePath

        acdc_output_filename = f'{basename}acdc_output_{new_endname}.csv'
        filePath = os.path.join(self.images_path, acdc_output_filename)
        self.acdc_output_csv_path = filePath

    def getCustomAnnotatedIDs(self):
        self.customAnnotIDs = {}

        if self.acdc_df_found is None:
            return

        if not self.acdc_df_found:
            return

        if self.customAnnotFound is None:
            return

        if not self.customAnnotFound:
            return

        for name in self.customAnnot.keys():
            self.customAnnotIDs[name] = {}
            if name not in self.acdc_df.columns:
                self.acdc_df[name] = 0
            for frame_i, df in self.acdc_df.groupby(level=0):
                series = df[name]
                series = series[series>0]
                annotatedIDs = list(series.index.get_level_values(1).unique())
                self.customAnnotIDs[name][frame_i] = annotatedIDs

    def getIsSegm3D(self):
        if self.SizeZ == 1:
            return False

        if self.segmFound is None:
            return False

        if not self.segmFound:
            return

        if self.SizeT > 1:
            return self.segm_data.ndim == 4
        else:
            return self.segm_data.ndim == 3

    def extractMetadata(self):
        self.metadata_df['values'] = self.metadata_df['values'].astype(str)
        if 'SizeT' in self.metadata_df.index:
            self.SizeT = float(self.metadata_df.at['SizeT', 'values'])
            self.SizeT = int(self.SizeT)
        elif self.last_md_df is not None and 'SizeT' in self.last_md_df.index:
            self.SizeT = float(self.last_md_df.at['SizeT', 'values'])
            self.SizeT = int(self.SizeT)
        else:
            self.SizeT = 1

        if 'SizeZ' in self.metadata_df.index:
            self.SizeZ = float(self.metadata_df.at['SizeZ', 'values'])
            self.SizeZ = int(self.SizeZ)
        elif self.last_md_df is not None and 'SizeZ' in self.last_md_df.index:
            self.SizeZ = float(self.last_md_df.at['SizeZ', 'values'])
            self.SizeZ = int(self.SizeZ)
        else:
            self.SizeZ = 1

        if 'SizeY' in self.metadata_df.index:
            self.SizeY = float(self.metadata_df.at['SizeY', 'values'])
            self.SizeY = int(self.SizeY)
            self.SizeX = float(self.metadata_df.at['SizeX', 'values'])
            self.SizeX = int(self.SizeX)
        else:
            if hasattr(self, 'img_data_shape'):
                self.SizeY, self.SizeX = self.img_data_shape[-2:]
            else:
                self.SizeY, self.SizeX = 1, 1

        self.isSegm3D = False
        if hasattr(self, 'segm_npz_path'):
            segmEndName = self.getSegmEndname()
            isSegm3Dkey = f'{segmEndName}_isSegm3D'        
            if 'isSegm3Dkey' in self.metadata_df.index:
                isSegm3D = str(self.metadata_df.at[isSegm3Dkey, 'values'])
                self.isSegm3D = isSegm3D.lower() == 'true'

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
             self.segmSizeT = float(
                 self.metadata_df.at['segmSizeT', 'values']
             )
             self.segmSizeT = int(self.segmSizeT)
        elif load_last_segmSizeT:
            self.segmSizeT = float(self.last_md_df.at['segmSizeT', 'values'])
            self.segmSizeT = int(self.segmSizeT)
        else:
            self.segmSizeT = self.SizeT

        self._additionalMetadataValues = {}
        for name in self.metadata_df.index:
            if name.startswith('__'):
                value = self.metadata_df.at[name, 'values']
                self._additionalMetadataValues[name] = value
        if not self._additionalMetadataValues:
            # Load metadata values saved in temp folder
            if os.path.exists(additional_metadata_path):
                self._additionalMetadataValues = read_json(
                    additional_metadata_path, desc='additional metadata'
                )

    def additionalMetadataValues(self):
        additionalMetadataValues = {}
        for name in self.metadata_df.index:
            if name.startswith('__'):
                value = self.metadata_df.at[name, 'values']
                key = name.replace('__', '', 1)
                additionalMetadataValues[key] = value
        return additionalMetadataValues

    def setNotFoundData(self):
        if self.segmFound is not None and not self.segmFound:
            self.segm_data = None
            # Segmentation file not found and a specifc one was requested
            # --> set the path
            if hasattr(self, '_segm_end_fn'):
                if self.basename.endswith('_'):
                    basename = self.basename
                else:
                    basename = f'{self.basename}_'
                base_path = os.path.join(self.images_path, basename)
                self.segm_npz_path = f'{base_path}{self._segm_end_fn}.npz'
        if self.acdc_df_found is not None and not self.acdc_df_found:
            self.acdc_df = None
            # Set the file path for selected acdc_output.csv file
            # since it was not found
            if hasattr(self, '_acdc_df_end_fn'):
                if self.basename.endswith('_'):
                    basename = self.basename
                else:
                    basename = f'{self.basename}_'
                base_path = os.path.join(self.images_path, basename)
                self.acdc_output_csv_path = f'{base_path}{self._acdc_df_end_fn}'
        if self.shiftsFound is not None and not self.shiftsFound:
            self.loaded_shifts = None
        if self.segmInfoFound is not None and not self.segmInfoFound:
            self.segmInfo_df = None
        if self.delROIsInfoFound is not None and not self.delROIsInfoFound:
            self.delROIsInfo_npz = None
        if self.bkgrDataFound is not None and not self.bkgrDataFound:
            self.bkgrData = None
        if self.bkgrROisFound is not None and not self.bkgrROisFound:
            # Do not load bkgrROIs if bkgrDataFound to avoid addMetrics to use it
            self.bkgrROIs = []
        if self.bkgrDataExists:
            # Do not load bkgrROIs if bkgrDataFound to avoid addMetrics to use it
            self.bkgrROIs = []
        if self.dataPrep_ROIcoordsFound is not None and not self.dataPrep_ROIcoordsFound:
            self.dataPrep_ROIcoords = None
        if self.last_tracked_i_found is not None and not self.last_tracked_i_found:
            self.last_tracked_i = None
        if self.TifPathFound is not None and not self.TifPathFound:
            self.tif_path = None
        if self.customAnnotFound is not None and not self.customAnnotFound:
            self.customAnnot = {}
        if self.combineMetricsFound is not None and not self.combineMetricsFound:
            self.setCombineMetricsConfig()

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

        self.SizeY, self.SizeX = self.img_data_shape[-2:]

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
            self.TimeIncrement = float(
                self.last_md_df.at['TimeIncrement', 'values']
            )
        if 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(
                self.last_md_df.at['PhysicalSizeX', 'values']
            )
        if 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(
                self.last_md_df.at['PhysicalSizeY', 'values']
            )
        if 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(
                self.last_md_df.at['PhysicalSizeZ', 'values']
            )
        if 'segmSizeT' in self.last_md_df.index:
            self.segmSizeT = int(self.last_md_df.at['segmSizeT', 'values'])

    def addEquationCombineMetrics(self, equation, colName, isMixedChannels):
        section = 'mixed_channels_equations' if isMixedChannels else 'equations'
        self.combineMetricsConfig[section][colName] = equation

    def setCombineMetricsConfig(self, ini_path=''):
        if ini_path:
            configPars = config.ConfigParser()
            configPars.read(ini_path)
        else:
            configPars = config.ConfigParser()

        if 'equations' not in configPars:
            configPars['equations'] = {}

        if 'mixed_channels_equations' not in configPars:
            configPars['mixed_channels_equations'] = {}

        if 'user_path_equations' not in configPars:
            configPars['user_path_equations'] = {}

        # Append channel specific equations from the user_path ini file
        userPathChEquations = configPars['user_path_equations']
        for chName in self.chNames:
            chName_equations = measurements.get_user_combine_metrics_equations(
                chName
            )
            chName_equations = {
                key:val for key, val in chName_equations.items()
                if key not in configPars['equations']
            }
            userPathChEquations = {**userPathChEquations, **chName_equations}
            configPars['user_path_equations'] = userPathChEquations

        # Append mixed channels equations from the user_path ini file
        configPars['mixed_channels_equations'] = {
            **configPars['mixed_channels_equations'],
            **measurements.get_user_combine_mixed_channels_equations()
        }

        self.combineMetricsConfig = configPars

    def saveCombineMetrics(self):
        with open(self.custom_combine_metrics_path, 'w') as configfile:
            self.combineMetricsConfig.write(configfile)

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

    def getSegmEndname(self):
        segmFilename = os.path.basename(self.segm_npz_path)
        segmFilename = os.path.splitext(segmFilename)[0]
        segmEndName = segmFilename[len(self.basename):]
        return segmEndName

    def saveSegmHyperparams(self, hyperparams, post_process_params):
        cp = config.ConfigParser()

        if os.path.exists(self.segm_hyperparams_ini_path):
            cp.read(self.segm_hyperparams_ini_path)

        segmEndName = self.getSegmEndname()
        now = datetime.now().strftime('%Y-%m-%d_%H:%M')
        section = f'{now}.{segmEndName}.segmentation'
        cp[section] = {'segmented_channel': self.user_ch_name}
        for key, value in hyperparams.items():
            cp[section][key] = str(value)

        section = f'{now}.{segmEndName}.post-processing'
        cp[section] = {}
        for key, value in post_process_params.items():
            cp[section][key] = str(value)

        with open(self.segm_hyperparams_ini_path, 'w') as configfile:
            cp.write(configfile)

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
        self.segm_hyperparams_ini_path = f'{base_path}segm_hyperparams.ini'
        self.btrack_tracks_h5_path = f'{base_path}btrack_tracks.h5'
        self.custom_annot_json_path = f'{base_path}custom_annot_params.json'
        self.custom_combine_metrics_path = f'{base_path}custom_combine_metrics.ini'

    def setBlankSegmData(self, SizeT, SizeZ, SizeY, SizeX):
        Y, X = self.img_data.shape[-2:]
        if self.segmFound is not None and not self.segmFound:
            if SizeT > 1 and self.isSegm3D:
                self.segm_data = np.zeros((SizeT, SizeZ, Y, X), int)
            elif self.isSegm3D:
                self.segm_data = np.zeros((SizeZ, Y, X), int)
            elif SizeT > 1:
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
            save=False,
            askSegm3D=True,
            forceEnableAskSegm3D=False,
        ):
        metadataWin = apps.QDialogMetadata(
            self.SizeT, self.SizeZ, self.TimeIncrement,
            self.PhysicalSizeZ, self.PhysicalSizeY, self.PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes,
            parent=self.parent, font=apps.font, imgDataShape=self.img_data_shape,
            posData=self, singlePos=singlePos, askSegm3D=askSegm3D,
            additionalValues=self._additionalMetadataValues,
            forceEnableAskSegm3D=forceEnableAskSegm3D
        )
        metadataWin.exec_()
        if metadataWin.cancel:
            return False

        self.SizeT = metadataWin.SizeT
        self.SizeZ = metadataWin.SizeZ
        self.SizeY, self.SizeX = self.img_data_shape[-2:]

        self.isSegm3D = metadataWin.isSegm3D

        self.loadSizeS = numPos
        self.loadSizeT = metadataWin.SizeT
        self.loadSizeZ = metadataWin.SizeZ

        source = metadataWin if ask_TimeIncrement else self
        self.TimeIncrement = source.TimeIncrement

        source = metadataWin if ask_PhysicalSizes else self
        self.PhysicalSizeZ = source.PhysicalSizeZ
        self.PhysicalSizeY = source.PhysicalSizeY
        self.PhysicalSizeX = source.PhysicalSizeX

        self._additionalMetadataValues = metadataWin._additionalValues
        if save:
            self.saveMetadata(additionalMetadata=metadataWin._additionalValues)
        return True

    def transferMetadata(self, from_posData):
        self.SizeT = from_posData.SizeT
        self.SizeZ = from_posData.SizeZ
        self.PhysicalSizeZ = from_posData.PhysicalSizeZ
        self.PhysicalSizeY = from_posData.PhysicalSizeY
        self.PhysicalSizeX = from_posData.PhysicalSizeX

    def updateSegmSizeT(self):
        segmSizeT = len(self.segm_data)
        if self.segmSizeT == segmSizeT:
            return
        self.segmSizeT = segmSizeT
        self.metadata_df.at['segmSizeT', 'values'] = segmSizeT
        self.metadataToCsv()

    def metadataToCsv(self, signals=None, mutex=None, waitCond=None):
        try:
            self.metadata_df.to_csv(self.metadata_csv_path)
        except PermissionError:
            print('='*20)
            traceback.print_exc()
            print('='*20)
            permissionErrorTxt = html_utils.paragraph(
                f'The below file is open in another app (Excel maybe?).<br><br>'
                f'{self.metadata_csv_path}<br><br>'
                'Close file and then press "Ok".'
            )
            if signals is None:
                msg = widgets.myMessageBox(self.parent)
                msg.setIcon(iconName='SP_MessageBoxCritical')
                msg.setWindowTitle('Permission denied')
                msg.addText(permissionErrorTxt)
                msg.addButton('  Ok  ')
                msg.exec_()
                self.metadata_df.to_csv(self.metadata_csv_path)
            else:
                mutex.lock()
                signals.sigPermissionError.emit(permissionErrorTxt, waitCond)
                waitCond.wait(mutex)
                mutex.unlock()
                self.metadata_df.to_csv(self.metadata_csv_path)

    def saveMetadata(
            self, signals=None, mutex=None, waitCond=None,
            additionalMetadata=None
        ):
        segmEndName = self.getSegmEndname()
        isSegm3Dkey = f'{segmEndName}_isSegm3D'
        if self.metadata_df is None:
            metadata_dict = {
                'SizeT': self.SizeT,
                'SizeZ': self.SizeZ,
                'SizeY': self.SizeY,
                'SizeX': self.SizeX,
                'TimeIncrement': self.TimeIncrement,
                'PhysicalSizeZ': self.PhysicalSizeZ,
                'PhysicalSizeY': self.PhysicalSizeY,
                'PhysicalSizeX': self.PhysicalSizeX,
                'segmSizeT': self.segmSizeT,
                isSegm3Dkey: self.isSegm3D
            }
            if additionalMetadata is not None:
                metadata_dict = {**metadata_dict, **additionalMetadata}
                for key in list(metadata_dict.keys()):
                    if key.startswith('__') and key not in additionalMetadata:
                        metadata_dict.pop(key)

            self.metadata_df = pd.DataFrame(metadata_dict, index=['values']).T
            self.metadata_df.index.name = 'Description'
        else:
            self.metadata_df.at['SizeT', 'values'] = self.SizeT
            self.metadata_df.at['SizeZ', 'values'] = self.SizeZ
            self.metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
            self.metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
            self.metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
            self.metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
            self.metadata_df.at['segmSizeT', 'values'] = self.segmSizeT
            self.metadata_df.at[isSegm3Dkey, 'values'] = self.isSegm3D
            if additionalMetadata is not None:
                for name, value in additionalMetadata.items():
                    self.metadata_df.at[name, 'values'] = value

                idx_to_drop = []
                for name in self.metadata_df.index:
                    if name.startswith('__') and name not in additionalMetadata:
                        idx_to_drop.append(name)

                self.metadata_df = self.metadata_df.drop(idx_to_drop)
        self.metadataToCsv(signals=signals, mutex=signals, waitCond=waitCond)
        self.metadata_df.to_csv(last_entries_metadata_path)
        if additionalMetadata is not None:
            with open(additional_metadata_path, mode='w') as file:
                json.dump(additionalMetadata, file, indent=2)

    def criticalExtNotValid(self, signals=None):
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
        elif signals is None:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            msg = QMessageBox()
            msg.critical(self.parent, err_title, err_msg, msg.Ok)
            return None
        elif signals is not None:
            raise FileNotFoundError(err_title)

class select_exp_folder:
    def __init__(self):
        self.exp_path = None

    def QtPrompt(
            self, parentQWidget, values,
            current=0, title='Select Position folder',
            CbLabel="Select folder to load:",
            showinexplorer_button=False, full_paths=None,
            allow_abort=True, show=False, toggleMulti=False
        ):
        font = QtGui.QFont()
        font.setPixelSize(13)
        win = apps.QtSelectItems(
            title, values, '', CbLabel=CbLabel, parent=parentQWidget,
            showInFileManagerPath=self.exp_path
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
        self.exp_path = exp_path
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            last_tracked_i_found = False
            pos_path = f'{exp_path}/{pos}'
            images_path = f'{exp_path}/{pos}/Images'
            filenames = myutils.listdir(images_path)
            for filename in filenames:
                if filename.find('acdc_output.csv') != -1:
                    last_tracked_i_found = True
                    acdc_df_path = f'{images_path}/{filename}'
                    acd_df = pd.read_csv(acdc_df_path)
                    last_tracked_i = max(acd_df['frame_i'])
                    break
            if last_tracked_i_found:
                values.append(f'{pos} (Last tracked frame: {last_tracked_i+1})')
            else:
                values.append(pos)
        self.values = values
        return values

    def get_values_dataprep(self, exp_path):
        self.exp_path = exp_path
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            is_prepped = False
            pos_path = f'{exp_path}/{pos}'
            images_path = f'{exp_path}/{pos}/Images'
            filenames = myutils.listdir(images_path)
            for filename in filenames:
                if filename.endswith('dataPrepROIs_coords.csv'):
                    is_prepped = True
                    break
                elif filename.endswith('dataPrep_bkgrROIs.json'):
                    is_prepped = True
                    break
                elif filename.endswith('aligned.npz'):
                    is_prepped = True
                    break
                elif filename.endswith('align_shift.npy'):
                    is_prepped = True
                    break
                elif filename.endswith('bkgrRoiData.npz'):
                    is_prepped = True
                    break
            if is_prepped:
                values.append(f'{pos} (already prepped)')
            else:
                values.append(pos)
        self.values = values
        return values

    def get_values_cca(self, exp_path):
        self.exp_path = exp_path
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
