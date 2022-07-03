# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPyTop HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TODO:
print('Importing GUI modules...')

import sys
import os
import shutil
import pathlib
import re
import traceback
import time
import datetime
import logging
import uuid
import json
import psutil
from importlib import import_module
from functools import partial
from tqdm import tqdm
from pprint import pprint
import time

import cv2
import math
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from functools import wraps
from skimage.color import gray2rgb, gray2rgba, label2rgb

from PyQt5.QtCore import (
    Qt, QFile, QTextStream, QSize, QRect, QRectF,
    QEventLoop, QTimer, QEvent, QObject, pyqtSignal,
    QThread, QMutex, QWaitCondition, QSettings
)
from PyQt5.QtGui import (
    QIcon, QKeySequence, QCursor, QKeyEvent, QGuiApplication,
    QPixmap, QColor
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton, QHBoxLayout,
    QMainWindow, QMenu, QToolBar, QGroupBox, QGridLayout,
    QScrollBar, QCheckBox, QToolButton, QSpinBox, QGroupBox,
    QComboBox, QDial, QButtonGroup, QActionGroup,
    QShortcut, QFileDialog, QDoubleSpinBox,
    QAbstractSlider, QMessageBox, QWidget, QDockWidget,
    QDockWidget, QGridLayout, QSizePolicy, QVBoxLayout
)

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# NOTE: Enable icons
from . import qrc_resources

# Custom modules
from . import base_cca_df, graphLayoutBkgrColor
from . import load, prompts, apps, workers, html_utils
from . import core, myutils, dataPrep, widgets
from . import measurements, printl
from .trackers.CellACDC import CellACDC_tracker
from .cca_functions import _calc_rot_vol
from .myutils import exec_time, setupLogger
from .help import welcome

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(cellacdc_path, 'temp')
settings_csv_path = os.path.join(temp_path, 'settings.csv')
favourite_func_metrics_csv_path = os.path.join(
    temp_path, 'favourite_func_metrics.csv'
)
custom_annot_path = os.path.join(temp_path, 'custom_annotations.json')

# Interpret image data as row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
np.random.seed(1568)

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)
pd.set_option('display.expand_frame_repr', False)

def qt_debug_trace():
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    import pdb; pdb.set_trace()

def get_data_exception_handler(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as e:
            try:
                if self.progressWin is not None:
                    self.progressWin.workerFinished = True
                    self.progressWin.close()
            except AttributeError:
                pass
            result = None
            posData = self.data[self.pos_i]
            acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
            segm_filename = os.path.basename(posData.segm_npz_path)
            traceback_str = traceback.format_exc()
            self.logger.exception(traceback_str)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.addShowInFileManagerButton(self.logs_path, txt='Show log file...')
            msg.setDetailedText(traceback_str)
            err_msg = html_utils.paragraph(f"""
                Error in function <code>{func.__name__}</code>.<br><br>
                One possbile explanation is that either the
                <code>{acdc_df_filename}</code> file<br>
                or the segmentation file <code>{segm_filename}</code><br>
                <b>are corrupted/damaged</b>.<br><br>
                <b>Try moving these files</b> (one by one) outside of the
                <code>{os.path.dirname(posData.relPath)}</code> folder
                <br>and reloading the data.<br><br>
                More details below or in the terminal/console.<br><br>
                Note that the <b>error details</b> from this session are
                also <b>saved in the following file</b>:<br><br>
                {self.log_path}<br><br>
                Please <b>send the log file</b> when reporting a bug, thanks!
            """)

            msg.critical(self, 'Critical error', err_msg)
            self.is_error_state = True
            raise e
        return result
    return inner_function

class trackingWorker(QObject):
    finished = pyqtSignal()
    critical = pyqtSignal(object)
    progress = pyqtSignal(str)
    debug = pyqtSignal(object)

    def __init__(self, posData, mainWin, video_to_track):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.posData = posData
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.tracker = self.mainWin.tracker
        self.video_to_track = video_to_track

    @workers.worker_exception_handler
    def run(self):
        self.mutex.lock()

        self.progress.emit('Tracking process started...')

        tracked_video = self.tracker.track(
            self.video_to_track, signals=self.signals,
            export_to=self.posData.btrack_tracks_h5_path
        )

        # Store new tracked video
        current_frame_i = self.posData.frame_i
        self.trackingOnNeverVisitedFrames = False
        for rel_frame_i, lab in enumerate(tracked_video):
            frame_i = rel_frame_i + self.mainWin.start_n - 1

            if self.posData.allData_li[frame_i]['labels'] is None:
                # repeating tracking on a never visited frame
                # --> modify only raw data and ask later what to do
                self.posData.segm_data[frame_i] = lab.copy()
                self.trackingOnNeverVisitedFrames = True
            else:
                # Get the rest of the stored metadata based on the new lab
                self.posData.allData_li[frame_i]['labels'] = lab.copy()
                self.posData.frame_i = frame_i
                self.mainWin.get_data()
                self.mainWin.store_data()

        # Back to current frame
        self.posData.frame_i = current_frame_i
        self.mainWin.get_data()

        self.mutex.unlock()
        self.finished.emit()

class relabelSequentialWorker(QObject):
    finished = pyqtSignal()
    critical = pyqtSignal(object)
    progress = pyqtSignal(str)
    sigRemoveItemsGUI = pyqtSignal(int)
    debug = pyqtSignal(object)

    def __init__(self, posData, mainWin):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.posData = posData
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()

    def progressNewIDs(self, inv):
        newIDs = inv.in_values
        oldIDs = inv.out_values
        li = list(zip(oldIDs, newIDs))
        s = '\n'.join([str(pair).replace(',', ' -->') for pair in li])
        s = f'IDs relabelled as follows:\n{s}'
        self.progress.emit(s)

    @workers.worker_exception_handler
    def run(self):
        self.mutex.lock()

        self.progress.emit('Relabelling process started...')

        posData = self.posData
        progressWin = self.mainWin.progressWin
        mainWin = self.mainWin

        current_lab = self.mainWin.get_2Dlab(posData.lab).copy()
        current_frame_i = posData.frame_i
        segm_data = []
        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict['labels']
            if lab is None:
                break
            segm_data.append(lab)
            if frame_i == current_frame_i:
                break

        if not segm_data:
            segm_data = np.array(current_lab)

        segm_data = np.array(segm_data)
        segm_data, fw, inv = skimage.segmentation.relabel_sequential(
            segm_data
        )
        self.progressNewIDs(inv)
        self.sigRemoveItemsGUI.emit(np.max(segm_data))

        self.progress.emit(
            'Updating stored data and cell cycle annotations '
            '(if present)...'
        )
        newIDs = list(inv.in_values)
        oldIDs = list(inv.out_values)
        newIDs.append(-1)
        oldIDs.append(-1)
        for frame_i, lab in enumerate(segm_data):
            posData.frame_i = frame_i
            posData.lab = lab
            mainWin.get_cca_df()
            if posData.cca_df is not None:
                mainWin.update_cca_df_relabelling(
                    posData, oldIDs, newIDs
                )
            mainWin.update_rp(draw=False)
            mainWin.store_data(mainThread=False)

        # Go back to current frame
        posData.frame_i = current_frame_i
        mainWin.get_data()

        self.mutex.unlock()
        self.finished.emit()

class saveDataWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    progressBar = pyqtSignal(int, int, float)
    critical = pyqtSignal(str)
    criticalMetrics = pyqtSignal(object)
    criticalPermissionError = pyqtSignal(str)
    metricsPbarProgress = pyqtSignal(int, int)
    askZsliceAbsent = pyqtSignal(str, object)
    customMetricsCritical = pyqtSignal(str)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.saveWin = mainWin.saveWin
        self.mutex = mainWin.mutex
        self.waitCond = mainWin.waitCond

    def addMetrics_acdc_df(self, df, rp, frame_i, lab, posData):
        """
        Function used to add metrics to the acdc_df.

        NOTE for 3D data: add the same metrics calculated from 2D data obtained
        with three different methods:
            - sum projection
            - mean projection
            - z-slice used for segmentation

        For background data there are three options:
            1. The user did not select any background ROI in data Prep
               --> save only autoBkgr which are all the pixels outside cells
            2. The user selected background ROI but did not crop
               --> get values from the ROI background in this function
            3. The user selected background ROI AND cropped
               --> background values are saved in posData.fluo_bkgrData_dict
                   and we calculate metrics from there
        """

        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeX = posData.PhysicalSizeX

        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX
        if self.mainWin.isSegm3D:
            vox_to_fl_3D = PhysicalSizeY*PhysicalSizeX*posData.PhysicalSizeZ
        numCells = len(rp)

        list_0s = [-2]*numCells
        IDs = list_0s.copy()
        IDs_vol_vox = list_0s.copy()
        IDs_area_pxl = list_0s.copy()
        IDs_vol_fl = list_0s.copy()
        IDs_area_um2 = list_0s.copy()
        if self.mainWin.isSegm3D:
            IDs_vol_vox_3D = list_0s.copy()
            IDs_vol_fl_3D = list_0s.copy()

        # Initialize fluo metrics arrays
        fluo_keys = list(posData.fluo_data_dict.keys())
        fluo_data = posData.fluo_data_dict[fluo_keys[0]][frame_i]
        is_3D = fluo_data.ndim == 3
        how_3Dto2D, _ = measurements.get_how_3Dto2D(is_3D, self.mainWin.isSegm3D)
        n = len(how_3Dto2D)
        numFluoChannels = len(fluo_keys)

        # Defined in function setMetricsFunc
        metrics_func = self.mainWin.metrics_func
        custom_func_dict = self.mainWin.custom_func_dict

        # Dictionary where values is a list of 0s with len=numCells
        # and key is 'channelName_metrics_how' (e.g. 'GFP_mean_zSlice')
        metrics_values = {
            f'{chName}_{metric}{how}':list_0s.copy()
            for metric in self.mainWin.all_metrics_names
            for chName in posData.loadedChNames
            for how in how_3Dto2D
        }
        custom_metrics_values = {
            f'{chName}_{metric}{how}':list_0s.copy()
            for metric in self.mainWin.custom_func_dict
            for chName in posData.loadedChNames
            for how in how_3Dto2D
        }

        tot_iter = (
            self.mainWin.total_metrics
            *len(posData.loadedChNames)
            *len(how_3Dto2D)
            *numCells
        )

        # self.metricsPbarProgress.emit(tot_iter, 0)

        # pbar = tqdm(total=tot_iter, ncols=100, unit='metric', leave=False)

        if self.mainWin.isSegm3D:
            outCellsMask3D = lab==0
            outCellsMaskProj = lab.max(axis=0) == 0
        else:
            outCellsMask2D = lab==0
            outCellsMaskZslice = outCellsMask2D
            outCellsMaskProj = outCellsMask2D

        # Compute ROI bkgrMask
        if posData.bkgrROIs:
            ROI_bkgrMask = np.zeros(posData.lab.shape, bool)
            if posData.bkgrROIs:
                for roi in posData.bkgrROIs:
                    xl, yl = [int(round(c)) for c in roi.pos()]
                    w, h = [int(round(c)) for c in roi.size()]
                    ROI_bkgrMask[yl:yl+h, xl:xl+w] = True
        else:
            ROI_bkgrMask = None

        # Iteare fluo channels and get 2D data from 3D if needed
        for chName, filename in zip(posData.loadedChNames, fluo_keys):
            fluo_data = posData.fluo_data_dict[filename][frame_i]
            bkgrArchive = posData.fluo_bkgrData_dict[filename]
            metricsToSkipChannel = self.mainWin.metricsToSkip.get(chName, [])
            fluo_data_projs = []
            bkgrData_medians = []
            bkgrData_means = []
            bkgrData_q75s = []
            bkgrData_q25s = []
            bkgrData_q95s = []
            bkgrData_q05s = []
            if posData.SizeZ > 1:
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
                        self.progress.emit(
                            f'z-slice for channel "{chName}" absent. '
                            'Follow instructions on pop-up dialogs.'
                        )
                        self.mutex.lock()
                        self.askZsliceAbsent.emit(filename, posData)
                        self.waitCond.wait(self.mutex)
                        self.mutex.unlock()
                        segmInfo_df = pd.read_csv(posData.segmInfo_df_csv_path)
                        index_col = ['filename', 'frame_i']
                        posData.segmInfo_df = segmInfo_df.set_index(index_col)
                        col = 'z_slice_used_dataPrep'
                        z_slice = posData.segmInfo_df.at[idx, col]

                fluo_data_z_maxP = fluo_data.max(axis=0)
                fluo_data_z_sumP = fluo_data.mean(axis=0)
                fluo_data_zSlice = fluo_data[z_slice]
                if self.mainWin.isSegm3D:
                    fluo_data_3D = fluo_data
                    outCellsMaskZslice = lab[z_slice]

                # how_3Dto2D = ['_maxProj', '_sumProj', '_zSlice']
                fluo_data_projs.append(fluo_data_z_maxP)
                fluo_data_projs.append(fluo_data_z_sumP)
                fluo_data_projs.append(fluo_data_zSlice)
                if self.mainWin.isSegm3D:
                    fluo_data_projs.append(fluo_data)

                if bkgrArchive is not None:
                    bkgrVals_z_maxP = []
                    bkgrVals_z_sumP = []
                    bkgrVals_zSlice = []
                    bkgrVals_3D = []
                    for roi_key in bkgrArchive.files:
                        roiData = bkgrArchive[roi_key]
                        if posData.SizeT > 1:
                            roiData = bkgrArchive[roi_key][frame_i]
                        roi_z_maxP = roiData.max(axis=0)
                        roi_z_sumP = roiData.mean(axis=0)
                        roi_zSlice = roiData[z_slice]
                        bkgrVals_z_maxP.extend(roi_z_maxP[roi_z_maxP!=0])
                        bkgrVals_z_sumP.extend(roi_z_sumP[roi_z_sumP!=0])
                        bkgrVals_zSlice.extend(roi_zSlice[roi_zSlice!=0])
                        if self.mainWin.isSegm3D:
                            bkgrVals_3D.extend(roiData[roiData!=0])
                    if not bkgrVals_z_maxP:
                        # issue 51: a user had an empty bkgr data
                        bkgrVals_z_maxP = [0]
                    if not bkgrVals_z_sumP:
                        # issue 51: a user had an empty bkgr data
                        bkgrVals_z_sumP = [0]
                    if not bkgrVals_zSlice:
                        # issue 51: a user had an empty bkgr data
                        bkgrVals_zSlice = [0]
                    bkgrData_medians.append(np.median(bkgrVals_z_maxP))
                    bkgrData_medians.append(np.median(bkgrVals_z_sumP))
                    bkgrData_medians.append(np.median(bkgrVals_zSlice))
                    if self.mainWin.isSegm3D:
                        bkgrData_medians.extend(np.median(bkgrVals_3D))

                    bkgrData_means.append(np.mean(bkgrVals_z_maxP))
                    bkgrData_means.append(np.mean(bkgrVals_z_sumP))
                    bkgrData_means.append(np.mean(bkgrVals_zSlice))
                    if self.mainWin.isSegm3D:
                        bkgrData_means.extend(np.mean(bkgrVals_3D))

                    bkgrData_q75s.append(np.quantile(bkgrVals_z_maxP, q=0.75))
                    bkgrData_q75s.append(np.quantile(bkgrVals_z_sumP, q=0.75))
                    bkgrData_q75s.append(np.quantile(bkgrVals_zSlice, q=0.75))
                    if self.mainWin.isSegm3D:
                        bkgrData_q75s.extend(np.quantile(bkgrVals_3D, q=0.75))

                    bkgrData_q25s.append(np.quantile(bkgrVals_z_maxP, q=0.25))
                    bkgrData_q25s.append(np.quantile(bkgrVals_z_sumP, q=0.25))
                    bkgrData_q25s.append(np.quantile(bkgrVals_zSlice, q=0.25))
                    if self.mainWin.isSegm3D:
                        bkgrData_q25s.extend(np.quantile(bkgrVals_3D, q=0.25))

                    bkgrData_q95s.append(np.quantile(bkgrVals_z_maxP, q=0.95))
                    bkgrData_q95s.append(np.quantile(bkgrVals_z_sumP, q=0.95))
                    bkgrData_q95s.append(np.quantile(bkgrVals_zSlice, q=0.95))
                    if self.mainWin.isSegm3D:
                        bkgrData_q95s.extend(np.quantile(bkgrVals_3D, q=0.95))

                    bkgrData_q05s.append(np.quantile(bkgrVals_z_maxP, q=0.05))
                    bkgrData_q05s.append(np.quantile(bkgrVals_z_sumP, q=0.05))
                    bkgrData_q05s.append(np.quantile(bkgrVals_zSlice, q=0.05))
                    if self.mainWin.isSegm3D:
                        bkgrData_q05s.extend(np.quantile(bkgrVals_3D, q=0.05))
            else:
                fluo_data_2D = fluo_data
                fluo_data_projs.append(fluo_data_2D)
                if bkgrArchive is not None:
                    # Note that if bkgrArchive is not None then
                    # ROI_bkgrMask is None (no .json file available)
                    bkgrVals_2D = []
                    for roi_key in bkgrArchive.files:
                        roiData = bkgrArchive[roi_key]
                        if posData.SizeT > 1:
                            roiData = bkgrArchive[roi_key][frame_i]
                        bkgrVals_2D.extend(roiData[roiData!=0])
                    if not bkgrVals_2D:
                        # issue 51: a user had an empty bkgr data
                        bkgrVals_2D = [0]
                    bkgrData_medians.append(np.median(bkgrVals_2D))
                    bkgrData_means.append(np.mean(bkgrVals_2D))
                    bkgrData_q75s.append(np.quantile(bkgrVals_2D, q=0.75))
                    bkgrData_q25s.append(np.quantile(bkgrVals_2D, q=0.25))
                    bkgrData_q95s.append(np.quantile(bkgrVals_2D, q=0.95))
                    bkgrData_q05s.append(np.quantile(bkgrVals_2D, q=0.05))

            # Iterate cells
            for i, obj in enumerate(rp):
                if self.mainWin.isSegm3D:
                    obj3Dslice = obj.slice
                    obj3Dimage = obj.image
                    obj2Dproj = obj.image.max(axis=0)
                    obj2Dslice = obj3Dslice[1:3]
                    min_z = obj.bbox[0]
                    z_slice = int(math.floor(obj.centroid[0]))
                    local_z = z_slice - min_z
                    obj2DzImage = obj.image[local_z]
                else:
                    obj2Dslice = obj.slice
                    obj2Dproj = obj.image
                    obj2DzImage = obj.image # self.mainWin.getObjImage(obj.image, obj.bbox)
                IDs[i] = obj.label
                # Calc volume
                vol_vox = None
                vol_fl = None
                if 'cell_vol_vox' in self.mainWin.sizeMetricsToSave:
                    IDs_vol_vox[i] = obj.vol_vox
                    IDs_vol_fl[i] = obj.vol_fl
                    vol_vox = obj.vol_vox
                    vol_fl = obj.vol_fl
                    if self.mainWin.isSegm3D:
                        IDs_vol_vox_3D[i] = obj.area
                        IDs_vol_fl_3D[i] = obj.area*vox_to_fl_3D

                if 'cell_area_pxl' in self.mainWin.sizeMetricsToSave:
                    IDs_area_pxl[i] = obj.area
                    IDs_area_um2[i] = obj.area*yx_pxl_to_um2

                # Iterate method of 3D to 2D
                # '_maxProj', '_meanProj', '_zSlice', '_3D'
                how_iterable = enumerate(zip(how_3Dto2D, fluo_data_projs))
                for k, (how, fluo_img) in how_iterable:

                    # fluo_img!=0 is required because when we align we pad with 0s
                    # instead of np.roll and we don't want to include those
                    # exact 0s in the backgrMask
                    if how == '_maxProj':
                        fluo_data_ID = fluo_img[obj2Dslice][obj2Dproj]
                        backgrMask = np.logical_and(outCellsMaskProj, fluo_img!=0)
                    elif how == '_meanProj':
                        fluo_data_ID = fluo_img[obj2Dslice][obj2Dproj]
                        backgrMask = np.logical_and(outCellsMaskProj, fluo_img!=0)
                    elif how == '_zSlice':
                        fluo_data_ID = fluo_img[obj2Dslice][obj2DzImage]
                        backgrMask = np.logical_and(outCellsMaskZslice, fluo_img!=0)
                    elif how == '_3D':
                        fluo_data_ID = fluo_img[obj3Dslice][obj3Dimage]
                        backgrMask = np.logical_and(outCellsMask3D, fluo_img!=0)
                    else:
                        # 2D data
                        fluo_data_ID = fluo_img[obj2Dslice][obj2DzImage]
                        backgrMask = np.logical_and(outCellsMask2D, fluo_img!=0)

                    bkgr_arr = fluo_img[backgrMask]
                    fluo_backgr = np.median(bkgr_arr)

                    bkgr_key = f'{chName}_autoBkgr_bkgrVal_median{how}'
                    if not bkgr_key in metricsToSkipChannel:
                        metrics_values[bkgr_key][i] = fluo_backgr

                    bkgr_key = f'{chName}_autoBkgr_bkgrVal_mean{how}'
                    if not bkgr_key in metricsToSkipChannel:
                        metrics_values[bkgr_key][i] = bkgr_arr.mean()

                    bkgr_key = f'{chName}_autoBkgr_bkgrVal_q75{how}'
                    if not bkgr_key in metricsToSkipChannel:
                        metrics_values[bkgr_key][i] = np.quantile(
                            bkgr_arr, q=0.75
                        )

                    bkgr_key = f'{chName}_autoBkgr_bkgrVal_q25{how}'
                    if not bkgr_key in metricsToSkipChannel:
                        metrics_values[bkgr_key][i] = np.quantile(
                            bkgr_arr, q=0.25
                        )

                    bkgr_key = f'{chName}_autoBkgr_bkgrVal_q95{how}'
                    if not bkgr_key in metricsToSkipChannel:
                        metrics_values[bkgr_key][i] = np.quantile(
                            bkgr_arr, q=0.95
                        )

                    bkgr_key = f'{chName}_autoBkgr_bkgrVal_q05{how}'
                    if not bkgr_key in metricsToSkipChannel:
                        metrics_values[bkgr_key][i] = np.quantile(
                            bkgr_arr, q=0.05
                        )

                    # Calculate metrics for each cell
                    for func_name, func in metrics_func.items():
                        key = f'{chName}_{func_name}{how}'
                        conc_keys = measurements.get_conc_keys(key)
                        is_ROIbkgr_func = (
                            func_name == 'amount_dataPrepBkgr' and
                                (ROI_bkgrMask is not None
                                or bkgrArchive is not None)
                        )
                        if func_name == 'amount_autoBkgr':
                            if not key in metricsToSkipChannel:
                                val = func(fluo_data_ID, fluo_backgr, obj.area)
                                metrics_values[key][i] = val
                                conc_key_vox, conc_key_fl = conc_keys
                                calc_conc_vox = (
                                    vol_vox is not None
                                    and conc_key_vox not in metricsToSkipChannel
                                )
                                if calc_conc_vox:
                                    # Compute concentration
                                    conc_vox = val/vol_vox
                                    metrics_values[conc_key_vox][i] = conc_vox

                                calc_conc_fl = (
                                    vol_fl is not None
                                    and conc_key_fl not in metricsToSkipChannel
                                )
                                if calc_conc_fl:
                                    conc_fl = val/vol_fl
                                    metrics_values[conc_key_fl][i] = conc_fl
                        elif is_ROIbkgr_func:
                            if ROI_bkgrMask is not None:
                                ROI_bkgrData = fluo_img[ROI_bkgrMask]
                                ROI_bkgrVal = np.median(ROI_bkgrData)
                            else:
                                ROI_bkgrVal = bkgrData_medians[k]

                            if not key in metricsToSkipChannel:
                                val = func(fluo_data_ID, ROI_bkgrVal, obj.area)
                                metrics_values[key][i] = val
                                conc_key_vox, conc_key_fl = conc_keys
                                calc_conc = (
                                    vol_vox is not None
                                    and conc_key_vox not in metricsToSkipChannel
                                )
                                if calc_conc:
                                    # Compute concentration
                                    if how == '_3D':
                                        conc_vox = val/IDs_vol_vox_3D[i]
                                        conc_fl = val/IDs_vol_fl_3D[i]
                                    else:
                                        conc_vox = val/vol_vox
                                        conc_fl = val/vol_fl
                                    metrics_values[conc_key_vox][i] = conc_vox
                                    metrics_values[conc_key_fl][i] = conc_fl

                            bkgr_key = f'{chName}_dataPrepBkgr_bkgrVal_median{how}'
                            if not bkgr_key in metricsToSkipChannel:
                                metrics_values[bkgr_key][i] = ROI_bkgrVal

                            bkgr_key = f'{chName}_dataPrepBkgr_bkgrVal_mean{how}'
                            if not bkgr_key in metricsToSkipChannel:
                                if ROI_bkgrMask is None:
                                    bkgr_val = bkgrData_means[k]
                                else:
                                    bkgr_val = ROI_bkgrData.mean()
                                metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_bkgrVal_q75{how}'
                            if not bkgr_key in metricsToSkipChannel:
                                if ROI_bkgrMask is None:
                                    bkgr_val = bkgrData_q75s[k]
                                else:
                                    bkgr_val = np.quantile(ROI_bkgrData, q=0.75)
                                metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_bkgrVal_q25{how}'
                            if not bkgr_key in metricsToSkipChannel:
                                if ROI_bkgrMask is None:
                                    bkgr_val = bkgrData_q25s[k]
                                else:
                                    bkgr_val = np.quantile(ROI_bkgrData, q=0.25)
                                metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_bkgrVal_q95{how}'
                            if not bkgr_key in metricsToSkipChannel:
                                if ROI_bkgrMask is None:
                                    bkgr_val = bkgrData_q95s[k]
                                else:
                                    bkgr_val = np.quantile(ROI_bkgrData, q=0.95)
                                metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_bkgrVal_q05{how}'
                            if not bkgr_key in metricsToSkipChannel:
                                if ROI_bkgrMask is None:
                                    bkgr_val = bkgrData_q05s[k]
                                else:
                                    bkgr_val = np.quantile(ROI_bkgrData, q=0.05)
                                metrics_values[bkgr_key][i] = bkgr_val

                        elif func_name.find('amount') == -1:
                            if not key in metricsToSkipChannel:
                                val = func(fluo_data_ID)
                                metrics_values[key][i] = val

                        # pbar.update()
                        # self.metricsPbarProgress.emit(-1, 1)
                    for custom_func_name, custom_func in custom_func_dict.items():
                        key = f'{chName}_{custom_func_name}{how}'
                        if key in metricsToSkipChannel:
                            # Skip metric because unchecked in set measurements
                            continue

                        if ROI_bkgrMask is not None:
                            ROI_bkgrData = fluo_img[ROI_bkgrMask]
                            ROI_bkgrVal = np.median(ROI_bkgrData)
                        elif bkgrArchive is not None:
                            ROI_bkgrVal = bkgrData_medians[k]
                        else:
                            ROI_bkgrVal = None
                        try:
                            custom_val = custom_func(
                                fluo_data_ID, fluo_backgr, ROI_bkgrVal
                            )
                            custom_metrics_values[key][i] = custom_val
                        except Exception as e:
                            self.customMetricsCritical.emit(
                                traceback.format_exc()
                            )
                            # self.mainWin.logger.info(traceback.format_exc())
                        # self.metricsPbarProgress.emit(-1, 1)
                        # pbar.update()

        if 'cell_area_pxl' in self.mainWin.sizeMetricsToSave:
            df['cell_area_pxl'] = pd.Series(
                data=IDs_area_pxl, index=IDs, dtype=float
            )
            df['cell_area_um2'] = pd.Series(
                data=IDs_area_um2, index=IDs, dtype=float
            )

        if 'cell_vol_vox' in self.mainWin.sizeMetricsToSave:
            df['cell_vol_vox'] = pd.Series(
                data=IDs_vol_vox, index=IDs, dtype=float
            )
            df['cell_vol_fl'] = pd.Series(
                data=IDs_vol_fl, index=IDs, dtype=float
            )
            if self.mainWin.isSegm3D:
                df['cell_vol_vox_3D'] = pd.Series(
                    data=IDs_vol_vox_3D, index=IDs, dtype=float
                )
                df['cell_vol_fl_3D'] = pd.Series(
                    data=IDs_vol_fl_3D, index=IDs, dtype=float
                )

        df_metrics = pd.DataFrame(metrics_values, index=IDs)

        # Drop metrics that were already calculated in a prev session
        df = df.drop(columns=df_metrics.columns, errors='ignore')
        df = df.join(df_metrics)

        if custom_metrics_values:
            # Drop custom metrics that were already calculated in a prev session
            df_custom_metrics = pd.DataFrame(custom_metrics_values, index=IDs)
            df = df.drop(columns=df_custom_metrics.columns, errors='ignore')
            df = df.join(df_custom_metrics)

        # pbar.close()

        # Join with regionprops_table
        if self.mainWin.regionPropsToSave:
            if 'label' not in self.mainWin.regionPropsToSave:
                self.mainWin.regionPropsToSave = (
                    'label', *self.mainWin.regionPropsToSave
                )
            rp_table = skimage.measure.regionprops_table(
                posData.lab, properties=self.mainWin.regionPropsToSave
            )
            df_rp = pd.DataFrame(rp_table).set_index('label')
            # Drop regionprops that were already calculated in a prev session
            df = df.drop(columns=df_rp.columns, errors='ignore')
            df = df.join(df_rp)

        # Remove 0s columns
        df = df.loc[:, (df != -2).any(axis=0)]

        return df

    def _dfEvalEquation(self, df, newColName, expr):
        try:
            df[newColName] = df.eval(expr)
        except:
            print('-'*20)
            self.mainWin.logger.info(traceback.format_exc())
            print('='*20)

    def _removeDeprecatedRows(self, df):
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

        return df

    def addCombineMetrics_acdc_df(self, posData, df):
        # Add channel specifc combined metrics
        config = posData.combineMetricsConfig
        for chName in posData.loadedChNames:
            metricsToSkipChannel = self.mainWin.metricsToSkip.get(chName, [])
            posDataEquations = config['equations']
            userPathChEquations = config['user_path_equations']
            for newColName, equation in posDataEquations.items():
                if newColName in metricsToSkipChannel:
                    continue
                self._dfEvalEquation(df, newColName, equation)
            for newColName, equation in userPathChEquations.items():
                if newColName in metricsToSkipChannel:
                    continue
                self._dfEvalEquation(df, newColName, equation)

        # Add mixed channels combined metrics
        mixedChannelsEquations = config['mixed_channels_equations']
        for newColName, equation in mixedChannelsEquations.items():
            if newColName not in self.mainWin.mixedChCombineMetricsToSave:
                continue
            self._dfEvalEquation(df, newColName, equation)

    def addVolumeMetrics(self, df, rp, posData):
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
        if self.mainWin.isSegm3D:
            IDs_vol_vox_3D = init_list.copy()
            IDs_vol_fl_3D = init_list.copy()

        for i, obj in enumerate(rp):
            IDs[i] = obj.label
            IDs_vol_vox[i] = obj.vol_vox
            IDs_vol_fl[i] = obj.vol_fl
            IDs_area_pxl[i] = obj.area
            IDs_area_um2[i] = obj.area*yx_pxl_to_um2
            if self.mainWin.isSegm3D:
                IDs_vol_vox_3D[i] = obj.area
                IDs_vol_fl_3D[i] = obj.area*vox_to_fl_3D

        df['cell_area_pxl'] = pd.Series(data=IDs_area_pxl, index=IDs, dtype=float)
        df['cell_vol_vox'] = pd.Series(data=IDs_vol_vox, index=IDs, dtype=float)
        df['cell_area_um2'] = pd.Series(data=IDs_area_um2, index=IDs, dtype=float)
        df['cell_vol_fl'] = pd.Series(data=IDs_vol_fl, index=IDs, dtype=float)
        if self.mainWin.isSegm3D:
            df['cell_vol_vox_3D'] = pd.Series(data=IDs_vol_vox_3D, index=IDs, dtype=float)
            df['cell_vol_fl_3D'] = pd.Series(data=IDs_vol_fl_3D, index=IDs, dtype=float)

        return df

    def addAdditionalMetadata(self, posData, df):
        for col, val in posData.additionalMetadataValues().items():
            if col in df.columns:
                df.pop(col)
            df.insert(0, col, val)

    def run(self):
        last_pos = self.mainWin.last_pos
        save_metrics = self.mainWin.save_metrics
        self.time_last_pbar_update = time.time()
        mode = self.mode
        for p, posData in enumerate(self.mainWin.data[:last_pos]):
            if self.saveWin.aborted:
                self.finished.emit()
                return

            current_frame_i = posData.frame_i

            if not self.mainWin.isSnapshot:
                last_tracked_i = self.mainWin.last_tracked_i
                if last_tracked_i is None:
                    self.mainWin.saveWin.aborted = True
                    self.finished.emit()
                    return
            elif self.mainWin.isSnapshot:
                last_tracked_i = 0

            if p == 0:
                self.progressBar.emit(0, last_pos*(last_tracked_i+1), 0)

            try:
                segm_npz_path = posData.segm_npz_path
                acdc_output_csv_path = posData.acdc_output_csv_path
                last_tracked_i_path = posData.last_tracked_i_path
                segm_npy = posData.segm_data
                npz_delROIs_info = {}
                delROIs_info_path = posData.delROIs_info_path
                acdc_df_li = []
                keys = []

                # Add segmented channel data for calc metrics if requested
                add_user_channel_data = True
                for chName in self.mainWin.chNamesToSkip:
                    skipUserChannel = (
                        posData.filename.endswith(chName)
                        or posData.filename.endswith(f'{chName}_aligned')
                    )
                    if skipUserChannel:
                        add_user_channel_data = False

                if add_user_channel_data:
                    posData.fluo_data_dict[posData.filename] = posData.img_data

                posData.fluo_bkgrData_dict[posData.filename] = posData.bkgrData

                self.mainWin.getChNames(posData)

                self.progress.emit(f'Saving {posData.relPath}')
                end_i = self.mainWin.save_until_frame_i
                for frame_i, data_dict in enumerate(posData.allData_li[:end_i+1]):
                    if self.saveWin.aborted:
                        self.finished.emit()
                        return

                    # Build segm_npy
                    lab = data_dict['labels']
                    posData.lab = lab
                    if lab is None:
                        break

                    if posData.SizeT > 1:
                        segm_npy[frame_i] = lab
                    else:
                        segm_npy = lab

                    acdc_df = data_dict['acdc_df']

                    if self.saveOnlySegm:
                        continue

                    if acdc_df is None:
                        continue

                    if not np.any(lab):
                        continue

                    # Build acdc_df and index it in each frame_i of acdc_df_li
                    try:
                        acdc_df = load.pd_bool_to_int(acdc_df, inplace=False)
                        rp = data_dict['regionprops']
                        if save_metrics:
                            acdc_df = self.addMetrics_acdc_df(
                                acdc_df, rp, frame_i, lab, posData
                            )
                        elif mode == 'Cell cycle analysis':
                            acdc_df = self.addVolumeMetrics(
                                acdc_df, rp, posData
                            )
                        acdc_df_li.append(acdc_df)
                        key = (frame_i, posData.TimeIncrement*frame_i)
                        keys.append(key)
                    except Exception as error:
                        self.mutex.lock()
                        self.criticalMetrics.emit(traceback.format_exc())
                        self.waitCond.wait(self.mutex)
                        self.mutex.unlock()
                        self.finished.emit()
                        return

                    t = time.time()
                    exec_time = t - self.time_last_pbar_update
                    self.progressBar.emit(1, -1, exec_time)
                    self.time_last_pbar_update = t

                # Save segmentation file
                np.savez_compressed(segm_npz_path, np.squeeze(segm_npy))
                posData.segm_data = segm_npy

                if posData.segmInfo_df is not None:
                    try:
                        posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
                    except PermissionError:
                        err_msg = (
                            'The below file is open in another app '
                            '(Excel maybe?).\n\n'
                            f'{posData.segmInfo_df_csv_path}\n\n'
                            'Close file and then press "Ok".'
                        )
                        self.mutex.lock()
                        self.criticalPermissionError.emit(err_msg)
                        self.waitCond.wait(self.mutex)
                        self.mutex.unlock()
                        posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

                if self.saveOnlySegm:
                    continue

                if add_user_channel_data:
                    posData.fluo_data_dict.pop(posData.filename)

                posData.fluo_bkgrData_dict.pop(posData.filename)

                if posData.SizeT > 1:
                    self.progress.emit('Almost done...')
                    self.progressBar.emit(0, 0, 0)

                if acdc_df_li:
                    all_frames_acdc_df = pd.concat(
                        acdc_df_li, keys=keys,
                        names=['frame_i', 'time_seconds', 'Cell_ID']
                    )
                    if save_metrics:
                        self.addCombineMetrics_acdc_df(
                            posData, all_frames_acdc_df
                        )

                    self.addAdditionalMetadata(posData, all_frames_acdc_df)

                    all_frames_acdc_df = self._removeDeprecatedRows(
                        all_frames_acdc_df
                    )
                    try:
                        # Save segmentation metadata
                        all_frames_acdc_df.to_csv(acdc_output_csv_path)
                        posData.acdc_df = all_frames_acdc_df
                    except PermissionError:
                        err_msg = (
                            'The below file is open in another app '
                            '(Excel maybe?).\n\n'
                            f'{acdc_output_csv_path}\n\n'
                            'Close file and then press "Ok".'
                        )
                        self.mutex.lock()
                        self.criticalPermissionError.emit(err_msg)
                        self.waitCond.wait(self.mutex)
                        self.mutex.unlock()

                        # Save segmentation metadata
                        all_frames_acdc_df.to_csv(acdc_output_csv_path)
                        posData.acdc_df = all_frames_acdc_df
                    except Exception as e:
                        self.mutex.lock()
                        self.critical.emit(traceback.format_exc())
                        self.waitCond.wait(self.mutex)
                        self.mutex.unlock()

                with open(last_tracked_i_path, 'w+') as txt:
                    txt.write(str(frame_i))

                # Save combined metrics equations
                posData.saveCombineMetrics()

                posData.last_tracked_i = last_tracked_i

                # Go back to current frame
                posData.frame_i = current_frame_i
                self.mainWin.get_data()

                if mode == 'Segmentation and Tracking' or mode == 'Viewer':
                    self.progress.emit(
                        f'Saved data until frame number {frame_i+1}'
                    )
                elif mode == 'Cell cycle analysis':
                    self.progress.emit(
                        'Saved cell cycle annotations until frame '
                        f'number {last_tracked_i+1}'
                    )
                # self.progressBar.emit(1)
            except Exception as e:
                self.mutex.lock()
                self.critical.emit(traceback.format_exc())
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
        if self.mainWin.isSnapshot:
            self.progress.emit(f'Saved all {p+1} Positions!')

        self.finished.emit()

class guiWin(QMainWindow):
    """Main Window."""

    def __init__(
            self, app, parent=None, buttonToRestore=None,
            mainWin=None, version=None
        ):
        """Initializer."""

        super().__init__(parent)

        self._version = version

        from .trackers.YeaZ import tracking as tracking_yeaz
        self.tracking_yeaz = tracking_yeaz

        from .config import parser_args
        self.debug = parser_args['debug']

        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        self.app = app
        self.closeGUI = False

    def run(self):
        self.is_win = sys.platform.startswith("win")
        if self.is_win:
            self.openFolderText = 'Show in Explorer...'
        else:
            self.openFolderText = 'Reveal in Finder...'

        self.is_error_state = False
        logger, logs_path, log_path, log_filename = setupLogger(
            module='gui'
        )
        if self._version is not None:
            logger.info(f'Initializing GUI v{self._version}...')
        else:
            logger.info(f'Initializing GUI...')
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        self.loadLastSessionSettings()

        self.progressWin = None
        self.slideshowWin = None
        self.ccaTableWin = None
        self.customAnnotButton = None
        self.data_loaded = False
        self.highlightedID = 0
        self.hoverLabelID = 0
        self.expandingID = -1
        self.isDilation = True
        self.flag = True
        self.currentPropsID = 0
        self.isSegm3D = False
        self.newSegmEndName = ''
        self.closeGUI = False

        self.setWindowTitle("Cell-ACDC - GUI")
        self.setWindowIcon(QIcon(":icon.ico"))
        self.setAcceptDrops(True)

        self.checkableButtons = []
        self.LeftClickButtons = []
        self.customAnnotDict = {}

        self.isSnapshot = False
        self.debugFlag = False
        self.pos_i = 0
        self.save_until_frame_i = 0
        self.countKeyPress = 0
        self.xHoverImg, self.yHoverImg = None, None

        # Buttons added to QButtonGroup will be mutually exclusive
        self.checkableQButtonsGroup = QButtonGroup(self)
        self.checkableQButtonsGroup.setExclusive(False)

        self.gui_createCursors()

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()
        self.gui_createControlsToolbar()
        self.gui_createLeftSideWidgets()
        self.gui_createPropsDockWidget()

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_createGraphicsPlots()
        self.gui_addGraphicsItems()

        self.gui_createImg1Widgets()
        self.gui_createLabWidgets()
        self.gui_addBottomWidgetsToBottomLayout()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = self.gui_createMainLayout()
        self.mainLayout = mainLayout

        mainContainer.setLayout(mainLayout)

        self.isEditActionsConnected = False

        self.show()

    def loadLastSessionSettings(self):
        self.settings_csv_path = settings_csv_path
        if os.path.exists(settings_csv_path):
            self.df_settings = pd.read_csv(
                settings_csv_path, index_col='setting'
            )
            if 'is_bw_inverted' not in self.df_settings.index:
                self.df_settings.at['is_bw_inverted', 'value'] = 'No'
            else:
                self.df_settings.loc['is_bw_inverted'] = (
                    self.df_settings.loc['is_bw_inverted'].astype(str)
                )
            if 'fontSize' not in self.df_settings.index:
                self.df_settings.at['fontSize', 'value'] = '12pt'
            if 'fontSize' in self.df_settings.index:
                _s = self.df_settings.at['fontSize', 'value']
                self.df_settings.at['fontSize', 'value'] = _s.replace('px', 'pt')
            if 'overlayColor' not in self.df_settings.index:
                self.df_settings.at['overlayColor', 'value'] = '255-255-0'
            if 'how_normIntensities' not in self.df_settings.index:
                raw = 'Do not normalize. Display raw image'
                self.df_settings.at['how_normIntensities', 'value'] = raw
        else:
            idx = ['is_bw_inverted', 'fontSize', 'overlayColor', 'how_normIntensities']
            values = ['No', '12px', '255-255-0', 'raw']
            self.df_settings = pd.DataFrame({
                'setting': idx,'value': values}
            ).set_index('setting')

    def dragEnterEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(file_path):
            exp_path = file_path
            basename = os.path.basename(file_path)
            if basename.find('Position_')!=-1 or basename=='Images':
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.logger.info(f'Dragged and dropped path "{file_path}"')
        basename = os.path.basename(file_path)
        if os.path.isdir(file_path):
            exp_path = file_path
            self.openFolder(exp_path=exp_path)
        else:
            self.openFile(file_path=file_path)

    def leaveEvent(self, event):
        if self.slideshowWin is not None:
            posData = self.data[self.pos_i]
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (
                (slideshowWinTop < mainWinBottom) and
                (slideshowWinLeft < mainWinRight)
            )

            autoActivate = (
                self.data_loaded and not
                overlap and not
                posData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.slideshowWin.setFocus(True)
                self.slideshowWin.activateWindow()

    def enterEvent(self, event):
        if self.slideshowWin is not None:
            posData = self.data[self.pos_i]
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (
                (slideshowWinTop < mainWinBottom) and
                (slideshowWinLeft < mainWinRight)
            )

            autoActivate = (
                self.data_loaded and not
                overlap and not
                posData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.setFocus(True)
                self.activateWindow()

    def isPanImageClick(self, mouseEvent, modifiers):
        left_click = mouseEvent.button() == Qt.MouseButton.LeftButton
        return modifiers == Qt.AltModifier and left_click

    def isMiddleClick(self, mouseEvent, modifiers):
        if sys.platform == 'darwin':
            middle_click = (
                mouseEvent.button() == Qt.MouseButton.LeftButton
                and modifiers == Qt.ControlModifier
            )
        else:
            middle_click = mouseEvent.button() == Qt.MouseButton.MidButton
        return middle_click

    def gui_createCursors(self):
        pixmap = QPixmap(":wand_cursor.svg")
        self.wandCursor = QCursor(pixmap, 16, 16)

        pixmap = QPixmap(":curv_cursor.svg")
        self.curvCursor = QCursor(pixmap, 16, 16)

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.openFileAction)
        # Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveAsAction)
        fileMenu.addAction(self.quickSaveAction)
        fileMenu.addAction(self.loadFluoAction)
        # Separator
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addSeparator()
        # Font size
        self.fontSize = self.df_settings.at['fontSize', 'value']
        self.fontSizeMenu = editMenu.addMenu("Font size")
        fontActionGroup = QActionGroup(self)
        fontActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        fs = int(re.findall(r'(\d+)pt', self.fontSize)[0])
        for i in range(2,25):
            action = QAction(self)
            action.setText(f'{i}')
            action.setCheckable(True)
            if i == fs:
                action.setChecked(True)
                self.fontSizeAction = action
            fontActionGroup.addAction(action)
            action = self.fontSizeMenu.addAction(action)
        editMenu.addAction(self.editTextIDsColorAction)
        editMenu.addAction(self.editOverlayColorAction)
        editMenu.addAction(self.manuallyEditCcaAction)
        editMenu.addAction(self.viewCcaTableAction)
        editMenu.addAction(self.enableSmartTrackAction)
        editMenu.addAction(self.enableAutoZoomToCellsAction)


        # Image menu
        ImageMenu = menuBar.addMenu("&Image")
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.imgPropertiesAction)
        filtersMenu = ImageMenu.addMenu("Filters")
        filtersMenu.addAction(self.gaussBlurAction)
        filtersMenu.addAction(self.diffGaussFilterAction)
        filtersMenu.addAction(self.edgeDetectorAction)
        filtersMenu.addAction(self.entropyFilterAction)
        normalizeIntensitiesMenu = ImageMenu.addMenu("Normalize intensities")
        normalizeIntensitiesMenu.addAction(self.normalizeRawAction)
        normalizeIntensitiesMenu.addAction(self.normalizeToFloatAction)
        # normalizeIntensitiesMenu.addAction(self.normalizeToUbyteAction)
        normalizeIntensitiesMenu.addAction(self.normalizeRescale0to1Action)
        normalizeIntensitiesMenu.addAction(self.normalizeByMaxAction)
        ImageMenu.addAction(self.invertBwAction)
        ImageMenu.addAction(self.shuffleCmapAction)
        ImageMenu.addAction(self.zoomToObjsAction)
        ImageMenu.addAction(self.zoomOutAction)

        # Segment menu
        SegmMenu = menuBar.addMenu("&Segment")
        SegmMenu.addSeparator()
        self.segmSingleFrameMenu = SegmMenu.addMenu('Segment displayed frame')
        for action in self.segmActions:
            self.segmSingleFrameMenu.addAction(action)

        self.segmSingleFrameMenu.addAction(self.addCustomModelAction)

        self.segmVideoMenu = SegmMenu.addMenu('Segment multiple frames')
        for action in self.segmActionsVideo:
            self.segmVideoMenu.addAction(action)

        self.segmVideoMenu.addAction(self.addCustomModelAction)

        SegmMenu.addAction(self.SegmActionRW)
        SegmMenu.addAction(self.postProcessSegmAction)
        SegmMenu.addAction(self.autoSegmAction)
        SegmMenu.addAction(self.relabelSequentialAction)
        SegmMenu.aboutToShow.connect(self.nonViewerEditMenuOpened)

        # Tracking menu
        trackingMenu = menuBar.addMenu("&Tracking")
        self.trackingMenu = trackingMenu
        trackingMenu.addSeparator()
        selectTrackAlgoMenu = trackingMenu.addMenu(
            'Select real-time tracking algorithm'
        )
        selectTrackAlgoMenu.addAction(self.trackWithAcdcAction)
        selectTrackAlgoMenu.addAction(self.trackWithYeazAction)

        trackingMenu.addAction(self.repeatTrackingVideoAction)

        trackingMenu.addAction(self.repeatTrackingMenuAction)
        trackingMenu.aboutToShow.connect(self.nonViewerEditMenuOpened)

        # Measurements menu
        measurementsMenu = menuBar.addMenu("&Measurements")
        self.measurementsMenu = measurementsMenu
        measurementsMenu.addSeparator()
        measurementsMenu.addAction(self.setMeasurementsAction)
        measurementsMenu.addAction(self.addCustomMetricAction)
        measurementsMenu.addAction(self.addCombineMetricAction)
        measurementsMenu.setDisabled(True)

        # Settings menu
        self.settingsMenu = QMenu("Settings", self)
        menuBar.addMenu(self.settingsMenu)
        self.settingsMenu.addSeparator()

        # Help menu
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.tipsAction)
        helpMenu.addAction(self.UserManualAction)
        # helpMenu.addAction(self.aboutAction)

    def gui_createToolBars(self):
        # File toolbar
        fileToolBar = self.addToolBar("File")
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addAction(self.showInExplorerAction)
        # fileToolBar.addAction(self.reloadAction)
        fileToolBar.addAction(self.undoAction)
        fileToolBar.addAction(self.redoAction)
        self.fileToolBar = fileToolBar
        self.setEnabledFileToolbar(False)

        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

        # Navigation toolbar
        navigateToolBar = QToolBar("Navigation", self)
        navigateToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolBar)
        navigateToolBar.addAction(self.prevAction)
        navigateToolBar.addAction(self.nextAction)
        navigateToolBar.addAction(self.findIdAction)

        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        self.slideshowButton.setToolTip('Open slideshow (Ctrl+W)')
        navigateToolBar.addWidget(self.slideshowButton)

        self.overlayButton = widgets.rightClickToolButton(parent=self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip('Overlay fluorescent image\n'
        'NOTE: Button has a green background if you successfully loaded fluorescent data\n\n'
        'If you need to overlay a different channel load (or load it again)\n'
        'from "File --> Load fluorescent images" menu.')
        navigateToolBar.addWidget(self.overlayButton)
        # self.checkableButtons.append(self.overlayButton)
        # self.checkableQButtonsGroup.addButton(self.overlayButton)

        self.rulerButton = QToolButton(self)
        self.rulerButton.setIcon(QIcon(":ruler.svg"))
        self.rulerButton.setCheckable(True)
        self.rulerButton.setToolTip(
            'Draw a straight line and show its length. '
            'Length is displayed on the bottom-right corner.'
        )
        navigateToolBar.addWidget(self.rulerButton)
        self.checkableButtons.append(self.rulerButton)
        self.LeftClickButtons.append(self.rulerButton)

        # fluorescent image color widget
        colorsToolBar = QToolBar("Colors", self)

        self.overlayColorButton = pg.ColorButton(self, color=(230,230,230))
        self.overlayColorButton.setDisabled(True)
        colorsToolBar.addWidget(self.overlayColorButton)

        self.textIDsColorButton = pg.ColorButton(self)
        colorsToolBar.addWidget(self.textIDsColorButton)

        self.addToolBar(colorsToolBar)
        colorsToolBar.setVisible(False)

        self.navigateToolBar = navigateToolBar

        # cca toolbar
        ccaToolBar = QToolBar("Cell cycle annotations", self)
        self.addToolBar(ccaToolBar)

        # Assign mother to bud button
        self.assignBudMothButton = QToolButton(self)
        self.assignBudMothButton.setIcon(QIcon(":assign-motherbud.svg"))
        self.assignBudMothButton.setCheckable(True)
        self.assignBudMothButton.setShortcut('a')
        self.assignBudMothButton.setVisible(False)
        self.assignBudMothButton.setToolTip(
            'Toggle "Assign bud to mother cell" mode ON/OFF\n\n'
            'ACTION: press with right button on bud and release on mother '
            '(right-click drag-and-drop)\n\n'
            'SHORTCUT: "A" key'
        )
        ccaToolBar.addWidget(self.assignBudMothButton)
        self.checkableButtons.append(self.assignBudMothButton)
        self.checkableQButtonsGroup.addButton(self.assignBudMothButton)

        # Set is_history_known button
        self.setIsHistoryKnownButton = QToolButton(self)
        self.setIsHistoryKnownButton.setIcon(QIcon(":history.svg"))
        self.setIsHistoryKnownButton.setCheckable(True)
        self.setIsHistoryKnownButton.setShortcut('u')
        self.setIsHistoryKnownButton.setVisible(False)
        self.setIsHistoryKnownButton.setToolTip(
            'Toggle "Annotate unknown history" mode ON/OFF\n\n'
            'EXAMPLE: useful for cells appearing from outside of the field of view\n\n'
            'ACTION: Right-click on cell\n\n'
            'SHORTCUT: "U" key'
        )
        ccaToolBar.addWidget(self.setIsHistoryKnownButton)
        self.checkableButtons.append(self.setIsHistoryKnownButton)
        self.checkableQButtonsGroup.addButton(self.setIsHistoryKnownButton)

        ccaToolBar.addAction(self.assignBudMothAutoAction)
        ccaToolBar.addAction(self.reInitCcaAction)
        ccaToolBar.setVisible(False)
        self.ccaToolBar = ccaToolBar

        # Edit toolbar
        editToolBar = QToolBar("Edit", self)
        self.addToolBar(editToolBar)

        self.brushButton = QToolButton(self)
        self.brushButton.setIcon(QIcon(":brush.svg"))
        self.brushButton.setCheckable(True)
        self.brushButton.setToolTip(
            'Edit segmentation labels with a circular brush.\n'
            'Increase brush size with UP/DOWN arrows on the keyboard.\n\n'
            'Default behaviour:\n\n'
            '   - Painting on the background will create a new label.\n'
            '   - Edit an existing label by starting to paint on the label\n'
            '     (brush cursor changes color when hovering an existing label).\n'
            '   - Painting is always UNDER existing labels, unless you press\n'
            '     "b" key twice quickly. If double-press is successfull, '
            '     then brush button is red and brush cursor always white.\n\n'
            'SHORTCUT: "B" key')
        editToolBar.addWidget(self.brushButton)
        self.checkableButtons.append(self.brushButton)
        self.LeftClickButtons.append(self.brushButton)

        self.eraserButton = QToolButton(self)
        self.eraserButton.setIcon(QIcon(":eraser.png"))
        self.eraserButton.setCheckable(True)
        self.eraserButton.setToolTip(
            'Erase segmentation labels with a circular eraser.\n'
            'Increase eraser size with UP/DOWN arrows on the keyboard.\n\n'
            'Default behaviour:\n\n'
            '   - Starting to erase from the background (cursor is a red circle)\n '
            '     will erase any labels you hover above.\n'
            '   - Starting to erase from a specific label will erase only that label\n'
            '     (cursor is a circle with the color of the label).\n'
            '   - To enforce erasing all labels no matter where you start from\n'
            '     double-press "X" key. If double-press is successfull,\n'
            '     then eraser button is red and eraser cursor always red.\n\n'
            'SHORTCUT: "X" key')
        editToolBar.addWidget(self.eraserButton)
        self.checkableButtons.append(self.eraserButton)
        self.LeftClickButtons.append(self.eraserButton)

        self.curvToolButton = QToolButton(self)
        self.curvToolButton.setIcon(QIcon(":curvature-tool.svg"))
        self.curvToolButton.setCheckable(True)
        self.curvToolButton.setShortcut('c')
        self.curvToolButton.setToolTip(
            'Toggle "Curvature tool" ON/OFF\n\n'
            'ACTION: left-clicks for manual spline anchors,\n'
            'right button for drawing auto-contour\n\n'
            'SHORTCUT: "C" key')
        editToolBar.addWidget(self.curvToolButton)
        self.LeftClickButtons.append(self.curvToolButton)
        # self.checkableButtons.append(self.curvToolButton)

        self.wandToolButton = QToolButton(self)
        self.wandToolButton.setIcon(QIcon(":magic_wand.svg"))
        self.wandToolButton.setCheckable(True)
        self.wandToolButton.setShortcut('w')
        self.wandToolButton.setToolTip(
            'Toggle "Magic wand tool" ON/OFF\n\n'
            'ACTION: left-click for single selection,\n'
            'or left-click and then drag for continous selection\n\n'
            'SHORTCUT: "W" key')
        editToolBar.addWidget(self.wandToolButton)
        self.LeftClickButtons.append(self.wandToolButton)

        self.hullContToolButton = QToolButton(self)
        self.hullContToolButton.setIcon(QIcon(":hull.svg"))
        self.hullContToolButton.setCheckable(True)
        self.hullContToolButton.setShortcut('k')
        self.hullContToolButton.setToolTip(
            'Toggle "Hull contour" ON/OFF\n\n'
            'ACTION: right-click on a cell to replace it with its hull contour.\n'
            'Use it to fill cracks and holes.\n\n'
            'SHORTCUT: "K" key')
        editToolBar.addWidget(self.hullContToolButton)
        self.checkableButtons.append(self.hullContToolButton)
        self.checkableQButtonsGroup.addButton(self.hullContToolButton)

        self.fillHolesToolButton = QToolButton(self)
        self.fillHolesToolButton.setIcon(QIcon(":fill_holes.svg"))
        self.fillHolesToolButton.setCheckable(True)
        self.fillHolesToolButton.setShortcut('f')
        self.fillHolesToolButton.setToolTip(
            'Toggle "Fill holes" ON/OFF\n\n'
            'ACTION: right-click on a cell to fill holes\n\n'
            'SHORTCUT: "F" key')
        editToolBar.addWidget(self.fillHolesToolButton)
        self.checkableButtons.append(self.fillHolesToolButton)
        self.checkableQButtonsGroup.addButton(self.fillHolesToolButton)

        self.moveLabelToolButton = QToolButton(self)
        self.moveLabelToolButton.setIcon(QIcon(":moveLabel.svg"))
        self.moveLabelToolButton.setCheckable(True)
        self.moveLabelToolButton.setShortcut('p')
        self.moveLabelToolButton.setToolTip(
            'Toggle "Move label (a.k.a. mask)" ON/OFF\n\n'
            'ACTION: right-click drag and drop a labels to move it around\n\n'
            'SHORTCUT: "P" key')
        editToolBar.addWidget(self.moveLabelToolButton)
        self.checkableButtons.append(self.moveLabelToolButton)
        self.checkableQButtonsGroup.addButton(self.moveLabelToolButton)

        self.expandLabelToolButton = QToolButton(self)
        self.expandLabelToolButton.setIcon(QIcon(":expandLabel.svg"))
        self.expandLabelToolButton.setCheckable(True)
        self.expandLabelToolButton.setShortcut('e')
        self.expandLabelToolButton.setToolTip(
            'Toggle "Expand/Shrink label (a.k.a. masks)" ON/OFF\n\n'
            'ACTION: leave mouse cursor on the label you want to expand/shrink'
            'and press arrow up/down on the keyboard to expand/shrink the mask.\n\n'
            'SHORTCUT: "E" key')
        editToolBar.addWidget(self.expandLabelToolButton)
        self.expandLabelToolButton.hide()
        self.checkableButtons.append(self.expandLabelToolButton)
        self.LeftClickButtons.append(self.expandLabelToolButton)
        self.checkableQButtonsGroup.addButton(self.expandLabelToolButton)

        self.editID_Button = QToolButton(self)
        self.editID_Button.setIcon(QIcon(":edit-id.svg"))
        self.editID_Button.setCheckable(True)
        self.editID_Button.setShortcut('n')
        self.editID_Button.setToolTip(
            'Toggle "Edit ID" mode ON/OFF\n\n'
            'EXAMPLE: manually change ID of a cell\n\n'
            'ACTION: right-click on cell\n\n'
            'SHORTCUT: "N" key')
        editToolBar.addWidget(self.editID_Button)
        self.checkableButtons.append(self.editID_Button)
        self.checkableQButtonsGroup.addButton(self.editID_Button)

        self.separateBudButton = QToolButton(self)
        self.separateBudButton.setIcon(QIcon(":separate-bud.svg"))
        self.separateBudButton.setCheckable(True)
        self.separateBudButton.setShortcut('s')
        self.separateBudButton.setToolTip(
            'Toggle "Automatic/manual separation" mode ON/OFF\n\n'
            'EXAMPLE: separate mother-bud fused together\n\n'
            'ACTION: right-click for automatic and left-click for manual\n\n'
            'SHORTCUT: "S" key'
        )
        editToolBar.addWidget(self.separateBudButton)
        self.checkableButtons.append(self.separateBudButton)
        self.checkableQButtonsGroup.addButton(self.separateBudButton)

        self.mergeIDsButton = QToolButton(self)
        self.mergeIDsButton.setIcon(QIcon(":merge-IDs.svg"))
        self.mergeIDsButton.setCheckable(True)
        self.mergeIDsButton.setShortcut('m')
        self.mergeIDsButton.setToolTip(
            'Toggle "Merge IDs" mode ON/OFF\n\n'
            'EXAMPLE: merge/fuse two cells together\n\n'
            'ACTION: right-click\n\n'
            'SHORTCUT: "M" key'
        )
        editToolBar.addWidget(self.mergeIDsButton)
        self.checkableButtons.append(self.mergeIDsButton)
        self.checkableQButtonsGroup.addButton(self.mergeIDsButton)

        self.binCellButton = QToolButton(self)
        self.binCellButton.setIcon(QIcon(":bin.svg"))
        self.binCellButton.setCheckable(True)
        self.binCellButton.setToolTip(
            'Toggle "Annotate cell as removed from analysis" mode ON/OFF\n\n'
            'EXAMPLE: annotate that a cell is removed from downstream analysis.\n'
            '"is_cell_excluded" set to True in acdc_output.csv table\n\n'
            'ACTION: right-click\n\n'
            'SHORTCUT: "R" key'
        )
        self.binCellButton.setShortcut("r")
        editToolBar.addWidget(self.binCellButton)
        self.checkableButtons.append(self.binCellButton)
        self.checkableQButtonsGroup.addButton(self.binCellButton)

        self.ripCellButton = QToolButton(self)
        self.ripCellButton.setIcon(QIcon(":rip.svg"))
        self.ripCellButton.setCheckable(True)
        self.ripCellButton.setToolTip(
            'Toggle "Annotate cell as dead" mode ON/OFF\n\n'
            'EXAMPLE: annotate that a cell is dead.\n'
            '"is_cell_dead" set to True in acdc_output.csv table\n\n'
            'ACTION: right-click\n\n'
            'SHORTCUT: "D" key'
        )
        self.ripCellButton.setShortcut("d")
        editToolBar.addWidget(self.ripCellButton)
        self.checkableButtons.append(self.ripCellButton)
        self.checkableQButtonsGroup.addButton(self.ripCellButton)

        editToolBar.addAction(self.addDelRoiAction)
        editToolBar.addAction(self.delBorderObjAction)

        editToolBar.addAction(self.repeatTrackingAction)

        # Widgets toolbar
        widgetsToolBar = QToolBar("Widgets", self)
        self.addToolBar(widgetsToolBar)

        self.disableTrackingCheckBox = QCheckBox("Disable tracking")
        self.disableTrackingCheckBox.setLayoutDirection(Qt.RightToLeft)
        self.disableTrackingAction = widgetsToolBar.addWidget(
                                            self.disableTrackingCheckBox)
        self.disableTrackingAction.setVisible(False)

        self.brushSizeSpinbox = QSpinBox()
        self.brushSizeSpinbox.setValue(4)
        brushSizeLabel = QLabel('   Size: ')
        brushSizeLabel.setBuddy(self.brushSizeSpinbox)
        self.brushSizeLabelAction = widgetsToolBar.addWidget(brushSizeLabel)
        self.brushSizeAction = widgetsToolBar.addWidget(self.brushSizeSpinbox)
        self.brushSizeLabelAction.setVisible(False)
        self.brushSizeAction.setVisible(False)

        widgetsToolBar.setVisible(False)
        self.widgetsToolBar = widgetsToolBar

        # Edit toolbar
        modeToolBar = QToolBar("Mode", self)
        self.addToolBar(modeToolBar)

        self.modeComboBox = QComboBox()
        self.modeItems = [
            'Segmentation and Tracking',
            'Cell cycle analysis',
            'Viewer',
            'Custom annotations'
        ]
        self.modeComboBox.addItems(self.modeItems)
        self.modeComboBoxLabel = QLabel('    Mode: ')
        self.modeComboBoxLabel.setBuddy(self.modeComboBox)
        modeToolBar.addWidget(self.modeComboBoxLabel)
        modeToolBar.addWidget(self.modeComboBox)
        modeToolBar.setVisible(False)


        self.modeToolBar = modeToolBar
        self.editToolBar = editToolBar
        self.editToolBar.setVisible(False)
        self.navigateToolBar.setVisible(False)

        self.gui_populateToolSettingsMenu()

        self.gui_createAnnotateToolbar()

        # toolbarSize = 58
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # ccaToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # widgetsToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # modeToolBar.setIconSize(QSize(toolbarSize, toolbarSize))

    def gui_createAnnotateToolbar(self):
        # Edit toolbar
        self.annotateToolbar = QToolBar("Custom annotations", self)
        self.annotateToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(Qt.LeftToolBarArea, self.annotateToolbar)
        self.annotateToolbar.addAction(self.addCustomAnnotationAction)
        self.annotateToolbar.addAction(self.viewAllCustomAnnotAction)
        self.annotateToolbar.setVisible(False)

    def gui_createMainLayout(self):
        mainLayout = QGridLayout()
        row = 0
        mainLayout.addLayout(self.leftSideDocksLayout, row, 0, 2, 1)
        mainLayout.addWidget(self.graphLayout, row, 1, 1, 2)
        mainLayout.addWidget(self.labelsGrad, row, 3)

        row += 1
        mainLayout.addLayout(
            self.bottomLayout, row, 1, 1, 2, alignment=Qt.AlignLeft
        )
        self.bottomLayout.row = row
        mainLayout.setRowStretch(row, 0)

        return mainLayout

    def gui_createPropsDockWidget(self):
        self.propsDockWidget = QDockWidget('Cell-ACDC objects', self)
        self.guiTabControl = widgets.guiTabControl(self.propsDockWidget)

        self.propsDockWidget.setWidget(self.guiTabControl)
        self.propsDockWidget.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )
        self.propsDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )

        self.addDockWidget(Qt.LeftDockWidgetArea, self.propsDockWidget)
        self.propsDockWidget.hide()

    def gui_createControlsToolbar(self):
        self.addToolBarBreak()

        self.wandControlsToolbar = QToolBar("Controls", self)
        self.wandToleranceSlider = widgets.sliderWithSpinBox(
            title='Tolerance', title_loc='in_line'
        )
        self.wandToleranceSlider.setValue(5)

        self.wandAutoFillCheckbox = QCheckBox('Auto-fill holes')

        col = 3
        self.wandToleranceSlider.layout.addWidget(
            self.wandAutoFillCheckbox, 0, col
        )

        col += 1
        self.wandToleranceSlider.layout.setColumnStretch(col, 21)

        self.wandControlsToolbar.addWidget(self.wandToleranceSlider)

        self.addToolBar(Qt.TopToolBarArea , self.wandControlsToolbar)
        self.wandControlsToolbar.setVisible(False)

    def gui_populateToolSettingsMenu(self):
        brushHoverModeActionGroup = QActionGroup(self)
        brushHoverModeActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        self.brushHoverCenterModeAction = QAction()
        self.brushHoverCenterModeAction.setCheckable(True)
        self.brushHoverCenterModeAction.setText(
            'Use center of the brush/eraser cursor to determine hover ID'
        )
        self.brushHoverCircleModeAction = QAction()
        self.brushHoverCircleModeAction.setCheckable(True)
        self.brushHoverCircleModeAction.setText(
            'Use the entire circle of the brush/eraser cursor to determine hover ID'
        )
        brushHoverModeActionGroup.addAction(self.brushHoverCenterModeAction)
        brushHoverModeActionGroup.addAction(self.brushHoverCircleModeAction)
        brushHoverModeMenu = self.settingsMenu.addMenu(
            'Brush/eraser cursor hovering mode'
        )
        brushHoverModeMenu.addAction(self.brushHoverCenterModeAction)
        brushHoverModeMenu.addAction(self.brushHoverCircleModeAction)

        if 'useCenterBrushCursorHoverID' not in self.df_settings.index:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'Yes'

        useCenterBrushCursorHoverID = self.df_settings.at[
            'useCenterBrushCursorHoverID', 'value'
        ] == 'Yes'
        self.brushHoverCenterModeAction.setChecked(useCenterBrushCursorHoverID)
        self.brushHoverCircleModeAction.setChecked(not useCenterBrushCursorHoverID)

        self.brushHoverCenterModeAction.toggled.connect(
            self.useCenterBrushCursorHoverIDtoggled
        )

        self.settingsMenu.addSeparator()

        for button in self.checkableQButtonsGroup.buttons():
            toolName = re.findall('Toggle "(.*)"', button.toolTip())[0]
            menu = self.settingsMenu.addMenu(f'{toolName} tool')
            action = QAction(button)
            action.setText('Keep tool active after using it')
            action.setCheckable(True)
            if toolName in self.df_settings.index:
                action.setChecked(True)
            action.toggled.connect(self.keepToolActiveActionToggled)
            menu.addAction(action)

        self.warnLostCellsAction = QAction()
        self.warnLostCellsAction.setText('Show pop-up warning for lost cells')
        self.warnLostCellsAction.setCheckable(True)
        self.warnLostCellsAction.setChecked(True)
        self.settingsMenu.addAction(self.warnLostCellsAction)

        warnEditingWithAnnotTexts = {
            'Delete ID': 'Show warning when deleting ID that has annotations',
            'Separate IDs': 'Show warning when separating IDs that have annotations',
            'Edit ID': 'Show warning when editing ID that has annotations',
            'Annotate ID as dead':
                'Show warning when annotating dead ID that has annotations',
            'Delete ID with eraser':
                'Show warning when erasing ID that has annotations',
            'Add new ID with brush tool':
                'Show warning when adding new ID (brush) that has annotations',
            'Merge IDs':
                'Show warning when merging IDs that have annotations',
            'Add new ID with curvature tool':
                'Show warning when adding new ID (curv. tool) that has annotations',
            'Add new ID with magic-wand':
                'Show warning when adding new ID (magic-wand) that has annotations',
            'Delete IDs using ROI':
                'Show warning when using ROIs to delete IDs that have annotations',
        }
        self.warnEditingWithAnnotActions = {}
        for key, desc in warnEditingWithAnnotTexts.items():
            action = QAction()
            action.setText(desc)
            action.setCheckable(True)
            action.setChecked(True)
            action.removeAnnot = False
            self.warnEditingWithAnnotActions[key] = action
            self.settingsMenu.addAction(action)


    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Permanent widget
        self.wcLabel = QLabel('')
        self.statusbar.addPermanentWidget(self.wcLabel)
        self.statusBarLabel = QLabel('')
        self.statusbar.addWidget(self.statusBarLabel)


    def gui_createActions(self):
        # File actions
        self.newAction = QAction(self)
        self.newAction.setText("&New")
        self.newAction.setIcon(QIcon(":file-new.svg"))
        self.openAction = QAction(
            QIcon(":folder-open.svg"), "&Load folder...", self
        )
        self.openFileAction = QAction(
            QIcon(":image.svg"),"&Open image/video file...", self
        )
        self.saveAction = QAction(QIcon(":file-save.svg"), "Save", self)
        self.saveAsAction = QAction("Save as...", self)
        self.quickSaveAction = QAction("Save only segm. file", self)
        self.loadFluoAction = QAction("Load fluorescent images...", self)
        # self.reloadAction = QAction(
        #     QIcon(":reload.svg"), "Reload segmentation file", self
        # )
        self.showInExplorerAction = QAction(
            QIcon(":drawer.svg"), f"&{self.openFolderText}", self
        )
        self.exitAction = QAction("&Exit", self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo", self)
        self.redoAction = QAction(QIcon(":redo.svg"), "Redo", self)
        # String-based key sequences
        self.newAction.setShortcut("Ctrl+N")
        self.openAction.setShortcut("Ctrl+O")
        self.saveAsAction.setShortcut("Ctrl+Shift+S")
        self.saveAction.setShortcut("Ctrl+Alt+S")
        self.quickSaveAction.setShortcut("Ctrl+S")
        self.undoAction.setShortcut("Ctrl+Z")
        self.redoAction.setShortcut("Ctrl+Y")
        # Help tips
        newTip = "Create a new segmentation file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setToolTip(newTip)
        self.newAction.setWhatsThis("Create a new empty segmentation file")

        self.findIdAction = QAction(self)
        self.findIdAction.setIcon(QIcon(":find.svg"))
        self.findIdAction.setShortcut('Ctrl+F')
        self.findIdAction.setToolTip(
            'Find and highlight ID (Ctrl+F). Press Esc to exist highlight mode'
        )

        # Edit actions
        models = myutils.get_list_of_models()
        self.segmActions = []
        self.modelNames = []
        self.acdcSegment_li = []
        self.models = []
        for model_name in models:
            action = QAction(f"{model_name}...")
            self.segmActions.append(action)
            self.modelNames.append(model_name)
            self.models.append(None)
            self.acdcSegment_li.append(None)
            action.setDisabled(True)

        self.addCustomModelAction = QAction('Custom model...', self)

        self.segmActionsVideo = []
        for model_name in models:
            action = QAction(f"{model_name}...")
            self.segmActionsVideo.append(action)
            action.setDisabled(True)
        self.SegmActionRW = QAction("Random walker...", self)
        self.SegmActionRW.setDisabled(True)

        self.postProcessSegmAction = QAction(
            "Segmentation post-processing...", self
        )
        self.postProcessSegmAction.setDisabled(True)
        self.postProcessSegmAction.setCheckable(True)

        self.prevAction = QAction(QIcon(":arrow-left.svg"),
                                        "Previous frame", self)
        self.nextAction = QAction(QIcon(":arrow-right.svg"),
                                  "Next Frame", self)
        self.prevAction.setShortcut("left")
        self.nextAction.setShortcut("right")
        # self.nextAction.setVisible(False)
        # self.prevAction.setVisible(False)

        self.repeatTrackingAction = QAction(
            QIcon(":repeat-tracking.svg"), "Repeat tracking", self
        )
        self.repeatTrackingAction.setToolTip(
            'Repeat tracking on current frame\n'
            'SHORTCUT: "Shift+T"'
        )
        self.repeatTrackingMenuAction = QAction(
            'Repeat tracking on current frame...', self
        )
        self.repeatTrackingMenuAction.setDisabled(True)
        self.repeatTrackingMenuAction.setShortcut('Shift+T')

        self.repeatTrackingVideoAction = QAction(
            'Repeat tracking on multiple frames...', self
        )
        self.repeatTrackingVideoAction.setDisabled(True)
        self.repeatTrackingVideoAction.setShortcut('Alt+Shift+T')

        trackingAlgosGroup = QActionGroup(self)
        self.trackWithAcdcAction = QAction('Cell-ACDC', self)
        self.trackWithAcdcAction.setCheckable(True)
        trackingAlgosGroup.addAction(self.trackWithAcdcAction)

        self.trackWithYeazAction = QAction('YeaZ', self)
        self.trackWithYeazAction.setCheckable(True)
        trackingAlgosGroup.addAction(self.trackWithYeazAction)


        self.trackWithAcdcAction.setChecked(True)
        if 'tracking_algorithm' in self.df_settings.index:
            trackingAlgo = self.df_settings.at['tracking_algorithm', 'value']
            if trackingAlgo == 'Cell-ACDC':
                self.trackWithAcdcAction.setChecked(True)
            elif trackingAlgo == 'YeaZ':
                self.trackWithYeazAction.setChecked(True)

        self.setMeasurementsAction = QAction('Set measurements...')
        self.addCustomMetricAction = QAction('Add custom measurement...')
        self.addCombineMetricAction = QAction('Add combined measurement...')

        # Standard key sequence
        # self.copyAction.setShortcut(QKeySequence.Copy)
        # self.pasteAction.setShortcut(QKeySequence.Paste)
        # self.cutAction.setShortcut(QKeySequence.Cut)
        # Help actions
        self.tipsAction = QAction("Tips and tricks...", self)
        self.UserManualAction = QAction("User Manual...", self)
        # self.aboutAction = QAction("&About...", self)

        # Assign mother to bud button
        self.assignBudMothAutoAction = QAction(self)
        self.assignBudMothAutoAction.setIcon(QIcon(":autoAssign.svg"))
        self.assignBudMothAutoAction.setVisible(False)
        self.assignBudMothAutoAction.setToolTip(
            'Automatically assign buds to mothers using YeastMate'
        )


        self.reInitCcaAction = QAction(self)
        self.reInitCcaAction.setIcon(QIcon(":reinitCca.svg"))
        self.reInitCcaAction.setVisible(False)
        self.reInitCcaAction.setToolTip(
            'Reinitialize cell cycle annotations table from this frame onward.\n'
            'NOTE: This will erase all the already annotated future frames information\n'
            '(from the current session not the saved information)'
        )

        self.editTextIDsColorAction = QAction('Edit text on IDs color...', self)
        self.editTextIDsColorAction.setDisabled(True)

        self.editOverlayColorAction = QAction('Edit overlay color...', self)
        self.editOverlayColorAction.setDisabled(True)

        self.manuallyEditCcaAction = QAction(
            'Edit cell cycle annotations...', self
        )
        self.manuallyEditCcaAction.setShortcut('Ctrl+Shift+P')
        self.manuallyEditCcaAction.setDisabled(True)

        self.viewCcaTableAction = QAction(
            'View cell cycle annotations...', self
        )
        self.viewCcaTableAction.setDisabled(True)
        self.viewCcaTableAction.setShortcut('Ctrl+P')

        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        checked = self.df_settings.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

        self.shuffleCmapAction =  QAction('Shuffle colormap...', self)
        self.shuffleCmapAction.setShortcut('Shift+S')

        self.normalizeRawAction = QAction(
            'Do not normalize. Display raw image', self)
        self.normalizeToFloatAction = QAction(
            'Convert to floating point format with values [0, 1]', self)
        # self.normalizeToUbyteAction = QAction(
        #     'Rescale to 8-bit unsigned integer format with values [0, 255]', self)
        self.normalizeRescale0to1Action = QAction(
            'Rescale to [0, 1]', self)
        self.normalizeByMaxAction = QAction(
            'Normalize by max value', self)
        self.normalizeRawAction.setCheckable(True)
        self.normalizeToFloatAction.setCheckable(True)
        # self.normalizeToUbyteAction.setCheckable(True)
        self.normalizeRescale0to1Action.setCheckable(True)
        self.normalizeByMaxAction.setCheckable(True)
        self.normalizeQActionGroup = QActionGroup(self)
        self.normalizeQActionGroup.addAction(self.normalizeRawAction)
        self.normalizeQActionGroup.addAction(self.normalizeToFloatAction)
        # self.normalizeQActionGroup.addAction(self.normalizeToUbyteAction)
        self.normalizeQActionGroup.addAction(self.normalizeRescale0to1Action)
        self.normalizeQActionGroup.addAction(self.normalizeByMaxAction)

        self.zoomToObjsAction = QAction(
            'Zoom to objects  (shortcut: H key)', self
        )
        self.zoomOutAction = QAction(
            'Zoom out  (shortcut: double press H key)', self
        )

        self.relabelSequentialAction = QAction(
            'Relabel IDs sequentially...', self
        )
        self.relabelSequentialAction.setShortcut('Ctrl+L')
        self.relabelSequentialAction.setDisabled(True)

        self.setLastUserNormAction()

        self.autoSegmAction = QAction(
            'Enable automatic segmentation', self)
        self.autoSegmAction.setCheckable(True)
        self.autoSegmAction.setDisabled(True)

        self.enableSmartTrackAction = QAction(
            'Smart handling of enabling/disabling tracking', self)
        self.enableSmartTrackAction.setCheckable(True)
        self.enableSmartTrackAction.setChecked(True)

        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self)
        self.enableAutoZoomToCellsAction.setCheckable(True)

        self.gaussBlurAction = QAction('Gaussian blur...', self)
        self.gaussBlurAction.setCheckable(True)

        self.diffGaussFilterAction = QAction('Sharpen...', self)
        self.diffGaussFilterAction.setCheckable(True)

        self.imgPropertiesAction = QAction('Properties...', self)
        self.imgPropertiesAction.setDisabled(True)

        self.edgeDetectorAction = QAction('Edge detection...', self)
        self.edgeDetectorAction.setCheckable(True)

        self.entropyFilterAction = QAction('Object detection...', self)
        self.entropyFilterAction.setCheckable(True)

        self.addDelRoiAction = QAction(self)
        self.addDelRoiAction.setIcon(QIcon(":addDelRoi.svg"))
        self.addDelRoiAction.setToolTip(
            'Add resizable rectangle. Every ID touched by the rectangle will be '
            'automaticaly deleted.\n '
            'Moving the rectangle will restore deleted IDs if they are not '
            'touched by it anymore.\n'
            'To delete rectangle right-click on it --> remove.')

        self.delBorderObjAction = QAction(self)
        self.delBorderObjAction.setIcon(QIcon(":delBorderObj.svg"))
        self.delBorderObjAction.setToolTip(
            'Remove segmented objects touching the border of the image'
        )

        self.addCustomAnnotationAction = QAction(self)
        self.addCustomAnnotationAction.setIcon(QIcon(":annotate.svg"))
        self.addCustomAnnotationAction.setToolTip('Add custom annotation')

        self.viewAllCustomAnnotAction = QAction(self)
        self.viewAllCustomAnnotAction.setCheckable(True)
        self.viewAllCustomAnnotAction.setIcon(QIcon(":eye.svg"))
        self.viewAllCustomAnnotAction.setToolTip('Show all custom annotations')

        # self.imgGradLabelsAlphaUpAction = QAction(self)
        # self.imgGradLabelsAlphaUpAction.setVisible(False)
        # self.imgGradLabelsAlphaUpAction.setShortcut('Ctrl+Up')

    def gui_connectActions(self):
        # Connect File actions
        self.newAction.triggered.connect(self.newFile)
        self.openAction.triggered.connect(self.openFolder)
        self.openFileAction.triggered.connect(self.openFile)
        self.saveAction.triggered.connect(self.saveData)
        self.saveAsAction.triggered.connect(self.saveAsData)
        self.quickSaveAction.triggered.connect(self.quickSave)
        self.showInExplorerAction.triggered.connect(self.showInExplorer_cb)
        self.exitAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)

        # Connect Help actions
        self.tipsAction.triggered.connect(self.showTipsAndTricks)
        self.UserManualAction.triggered.connect(myutils.showUserManual)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

        self.showPropsDockButton.clicked.connect(self.showPropsDockWidget)

        self.addCustomAnnotationAction.triggered.connect(
            self.addCustomAnnotation
        )
        self.viewAllCustomAnnotAction.toggled.connect(
            self.viewAllCustomAnnot
        )
        self.addCustomModelAction.triggered.connect(
            self.showInstructionsCustomModel
        )

    def gui_connectEditActions(self):
        self.showInExplorerAction.setEnabled(True)
        self.setEnabledFileToolbar(True)
        self.loadFluoAction.setEnabled(True)
        self.isEditActionsConnected = True

        self.fontSizeMenu.triggered.connect(self.changeFontSize)
        self.prevAction.triggered.connect(self.prev_cb)
        self.nextAction.triggered.connect(self.next_cb)
        self.overlayButton.toggled.connect(self.overlay_cb)
        self.overlayButton.sigRightClick.connect(self.showOverlayContextMenu)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.loadFluoAction.triggered.connect(self.loadFluo_cb)
        # self.reloadAction.triggered.connect(self.reload_cb)
        self.findIdAction.triggered.connect(self.findID)
        self.slideshowButton.toggled.connect(self.launchSlideshow)

        self.segmSingleFrameMenu.triggered.connect(self.segmFrameCallback)
        self.segmVideoMenu.triggered.connect(self.segmVideoCallback)

        self.SegmActionRW.triggered.connect(self.randomWalkerSegm)
        self.postProcessSegmAction.toggled.connect(self.postProcessSegm)
        self.autoSegmAction.toggled.connect(self.autoSegm_cb)
        self.disableTrackingCheckBox.clicked.connect(self.disableTracking)
        self.repeatTrackingAction.triggered.connect(self.repeatTracking)
        self.repeatTrackingMenuAction.triggered.connect(self.repeatTracking)
        self.repeatTrackingVideoAction.triggered.connect(
            self.repeatTrackingVideo
        )
        self.trackWithAcdcAction.toggled.connect(self.storeTrackingAlgo)
        self.trackWithYeazAction.toggled.connect(self.storeTrackingAlgo)


        self.brushButton.toggled.connect(self.Brush_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.wandToolButton.toggled.connect(self.wand_cb)
        self.reInitCcaAction.triggered.connect(self.reInitCca)
        self.assignBudMothAutoAction.triggered.connect(
            self.autoAssignBud_YeastMate
        )

        self.expandLabelToolButton.toggled.connect(self.expandLabelCallback)

        # self.repeatAutoCcaAction.triggered.connect(self.repeatAutoCca)
        self.manuallyEditCcaAction.triggered.connect(self.manualEditCca)
        self.invertBwAction.toggled.connect(self.invertBw)

        self.enableSmartTrackAction.toggled.connect(self.enableSmartTrack)
        # Brush/Eraser size action
        self.brushSizeSpinbox.valueChanged.connect(self.brushSize_cb)
        # Mode
        self.modeComboBox.currentIndexChanged.connect(self.changeMode)
        self.modeComboBox.activated.connect(self.clearComboBoxFocus)
        self.equalizeHistPushButton.clicked.connect(self.equalizeHist)
        self.editOverlayColorAction.triggered.connect(self.toggleOverlayColorButton)
        self.editTextIDsColorAction.triggered.connect(self.toggleTextIDsColorButton)
        self.overlayColorButton.sigColorChanging.connect(self.updateOlColors)
        self.textIDsColorButton.sigColorChanging.connect(self.updateTextIDsColors)
        self.textIDsColorButton.sigColorChanged.connect(self.saveTextIDsColors)
        self.alphaScrollBar.valueChanged.connect(self.updateOverlay)

        self.setMeasurementsAction.triggered.connect(self.showSetMeasurements)
        self.addCustomMetricAction.triggered.connect(self.addCustomMetric)
        self.addCombineMetricAction.triggered.connect(self.addCombineMetric)

        self.labelsGrad.colorButton.sigColorChanging.connect(self.updateBkgrColor)
        self.labelsGrad.colorButton.sigColorChanged.connect(self.saveBkgrColor)
        self.labelsGrad.sigGradientChangeFinished.connect(self.updateLabelsCmap)
        self.labelsGrad.sigGradientChanged.connect(self.ticksCmapMoved)
        self.labelsGrad.textColorButton.sigColorChanging.connect(
            self.updateTextLabelsColor
        )
        self.labelsGrad.textColorButton.sigColorChanged.connect(
            self.saveTextLabelsColor
        )
        self.labelsGrad.editFontSizeAction.triggered.connect(self.editFontSize)

        self.labelsGrad.shuffleCmapAction.triggered.connect(self.shuffle_cmap)
        self.shuffleCmapAction.triggered.connect(self.shuffle_cmap)
        self.labelsGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.labelsGrad.hideLabelsImgAction.toggled.connect(self.hideLabels)
        self.labelsGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )

        self.imgGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.imgGrad.labelsAlphaSlider.valueChanged.connect(
            self.updateLabelsAlpha
        )
        self.imgGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )

        # Drawing mode
        self.drawIDsContComboBox.currentIndexChanged.connect(
                                                self.drawIDsContComboBox_cb)
        self.drawIDsContComboBox.activated.connect(self.clearComboBoxFocus)
        self.gaussBlurAction.toggled.connect(self.gaussBlur)
        self.diffGaussFilterAction.toggled.connect(self.diffGaussCallback)
        self.edgeDetectorAction.toggled.connect(self.edgeDetection)
        self.entropyFilterAction.toggled.connect(self.entropyFilter)
        self.addDelRoiAction.triggered.connect(self.addDelROI)
        self.delBorderObjAction.triggered.connect(self.delBorderObj)

        self.imgGrad.sigLookupTableChanged.connect(self.imgGradLUT_cb)
        self.imgGrad.gradient.sigGradientChangeFinished.connect(
            self.imgGradLUTfinished_cb
        )


        self.normalizeQActionGroup.triggered.connect(self.saveNormAction)
        self.imgPropertiesAction.triggered.connect(self.editImgProperties)

        self.guiTabControl.propsQGBox.idSB.valueChanged.connect(
            self.updatePropsWidget
        )
        self.guiTabControl.highlightCheckbox.toggled.connect(
            self.highlightIDcheckBoxToggled
        )
        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        intensMeasurQGBox.additionalMeasCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )
        intensMeasurQGBox.channelCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )

        self.relabelSequentialAction.triggered.connect(
            self.relabelSequentialCallback
        )

        self.zoomToObjsAction.triggered.connect(self.zoomToObjsActionCallback)
        self.zoomOutAction.triggered.connect(self.zoomOut)

        self.viewCcaTableAction.triggered.connect(self.viewCcaTable)

    def gui_createLeftSideWidgets(self):
        self.leftSideDocksLayout = QVBoxLayout()
        self.showPropsDockButton = widgets.expandCollapseButton()
        self.showPropsDockButton.setDisabled(True)
        self.showPropsDockButton.setFocusPolicy(Qt.NoFocus)
        self.showPropsDockButton.setToolTip('Show object properties')
        self.leftSideDocksLayout.addWidget(self.showPropsDockButton)
        self.leftSideDocksLayout.setSpacing(0)
        self.leftSideDocksLayout.setContentsMargins(0,0,0,0)

    def gui_createImg1Widgets(self):
        _font = QtGui.QFont()
        # _font.setPixelSize(13)
        _font.setPixelSize(12)

        # Toggle contours/ID comboboxf
        self.drawIDsContComboBoxSegmItems = [
            'Draw IDs and contours',
            'Draw IDs and overlay segm. masks',
            'Draw only cell cycle info',
            'Draw cell cycle info and contours',
            'Draw cell cycle info and overlay segm. masks',
            'Draw only mother-bud lines',
            'Draw only IDs',
            'Draw only contours',
            'Draw only overlay segm. masks',
            'Draw nothing'
        ]
        # self.drawIDsContComboBoxCcaItems = [
        #     'Draw IDs and contours',
        #     'Draw IDs and overlay segm. masks',
        #     'Draw only cell cycle info',
        #     'Draw cell cycle info and contours',
        #     'Draw cell cycle info and overlay segm. masks',
        #     'Draw only mother-bud lines',
        #     'Draw only IDs',
        #     'Draw only contours',
        #     'Draw only overlay segm. masks',
        #     'Draw nothing'
        # ]
        self.drawIDsContComboBox = QComboBox()
        self.drawIDsContComboBox.setFont(_font)
        self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
        # Always adjust combobox width to largest item
        self.drawIDsContComboBox.setSizeAdjustPolicy(
                    self.drawIDsContComboBox.AdjustToContents)

        # Frames scrollbar
        self.navigateScrollBar = widgets.navigateQScrollBar(Qt.Horizontal)
        self.navigateScrollBar.setDisabled(True)
        self.navigateScrollBar.setMinimum(1)
        self.navigateScrollBar.setMaximum(1)
        self.navigateScrollBar.setToolTip(
            'NOTE: The maximum frame number that can be visualized with this '
            'scrollbar\n'
            'is the last visited frame with the selected mode\n'
            '(see "Mode" selector on the top-right).\n\n'
            'If the scrollbar does not move it means that you never visited\n'
            'any frame with current mode.\n\n'
            'Note that the "Viewer" mode allows you to scroll ALL frames.'
        )
        t_label = QLabel('frame n.  ')
        t_label.setFont(_font)
        self.t_label = t_label

        # z-slice scrollbars
        self.zSliceScrollBar = widgets.linkedQScrollbar(Qt.Horizontal)
        _z_label = QLabel('z-slice  ')
        _z_label.setFont(_font)
        self.z_label = _z_label

        self.zProjComboBox = QComboBox()
        self.zProjComboBox.setFont(_font)
        self.zProjComboBox.addItems([
            'single z-slice',
            'max z-projection',
            'mean z-projection',
            'median z-proj.'
        ])

        self.zSliceOverlay_SB = QScrollBar(Qt.Horizontal)
        _z_label = QLabel('overlay z-slice  ')
        _z_label.setFont(_font)
        self.overlay_z_label = _z_label

        self.zProjOverlay_CB = QComboBox()
        self.zProjOverlay_CB.setFont(_font)
        self.zProjOverlay_CB.addItems(['single z-slice',
                                       'max z-projection',
                                       'mean z-projection',
                                       'median z-proj.',
                                       'same as above'])
        self.zProjOverlay_CB.setCurrentIndex(1)
        self.zSliceOverlay_SB.setDisabled(True)

        # Fluorescent overlay alpha
        alphaScrollBar_label = QLabel('Overlay alpha  ')
        alphaScrollBar_label.setFont(_font)
        alphaScrollBar = QScrollBar(Qt.Horizontal)

        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(20)
        alphaScrollBar.setToolTip(
            'Control the alpha value of the overlay.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescent data visible'
        )
        alphaScrollBar.setDisabled(True)
        self.alphaScrollBar = alphaScrollBar
        self.alphaScrollBar_label = alphaScrollBar_label
        self.img1BottomGroupbox = self.gui_addImg1BottomWidgets()

    def gui_addImg1BottomWidgets(self):
        bottomLeftLayout = QGridLayout()
        container = QGroupBox('Left image controls')

        row = 0
        bottomLeftLayout.addWidget(
            self.drawIDsContComboBox, row, 1, alignment=Qt.AlignCenter
        )

        row += 1
        bottomLeftLayout.addWidget(self.t_label, row, 0, alignment=Qt.AlignRight)
        bottomLeftLayout.addWidget(self.navigateScrollBar, row, 1)
        sp = self.navigateScrollBar.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.navigateScrollBar.setSizePolicy(sp)

        row += 1
        bottomLeftLayout.addWidget(
            self.z_label, row, 0, alignment=Qt.AlignRight
        )
        bottomLeftLayout.addWidget(self.zSliceScrollBar, row, 1)
        bottomLeftLayout.addWidget(self.zProjComboBox, row, 2)

        row += 1
        bottomLeftLayout.addWidget(
            self.overlay_z_label, row, 0, alignment=Qt.AlignRight
        )
        bottomLeftLayout.addWidget(self.zSliceOverlay_SB, row, 1)

        bottomLeftLayout.addWidget(self.zProjOverlay_CB, row, 2)

        row += 1
        bottomLeftLayout.addWidget(
            self.alphaScrollBar_label, row, 0, alignment=Qt.AlignRight
        )
        bottomLeftLayout.addWidget(self.alphaScrollBar, row, 1)
        sp = self.alphaScrollBar_label.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.alphaScrollBar_label.setSizePolicy(sp)
        sp = self.alphaScrollBar.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.alphaScrollBar.setSizePolicy(sp)

        bottomLeftLayout.setColumnStretch(0,0)
        bottomLeftLayout.setColumnStretch(1,3)
        bottomLeftLayout.setColumnStretch(2,0)

        container.setLayout(bottomLeftLayout)
        return container

    def gui_createLabWidgets(self):
        bottomRightLayout = QGridLayout()
        self.labBottomGroupbox = QGroupBox('Segmentation image controls')
        self.labBottomGroupbox.setCheckable(True)
        self.labBottomGroupbox.setChecked(False)

        font = QtGui.QFont()
        font.setPixelSize(12)

        row = 0
        self.z_label_lab = QLabel('z-slice  ')
        bottomRightLayout.addWidget(
            self.z_label_lab, row, 0, alignment=Qt.AlignRight
        )
        self.zSliceScrollBarLab = widgets.labelledQScrollbar(Qt.Horizontal)
        self.zSliceScrollBarLab.moving = False
        self.zSliceScrollBarLab.setLabel(self.z_label_lab)
        self.zSliceScrollBar.linkScrollBar(self.zSliceScrollBarLab)
        bottomRightLayout.addWidget(self.zSliceScrollBarLab, row, 1)

        bottomRightLayout.setColumnStretch(0,0)
        bottomRightLayout.setColumnStretch(1,3)
        # bottomRightLayout.setColumnStretch(2,0)

        self.labBottomGroupbox.setLayout(bottomRightLayout)

        self.labBottomGroupbox.toggled.connect(self.labControlsToggled)

    def labControlsToggled(self, checked):
        if checked:
            self.zSliceScrollBar.unlinkScrollBar()
            self.connectZSliceScrollBarLab()
        else:
            self.zSliceScrollBar.linkScrollBar(self.zSliceScrollBarLab)
            self.connectZSliceScrollBarLab(disconnect=True)

    def connectZSliceScrollBarLab(self, disconnect=False):
        if disconnect:
            self.zSliceScrollBarLab.actionTriggered.disconnect()
            self.zSliceScrollBarLab.sliderReleased.disconnect()
        else:
            self.zSliceScrollBarLab.actionTriggered.connect(
                self.updateZsliceLab
            )
            self.zSliceScrollBarLab.sliderReleased.connect(
                self.zSliceScrollBarLabReleased
            )

    def gui_addBottomWidgetsToBottomLayout(self):
        self.bottomLayout = QHBoxLayout()
        self.bottomLayout.addStretch(1)
        self.bottomLayout.addWidget(self.img1BottomGroupbox)
        self.bottomLayout.addStretch(1)
        self.bottomLayout.addWidget(self.labBottomGroupbox)
        self.bottomLayout.addStretch(1)
        self.setBottomLayoutStretch()

    def gui_createGraphicsPlots(self):
        self.graphLayout = pg.GraphicsLayoutWidget()
        if self.invertBwAction.isChecked():
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.titleColor = 'k'
        else:
            self.titleColor = 'w'

        # Left plot
        self.ax1 = pg.PlotItem()
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=1, col=1)

        # Right plot
        self.ax2 = pg.PlotItem()
        self.ax2.setAspectLocked(True)
        self.ax2.invertY(True)
        self.ax2.hideAxis('bottom')
        self.ax2.hideAxis('left')
        self.graphLayout.addItem(self.ax2, row=1, col=2)

    def gui_addGraphicsItems(self):
        # Auto image adjustment button
        proxy = QtGui.QGraphicsProxyWidget()
        equalizeHistPushButton = QPushButton("Auto")
        if not self.invertBwAction.isChecked():
            equalizeHistPushButton.setStyleSheet(
                'QPushButton {background-color: #282828; color: #F0F0F0;}'
            )
        self.equalizeHistPushButton = equalizeHistPushButton
        proxy.setWidget(equalizeHistPushButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        self.equalizeHistPushButton = equalizeHistPushButton

        # Left image histogram
        self.imgGrad = widgets.myHistogramLUTitem()
        # Disable histogram default context Menu event
        self.imgGrad.vb.raiseContextMenu = lambda x: None
        self.imgGrad.restoreState(self.df_settings)
        self.graphLayout.addItem(self.imgGrad, row=1, col=0)

        # Colormap gradient widget
        self.labelsGrad = widgets.labelsGradientWidget()
        try:
            stateFound = self.labelsGrad.restoreState(self.df_settings)
        except Exception as e:
            self.logger.exception(traceback.format_exc())
            print('======================================')
            self.logger.info(
                'Failed to restore previously used colormap. '
                'Using default colormap "viridis"'
            )
            self.labelsGrad.item.loadPreset('viridis')

        # Title
        self.titleLabel = pg.LabelItem(
            justify='center', color=self.titleColor, size='14pt'
        )
        self.titleLabel.setText(
            'Drag and drop image file or go to File --> Open folder...')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)

        # # Current frame text
        # self.frameLabel = pg.LabelItem(justify='center', color=self.titleColor, size='14pt')
        # self.frameLabel.setText(' ')
        # self.graphLayout.addItem(self.frameLabel, row=2, col=1, colspan=2)

    def gui_setImg1TextColors(self, r, g, b, custom=False):
        if custom:
            self.ax1_oldIDcolor = (r, g, b)
            self.ax1_S_oldCellColor = (int(r*0.9), int(r*0.9), int(b*0.9))
            self.ax1_G1cellColor = (int(r*0.8), int(g*0.8), int(b*0.8), 178)
        else:
            self.ax1_oldIDcolor = (255, 255, 255) # white
            self.ax1_S_oldCellColor = (229, 229, 229)
            self.ax1_G1cellColor = (204, 204, 204, 178)
        self.ax1_divAnnotColor = (245, 188, 1) # orange

    def gui_addPlotItems(self):
        if 'textIDsColor' in self.df_settings.index:
            rgbString = self.df_settings.at['textIDsColor', 'value']
            r, g, b = myutils.rgb_str_to_values(rgbString)
            self.gui_setImg1TextColors(r, g, b, custom=True)
            self.textIDsColorButton.setColor((r, g, b))
        else:
            self.gui_setImg1TextColors(0,0,0, custom=False)

        if 'labels_text_color' in self.df_settings.index:
            rgbString = self.df_settings.at['labels_text_color', 'value']
            r, g, b = myutils.rgb_str_to_values(rgbString)
            self.ax2_textColor = (r, g, b)
        else:
            self.ax2_textColor = (255, 0, 0)

        # Blank image
        self.blank = np.zeros((256,256), np.uint8)
        if self.invertBwAction.isChecked():
            self.blank = self.blank + graphLayoutBkgrColor[0]

        # Left image
        self.img1 = pg.ImageItem(self.blank)
        self.ax1.addItem(self.img1)
        # Disconnect colormaps actions and connect to gradientCmapContextMenuClicked
        for action in self.imgGrad.gradient.menu.actions():
            try:
                action.name
                action.triggered.disconnect()
                action.triggered.connect(self.gradientCmapContextMenuClicked)
            except (AttributeError, TypeError):
                pass

        # Right image
        self.img2 = widgets.labImageItem(self.blank)
        self.ax2.addItem(self.img2)

        # Brush circle img1
        self.ax1_BrushCircle = pg.ScatterPlotItem()
        self.ax1_BrushCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=pg.mkBrush((255,255,255,50)),
                                 pen=pg.mkPen(width=2))
        self.ax1.addItem(self.ax1_BrushCircle)

        # Eraser circle img2
        self.ax2_EraserCircle = pg.ScatterPlotItem()
        self.eraserCirclePen = pg.mkPen(width=1.5, color='r')
        self.ax2_EraserCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=None,
                                 pen=self.eraserCirclePen)
        self.ax2.addItem(self.ax2_EraserCircle)
        self.ax2_EraserX = pg.ScatterPlotItem()
        self.ax2_EraserX.setData([], [], symbol='x', pxMode=False, size=3,
                                      brush=pg.mkBrush(color=(255,0,0,50)),
                                      pen=pg.mkPen(width=1.5, color='r'))
        self.ax2.addItem(self.ax2_EraserX)

        # Eraser circle img1
        self.ax1_EraserCircle = pg.ScatterPlotItem()
        self.ax1_EraserCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=None,
                                 pen=self.eraserCirclePen)
        self.ax1.addItem(self.ax1_EraserCircle)
        self.ax1_EraserX = pg.ScatterPlotItem()
        self.ax1_EraserX.setData([], [], symbol='x', pxMode=False, size=3,
                                      brush=pg.mkBrush(color=(255,0,0,50)),
                                      pen=pg.mkPen(width=1, color='r'))
        self.ax1.addItem(self.ax1_EraserX)

        # Brush circle img2
        self.ax2_BrushCirclePen = pg.mkPen(width=2)
        self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
        self.ax2_BrushCircle = pg.ScatterPlotItem()
        self.ax2_BrushCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=self.ax2_BrushCircleBrush,
                                 pen=self.ax2_BrushCirclePen)
        self.ax2.addItem(self.ax2_BrushCircle)

        # Random walker markers colors
        self.RWbkgrColor = (255,255,0)
        self.RWforegrColor = (124,5,161)


        # # Experimental: brush cursors
        # self.eraserCursor = QCursor(QIcon(":eraser.png").pixmap(30, 30))
        # brushCursorPixmap = QIcon(":brush-cursor.png").pixmap(32, 32)
        # self.brushCursor = QCursor(brushCursorPixmap, 16, 16)

        # Annotated metadata markers (ScatterPlotItem)
        self.ax2_binnedIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax2_binnedIDs_ScatterPlot.setData(
                                 [], [], symbol='t', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=3, color='r'))
        self.ax2.addItem(self.ax2_binnedIDs_ScatterPlot)
        self.ax2_ripIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax2_ripIDs_ScatterPlot.setData(
                                 [], [], symbol='x', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=2, color='r'))
        self.ax2.addItem(self.ax2_ripIDs_ScatterPlot)

        self.ax1_binnedIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax1_binnedIDs_ScatterPlot.setData(
                                 [], [], symbol='t', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=3, color='r'))
        self.ax1.addItem(self.ax1_binnedIDs_ScatterPlot)
        self.ax1_ripIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax1_ripIDs_ScatterPlot.setData(
                                 [], [], symbol='x', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=2, color='r'))
        self.ax1.addItem(self.ax1_ripIDs_ScatterPlot)

        # Ruler plotItem and scatterItem
        rulerPen = pg.mkPen(color='r', style=Qt.DashLine, width=2)
        self.ax1_rulerPlotItem = pg.PlotDataItem(pen=rulerPen)
        self.ax1_rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2)
        )
        self.ax1.addItem(self.ax1_rulerPlotItem)
        self.ax1.addItem(self.ax1_rulerAnchorsItem)

        # Experimental: scatter plot to add a point marker
        self.ax1_point_ScatterPlot = pg.ScatterPlotItem()
        self.ax1_point_ScatterPlot.setData(
            [], [], symbol='o', pxMode=False, size=3,
            pen=pg.mkPen(width=2, color='r'),
            brush=pg.mkBrush((255,0,0,50))
        )
        self.ax1.addItem(self.ax1_point_ScatterPlot)

    def _warn_too_many_items(self, numItems):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            You loaded a segmentation mask that has <b>{numItems} objects</b>.<br><br>
            Creating graphical items for this many objects could take
            a <b>long time</b>.<br><br>
            We <b>recommend disabling graphical items</b>. The graphical items
            <b>will be created "on-demand"</b> when you visualize a new time-point,
            resulting in slighlty higher loading time of a new time-point.<br><br>
            <i>Note that the ID of the object will still be displayed on the
            bottom-right corner when you hover on it with the mouse.</i><br><br>
            What do you want to do?
        """)
        _, doNotCreateItemsButton, _ = msg.warning(
            self, 'Too many objects', txt,
            buttonsTexts=(
                'Cancel', ' Ok, create on-demand ',
                'Try anyway'
            )
        )
        return msg.cancel, msg.clickedButton==doNotCreateItemsButton

    def gui_createIDsAxesItems(self):
        allIDs = set()
        self.logger.info('Counting total number of segmented objects...')
        for lab in tqdm(self.data[self.pos_i].segm_data, ncols=100):
            IDs = [obj.label for obj in skimage.measure.regionprops(lab)]
            allIDs.update(IDs)

        numItems = (self.data[self.pos_i].segm_data).max()
        if numItems == 0:
            numItems = 100
            # Pre-generate 100 items to make first 100 new objects faster
            allIDs = range(1,numItems+1)

        self.ax1_ContoursCurves = [None]*numItems
        self.ax2_ContoursCurves = [None]*numItems
        self.ax1_BudMothLines = [None]*numItems
        self.ax1_LabelItemsIDs = [None]*numItems
        self.ax2_LabelItemsIDs = [None]*numItems

        if numItems > 500:
            cancel, doNotCreateItems = self._warn_too_many_items(numItems)
            if cancel:
                self.progressWin.workerFinished = True
                self.progressWin.close()
                return
            elif doNotCreateItems:
                self.logger.info(f'Graphical items creation aborted.')
                self.progressWin.workerFinished = True
                self.progressWin.close()
                drawModes = self.drawIDsContComboBoxSegmItems
                # self.drawIDsContComboBoxSegmItems = drawModes
                # self.drawIDsContComboBox.addItems(drawModes)
                df = self.df_settings
                df.at['how_draw_annotations', 'value'] = drawModes[-2]
                self.drawIDsContComboBox.setCurrentText(drawModes[-2])
                self.loadingDataCompleted()
                return

        self.logger.info(f'Creating {len(allIDs)} axes items...')
        for ID in tqdm(allIDs, ncols=100):
            self.ax1_ContoursCurves[ID-1] = pg.PlotDataItem()
            self.ax1_BudMothLines[ID-1] = pg.PlotDataItem()
            self.ax1_LabelItemsIDs[ID-1] = widgets.myLabelItem()
            self.ax2_LabelItemsIDs[ID-1] = widgets.myLabelItem()
            self.ax2_ContoursCurves[ID-1] = pg.PlotDataItem()

        self.progressWin.mainPbar.setMaximum(0)
        self.gui_addCreatedAxesItems()
        self.progressWin.workerFinished = True
        self.progressWin.close()

        self.loadingDataCompleted()

    def gui_createContourPens(self):
        if 'contLineWeight' in self.df_settings.index:
            val = self.df_settings.at['contLineWeight', 'value']
            self.contLineWeight = int(val)
        else:
            self.contLineWeight = 2
        if 'contLineColor' in self.df_settings.index:
            val = self.df_settings.at['contLineColor', 'value']
            rgba = myutils.rgba_str_to_values(val)
            self.contLineColor = [max(0, v-50) for v in rgba]
            self.newIDlineColor = [min(255, v+50) for v in self.contLineColor]
        else:
            self.contLineColor = (205, 0, 0, 220)
            self.newIDlineColor = (255, 0, 0, 255)

        try:
            self.imgGrad.contoursColorButton.sigColorChanging.disconnect()
            self.imgGrad.contoursColorButton.sigColorChanged.disconnect()
        except Exception as e:
            pass
        try:
            for act in self.imgGrad.contLineWightActionGroup.actions():
                act.toggled.disconnect()
        except Exception as e:
            pass
        for act in self.imgGrad.contLineWightActionGroup.actions():
            if act.lineWeight == self.contLineWeight:
                act.setChecked(True)
        self.imgGrad.contoursColorButton.setColor(self.contLineColor[:3])

        self.imgGrad.contoursColorButton.sigColorChanging.connect(
            self.updateContColour
        )
        self.imgGrad.contoursColorButton.sigColorChanged.connect(
            self.saveContColour
        )
        for act in self.imgGrad.contLineWightActionGroup.actions():
            act.toggled.connect(self.contLineWeightToggled)

        # Contours pens
        self.oldIDs_cpen = pg.mkPen(
            color=self.contLineColor, width=self.contLineWeight
        )
        self.newIDs_cpen = pg.mkPen(
            color=self.newIDlineColor, width=self.contLineWeight+1
        )
        self.tempNewIDs_cpen = pg.mkPen(
            color='g', width=self.contLineWeight+1
        )
        self.lostIDs_cpen = pg.mkPen(
            color=(245, 184, 0, 100), width=self.contLineWeight+2
        )

    def gui_createGraphicsItems(self):
        self.gui_createContourPens()

        # Lost ID question mark text color
        self.lostIDs_qMcolor = (245, 184, 0)

        # New bud-mother line pen
        self.NewBudMoth_Pen = pg.mkPen(
            color='r', width=3, style=Qt.DashLine
        )

        # Old bud-mother line pen
        self.OldBudMoth_Pen = pg.mkPen(
            color=(255,165,0), width=2, style=Qt.DashLine
        )

        # Temporary line item connecting bud to new mother
        self.BudMothTempLine = pg.PlotDataItem(pen=self.NewBudMoth_Pen)
        self.ax1.addItem(self.BudMothTempLine)

        # Red/green border rect item
        self.GreenLinePen = pg.mkPen(color='g', width=2)
        self.RedLinePen = pg.mkPen(color='r', width=2)
        self.ax1BorderLine = pg.PlotDataItem()
        self.ax1.addItem(self.ax1BorderLine)
        self.ax2BorderLine = pg.PlotDataItem(pen=pg.mkPen(color='r', width=2))
        self.ax2.addItem(self.ax2BorderLine)

        # Create enough PlotDataItems and LabelItems to draw contours and IDs.
        self.progressWin = apps.QDialogWorkerProgress(
            title='Creating axes items', parent=self,
            pbarDesc='Creating axes items (see progress in the terminal)...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        QTimer.singleShot(50, self.gui_createIDsAxesItems)

    def gui_connectGraphicsEvents(self):
        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img2.hoverEvent = self.gui_hoverEventImg2
        self.img1.mousePressEvent = self.gui_mousePressEventImg1
        self.img1.mouseMoveEvent = self.gui_mouseDragEventImg1
        self.img1.mouseReleaseEvent = self.gui_mouseReleaseEventImg1
        self.img2.mousePressEvent = self.gui_mousePressEventImg2
        self.img2.mouseMoveEvent = self.gui_mouseDragEventImg2
        self.img2.mouseReleaseEvent = self.gui_mouseReleaseEventImg2
        self.imgGrad.gradient.showMenu = self.gui_gradientContextMenuEvent
        # self.imgGrad.vb.contextMenuEvent = self.gui_gradientContextMenuEvent

    def gui_initImg1BottomWidgets(self):
        self.zSliceScrollBar.hide()
        self.zProjComboBox.hide()
        self.z_label.hide()
        self.zSliceOverlay_SB.hide()
        self.zProjOverlay_CB.hide()
        self.overlay_z_label.hide()
        self.alphaScrollBar.hide()
        self.alphaScrollBar_label.hide()

    @myutils.exception_handler
    def gui_mousePressEventImg2(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        isMod = ctrl or alt
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        left_click = event.button() == Qt.MouseButton.LeftButton and not isMod
        middle_click = self.isMiddleClick(event, modifiers)
        right_click = event.button() == Qt.MouseButton.RightButton and not isMod
        isPanImageClick = self.isPanImageClick(event, modifiers)
        eraserON = self.eraserButton.isChecked()
        brushON = self.brushButton.isChecked()
        separateON = self.separateBudButton.isChecked()

        # Drag image if neither brush or eraser are On pressed
        dragImg = (
            left_click and not eraserON and not
            brushON and not separateON
        )
        if isPanImageClick:
            dragImg = True

        # Enable dragging of the image window like pyqtgraph original code
        if dragImg:
            pg.ImageItem.mousePressEvent(self.img2, event)
            event.ignore()
            return

        if mode == 'Viewer' and middle_click:
            self.startBlinkingModeCB()
            event.ignore()
            return

        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        Y, X = self.get_2Dlab(posData.lab).shape
        if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        else:
            return

        # Check if right click on ROI
        isClickOnDelRoi = self.gui_clickedDelRoi(event, left_click, right_click)
        if isClickOnDelRoi:
            return

        # show gradient widget menu if none of the right-click actions are ON
        # and event is not coming from image 1
        is_right_click_action_ON = any([
            b.isChecked() for b in self.checkableQButtonsGroup.buttons()
        ])
        is_right_click_custom_ON = any([
            b.isChecked() for b in self.customAnnotDict.keys()
        ])
        is_event_from_img1 = False
        if hasattr(event, 'isImg1Sender'):
            is_event_from_img1 = event.isImg1Sender
        showLabelsGradMenu = (
            right_click and not is_right_click_action_ON
            and not is_event_from_img1
        )
        if showLabelsGradMenu:
            self.labelsGrad.showMenu(event)
            event.ignore()
            return

        editInViewerMode = (
            (is_right_click_action_ON or is_right_click_custom_ON)
            and (right_click or middle_click) and mode=='Viewer'
        )

        if editInViewerMode:
            self.startBlinkingModeCB()
            event.ignore()
            return

        # Left-click is used for brush, eraser, separate bud and curvature tool
        # Brush and eraser are mutually exclusive but we want to keep the eraser
        # or brush ON and disable them temporarily to allow left-click with
        # separate ON
        canErase = eraserON and not separateON and not dragImg
        canBrush = brushON and not separateON and not dragImg
        canDelete = mode == 'Segmentation and Tracking' or self.isSnapshot

        # Erase with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        if left_click and canErase:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = self.get_2Dlab(posData.lab).shape
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.yPressAx2, self.xPressAx2 = y, x
            # Keep a global mask to compute which IDs got erased
            self.erasedIDs = []
            lab_2D = self.get_2Dlab(posData.lab)
            self.erasedID = self.getHoverID(xdata, ydata)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True

            # If user double-pressed 'b' then erase over ALL labels
            color = self.eraserButton.palette().button().color().name()
            eraseOnlyOneID = (
                color != self.doublePressKeyButtonColor
                and self.erasedID != 0
            )
            if eraseOnlyOneID:
                mask[lab_2D!=self.erasedID] = False

            self.eraseOnlyOneID = eraseOnlyOneID

            self.erasedIDs.extend(lab_2D[mask])
            self.applyEraserMask(mask)
            self.setImageImg2()

            self.isMouseDragImg2 = True

        # Paint with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        elif left_click and canBrush:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.yPressAx2, self.xPressAx2 = y, x

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            ID = self.getHoverID(xdata, ydata)

            # If user double-pressed 'b' then draw over the labels
            color = self.brushButton.palette().button().color().name()
            drawUnder = color != self.doublePressKeyButtonColor

            if ID > 0 and drawUnder:
                self.ax2BrushID = ID
                self.isNewID = False
            else:
                self.setBrushID()
                self.ax2BrushID = posData.brushID
                self.isNewID = True

            self.updateLookuptable(lenNewLut=self.ax2BrushID+1)
            self.isMouseDragImg2 = True

            # Draw new objects
            localLab = lab_2D[ymin:ymax, xmin:xmax]
            mask = diskMask.copy()
            if drawUnder:
                mask[localLab!=0] = False

            self.applyBrushMask(ymin, xmin, ymax, xmax, mask, self.ax2BrushID)

            self.setImageImg2(updateLookuptable=False)
            self.lastHoverID = -1

        # Delete entire ID (set to 0)
        elif middle_click and canDelete:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            delID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if delID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                delID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to delete',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                delID_prompt.exec_()
                if delID_prompt.cancel:
                    return
                delID = delID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    delID, 'Delete ID',
                                    posData.doNotShowAgain_DelID,
                                    posData.UndoFutFrames_DelID,
                                    posData.applyFutFrames_DelID)

            if UndoFutFrames is None:
                return

            posData.doNotShowAgain_DelID = doNotShowAgain
            posData.UndoFutFrames_DelID = UndoFutFrames
            posData.applyFutFrames_DelID = applyFutFrames

            self.current_frame_i = posData.frame_i

            # Apply Delete ID to future frames if requested
            if applyFutFrames:
                # Store current data before going to future frames
                self.store_data()
                for i in range(posData.frame_i+1, endFrame_i+1):
                    lab = posData.allData_li[i]['labels']
                    if lab is None:
                        break

                    lab[lab==delID] = 0

                    # Store change
                    posData.allData_li[i]['labels'] = lab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    self.store_data()

            # Back to current frame
            if applyFutFrames:
                posData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)
            posData.lab[posData.lab==delID] = 0

            # Update data (rp, etc)
            self.update_rp()

            self.warnEditingWithCca_df('Delete ID')

            self.setImageImg2()

            # Remove contour and LabelItem of deleted ID
            self.ax1_ContoursCurves[delID-1].setData([], [])
            self.ax1_LabelItemsIDs[delID-1].setText('')
            self.ax2_LabelItemsIDs[delID-1].setText('')

            self.setTitleText()
            self.highlightLostNew()
            # self.checkIDsMultiContour()

        # Separate bud
        elif (right_click or left_click) and self.separateBudButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(self.get_2Dlab(posData.lab), y, x)
                sepID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to split',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                sepID_prompt.exec_()
                if sepID_prompt.cancel:
                    return
                else:
                    ID = sepID_prompt.EntryID

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            max_ID = np.max(posData.lab)

            if right_click:
                lab2D, success = self.auto_separate_bud_ID(
                    ID, self.get_2Dlab(posData.lab), posData.rp, max_ID,
                    enforce=True
                )
                self.set_2Dlab(lab2D)
            else:
                success = False

            # If automatic bud separation was not successfull call manual one
            if not success:
                posData.disableAutoActivateViewerWindow = True
                img = self.getDisplayedCellsImg()
                manualSep = apps.manualSeparateGui(
                    self.get_2Dlab(posData.lab), ID, img,
                    fontSize=self.fontSize,
                    IDcolor=self.img2.lut[ID],
                    parent=self
                )
                manualSep.show()
                manualSep.centerWindow()
                loop = QEventLoop(self)
                manualSep.loop = loop
                loop.exec_()
                if manualSep.cancel:
                    posData.disableAutoActivateViewerWindow = False
                    if not self.separateBudButton.findChild(QAction).isChecked():
                        self.separateBudButton.setChecked(False)
                    return
                lab2D = self.get_2Dlab(posData.lab)
                lab2D[manualSep.lab!=0] = manualSep.lab[manualSep.lab!=0]
                self.set_2Dlab(lab2D)
                posData.disableAutoActivateViewerWindow = False

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in posData.rp]
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            # Update all images
            self.updateALLimg()
            self.warnEditingWithCca_df('Separate IDs')
            self.store_data()

            if not self.separateBudButton.findChild(QAction).isChecked():
                self.separateBudButton.setChecked(False)

        # Fill holes
        elif right_click and self.fillHolesToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                clickedBkgrID = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'fill the holes of',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                clickedBkgrID.exec_()
                if clickedBkgrID.cancel:
                    return
                else:
                    ID = clickedBkgrID.EntryID

            if ID in posData.lab:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                obj_idx = posData.IDs.index(ID)
                obj = posData.rp[obj_idx]
                objMask = self.getObjImage(obj.image, obj.bbox)
                localFill = scipy.ndimage.binary_fill_holes(objMask)
                posData.lab[self.getObjSlice(obj.slice)][localFill] = ID

                self.update_rp()
                self.updateALLimg()

                if not self.fillHolesToolButton.findChild(QAction).isChecked():
                    self.fillHolesToolButton.setChecked(False)

        # Hull contour
        elif right_click and self.hullContToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'replace with Hull contour',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            if ID in posData.lab:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                obj_idx = posData.IDs.index(ID)
                obj = posData.rp[obj_idx]
                objMask = self.getObjImage(obj.image, obj.bbox)
                localHull = skimage.morphology.convex_hull_image(objMask)
                posData.lab[self.getObjSlice(obj.slice)][localHull] = ID

                self.update_rp()
                self.updateALLimg()

                if not self.hullContToolButton.findChild(QAction).isChecked():
                    self.hullContToolButton.setChecked(False)

        # Move label
        elif right_click and self.moveLabelToolButton.isChecked():
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            x, y = event.pos().x(), event.pos().y()
            self.imgRGB = self.img1.image.copy()
            self.startMovingLabel(x, y)

        # Fill holes
        elif right_click and self.fillHolesToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                clickedBkgrID = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'fill the holes of',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                clickedBkgrID.exec_()
                if clickedBkgrID.cancel:
                    return
                else:
                    ID = clickedBkgrID.EntryID

        # Merge IDs
        elif right_click and self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here first ID that you want to merge',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.firstID = ID

        # Edit ID
        elif right_click and self.editID_Button.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                editID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to replace with a new one',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                editID_prompt.show(block=True)

                if editID_prompt.cancel:
                    return
                else:
                    ID = editID_prompt.EntryID

            obj_idx = posData.IDs.index(ID)
            y, x = posData.rp[obj_idx].centroid[-2:]
            xdata, ydata = int(x), int(y)

            posData.disableAutoActivateViewerWindow = True
            prev_IDs = posData.IDs.copy()
            editID = apps.editID_QWidget(ID, posData.IDs, parent=self)
            editID.show(block=True)
            if editID.cancel:
                posData.disableAutoActivateViewerWindow = False
                if not self.editID_Button.findChild(QAction).isChecked():
                    self.editID_Button.setChecked(False)
                return

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                ID, 'Edit ID', posData.doNotShowAgain_EditID,
                posData.UndoFutFrames_EditID, posData.applyFutFrames_EditID,
                applyTrackingB=True
            )

            if UndoFutFrames is None:
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            for old_ID, new_ID in editID.how:
                self.addNewItems(new_ID)

                if new_ID in prev_IDs:
                    tempID = np.max(posData.lab) + 1
                    posData.lab[posData.lab == old_ID] = tempID
                    posData.lab[posData.lab == new_ID] = old_ID
                    posData.lab[posData.lab == tempID] = new_ID

                    old_ID_idx = prev_IDs.index(old_ID)
                    new_ID_idx = prev_IDs.index(new_ID)

                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = posData.rp[old_ID_idx]
                    y, x = self.getObjCentroid(obj.centroid)
                    y, x = int(y), int(x)
                    posData.editID_info.append((y, x, new_ID))
                    obj = posData.rp[new_ID_idx]
                    y, x = self.getObjCentroid(obj.centroid)
                    y, x = int(y), int(x)
                    posData.editID_info.append((y, x, old_ID))
                else:
                    posData.lab[posData.lab == old_ID] = new_ID
                    old_ID_idx = posData.IDs.index(old_ID)

                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = posData.rp[old_ID_idx]
                    y, x = self.getObjCentroid(obj.centroid)
                    y, x = int(y), int(x)
                    posData.editID_info.append((y, x, new_ID))

            # Update rps
            self.update_rp()

            # Since we manually changed an ID we don't want to repeat tracking
            self.setTitleText()
            self.highlightLostNew()
            # self.checkIDsMultiContour()

            # Update colors for the edited IDs
            self.updateLookuptable()

            self.warnEditingWithCca_df('Edit ID')

            self.updateALLimg()
            if not self.editID_Button.findChild(QAction).isChecked():
                self.editID_Button.setChecked(False)

            posData.disableAutoActivateViewerWindow = True

            # Perform desired action on future frames
            posData.doNotShowAgain_EditID = doNotShowAgain
            posData.UndoFutFrames_EditID = UndoFutFrames
            posData.applyFutFrames_EditID = applyFutFrames

            self.current_frame_i = posData.frame_i

            if applyFutFrames:
                # Store data for current frame
                self.store_data()
                if endFrame_i is None:
                    self.app.restoreOverrideCursor()
                    return
                for i in range(posData.frame_i+1, endFrame_i+1):
                    posData.frame_i = i
                    self.get_data()
                    if self.onlyTracking:
                        self.tracking(enforce=True)
                    else:
                        for old_ID, new_ID in editID.how:
                            if new_ID in prev_IDs:
                                tempID = posData.lab.max() + 1
                                posData.lab[posData.lab == old_ID] = tempID
                                posData.lab[posData.lab == new_ID] = old_ID
                                posData.lab[posData.lab == tempID] = new_ID
                            else:
                                posData.lab[posData.lab == old_ID] = new_ID
                        self.update_rp(draw=False)
                    self.store_data()

                # Back to current frame
                posData.frame_i = self.current_frame_i
                self.get_data()
                self.app.restoreOverrideCursor()

        # Annotate cell as removed from the analysis
        elif right_click and self.binCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                binID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to remove from the analysis',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                binID_prompt.exec_()
                if binID_prompt.cancel:
                    return
                else:
                    ID = binID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    ID, 'Exclude cell from analysis',
                                    posData.doNotShowAgain_BinID,
                                    posData.UndoFutFrames_BinID,
                                    posData.applyFutFrames_BinID)

            if UndoFutFrames is None:
                return

            posData.doNotShowAgain_BinID = doNotShowAgain
            posData.UndoFutFrames_BinID = UndoFutFrames
            posData.applyFutFrames_BinID = applyFutFrames

            self.current_frame_i = posData.frame_i

            # Apply Exclude cell from analysis to future frames if requested
            if applyFutFrames:
                # Store current data before going to future frames
                self.store_data()
                for i in range(posData.frame_i+1, endFrame_i+1):
                    posData.frame_i = i
                    self.get_data()
                    if ID in posData.binnedIDs:
                        posData.binnedIDs.remove(ID)
                    else:
                        posData.binnedIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data()

                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                posData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in posData.binnedIDs:
                posData.binnedIDs.remove(ID)
            else:
                posData.binnedIDs.add(ID)

            self.update_rp_metadata()

            # Gray out ore restore binned ID
            self.updateLookuptable()

            if not self.binCellButton.findChild(QAction).isChecked():
                self.binCellButton.setChecked(False)

        # Annotate cell as dead
        elif right_click and self.ripCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                ripID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as dead',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                ripID_prompt.exec_()
                if ripID_prompt.cancel:
                    return
                else:
                    ID = ripID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    ID, 'Annotate cell as dead',
                                    posData.doNotShowAgain_RipID,
                                    posData.UndoFutFrames_RipID,
                                    posData.applyFutFrames_RipID)

            if UndoFutFrames is None:
                return

            posData.doNotShowAgain_RipID = doNotShowAgain
            posData.UndoFutFrames_RipID = UndoFutFrames
            posData.applyFutFrames_RipID = applyFutFrames

            self.current_frame_i = posData.frame_i

            # Apply Edit ID to future frames if requested
            if applyFutFrames:
                # Store current data before going to future frames
                self.store_data()
                for i in range(posData.frame_i+1, endFrame_i+1):
                    posData.frame_i = i
                    self.get_data()
                    if ID in posData.ripIDs:
                        posData.ripIDs.remove(ID)
                    else:
                        posData.ripIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data()
                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                posData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in posData.ripIDs:
                posData.ripIDs.remove(ID)
            else:
                posData.ripIDs.add(ID)

            self.update_rp_metadata()

            # Gray out dead ID
            self.updateLookuptable()
            self.store_data()

            self.warnEditingWithCca_df('Annotate ID as dead')

            if not self.ripCellButton.findChild(QAction).isChecked():
                self.ripCellButton.setChecked(False)

    def expandLabelCallback(self, checked):
        self.disconnectLeftClickButtons()
        self.uncheckLeftClickButtons(self.sender())
        self.connectLeftClickButtons()
        self.expandFootprintSize = 1

    def expandLabel(self, dilation=True):
        posData = self.data[self.pos_i]
        if self.hoverLabelID == 0:
            self.isExpandingLabel = False
            return

        # Reinitialize label to expand when we hover on a different ID
        # or we change direction
        reinitExpandingLab = (
            self.expandingID != self.hoverLabelID
            or dilation != self.isDilation
        )

        ID = self.hoverLabelID

        obj = posData.rp[posData.IDs.index(ID)]

        if reinitExpandingLab:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            # hoverLabelID different from previously expanded ID --> reinit
            self.isExpandingLabel = True
            self.expandingID = ID
            self.expandingIDColor = posData.lut[ID]/255
            self.expandingLab = np.zeros_like(self.currentLab2D)
            self.expandingLab[obj.coords[:,-2], obj.coords[:,-1]] = ID
            self.expandFootprintSize = 1
            self.imgRGB = self.img1.image.copy()

        prevCoords = (obj.coords[:,-2], obj.coords[:,-1])
        self.currentLab2D[obj.coords[:,-2], obj.coords[:,-1]] = 0
        lab_2D = self.get_2Dlab(posData.lab)
        lab_2D[obj.coords[:,-2], obj.coords[:,-1]] = 0

        footprint = skimage.morphology.disk(self.expandFootprintSize)
        if dilation:
            expandedLab = skimage.morphology.dilation(
                self.expandingLab, footprint
            )
            self.isDilation = True
        else:
            expandedLab = skimage.morphology.erosion(
                self.expandingLab, footprint
            )
            self.isDilation = False

        # Prevent expanding into neighbouring labels
        expandedLab[self.currentLab2D>0] = 0

        # Get coords of the dilated/eroded object
        expandedObj = skimage.measure.regionprops(expandedLab)[0]
        expandedObjCoords = (expandedObj.coords[:,-2], expandedObj.coords[:,-1])

        # Add the dilated/erored object
        self.currentLab2D[expandedObjCoords] = self.expandingID
        lab_2D[expandedObjCoords] = self.expandingID

        self.set_2Dlab(lab_2D)

        self.update_rp()

        if not self.labelsGrad.hideLabelsImgAction.isChecked():
            self.img2.setImage(img=self.currentLab2D, autoLevels=False)

        self.setTempImg1ExpandLabel(prevCoords, expandedObjCoords)

    def startMovingLabel(self, xPos, yPos):
        posData = self.data[self.pos_i]
        xdata, ydata = int(xPos), int(yPos)
        lab_2D = self.get_2Dlab(posData.lab)
        ID = lab_2D[ydata, xdata]
        if ID == 0:
            self.isMovingLabel = False
            return

        posData = self.data[self.pos_i]
        self.isMovingLabel = True
        self.movingID = ID
        self.prevMovePos = (xdata, ydata)
        self.movingIDColor = posData.lut[ID]/255
        movingObj = posData.rp[posData.IDs.index(ID)]
        self.movingObjCoords = movingObj.coords.copy()


    def dragLabel(self, xPos, yPos):
        posData = self.data[self.pos_i]
        lab_2D = self.get_2Dlab(posData.lab)
        Y, X = lab_2D.shape
        xdata, ydata = int(xPos), int(yPos)
        if xdata<0 or ydata<0 or xdata>=X or ydata>=Y:
            return

        xStart, yStart = self.prevMovePos
        deltaX = xdata-xStart
        deltaY = ydata-yStart

        yy, xx = self.movingObjCoords[:,-2], self.movingObjCoords[:,-1]
        if self.isSegm3D:
            zz = self.movingObjCoords[:,0]
            posData.lab[zz, yy, xx] = 0
            prevCoords = (zz.copy(), yy.copy(), xx.copy())
        else:
            posData.lab[yy, xx] = 0
            prevCoords = (None, yy.copy(), xx.copy())
        self.currentLab2D[yy, xx] = 0

        self.movingObjCoords[:,-2] = self.movingObjCoords[:,-2]+deltaY
        self.movingObjCoords[:,-1] = self.movingObjCoords[:,-1]+deltaX

        yy, xx = self.movingObjCoords[:,-2], self.movingObjCoords[:,-1]

        yy[yy<0] = 0
        xx[xx<0] = 0
        yy[yy>=Y] = Y-1
        xx[xx>=X] = X-1

        if self.isSegm3D:
            zz = self.movingObjCoords[:,0]
            posData.lab[zz, yy, xx] = self.movingID
            newCoords = (zz.copy(), yy.copy(), xx.copy())
        else:
            posData.lab[yy, xx] = self.movingID
            newCoords = (None, yy.copy(), xx.copy())

        if not self.labelsGrad.hideLabelsImgAction.isChecked():
            self.img2.setImage(self.currentLab2D)

        self.setTempImg1MoveLabel(prevCoords)

        self.prevMovePos = (xdata, ydata)

    @myutils.exception_handler
    def gui_mouseDragEventImg1(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        Y, X = self.get_2Dlab(posData.lab).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            return

        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.drawAutoContour(y, x)

        # Brush dragging mouse --> keep painting
        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            # Build brush mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            # If user double-pressed 'b' then draw over the labels
            color = self.brushButton.palette().button().color().name()
            drawUnder = color != self.doublePressKeyButtonColor
            if drawUnder:
                mask[lab_2D!=0] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.brushButton, brush=self.ax2_BrushCircleBrush
                )

            # Apply brush mask
            self.applyBrushMask(0,0,0,0, mask, posData.brushID, isLocal=False)

            self.setImageImg2(updateLookuptable=False)

            brushMask = self.get_2Dlab(posData.lab) == posData.brushID
            self.setTempImg1Brush(brushMask)

        # Eraser dragging mouse --> keep erasing
        elif self.isMouseDragImg1 and self.eraserButton.isChecked():
            posData = self.data[self.pos_i]
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()

            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[lab_2D!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )


            self.erasedIDs.extend(lab_2D[mask])
            self.applyEraserMask(mask)

            self.setImageImg2()

            self.erasesedLab = np.zeros_like(posData.lab)
            for erasedID in np.unique(self.erasedIDs):
                if erasedID == 0:
                    continue
                self.erasesedLab[posData.lab==erasedID] = erasedID

            how = self.drawIDsContComboBox.currentText()
            self.setTempImg1Eraser(mask)

        # Move label dragging mouse --> keep moving
        elif self.isMovingLabel and self.moveLabelToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            self.dragLabel(x, y)

        # Wand dragging mouse --> keep doing the magic
        elif self.isMouseDragImg1 and self.wandToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            tol = self.wandToleranceSlider.value()
            flood_mask = skimage.segmentation.flood(
                self.flood_img, (ydata, xdata), tolerance=tol
            )
            drawUnderMask = np.logical_or(
                posData.lab==0, posData.lab==posData.brushID
            )
            flood_mask = np.logical_and(flood_mask, drawUnderMask)

            self.flood_mask[flood_mask] = True

            if self.wandAutoFillCheckbox.isChecked():
                self.flood_mask = scipy.ndimage.binary_fill_holes(
                    self.flood_mask
                )

            if np.any(self.flood_mask):
                mask = np.logical_or(
                    self.flood_mask,
                    posData.lab==posData.brushID
                )
                self.setTempImg1Brush(mask)

    def highlightIDcheckBoxToggled(self, checked):
        if not checked:
            self.highlightedID = 0
            self.initLookupTableLab()
            self.updateALLimg()
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            self.updatePropsWidget(self.highlightedID)

    def updatePropsWidget(self, ID):
        if isinstance(ID, str):
            # Function called by currentTextChanged of channelCombobox or
            # additionalMeasCombobox. We set elf.currentPropsID = 0 to force update
            ID = self.guiTabControl.propsQGBox.idSB.value()
            self.currentPropsID = 0

        update = (
            self.propsDockWidget.isVisible()
            and ID != 0 and ID!=self.currentPropsID
        )
        if not update:
            return

        posData = self.data[self.pos_i]
        if posData.rp is None:
            self.update_rp()

        if not posData.IDs:
            # empty segmentation mask
            return

        if self.guiTabControl.highlightCheckbox.isChecked():
            self.highlightSearchedID(ID)

        propsQGBox = self.guiTabControl.propsQGBox

        if ID not in posData.IDs:
            s = f'Object ID {ID} does not exist'
            propsQGBox.notExistingIDLabel.setText(s)
            return

        propsQGBox.notExistingIDLabel.setText('')
        self.currentPropsID = ID
        propsQGBox.idSB.setValue(ID)
        obj_idx = posData.IDs.index(ID)
        obj = posData.rp[obj_idx]

        if self.isSegm3D:
            if self.zProjComboBox.currentText() == 'single z-slice':
                local_z = self.z_lab() - obj.bbox[0]
                area_pxl = np.count_nonzero(obj.image[local_z])
            else:
                area_pxl = np.count_nonzero(obj.image.max(axis=0))
        else:
            area_pxl = obj.area

        propsQGBox.cellAreaPxlSB.setValue(area_pxl)

        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeX = posData.PhysicalSizeX
        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX

        area_um2 = area_pxl*yx_pxl_to_um2

        propsQGBox.cellAreaUm2DSB.setValue(area_um2)

        if self.isSegm3D:
            PhysicalSizeZ = posData.PhysicalSizeZ
            vol_vox_3D = obj.area
            vol_fl_3D = vol_vox_3D*PhysicalSizeZ*PhysicalSizeY*PhysicalSizeX
            propsQGBox.cellVolVox3D_SB.setValue(vol_vox_3D)
            propsQGBox.cellVolFl3D_DSB.setValue(vol_fl_3D)

        vol_vox, vol_fl = _calc_rot_vol(
            obj, PhysicalSizeY, PhysicalSizeX
        )
        propsQGBox.cellVolVoxSB.setValue(vol_vox)
        propsQGBox.cellVolFlDSB.setValue(vol_fl)


        minor_axis_length = max(1, obj.minor_axis_length)
        elongation = obj.major_axis_length/minor_axis_length
        propsQGBox.elongationDSB.setValue(elongation)

        solidity = obj.solidity
        propsQGBox.solidityDSB.setValue(solidity)

        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        selectedChannel = intensMeasurQGBox.channelCombobox.currentText()
        if selectedChannel == self.user_ch_name:
            imgData = posData.img_data[posData.frame_i]
        else:
            _, filename = self.getPathFromChName(selectedChannel, posData)
            imgData = posData.fluo_data_dict[filename][posData.frame_i]

        objData = imgData[obj.slice][obj.image]

        intensMeasurQGBox.minimumDSB.setValue(np.min(objData))
        intensMeasurQGBox.maximumDSB.setValue(np.max(objData))
        intensMeasurQGBox.meanDSB.setValue(np.mean(objData))
        intensMeasurQGBox.medianDSB.setValue(np.median(objData))

        funcDesc = intensMeasurQGBox.additionalMeasCombobox.currentText()
        func = intensMeasurQGBox.additionalMeasCombobox.functions[funcDesc]
        if funcDesc == 'Concentration':
            bkgrVal = np.median(imgData[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount/vol_vox
        elif funcDesc == 'Amount':
            bkgrVal = np.median(imgData[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount
        else:
            value = func(objData)

        intensMeasurQGBox.additionalMeasCombobox.indicator.setValue(value)


    def gui_hoverEventImg1(self, event):
        posData = self.data[self.pos_i]
        # Update x, y, value label bottom right
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        # Cursor left image --> restore cursor
        if event.isExit() and self.app.overrideCursor() is not None:
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

        # Alt key was released --> restore cursor
        modifiers = QGuiApplication.keyboardModifiers()
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        setBrushCursor = (
            self.brushButton.isChecked() and not event.isExit()
            and (noModifier or shift)
        )
        setEraserCursor = (
            self.eraserButton.isChecked() and not event.isExit()
            and noModifier
        )
        if setBrushCursor or setEraserCursor:
            self.app.setOverrideCursor(Qt.CrossCursor)

        setWandCursor = (
            self.wandToolButton.isChecked() and not event.isExit()
            and noModifier
        )
        if setWandCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(self.wandCursor)

        setMoveLabelCursor = (
            self.moveLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )

        setExpandLabelCursor = (
            self.expandLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )

        setCurvCursor = (
            self.curvToolButton.isChecked() and not event.isExit()
            and noModifier
        )
        if setCurvCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(self.curvCursor)

        setCustomAnnotCursor = (
            self.customAnnotButton is not None and not event.isExit()
            and noModifier
        )
        if setCustomAnnotCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.PointingHandCursor)

        # Cursor is moving on image while Alt key is pressed --> pan cursor
        alt = QGuiApplication.keyboardModifiers() == Qt.AltModifier
        setPanImageCursor = alt and not event.isExit()
        if setPanImageCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)

        drawRulerLine = (
            self.rulerButton.isChecked() and self.rulerHoverON
            and not event.isExit()
        )
        if drawRulerLine:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
            if self.isCtrlDown:
                ydata = yyRA[0]
            self.ax1_rulerPlotItem.setData([xxRA[0], xdata], [yyRA[0], ydata])

        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.img1.image
            Y, X = _img.shape[:2]
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                maxVal = _img.max()
                ID = self.currentLab2D[ydata, xdata]
                self.updatePropsWidget(ID)
                if posData.IDs:
                    maxID = max(posData.IDs)
                else:
                    maxID = 0
                if _img.ndim > 2:
                    val = [v for v in val]
                    value = f'{val}'
                    val_l0 = self.img_layer0[ydata, xdata]
                    val_str = f'rgb={val}, value_l0={val_l0:.2f}'
                else:
                    val_str = f'value={val:.2f}'
                txt = (
                    f'x={x:.2f}, y={y:.2f}, {val_str}, '
                    f'max={maxVal:.2f}, ID={ID}, max_ID={maxID}, '
                    f'num. of objects={len(posData.IDs)}'
                )
                xx, yy = self.ax1_rulerPlotItem.getData()
                if xx is not None:
                    lenPxl = math.sqrt((xx[0]-xx[1])**2 + (yy[0]-yy[1])**2)
                    pxlToUm = self.data[self.pos_i].PhysicalSizeX
                    lenTxt = (
                        f'length={lenPxl:.2f} pxl ({lenPxl*pxlToUm:.2f} um)'
                    )
                    txt = f'{txt}, {lenTxt}'
                self.wcLabel.setText(txt)
            else:
                self.clickedOnBud = False
                self.BudMothTempLine.setData([], [])
                self.wcLabel.setText(f'')

        if setMoveLabelCursor or setExpandLabelCursor:
            x, y = event.pos()
            self.updateHoverLabelCursor(x, y)

        # Draw eraser circle
        if setEraserCursor:
            x, y = event.pos()
            self.updateEraserCursor(x, y)
            self.hideItemsHoverBrush(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )

        # Draw Brush circle
        if setBrushCursor:
            x, y = event.pos()
            self.updateBrushCursor(x, y)
            self.hideItemsHoverBrush(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

        drawMothBudLine = (
            self.assignBudMothButton.isChecked() and self.clickedOnBud
            and not event.isExit()
        )
        if drawMothBudLine:
            x, y = event.pos()
            y2, x2 = y, x
            xdata, ydata = int(x), int(y)
            y1, x1 = self.yClickBud, self.xClickBud
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                self.BudMothTempLine.setData([x1, x2], [y1, y2])
            else:
                obj_idx = posData.IDs.index(ID)
                obj = posData.rp[obj_idx]
                y2, x2 = self.getObjCentroid(obj.centroid)
                self.BudMothTempLine.setData([x1, x2], [y1, y2])

        # Temporarily draw spline curve
        # see https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        drawSpline = (
            self.curvToolButton.isChecked() and self.splineHoverON
            and not event.isExit()
        )
        if drawSpline:
            x, y = event.pos()
            xx, yy = self.curvAnchors.getData()
            hoverAnchors = self.curvAnchors.pointsAt(event.pos())
            per=False
            # If we are hovering the starting point we generate
            # a closed spline
            if len(xx) >= 2:
                if len(hoverAnchors)>0:
                    xA_hover, yA_hover = hoverAnchors[0].pos()
                    if xx[0]==xA_hover and yy[0]==yA_hover:
                        per=True
                if per:
                    # Append start coords and close spline
                    xx = np.r_[xx, xx[0]]
                    yy = np.r_[yy, yy[0]]
                    xi, yi = self.getSpline(xx, yy, per=per)
                    # self.curvPlotItem.setData([], [])
                else:
                    # Append mouse coords
                    xx = np.r_[xx, x]
                    yy = np.r_[yy, y]
                    xi, yi = self.getSpline(xx, yy, per=per)
                self.curvHoverPlotItem.setData(xi, yi)

    def gui_hoverEventImg2(self, event):
        posData = self.data[self.pos_i]
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        # Cursor left image --> restore cursor
        if event.isExit() and self.app.overrideCursor() is not None:
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

        # Alt key was released --> restore cursor
        modifiers = QGuiApplication.keyboardModifiers()
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        setBrushCursor = (
            self.brushButton.isChecked() and not event.isExit()
            and (noModifier or shift)
        )
        setEraserCursor = (
            self.eraserButton.isChecked() and not event.isExit()
            and noModifier
        )
        if setBrushCursor or setEraserCursor:
            self.app.setOverrideCursor(Qt.CrossCursor)

        setMoveLabelCursor = (
            self.moveLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )

        setExpandLabelCursor = (
            self.expandLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )

        # Cursor is moving on image while Alt key is pressed --> pan cursor
        alt = QGuiApplication.keyboardModifiers() == Qt.AltModifier
        setPanImageCursor = alt and not event.isExit()
        if setPanImageCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)

        # Update x, y, value label bottom right
        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.currentLab2D
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                if posData.IDs:
                    maxID = max(posData.IDs)
                else:
                    maxID = 0
                maxVal = np.max(self.img1.image)
                img1_val = self.img1.image[ydata, xdata]
                if self.img1.image.ndim > 2:
                    img1_val = [v for v in img1_val]
                    val_l0 = self.img_layer0[ydata, xdata]
                    val_str = f'rgb={img1_val}, value_l0={val_l0:.2f}'
                else:
                    val_str = f'value={img1_val:.2f}'
                self.wcLabel.setText(
                    f'x={x:.2f}, y={y:.2f}, {val_str}, '
                    f'max={maxVal:.2f}, ID={val}, max_ID={maxID}, '
                    f'num. of objects={len(posData.IDs)}'
                )
            else:
                if self.eraserButton.isChecked() or self.brushButton.isChecked():
                    self.gui_mouseReleaseEventImg2(event)
                self.wcLabel.setText(f'')

        if setMoveLabelCursor or setExpandLabelCursor:
            x, y = event.pos()
            self.updateHoverLabelCursor(x, y)

        # Draw eraser circle
        if setEraserCursor:
            x, y = event.pos()
            self.updateEraserCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )

        # Draw Brush circle
        if setBrushCursor:
            x, y = event.pos()
            self.updateBrushCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

    def gui_gradientContextMenuEvent(self, event):
        self.imgGrad.gradient.menu.removeAction(self.userChNameAction)
        self.imgGrad.gradient.menu.addAction(self.userChNameAction)
        posData = self.data[self.pos_i]
        if self.overlayButton.isChecked():
            for action in posData.fluoDataChNameActions:
                self.imgGrad.gradient.menu.removeAction(action)
                self.imgGrad.gradient.menu.addAction(action)
        try:
            # Convert QPointF to QPoint
            self.imgGrad.gradient.menu.popup(event.screenPos().toPoint())
        except AttributeError:
            self.imgGrad.gradient.menu.popup(event.screenPos())
        # self.imgGrad.gradient.showMenu(event)

    @myutils.exception_handler
    def gui_mouseDragEventImg2(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        Y, X = self.get_2Dlab(posData.lab).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            return

        # Eraser dragging mouse --> keep erasing
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            posData = self.data[self.pos_i]
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[lab_2D!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )

            self.erasedIDs.extend(lab_2D[mask])

            self.applyEraserMask(mask)
            self.setImageImg2(updateLookuptable=False)

        # Brush paint dragging mouse --> keep painting
        if self.isMouseDragImg2 and self.brushButton.isChecked():
            posData = self.data[self.pos_i]
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            # Build brush mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            # If user double-pressed 'b' then draw over the labels
            color = self.brushButton.palette().button().color().name()
            if color != self.doublePressKeyButtonColor:
                mask[posData.lab!=0] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.eraserButton, brush=self.ax2_BrushCircleBrush
                )

            # Apply brush mask
            self.applyBrushMask(0,0,0,0, mask, self.ax2BrushID, isLocal=False)

            self.setImageImg2()

        # Move label dragging mouse --> keep moving
        elif self.isMovingLabel and self.moveLabelToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            self.dragLabel(x, y)

    @myutils.exception_handler
    def gui_mouseReleaseEventImg2(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        Y, X = self.get_2Dlab(posData.lab).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            self.isMouseDragImg2 = False
            self.updateALLimg()
            return

        # Eraser mouse release --> update IDs and contours
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            self.isMouseDragImg2 = False
            erasedIDs = np.unique(self.erasedIDs)

            # Update data (rp, etc)
            self.update_rp()
            self.updateALLimg(updateFilters=True, useEraserImg=True)

            for ID in erasedIDs:
                if ID not in posData.lab:
                    self.warnEditingWithCca_df('Delete ID with eraser')
                    break

        # Brush button mouse release --> update IDs and contours
        elif self.isMouseDragImg2 and self.brushButton.isChecked():
            self.isMouseDragImg2 = False

            self.update_rp()
            if self.isNewID:
                self.tracking(enforce=True, assign_unique_new_IDs=False)

            self.updateALLimg(updateFilters=True)
            if self.isNewID:
                self.warnEditingWithCca_df('Add new ID with brush tool')

        # Move label mouse released, update move
        elif self.isMovingLabel and self.moveLabelToolButton.isChecked():
            self.isMovingLabel = False

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            self.updateALLimg(useStoredGaussFiltered=True)

            if not self.moveLabelToolButton.findChild(QAction).isChecked():
                self.moveLabelToolButton.setChecked(False)

        # Merge IDs
        elif self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to merge with ID '
                         f'{self.firstID}',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            posData.lab[posData.lab==ID] = self.firstID

            # Mask to keep track of which ID needs redrawing of the contours
            mergedID_mask = self.get_2Dlab(posData.lab)==self.firstID

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            self.updateALLimg()
            if not self.mergeIDsButton.findChild(QAction).isChecked():
                self.mergeIDsButton.setChecked(False)
            self.store_data()
            self.warnEditingWithCca_df('Merge IDs')

    @myutils.exception_handler
    def gui_mouseReleaseEventImg1(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        Y, X = self.get_2Dlab(posData.lab).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            self.isMouseDragImg2 = False
            self.updateALLimg()
            return

        if mode=='Segmentation and Tracking' or self.isSnapshot:
            # Allow right-click actions on both images
            self.gui_mouseReleaseEventImg2(event)

        # Right-click curvature tool mouse release
        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            self.isRightClickDragImg1 = False
            try:
                self.splineToObj(isRightClick=True)
                self.update_rp()
                self.tracking(enforce=True, assign_unique_new_IDs=False)
                self.updateALLimg()
                self.warnEditingWithCca_df('Add new ID with curvature tool')
                self.clearCurvItems()
                self.curvTool_cb(True)
            except ValueError:
                self.clearCurvItems()
                self.curvTool_cb(True)
                pass

        # Eraser mouse release --> update IDs and contours
        elif self.isMouseDragImg1 and self.eraserButton.isChecked():
            self.isMouseDragImg1 = False
            erasedIDs = np.unique(self.erasedIDs)

            # Update data (rp, etc)
            self.update_rp()
            self.updateALLimg(useStoredGaussFiltered=True, useEraserImg=True)

            for ID in erasedIDs:
                if ID not in posData.IDs:
                    self.warnEditingWithCca_df('Delete ID with eraser')
                    break

        # Brush button mouse release
        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            self.isMouseDragImg1 = False

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            # Update colors to include a new color for the new ID
            self.updateALLimg(useStoredGaussFiltered=True)
            if self.isNewID:
                self.warnEditingWithCca_df('Add new ID with brush tool')
            self.isNewID = False

        # Wand tool release, add new object
        elif self.isMouseDragImg1 and self.wandToolButton.isChecked():
            self.isMouseDragImg1 = False

            posData = self.data[self.pos_i]
            posData.lab[self.flood_mask] = posData.brushID

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            self.updateALLimg()
            self.warnEditingWithCca_df('Add new ID with magic-wand')

        # Move label mouse released, update move
        elif self.isMovingLabel and self.moveLabelToolButton.isChecked():
            self.isMovingLabel = False

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            self.updateALLimg(useStoredGaussFiltered=True)

            if not self.moveLabelToolButton.findChild(QAction).isChecked():
                self.moveLabelToolButton.setChecked(False)

        # Assign mother to bud
        elif self.assignBudMothButton.isChecked() and self.clickedOnBud:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == self.get_2Dlab(posData.lab)[self.yClickBud, self.xClickBud]:
                return

            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                mothID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as mother cell',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                mothID_prompt.exec_()
                if mothID_prompt.cancel:
                    return
                else:
                    ID = mothID_prompt.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            relationship = posData.cca_df.at[ID, 'relationship']
            ccs = posData.cca_df.at[ID, 'cell_cycle_stage']
            is_history_known = posData.cca_df.at[ID, 'is_history_known']
            # We allow assiging a cell in G1 as mother only on first frame
            # OR if the history is unknown
            if relationship == 'bud' and posData.frame_i > 0 and is_history_known:
                self.assignBudMothButton.setChecked(False)
                txt = html_utils.paragraph(
                    f'You clicked on <b>ID {ID}</b> which is a <b>BUD</b>.<br>'
                    'To assign a bud to a cell <b>start by clicking on a bud</b> '
                    'and release on a cell in G1'
                )
                msg = widgets.myMessageBox()
                msg.critical(
                    self, 'Released on a bud', txt
                )
                self.assignBudMothButton.setChecked(True)
                return

            elif ccs != 'G1' and posData.frame_i > 0:
                self.assignBudMothButton.setChecked(False)
                txt = html_utils.paragraph(
                    f'You clicked on <b>ID={ID}</b> which is <b>NOT in G1</b>.<br>'
                    'To assign a bud to a cell start by clicking on a bud '
                    'and release on a cell in G1'
                )
                msg = widgets.myMessageBox()
                msg.critical(
                    self, 'Released on a cell NOT in G1', txt
                )
                self.assignBudMothButton.setChecked(True)
                return

            elif posData.frame_i == 0:
                # Check that clicked bud actually is smaller that mother
                # otherwise warn the user that he might have clicked first
                # on a mother
                budID = self.get_2Dlab(posData.lab)[self.yClickBud, self.xClickBud]
                new_mothID = self.get_2Dlab(posData.lab)[ydata, xdata]
                bud_obj_idx = posData.IDs.index(budID)
                new_moth_obj_idx = posData.IDs.index(new_mothID)
                rp_budID = posData.rp[bud_obj_idx]
                rp_new_mothID = posData.rp[new_moth_obj_idx]
                if rp_budID.area >= rp_new_mothID.area:
                    self.assignBudMothButton.setChecked(False)
                    msg = widgets.myMessageBox()
                    txt = (
                        f'You clicked FIRST on ID {budID} and then on {new_mothID}.\n'
                        f'For me this means that you want ID {budID} to be the '
                        f'BUD of ID {new_mothID}.\n'
                        f'However <b>ID {budID} is bigger than {new_mothID}</b> '
                        f'so maybe you shoul have clicked FIRST on {new_mothID}?\n\n'
                        'What do you want me to do?'
                    )
                    txt = html_utils.paragraph(txt)
                    swapButton, keepButton = msg.warning(
                        self, 'Which one is bud?', txt,
                        buttonsTexts=(
                            f'Assign ID {new_mothID} as the bud of ID {budID}',
                            f'Keep ID {budID} as the bud of  ID {new_mothID}'
                        )
                    )
                    if msg.clickedButton == swapButton:
                        (xdata, ydata,
                        self.xClickBud, self.yClickBud) = (
                            self.xClickBud, self.yClickBud,
                            xdata, ydata
                        )
                    self.assignBudMothButton.setChecked(True)

            elif is_history_known and not self.clickedOnHistoryKnown:
                self.assignBudMothButton.setChecked(False)
                budID = self.get_2Dlab(posData.lab)[ydata, xdata]
                # Allow assigning an unknown cell ONLY to another unknown cell
                txt = (
                    f'You started by clicking on ID {budID} which has '
                    'UNKNOWN history, but you then clicked/released on '
                    f'ID {ID} which has KNOWN history.\n\n'
                    'Only two cells with UNKNOWN history can be assigned as '
                    'relative of each other.'
                )
                msg = QMessageBox()
                msg.critical(
                    self, 'Released on a cell with KNOWN history', txt, msg.Ok
                )
                self.assignBudMothButton.setChecked(True)
                return

            self.clickedOnHistoryKnown = is_history_known
            self.xClickMoth, self.yClickMoth = xdata, ydata
            self.assignBudMoth()

            if not self.assignBudMothButton.findChild(QAction).isChecked():
                self.assignBudMothButton.setChecked(False)

            self.clickedOnBud = False
            self.BudMothTempLine.setData([], [])

    def gui_clickedDelRoi(self, event, left_click, right_click):
        posData = self.data[self.pos_i]
        x, y = event.pos().x(), event.pos().y()

        # Check if right click on ROI
        delROIs = (
            posData.allData_li[posData.frame_i]['delROIs_info']['rois'].copy()
        )
        for r, roi in enumerate(delROIs):
            x0, y0 = [int(c) for c in roi.pos()]
            w, h = [int(c) for c in roi.size()]
            x1, y1 = x0+w, y0+h
            clickedOnROI = (
                x>=x0 and x<=x1 and y>=y0 and y<=y1
            )
            raiseContextMenuRoi = right_click and clickedOnROI
            dragRoi = left_click and clickedOnROI
            if raiseContextMenuRoi:
                self.roi_to_del = roi
                self.roiContextMenu = QMenu(self)
                separator = QAction(self)
                separator.setSeparator(True)
                self.roiContextMenu.addAction(separator)
                action = QAction('Remove ROI')
                action.triggered.connect(self.removeDelROI)
                self.roiContextMenu.addAction(action)
                self.roiContextMenu.exec_(event.screenPos())
                return True
            elif dragRoi:
                event.ignore()
                return True
        return False

    @myutils.exception_handler
    def gui_mousePressEventImg1(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        isMod = ctrl or alt
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        isCcaMode = mode == 'Cell cycle analysis'
        isCustomAnnotMode = mode == 'Custom annotations'
        left_click = event.button() == Qt.MouseButton.LeftButton and not isMod
        middle_click = self.isMiddleClick(event, modifiers)
        right_click = event.button() == Qt.MouseButton.RightButton
        isPanImageClick = self.isPanImageClick(event, modifiers)
        brushON = self.brushButton.isChecked()
        curvToolON = self.curvToolButton.isChecked()
        histON = self.setIsHistoryKnownButton.isChecked()
        eraserON = self.eraserButton.isChecked()
        rulerON = self.rulerButton.isChecked()
        wandON = self.wandToolButton.isChecked() and not isPanImageClick

        # Check if right click on ROI
        isClickOnDelRoi = self.gui_clickedDelRoi(event, left_click, right_click)
        if isClickOnDelRoi:
            return

        dragImgLeft = (
            left_click and not brushON and not histON
            and not curvToolON and not eraserON and not rulerON
            and not wandON
        )
        if isPanImageClick:
            dragImgLeft = True

        is_right_click_custom_ON = any([
            b.isChecked() for b in self.customAnnotDict.keys()
        ])

        canAnnotateDivision = (
             not self.assignBudMothButton.isChecked()
             and not self.setIsHistoryKnownButton.isChecked()
             and not self.curvToolButton.isChecked()
             and not is_right_click_custom_ON
        )

        # In timelapse mode division can be annotated if isCcaMode and right-click
        # while in snapshot mode with Ctrl+rigth-click
        isAnnotateDivision = (
            (right_click and isCcaMode and canAnnotateDivision)
            or (right_click and ctrl and self.isSnapshot)
        )

        isCustomAnnot = (
            (right_click or dragImgLeft)
            and (isCustomAnnotMode or self.isSnapshot)
            and self.customAnnotButton is not None
        )

        is_right_click_action_ON = any([
            b.isChecked() for b in self.checkableQButtonsGroup.buttons()
        ])

        isOnlyRightClick = (
            right_click and canAnnotateDivision and not isAnnotateDivision
            and not isMod and not is_right_click_action_ON
            and not is_right_click_custom_ON
        )

        if isOnlyRightClick:
            self.gui_gradientContextMenuEvent(event)
            event.ignore()
            return

        # Left click actions
        canCurv = (
            curvToolON and not self.assignBudMothButton.isChecked()
            and not brushON and not dragImgLeft and not eraserON)
        canBrush = (
            brushON and not curvToolON and not rulerON
            and not dragImgLeft and not eraserON and not wandON)
        canErase = (
            eraserON and not curvToolON and not rulerON
            and not dragImgLeft and not brushON and not wandON)
        canRuler = (
            rulerON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not wandON)
        canWand = (
            wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON)

        # Enable dragging of the image window like pyqtgraph original code
        if dragImgLeft and not isCustomAnnot:
            pg.ImageItem.mousePressEvent(self.img1, event)
            event.ignore()
            return

        dragImgMiddle = middle_click
        # if dragImgMiddle:
        #     pg.ImageItem.mousePressEvent(self.img1, event)
        #     event.ignore()
        #     return

        if mode == 'Viewer' and not canRuler:
            self.startBlinkingModeCB()
            event.ignore()
            return

        # Allow right-click or middle-click actions on both images
        eventOnImg2 = (
            (right_click or middle_click)
            and (mode=='Segmentation and Tracking' or self.isSnapshot)
            and not isAnnotateDivision
        )
        if eventOnImg2:
            event.isImg1Sender = True
            self.gui_mousePressEventImg2(event)

        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        Y, X = self.get_2Dlab(posData.lab).shape
        if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        else:
            return

        # Paint new IDs with brush and left click on the left image
        if left_click and canBrush:
            # Store undo state before modifying stuff

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape
            self.storeUndoRedoStates(False)

            # If user double-pressed 'b' then draw over the labels
            color = self.brushButton.palette().button().color().name()
            drawUnder = color != self.doublePressKeyButtonColor

            ID = self.getHoverID(xdata, ydata)

            if ID > 0 and drawUnder:
                posData.brushID = ID
                self.isNewID = False
            else:
                # Update brush ID. Take care of disappearing cells to remember
                # to not use their IDs anymore in the future
                self.isNewID = True
                self.setBrushID()
                self.updateLookuptable(lenNewLut=posData.brushID+1)

            self.brushColor = posData.lut[posData.brushID]/255

            self.yPressAx2, self.xPressAx2 = y, x

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            self.isMouseDragImg1 = True

            # Draw new objects
            localLab = lab_2D[ymin:ymax, xmin:xmax]
            mask = diskMask.copy()
            if drawUnder:
                mask[localLab!=0] = False

            self.applyBrushMask(ymin, xmin, ymax, xmax, mask, posData.brushID)

            self.setImageImg2(updateLookuptable=False)

            img = self.img1.image.copy()
            how = self.drawIDsContComboBox.currentText()
            if img.ndim > 2:
                # image is already RGB
                self.imgRGB = img/np.max(img)
            else:
                img = img/np.max(img)
                self.imgRGB = gray2rgb(img)

            brushMask = self.get_2Dlab(posData.lab) == posData.brushID
            self.setTempImg1Brush(brushMask)

            self.lastHoverID = -1

        elif left_click and canErase:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = self.get_2Dlab(posData.lab).shape
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.yPressAx2, self.xPressAx2 = y, x
            # Keep a list of erased IDs got erased
            self.erasedIDs = []
            lab_2D = self.get_2Dlab(posData.lab)
            self.erasedID = self.getHoverID(xdata, ydata)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True


            # If user double-pressed 'b' then erase over ALL labels
            color = self.eraserButton.palette().button().color().name()
            eraseOnlyOneID = (
                color != self.doublePressKeyButtonColor
                and self.erasedID != 0
            )

            self.eraseOnlyOneID = eraseOnlyOneID

            if eraseOnlyOneID:
                mask[lab_2D!=self.erasedID] = False

            self.erasedIDs.extend(lab_2D[mask])

            self.applyEraserMask(mask)

            self.setTempImg1Eraser(None, init=True)

            for erasedID in np.unique(self.erasedIDs):
                if erasedID == 0:
                    continue
                self.erasesedLab[lab_2D==erasedID] = erasedID

            self.getDisplayedCellsImg()
            self.setTempImg1Eraser(mask)

            self.img2.updateImage()
            self.isMouseDragImg1 = True

        elif left_click and canRuler:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            if not self.rulerHoverON:
                self.ax1_rulerAnchorsItem.setData([xdata], [ydata])
                self.rulerHoverON = True
            else:
                self.rulerHoverON = False
                xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
                if self.isCtrlDown:
                    ydata = yyRA[0]
                self.ax1_rulerPlotItem.setData(
                    [xxRA[0], xdata], [yyRA[0], ydata]
                )
                self.ax1_rulerAnchorsItem.setData(
                    [xxRA[0], xdata], [yyRA[0], ydata]
                )

        elif right_click and canCurv:
            # Draw manually assisted auto contour
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = self.get_2Dlab(posData.lab).shape

            self.autoCont_x0 = xdata
            self.autoCont_y0 = ydata
            self.xxA_autoCont, self.yyA_autoCont = [], []
            self.curvAnchors.addPoints([x], [y])
            img = self.getDisplayedCellsImg()
            self.autoContObjMask = np.zeros(img.shape, np.uint8)
            self.isRightClickDragImg1 = True

        elif left_click and canCurv:
            # Draw manual spline
            x, y = event.pos().x(), event.pos().y()
            Y, X = self.get_2Dlab(posData.lab).shape

            # Check if user clicked on starting anchor again --> close spline
            closeSpline = False
            clickedAnchors = self.curvAnchors.pointsAt(event.pos())
            xxA, yyA = self.curvAnchors.getData()
            if len(xxA)>0:
                if len(xxA) == 1:
                    self.splineHoverON = True
                x0, y0 = xxA[0], yyA[0]
                if len(clickedAnchors)>0:
                    xA_clicked, yA_clicked = clickedAnchors[0].pos()
                    if x0==xA_clicked and y0==yA_clicked:
                        x = x0
                        y = y0
                        closeSpline = True

            # Add anchors
            self.curvAnchors.addPoints([x], [y])
            try:
                xx, yy = self.curvHoverPlotItem.getData()
                self.curvPlotItem.setData(xx, yy)
            except Exception as e:
                # traceback.print_exc()
                pass

            if closeSpline:
                self.splineHoverON = False
                self.splineToObj()
                self.update_rp()
                self.tracking(enforce=True, assign_unique_new_IDs=False)
                self.updateALLimg()
                self.warnEditingWithCca_df('Add new ID with curvature tool')
                self.clearCurvItems()
                self.curvTool_cb(True)

        elif left_click and canWand:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = self.get_2Dlab(posData.lab).shape
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            posData.brushID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if posData.brushID == 0:
                self.setBrushID()
                self.updateLookuptable(
                    lenNewLut=np.max(posData.lab)+posData.brushID+1
                )
            self.brushColor = self.img2.lut[posData.brushID]/255

            img = self.img1.image.copy()
            img = img/np.max(img)
            if img.ndim > 2:
                self.imgRGB = img/np.max(img)
            else:
                self.imgRGB = gray2rgb(img)

            # NOTE: flood is on mousedrag or release
            tol = self.wandToleranceSlider.value()
            self.flood_img = myutils.to_uint8(self.getDisplayedCellsImg())
            flood_mask = skimage.segmentation.flood(
                self.flood_img, (ydata, xdata), tolerance=tol
            )
            bkgrLabMask = self.get_2Dlab(posData.lab)==0

            drawUnderMask = np.logical_or(
                posData.lab==0, posData.lab==posData.brushID
            )
            self.flood_mask = np.logical_and(flood_mask, drawUnderMask)

            if self.wandAutoFillCheckbox.isChecked():
                self.flood_mask = scipy.ndimage.binary_fill_holes(
                    self.flood_mask
                )

            if np.any(self.flood_mask):
                mask = np.logical_or(
                    self.flood_mask,
                    posData.lab==posData.brushID
                )
                self.setTempImg1Brush(mask)
            self.isMouseDragImg1 = True

        # Annotate cell cycle division
        elif isAnnotateDivision:
            if posData.frame_i <= 0 and not self.isSnapshot:
                return

            if posData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                divID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as divided',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                divID_prompt.exec_()
                if divID_prompt.cancel:
                    return
                else:
                    ID = divID_prompt.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            if not self.isSnapshot:
                # Annotate or undo division
                self.manualCellCycleAnnotation(ID)
            else:
                self.undoBudMothAssignment(ID)

        # Assign bud to mother (mouse down on bud)
        elif right_click and self.assignBudMothButton.isChecked():
            if self.clickedOnBud:
                # NOTE: self.clickedOnBud is set to False when assigning a mother
                # is successfull in mouse release event
                # We still have to click on a mother
                return

            if posData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                budID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID of a bud you want to correct mother assignment',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                budID_prompt.exec_()
                if budID_prompt.cancel:
                    return
                else:
                    ID = budID_prompt.EntryID

            obj_idx = posData.IDs.index(ID)
            y, x = posData.rp[obj_idx].centroid
            xdata, ydata = int(x), int(y)

            relationship = posData.cca_df.at[ID, 'relationship']
            is_history_known = posData.cca_df.at[ID, 'is_history_known']
            self.clickedOnHistoryKnown = is_history_known
            # We allow assiging a cell in G1 as bud only on first frame
            # OR if the history is unknown
            if relationship != 'bud' and posData.frame_i > 0 and is_history_known:
                txt = (f'You clicked on ID {ID} which is NOT a bud.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QMessageBox()
                msg.critical(
                    self, 'Not a bud', txt, msg.Ok
                )
                return

            self.clickedOnBud = True
            self.xClickBud, self.yClickBud = xdata, ydata

        # Annotate (or undo) that cell has unknown history
        elif right_click and self.setIsHistoryKnownButton.isChecked():
            if posData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                unknownID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as '
                         '"history UNKNOWN/KNOWN"',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                unknownID_prompt.exec_()
                if unknownID_prompt.cancel:
                    return
                else:
                    ID = unknownID_prompt.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            self.annotateIsHistoryKnown(ID)
            if not self.setIsHistoryKnownButton.findChild(QAction).isChecked():
                self.setIsHistoryKnownButton.setChecked(False)

        elif isCustomAnnot:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.nearest_nonzero(
                    self.get_2Dlab(posData.lab), y, x
                )
                clickedBkgrDialog = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as divided',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID)
                )
                clickedBkgrDialog.exec_()
                if clickedBkgrDialog.cancel:
                    return
                else:
                    ID = clickedBkgrDialog.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            button = self.doCustomAnnotation(ID)
            keepActive = self.customAnnotDict[button]['state']['keepActive']
            if not keepActive:
                button.setChecked(False)


    def gui_addCreatedAxesItems(self):
        allItems = zip(
            self.ax1_ContoursCurves,
            self.ax2_ContoursCurves,
            self.ax1_LabelItemsIDs,
            self.ax2_LabelItemsIDs,
            self.ax1_BudMothLines
        )
        self.logger.info(f'Adding {len(self.ax1_ContoursCurves)} axes items...')
        pbar = tqdm(total=len(self.ax1_ContoursCurves), ncols=100)
        for items_ID in allItems:
            (ax1ContCurve, ax2ContCurve,
            ax1_IDlabel, ax2_IDlabel,
            BudMothLine) = items_ID

            pbar.update()

            if ax1ContCurve is None:
                continue

            self.ax1.addItem(ax1ContCurve)
            self.ax1.addItem(BudMothLine)
            self.ax1.addItem(ax1_IDlabel)

            self.ax2.addItem(ax2_IDlabel)
            self.ax2.addItem(ax2ContCurve)
        pbar.close()

    def relabelSequentialCallback(self):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer' or mode == 'Cell cycle analysis':
            self.startBlinkingModeCB()
            return
        self.storeUndoRedoStates(False)
        posData = self.data[self.pos_i]
        if posData.SizeT > 1:
            self.progressWin = apps.QDialogWorkerProgress(
                title='Re-labelling sequential', parent=self,
                pbarDesc='Relabelling sequential...'
            )
            self.progressWin.show(self.app)
            self.progressWin.mainPbar.setMaximum(0)
            self.startRelabellingWorker(posData)
        else:
            posData.lab, fw, inv = skimage.segmentation.relabel_sequential(
                posData.lab
            )
            # Update annotations based on relabelling
            newIDs = list(inv.in_values)
            oldIDs = list(inv.out_values)
            newIDs.append(-1)
            oldIDs.append(-1)
            self.update_cca_df_relabelling(posData, oldIDs, newIDs)
            self.store_data()
            self.update_rp()
            li = list(zip(oldIDs, newIDs))
            s = '\n'.join([str(pair).replace(',', ' -->') for pair in li])
            s = f'IDs relabelled as follows:\n{s}'
            self.logger.info(s)

        self.updateALLimg()

    def storeTrackingAlgo(self, checked):
        if not checked:
            return

        trackingAlgo = self.sender().text()
        self.df_settings.at['tracking_algorithm', 'value'] = trackingAlgo
        self.df_settings.to_csv(self.settings_csv_path)

        if self.sender().text() == 'YeaZ':
            msg = QMessageBox()
            info_txt = html_utils.paragraph(f"""
                Note that YeaZ tracking algorithm tends to be sliglhtly more accurate
                overall, but it is <b>less capable of detecting segmentation
                errors.</b><br><br>
                If you need to correct as many segmentation errors as possible
                we recommend using Cell-ACDC tracking algorithm.
            """)
            msg.information(self, 'Info about YeaZ', info_txt, msg.Ok)

    def findID(self):
        posData = self.data[self.pos_i]
        searchIDdialog = apps.QLineEditDialog(
            title='Search object by ID',
            msg='Enter object ID to find and highlight',
            parent=self, allowedValues=posData.IDs
        )
        searchIDdialog.exec_()
        if searchIDdialog.cancel:
            return
        self.highlightSearchedID(searchIDdialog.EntryID)
        propsQGBox = self.guiTabControl.propsQGBox
        propsQGBox.idSB.setValue(searchIDdialog.EntryID)

    def workerProgress(self, text, loggerLevel='INFO'):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

    def workerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
        self.logger.info('Worker process ended.')
        self.updateALLimg()
        self.titleLabel.setText('Done', color='w')

    def trackingWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
        self.logger.info('Worker process ended.')
        if self.trackingWorker.trackingOnNeverVisitedFrames:
            msg = widgets.myMessageBox()
            title = 'Disable real-time tracking?'
            txt = (
                'You perfomed tracking on frames that you have '
                '<b>never visited.</b><br><br>'
                'Cell-ACDC default behaviour is to <b>track them again</b> when you '
                'will visit them.<br><br>'
                'However, you can <b>overwrite this behaviour</b> and explicitly '
                'disable tracking for all of the frames you already tracked.<br><br>'
                'What do you want me to do?'
            )
            _, disableTrackingButton = msg.information(
                self, title, html_utils.paragraph(txt),
                buttonsTexts=(
                    'Keep real-time tracking active (RECOMMENDED)',
                    'Disable real-time tracking'
                )
            )
            if msg.clickedButton == disableTrackingButton:
                posData = self.data[self.pos_i]
                current_frame_i = posData.frame_i
                for frame_i in range(self.start_n-1, self.stop_n):
                    posData.frame_i = frame_i
                    self.get_data()
                    self.store_data()
                posData.last_tracked_i = frame_i
                self.setNavigateScrollBarMaximum()

                # Back to current frame
                posData.frame_i = current_frame_i
                self.get_data()
        posData = self.data[self.pos_i]
        posData.updateSegmSizeT()
        self.updateALLimg()
        self.titleLabel.setText('Done', color='w')

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)

    def startTrackingWorker(self, posData, video_to_track):
        self.thread = QThread()
        self.trackingWorker = trackingWorker(posData, self, video_to_track)
        self.trackingWorker.moveToThread(self.thread)
        self.trackingWorker.finished.connect(self.thread.quit)
        self.trackingWorker.finished.connect(self.trackingWorker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.trackingWorker.signals = myutils.signals()
        self.trackingWorker.signals.progress = self.trackingWorker.progress
        self.trackingWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.trackingWorker.progress.connect(self.workerProgress)
        self.trackingWorker.critical.connect(self.workerCritical)
        self.trackingWorker.finished.connect(self.trackingWorkerFinished)

        self.trackingWorker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.trackingWorker.run)
        self.thread.start()

    def startRelabellingWorker(self, posData):
        self.thread = QThread()
        self.worker = relabelSequentialWorker(posData, self)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        # self.worker.sigRemoveItemsGUI.connect(self.removeGraphicsItemsIDs)
        self.worker.progress.connect(self.workerProgress)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.workerFinished)

        self.worker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def workerDebug(self, item):
        print(f'Updating frame {item.frame_i}')
        print(item.cca_df)
        stored_lab = item.allData_li[item.frame_i]['labels']
        apps.imshow_tk(item.lab, additional_imgs=[stored_lab])
        self.worker.waitCond.wakeAll()

    def keepToolActiveActionToggled(self, checked):
        parentToolButton = self.sender().parentWidget()
        toolName = re.findall('Toggle "(.*)"', parentToolButton.toolTip())[0]
        self.df_settings.at[toolName, 'value'] = 'keepActive'
        self.df_settings.to_csv(self.settings_csv_path)

    def determineSlideshowWinPos(self):
        screens = self.app.screens()
        self.numScreens = len(screens)
        winScreen = self.screen()

        # Center main window and determine location of slideshow window
        # depending on number of screens available
        if self.numScreens > 1:
            for screen in screens:
                if screen != winScreen:
                    winScreen = screen
                    break

        winScreenGeom = winScreen.geometry()
        winScreenCenter = winScreenGeom.center()
        winScreenCenterX = winScreenCenter.x()
        winScreenCenterY = winScreenCenter.y()
        winScreenLeft = winScreenGeom.left()
        winScreenTop = winScreenGeom.top()
        self.slideshowWinLeft = winScreenCenterX - int(850/2)
        self.slideshowWinTop = winScreenCenterY - int(800/2)

    def nonViewerEditMenuOpened(self):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            self.startBlinkingModeCB()

    def getDistantGray(self, desiredGray, bkgrGray):
        isDesiredSimilarToBkgr = (
            abs(desiredGray-bkgrGray) < 0.3
        )
        if isDesiredSimilarToBkgr:
            return 1-desiredGray
        else:
            return desiredGray

    def RGBtoGray(self, R, G, B):
        # see https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
        C_linear = (0.2126*R + 0.7152*G + 0.0722*B)/255
        if C_linear <= 0.0031309:
            gray = 12.92*C_linear
        else:
            gray = 1.055*(C_linear)**(1/2.4) - 0.055
        return gray

    def ruler_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
        else:
            self.rulerHoverON = False
            self.ax1_rulerPlotItem.setData([], [])
            self.ax1_rulerAnchorsItem.setData([], [])

    def getOptimalLabelItemColor(self, LabelItemID, desiredGray):
        img = self.img1.image
        img = img/np.max(img)
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        x, y = LabelItemID.pos()
        w, h, x, y = int(w), int(h), int(x), int(y)
        colors = img[y:y+h, x:x+w]
        bkgrGray = colors.mean(axis=(0,1))
        if isinstance(bkgrGray, np.ndarray):
            bkgrGray = self.RGBtoGray(*bkgrGray)
        optimalGray = self.getDistantGray(desiredGray, bkgrGray)
        return optimalGray

    def editImgProperties(self, checked=True):
        posData = self.data[self.pos_i]
        posData.askInputMetadata(
            len(self.data),
            ask_SizeT=True,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            save=True, singlePos=True,
            askSegm3D=False
        )

    def setHoverToolSymbolData(self, xx, yy, ScatterItems, size=None):
        for item in ScatterItems:
            if size is None:
                item.setData(xx, yy)
            else:
                item.setData(xx, yy, size=size)

    def getHoverID(self, xdata, ydata):
        ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
        posData = self.data[self.pos_i]
        lab_2D = self.get_2Dlab(posData.lab)
        ID = lab_2D[ydata, xdata]
        self.isHoverZneighID = False
        if self.isSegm3D:
            z = self.z_lab()
            SizeZ = posData.lab.shape[0]
            doNotLinkThroughZ = (
                self.brushButton.isChecked() and self.isShiftDown
            )
            if doNotLinkThroughZ:
                if self.brushHoverCenterModeAction.isChecked() or ID>0:
                    hoverID = ID
                else:
                    masked_lab = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                    hoverID = np.bincount(masked_lab).argmax()
            else:
                if z > 0:
                    ID_z_under = posData.lab[z-1, ydata, xdata]
                    if self.brushHoverCenterModeAction.isChecked() or ID_z_under>0:
                        hoverIDa = ID_z_under
                    else:
                        lab = posData.lab
                        masked_lab_a = lab[z-1, ymin:ymax, xmin:xmax][diskMask]
                        hoverIDa = np.bincount(masked_lab_a).argmax()
                else:
                    hoverIDa = 0

                if self.brushHoverCenterModeAction.isChecked() or ID>0:
                    hoverIDb = lab_2D[ydata, xdata]
                else:
                    masked_lab_b = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                    hoverIDb = np.bincount(masked_lab_b).argmax()

                if z < SizeZ-1:
                    ID_z_above = posData.lab[z+1, ydata, xdata]
                    if self.brushHoverCenterModeAction.isChecked() or ID_z_above>0:
                        hoverIDc = ID_z_above
                    else:
                        lab = posData.lab
                        masked_lab_c = lab[z+1, ymin:ymax, xmin:xmax][diskMask]
                        hoverIDc = np.bincount(masked_lab_c).argmax()
                else:
                    hoverIDc = 0

                if hoverIDa > 0:
                    hoverID = hoverIDa
                    self.isHoverZneighID = True
                elif hoverIDb > 0:
                    hoverID = hoverIDb
                elif hoverIDc > 0:
                    hoverID = hoverIDc
                    self.isHoverZneighID = True
                else:
                    hoverID = 0
        else:
            if self.brushHoverCenterModeAction.isChecked() or ID>0:
                hoverID = ID
            else:
                masked_lab = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                hoverID = np.bincount(masked_lab).argmax()

        return hoverID

    def setHoverToolSymbolColor(
            self, xdata, ydata, pen, ScatterItems, button,
            brush=None, hoverRGB=None, ID=None
        ):

        posData = self.data[self.pos_i]
        Y, X = self.get_2Dlab(posData.lab).shape
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            return

        if ID is None:
            hoverID = self.getHoverID(xdata, ydata)
        else:
            hoverID = ID

        color = button.palette().button().color().name()
        drawAbove = color == self.doublePressKeyButtonColor
        if hoverID == 0 or drawAbove:
            for item in ScatterItems:
                item.setPen(pen)
                item.setBrush(brush)
        else:
            try:
                rgb = self.img2.lut[hoverID]
                rgb = rgb if hoverRGB is None else hoverRGB
                rgbPen = np.clip(rgb*1.2, 0, 255)
                for item in ScatterItems:
                    item.setPen(*rgbPen, width=2)
                    item.setBrush(*rgb, 100)
            except IndexError:
                pass
        
        checkChangeID = (
            self.isHoverZneighID and not self.isShiftDown
            and self.lastHoverID != hoverID
        )
        if checkChangeID:
            # We are hovering an ID in z+1 or z-1
            self.restoreBrushID = hoverID
            self.changeBrushID()
        
        self.lastHoverID = hoverID

    def getCheckNormAction(self):
        normalize = False
        how = ''
        for action in self.normalizeQActionGroup.actions():
            if action.isChecked():
                how = action.text()
                normalize = True
                break
        return action, normalize, how

    def normalizeIntensities(self, img):
        action, normalize, how = self.getCheckNormAction()
        if not normalize:
            return img
        if how == 'Do not normalize. Display raw image':
            return img
        elif how == 'Convert to floating point format with values [0, 1]':
            img = myutils.uint_to_float(img)
            return img
        # elif how == 'Rescale to 8-bit unsigned integer format with values [0, 255]':
        #     img = skimage.img_as_float(img)
        #     img = (img*255).astype(np.uint8)
        #     return img
        elif how == 'Rescale to [0, 1]':
            img = skimage.img_as_float(img)
            img = skimage.exposure.rescale_intensity(img)
            return img
        elif how == 'Normalize by max value':
            img = img/np.max(img)
        return img

    def removeAlldelROIsCurrentFrame(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        rois = delROIs_info['rois'].copy()
        for roi in delROIs_info['rois']:
            self.ax2.removeItem(roi)

        # Collect garbage ROIs:
        for item in self.ax2.items:
            if isinstance(item, pg.ROI):
                self.ax2.removeItem(item)

    def removeDelROI(self, event):
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        self.store_data()
        for i in range(posData.frame_i, posData.SizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            if self.roi_to_del in delROIs_info['rois']:
                posData.frame_i = i
                idx = delROIs_info['rois'].index(self.roi_to_del)
                # Restore deleted IDs from already visited frames
                if posData.allData_li[i]['labels'] is not None:
                    if len(delROIs_info['delIDsROI'][idx]) > 1:
                        posData.lab = posData.allData_li[i]['labels']
                        self.restoreAnnotDelROI(self.roi_to_del, enforce=True)
                        posData.allData_li[i]['labels'] = posData.lab
                        self.get_data()
                        self.store_data()
                delROIs_info['rois'].pop(idx)
                delROIs_info['delMasks'].pop(idx)
                delROIs_info['delIDsROI'].pop(idx)

        # Back to current frame
        posData.frame_i = current_frame_i
        posData.lab = posData.allData_li[posData.frame_i]['labels']
        if self.labelsGrad.hideLabelsImgAction.isChecked():
            self.ax1.removeItem(self.roi_to_del)
        else:
            self.ax2.removeItem(self.roi_to_del)
        self.get_data()
        self.updateALLimg()

    def getPolygonBrush(self, yxc2, Y, X):
        # see https://en.wikipedia.org/wiki/Tangent_lines_to_circles
        y1, x1 = self.yPressAx2, self.xPressAx2
        y2, x2 = yxc2
        R = self.brushSizeSpinbox.value()
        r = R

        arcsin_den = np.sqrt((x2-x1)**2+(y2-y1)**2)
        arctan_den = (x2-x1)
        if arcsin_den!=0 and arctan_den!=0:
            beta = np.arcsin((R-r)/arcsin_den)
            gamma = -np.arctan((y2-y1)/arctan_den)
            alpha = gamma-beta
            x3 = x1 + r*np.sin(alpha)
            y3 = y1 + r*np.cos(alpha)
            x4 = x2 + R*np.sin(alpha)
            y4 = y2 + R*np.cos(alpha)

            alpha = gamma+beta
            x5 = x1 - r*np.sin(alpha)
            y5 = y1 - r*np.cos(alpha)
            x6 = x2 - R*np.sin(alpha)
            y6 = y2 - R*np.cos(alpha)

            rr_poly, cc_poly = skimage.draw.polygon([y3, y4, y6, y5],
                                                    [x3, x4, x6, x5],
                                                    shape=(Y, X))
        else:
            rr_poly, cc_poly = [], []

        self.yPressAx2, self.xPressAx2 = y2, x2
        return rr_poly, cc_poly

    def get_dir_coords(self, alfa_dir, yd, xd, shape, connectivity=1):
        h, w = shape
        y_above = yd+1 if yd+1 < h else yd
        y_below = yd-1 if yd > 0 else yd
        x_right = xd+1 if xd+1 < w else xd
        x_left = xd-1 if xd > 0 else xd
        if alfa_dir == 0:
            yy = [y_below, y_below, yd, y_above, y_above]
            xx = [xd, x_right, x_right, x_right, xd]
        elif alfa_dir == 45:
            yy = [y_below, y_below, y_below, yd, y_above]
            xx = [x_left, xd, x_right, x_right, x_right]
        elif alfa_dir == 90:
            yy = [yd, y_below, y_below, y_below, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        elif alfa_dir == 135:
            yy = [y_above, yd, y_below, y_below, y_below]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -180 or alfa_dir == 180:
            yy = [y_above, y_above, yd, y_below, y_below]
            xx = [xd, x_left, x_left, x_left, xd]
        elif alfa_dir == -135:
            yy = [y_below, yd, y_above, y_above, y_above]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -90:
            yy = [yd, y_above, y_above, y_above, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        else:
            yy = [y_above, y_above, y_above, yd, y_below]
            xx = [x_left, xd, x_right, x_right, x_right]
        if connectivity == 1:
            return yy[1:4], xx[1:4]
        else:
            return yy, xx

    def drawAutoContour(self, y2, x2):
        y1, x1 = self.autoCont_y0, self.autoCont_x0
        Dy = abs(y2-y1)
        Dx = abs(x2-x1)
        edge = self.getDisplayedCellsImg()
        if Dy != 0 or Dx != 0:
            # NOTE: numIter takes care of any lag in mouseMoveEvent
            numIter = int(round(max((Dy, Dx))))
            alfa = np.arctan2(y1-y2, x2-x1)
            base = np.pi/4
            alfa_dir = round((base * round(alfa/base))*180/np.pi)
            for _ in range(numIter):
                y1, x1 = self.autoCont_y0, self.autoCont_x0
                yy, xx = self.get_dir_coords(alfa_dir, y1, x1, edge.shape)
                a_dir = edge[yy, xx]
                if self.invertBwAction.isChecked() and self.imgCmapName == 'grey':
                    min_int = np.min(a_dir) # if int_val > ta else np.min(a_dir)
                else:
                    min_int = np.max(a_dir)
                min_i = list(a_dir).index(min_int)
                y, x = yy[min_i], xx[min_i]
                try:
                    xx, yy = self.curvHoverPlotItem.getData()
                except TypeError:
                    xx, yy = [], []
                xx = np.r_[xx, x]
                yy = np.r_[yy, y]
                try:
                    self.curvHoverPlotItem.setData(xx, yy)
                    self.curvPlotItem.setData(xx, yy)
                except TypeError:
                    pass
                self.autoCont_y0, self.autoCont_x0 = y, x
                # self.smoothAutoContWithSpline()

    def smoothAutoContWithSpline(self, n=3):
        try:
            xx, yy = self.curvHoverPlotItem.getData()
            # Downsample by taking every nth coord
            xxA, yyA = xx[::n], yy[::n]
            rr, cc = skimage.draw.polygon(yyA, xxA)
            self.autoContObjMask[rr, cc] = 1
            rp = skimage.measure.regionprops(self.autoContObjMask)
            if not rp:
                return
            obj = rp[0]
            cont = self.getObjContours(obj)
            xxC, yyC = cont[:,0], cont[:,1]
            xxA, yyA = xxC[::n], yyC[::n]
            self.xxA_autoCont, self.yyA_autoCont = xxA, yyA
            xxS, yyS = self.getSpline(xxA, yyA, per=True, appendFirst=True)
            if len(xxS)>0:
                self.curvPlotItem.setData(xxS, yyS)
        except TypeError:
            pass

    def updateIsHistoryKnown():
        """
        This function is called every time the user saves and it is used
        for updating the status of cells where we don't know the history

        There are three possibilities:

        1. The cell with unknown history is a BUD
           --> we don't know when that  bud emerged --> 'emerg_frame_i' = -1
        2. The cell with unknown history is a MOTHER cell
           --> we don't know emerging frame --> 'emerg_frame_i' = -1
               AND generation number --> we start from 'generation_num' = 2
        3. The cell with unknown history is a CELL in G1
           --> we don't know emerging frame -->  'emerg_frame_i' = -1
               AND generation number --> we start from 'generation_num' = 2
               AND relative's ID in the previous cell cycle --> 'relative_ID' = -1
        """
        pass

    def getStatusKnownHistoryBud(self, ID):
        posData = self.data[self.pos_i]
        cca_df_ID = None
        for i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            is_cell_existing = is_bud_existing = ID in cca_df_i.index
            if not is_cell_existing:
                cca_df_ID = pd.Series({
                    'cell_cycle_stage': 'S',
                    'generation_num': 0,
                    'relative_ID': -1,
                    'relationship': 'bud',
                    'emerg_frame_i': i+1,
                    'division_frame_i': -1,
                    'is_history_known': True,
                    'corrected_assignment': False
                })
                return cca_df_ID

    def setHistoryKnowledge(self, ID, cca_df):
        posData = self.data[self.pos_i]
        is_history_known = cca_df.at[ID, 'is_history_known']
        if is_history_known:
            cca_df.at[ID, 'is_history_known'] = False
            cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
            cca_df.at[ID, 'generation_num'] += 2
            cca_df.at[ID, 'emerg_frame_i'] = -1
            cca_df.at[ID, 'relative_ID'] = -1
            cca_df.at[ID, 'relationship'] = 'mother'
        else:
            cca_df.loc[ID] = posData.ccaStatus_whenEmerged[ID]

    def annotateIsHistoryKnown(self, ID):
        """
        This function is used for annotating that a cell has unknown or known
        history. Cells with unknown history are for example the cells already
        present in the first frame or cells that appear in the frame from
        outside of the field of view.

        With this function we simply set 'is_history_known' to False.
        When the users saves instead we update the entire staus of the cell
        with unknown history with the function "updateIsHistoryKnown()"
        """
        posData = self.data[self.pos_i]
        is_history_known = posData.cca_df.at[ID, 'is_history_known']
        relID = posData.cca_df.at[ID, 'relative_ID']
        if relID in posData.cca_df.index:
            relID_cca = self.getStatus_RelID_BeforeEmergence(ID, relID)

        if is_history_known:
            # Save status of ID when emerged to allow undoing
            statusID_whenEmerged = self.getStatusKnownHistoryBud(ID)
            if statusID_whenEmerged is None:
                return
            posData.ccaStatus_whenEmerged[ID] = statusID_whenEmerged

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        self.setHistoryKnowledge(ID, posData.cca_df)

        if relID in posData.cca_df.index:
            # If the cell with unknown history has a relative ID assigned to it
            # we set the cca of it to the status it had BEFORE the assignment
            posData.cca_df.loc[relID] = relID_cca

        # Update cell cycle info LabelItems
        obj_idx = posData.IDs.index(ID)
        rp_ID = posData.rp[obj_idx]
        self.drawID_and_Contour(rp_ID, drawContours=False)

        if relID in posData.IDs:
            relObj_idx = posData.IDs.index(relID)
            rp_relID = posData.rp[relObj_idx]
            self.drawID_and_Contour(rp_relID, drawContours=False)

        self.store_cca_df()

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(posData.cca_df)

        # Correct future frames
        for i in range(posData.frame_i+1, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # For some reason ID disappeared from this frame
                continue
            else:
                self.setHistoryKnowledge(ID, cca_df_i)
                if relID in IDs:
                    cca_df_i.loc[relID] = relID_cca
                self.store_cca_df(frame_i=i, cca_df=cca_df_i)


        # Correct past frames
        for i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # we reached frame where ID was not existing yet
                break
            else:
                relID = cca_df_i.at[ID, 'relative_ID']
                self.setHistoryKnowledge(ID, cca_df_i)
                if relID in IDs:
                    cca_df_i.loc[relID] = relID_cca
                self.store_cca_df(frame_i=i, cca_df=cca_df_i)

    def annotateDivision(self, cca_df, ID, relID):
        # Correct as follows:
        # If S then assign to G1 and +1 on generation number
        posData = self.data[self.pos_i]
        store = False
        cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
        gen_num_clickedID = cca_df.at[ID, 'generation_num']
        cca_df.at[ID, 'generation_num'] += 1
        cca_df.at[ID, 'division_frame_i'] = posData.frame_i
        cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
        gen_num_relID = cca_df.at[relID, 'generation_num']
        cca_df.at[relID, 'generation_num'] = gen_num_relID+1
        cca_df.at[relID, 'division_frame_i'] = posData.frame_i
        if gen_num_clickedID < gen_num_relID:
            cca_df.at[ID, 'relationship'] = 'mother'
        else:
            cca_df.at[relID, 'relationship'] = 'mother'
        store = True
        return store

    def undoDivisionAnnotation(self, cca_df, ID, relID):
        # Correct as follows:
        # If G1 then correct to S and -1 on generation number
        store = False
        cca_df.at[ID, 'cell_cycle_stage'] = 'S'
        gen_num_clickedID = cca_df.at[ID, 'generation_num']
        cca_df.at[ID, 'generation_num'] -= 1
        cca_df.at[ID, 'division_frame_i'] = -1
        cca_df.at[relID, 'cell_cycle_stage'] = 'S'
        gen_num_relID = cca_df.at[relID, 'generation_num']
        cca_df.at[relID, 'generation_num'] -= 1
        cca_df.at[relID, 'division_frame_i'] = -1
        if gen_num_clickedID < gen_num_relID:
            cca_df.at[ID, 'relationship'] = 'bud'
        else:
            cca_df.at[relID, 'relationship'] = 'bud'
        store = True
        return store

    def undoBudMothAssignment(self, ID):
        posData = self.data[self.pos_i]
        relID = posData.cca_df.at[ID, 'relative_ID']
        ccs = posData.cca_df.at[ID, 'cell_cycle_stage']
        if ccs == 'G1':
            return
        posData.cca_df.at[ID, 'relative_ID'] = -1
        posData.cca_df.at[ID, 'generation_num'] = 2
        posData.cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
        posData.cca_df.at[ID, 'relationship'] = 'mother'
        if relID in posData.cca_df.index:
            posData.cca_df.at[relID, 'relative_ID'] = -1
            posData.cca_df.at[relID, 'generation_num'] = 2
            posData.cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
            posData.cca_df.at[relID, 'relationship'] = 'mother'

        obj_idx = posData.IDs.index(ID)
        relObj_idx = posData.IDs.index(relID)
        rp_ID = posData.rp[obj_idx]
        rp_relID = posData.rp[relObj_idx]

        self.store_cca_df()

        # Update cell cycle info LabelItems
        self.drawID_and_Contour(rp_ID, drawContours=False)
        self.drawID_and_Contour(rp_relID, drawContours=False)


        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(posData.cca_df)

    def manualCellCycleAnnotation(self, ID):
        """
        This function is used for both annotating division or undoing the
        annotation. It can be called on any frame.

        If we annotate division (right click on a cell in S) then it will
        check if there are future frames to correct.
        Frames to correct are those frames where both the mother and the bud
        are annotated as S phase cells.
        In this case we assign all those frames to G1, relationship to mother,
        and +1 generation number

        If we undo the annotation (right click on a cell in G1) then it will
        correct both past and future annotated frames (if present).
        Frames to correct are those frames where both the mother and the bud
        are annotated as G1 phase cells.
        In this case we assign all those frames to G1, relationship back to
        bud, and -1 generation number
        """
        posData = self.data[self.pos_i]

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        # Correct current frame
        clicked_ccs = posData.cca_df.at[ID, 'cell_cycle_stage']
        relID = posData.cca_df.at[ID, 'relative_ID']

        if relID not in posData.IDs:
            return

        ccs_relID = posData.cca_df.at[relID, 'cell_cycle_stage']
        if clicked_ccs == 'S':
            store = self.annotateDivision(posData.cca_df, ID, relID)
            self.store_cca_df()
        else:
            store = self.undoDivisionAnnotation(posData.cca_df, ID, relID)
            self.store_cca_df()

        obj_idx = posData.IDs.index(ID)
        relObj_idx = posData.IDs.index(relID)
        rp_ID = posData.rp[obj_idx]
        rp_relID = posData.rp[relObj_idx]

        # Update cell cycle info LabelItems
        self.drawID_and_Contour(rp_ID, drawContours=False)
        self.drawID_and_Contour(rp_relID, drawContours=False)

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(posData.cca_df)


        # Correct future frames
        for i in range(posData.frame_i+1, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # For some reason ID disappeared from this frame
                continue
            else:
                ccs = cca_df_i.at[ID, 'cell_cycle_stage']
                relID = cca_df_i.at[ID, 'relative_ID']
                ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
                if clicked_ccs == 'S':
                    if ccs == 'G1':
                        # Cell is in G1 in the future again so stop annotating
                        break
                    self.annotateDivision(cca_df_i, ID, relID)
                    self.store_cca_df(frame_i=i, cca_df=cca_df_i)
                else:
                    if ccs == 'S':
                        # Cell is in S in the future again so stop undoing (break)
                        # also leave a 1 frame duration G1 to avoid a continuous
                        # S phase
                        self.annotateDivision(cca_df_i, ID, relID)
                        self.store_cca_df(frame_i=i, cca_df=cca_df_i)
                        break
                    store = self.undoDivisionAnnotation(cca_df_i, ID, relID)
                    self.store_cca_df(frame_i=i, cca_df=cca_df_i)

        # Correct past frames
        for i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if ID not in cca_df_i.index or relID not in cca_df_i.index:
                # Bud did not exist at frame_i = i
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            relID = cca_df_i.at[ID, 'relative_ID']
            ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
            if ccs == 'S':
                # We correct only those frames in which the ID was in 'G1'
                break
            else:
                store = self.undoDivisionAnnotation(cca_df_i, ID, relID)
                self.store_cca_df(frame_i=i, cca_df=cca_df_i)

    def warnMotherNotEligible(self, new_mothID, budID, i, why):
        if why == 'not_G1_in_the_future':
            err_msg = html_utils.paragraph(f"""
                The requested cell in G1 (ID={new_mothID})
                at future frame {i+1} has a bud assigned to it,
                therefore it cannot be assigned as the mother
                of bud ID {budID}.<br>
                You can assign a cell as the mother of bud ID {budID}
                only if this cell is in G1 for the
                entire life of the bud.<br><br>
                One possible solution is to click on "cancel", go to
                frame {i+1} and  assign the bud of cell {new_mothID}
                to another cell.\n'
                A second solution is to assign bud ID {budID} to cell
                {new_mothID} anyway by clicking "Apply".<br><br>
                However to ensure correctness of
                future assignments the system will delete any cell cycle
                information from frame {i+1} to the end. Therefore, you
                will have to visit those frames again.<br><br>
                The deletion of cell cycle information
                <b>CANNOT BE UNDONE!</b>
                Saved data is not changed of course.<br><br>
                Apply assignment or cancel process?
            """)
            msg = QMessageBox()
            enforce_assignment = msg.warning(
               self, 'Cell not eligible', err_msg, msg.Apply | msg.Cancel
            )
            cancel = enforce_assignment == msg.Cancel
        elif why == 'not_G1_in_the_past':
            err_msg = html_utils.paragraph(f"""
                The requested cell in G1
                (ID={new_mothID}) at past frame {i+1}
                has a bud assigned to it, therefore it cannot be
                assigned as mother of bud ID {budID}.<br>
                You can assign a cell as the mother of bud ID {budID}
                only if this cell is in G1 for the entire life of the bud.<br>
                One possible solution is to first go to frame {i+1} and
                assign the bud of cell {new_mothID} to another cell.
            """)
            msg = QMessageBox()
            msg.warning(
               self, 'Cell not eligible', err_msg, msg.Ok
            )
            cancel = None
        elif why == 'single_frame_G1_duration':
            err_msg = html_utils.paragraph(f"""
                Assigning bud ID {budID} to cell in G1
                (ID={new_mothID}) would result in no G1 phase at all between
                previous cell cycle and current cell cycle.
                This is very confusing for me, sorry.<br><br>
                The solution is to remove cell division anotation on cell
                {new_mothID} (right-click on it on current frame) and then
                annotate division on any frame before current frame number {i+1}.
                This will gurantee a G1 duration of cell {new_mothID}
                of <b>at least 1 frame</b>. Thanks.
            """)
            msg = widgets.myMessageBox()
            msg.warning(
               self, 'Cell not eligible', err_msg
            )
            cancel = None
        return cancel

    def checkMothEligibility(self, budID, new_mothID):
        """
        Check that the new mother is in G1 for the entire life of the bud
        and that the G1 duration is > than 1 frame
        """
        posData = self.data[self.pos_i]
        eligible = True

        G1_duration = 0
        # Check future frames
        for i in range(posData.frame_i, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            if cca_df_i is None:
                # ith frame was not visited yet
                break

            is_still_bud = cca_df_i.at[budID, 'relationship'] == 'bud'
            if not is_still_bud:
                break

            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1':
                cancel = self.warnMotherNotEligible(
                    new_mothID, budID, i, 'not_G1_in_the_future'
                )
                if cancel or G1_duration == 1:
                    eligible = False
                    return eligible
                else:
                    self.remove_future_cca_df(i)
                    break

            G1_duration += 1

        # Check past frames
        for i in range(posData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            is_moth_existing = new_mothID in cca_df_i.index

            if not is_moth_existing:
                # Mother not existing because it appeared from outside FOV
                break

            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1' and is_bud_existing:
                # Requested mother not in G1 in the past
                # during the life of the bud (is_bud_existing = True)
                self.warnMotherNotEligible(
                    new_mothID, budID, i, 'not_G1_in_the_past'
                )
                eligible = False
                return eligible

            if ccs != 'G1':
                # Stop counting G1 duration of the requested mother
                break

            G1_duration += 1

        if G1_duration == 1:
            # G1_duration of the mother is single frame --> not eligible
            eligible = False
            self.warnMotherNotEligible(
                new_mothID, budID, posData.frame_i, 'single_frame_G1_duration'
            )
        return eligible

    def getStatus_RelID_BeforeEmergence(self, budID, curr_mothID):
        posData = self.data[self.pos_i]
        # Get status of the current mother before it had budID assigned to it
        for i in range(posData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                if curr_mothID in cca_df_i.index:
                    return cca_df_i.loc[curr_mothID]
                else:
                    # The bud emerged together with the mother because
                    # they appeared together from outside of the fov
                    # and they were trated as new IDs bud in S0
                    return pd.Series({
                        'cell_cycle_stage': 'S',
                        'generation_num': 0,
                        'relative_ID': -1,
                        'relationship': 'bud',
                        'emerg_frame_i': i+1,
                        'division_frame_i': -1,
                        'is_history_known': True,
                        'corrected_assignment': False
                    })

    def assignBudMoth(self):
        """
        This function is used for correcting automatic mother-bud assignment.

        It can be called at any frame of the bud life.

        There are three cells involved: bud, current mother, new mother.

        Eligibility:
            - User clicked first on a bud (checked at click time)
            - User released mouse button on a cell in G1 (checked at release time)
            - The new mother MUST be in G1 for all the frames of the bud life
              --> if not warn

        Result:
            - The bud only changes relative ID to the new mother
            - The new mother changes relative ID and stage to 'S'
            - The old mother changes its entire status to the status it had
              before being assigned to the clicked bud
        """
        posData = self.data[self.pos_i]
        budID = self.get_2Dlab(posData.lab)[self.yClickBud, self.xClickBud]
        new_mothID = self.get_2Dlab(posData.lab)[self.yClickMoth, self.xClickMoth]

        if budID == new_mothID:
            return

        # Allow partial initialization of cca_df with mouse
        singleFrameCca = (
            (posData.frame_i == 0 and budID != new_mothID)
            or (self.isSnapshot and budID != new_mothID)
        )
        if singleFrameCca:
            newMothCcs = posData.cca_df.at[new_mothID, 'cell_cycle_stage']
            if not newMothCcs == 'G1':
                err_msg = (
                    'You are assigning the bud to a cell that is not in G1!'
                )
                msg = QMessageBox()
                msg.critical(
                   self, 'New mother not in G1!', err_msg, msg.Ok
                )
                return
            # Store cca_df for undo action
            undoId = uuid.uuid4()
            self.storeUndoRedoCca(0, posData.cca_df, undoId)
            currentRelID = posData.cca_df.at[budID, 'relative_ID']
            if currentRelID in posData.cca_df.index:
                posData.cca_df.at[currentRelID, 'relative_ID'] = -1
                posData.cca_df.at[currentRelID, 'generation_num'] = 2
                posData.cca_df.at[currentRelID, 'cell_cycle_stage'] = 'G1'
                currentRelObjIdx = posData.IDs.index(currentRelID)
                currentRelObj = posData.rp[currentRelObjIdx]
                self.drawID_and_Contour(currentRelObj, drawContours=False)
            posData.cca_df.at[budID, 'relationship'] = 'bud'
            posData.cca_df.at[budID, 'generation_num'] = 0
            posData.cca_df.at[budID, 'relative_ID'] = new_mothID
            posData.cca_df.at[budID, 'cell_cycle_stage'] = 'S'
            posData.cca_df.at[new_mothID, 'relative_ID'] = budID
            posData.cca_df.at[new_mothID, 'generation_num'] = 2
            posData.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'
            bud_obj_idx = posData.IDs.index(budID)
            new_moth_obj_idx = posData.IDs.index(new_mothID)
            rp_budID = posData.rp[bud_obj_idx]
            rp_new_mothID = posData.rp[new_moth_obj_idx]
            self.drawID_and_Contour(rp_budID, drawContours=False)
            self.drawID_and_Contour(rp_new_mothID, drawContours=False)
            self.store_cca_df()
            return

        curr_mothID = posData.cca_df.at[budID, 'relative_ID']

        eligible = self.checkMothEligibility(budID, new_mothID)
        if not eligible:
            return

        if curr_mothID in posData.cca_df.index:
            curr_moth_cca = self.getStatus_RelID_BeforeEmergence(
                                                         budID, curr_mothID)

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)
        # Correct current frames and update LabelItems
        posData.cca_df.at[budID, 'relative_ID'] = new_mothID
        posData.cca_df.at[budID, 'generation_num'] = 0
        posData.cca_df.at[budID, 'relative_ID'] = new_mothID
        posData.cca_df.at[budID, 'relationship'] = 'bud'
        posData.cca_df.at[budID, 'corrected_assignment'] = True
        posData.cca_df.at[budID, 'cell_cycle_stage'] = 'S'

        posData.cca_df.at[new_mothID, 'relative_ID'] = budID
        posData.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'
        posData.cca_df.at[new_mothID, 'relationship'] = 'mother'

        if curr_mothID in posData.cca_df.index:
            # Cells with UNKNOWN history has relative's ID = -1
            # which is not an existing cell
            posData.cca_df.loc[curr_mothID] = curr_moth_cca

        bud_obj_idx = posData.IDs.index(budID)
        new_moth_obj_idx = posData.IDs.index(new_mothID)
        if curr_mothID in posData.cca_df.index:
            curr_moth_obj_idx = posData.IDs.index(curr_mothID)
        rp_budID = posData.rp[bud_obj_idx]
        rp_new_mothID = posData.rp[new_moth_obj_idx]


        self.drawID_and_Contour(rp_budID, drawContours=False)
        self.drawID_and_Contour(rp_new_mothID, drawContours=False)

        if curr_mothID in posData.cca_df.index:
            rp_curr_mothID = posData.rp[curr_moth_obj_idx]
            self.drawID_and_Contour(rp_curr_mothID, drawContours=False)

        self.checkMultiBudMoth(draw=True)

        self.store_cca_df()

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(posData.cca_df)

        # Correct future frames
        for i in range(posData.frame_i+1, posData.SizeT):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            IDs = cca_df_i.index
            if budID not in IDs or new_mothID not in IDs:
                # For some reason ID disappeared from this frame
                continue

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            bud_relationship = cca_df_i.at[budID, 'relationship']
            bud_ccs = cca_df_i.at[budID, 'cell_cycle_stage']

            if bud_relationship == 'mother' and bud_ccs == 'S':
                # The bud at the ith frame budded itself --> stop
                break

            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'generation_num'] = 0
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'relationship'] = 'bud'
            cca_df_i.at[budID, 'cell_cycle_stage'] = 'S'

            newMoth_bud_ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if newMoth_bud_ccs == 'G1':
                # Assign bud to new mother only if the new mother is in G1
                # This can happen if the bud already has a G1 annotated
                cca_df_i.at[new_mothID, 'relative_ID'] = budID
                cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'
                cca_df_i.at[new_mothID, 'relationship'] = 'mother'

            if curr_mothID in cca_df_i.index:
                # Cells with UNKNOWN history has relative's ID = -1
                # which is not an existing cell
                cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i)

        # Correct past frames
        for i in range(posData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'generation_num'] = 0
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'relationship'] = 'bud'
            cca_df_i.at[budID, 'cell_cycle_stage'] = 'S'

            cca_df_i.at[new_mothID, 'relative_ID'] = budID
            cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'
            cca_df_i.at[new_mothID, 'relationship'] = 'mother'

            if curr_mothID in cca_df_i.index:
                # Cells with UNKNOWN history has relative's ID = -1
                # which is not an existing cell
                cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i)

    def getSpline(self, xx, yy, resolutionSpace=None, per=False, appendFirst=False):
        # Remove duplicates
        valid = np.where(np.abs(np.diff(xx)) + np.abs(np.diff(yy)) > 0)
        if appendFirst:
            xx = np.r_[xx[valid], xx[-1], xx[0]]
            yy = np.r_[yy[valid], yy[-1], yy[0]]
        else:
            xx = np.r_[xx[valid], xx[-1]]
            yy = np.r_[yy[valid], yy[-1]]

        # Interpolate splice
        if resolutionSpace is None:
            resolutionSpace = self.hoverLinSpace
        k = 2 if len(xx) == 3 else 3
        try:
            tck, u = scipy.interpolate.splprep(
                        [xx, yy], s=0, k=k, per=per)
            xi, yi = scipy.interpolate.splev(resolutionSpace, tck)
            return xi, yi
        except (ValueError, TypeError):
            # Catch errors where we know why splprep fails
            return [], []

    def uncheckQButton(self, button):
        # Manual exclusive where we allow to uncheck all buttons
        for b in self.checkableQButtonsGroup.buttons():
            if b != button:
                b.setChecked(False)

    def delBorderObj(self, checked):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        posData = self.data[self.pos_i]
        posData.lab = skimage.segmentation.clear_border(
            posData.lab, buffer_size=1
        )
        oldIDs = posData.IDs.copy()
        self.update_rp()
        removedIDs = [ID for ID in oldIDs if ID not in posData.IDs]
        if posData.cca_df is not None:
            posData.cca_df = posData.cca_df.drop(index=removedIDs)
        self.store_data()
        self.updateALLimg()

    def addDelROI(self, event):
        posData = self.data[self.pos_i]
        self.warnEditingWithCca_df('Delete IDs using ROI')
        roi = self.getDelROI()
        for i in range(posData.frame_i, posData.SizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            delROIs_info['rois'].append(roi)
            delROIs_info['delMasks'].append(np.zeros_like(posData.lab))
            delROIs_info['delIDsROI'].append(set())
        if self.labelsGrad.hideLabelsImgAction.isChecked():
            self.ax1.addItem(roi)
        else:
            self.ax2.addItem(roi)
        self.applyDelROIimg1(None, init=True)

    def getDelROI(self, xl=None, yb=None, w=32, h=32):
        posData = self.data[self.pos_i]
        if xl is None:
            xRange, yRange = self.ax1.viewRange()
            xl = 0 if xRange[0] < 0 else xRange[0]
            yb = 0 if yRange[0] < 0 else yRange[0]
        Y, X = self.currentLab2D.shape
        roi = pg.ROI(
            [xl, yb], [w, h],
            rotatable=False,
            removable=True,
            pen=pg.mkPen(color='r'),
            maxBounds=QRectF(QRect(0,0,X,Y))
        )

        roi.handleSize = 7

        ## handles scaling horizontally around center
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        roi.addScaleHandle([0, 0.5], [1, 0.5])

        ## handles scaling vertically from opposite edge
        roi.addScaleHandle([0.5, 0], [0.5, 1])
        roi.addScaleHandle([0.5, 1], [0.5, 0])

        ## handles scaling both vertically and horizontally
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        roi.addScaleHandle([1, 0], [0, 1])

        roi.sigRegionChanged.connect(self.delROImoving)
        roi.sigRegionChangeFinished.connect(self.delROImovingFinished)
        return roi

    def delROImoving(self, roi):
        roi.setPen(color=(255,255,0))
        # First bring back IDs if the ROI moved away
        self.restoreAnnotDelROI(roi)
        self.update_rp()
        self.setImageImg2()
        self.applyDelROIimg1(roi)

    def delROImovingFinished(self, roi):
        roi.setPen(color='r')
        self.update_rp()
        self.updateALLimg()

    def restoreAnnotDelROI(self, roi, enforce=True):
        posData = self.data[self.pos_i]
        x0, y0 = [int(c) for c in roi.pos()]
        w, h = [int(c) for c in roi.size()]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        idx = delROIs_info['rois'].index(roi)
        delMask = delROIs_info['delMasks'][idx]
        delIDs = delROIs_info['delIDsROI'][idx]
        ROImask = np.zeros(self.currentLab2D.shape, bool)
        ROImask[y0:y0+h, x0:x0+w] = True
        overlapROIdelIDs = np.unique(delMask[ROImask])
        lab2D = self.get_2Dlab(posData.lab)
        restoredIDs = set()
        for ID in delIDs:
            if ID in overlapROIdelIDs and not enforce:
                continue
            delMaskID = delMask==ID
            self.currentLab2D[delMaskID] = ID
            lab2D[delMaskID] = ID
            self.restoreDelROIimg1(delMaskID, ID)
            delMask[delMaskID] = 0
            restoredIDs.add(ID)
        delROIs_info['delIDsROI'][idx] = delIDs - restoredIDs
        self.set_2Dlab(lab2D)

    def restoreDelROIimg1(self, delMaskID, delID):
        posData = self.data[self.pos_i]
        how = self.drawIDsContComboBox.currentText()
        if how.find('nothing') != -1:
            return
        elif how.find('contours') != -1:
            idx = delID-1
            obj_idx = posData.IDs.index(delID)
            obj = posData.rp[obj_idx]
            curveID = self.ax1_ContoursCurves[idx]
            cont = self.getObjContours(obj)
            curveID.setData(
                cont[:,0], cont[:,1], pen=self.oldIDs_cpen
            )
            self.ax1_LabelItemsIDs[delID-1].setText(f'{delID}')
        elif how.find('overlay segm. masks') != -1:
            self.ax1_LabelItemsIDs[delID-1].setText(f'{delID}')
            alpha = self.imgGrad.labelsAlphaSlider.value()
            overlayRGB = self.imgRGB.copy()/255
            color = posData.lut[delID]/255
            overlay = overlayRGB[delMaskID>0]*(1.0-alpha) + color*alpha
            overlayRGB[delMaskID>0] = overlay
            overlayRGB = (np.clip(overlayRGB, 0, 1)*255).astype(np.uint8)
            self.img1.setImage(overlayRGB)

    def getDelROIlab(self):
        posData = self.data[self.pos_i]
        DelROIlab = self.get_2Dlab(posData.lab, force_z=False)
        allDelIDs = set()
        # Iterate rois and delete IDs
        for roi in posData.allData_li[posData.frame_i]['delROIs_info']['rois']:
            if roi not in self.ax2.items and roi not in self.ax1.items:
                continue
            ROImask = np.zeros(posData.lab.shape, bool)
            delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
            idx = delROIs_info['rois'].index(roi)
            delObjROImask = delROIs_info['delMasks'][idx]
            delIDsROI = delROIs_info['delIDsROI'][idx]
            x0, y0 = [int(c) for c in roi.pos()]
            w, h = [int(c) for c in roi.size()]
            ROImask[y0:y0+h, x0:x0+w] = True
            delIDs = np.unique(posData.lab[ROImask])
            if delIDs[0] == 0:
                delIDs = delIDs[1:]
            delIDsROI.update(delIDs)
            allDelIDs.update(delIDs)
            _DelROIlab = self.get_2Dlab(posData.lab).copy()
            for obj in posData.rp:
                ID = obj.label
                if ID in delIDs:
                    delObjROImask[posData.lab==ID] = ID
                    _DelROIlab[posData.lab==ID] = 0
            DelROIlab[_DelROIlab == 0] = 0
            # Keep a mask of deleted IDs to bring them back when roi moves
            delROIs_info['delMasks'][idx] = delObjROImask
            delROIs_info['delIDsROI'][idx] = delIDsROI
        return allDelIDs, DelROIlab

    def gaussBlur(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPixelSize(13)
            self.gaussWin = apps.gaussBlurDialog(self)
            self.gaussWin.setFont(font)
            self.gaussWin.show()
        else:
            self.gaussWin.close()
            self.gaussWin = None

    def diffGaussCallback(self, checked):
        if checked:
            posData = self.data[self.pos_i]
            channels = [posData.filename]
            try:
                channels.extend(list(posData.ol_data_dict.keys()))
            except Exception as e:
                pass
            self.diffGaussFilterWin = apps.diffGaussFilterDialog(
                parent=self, is3D=posData.SizeZ>1, channels=channels
            )
            if posData.SizeZ > 1:
                self.diffGaussFilterWin.initSpotmaxValues(posData)
            self.diffGaussFilterWin.sigClose.connect(
                self.diffGaussFilterWinClosed
            )
            self.diffGaussFilterWin.sigValueChanged.connect(
                self.diffGaussFilterWinValueChanged
            )
            self.diffGaussFilterWin.sigRemoveFilterClicked.connect(
                self.updateALLimg
            )
            self.diffGaussFilterWin.show()
            self.diffGaussFilterWin.valueChanged()
        else:
            self.diffGaussFilterWin.sigClose.disconnect()
            self.diffGaussFilterWin.sigValueChanged.disconnect()
            self.diffGaussFilterWin.sigRemoveFilterClicked.disconnect()
            self.diffGaussFilterWin.close()
            self.diffGaussFilterWin = None
            self.updateALLimg()

    def diffGaussFilterWinClosed(self):
        self.diffGaussFilterAction.setChecked(False)

    def getDiffGaussFilteredImg(self, imgData, sigmas):
        posData = self.data[self.pos_i]
        sigmas1, sigmas2 = sigmas
        sigma1_yx = sigmas1 if isinstance(sigmas1, float) else sigmas1[1]
        sigma2_yx = sigmas2 if isinstance(sigmas2, float) else sigmas2[1]
        if sigma1_yx>0:
            filtered1 = skimage.filters.gaussian(imgData, sigma=sigmas1)
        else:
            filtered1 = myutils.uint_to_float(imgData)

        if sigma2_yx>0:
            filtered2 = skimage.filters.gaussian(imgData, sigma=sigmas2)
        else:
            filtered2 = myutils.uint_to_float(imgData)

        resultFiltered = filtered1 - filtered2
        self.diffGaussFilteredData = resultFiltered
        return resultFiltered

    def diffGaussFilterWinValueChanged(self, sigmas, filename):
        posData = self.data[self.pos_i]
        _imgData = self.getImageDataFromFilename(filename)
        if _imgData is None:
            return
        imgData = _imgData.copy()
        filteredData = self.getDiffGaussFilteredImg(imgData, sigmas)
        if posData.SizeZ > 1:
            img = self.get_2Dimg_from_3D(filteredData)
        else:
            img = filteredData
        img = self.getImageWithCmap(img=img)
        self.updateALLimg(
            image=img, updateFilters=False, updateDiffGaussFilter=False
        )

    def edgeDetection(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPixelSize(13)
            self.edgeWin = apps.edgeDetectionDialog(self)
            self.edgeWin.setFont(font)
            self.edgeWin.show()
        else:
            self.edgeWin.close()
            self.edgeWin = None

    def entropyFilter(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPixelSize(13)
            self.entropyWin = apps.entropyFilterDialog(self)
            self.entropyWin.setFont(font)
            self.entropyWin.show()
        else:
            self.entropyWin.close()
            self.entropyWin = None

    def enableSmartTrack(self, checked):
        posData = self.data[self.pos_i]
        # Disable tracking for already visited frames

        if posData.allData_li[posData.frame_i]['labels'] is not None:
            trackingEnabled = True
        else:
            trackingEnabled = False

        if checked:
            self.UserEnforced_DisabledTracking = False
            self.UserEnforced_Tracking = False
        else:
            if trackingEnabled:
                self.UserEnforced_DisabledTracking = True
                self.UserEnforced_Tracking = False
            else:
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    def invertBw(self, checked):
        self.labelsGrad.invertBwAction.toggled.disconnect()
        self.labelsGrad.invertBwAction.setChecked(checked)
        self.labelsGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        self.imgGrad.invertBwAction.toggled.disconnect()
        self.imgGrad.invertBwAction.setChecked(checked)
        self.imgGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        if self.slideshowWin is not None:
            self.slideshowWin.is_bw_inverted = checked
            self.slideshowWin.update_img()
        self.df_settings.at['is_bw_inverted', 'value'] = 'Yes' if checked else 'No'
        self.df_settings.to_csv(self.settings_csv_path)
        if checked:
            # Light mode
            self.equalizeHistPushButton.setStyleSheet('')
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.ax2_BrushCirclePen = pg.mkPen((0,0,0), width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((0,0,0,50))
            self.ax1_oldIDcolor = [255-v for v in self.ax1_oldIDcolor]
            self.ax1_G1cellColor = [255-v for v in self.ax1_G1cellColor[:3]]
            self.ax1_G1cellColor.append(178)
            self.ax1_S_oldCellColor = [255-v for v in self.ax1_S_oldCellColor]
            self.titleColor = 'k'
        else:
            # Dark mode
            self.equalizeHistPushButton.setStyleSheet(
                'QPushButton {background-color: #282828; color: #F0F0F0;}'
            )
            self.graphLayout.setBackground('default')
            self.ax2_BrushCirclePen = pg.mkPen(width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
            self.ax1_oldIDcolor = [255-v for v in self.ax1_oldIDcolor]
            self.ax1_G1cellColor = [255-v for v in self.ax1_G1cellColor[:3]]
            self.ax1_G1cellColor.append(178)
            self.ax1_S_oldCellColor = [255-v for v in self.ax1_S_oldCellColor]
            self.titleColor = 'w'

        self.updateALLimg(updateLabelItemColor=False)

    def saveNormAction(self, action):
        how = action.text()
        self.df_settings.at['how_normIntensities', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateALLimg(only_ax1=True, updateFilters=True)

    def setLastUserNormAction(self):
        how = self.df_settings.at['how_normIntensities', 'value']
        for action in self.normalizeQActionGroup.actions():
            if action.text() == how:
                action.setChecked(True)
                break

    def editFontSize(self):
        self.fontSizeMenu.popup(QCursor.pos())

    @myutils.exception_handler
    def changeFontSize(self, action):
        self.fontSize = f'{action.text()}pt'
        self.df_settings.at['fontSize', 'value'] = self.fontSize
        self.df_settings.to_csv(self.settings_csv_path)
        LIs = zip(self.ax1_LabelItemsIDs, self.ax2_LabelItemsIDs)
        for ax1_LI, ax2_LI in LIs:
            if ax1_LI is None:
                continue
            x1, y1 = ax1_LI.pos().x(), ax1_LI.pos().y()
            if x1>0:
                w, h = ax1_LI.rect().right(), ax1_LI.rect().bottom()
                xc, yc = x1+w/2, y1+h/2
            ax1_LI.setAttr('size', self.fontSize)
            ax1_LI.setText(ax1_LI.text)
            if x1>0:
                w, h = ax1_LI.rect().right(), ax1_LI.rect().bottom()
                ax1_LI.setPos(xc-w/2, yc-h/2)
            x2, y2 = ax2_LI.pos().x(), ax2_LI.pos().y()
            if x2>0:
                w, h = ax2_LI.rect().right(), ax2_LI.rect().bottom()
                xc, yc = x2+w/2, y2+h/2
            ax2_LI.setAttr('size', self.fontSize)
            ax2_LI.setText(ax2_LI.text)
            if x2>0:
                w, h = ax2_LI.rect().right(), ax2_LI.rect().bottom()
                ax2_LI.setPos(xc-w/2, yc-h/2)

    def enableZstackWidgets(self, enabled):
        if enabled:
            myutils.setRetainSizePolicy(self.zSliceScrollBar)
            myutils.setRetainSizePolicy(self.zProjComboBox)
            myutils.setRetainSizePolicy(self.z_label)
            myutils.setRetainSizePolicy(self.zSliceOverlay_SB)
            myutils.setRetainSizePolicy(self.zProjOverlay_CB)
            myutils.setRetainSizePolicy(self.overlay_z_label)
            self.zSliceScrollBar.setDisabled(False)
            self.zProjComboBox.show()
            self.zSliceScrollBar.show()
            self.z_label.show()
        else:
            myutils.setRetainSizePolicy(self.zSliceScrollBar, retain=False)
            myutils.setRetainSizePolicy(self.zProjComboBox, retain=False)
            myutils.setRetainSizePolicy(self.z_label, retain=False)
            myutils.setRetainSizePolicy(self.zSliceOverlay_SB, retain=False)
            myutils.setRetainSizePolicy(self.zProjOverlay_CB, retain=False)
            myutils.setRetainSizePolicy(self.overlay_z_label, retain=False)
            self.zSliceScrollBar.setDisabled(True)
            self.zProjComboBox.hide()
            self.zSliceScrollBar.hide()
            self.z_label.hide()

    def reInitCca(self):
        if not self.isSnapshot:
            txt = (
                'If you decide to continue ALL cell cycle annotations from this '
                'frame to the end will be erased from current session '
                '(saved data is not touched of course)\n\n'
                'To annotate future frames again you will have to revisit them.\n\n'
                'Do you want to continue?'
            )
            msg = QMessageBox()
            reinit = msg.warning(
               self, 'Cell not eligible', txt, msg.Yes | msg.Cancel
            )
            posData = self.data[self.pos_i]
            if reinit == msg.Cancel:
                return
            # Go to previous frame without storing and then back to current
            if posData.frame_i > 0:
                posData.frame_i -= 1
                self.get_data()
                self.remove_future_cca_df(posData.frame_i)
                self.next_frame()
            else:
                posData.cca_df = self.getBaseCca_df()
                self.remove_future_cca_df(posData.frame_i)
                self.updateALLimg()
        else:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            posData = self.data[self.pos_i]
            posData.cca_df = self.getBaseCca_df()
            self.store_data()
            self.updateALLimg()

    def repeatAutoCca(self):
        # Do not allow automatic bud assignment if there are future
        # frames that already contain anotations
        posData = self.data[self.pos_i]
        next_df = posData.allData_li[posData.frame_i+1]['acdc_df']
        if next_df is not None:
            if 'cell_cycle_stage' in next_df.columns:
                msg = QMessageBox()
                warn_cca = msg.critical(
                    self, 'Future visited frames detected!',
                    'Automatic bud assignment CANNOT be performed becasue '
                    'there are future frames that already contain cell cycle '
                    'annotations. The behaviour in this case cannot be predicted.\n\n'
                    'We suggest assigning the bud manually OR use the '
                    '"Renitialize cell cycle annotations" button which properly '
                    'reinitialize future frames.',
                    msg.Ok
                )
                return

        correctedAssignIDs = (
                posData.cca_df[posData.cca_df['corrected_assignment']].index)
        NeverCorrectedAssignIDs = [ID for ID in posData.new_IDs
                                   if ID not in correctedAssignIDs]

        # Store cca_df temporarily if attempt_auto_cca fails
        posData.cca_df_beforeRepeat = posData.cca_df.copy()

        if not all(NeverCorrectedAssignIDs):
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
            if notEnoughG1Cells or not proceed:
                posData.cca_df = posData.cca_df_beforeRepeat
            else:
                self.updateALLimg()
            return

        msg = QMessageBox()
        msg.setIcon(msg.Question)
        msg.setText(
            'Do you want to automatically assign buds to mother cells for '
            'ALL the new cells in this frame (excluding cells with unknown history) '
            'OR only the cells where you never clicked on?'
        )
        msg.setDetailedText(
            f'New cells that you never touched:\n\n{NeverCorrectedAssignIDs}')
        enforceAllButton = QPushButton('ALL new cells')
        b = QPushButton('Only cells that I never corrected assignment')
        msg.addButton(b, msg.YesRole)
        msg.addButton(enforceAllButton, msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == enforceAllButton:
            notEnoughG1Cells, proceed = self.attempt_auto_cca(enforceAll=True)
        else:
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
        if notEnoughG1Cells or not proceed:
            posData.cca_df = posData.cca_df_beforeRepeat
        else:
            self.updateALLimg()

    def manualEditCca(self):
        posData = self.data[self.pos_i]
        editCcaWidget = apps.editCcaTableWidget(posData.cca_df, parent=self)
        editCcaWidget.exec_()
        if editCcaWidget.cancel:
            return
        posData.cca_df = editCcaWidget.cca_df
        self.checkMultiBudMoth()
        self.updateALLimg()

    def drawIDsContComboBox_cb(self, idx):
        self.updateALLimg(updateDiffGaussFilter=True)
        how = self.drawIDsContComboBox.currentText()
        self.df_settings.at['how_draw_annotations', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'
        IDs_and_masks = how == 'Draw IDs and overlay segm. masks'
        onlyMasks = how == 'Draw only overlay segm. masks'

        # if how.find('segm. masks') != -1:
        #     self.imgGrad.labelsAlphaMenu.setDisabled(False)
        # else:
        #     self.imgGrad.labelsAlphaMenu.setDisabled(True)

        # Clear contours if requested
        if how.find('contours') == -1 or nothing:
            allCont = zip(self.ax1_ContoursCurves,
                          self.ax2_ContoursCurves)
            for ax1ContCurve, ax2ContCurve in allCont:
                if ax1ContCurve is None:
                    continue
                if ax1ContCurve.getData()[0] is not None:
                    ax1ContCurve.setData([], [])
                if ax2ContCurve.getData()[0] is not None:
                    ax2ContCurve.setData([], [])

        # Clear LabelItems IDs if requested (draw nothing or only contours)
        if onlyCont or nothing or onlyMothBudLines:
            for _IDlabel1 in self.ax1_LabelItemsIDs:
                if _IDlabel1 is None:
                    continue
                _IDlabel1.setText('')

        # Clear mother-bud lines if Requested
        drawLines = only_ccaInfo or ccaInfo_and_cont or onlyMothBudLines
        if not drawLines:
            for BudMothLine in self.ax1_BudMothLines:
                if BudMothLine is None:
                    continue
                if BudMothLine.getData()[0] is not None:
                    BudMothLine.setData([], [])

        if self.eraserButton.isChecked():
            self.setTempImg1Eraser(None, init=True)

    def mousePressColorButton(self, event):
        posData = self.data[self.pos_i]
        items = list(posData.fluo_data_dict.keys())
        if len(items)>1:
            selectFluo = apps.QDialogListbox(
                'Select image',
                'Select which fluorescent image you want to update the color of\n',
                items, multiSelection=False, parent=self
            )
            selectFluo.exec_()
            keys = selectFluo.selectedItemsText
            key = keys[0]
            if selectFluo.cancel or not keys:
                return
            else:
                self._key = keys[0]
        else:
            self._key = items[0]
        self.overlayColorButton.selectColor()

    def setEnabledCcaToolbar(self, enabled=False):
        self.manuallyEditCcaAction.setDisabled(False)
        self.viewCcaTableAction.setDisabled(False)
        self.ccaToolBar.setVisible(enabled)
        for action in self.ccaToolBar.actions():
            button = self.ccaToolBar.widgetForAction(action)
            action.setVisible(enabled)
            button.setEnabled(enabled)

    def setEnabledEditToolbarButton(self, enabled=False):
        for action in self.segmActions:
            action.setEnabled(enabled)

        for action in self.segmActionsVideo:
            action.setEnabled(enabled)

        self.SegmActionRW.setEnabled(enabled)
        self.relabelSequentialAction.setEnabled(enabled)
        self.repeatTrackingMenuAction.setEnabled(enabled)
        self.repeatTrackingVideoAction.setEnabled(enabled)
        self.postProcessSegmAction.setEnabled(enabled)
        self.autoSegmAction.setEnabled(enabled)
        self.editToolBar.setVisible(enabled)
        mode = self.modeComboBox.currentText()
        ccaON = mode == 'Cell cycle analysis'
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            # Keep binCellButton active in cca mode
            if button==self.binCellButton and not enabled and ccaON:
                action.setVisible(True)
                button.setEnabled(True)
            else:
                action.setVisible(enabled)
                button.setEnabled(enabled)
        if not enabled:
            self.setUncheckedAllButtons()

    def setEnabledFileToolbar(self, enabled):
        for action in self.fileToolBar.actions():
            button = self.fileToolBar.widgetForAction(action)
            if action == self.openAction or action == self.newAction:
                continue
            action.setEnabled(enabled)
            button.setEnabled(enabled)

    def reconnectUndoRedo(self):
        try:
            self.undoAction.triggered.disconnect()
            self.redoAction.triggered.disconnect()
        except Exception as e:
            pass
        mode = self.modeComboBox.currentText()
        if mode == 'Segmentation and Tracking' or mode == 'Snapshot':
            self.undoAction.triggered.connect(self.undo)
            self.redoAction.triggered.connect(self.redo)
        elif mode == 'Cell cycle analysis':
            self.undoAction.triggered.connect(self.UndoCca)
        elif mode == 'Custom annotations':
            self.undoAction.triggered.connect(self.undoCustomAnnotation)
        else:
            self.undoAction.setDisabled(True)
            self.redoAction.setDisabled(True)

    def setEnabledWidgetsToolbar(self, enabled):
        self.widgetsToolBar.setVisible(enabled)
        for action in self.widgetsToolBar.actions():
            widget = self.widgetsToolBar.widgetForAction(action)
            if widget == self.disableTrackingCheckBox:
                action.setVisible(enabled)
                widget.setEnabled(enabled)

    def enableSizeSpinbox(self, enabled):
        self.brushSizeLabelAction.setVisible(enabled)
        self.brushSizeAction.setVisible(enabled)

    def reload_cb(self):
        posData = self.data[self.pos_i]
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        labData = np.load(posData.segm_npz_path)
        # Keep compatibility with .npy and .npz files
        try:
            lab = labData['arr_0'][posData.frame_i]
        except Exception as e:
            lab = labData[posData.frame_i]
        posData.segm_data[posData.frame_i] = lab.copy()
        self.get_data()
        self.tracking()
        self.updateALLimg()

    def clearComboBoxFocus(self, mode):
        # Remove focus from modeComboBox to avoid the key_up changes its value
        self.sender().clearFocus()
        try:
            self.timer.stop()
            self.modeComboBox.setStyleSheet('background-color: none')
        except Exception as e:
            pass

    def changeMode(self, idx):
        self.reconnectUndoRedo()
        posData = self.data[self.pos_i]
        mode = self.modeComboBox.itemText(idx)
        self.annotateToolbar.setVisible(False)
        if mode == 'Segmentation and Tracking':
            self.trackingMenu.setDisabled(False)
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(True)
            self.initSegmTrackMode()
            self.setEnabledEditToolbarButton(enabled=True)
            self.addExistingDelROIs()
            self.checkTrackingEnabled()
            self.setEnabledCcaToolbar(enabled=False)
            # self.drawIDsContComboBox.clear()
            # self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            for BudMothLine in self.ax1_BudMothLines:
                if BudMothLine is None:
                    continue
                if BudMothLine.getData()[0] is not None:
                    BudMothLine.setData([], [])
            if posData.cca_df is not None:
                self.store_cca_df()
        elif mode == 'Cell cycle analysis':
            proceed = self.init_cca()
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(False)
            if proceed:
                self.setEnabledEditToolbarButton(enabled=False)
                if self.isSnapshot:
                    self.editToolBar.setVisible(True)
                self.setEnabledCcaToolbar(enabled=True)
                self.removeAlldelROIsCurrentFrame()
                # self.disableTrackingCheckBox.setChecked(True)
                how = self.drawIDsContComboBox.currentText()
                if how.find('segm. masks') != -1:
                    self.drawIDsContComboBox.setCurrentText(
                        'Draw cell cycle info and overlay segm. masks'
                    )
                elif how.find('contours') != -1:
                    self.drawIDsContComboBox.setCurrentText(
                        'Draw cell cycle info and contours'
                    )
                else:
                    self.drawIDsContComboBox.setCurrentText(
                        'Draw only cell cycle info'
                    )
        elif mode == 'Viewer':
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(False)
            self.setEnabledEditToolbarButton(enabled=False)
            self.setEnabledCcaToolbar(enabled=False)
            self.removeAlldelROIsCurrentFrame()
            # self.disableTrackingCheckBox.setChecked(True)
            # currentMode = self.drawIDsContComboBox.currentText()
            # self.drawIDsContComboBox.clear()
            # self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxCcaItems)
            # self.drawIDsContComboBox.setCurrentText(currentMode)
            self.navigateScrollBar.setMaximum(posData.SizeT)
        elif mode == 'Custom annotations':
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(False)
            self.setEnabledEditToolbarButton(enabled=False)
            self.setEnabledCcaToolbar(enabled=False)
            self.removeAlldelROIsCurrentFrame()
            self.annotateToolbar.setVisible(True)
        elif mode == 'Snapshot':
            self.reconnectUndoRedo()
            self.setEnabledSnapshotMode()

    def setEnabledSnapshotMode(self):
        posData = self.data[self.pos_i]
        self.manuallyEditCcaAction.setDisabled(False)
        self.viewCcaTableAction.setDisabled(False)
        for action in self.segmActions:
            action.setDisabled(False)
        self.SegmActionRW.setDisabled(False)
        if posData.SizeT == 1:
            self.segmVideoMenu.setDisabled(True)
        self.relabelSequentialAction.setDisabled(False)
        self.trackingMenu.setDisabled(True)
        self.postProcessSegmAction.setDisabled(False)
        self.autoSegmAction.setDisabled(False)
        self.ccaToolBar.setVisible(True)
        self.editToolBar.setVisible(True)
        self.modeToolBar.setVisible(False)
        for action in self.ccaToolBar.actions():
            button = self.ccaToolBar.widgetForAction(action)
            if button == self.assignBudMothButton:
                button.setDisabled(False)
                action.setVisible(True)
            elif action == self.reInitCcaAction:
                action.setVisible(True)
            elif action == self.assignBudMothAutoAction and posData.SizeT==1:
                action.setVisible(True)
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            action.setVisible(True)
            button.setEnabled(True)
        # self.disableTrackingCheckBox.setChecked(True)
        self.disableTrackingCheckBox.setDisabled(True)
        self.repeatTrackingAction.setVisible(False)
        button = self.editToolBar.widgetForAction(self.repeatTrackingAction)
        button.setDisabled(True)
        self.setEnabledWidgetsToolbar(False)

    def launchSlideshow(self):
        posData = self.data[self.pos_i]
        self.determineSlideshowWinPos()
        if self.slideshowButton.isChecked():
            self.slideshowWin = apps.imageViewer(
                parent=self,
                button_toUncheck=self.slideshowButton,
                linkWindow=posData.SizeT > 1,
                enableOverlay=True
            )
            h = self.drawIDsContComboBox.size().height()
            self.slideshowWin.framesScrollBar.setFixedHeight(h)
            self.slideshowWin.overlayButton.setChecked(
                self.overlayButton.isChecked()
            )
            self.slideshowWin.update_img()
            self.slideshowWin.show(
                left=self.slideshowWinLeft, top=self.slideshowWinTop
            )
        else:
            self.slideshowWin.close()
            self.slideshowWin = None

    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

    def convexity_defects(self, img, eps_percent):
        img = img.astype(np.uint8)
        contours, _ = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects

    def auto_separate_bud_ID(self, ID, lab, rp, max_ID, max_i=1,
                             enforce=False, eps_percent=0.01):
        lab_ID_bool = lab == ID
        # First try separating by labelling
        lab_ID = lab_ID_bool.astype(int)
        rp_ID = skimage.measure.regionprops(lab_ID)
        setRp = self.separateByLabelling(lab_ID, rp_ID, maxID=np.max(lab))
        if setRp:
            success = True
            lab[lab_ID_bool] = lab_ID[lab_ID_bool]
            return lab, success

        cnt, defects = self.convexity_defects(lab_ID_bool, eps_percent)
        success = False
        if defects is not None:
            if len(defects) == 2:
                num_obj_watshed = 0
                # Separate only if it was a separation also with watershed method
                if num_obj_watshed > 2 or enforce:
                    defects_points = [0]*len(defects)
                    for i, defect in enumerate(defects):
                        s,e,f,d = defect[0]
                        x,y = tuple(cnt[f][0])
                        defects_points[i] = (y,x)
                    (r0, c0), (r1, c1) = defects_points
                    rr, cc, _ = skimage.draw.line_aa(r0, c0, r1, c1)
                    sep_bud_img = np.copy(lab_ID_bool)
                    sep_bud_img[rr, cc] = False
                    sep_bud_label = skimage.measure.label(
                                               sep_bud_img, connectivity=2)
                    rp_sep = skimage.measure.regionprops(sep_bud_label)
                    IDs_sep = [obj.label for obj in rp_sep]
                    areas = [obj.area for obj in rp_sep]
                    curr_ID_bud = IDs_sep[areas.index(min(areas))]
                    curr_ID_moth = IDs_sep[areas.index(max(areas))]
                    orig_sblab = np.copy(sep_bud_label)
                    # sep_bud_label = np.zeros_like(sep_bud_label)
                    sep_bud_label[orig_sblab==curr_ID_moth] = ID
                    sep_bud_label[orig_sblab==curr_ID_bud] = max_ID+max_i
                    # sep_bud_label *= (max_ID+max_i)
                    temp_sep_bud_lab = sep_bud_label.copy()
                    for r, c in zip(rr, cc):
                        if lab_ID_bool[r, c]:
                            nearest_ID = self.nearest_nonzero(
                                                    sep_bud_label, r, c)
                            temp_sep_bud_lab[r,c] = nearest_ID
                    sep_bud_label = temp_sep_bud_lab
                    sep_bud_label_mask = sep_bud_label != 0
                    # plt.imshow_tk(sep_bud_label, dots_coords=np.asarray(defects_points))
                    lab[sep_bud_label_mask] = sep_bud_label[sep_bud_label_mask]
                    max_i += 1
                    success = True
        return lab, success

    def disconnectLeftClickButtons(self):
        for button in self.LeftClickButtons:
            button.toggled.disconnect()

    def uncheckLeftClickButtons(self, sender):
        for button in self.LeftClickButtons:
            if button != sender:
                button.setChecked(False)
        self.wandControlsToolbar.setVisible(False)
        self.enableSizeSpinbox(False)

    def connectLeftClickButtons(self):
        self.brushButton.toggled.connect(self.Brush_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.wandToolButton.toggled.connect(self.wand_cb)
        self.expandLabelToolButton.toggled.connect(self.expandLabelCallback)

    def brushSize_cb(self, value):
        self.ax2_EraserCircle.setSize(value*2)
        self.ax1_BrushCircle.setSize(value*2)
        self.ax2_BrushCircle.setSize(value*2)
        self.ax1_EraserCircle.setSize(value*2)
        self.ax2_EraserX.setSize(value)
        self.ax1_EraserX.setSize(value)
        self.setDiskMask()

    def wand_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.wandToolButton)
            self.connectLeftClickButtons()
            self.wandControlsToolbar.setVisible(True)
        else:
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()
            self.wandControlsToolbar.setVisible(False)
        if self.labelsGrad.hideLabelsImgAction.isChecked():
            self.ax1.vb.autoRange()

    def restoreHoveredID(self):
        posData = self.data[self.pos_i]
        if self.ax1BrushHoverID in posData.IDs:
            obj_idx = posData.IDs.index(self.ax1BrushHoverID)
            obj = posData.rp[obj_idx]
            self.drawID_and_Contour(obj)
        elif self.ax1BrushHoverID in posData.lost_IDs:
            prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            obj_idx = [obj.label for obj in prev_rp].index(self.ax1BrushHoverID)
            obj = prev_rp[obj_idx]
            self.highlightLost_obj(obj)

    def hideItemsHoverBrush(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.currentLab2D
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        posData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2

        ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        if ID == 0:
            prev_lab = posData.allData_li[posData.frame_i-1]['labels']
            if prev_lab is None:
                self.restoreHoveredID()
                return
            ID = self.get_2Dlab(prev_lab)[ydata, xdata]

        # Restore ID previously hovered
        if ID != self.ax1BrushHoverID and not self.isMouseDragImg1:
            self.restoreHoveredID()

        # Hide items hover ID
        if ID != 0:
            try:
                contCurve = self.ax1_ContoursCurves[ID-1]
            except IndexError:
                return

            if contCurve is None:
                return

            contCurve.setData([], [])
            self.ax1_LabelItemsIDs[ID-1].setText('')
            self.ax1BrushHoverID = ID
        else:
            prev_lab = posData.allData_li[posData.frame_i-1]['labels']
            rp = posData.allData_li[posData.frame_i-1]['regionprops']
            self.ax1BrushHoverID = 0

    def updateBrushCursor(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.currentLab2D
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        posData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            size=size
        )
        self.setHoverToolSymbolColor(
            xdata, ydata, self.ax2_BrushCirclePen,
            (self.ax2_BrushCircle, self.ax1_BrushCircle),
            self.brushButton, brush=self.ax2_BrushCircleBrush
        )

    def Brush_cb(self, checked):
        if checked:
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)
            self.setBrushID()

            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.eraserButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
            self.enableSizeSpinbox(True)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.enableSizeSpinbox(False)
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

    def setDiskMask(self):
        brushSize = self.brushSizeSpinbox.value()
        # diam = brushSize*2
        # center = (brushSize, brushSize)
        # diskShape = (diam+1, diam+1)
        # diskMask = np.zeros(diskShape, bool)
        # rr, cc = skimage.draw.disk(center, brushSize+1, shape=diskShape)
        # diskMask[rr, cc] = True
        self.diskMask = skimage.morphology.disk(brushSize, dtype=bool)

    def getDiskMask(self, xdata, ydata):
        Y, X = self.currentLab2D.shape[-2:]

        brushSize = self.brushSizeSpinbox.value()
        yBottom, xLeft = ydata-brushSize, xdata-brushSize
        yTop, xRight = ydata+brushSize+1, xdata+brushSize+1

        if xLeft<0:
            if yBottom<0:
                # Disk mask out of bounds top-left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[-yBottom:, -xLeft:]
                yBottom = 0
            elif yTop>Y:
                # Disk mask out of bounds bottom-left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[0:Y-yBottom, -xLeft:]
                yTop = Y
            else:
                # Disk mask out of bounds on the left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[:, -xLeft:]
            xLeft = 0

        elif xRight>X:
            if yBottom<0:
                # Disk mask out of bounds top-right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[-yBottom:, 0:X-xLeft]
                yBottom = 0
            elif yTop>Y:
                # Disk mask out of bounds bottom-right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[0:Y-yBottom, 0:X-xLeft]
                yTop = Y
            else:
                # Disk mask out of bounds on the right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[:, 0:X-xLeft]
            xRight = X

        elif yBottom<0:
            # Disk mask out of bounds on top
            diskMask = self.diskMask.copy()
            diskMask = diskMask[-yBottom:]
            yBottom = 0

        elif yTop>Y:
            # Disk mask out of bounds on bottom
            diskMask = self.diskMask.copy()
            diskMask = diskMask[0:Y-yBottom]
            yTop = Y

        else:
            # Disk mask fully inside the image
            diskMask = self.diskMask

        return yBottom, xLeft, yTop, xRight, diskMask

    def setBrushID(self, useCurrentLab=True):
        # Make sure that the brushed ID is always a new one based on
        # already visited frames
        posData = self.data[self.pos_i]
        if useCurrentLab:
            newID = np.max(posData.lab)
        else:
            newID = 1
        for frame_i, storedData in enumerate(posData.allData_li):
            lab = storedData['labels']
            if lab is not None:
                _max = np.max(lab)
                if _max > newID:
                    newID = _max
            else:
                break

        for y, x, manual_ID in posData.editID_info:
            if manual_ID > newID:
                newID = manual_ID
        posData.brushID = newID+1

    def equalizeHist(self):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        img = self.getDisplayedCellsImg()
        self.updateALLimg()

    def curvTool_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.curvToolButton)
            self.connectLeftClickButtons()
            self.hoverLinSpace = np.linspace(0, 1, 1000)
            self.curvPlotItem = pg.PlotDataItem(pen=self.newIDs_cpen)
            self.curvHoverPlotItem = pg.PlotDataItem(pen=self.oldIDs_cpen)
            self.curvAnchors = pg.ScatterPlotItem(
                symbol='o', size=9,
                brush=pg.mkBrush((255,0,0,50)),
                pen=pg.mkPen((255,0,0), width=2),
                hoverable=True, hoverPen=pg.mkPen((255,0,0), width=3),
                hoverBrush=pg.mkBrush((255,0,0))
            )
            self.ax1.addItem(self.curvAnchors)
            self.ax1.addItem(self.curvPlotItem)
            self.ax1.addItem(self.curvHoverPlotItem)
            self.splineHoverON = True
            posData.curvPlotItems.append(self.curvPlotItem)
            posData.curvAnchorsItems.append(self.curvAnchors)
            posData.curvHoverItems.append(self.curvHoverPlotItem)
        else:
            self.splineHoverON = False
            self.isRightClickDragImg1 = False
            self.clearCurvItems()
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

    def updateHoverLabelCursor(self, x, y):
        if x is None:
            self.hoverLabelID = 0
            return

        xdata, ydata = int(x), int(y)
        Y, X = self.currentLab2D.shape
        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        ID = self.currentLab2D[ydata, xdata]
        self.hoverLabelID = ID

        if ID == 0:
            if self.highlightedID != 0:
                self.updateALLimg()
                self.highlightedID = 0
            return

        if self.app.overrideCursor() != Qt.SizeAllCursor:
            self.app.setOverrideCursor(Qt.SizeAllCursor)
        self.highlightSearchedID(ID)

    def updateEraserCursor(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.currentLab2D
        Y, X = _img.shape
        posData = self.data[self.pos_i]

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        posData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], (self.ax1_EraserCircle, self.ax2_EraserCircle),
            size=size
        )
        self.setHoverToolSymbolData(
            [x], [y], (self.ax1_EraserX, self.ax2_EraserX),
            size=int(size/2)
        )

        isMouseDrag = (
            self.isMouseDragImg1 or self.isMouseDragImg2
        )

        if not isMouseDrag:
            self.setHoverToolSymbolColor(
                xdata, ydata, self.eraserCirclePen,
                (self.ax2_EraserCircle, self.ax1_EraserCircle),
                self.eraserButton, hoverRGB=None
            )

    def Eraser_cb(self, checked):
        if checked:
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)
            self.setTempImg1Eraser(None, init=True)
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.brushButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
            self.enableSizeSpinbox(True)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.enableSizeSpinbox(False)
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()
            self.updateALLimg()

    @myutils.exception_handler
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_T:
            if self.debug:
                raise FileNotFoundError
                posData = self.data[self.pos_i]
                print(posData.ol_data.keys())
                values = posData.ol_data.values()
                values = list(values)
                print((values[0][posData.frame_i]).max())
                print(posData.img_data[posData.frame_i].max())
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return
        if ev.key() == Qt.Key_Control:
            self.isCtrlDown = True
        modifiers = ev.modifiers()
        isAltModifier = modifiers == Qt.AltModifier
        isCtrlModifier = modifiers == Qt.ControlModifier
        isShiftModifier = modifiers == Qt.ShiftModifier
        self.isZmodifier = ev.key()==Qt.Key_Z
        if isShiftModifier:
            self.isShiftDown = True
            # Force default brush symbol with shift down
            self.setHoverToolSymbolColor(
                1, 1, self.ax2_BrushCirclePen,
                (self.ax2_BrushCircle, self.ax1_BrushCircle),
                self.brushButton, brush=self.ax2_BrushCircleBrush,
                ID=0
            )
            self.changeBrushID()        
        isBrushActive = (
            self.brushButton.isChecked() or self.eraserButton.isChecked()
        )
        isExpandLabelActive = self.expandLabelToolButton.isChecked()
        isWandActive = self.wandToolButton.isChecked()
        how = self.drawIDsContComboBox.currentText()
        isOverlaySegm = how.find('overlay segm. masks') != -1
        if ev.key()==Qt.Key_Up and not isCtrlModifier:
            if isBrushActive:
                brushSize = self.brushSizeSpinbox.value()
                self.brushSizeSpinbox.setValue(brushSize+1)
            elif isWandActive:
                wandTolerance = self.wandToleranceSlider.value()
                self.wandToleranceSlider.setValue(wandTolerance+1)
            elif isExpandLabelActive:
                self.expandLabel(dilation=True)
                self.expandFootprintSize += 1
        elif ev.key()==Qt.Key_Down and not isCtrlModifier:
            if isBrushActive:
                brushSize = self.brushSizeSpinbox.value()
                self.brushSizeSpinbox.setValue(brushSize-1)
            elif isWandActive:
                wandTolerance = self.wandToleranceSlider.value()
                self.wandToleranceSlider.setValue(wandTolerance-1)
            elif isExpandLabelActive:
                self.expandLabel(dilation=False)
                self.expandFootprintSize += 1
        elif ev.key() == Qt.Key_Escape:
            self.setUncheckedAllButtons()
            if self.highlightedID != 0:
                self.highlightedID = 0
                self.guiTabControl.highlightCheckbox.setChecked(False)
                self.highlightIDcheckBoxToggled(False)
                # self.updateALLimg()
        elif isAltModifier:
            isCursorSizeAll = self.app.overrideCursor() == Qt.SizeAllCursor
            # Alt is pressed while cursor is on images --> set SizeAllCursor
            if self.xHoverImg is not None and not isCursorSizeAll:
                self.app.setOverrideCursor(Qt.SizeAllCursor)
            if ev.key() == Qt.Key_C:
                font = QtGui.QFont()
                font.setPixelSize(13)
                win = apps.QDialogEntriesWidget(
                    ['Z coord.', 'Y coord.', 'X coord.'], ['0', '0', '0'],
                    winTitle='Point coordinates',
                    parent=self, font=font
                )
                win.show()
                win.QLEs[0].setFocus()
                win.QLEs[0].selectAll()
                win.exec_()
                z, y, x = win.entriesTxt
                z, y, x = int(z), int(y), int(x)
                posData = self.data[self.pos_i]
                if posData.SizeZ > 1:
                    self.zSliceScrollBar.setSliderPosition(z)
                self.ax1_point_ScatterPlot.setData([x], [y])
        elif isCtrlModifier and isOverlaySegm:
            if ev.key() == Qt.Key_Up:
                val = self.imgGrad.labelsAlphaSlider.value()
                delta = 5/self.imgGrad.labelsAlphaSlider.maximum()
                val = val+delta
                self.imgGrad.labelsAlphaSlider.setValue(val, emitSignal=True)
            elif ev.key() == Qt.Key_Down:
                val = self.imgGrad.labelsAlphaSlider.value()
                delta = 5/self.imgGrad.labelsAlphaSlider.maximum()
                val = val-delta
                self.imgGrad.labelsAlphaSlider.setValue(val, emitSignal=True)
        elif ev.key() == Qt.Key_H:
            self.zoomToCells(enforce=True)
            if self.countKeyPress == 0:
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                self.Button = None
                QTimer.singleShot(400, self.doubleKeyTimerCallBack)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.ax1.vb.autoRange()
                self.isKeyDoublePress = True
                self.countKeyPress = 0
        elif ev.key() == Qt.Key_Space:
            if self.countKeyPress == 0:
                # Single press --> wait that it's not double press
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                QTimer.singleShot(300, self.doubleKeySpacebarTimerCallback)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                # Double press --> toggle draw nothing
                how = self.drawIDsContComboBox.currentText()
                if how.find('nothing') == -1:
                    self.prev_how = how
                    self.drawIDsContComboBox.setCurrentText('Draw nothing')
                else:
                    try:
                        self.drawIDsContComboBox.setCurrentText(self.prev_how)
                    except Exception as e:
                        # traceback.print_exc()
                        pass
                self.countKeyPress = 0
        elif ev.key() == Qt.Key_B or ev.key() == Qt.Key_X:
            mode = self.modeComboBox.currentText()
            if mode == 'Cell cycle analysis' or mode == 'Viewer':
                return
            if ev.key() == Qt.Key_B:
                self.Button = self.brushButton
            else:
                self.Button = self.eraserButton

            if self.countKeyPress == 0:
                # If first time clicking B activate brush and start timer
                # to catch double press of B
                if not self.Button.isChecked():
                    self.uncheck = False
                    self.Button.setChecked(True)
                else:
                    self.uncheck = True
                self.countKeyPress = 1
                self.isKeyDoublePress = False
                self.doubleKeyTimeElapsed = False

                QTimer.singleShot(400, self.doubleKeyTimerCallBack)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                color = self.Button.palette().button().color().name()
                if color == self.doublePressKeyButtonColor:
                    c = self.defaultToolBarButtonColor
                else:
                    c = self.doublePressKeyButtonColor
                self.Button.setStyleSheet(f'background-color: {c}')
                self.countKeyPress = 0
                if self.xHoverImg is not None:
                    xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
                    if ev.key() == Qt.Key_B:
                        self.setHoverToolSymbolColor(
                            xdata, ydata, self.ax2_BrushCirclePen,
                            (self.ax2_BrushCircle, self.ax1_BrushCircle),
                            self.brushButton, brush=self.ax2_BrushCircleBrush
                        )
                    elif ev.key() == Qt.Key_X:
                        self.setHoverToolSymbolColor(
                            xdata, ydata, self.eraserCirclePen,
                            (self.ax2_EraserCircle, self.ax1_EraserCircle),
                            self.eraserButton
                        )

    def doubleKeyTimerCallBack(self):
        if self.isKeyDoublePress:
            self.doubleKeyTimeElapsed = False
            return
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0
        if self.Button is None:
            return

        isBrushChecked = self.Button.isChecked()
        if isBrushChecked and self.uncheck:
            self.Button.setChecked(False)
        c = self.defaultToolBarButtonColor
        self.Button.setStyleSheet(f'background-color: {c}')

    def doubleKeySpacebarTimerCallback(self):
        if self.isKeyDoublePress:
            self.doubleKeyTimeElapsed = False
            return
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0

        # Spacebar single press --> toggle next visualization
        currentIndex = self.drawIDsContComboBox.currentIndex()
        nItems = self.drawIDsContComboBox.count()
        nextIndex = currentIndex+1
        if nextIndex < nItems:
            self.drawIDsContComboBox.setCurrentIndex(nextIndex)
        else:
            self.drawIDsContComboBox.setCurrentIndex(0)

    def keyReleaseEvent(self, ev):
        if self.app.overrideCursor() == Qt.SizeAllCursor:
            self.app.restoreOverrideCursor()
        if ev.key() == Qt.Key_Control:
            self.isCtrlDown = False
        elif ev.key() == Qt.Key_Shift:
            # Restore normal brush cursor when releasing shift
            xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
            self.setHoverToolSymbolColor(
                xdata, ydata, self.ax2_BrushCirclePen,
                (self.ax2_BrushCircle, self.ax1_BrushCircle),
                self.brushButton, brush=self.ax2_BrushCircleBrush
            )
            self.changeBrushID()
            self.isShiftDown = False
        elif ev.key() == Qt.Key_Z:
            self.isZmodifier = False
        canRepeat = (
            ev.key() == Qt.Key_Left
            or ev.key() == Qt.Key_Right
            or ev.key() == Qt.Key_Up
            or ev.key() == Qt.Key_Down
            or ev.key() == Qt.Key_Control
            or ev.key() == Qt.Key_Z
        )
        if canRepeat:
            return
        if ev.isAutoRepeat():
            msg = QMessageBox()
            msg.critical(
                self, 'Release the key!',
                f'Please, do not keep the key "{ev.text()}" pressed! It confuses me.\n'
                'You do not need to keep it pressed.\n\n '
                'Thanks!',
                msg.Ok
            )

    def setUncheckedAllButtons(self):
        self.clickedOnBud = False
        try:
            self.BudMothTempLine.setData([], [])
        except Exception as e:
            pass
        for button in self.checkableButtons:
            button.setChecked(False)
        self.splineHoverON = False
        self.rulerHoverON = False
        self.isRightClickDragImg1 = False
        self.clearCurvItems(removeItems=False)

    def propagateChange(self, modID, modTxt, doNotShow, UndoFutFrames,
                        applyFutFrames, applyTrackingB=False):
        """
        This function determines whether there are already visited future frames
        that contains "modID". If so, it triggers a pop-up asking the user
        what to do (propagate change to future frames o not)
        """
        posData = self.data[self.pos_i]
        # Do not check the future for the last frame
        if posData.frame_i+1 == posData.SizeT:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        areFutureIDs_affected = []
        # Get number of future frames already visited and checked if future
        # frames has an ID affected by the change
        for i in range(posData.frame_i+1, posData.SizeT):
            if posData.allData_li[i]['labels'] is None:
                i -= 1
                break
            else:
                futureIDs = np.unique(posData.allData_li[i]['labels'])
                if modID in futureIDs:
                    areFutureIDs_affected.append(True)

        if i == posData.frame_i+1:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        if not areFutureIDs_affected:
            # There are future frames but they are not affected by the change
            return UndoFutFrames, False, None, doNotShow

        # Ask what to do unless the user has previously checked doNotShowAgain
        if doNotShow:
            endFrame_i = i
            return UndoFutFrames, applyFutFrames, endFrame_i, doNotShow
        else:
            ffa = apps.FutureFramesAction_QDialog(
                    posData.frame_i+1, i, modTxt, applyTrackingB=applyTrackingB,
                    parent=self)
            ffa.exec_()
            decision = ffa.decision

            if decision is None:
                return None, None, None, doNotShow

            endFrame_i = ffa.endFrame_i
            doNotShowAgain = ffa.doNotShowCheckbox.isChecked()

            self.onlyTracking = False
            if decision == 'apply_and_reinit':
                UndoFutFrames = True
                applyFutFrames = False
            elif decision == 'apply_and_NOTreinit':
                UndoFutFrames = False
                applyFutFrames = False
            elif decision == 'apply_to_all':
                UndoFutFrames = False
                applyFutFrames = True
            elif decision == 'apply_to_range':
                UndoFutFrames = False
                applyFutFrames = True
            elif decision == 'only_tracking':
                UndoFutFrames = False
                applyFutFrames = True
                self.onlyTracking = True
        return UndoFutFrames, applyFutFrames, endFrame_i, doNotShowAgain

    def addCcaState(self, frame_i, cca_df, undoId):
        posData = self.data[self.pos_i]
        posData.UndoRedoCcaStates[frame_i].insert(0,
                                     {'id': undoId,
                                      'cca_df': cca_df.copy()})

    def addCurrentState(self):
        posData = self.data[self.pos_i]
        if posData.cca_df is not None:
            cca_df = posData.cca_df.copy()
        else:
            cca_df = None

        if hasattr(self, 'imgRGB'):
            imgRGB = self.imgRGB.copy()
        else:
            imgRGB = None

        state = {
            'image': self.img1.image.copy(),
            'imgRGB': imgRGB,
            'labels': posData.lab.copy(),
            'editID_info': posData.editID_info.copy(),
            'binnedIDs': posData.binnedIDs.copy(),
            'ripIDs': posData.ripIDs.copy(),
            'cca_df': cca_df
        }
        posData.UndoRedoStates[posData.frame_i].insert(0, state)

    def getCurrentState(self):
        posData = self.data[self.pos_i]
        i = posData.frame_i
        c = self.UndoCount
        state = posData.UndoRedoStates[i][c]
        if state['imgRGB'] is not None:
            self.imgRGB = state['imgRGB'].copy()
        image_left = state['image'].copy()
        posData.lab = state['labels'].copy()
        posData.editID_info = state['editID_info'].copy()
        posData.binnedIDs = state['binnedIDs'].copy()
        posData.ripIDs = state['ripIDs'].copy()
        cca_df = state['cca_df']
        if cca_df is not None:
            posData.cca_df = state['cca_df'].copy()
        else:
            posData.cca_df = None
        return image_left

    def storeUndoRedoStates(self, UndoFutFrames):
        posData = self.data[self.pos_i]
        if UndoFutFrames:
            # Since we modified current frame all future frames that were already
            # visited are not valid anymore. Undo changes there
            self.undo_changes_future_frames()

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCount = 0
        self.undoAction.setEnabled(True)
        self.addCurrentState()
        # Keep only 5 Undo/Redo states
        if len(posData.UndoRedoStates[posData.frame_i]) > 5:
            posData.UndoRedoStates[posData.frame_i].pop(-1)

    def storeUndoRedoCca(self, frame_i, cca_df, undoId):
        if self.isSnapshot:
            # For snapshot mode we don't store anything because we have only
            # segmentation undo action active
            return
        """
        Store current cca_df along with a unique id to know which cca_df needs
        to be restored
        """

        posData = self.data[self.pos_i]

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCcaCount = 0
        self.undoAction.setEnabled(True)

        self.addCcaState(frame_i, cca_df, undoId)

        # Keep only 10 Undo/Redo states
        if len(posData.UndoRedoCcaStates[frame_i]) > 10:
            posData.UndoRedoCcaStates[frame_i].pop(-1)

    def undoCustomAnnotation(self):
        pass

    def UndoCca(self):
        posData = self.data[self.pos_i]
        # Undo current ccaState
        storeState = False
        if self.UndoCount == 0:
            undoId = uuid.uuid4()
            self.addCcaState(posData.frame_i, posData.cca_df, undoId)
            storeState = True


        # Get previously stored state
        self.UndoCount += 1
        currentCcaStates = posData.UndoRedoCcaStates[posData.frame_i]
        prevCcaState = currentCcaStates[self.UndoCount]
        posData.cca_df = prevCcaState['cca_df']
        self.store_cca_df()
        self.updateALLimg()

        # Check if we have undone all states
        if len(currentCcaStates) > self.UndoCount:
            # There are no states left to undo for current frame_i
            self.undoAction.setEnabled(False)

        # Undo all past and future frames that has a last status inserted
        # when modyfing current frame
        prevStateId = prevCcaState['id']
        for frame_i in range(0, posData.SizeT):
            if storeState:
                cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
                if cca_df_i is None:
                    break
                # Store current state to enable redoing it
                self.addCcaState(frame_i, cca_df_i, undoId)

            CcaStates_i = posData.UndoRedoCcaStates[frame_i]
            if len(CcaStates_i) <= self.UndoCount:
                # There are no states to undo for frame_i
                continue

            CcaState_i = CcaStates_i[self.UndoCount]
            id_i = CcaState_i['id']
            if id_i != prevStateId:
                # The id of the state in frame_i is different from current frame
                continue

            cca_df_i = CcaState_i['cca_df']
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i)

    def undo(self):
        posData = self.data[self.pos_i]
        if self.UndoCount == 0:
            # Store current state to enable redoing it
            self.addCurrentState()

        # Get previously stored state
        if self.UndoCount < len(posData.UndoRedoStates[posData.frame_i])-1:
            self.UndoCount += 1
            # Since we have undone then it is possible to redo
            self.redoAction.setEnabled(True)

            # Restore state
            image_left = self.getCurrentState()
            self.update_rp()
            self.setTitleText()
            self.updateALLimg(image=image_left, overlayMasks=False)
            self.store_data()

        if not self.UndoCount < len(posData.UndoRedoStates[posData.frame_i])-1:
            # We have undone all available states
            self.undoAction.setEnabled(False)

    def redo(self):
        posData = self.data[self.pos_i]
        # Get previously stored state
        if self.UndoCount > 0:
            self.UndoCount -= 1
            # Since we have redone then it is possible to undo
            self.undoAction.setEnabled(True)

            # Restore state
            image_left = self.getCurrentState()
            self.update_rp()
            self.setTitleText()
            self.updateALLimg(image=image_left, overlayMasks=False)
            self.store_data()

        if not self.UndoCount > 0:
            # We have redone all available states
            self.redoAction.setEnabled(False)

    def disableTracking(self, isChecked):
        # Event called ONLY if the user click on Disable tracking
        # NOT called if setChecked is called. This allows to keep track
        # of the user choice. This way user con enforce tracking
        # NOTE: I know two booleans doing the same thing is overkill
        # but the code is more readable when we actually need them

        posData = self.data[self.pos_i]

        # Turn off smart tracking
        self.enableSmartTrackAction.toggled.disconnect()
        self.enableSmartTrackAction.setChecked(False)
        self.enableSmartTrackAction.toggled.connect(self.enableSmartTrack)
        if isChecked:
            self.UserEnforced_DisabledTracking = True
            self.UserEnforced_Tracking = False
        else:
            warn_txt = (

            'You requested to explicitly ENABLE tracking. This will '
            'overwrite the default behaviour of not tracking already '
            'visited/checked frames.\n\n'
            'On all future frames that you will visit tracking '
            'will be automatically performed unless you explicitly '
            'disable tracking by clicking "Disable tracking" again.\n\n'
            'To reactive smart handling of enabling/disabling tracking '
            'go to Edit --> Smart handling of enabling/disabling tracking.\n\n'
            'If you need to repeat tracking ONLY on the current frame you '
            'can use the "Repeat tracking" button on the toolbar instead.\n\n'
            'Are you sure you want to proceed with ENABLING tracking from now on?'

            )
            msg = QMessageBox(self)
            msg.setWindowTitle('Enable tracking?')
            msg.setIcon(msg.Warning)
            msg.setText(warn_txt)
            msg.addButton(msg.Yes)
            noButton = QPushButton('No')
            msg.addButton(noButton, msg.NoRole)
            msg.exec_()
            enforce_Tracking = msg.clickedButton()
            if msg.clickedButton() == noButton:
                pass
                # self.disableTrackingCheckBox.setChecked(True)
            else:
                self.repeatTracking()
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    @myutils.exception_handler
    def repeatTrackingVideo(self):
        posData = self.data[self.pos_i]
        win = apps.selectTrackerGUI(
            posData.SizeT, currentFrameNo=posData.frame_i+1
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Tracking aborted.')
            return

        trackerName = win.selectedItemsText[0]
        self.logger.info(f'Importing {trackerName} tracker...')
        trackerModule = import_module(
            f'trackers.{trackerName}.{trackerName}_tracker'
        )

        params = {}
        if trackerName == 'BayesianTracker':
            if self.isSegm3D:
                labShape = posData.lab.shape
            else:
                labShape = (1, *posData.lab.shape)
            paramsWin = apps.BayesianTrackerParamsWin(labShape, parent=self)
            paramsWin.exec_()
            if paramsWin.cancel:
                pass
            params = paramsWin.params
        elif trackerName == 'CellACDC':
            paramsWin = apps.CellACDCTrackerParamsWin(parent=self)
            paramsWin.exec_()
            params = paramsWin.params

        self.tracker = trackerModule.tracker(**params)
        start_n = win.startFrame
        stop_n = win.stopFrame
        video_to_track = posData.segm_data
        for frame_i in range(start_n-1, stop_n):
            data_dict = posData.allData_li[frame_i]
            lab = data_dict['labels']
            if lab is None:
                break

            video_to_track[frame_i] = lab
        video_to_track = video_to_track[start_n-1:stop_n]

        self.start_n = start_n
        self.stop_n = stop_n

        self.progressWin = apps.QDialogWorkerProgress(
            title='Tracking', parent=self,
            pbarDesc=f'Tracking from frame n. {start_n} to {stop_n}...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(stop_n-start_n)
        self.startTrackingWorker(posData, video_to_track)

    def repeatTracking(self):
        posData = self.data[self.pos_i]
        prev_lab = self.get_2Dlab(posData.lab).copy()
        self.tracking(enforce=True, DoManualEdit=False)
        if posData.editID_info:
            editIDinfo = [
                f'Replace ID {posData.lab[y,x]} with {newID}'
                for y, x, newID in posData.editID_info
            ]
            msg = QMessageBox(self)
            msg.setWindowTitle('Repeat tracking mode')
            msg.setIcon(msg.Question)
            msg.setText("You requested to repeat tracking but there are "
                        "the following manually edited IDs:\n\n"
                        f"{editIDinfo}\n\n"
                        "Do you want to keep these edits or ignore them?")
            keepManualEditButton = QPushButton('Keep manually edited IDs')
            msg.addButton(keepManualEditButton, msg.YesRole)
            msg.addButton(QPushButton('Ignore'), msg.NoRole)
            msg.exec_()
            if msg.clickedButton() == keepManualEditButton:
                allIDs = [obj.label for obj in posData.rp]
                lab2D = self.get_2Dlab(posData.lab)
                self.manuallyEditTracking(lab2D, allIDs)
                self.update_rp()
                self.setTitleText()
                self.highlightLostNew()
                # self.checkIDsMultiContour()
            else:
                posData.editID_info = []
        self.updateALLimg()
        if np.any(posData.lab != prev_lab):
            self.warnEditingWithCca_df('Repeat tracking')

    def autoSegm_cb(self, checked):
        if checked:
            self.askSegmParam = True
            # Ask which model
            models = myutils.get_list_of_models()
            win = apps.QDialogListbox(
                'Select model',
                'Select model to use for segmentation: ',
                models,
                multiSelection=False,
                parent=self
            )
            win.exec_()
            model_name = win.selectedItemsText[0]
            self.segmModelName = model_name
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.updateALLimg()
            self.computeSegm()
            self.askSegmParam = False
        else:
            self.segmModelName = None

    def randomWalkerSegm(self):
        # self.RWbkgrScatterItem = pg.ScatterPlotItem(
        #     symbol='o', size=2,
        #     brush=self.RWbkgrBrush,
        #     pen=self.RWbkgrPen
        # )
        # self.ax1.addItem(self.RWbkgrScatterItem)
        #
        # self.RWforegrScatterItem = pg.ScatterPlotItem(
        #     symbol='o', size=2,
        #     brush=self.RWforegrBrush,
        #     pen=self.RWforegrPen
        # )
        # self.ax1.addItem(self.RWforegrScatterItem)

        font = QtGui.QFont()
        font.setPixelSize(13)

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        self.segmModelName = 'randomWalker'
        self.randomWalkerWin = apps.randomWalkerDialog(self)
        self.randomWalkerWin.setFont(font)
        self.randomWalkerWin.show()
        self.randomWalkerWin.setSize()

    def postProcessSegm(self, checked):
        if checked:
            self.postProcessSegmWin = apps.postProcessSegmDialog(self)
            self.postProcessSegmWin.sigClosed.connect(
                self.postProcessSegmWinClosed
            )
            self.postProcessSegmWin.show()
        else:
            self.postProcessSegmWin.close()
            self.postProcessSegmWin = None

    def postProcessSegmWinClosed(self):
        self.postProcessSegmWin = None
        self.postProcessSegmAction.toggled.disconnect()
        self.postProcessSegmAction.setChecked(False)
        self.postProcessSegmAction.toggled.connect(self.postProcessSegm)

    def readSavedCustomAnnot(self):
        tempAnnot = {}
        if os.path.exists(custom_annot_path):
            self.logger.info('Loading saved custom annotations...')
            tempAnnot = load.read_json(
                custom_annot_path, logger_func=self.logger.info
            )

        posData = self.data[self.pos_i]
        self.savedCustomAnnot = {**tempAnnot, **posData.customAnnot}

    def addCustomAnnotationSavedPos(self):
        posData = self.data[self.pos_i]
        for name, annotState in posData.customAnnot.items():
            # Check if button is already present and update only annotated IDs
            buttons = [b for b in self.customAnnotDict.keys() if b.name==name]
            if buttons:
                toolButton = buttons[0]
                allAnnotedIDs = self.customAnnotDict[toolButton]['annotatedIDs']
                allAnnotedIDs[self.pos_i] = posData.customAnnotIDs.get(name, {})
                continue

            try:
                symbol = re.findall(r"\'(\w+)\'", annotState['symbol'])[0]
            except Exception as e:
                traceback.print_exc()
                symbol = 'o'
            symbolColor = QColor(*annotState['symbolColor'])
            shortcut = annotState['shortcut']
            if shortcut is not None:
                keySequence = widgets.macShortcutToQKeySequence(shortcut)
                keySequence = QKeySequence(keySequence)
            else:
                keySequence = None
            toolTip = myutils.getCustomAnnotTooltip(annotState)
            keepActive = annotState.get('keepActive', True)
            isHideChecked = annotState.get('isHideChecked', True)

            toolButton, action = self.addCustomAnnotationButton(
                symbol, symbolColor, keySequence, toolTip, name,
                keepActive, isHideChecked
            )
            allPosAnnotIDs = [
                pos.customAnnotIDs.get(name, {}) for pos in self.data
            ]
            self.customAnnotDict[toolButton] = {
                'action': action,
                'state': annotState,
                'annotatedIDs': allPosAnnotIDs
            }

            self.addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)

    def addCustomAnnotationButton(
            self, symbol, symbolColor, keySequence, toolTip, annotName,
            keepActive, isHideChecked
        ):
        toolButton = widgets.customAnnotToolButton(
            symbol, symbolColor, parent=self, keepToolActive=keepActive,
            isHideChecked=isHideChecked
        )
        toolButton.setCheckable(True)
        self.checkableQButtonsGroup.addButton(toolButton)
        if keySequence is not None:
            toolButton.setShortcut(keySequence)
        toolButton.setToolTip(toolTip)
        toolButton.name = annotName
        toolButton.toggled.connect(self.customAnnotButtonClicked)
        toolButton.sigRemoveAction.connect(self.removeCustomAnnotButton)
        toolButton.sigKeepActiveAction.connect(self.customAnnotKeepActive)
        toolButton.sigHideAction.connect(self.customAnnotHide)
        toolButton.sigModifyAction.connect(self.customAnnotModify)
        action = self.annotateToolbar.addWidget(toolButton)
        return toolButton, action

    def addCustomAnnnotScatterPlot(
            self, symbolColor, symbol, toolButton
        ):
        # Add scatter plot item
        symbolColorBrush = [0, 0, 0, 50]
        symbolColorBrush[:3] = symbolColor.getRgb()[:3]
        scatterPlotItem = pg.ScatterPlotItem()
        scatterPlotItem.setData(
            [], [], symbol=symbol, pxMode=False,
            brush=pg.mkBrush(symbolColorBrush), size=15,
            pen=pg.mkPen(width=3, color=symbolColor),
            hoverable=True, hoverBrush=pg.mkBrush(symbolColor),
            tip=None
        )
        scatterPlotItem.sigHovered.connect(self.customAnnotHovered)
        scatterPlotItem.button = toolButton
        self.customAnnotDict[toolButton]['scatterPlotItem'] = scatterPlotItem
        self.ax1.addItem(scatterPlotItem)

    def customAnnotHovered(self, scatterPlotItem, points, event):
        # Show tool tip when hovering an annotation with annotation name and ID
        vb = scatterPlotItem.getViewBox()
        if vb is None:
            return
        if len(points) > 0:
            posData = self.data[self.pos_i]
            point = points[0]
            x, y = point.pos().x(), point.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            vb.setToolTip(
                f'Annotation name: {scatterPlotItem.button.name}\n'
                f'ID = {ID}'
            )
        else:
            vb.setToolTip('')

    def addCustomAnnotation(self):
        self.readSavedCustomAnnot()

        self.addAnnotWin = apps.customAnnotationDialog(
            self.savedCustomAnnot, parent=self
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            return

        symbol = self.addAnnotWin.symbol
        symbolColor = self.addAnnotWin.state['symbolColor']
        keySequence = self.addAnnotWin.shortcutWidget.widget.keySequence
        toolTip = self.addAnnotWin.toolTip
        name = self.addAnnotWin.state['name']
        keepActive = self.addAnnotWin.state.get('keepActive', True)
        isHideChecked = self.addAnnotWin.state.get('isHideChecked', True)
        toolButton, action = self.addCustomAnnotationButton(
            symbol, symbolColor, keySequence, toolTip, name,
            keepActive, isHideChecked
        )

        self.customAnnotDict[toolButton] = {
            'action': action,
            'state': self.addAnnotWin.state,
            'annotatedIDs': [{} for _ in range(len(self.data))]
        }

        # Save custom annotation to cellacdc/temp/custom_annotations.json
        state_to_save = self.addAnnotWin.state.copy()
        state_to_save['symbolColor'] = tuple(symbolColor.getRgb())
        self.savedCustomAnnot[name] = state_to_save
        for posData in self.data:
            posData.customAnnot[name] = state_to_save
        self.saveCustomAnnot()

        # Add scatter plot item
        self.addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)

        # Add 0s column to acdc_df
        posData = self.data[self.pos_i]
        for frame_i, data_dict in enumerate(posData.allData_li):
            acdc_df = data_dict['acdc_df']
            if acdc_df is None:
                continue
            acdc_df[self.addAnnotWin.state['name']] = 0
        if posData.acdc_df is not None:
            posData.acdc_df[self.addAnnotWin.state['name']] = 0

    def viewAllCustomAnnot(self, checked):
        if not checked:
            # Clear all annotations before showing only checked
            for button in self.customAnnotDict.keys():
                self.clearScatterPlotCustomAnnotButton(button)
        self.doCustomAnnotation(0)

    def clearScatterPlotCustomAnnotButton(self, button):
        scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
        scatterPlotItem.setData([], [])

    def saveCustomAnnot(self, only_temp=False):
        if not hasattr(self, 'savedCustomAnnot'):
            return

        if not self.savedCustomAnnot:
            return

        self.logger.info('Saving custom annotations parameters...')
        # Save to cell acdc temp path
        with open(custom_annot_path, mode='w') as file:
            json.dump(self.savedCustomAnnot, file, indent=2)

        if only_temp:
            return

        # Save to pos path
        for posData in self.data:
            if not posData.customAnnot:
                if os.path.exists(posData.custom_annot_json_path):
                    try:
                        os.remove(posData.custom_annot_json_path)
                    except Exception as e:
                        self.logger.info(traceback.format_exc())
                continue
            with open(posData.custom_annot_json_path, mode='w') as file:
                json.dump(posData.customAnnot, file, indent=2)

    def customAnnotKeepActive(self, button):
        self.customAnnotDict[button]['state']['keepActive'] = button.keepToolActive

    def customAnnotHide(self, button):
        self.customAnnotDict[button]['state']['isHideChecked'] = button.isHideChecked
        clearAnnot = (
            not button.isChecked() and button.isHideChecked
            and not self.viewAllCustomAnnotAction.isChecked()
        )
        if clearAnnot:
            # User checked hide annot with the button not active --> clear
            self.clearScatterPlotCustomAnnotButton(button)
        elif not button.isChecked():
            # User uncheked hide annot with the button not active --> show
            self.doCustomAnnotation(0)

    def deleteSelectedAnnot(self, items):
        self.saveCustomAnnot(only_temp=True)

    def customAnnotModify(self, button):
        state = self.customAnnotDict[button]['state']
        self.addAnnotWin = apps.customAnnotationDialog(
            self.savedCustomAnnot, state=state
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            return

        # Rename column if existing
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        if acdc_df is not None:
            old_name = self.customAnnotDict[button]['state']['name']
            new_name = self.addAnnotWin.state['name']
            acdc_df = acdc_df.rename(columns={old_name: new_name})
            posData.allData_li[posData.frame_i]['acdc_df'] = acdc_df

        self.customAnnotDict[button]['state'] = self.addAnnotWin.state

        name = self.addAnnotWin.state['name']
        state_to_save = self.addAnnotWin.state.copy()
        symbolColor = self.addAnnotWin.state['symbolColor']
        state_to_save['symbolColor'] = tuple(symbolColor.getRgb())
        self.savedCustomAnnot[name] = self.addAnnotWin.state
        self.saveCustomAnnot()

        symbol = self.addAnnotWin.symbol
        symbolColor = self.customAnnotDict[button]['state']['symbolColor']
        button.setColor(symbolColor)
        button.update()
        symbolColorBrush = [0, 0, 0, 50]
        symbolColorBrush[:3] = symbolColor.getRgb()[:3]
        scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
        xx, yy = scatterPlotItem.getData()
        if xx is None:
            xx, yy = [], []
        scatterPlotItem.setData(
            xx, yy, symbol=symbol, pxMode=False,
            brush=pg.mkBrush(symbolColorBrush), size=15,
            pen=pg.mkPen(width=3, color=symbolColor)
        )

    def doCustomAnnotation(self, ID):
        # NOTE: pass 0 for ID to not add
        posData = self.data[self.pos_i]
        if self.viewAllCustomAnnotAction.isChecked():
            # User requested to show all annotations --> iterate all buttons
            buttons = list(self.customAnnotDict.keys())
        else:
            # Annotate if the button is active or isHideChecked is False
            buttons = [
                b for b in self.customAnnotDict.keys()
                if (b.isChecked() or not b.isHideChecked)
            ]
            if not buttons:
                return

        for button in buttons:
            annotatedIDs = self.customAnnotDict[button]['annotatedIDs'][self.pos_i]
            annotIDs_frame_i = annotatedIDs.get(posData.frame_i, [])
            if ID in annotIDs_frame_i:
                annotIDs_frame_i.remove(ID)
            elif ID != 0:
                annotIDs_frame_i.append(ID)

            annotPerButton = self.customAnnotDict[button]
            allAnnotedIDs = annotPerButton['annotatedIDs']
            posAnnotedIDs = allAnnotedIDs[self.pos_i]
            posAnnotedIDs[posData.frame_i] = annotIDs_frame_i

            state = self.customAnnotDict[button]['state']
            acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
            if acdc_df is None:
                # visiting new frame for single time-point annot type do nothing
                return

            acdc_df[state['name']] = 0

            xx, yy = [], []
            for annotID in annotIDs_frame_i:
                obj_idx = posData.IDs.index(annotID)
                obj = posData.rp[obj_idx]
                y, x = self.getObjCentroid(obj.centroid)
                xx.append(x)
                yy.append(y)
                acdc_df.at[annotID, state['name']] = 1

            scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
            scatterPlotItem.setData(xx, yy)

            posData.allData_li[posData.frame_i]['acdc_df'] = acdc_df

        return buttons[0]

    def removeCustomAnnotButton(self, button):
        name = self.customAnnotDict[button]['state']['name']
        # remove annotation from position
        for posData in self.data:
            posData.customAnnot.pop(name)
            if posData.acdc_df is not None:
                posData.acdc_df = posData.acdc_df.drop(
                    columns=name, errors='ignore'
                )
                for frame_i, data_dict in enumerate(posData.allData_li):
                    acdc_df = data_dict['acdc_df']
                    if acdc_df is None:
                        continue
                    acdc_df = acdc_df.drop(columns=name, errors='ignore')
                    posData.allData_li[frame_i]['acdc_df'] = acdc_df

        self.clearScatterPlotCustomAnnotButton(button)

        action = self.customAnnotDict[button]['action']
        self.annotateToolbar.removeAction(action)
        self.checkableQButtonsGroup.removeButton(button)
        self.customAnnotDict.pop(button)
        # self.savedCustomAnnot.pop(name)

        self.saveCustomAnnot()

    def customAnnotButtonClicked(self, checked):
        if checked:
            self.customAnnotButton = self.sender()
            # Uncheck the other buttons
            for button in self.customAnnotDict.keys():
                if button == self.sender():
                    continue

                button.toggled.disconnect()
                button.setChecked(False)
                button.toggled.connect(self.customAnnotButtonClicked)
            self.doCustomAnnotation(0)
        else:
            self.customAnnotButton = None
            button = self.sender()
            if self.viewAllCustomAnnotAction.isChecked():
                return
            if not button.isHideChecked:
                return
            self.clearScatterPlotCustomAnnotButton(button)

    def segmFrameCallback(self, action):
        idx = self.segmActions.index(action)
        model_name = self.modelNames[idx]
        self.repeatSegm(model_name=model_name, askSegmParams=True)

    def segmVideoCallback(self, action):
        posData = self.data[self.pos_i]
        win = apps.startStopFramesDialog(
            posData.SizeT, currentFrameNum=posData.frame_i+1
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Segmentation on multiple frames aborted.')
            return

        idx = self.segmActionsVideo.index(action)
        model_name = self.modelNames[idx]
        self.repeatSegmVideo(model_name, win.startFrame, win.stopFrame)

    @myutils.exception_handler
    def repeatSegm(self, model_name='', askSegmParams=False):
        idx = self.modelNames.index(model_name)
        # Ask segm parameters if not already set
        # and not called by segmSingleFrameMenu (askSegmParams=False)
        if not askSegmParams:
            askSegmParams = self.segment2D_kwargs is None

        self.downloadWin = apps.downloadModel(model_name, parent=self)
        self.downloadWin.download()

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        posData = self.data[self.pos_i]
        # Check if model needs to be imported
        acdcSegment = self.acdcSegment_li[idx]
        if acdcSegment is None:
            self.logger.info(f'Importing {model_name}...')
            acdcSegment = import_module(f'models.{model_name}.acdcSegment')
            self.acdcSegment_li[idx] = acdcSegment

        # Ask parameters if the user clicked on the action
        # Otherwise this function is called by "computeSegm" function and
        # we use loaded parameters
        if askSegmParams:
            self.segmModelName = model_name
            # Read all models parameters
            init_params, segment_params = myutils.getModelArgSpec(acdcSegment)
            # Prompt user to enter the model parameters
            try:
                url = acdcSegment.help_url
            except AttributeError:
                url = None

            win = apps.QDialogModelParams(
                init_params,
                segment_params,
                model_name, parent=self,
                url=url
            )
            win.exec_()
            if win.cancel:
                self.logger.info('Segmentation process cancelled.')
                self.titleLabel.setText('Segmentation process cancelled.')
                return

            self.segment2D_kwargs = win.segment2D_kwargs
            self.minSize = win.minSize
            self.minSolidity = win.minSolidity
            self.maxElongation = win.maxElongation
            self.applyPostProcessing = win.applyPostProcessing

            model = acdcSegment.Model(**win.init_kwargs)
            self.models[idx] = model

            postProcessParams = {
                'model': model_name,
                'minSize': self.minSize,
                'minSolidity': self.minSolidity,
                'maxElongation': self.maxElongation,
                'applied_postprocessing': self.applyPostProcessing
            }
            posData.saveSegmHyperparams(self.segment2D_kwargs, postProcessParams)
        else:
            model = self.models[idx]

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        self.segment3D = False
        if self.isSegm3D and self.askRepeatSegment3D:
            msg = widgets.myMessageBox(showCentered=False)
            msg.addDoNotShowAgainCheckbox(text='Do not ask again')
            txt = html_utils.paragraph(
                'Do you want to segment the <b>entire z-stack</b> or only the '
                '<b>current z-slice</b>?'
            )
            _, segment3DButton, _ = msg.question(
                self, '3D segmentation?', txt,
                buttonsTexts=('Cancel', 'Segment 3D z-stack', 'Segment 2D z-slice')
            )
            if msg.cancel:
                self.titleLabel.setText('Segmentation process aborted.')
                self.logger.info('Segmentation process aborted.')
                return
            self.segment3D = msg.clickedButton == segment3DButton
            if msg.doNotShowAgainCheckbox.isChecked():
                self.askRepeatSegment3D = False

        self.model = model

        self.thread = QThread()
        self.worker = workers.segmWorker(self)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.segmWorkerFinished)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    @myutils.exception_handler
    def repeatSegmVideo(self, model_name, startFrameNum, stopFrameNum):
        idx = self.modelNames.index(model_name)

        self.downloadWin = apps.downloadModel(model_name, parent=self)
        self.downloadWin.download()

        posData = self.data[self.pos_i]
        # Check if model needs to be imported
        acdcSegment = self.acdcSegment_li[idx]
        if acdcSegment is None:
            self.logger.info(f'Importing {model_name}...')
            acdcSegment = import_module(f'models.{model_name}.acdcSegment')
            self.acdcSegment_li[idx] = acdcSegment

        # Read all models parameters
        init_params, segment_params = myutils.getModelArgSpec(acdcSegment)
        # Prompt user to enter the model parameters
        try:
            url = acdcSegment.help_url
        except AttributeError:
            url = None

        win = apps.QDialogModelParams(
            init_params,
            segment_params,
            model_name, parent=self,
            url=url
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return

        model = acdcSegment.Model(**win.init_kwargs)

        self.undo_changes_future_frames(from_frame_i=startFrameNum-1)

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        self.progressWin = apps.QDialogWorkerProgress(
            title='Segmenting video', parent=self,
            pbarDesc=f'Segmenting from frame n. {startFrameNum} to {stopFrameNum}...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(stopFrameNum-startFrameNum)

        self.thread = QThread()
        self.worker = workers.segmVideoWorker(
            posData, win, model, startFrameNum, stopFrameNum
        )
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.segmVideoWorkerFinished)
        self.worker.progressBar.connect(self.workerUpdateProgressbar)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def segmVideoWorkerFinished(self, exec_time):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        posData = self.data[self.pos_i]

        self.get_data()
        self.tracking(enforce=True)
        self.updateALLimg()

        txt = f'Done. Segmentation computed in {exec_time:.3f} s'
        self.logger.info('-----------------')
        self.logger.info(txt)
        self.logger.info('=================')
        self.titleLabel.setText(txt, color='g')

    @myutils.exception_handler
    def workerCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.thread.quit()
        raise error

    def debugSegmWorker(self, lab):
        apps.imshow_tk(lab)

    def segmWorkerFinished(self, lab, exec_time):
        posData = self.data[self.pos_i]

        if posData.segmInfo_df is not None and posData.SizeZ>1:
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, 'resegmented_in_gui'] = True

        if lab.ndim == 2 and self.isSegm3D:
            self.set_2Dlab(lab)
        else:
            posData.lab = lab.copy()
        self.update_rp()
        self.tracking(enforce=True)
        self.updateALLimg()
        self.ax1.vb.autoRange()
        self.warnEditingWithCca_df('Repeat segmentation')

        txt = f'Done. Segmentation computed in {exec_time:.3f} s'
        self.logger.info('-----------------')
        self.logger.info(txt)
        self.logger.info('=================')
        self.titleLabel.setText(txt, color='g')
        self.checkIfAutoSegm()

    def getDisplayedCellsImg(self):
        how = self.drawIDsContComboBox.currentText()
        if self.overlayButton.isChecked():
            img = self.ol_cells_img
        else:
            img = self.img_layer0

        if self.imgCmapName == 'grey' and self.invertBwAction.isChecked():
            # Neural network requires non-inverted bw images --> invert back
            img = -img+np.max(img)

        return img

    def autoAssignBud_YeastMate(self):
        if not self.is_win:
            txt = (
                'YeastMate is available only on Windows OS.'
                'We are working on expading support also on macOS and Linux.\n\n'
                'Thank you for your patience!'
            )
            msg = QMessageBox()
            msg.critical(
                self, 'Supported only on Windows', txt, msg.Ok
            )
            return


        model_name = 'YeastMate'
        idx = self.modelNames.index(model_name)

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        t0 = time.time()
        posData = self.data[self.pos_i]
        # Check if model needs to be imported
        acdcSegment = self.acdcSegment_li[idx]
        if acdcSegment is None:
            acdcSegment = import_module(f'models.{model_name}.acdcSegment')
            self.acdcSegment_li[idx] = acdcSegment

        # Read all models parameters
        init_params, segment_params = myutils.getModelArgSpec(acdcSegment)
        # Prompt user to enter the model parameters
        try:
            url = acdcSegment.url_help()
        except AttributeError:
            url = None

        win = apps.QDialogModelParams(
            init_params,
            segment_params,
            model_name, url=url)
        win.exec_()
        if win.cancel:
            self.titleLabel.setText('Segmentation aborted.')
            return

        self.segment2D_kwargs = win.segment2D_kwargs
        model = acdcSegment.Model(**win.init_kwargs)
        self.models[idx] = model

        img = self.getDisplayedCellsImg()
        # if self.gaussWin is None:
        #     img = skimage.filters.gaussian(img, sigma=1)
        # img = skimage.exposure.equalize_adapthist(skimage.img_as_float(img))

        posData.cca_df = model.predictCcaState(img, posData.lab)
        self.store_data()
        self.updateALLimg()

        self.titleLabel.setText('Budding event prediction done.', color='g')

    def next_cb(self):
        if self.isZmodifier:
            stepAddAction = QAbstractSlider.SliderSingleStepAdd
            self.zSliceScrollBar.triggerAction(stepAddAction)
            return

        if self.isSnapshot:
            self.next_pos()
        else:
            self.next_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)

    def prev_cb(self):
        if self.isZmodifier:
            stepSubAction = QAbstractSlider.SliderSingleStepSub
            self.zSliceScrollBar.triggerAction(stepSubAction)
            return

        if self.isSnapshot:
            self.prev_pos()
        else:
            self.prev_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)

    def zoomOut(self):
        self.ax1.vb.autoRange()

    def zoomToObjsActionCallback(self):
        self.zoomToCells(enforce=True)

    def zoomToCells(self, enforce=False):
        if not self.enableAutoZoomToCellsAction.isChecked() and not enforce:
            return

        posData = self.data[self.pos_i]
        lab_mask = (posData.lab>0).astype(np.uint8)
        rp = skimage.measure.regionprops(lab_mask)
        if not rp:
            Y, X = self.get_2Dlab(posData.lab).shape
            xRange = -0.5, X+0.5
            yRange = -0.5, Y+0.5
        else:
            obj = rp[0]
            min_row, min_col, max_row, max_col = self.getObjBbox(obj.bbox)
            xRange = min_col-10, max_col+10
            yRange = max_row+10, min_row-10

        self.ax1.setRange(xRange=xRange, yRange=yRange)

    def viewCcaTable(self):
        posData = self.data[self.pos_i]
        self.logger.info('========================')
        self.logger.info('CURRENT Cell cycle analysis table:')
        self.logger.info(posData.cca_df)
        self.logger.info('------------------------')
        self.logger.info(f'STORED Cell cycle analysis table for frame {posData.frame_i+1}:')
        df = posData.allData_li[posData.frame_i]['acdc_df']
        if 'cell_cycle_stage' in df.columns:
            cca_df = df[self.cca_df_colnames]
            self.logger.info(cca_df)
            cca_df = cca_df.merge(posData.cca_df, how='outer',
                                  left_index=True, right_index=True,
                                  suffixes=('_STORED', '_CURRENT'))
            cca_df = cca_df.reindex(sorted(cca_df.columns), axis=1)
            num_cols = len(cca_df.columns)
            for j in range(0,num_cols,2):
                df_j_x = cca_df.iloc[:,j]
                df_j_y = cca_df.iloc[:,j+1]
                if any(df_j_x!=df_j_y):
                    self.logger.info('------------------------')
                    self.logger.info('DIFFERENCES:')
                    diff_df = cca_df.iloc[:,j:j+2]
                    diff_mask = diff_df.iloc[:,0]!=diff_df.iloc[:,1]
                    self.logger.info(diff_df[diff_mask])
        else:
            cca_df = None
            self.logger.info(cca_df)
        self.logger.info('========================')
        if posData.cca_df is None:
            return
        df = posData.cca_df.reset_index()
        if self.ccaTableWin is None:
            self.ccaTableWin = apps.pdDataFrameWidget(df, parent=self)
            self.ccaTableWin.show()
            self.ccaTableWin.setGeometryWindow()
        else:
            self.ccaTableWin.setFocus(True)
            self.ccaTableWin.activateWindow()
            self.ccaTableWin.updateTable(posData.cca_df)

    def updateScrollbars(self):
        self.updateItemsMousePos()
        self.updateFramePosLabel()
        posData = self.data[self.pos_i]
        pos = self.pos_i+1 if self.isSnapshot else posData.frame_i+1
        self.navigateScrollBar.setSliderPosition(pos)
        if posData.SizeZ > 1:
            idx = (posData.filename, posData.frame_i)
            z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            self.zSliceScrollBar.setSliderPosition(z)
            how = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
            self.zProjComboBox.setCurrentText(how)
            self.zSliceScrollBar.setMaximum(posData.SizeZ-1)

    def updateItemsMousePos(self):
        if self.brushButton.isChecked():
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)

        if self.eraserButton.isChecked():
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)

    @myutils.exception_handler
    def postProcessing(self):
        if self.postProcessSegmWin is not None:
            self.postProcessSegmWin.setPosData()
            posData = self.data[self.pos_i]
            lab, delIDs = self.postProcessSegmWin.apply()
            if posData.allData_li[posData.frame_i]['labels'] is None:
                posData.lab = lab.copy()
                self.update_rp()
            else:
                posData.allData_li[posData.frame_i]['labels'] = lab
                self.get_data()

    def updateSharpenFilterWindows(self, prev_pos_i):
        if self.diffGaussFilterWin is None:
            return
        oldFilename = self.diffGaussFilterWin.channelsComboBox.currentText()
        prevPosData = self.data[prev_pos_i]
        filterEndName = oldFilename[len(prevPosData.basename)]

        posData = self.data[self.pos_i]
        loadedFilenames = [posData.filename]
        if posData.fluo_data_dict:
            loadedFilenames.extend(posData.fluo_data_dict.keys())

        self.diffGaussFilterWin.sigValueChanged.disconnect()
        self.diffGaussFilterWin.channelsComboBox.clear()
        self.diffGaussFilterWin.channelsComboBox.addItems(loadedFilenames)
        if filterEndName is not None:
            for loadedFilename in loadedFilenames:
                if loadedFilename.endswith(filterEndName):
                    self.diffGaussFilterWin.channelsComboBox.setCurrentText(
                        loadedFilename
                    )
        self.diffGaussFilterWin.sigValueChanged.connect(
            self.diffGaussFilterWinValueChanged
        )

    def next_pos(self):
        self.store_data(debug=False)
        prev_pos_i = self.pos_i
        if self.pos_i < self.num_pos-1:
            self.pos_i += 1
        else:
            self.logger.info('You reached last position.')
            self.pos_i = 0
        self.addCustomAnnotationSavedPos()
        self.setImageNameText()
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.postProcessing()
        self.updateSharpenFilterWindows(prev_pos_i)
        self.updateALLimg(updateFilters=True, updateLabelItemColor=False)
        self.zoomToCells()
        self.updateScrollbars()
        self.computeSegm()

    def prev_pos(self):
        self.store_data(debug=False)
        prev_pos_i = self.pos_i
        if self.pos_i > 0:
            self.pos_i -= 1
        else:
            self.logger.info('You reached first position.')
            self.pos_i = self.num_pos-1
        self.addCustomAnnotationSavedPos()
        self.setImageNameText()
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.postProcessing()
        self.updateSharpenFilterWindows(prev_pos_i)
        self.updateALLimg(updateFilters=True)
        self.zoomToCells()
        self.updateScrollbars()

    def updateViewerWindow(self):
        if self.slideshowWin is None:
            return

        if self.slideshowWin.linkWindow is None:
            return

        if not self.slideshowWin.linkWindowCheckbox.isChecked():
            return

        posData = self.data[self.pos_i]
        self.slideshowWin.frame_i = posData.frame_i
        self.slideshowWin.update_img()

    def next_frame(self):
        mode = str(self.modeComboBox.currentText())
        isSegmMode =  mode == 'Segmentation and Tracking'
        posData = self.data[self.pos_i]
        if posData.frame_i < posData.SizeT-1:
            if 'lost' in self.titleLabel.text and isSegmMode:
                if self.warnLostCellsAction.isChecked():
                    msg = widgets.myMessageBox()
                    warn_msg = html_utils.paragraph(
                        'Current frame (compared to previous frame) '
                        'has <b>lost the following cells</b>:<br><br>'
                        f'{posData.lost_IDs}<br><br>'
                        'Are you <b>sure</b> you want to continue?<br>'
                    )
                    checkBox = QCheckBox('Do not show again')
                    noButton, yesButton = msg.warning(
                        self, 'Lost cells!', warn_msg,
                        buttonsTexts=('No', 'Yes'),
                        widgets=checkBox
                    )
                    doNotWarnLostCells = not checkBox.isChecked()
                    self.warnLostCellsAction.setChecked(doNotWarnLostCells)
                    if msg.clickedButton == noButton:
                        return
            if 'multiple' in self.titleLabel.text and mode != 'Viewer':
                msg = QMessageBox()
                warn_msg = (
                    'Current frame contains cells with MULTIPLE contours '
                    '(see title message above the images)!\n\n'
                    'This is potentially an issue indicating that two distant cells '
                    'have been merged.\n\n'
                    'Are you sure you want to continue?'
                )
                proceed_with_multi = msg.warning(
                   self, 'Multiple contours detected!', warn_msg, msg.Yes | msg.No
                )
                if proceed_with_multi == msg.No:
                    return

            if posData.frame_i <= 0 and mode == 'Cell cycle analysis':
                IDs = [obj.label for obj in posData.rp]
                editCcaWidget = apps.editCcaTableWidget(
                    posData.cca_df, parent=self,
                    title='Initialize cell cycle annotations'
                )
                editCcaWidget.exec_()
                if editCcaWidget.cancel:
                    return
                if posData.cca_df is not None:
                    is_cca_same_as_stored = (
                        (posData.cca_df == editCcaWidget.cca_df).all(axis=None)
                    )
                    if not is_cca_same_as_stored:
                        reinit_cca = self.warnEditingWithCca_df(
                            'Reinitialize cell cyle annotations first frame',
                            return_answer=True
                        )
                        if reinit_cca:
                            self.remove_future_cca_df(0)
                posData.cca_df = editCcaWidget.cca_df
                self.store_cca_df()

            # Store data for current frame
            if mode != 'Viewer':
                self.store_data(debug=False)
            # Go to next frame
            posData.frame_i += 1
            self.removeAlldelROIsCurrentFrame()
            proceed_cca, never_visited = self.get_data()
            if not proceed_cca:
                posData.frame_i -= 1
                self.get_data()
                return
            self.postProcessing()
            self.tracking(storeUndo=True)
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
            if notEnoughG1Cells or not proceed:
                posData.frame_i -= 1
                self.get_data()
                return
            self.updateALLimg(
                never_visited=never_visited,
                updateFilters=True,
                updateLabelItemColor=False
            )
            self.updateViewerWindow()
            self.setNavigateScrollBarMaximum()
            self.updateScrollbars()
            self.computeSegm()
            self.zoomToCells()
        else:
            # Store data for current frame
            self.store_data()
            msg = 'You reached the last segmented frame!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)

    def setNavigateScrollBarMaximum(self):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Segmentation and Tracking':
            if posData.last_tracked_i is not None:
                if posData.frame_i > posData.last_tracked_i:
                    self.navigateScrollBar.setMaximum(posData.frame_i+1)
                else:
                    self.navigateScrollBar.setMaximum(posData.last_tracked_i+1)
            else:
                self.navigateScrollBar.setMaximum(posData.frame_i+1)
        elif mode == 'Cell cycle analysis':
            if posData.frame_i > self.last_cca_frame_i:
                self.navigateScrollBar.setMaximum(posData.frame_i+1)

    def prev_frame(self):
        posData = self.data[self.pos_i]
        if posData.frame_i > 0:
            self.store_data()
            self.removeAlldelROIsCurrentFrame()
            posData.frame_i -= 1
            _, never_visited = self.get_data()
            self.postProcessing()
            self.tracking()
            self.updateALLimg(never_visited=never_visited,
                              updateSharp=True, updateBlur=True,
                              updateEntropy=True)
            self.updateScrollbars()
            self.zoomToCells()
            self.updateViewerWindow()
        else:
            msg = 'You reached the first frame!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)

    def loadSelectedData(self, user_ch_file_paths, user_ch_name):
        data = []
        numPos = len(user_ch_file_paths)
        self.user_ch_file_paths = user_ch_file_paths

        required_ram = myutils.getMemoryFootprint(user_ch_file_paths)
        proceed = self.checkMemoryRequirements(required_ram)
        if not proceed:
            self.loadingDataAborted()
            return

        self.logger.info(f'Reading {user_ch_name} channel metadata...')
        # Get information from first loaded position
        posData = load.loadData(user_ch_file_paths[0], user_ch_name)
        posData.getBasenameAndChNames()
        posData.buildPaths()

        # Get end name of every existing segmentation file
        existingSegmEndNames = set()
        for filePath in user_ch_file_paths:
            _posData = load.loadData(filePath, user_ch_name)
            _posData.getBasenameAndChNames()
            segm_files = load.get_segm_files(_posData.images_path)
            _existingEndnames = load.get_existing_segm_endnames(
                _posData.basename, segm_files
            )
            existingSegmEndNames.update(_existingEndnames)

        selectedSegmEndName = ''
        self.newSegmEndName = ''
        if self.isNewFile:
            # Remove the 'segm_' part to allow filenameDialog to check if
            # a new file is existing (since we only ask for the part after
            # 'segm_')
            existingEndNames = [
                n.replace('segm', '', 1).replace('_', '', 1)
                for n in existingSegmEndNames
            ]
            win = apps.filenameDialog(
                basename=f'{posData.basename}segm',
                hintText='Insert a <b>filename</b> for the segmentation file:<br>',
                existingNames=existingEndNames
            )
            win.exec_()
            if win.cancel:
                self.loadingDataAborted()
                return
            self.newSegmEndName = win.entryText
        else:
            if len(existingSegmEndNames) > 1:
                win = apps.QDialogMultiSegmNpz(
                    existingSegmEndNames, self.exp_path, parent=self
                )
                win.exec_()
                if win.cancel:
                    self.loadingDataAborted()
                    return

                selectedSegmEndName = win.selectedItemText
            elif len(existingSegmEndNames) == 1:
                selectedSegmEndName = list(existingSegmEndNames)[0]

        posData.loadImgData()
        posData.loadOtherFiles(
            load_segm_data=True,
            load_metadata=True,
            create_new_segm=self.isNewFile,
            new_endname=self.newSegmEndName,
            end_filename_segm=selectedSegmEndName
        )
        self.selectedSegmEndName = selectedSegmEndName
        self.labelBoolSegm = posData.labelBoolSegm
        posData.labelSegmData()

        print('')
        self.logger.info(
            f'Segmentation filename: {posData.segm_npz_path}'
        )

        proceed = posData.askInputMetadata(
            self.num_pos,
            ask_SizeT=self.num_pos==1,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            singlePos=False,
            save=True
        )
        if not proceed:
            self.loadingDataAborted()
            return

        self.isSegm3D = posData.isSegm3D
        self.SizeT = posData.SizeT
        self.SizeZ = posData.SizeZ
        self.TimeIncrement = posData.TimeIncrement
        self.PhysicalSizeZ = posData.PhysicalSizeZ
        self.PhysicalSizeY = posData.PhysicalSizeY
        self.PhysicalSizeX = posData.PhysicalSizeX
        self.loadSizeS = posData.loadSizeS
        self.loadSizeT = posData.loadSizeT
        self.loadSizeZ = posData.loadSizeZ

        self.isH5chunk = (
            posData.ext == '.h5'
            and (self.loadSizeT != self.SizeT
                or self.loadSizeZ != self.SizeZ)
        )

        required_ram = posData.checkH5memoryFootprint()*self.loadSizeS
        if required_ram > 0:
            proceed = self.checkMemoryRequirements(required_ram)
            if not proceed:
                self.loadingDataAborted()
                return

        if posData.SizeT == 1:
            self.isSnapshot = True
        else:
            self.isSnapshot = False

        self.progressWin = apps.QDialogWorkerProgress(
            title='Loading data...', parent=self,
            pbarDesc=f'Loading "{user_ch_file_paths[0]}"...'
        )
        self.progressWin.show(self.app)

        func = partial(
            self.startLoadDataWorker, user_ch_file_paths, user_ch_name,
            posData
        )
        QTimer.singleShot(150, func)

    @myutils.exception_handler
    def startLoadDataWorker(self, user_ch_file_paths, user_ch_name, firstPosData):
        self.funcDescription = 'loading data'

        self.thread = QThread()
        self.loadDataMutex = QMutex()
        self.loadDataWaitCond = QWaitCondition()

        self.loadDataWorker = workers.loadDataWorker(
            self, user_ch_file_paths, user_ch_name, firstPosData
        )

        self.loadDataWorker.moveToThread(self.thread)
        self.loadDataWorker.signals.finished.connect(self.thread.quit)
        self.loadDataWorker.signals.finished.connect(
            self.loadDataWorker.deleteLater
        )
        self.thread.finished.connect(self.thread.deleteLater)

        self.loadDataWorker.signals.finished.connect(
            self.loadDataWorkerFinished
        )
        self.loadDataWorker.signals.progress.connect(self.workerProgress)
        self.loadDataWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.loadDataWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.loadDataWorker.signals.critical.connect(
            self.workerCritical
        )
        self.loadDataWorker.signals.dataIntegrityCritical.connect(
            self.loadDataWorkerDataIntegrityCritical
        )
        self.loadDataWorker.signals.dataIntegrityWarning.connect(
            self.loadDataWorkerDataIntegrityWarning
        )
        self.loadDataWorker.signals.sigPermissionError.connect(
            self.workerPermissionError
        )

        self.thread.started.connect(self.loadDataWorker.run)
        self.thread.start()

    def workerPermissionError(self, txt, waitCond):
        msg = widgets.myMessageBox(parent=self)
        msg.setIcon(iconName='SP_MessageBoxCritical')
        msg.setWindowTitle('Permission denied')
        msg.addText(txt)
        msg.addButton('  Ok  ')
        msg.exec_()
        waitCond.wakeAll()

    def loadDataWorkerDataIntegrityCritical(self):
        errTitle = 'All loaded positions contains frames over time!'
        self.titleLabel.setText(errTitle, color='r')

        msg = widgets.myMessageBox(parent=self)

        err_msg = html_utils.paragraph(f"""
            {errTitle}.<br><br>
            To load data that contains frames over time you have to select
            only ONE position.
        """)
        msg.setIcon(iconName='SP_MessageBoxCritical')
        msg.setWindowTitle('Loaded multiple positions with frames!')
        msg.addText(err_msg)
        msg.addButton('Ok')
        msg.show(block=True)

    @myutils.exception_handler
    def loadDataWorkerFinished(self, data):
        self.funcDescription = 'loading data worker finished'
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        if data is None or data=='abort':
            self.loadingDataAborted()
            return

        self.pos_i = 0
        self.data = data
        self.gui_createGraphicsItems()
        return True

    def loadingDataCompleted(self):
        posData = self.data[self.pos_i]

        self.guiTabControl.addChannels([posData.user_ch_name])
        self.showPropsDockButton.setDisabled(False)

        self.init_segmInfo_df()
        self.connectScrollbars()
        self.initPosAttr()
        self.initFluoData()
        self.navigateScrollBar.setSliderPosition(posData.frame_i+1)
        if posData.SizeZ > 1:
            idx = (posData.filename, posData.frame_i)
            how = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
            self.zProjComboBox.setCurrentText(how)

        self.create_chNamesQActionGroup(self.user_ch_name)

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()
        if not self.isEditActionsConnected:
            self.gui_connectEditActions()

        self.setFramesSnapshotMode()

        self.enableZstackWidgets(posData.SizeZ > 1)

        self.img1BottomGroupbox.show()
        isLabVisible = self.df_settings.at['isLabelsVisible', 'value'] == 'Yes'
        if self.isSegm3D and isLabVisible:
            self.labBottomGroupbox.show()
        self.updateScrollbars()
        self.fontSizeAction.setChecked(True)
        self.openAction.setEnabled(True)
        self.editTextIDsColorAction.setDisabled(False)
        self.imgPropertiesAction.setEnabled(True)
        self.navigateToolBar.setVisible(True)
        if 'isLabelsVisible' in self.df_settings.index:
            val = self.df_settings.at['isLabelsVisible', 'value'] == 'No'
            if val:
                self.labelsGrad.hideLabelsImgAction.setChecked(True)

        self.readSavedCustomAnnot()
        self.addCustomAnnotationSavedPos()

        self.setBottomLayoutStretch()
        self.setAxesMaxRange()
        self.setImageNameText()

        self.initLookupTableLab()
        self.updateALLimg()
        self.restoreSavedSettings()

        self.setMetricsFunc()

        self.titleLabel.setText(
            'Data successfully loaded.',
            color=self.titleColor
        )

        QTimer.singleShot(200, self.autoRange)

    def restoreSavedSettings(self):
        if 'how_draw_annotations' in self.df_settings.index:
            how = self.df_settings.at['how_draw_annotations', 'value']
            self.drawIDsContComboBox.setCurrentText(how)

    def setImageNameText(self):
        self.statusbar.clearMessage()
        posData = self.data[self.pos_i]
        segmentedChannelname = posData.filename[len(posData.basename):]
        segmEndName = os.path.basename(posData.segm_npz_path)[len(posData.basename):]
        txt = (
            f'Segmented channel: {segmentedChannelname}, '
            f'Segmentation file name: {segmEndName}'
        )
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)

    def autoRange(self):
        if not self.labelsGrad.hideLabelsImgAction.isChecked():
            self.ax2.vb.autoRange()
        self.ax1.vb.autoRange()

    def setAxesMaxRange(self):
        return

        # Get current maxRange and prevent from zooming out too far
        screenSize = self.screen().size()
        xRangeScreen, yRangeScreen = screenSize.width(), screenSize.height()
        Y, X = self.img1.image.shape[:2]
        xRange = Y/yRangeScreen*xRangeScreen
        if xRange > X and xRangeScreen > yRangeScreen:
            self.ax1.setLimits(maxXRange=int(xRange*2))

    def setFramesSnapshotMode(self):
        self.measurementsMenu.setDisabled(False)
        if self.isSnapshot:
            self.disableTrackingCheckBox.setDisabled(True)
            try:
                self.drawIDsContComboBox.currentIndexChanged.disconnect()
            except Exception as e:
                pass

            self.repeatTrackingAction.setDisabled(True)
            self.logger.info('Setting GUI mode to "Snapshots"...')
            self.modeComboBox.clear()
            self.modeComboBox.addItems(['Snapshot'])
            self.modeComboBox.setDisabled(True)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            self.drawIDsContComboBox.setCurrentIndex(1)
            self.modeToolBar.setVisible(False)
            self.modeComboBox.setCurrentText('Snapshot')
            self.annotateToolbar.setVisible(True)
            self.drawIDsContComboBox.currentIndexChanged.connect(
                self.drawIDsContComboBox_cb
            )
        else:
            self.annotateToolbar.setVisible(False)
            self.disableTrackingCheckBox.setDisabled(False)
            self.repeatTrackingAction.setDisabled(False)
            self.modeComboBox.setDisabled(False)
            try:
                self.modeComboBox.activated.disconnect()
                self.modeComboBox.currentIndexChanged.disconnect()
                self.drawIDsContComboBox.currentIndexChanged.disconnect()
            except Exception as e:
                pass
                # traceback.print_exc()
            self.modeComboBox.clear()
            self.modeComboBox.addItems(self.modeItems)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            self.modeComboBox.currentIndexChanged.connect(self.changeMode)
            self.modeComboBox.activated.connect(self.clearComboBoxFocus)
            self.drawIDsContComboBox.currentIndexChanged.connect(
                                                    self.drawIDsContComboBox_cb)
            self.modeComboBox.setCurrentText('Viewer')

    def checkIfAutoSegm(self):
        """
        If there are any frame or position with empty segmentation mask
        ask whether automatic segmentation should be turned ON
        """
        if self.autoSegmAction.isChecked():
            return
        if self.autoSegmDoNotAskAgain:
            return

        ask = False
        for posData in self.data:
            if posData.SizeT > 1:
                for lab in posData.segm_data:
                    if not np.any(lab):
                        ask = True
                        txt = 'frames'
                        break
            else:
                if not np.any(posData.segm_data):
                    ask = True
                    txt = 'positions'
                    break

        if not ask:
            return

        questionTxt = html_utils.paragraph(
            f'Some or all loaded {txt} contain <b>empty segmentation masks</b>.<br><br>'
            'Do you want to <b>activate automatic segmentation</b><sup>*</sup> '
            f'when visiting these {txt}?<br><br>'
            '<i>* Automatic segmentation can always be turned ON/OFF from the menu<br>'
            '  <code>Edit --> Segmentation --> Enable automatic segmentation</code><br><br></i>'
            f'NOTE: you can automatically segment all {txt} using the<br>'
            '    segmentation module.'
        )
        msg = widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.question(
            self, 'Automatic segmentation?', questionTxt,
            buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton == yesButton:
            self.autoSegmAction.setChecked(True)
        else:
            self.autoSegmDoNotAskAgain = True
            self.autoSegmAction.setChecked(False)

    def updateZsliceLab(self, action):
        self.zSliceScrollBarLab.updateLabel()
        if action == QAbstractSlider.SliderMove:
            if not self.zSliceScrollBarLab.moving:
                self.clearAllItems()
                self.zSliceScrollBarLab.moving = True
            self.setImageImg2()
        else:
            self.updateALLimg(
                useStoredGaussFiltered=True, updateDiffGaussFilter=True
            )

    def zSliceScrollBarLabReleased(self):
        self.zSliceScrollBarLab.moving = False
        self.updateALLimg(
            useStoredGaussFiltered=True, updateDiffGaussFilter=True
        )

    def init_segmInfo_df(self):
        for posData in self.data:
            if posData.SizeZ > 1 and posData.segmInfo_df is not None:
                if 'z_slice_used_gui' not in posData.segmInfo_df.columns:
                    posData.segmInfo_df['z_slice_used_gui'] = (
                                    posData.segmInfo_df['z_slice_used_dataPrep']
                                    )
                if 'which_z_proj_gui' not in posData.segmInfo_df.columns:
                    posData.segmInfo_df['which_z_proj_gui'] = (
                                    posData.segmInfo_df['which_z_proj']
                                    )
                posData.segmInfo_df['resegmented_in_gui'] = False
                posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

            NO_segmInfo = (
                posData.segmInfo_df is None
                or posData.filename not in posData.segmInfo_df.index
            )
            if NO_segmInfo and posData.SizeZ > 1:
                filename = posData.filename
                df = myutils.getDefault_SegmInfo_df(posData, filename)
                if posData.segmInfo_df is None:
                    posData.segmInfo_df = df
                else:
                    posData.segmInfo_df = pd.concat([df, posData.segmInfo_df])
                    unique_idx = ~posData.segmInfo_df.index.duplicated()
                    posData.segmInfo_df = posData.segmInfo_df[unique_idx]
                posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

    def connectScrollbars(self):
        self.t_label.show()
        self.navigateScrollBar.show()
        self.navigateScrollBar.setDisabled(False)

        if self.data[0].SizeZ > 1:
            self.enableZstackWidgets(True)
            self.zSliceScrollBar.setMaximum(self.data[0].SizeZ-1)
            try:
                self.zSliceScrollBar.actionTriggered.disconnect()
                self.zSliceScrollBar.sliderReleased.disconnect()
                self.zProjComboBox.currentTextChanged.disconnect()
                self.zProjComboBox.activated.disconnect()
            except Exception as e:
                pass
            self.zSliceScrollBar.actionTriggered.connect(
                self.zSliceScrollBarActionTriggered
            )
            self.zSliceScrollBar.sliderReleased.connect(
                self.zSliceScrollBarReleased
            )
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
            self.zProjComboBox.activated.connect(self.clearComboBoxFocus)
            if self.isSegm3D and self.labBottomGroupbox.isChecked():
                self.connectZSliceScrollBarLab()

        posData = self.data[self.pos_i]
        if posData.SizeT == 1:
            self.t_label.setText('Position n. ')
            self.navigateScrollBar.setMinimum(1)
            self.navigateScrollBar.setMaximum(len(self.data))
            self.navigateScrollBar.setAbsoluteMaximum(len(self.data))
            try:
                self.navigateScrollBar.sliderMoved.disconnect()
                self.navigateScrollBar.sliderReleased.disconnect()
                self.navigateScrollBar.actionTriggered.disconnect()
            except TypeError:
                pass
            self.navigateScrollBar.sliderMoved.connect(
                self.PosScrollBarMoved
            )
            self.navigateScrollBar.sliderReleased.connect(
                self.PosScrollBarReleased
            )
            self.navigateScrollBar.actionTriggered.connect(
                self.PosScrollBarAction
            )
        else:
            self.navigateScrollBar.setMinimum(1)
            self.navigateScrollBar.setAbsoluteMaximum(posData.SizeT)
            if posData.last_tracked_i is not None:
                self.navigateScrollBar.setMaximum(posData.last_tracked_i+1)
            try:
                self.navigateScrollBar.sliderMoved.disconnect()
                self.navigateScrollBar.sliderReleased.disconnect()
                self.navigateScrollBar.actionTriggered.disconnect()
            except Exception as e:
                pass
            self.t_label.setText('frame n.  ')
            self.navigateScrollBar.sliderMoved.connect(
                self.framesScrollBarMoved
            )
            self.navigateScrollBar.sliderReleased.connect(
                self.framesScrollBarReleased
            )
            self.navigateScrollBar.actionTriggered.connect(
                self.framesScrollBarAction
            )

    def zSliceScrollBarActionTriggered(self, action):
        self.update_z_slice(self.zSliceScrollBar.sliderPosition())

    def zSliceScrollBarReleased(self):
        self.zSliceScrollBarLab.moving = False
        self.update_z_slice(self.zSliceScrollBar.sliderPosition())

    def update_z_slice(self, z):
        posData = self.data[self.pos_i]
        idx = (posData.filename, posData.frame_i)
        posData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
        self.updateALLimg(
            only_ax1=self.updateOnlyImg(), useStoredGaussFiltered=True,
            updateDiffGaussFilter=True
        )

    def update_overlay_z_slice(self, z):
        posData = self.data[self.pos_i]
        filename = list(posData.ol_data.keys())[0]
        idx = (filename, posData.frame_i)
        posData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
        self.getOverlayImg(setImg=True)

    def updateOverlayZproj(self, how):
        self.getOverlayImg(setImg=True)
        if how.find('max') != -1 or how == 'same as above':
            self.overlay_z_label.setStyleSheet('color: gray')
            self.zSliceOverlay_SB.setDisabled(True)
        else:
            self.overlay_z_label.setStyleSheet('color: black')
            self.zSliceOverlay_SB.setDisabled(False)

    def updateZproj(self, how):
        for p, posData in enumerate(self.data[self.pos_i:]):
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, 'which_z_proj_gui'] = how
        posData = self.data[self.pos_i]
        if how == 'single z-slice':
            self.zSliceScrollBar.setDisabled(False)
            self.z_label.setStyleSheet('color: black')
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.z_label.setStyleSheet('color: gray')
            self.updateALLimg(only_ax1=self.updateOnlyImg())

    def clearItems_IDs(self, IDs_to_clear):
        for ID in IDs_to_clear:
            if self.ax1_ContoursCurves[ID-1] is None:
                continue

            self.ax1_ContoursCurves[ID-1].setData([], [])
            self.ax2_ContoursCurves[ID-1].setData([], [])
            self.ax1_LabelItemsIDs[ID-1].setText('')
            self.ax2_LabelItemsIDs[ID-1].setText('')
            self.ax1_BudMothLines[ID-1].setData([], [])

    def removeGraphicsItemsIDs(self, maxID):
        itemsToRemove = zip(
            self.ax1_ContoursCurves[maxID:],
            self.ax2_ContoursCurves[maxID:],
            self.ax1_LabelItemsIDs[maxID:],
            self.ax2_LabelItemsIDs[maxID:],
            self.ax1_BudMothLines[maxID:]
        )
        for items in itemsToRemove:
            (ax1ContCurve, ax2ContCurve,
            _IDlabel1, _IDlabel2,
            BudMothLine) = items

            if ax1ContCurve is None:
                continue

            self.ax1.removeItem(ax1ContCurve)
            self.ax1.removeItem(_IDlabel1)
            self.ax1.removeItem(BudMothLine)
            self.ax2.removeItem(ax2ContCurve)
            self.ax2.removeItem(_IDlabel2)

        self.ax1_ContoursCurves = self.ax1_ContoursCurves[:maxID]
        self.ax2_ContoursCurves = self.ax2_ContoursCurves[:maxID]
        self.ax1_LabelItemsIDs = self.ax1_LabelItemsIDs[:maxID]
        self.ax2_LabelItemsIDs = self.ax2_LabelItemsIDs[:maxID]
        self.ax1_BudMothLines = self.ax1_BudMothLines[:maxID]

    def clearLabAnnotations(self):
        labAnnot = zip(self.ax2_ContoursCurves, self.ax2_LabelItemsIDs)
        for ax2ContCurve, _IDlabel2 in labAnnot:
            if ax2ContCurve.getData()[0] is not None:
                ax2ContCurve.setData([], [])
            _IDlabel2.setText('')

    def clearAllItems(self):
        self.ax2_binnedIDs_ScatterPlot.setData([], [])
        self.ax2_ripIDs_ScatterPlot.setData([], [])
        self.ax1_binnedIDs_ScatterPlot.setData([], [])
        self.ax1_ripIDs_ScatterPlot.setData([], [])

        allItems = zip(
            self.ax1_ContoursCurves,
            self.ax2_ContoursCurves,
            self.ax1_LabelItemsIDs,
            self.ax2_LabelItemsIDs,
            self.ax1_BudMothLines
        )
        for idx, items_ID in enumerate(allItems):
            (ax1ContCurve, ax2ContCurve,
            _IDlabel1, _IDlabel2,
            BudMothLine) = items_ID

            if ax1ContCurve is None:
                continue

            if ax1ContCurve.getData()[0] is not None:
                ax1ContCurve.setData([], [])
            if ax2ContCurve.getData()[0] is not None:
                ax2ContCurve.setData([], [])
            if BudMothLine.getData()[0] is not None:
                BudMothLine.setData([], [])
            _IDlabel1.setText('')
            _IDlabel2.setText('')

    def clearCurvItems(self, removeItems=True):
        try:
            posData = self.data[self.pos_i]
            curvItems = zip(posData.curvPlotItems,
                            posData.curvAnchorsItems,
                            posData.curvHoverItems)
            for plotItem, curvAnchors, hoverItem in curvItems:
                plotItem.setData([], [])
                curvAnchors.setData([], [])
                hoverItem.setData([], [])
                if removeItems:
                    self.ax1.removeItem(plotItem)
                    self.ax1.removeItem(curvAnchors)
                    self.ax1.removeItem(hoverItem)

            if removeItems:
                posData.curvPlotItems = []
                posData.curvAnchorsItems = []
                posData.curvHoverItems = []
        except AttributeError:
            # traceback.print_exc()
            pass

    def splineToObj(self, xxA=None, yyA=None, isRightClick=False):
        posData = self.data[self.pos_i]
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        if isRightClick:
            xxS, yyS = self.curvPlotItem.getData()
            if xxS is None:
                self.setUncheckedAllButtons()
                return
            N = len(xxS)
            self.smoothAutoContWithSpline(n=int(N*0.05))

        xxS, yyS = self.curvPlotItem.getData()

        self.setBrushID()
        newIDMask = np.zeros(posData.lab.shape, bool)
        rr, cc = skimage.draw.polygon(yyS, xxS)
        newIDMask[rr, cc] = True
        newIDMask[posData.lab!=0] = False
        posData.lab[newIDMask] = posData.brushID

    def addFluoChNameContextMenuAction(self, ch_name):
        posData = self.data[self.pos_i]
        allTexts = [
            action.text() for action in self.chNamesQActionGroup.actions()
        ]
        if ch_name not in allTexts:
            action = QAction(self)
            action.setText(ch_name)
            action.setCheckable(True)
            self.chNamesQActionGroup.addAction(action)
            action.setChecked(True)
            posData.fluoDataChNameActions.append(action)

    def computeSegm(self):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer' or mode == 'Cell cycle analysis':
            return

        if np.any(posData.lab):
            # Do not compute segm if there is already a mask
            return

        if not self.autoSegmAction.isChecked():
            # Compute segmentations that have an open window
            if self.segmModelName == 'randomWalker':
                self.randomWalkerWin.getImage()
                self.randomWalkerWin.computeMarkers()
                self.randomWalkerWin.computeSegm()
                self.update_rp()
                self.tracking(enforce=True)
                self.updateALLimg()
                self.store_data()
                self.warnEditingWithCca_df('Random Walker segmentation')
            else:
                return

        self.repeatSegm(model_name=self.segmModelName)
        self.update_rp()

    def initImgCmap(self):
        if not 'img_cmap' in self.df_settings.index:
            self.df_settings.at['img_cmap', 'value'] = 'grey'
        self.imgCmapName = self.df_settings.at['img_cmap', 'value']
        self.imgCmap = self.imgGrad.cmaps[self.imgCmapName]
        if self.imgCmapName != 'grey':
            # To ensure mapping to colors we need to normalize image
            self.normalizeByMaxAction.setChecked(True)


    def initGlobalAttr(self):
        self.setOverlayColors()

        self.initImgCmap()

        self.splineHoverON = False
        self.rulerHoverON = False
        self.isCtrlDown = False
        self.isShiftDown = False
        self.autoContourHoverON = False
        self.navigateScrollBarStartedMoving = True

        self.segment2D_kwargs = None
        self.segmModelName = None

        self.autoSegmDoNotAskAgain = False

        self.clickedOnBud = False
        self.gaussWin = None
        self.diffGaussFilterWin = None
        self.postProcessSegmWin = None
        self.edgeWin = None
        self.entropyWin = None

        self.UserEnforced_DisabledTracking = False
        self.UserEnforced_Tracking = False

        self.ax1BrushHoverID = 0

        # Plots items
        self.data_loaded = True
        self.isMouseDragImg2 = False
        self.isMouseDragImg1 = False
        self.isMovingLabel = False
        self.isRightClickDragImg1 = False

        self.cca_df_colnames = list(base_cca_df.keys())
        self.cca_df_dtypes = [
            str, int, int, str, int, int, bool, bool
        ]
        self.cca_df_default_values = list(base_cca_df.values())
        self.cca_df_int_cols = [
            'generation_num',
            'relative_ID',
            'emerg_frame_i',
            'division_frame_i'
        ]
        self.cca_df_bool_col = [
            'is_history_known',
            'corrected_assignment'
        ]

        # Metrics
        self.initMetricsToSave()

    def initMetricsToSave(self):
        self.chNamesToSkip = []
        self.metricsToSkip = {}
        self.regionPropsToSave = measurements.get_props_names()
        self.mixedChCombineMetricsToSave = list(
            measurements.get_user_combine_mixed_channels_desc().keys()
        )
        self.sizeMetricsToSave = list(
            measurements.get_size_metrics_desc().keys()
        )

    def initPosAttr(self):
        for p, posData in enumerate(self.data):
            self.pos_i = p
            posData.curvPlotItems = []
            posData.curvAnchorsItems = []
            posData.curvHoverItems = []
            posData.fluoDataChNameActions = []
            posData.manualContrastKey = posData.filename

            posData.HDDmaxID = np.max(posData.segm_data)

            # Decision on what to do with changes to future frames attr
            posData.doNotShowAgain_EditID = False
            posData.UndoFutFrames_EditID = False
            posData.applyFutFrames_EditID = False

            posData.doNotShowAgain_RipID = False
            posData.UndoFutFrames_RipID = False
            posData.applyFutFrames_RipID = False

            posData.doNotShowAgain_DelID = False
            posData.UndoFutFrames_DelID = False
            posData.applyFutFrames_DelID = False

            posData.doNotShowAgain_BinID = False
            posData.UndoFutFrames_BinID = False
            posData.applyFutFrames_BinID = False

            posData.disableAutoActivateViewerWindow = False
            posData.new_IDs = []
            posData.lost_IDs = []
            posData.multiBud_mothIDs = [2]
            posData.UndoRedoStates = [[] for _ in range(posData.SizeT)]
            posData.UndoRedoCcaStates = [[] for _ in range(posData.SizeT)]

            posData.ol_data_dict = {}
            posData.ol_data = None

            # Colormap
            self.setLut(posData)

            posData.allData_li = [{
                'regionprops': None,
                'labels': None,
                'acdc_df': None,
                'delROIs_info': {
                    'rois': [], 'delMasks': [], 'delIDsROI': []
                },
                'histoLevels': {}
                } for i in range(posData.SizeT)
            ]

            posData.ccaStatus_whenEmerged = {}

            posData.frame_i = 0
            posData.brushID = 0
            posData.binnedIDs = set()
            posData.ripIDs = set()
            posData.multiContIDs = set()
            posData.cca_df = None
            if posData.last_tracked_i is not None:
                last_tracked_num = posData.last_tracked_i+1
                # Load previous session data
                # Keep track of which ROIs have already been added
                # in previous frame
                delROIshapes = [[] for _ in range(posData.SizeT)]
                for i in range(last_tracked_num):
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(enforce=True)
                    # self.load_delROIs_info(delROIshapes, last_tracked_num)
                    posData.binnedIDs = set()
                    posData.ripIDs = set()

                # Ask whether to resume from last frame
                if last_tracked_num>1:
                    msg = widgets.myMessageBox()
                    txt = html_utils.paragraph(
                        'The system detected a previous session ended '
                        f'at frame {last_tracked_num}.<br><br>'
                        f'Do you want to <b>resume from frame '
                        f'{last_tracked_num}?</b>'
                    )
                    noButton, yesButton = msg.question(
                        self, 'Start from last session?', txt,
                        buttonsTexts=(' No ', 'Yes')
                    )
                    if msg.clickedButton == yesButton:
                        posData.frame_i = posData.last_tracked_i
                    else:
                        posData.frame_i = 0

        # Back to first position
        self.pos_i = 0
        self.get_data(debug=False)
        self.store_data()
        # self.updateALLimg()

        # Link Y and X axis of both plots to scroll zoom and pan together
        self.ax2.vb.setYLink(self.ax1.vb)
        self.ax2.vb.setXLink(self.ax1.vb)

    def PosScrollBarAction(self, action):
        if action == QAbstractSlider.SliderSingleStepAdd:
            self.next_cb()
        elif action == QAbstractSlider.SliderSingleStepSub:
            self.prev_cb()
        elif action == QAbstractSlider.SliderPageStepAdd:
            self.PosScrollBarReleased()
        elif action == QAbstractSlider.SliderPageStepSub:
            self.PosScrollBarReleased()

    def PosScrollBarMoved(self, pos_n):
        self.pos_i = pos_n-1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateFilters=False, updateDiffGaussFilter=False)

    def PosScrollBarReleased(self):
        self.pos_i = self.navigateScrollBar.sliderPosition()-1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg()

    def framesScrollBarAction(self, action):
        if action == QAbstractSlider.SliderSingleStepAdd:
            # Clicking on dialogs triggered by next_cb might trigger
            # pressEvent of navigateQScrollBar, avoid that
            self.navigateScrollBar.disableCustomPressEvent()
            self.next_cb()
            QTimer.singleShot(100, self.navigateScrollBar.enableCustomPressEvent)
        elif action == QAbstractSlider.SliderSingleStepSub:
            self.prev_cb()
        elif action == QAbstractSlider.SliderPageStepAdd:
            self.framesScrollBarReleased()
        elif action == QAbstractSlider.SliderPageStepSub:
            self.framesScrollBarReleased()

    def framesScrollBarMoved(self, frame_n):
        posData = self.data[self.pos_i]
        posData.frame_i = frame_n-1
        if posData.allData_li[posData.frame_i]['labels'] is None:
            posData.lab = posData.segm_data[posData.frame_i]
        else:
            posData.lab = posData.allData_li[posData.frame_i]['labels']

        if self.overlayButton.isChecked():
            img = self.getOverlayImg(setImg=False)
        else:
            img = self.getImageWithCmap()

        if self.navigateScrollBarStartedMoving:
            self.clearAllItems()
        self.t_label.setText(
            f'frame n. {posData.frame_i+1}/{posData.SizeT}'
        )
        self.img1.setImage(img)
        if not self.labelsGrad.hideLabelsImgAction.isChecked():
            self.img2.setImage(posData.lab, z=self.z_lab(), autoLevels=False)
        self.updateLookuptable()
        self.updateFramePosLabel()
        self.updateViewerWindow()
        self.navigateScrollBarStartedMoving = False

    def framesScrollBarReleased(self):
        self.navigateScrollBarStartedMoving = True
        posData = self.data[self.pos_i]
        posData.frame_i = self.navigateScrollBar.sliderPosition()-1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg()

    def unstore_data(self):
        posData = self.data[self.pos_i]
        posData.allData_li[posData.frame_i] = {
            'regionprops': [],
            'labels': None,
            'acdc_df': None,
            'delROIs_info': {
                'rois': [], 'delMasks': [], 'delIDsROI': []
            },
            'histoLevels': {}
        }

    def store_data(self, pos_i=None, enforce=True, debug=False, mainThread=True):
        pos_i = self.pos_i if pos_i is None else pos_i
        posData = self.data[pos_i]
        if posData.frame_i < 0:
            # In some cases we set frame_i = -1 and then call next_frame
            # to visualize frame 0. In that case we don't store data
            # for frame_i = -1
            return

        mode = str(self.modeComboBox.currentText())

        if mode == 'Viewer' and not enforce:
            return

        posData.allData_li[posData.frame_i]['regionprops'] = posData.rp.copy()
        posData.allData_li[posData.frame_i]['labels'] = posData.lab.copy()

        # Store dynamic metadata
        is_cell_dead_li = [False]*len(posData.rp)
        is_cell_excluded_li = [False]*len(posData.rp)
        IDs = [0]*len(posData.rp)
        xx_centroid = [0]*len(posData.rp)
        yy_centroid = [0]*len(posData.rp)
        areManuallyEdited = [0]*len(posData.rp)
        editedNewIDs = [vals[2] for vals in posData.editID_info]
        for i, obj in enumerate(posData.rp):
            is_cell_dead_li[i] = obj.dead
            is_cell_excluded_li[i] = obj.excluded
            IDs[i] = obj.label
            xx_centroid[i] = int(self.getObjCentroid(obj.centroid)[1])
            yy_centroid[i] = int(self.getObjCentroid(obj.centroid)[0])
            if obj.label in editedNewIDs:
                areManuallyEdited[i] = 1

        try:
            posData.STOREDmaxID = max(IDs)
        except ValueError:
            posData.STOREDmaxID = 0

        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        if acdc_df is None:
            posData.allData_li[posData.frame_i]['acdc_df'] = pd.DataFrame(
                {
                    'Cell_ID': IDs,
                    'is_cell_dead': is_cell_dead_li,
                    'is_cell_excluded': is_cell_excluded_li,
                    'x_centroid': xx_centroid,
                    'y_centroid': yy_centroid,
                    'was_manually_edited': areManuallyEdited
                }
            ).set_index('Cell_ID')
        else:
            # Filter or add IDs that were not stored yet
            acdc_df = acdc_df.drop(columns=['time_seconds'], errors='ignore')
            acdc_df = acdc_df.reindex(IDs, fill_value=0)
            acdc_df['is_cell_dead'] = is_cell_dead_li
            acdc_df['is_cell_excluded'] = is_cell_excluded_li
            acdc_df['x_centroid'] = xx_centroid
            acdc_df['y_centroid'] = yy_centroid
            acdc_df['was_manually_edited'] = areManuallyEdited
            posData.allData_li[posData.frame_i]['acdc_df'] = acdc_df

        self.store_cca_df(pos_i=pos_i, mainThread=mainThread)

    def nearest_point_2Dyx(self, points, all_others):
        """
        Given 2D array of [y, x] coordinates points and all_others return the
        [y, x] coordinates of the two points (one from points and one from all_others)
        that have the absolute minimum distance
        """
        # Compute 3D array where each ith row of each kth page is the element-wise
        # difference between kth row of points and ith row in all_others array.
        # (i.e. diff[k,i] = points[k] - all_others[i])
        diff = points[:, np.newaxis] - all_others
        # Compute 2D array of distances where
        # dist[i, j] = euclidean dist (points[i],all_others[j])
        dist = np.linalg.norm(diff, axis=2)
        # Compute i, j indexes of the absolute minimum distance
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        nearest_point = all_others[j]
        point = points[i]
        min_dist = np.min(dist)
        return min_dist, nearest_point

    def checkMultiBudMoth(self, draw=False):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            posData.multiBud_mothIDs = []
            return

        cca_df_S = posData.cca_df[posData.cca_df['cell_cycle_stage'] == 'S']
        cca_df_S_bud = cca_df_S[cca_df_S['relationship'] == 'bud']
        relIDs_of_S_bud = cca_df_S_bud['relative_ID']
        duplicated_relIDs_mask = relIDs_of_S_bud.duplicated(keep=False)
        duplicated_cca_df_S = cca_df_S_bud[duplicated_relIDs_mask]
        multiBud_mothIDs = duplicated_cca_df_S['relative_ID'].unique()
        posData.multiBud_mothIDs = multiBud_mothIDs
        multiBudInfo = []
        for multiBud_ID in multiBud_mothIDs:
            duplicatedBuds_df = cca_df_S_bud[
                                    cca_df_S_bud['relative_ID'] == multiBud_ID]
            duplicatedBudIDs = duplicatedBuds_df.index.to_list()
            info = f'Mother ID {multiBud_ID} has bud IDs {duplicatedBudIDs}'
            multiBudInfo.append(info)
        if multiBudInfo:
            multiBudInfo_format = '\n'.join(multiBudInfo)
            self.MultiBudMoth_msg = QMessageBox()
            self.MultiBudMoth_msg.setWindowTitle(
                                  'Mother with multiple buds assigned to it!')
            self.MultiBudMoth_msg.setText(multiBudInfo_format)
            self.MultiBudMoth_msg.setIcon(self.MultiBudMoth_msg.Warning)
            self.MultiBudMoth_msg.setDefaultButton(self.MultiBudMoth_msg.Ok)
            self.MultiBudMoth_msg.exec_()
        if draw:
            self.highlightmultiBudMoth()

    def isCurrentFrameCcaVisited(self):
        posData = self.data[self.pos_i]
        curr_df = posData.allData_li[posData.frame_i]['acdc_df']
        return curr_df is not None and 'cell_cycle_stage' in curr_df.columns

    def warnScellsGone(self, ScellsIDsGone, frame_i):
        msg = QMessageBox()
        text = html_utils.paragraph(f"""
            In the next frame the followning cells' IDs in S/G2/M
            (highlighted with a yellow contour) <b>will disappear</b>:<br><br>
            {ScellsIDsGone}<br><br>
            These cells are either buds or mother whose <b>related IDs will not
            disappear</b>. This is likely due to cell division happening in
            previous frame and the divided bud or mother will be
            washed away.<br><br>
            If you decide to continue these cells will be <b>automatically
            annotated as divided at frame number {frame_i}</b>.<br><br>
            Do you want to continue?
        """)
        answer = msg.warning(
           self, 'Cells in "S/G2/M" disappeared!', text,
           msg.Yes | msg.Cancel
        )
        return answer == msg.Yes

    def checkScellsGone(self):
        """Check if there are cells in S phase whose relative disappear in
        current frame. Allow user to choose between automatically assign
        division to these cells or cancel and not visit the frame.

        Returns
        -------
        bool
            False if there are no cells disappeared or the user decided
            to accept automatic division.
        """
        automaticallyDividedIDs = []

        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            # No cell cycle analysis mode --> do nothing
            return False, automaticallyDividedIDs

        posData = self.data[self.pos_i]


        if posData.allData_li[posData.frame_i]['labels'] is None:
            # Frame never visited/checked in segm mode --> autoCca_df will raise
            # a critical message
            return False, automaticallyDividedIDs

        # if self.isCurrentFrameCcaVisited():
        #     # Frame already visited in cca mode --> do nothing
        #     return False, automaticallyDividedIDs

        # Check if there are S cells that either only mother or only
        # bud disappeared and automatically assign division to it
        # or abort visiting this frame
        prev_acdc_df = posData.allData_li[posData.frame_i-1]['acdc_df']
        prev_cca_df = prev_acdc_df[self.cca_df_colnames].copy()

        ScellsIDsGone = []
        for ccSeries in prev_cca_df.itertuples():
            ID = ccSeries.Index
            ccs = ccSeries.cell_cycle_stage
            if ccs != 'S':
                continue

            relID = ccSeries.relative_ID
            if relID == -1:
                continue

            # Check is relID is gone while ID stays
            if relID not in posData.IDs and ID in posData.IDs:
                ScellsIDsGone.append(relID)

        if not ScellsIDsGone:
            # No cells in S that disappears --> do nothing
            return False, automaticallyDividedIDs

        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        for obj in prev_rp:
            if obj.label in ScellsIDsGone:
                self.highlight_obj(obj)

        self.highlightNewIDs_ccaFailed()
        proceed = self.warnScellsGone(ScellsIDsGone, posData.frame_i)
        if not proceed:
            return True, automaticallyDividedIDs

        for IDgone in ScellsIDsGone:
            relID = prev_cca_df.at[IDgone, 'relative_ID']
            self.annotateDivision(prev_cca_df, IDgone, relID)
            automaticallyDividedIDs.append(relID)

        self.store_cca_df(frame_i=posData.frame_i-1, cca_df=prev_cca_df)

        return False, automaticallyDividedIDs

    def attempt_auto_cca(self, enforceAll=False):
        posData = self.data[self.pos_i]
        try:
            notEnoughG1Cells, proceed = self.autoCca_df(
                enforceAll=enforceAll
            )
            if not proceed:
                return notEnoughG1Cells, proceed
            mode = str(self.modeComboBox.currentText())
            if posData.cca_df is None or mode.find('Cell cycle') == -1:
                notEnoughG1Cells = False
                proceed = True
                return notEnoughG1Cells, proceed
            if posData.cca_df.isna().any(axis=None):
                raise ValueError('Cell cycle analysis table contains NaNs')
            self.checkMultiBudMoth()
            return notEnoughG1Cells, proceed
        except Exception as e:
            self.logger.info('')
            self.logger.info('====================================')
            traceback.print_exc()
            self.logger.info('====================================')
            self.logger.info('')
            self.highlightNewIDs_ccaFailed()
            msg = QMessageBox(self)
            msg.setIcon(msg.Critical)
            msg.setWindowTitle('Failed cell cycle analysis')
            msg.setDefaultButton(msg.Ok)
            msg.setText(
                f'Cell cycle analysis for frame {posData.frame_i+1} failed!\n\n'
                'This can have multiple reasons:\n\n'
                '1. Segmentation or tracking errors --> Switch to \n'
                '   "Segmentation and Tracking" mode and check/correct next frame,\n'
                '   before attempting cell cycle analysis again.\n\n'
                '2. Edited a frame in "Segmentation and Tracking" mode\n'
                '   that already had cell cyce annotations -->\n'
                '   click on "Reinitialize cell cycle annotations" button,\n'
                '   and try again.')
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            return False, False

    def highlightIDs(self, IDs, pen):
        pass

    def warnFrameNeverVisitedSegmMode(self):
        msg = QMessageBox()
        warn_cca = msg.critical(
            self, 'Next frame NEVER visited',
            'Next frame was never visited in "Segmentation and Tracking"'
            'mode.\n You cannot perform cell cycle analysis on frames'
            'where segmentation and/or tracking errors were not'
            'checked/corrected.\n\n'
            'Switch to "Segmentation and Tracking" mode '
            'and check/correct next frame,\n'
            'before attempting cell cycle analysis again',
            msg.Ok
        )
        return False

    def autoCca_df(self, enforceAll=False):
        """
        Assign each bud to a mother with scipy linear sum assignment
        (Hungarian or Munkres algorithm). First we build a cost matrix where
        each (i, j) element is the minimum distance between bud i and mother j.
        Then we minimize the cost of assigning each bud to a mother, and finally
        we write the assignment info into cca_df
        """
        proceed = True
        notEnoughG1Cells = False
        ScellsGone = False

        posData = self.data[self.pos_i]

        # Skip cca if not the right mode
        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            return notEnoughG1Cells, proceed


        # Make sure that this is a visited frame in segmentation tracking mode
        if posData.allData_li[posData.frame_i]['labels'] is None:
            proceed = self.warnFrameNeverVisitedSegmMode()
            return notEnoughG1Cells, proceed

        # Determine if this is the last visited frame for repeating
        # bud assignment on non manually corrected_assignment buds.
        # The idea is that the user could have assigned division on a cell
        # by going previous and we want to check if this cell could be a
        # "better" mother for those non manually corrected buds
        lastVisited = False
        curr_df = posData.allData_li[posData.frame_i]['acdc_df']
        if curr_df is not None:
            if 'cell_cycle_stage' in curr_df.columns and not enforceAll:
                posData.new_IDs = [ID for ID in posData.new_IDs
                                if curr_df.at[ID, 'is_history_known']
                                and curr_df.at[ID, 'cell_cycle_stage'] == 'S']
                if posData.frame_i+1 < posData.SizeT:
                    next_df = posData.allData_li[posData.frame_i+1]['acdc_df']
                    if next_df is None:
                        lastVisited = True
                    else:
                        if 'cell_cycle_stage' not in next_df.columns:
                            lastVisited = True
                else:
                    lastVisited = True

        # Use stored cca_df and do not modify it with automatic stuff
        if posData.cca_df is not None and not enforceAll and not lastVisited:
            return notEnoughG1Cells, proceed

        # Keep only correctedAssignIDs if requested
        # For the last visited frame we perform assignment again only on
        # IDs where we didn't manually correct assignment
        if lastVisited and not enforceAll:
            correctedAssignIDs = curr_df[curr_df['corrected_assignment']].index
            posData.new_IDs = [
                ID for ID in posData.new_IDs
                if ID not in correctedAssignIDs
            ]

        # Check if there are some S cells that disappeared
        abort, automaticallyDividedIDs = self.checkScellsGone()
        if abort:
            notEnoughG1Cells = False
            proceed = False
            return notEnoughG1Cells, proceed

        # Get previous dataframe
        acdc_df = posData.allData_li[posData.frame_i-1]['acdc_df']
        prev_cca_df = acdc_df[self.cca_df_colnames].copy()

        if posData.cca_df is None:
            posData.cca_df = prev_cca_df
        else:
            posData.cca_df = curr_df[self.cca_df_colnames].copy()

        # If there are no new IDs we are done
        if not posData.new_IDs:
            proceed = True
            self.store_cca_df()
            return notEnoughG1Cells, proceed

        # Get cells in G1 (exclude dead) and check if there are enough cells in G1
        prev_df_G1 = prev_cca_df[prev_cca_df['cell_cycle_stage']=='G1']
        prev_df_G1 = prev_df_G1[~acdc_df.loc[prev_df_G1.index]['is_cell_dead']]
        IDsCellsG1 = set(prev_df_G1.index)
        if lastVisited or enforceAll:
            # If we are repeating auto cca for last visited frame
            # then we also add the cells in G1 that we already know
            # at current frame
            df_G1 = posData.cca_df[posData.cca_df['cell_cycle_stage']=='G1']
            IDsCellsG1.update(df_G1.index)

        # remove cells that disappeared
        IDsCellsG1 = [ID for ID in IDsCellsG1 if ID in posData.IDs]

        numCellsG1 = len(IDsCellsG1)
        numNewCells = len(posData.new_IDs)
        if numCellsG1 < numNewCells:
            self.highlightNewIDs_ccaFailed()
            msg = QMessageBox()
            warn_cca = msg.warning(
                self, 'No cells in G1!',
                f'In the next frame {numNewCells} new cells will '
                'appear (GREEN contour objects, left image).\n\n'
                f'However there are only {numCellsG1} cells '
                'in G1 available.\n\n'
                'You can either cancel the operation and "free" a cell '
                'by first annotating division on it or continue.\n\n'
                'If you continue the new cell will be annotated as a cell in G1 '
                'with unknown history.\n\n'
                'If you are not sure, before clicking "Yes" or "Cancel", you can '
                'preview (green contour objects, left image) '
                'where the new cells will appear.\n\n'
                'Do you want to continue?\n',
                msg.Yes | msg.Cancel
            )
            if warn_cca == msg.Yes:
                notEnoughG1Cells = False
                proceed = True
                # Annotate the new IDs with unknown history
                for ID in posData.new_IDs:
                    posData.cca_df.loc[ID] = pd.Series({
                        'cell_cycle_stage': 'G1',
                        'generation_num': 2,
                        'relative_ID': -1,
                        'relationship': 'mother',
                        'emerg_frame_i': -1,
                        'division_frame_i': -1,
                        'is_history_known': False,
                        'corrected_assignment': False
                    })
                    cca_df_ID = self.getStatusKnownHistoryBud(ID)
                    posData.ccaStatus_whenEmerged[ID] = cca_df_ID
            else:
                notEnoughG1Cells = True
                proceed = False
            return notEnoughG1Cells, proceed

        # Compute new IDs contours
        newIDs_contours = []
        for obj in posData.rp:
            ID = obj.label
            if ID in posData.new_IDs:
                cont = self.getObjContours(obj)
                newIDs_contours.append(cont)

        # Compute cost matrix
        cost = np.full((numCellsG1, numNewCells), np.inf)
        for obj in posData.rp:
            ID = obj.label
            if ID in IDsCellsG1:
                cont = self.getObjContours(obj)
                i = IDsCellsG1.index(ID)
                for j, newID_cont in enumerate(newIDs_contours):
                    min_dist, nearest_xy = self.nearest_point_2Dyx(
                        cont, newID_cont
                    )
                    cost[i, j] = min_dist

        # Run hungarian (munkres) assignment algorithm
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)

        # Assign buds to mothers
        for i, j in zip(row_idx, col_idx):
            mothID = IDsCellsG1[i]
            budID = posData.new_IDs[j]

            # If we are repeating assignment for the bud then we also have to
            # correct the possibily wrong mother first
            if budID in posData.cca_df.index:
                relID = posData.cca_df.at[budID, 'relative_ID']
                if relID in prev_cca_df.index:
                    posData.cca_df.loc[relID] = prev_cca_df.loc[relID]


            posData.cca_df.at[mothID, 'relative_ID'] = budID
            posData.cca_df.at[mothID, 'cell_cycle_stage'] = 'S'

            posData.cca_df.loc[budID] = pd.Series({
                'cell_cycle_stage': 'S',
                'generation_num': 0,
                'relative_ID': mothID,
                'relationship': 'bud',
                'emerg_frame_i': posData.frame_i,
                'division_frame_i': -1,
                'is_history_known': True,
                'corrected_assignment': False
            })


        # Keep only existing IDs
        posData.cca_df = posData.cca_df.loc[posData.IDs]

        self.store_cca_df()
        proceed = True
        return notEnoughG1Cells, proceed

    def getObjContours(self, obj, appendMultiContID=True):
        contours, _ = cv2.findContours(
           self.getObjImage(obj.image, obj.bbox).astype(np.uint8),
           cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.array([[np.nan, np.nan]])
        min_y, min_x, _, _ = self.getObjBbox(obj.bbox)
        cont = np.squeeze(contours[0], axis=1)
        if len(contours)>1 and appendMultiContID:
            posData = self.data[self.pos_i]
            if obj.label in posData.IDs:
                posData.multiContIDs.add(obj.label)
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        return cont

    def getObjBbox(self, obj_bbox):
        if self.isSegm3D and len(obj_bbox)==6:
            obj_bbox = (obj_bbox[1], obj_bbox[2], obj_bbox[4], obj_bbox[5])
            return obj_bbox
        else:
            return obj_bbox

    def z_lab(self):
        if self.isSegm3D:
            return self.zSliceScrollBarLab.sliderPosition()
        else:
            return None

    def get_2Dlab(self, lab, force_z=True):
        if self.isSegm3D:
            if force_z:
                return lab[self.z_lab()]
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if self.labBottomGroupbox.isChecked() or isZslice:
                return lab[self.z_lab()]
            else:
                return lab.max(axis=0)
        else:
            return lab

    def applyEraserMask(self, mask):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if self.labBottomGroupbox.isChecked() or isZslice:
                posData.lab[self.z_lab(), mask] = 0
            else:
                posData.lab[:, mask] = 0
        else:
            posData.lab[mask] = 0
    
    def changeBrushID(self):
        if not self.isSegm3D:
            # Changing brush ID with shift is only for 3D segm
            return

        if not self.brushButton.isChecked():
            # Brush if not active
            return
        
        if not self.isMouseDragImg2 and not self.isMouseDragImg1:
            # Mouse if not brushing at the moment
            return

        posData = self.data[self.pos_i]
        forceNewObj = not self.isNewID
        
        if forceNewObj:
            # Shift is down --> force new object with brush
            # e.g., 24 --> 28: 
            # 24 is hovering ID that we store as self.prevBrushID
            # 24 object becomes 28 that is the new posData.brushID
            self.isNewID = True
            self.changedID = posData.brushID
            self.restoreBrushID = posData.brushID
            # Set a new ID
            self.setBrushID()
            self.ax2BrushID = posData.brushID
        else:
            # Shift released or hovering on ID in z+-1 
            # --> restore brush ID from before shift was pressed or from 
            # when we started brushing from outside an object 
            # but we hovered on ID in z+-1 while dragging.
            # We change the entire 28 object to 24 so before changing the 
            # brush ID back to 24 we builg the mask with 28 to change it to 24
            self.isNewID = False
            self.changedID = posData.brushID
            # Restore ID   
            posData.brushID = self.restoreBrushID
            self.ax2BrushID = self.restoreBrushID
               
        brushID = posData.brushID
        brushIDmask = self.get_2Dlab(posData.lab) == self.changedID
        self.applyBrushMask(1,1,1,1, brushIDmask, brushID, isLocal=False)
        if self.isMouseDragImg1:
            self.brushColor = posData.lut[posData.brushID]/255
            self.setTempImg1Brush(brushIDmask)

    def applyBrushMask(self, ymin, xmin, ymax, xmax, mask, ID, isLocal=True):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if self.labBottomGroupbox.isChecked() or isZslice:
                if isLocal:
                    posData.lab[self.z_lab(), ymin:ymax, xmin:xmax][mask] = ID
                else:
                    posData.lab[self.z_lab()][mask] = ID
            else:
                if isLocal:
                    for z in range(len(posData.lab)):
                        posData.lab[z, ymin:ymax, xmin:xmax][mask] = ID
                else:
                    posData.lab[:, mask] = ID
        else:
            if isLocal:
                posData.lab[ymin:ymax, xmin:xmax][mask] = ID
            else:
                posData.lab[mask] = ID

    def get_2Drp(self, lab=None):
        if self.isSegm3D:
            if lab is None:
                # self.currentLab2D is defined at self.setImageImg2()
                lab = self.currentLab2D
            lab = self.get_2Dlab(lab)
            rp = skimage.measure.regionprops(lab)
            return rp
        else:
            return self.data[self.pos_i].rp

    def set_2Dlab(self, lab2D):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            posData.lab[self.z_lab()] = lab2D
        else:
            posData.lab = lab2D

    def get_labels(self, is_stored=False, frame_i=None, return_existing=False):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        existing = True
        if is_stored:
            labels = posData.allData_li[frame_i]['labels'].copy()
        else:
            try:
                labels = posData.segm_data[frame_i].copy()
            except IndexError:
                existing = False
                # Visting a frame that was not segmented --> empty masks
                if self.isSegm3D:
                    shape = (posData.SizeZ, posData.SizeY, posData.SizeX)
                else:
                    shape = (posData.SizeY, posData.SizeX)
                labels = np.zeros(shape, dtype=np.uint16)
        if return_existing:
            return labels, existing
        else:
            return labels

    def _get_editID_info(self, df):
        if 'was_manually_edited' not in df.columns:
            return []
        manually_edited_df = df[df['was_manually_edited'] > 0]
        editID_info = [
            (row.y_centroid, row.x_centroid, row.Index)
            for row in manually_edited_df.itertuples()
        ]
        return editID_info

    @get_data_exception_handler
    def get_data(self, debug=False):
        posData = self.data[self.pos_i]
        proceed_cca = True
        if posData.frame_i > 2:
            # Remove undo states from 4 frames back to avoid memory issues
            posData.UndoRedoStates[posData.frame_i-4] = []
            # Check if current frame contains undo states (not empty list)
            if posData.UndoRedoStates[posData.frame_i]:
                self.undoAction.setDisabled(False)
            else:
                self.undoAction.setDisabled(True)
        self.UndoCount = 0
        # If stored labels is None then it is the first time we visit this frame
        if posData.allData_li[posData.frame_i]['labels'] is None:
            posData.editID_info = []
            never_visited = True
            if str(self.modeComboBox.currentText()) == 'Cell cycle analysis':
                # Warn that we are visiting a frame that was never segm-checked
                # on cell cycle analysis mode
                msg = QMessageBox()
                warn_cca = msg.critical(
                    self, 'Never checked segmentation on requested frame',
                    'Segmentation and Tracking was never checked from '
                    f'frame {posData.frame_i+1} onward.\n To ensure correct cell '
                    'cell cycle analysis you have to first visit frames '
                    f'{posData.frame_i+1}-end with "Segmentation and Tracking" mode.',
                    msg.Ok
                )
                proceed_cca = False
                return proceed_cca, never_visited
            # Requested frame was never visited before. Load from HDD
            posData.lab = self.get_labels()
            posData.rp = skimage.measure.regionprops(posData.lab)
            if posData.acdc_df is not None:
                frames = posData.acdc_df.index.get_level_values(0)
                if posData.frame_i in frames:
                    # Since there was already segmentation metadata from
                    # previous closed session add it to current metadata
                    df = posData.acdc_df.loc[posData.frame_i].copy()
                    binnedIDs_df = df[df['is_cell_excluded']]
                    binnedIDs = set(binnedIDs_df.index).union(posData.binnedIDs)
                    posData.binnedIDs = binnedIDs
                    ripIDs_df = df[df['is_cell_dead']]
                    ripIDs = set(ripIDs_df.index).union(posData.ripIDs)
                    posData.ripIDs = ripIDs
                    posData.editID_info.extend(self._get_editID_info(df))
                    # Load cca df into current metadata
                    if 'cell_cycle_stage' in df.columns:
                        if any(df['cell_cycle_stage'].isna()):
                            if 'is_history_known' not in df.columns:
                                df['is_history_known'] = True
                            if 'corrected_assignment' not in df.columns:
                                df['corrected_assignment'] = True
                            df = df.drop(labels=self.cca_df_colnames, axis=1)
                        else:
                            # Convert to ints since there were NaN
                            cols = self.cca_df_int_cols
                            df[cols] = df[cols].astype(int)
                    i = posData.frame_i
                    posData.allData_li[i]['acdc_df'] = df.copy()
            self.get_cca_df()
        else:
            # Requested frame was already visited. Load from RAM.
            never_visited = False
            posData.lab = self.get_labels(is_stored=True)
            posData.rp = skimage.measure.regionprops(posData.lab)
            df = posData.allData_li[posData.frame_i]['acdc_df']
            binnedIDs_df = df[df['is_cell_excluded']]
            posData.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df['is_cell_dead']]
            posData.ripIDs = set(ripIDs_df.index)
            posData.editID_info = self._get_editID_info(df)
            self.get_cca_df()

        self.update_rp_metadata(draw=False)
        posData.IDs = [obj.label for obj in posData.rp]
        return proceed_cca, never_visited

    def load_delROIs_info(self, delROIshapes, last_tracked_num):
        posData = self.data[self.pos_i]
        delROIsInfo_npz = posData.delROIsInfo_npz
        if delROIsInfo_npz is None:
            return
        for file in posData.delROIsInfo_npz.files:
            if not file.startswith(f'{posData.frame_i}_'):
                continue

            delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
            if file.startswith(f'{posData.frame_i}_delMask'):
                delMask = delROIsInfo_npz[file]
                delROIs_info['delMasks'].append(delMask)
            elif file.startswith(f'{posData.frame_i}_delIDs'):
                delIDsROI = set(delROIsInfo_npz[file])
                delROIs_info['delIDsROI'].append(delIDsROI)
            elif file.startswith(f'{posData.frame_i}_roi'):
                Y, X = self.get_2Dlab(posData.lab).shape
                x0, y0, w, h = delROIsInfo_npz[file]
                addROI = (
                    posData.frame_i==0 or
                    [x0, y0, w, h] not in delROIshapes[posData.frame_i]
                )
                if addROI:
                    roi = self.getDelROI(xl=x0, yb=y0, w=w, h=h)
                    for i in range(posData.frame_i, last_tracked_num):
                        delROIs_info_i = posData.allData_li[i]['delROIs_info']
                        delROIs_info_i['rois'].append(roi)
                        delROIshapes[i].append([x0, y0, w, h])

    def addIDBaseCca_df(self, posData, ID):
        if ID <= 0:
            # When calling update_cca_df_deletedIDs we add relative IDs
            # but they could be -1 for cells in G1
            return

        _zip = zip(
            self.cca_df_colnames,
            self.cca_df_default_values,
        )
        if posData.cca_df.empty:
            posData.cca_df = pd.DataFrame(
                {col: val for col, val in _zip},
                index=[ID]
            )
        else:
            for col, val in _zip:
                posData.cca_df.at[ID, col] = val
        self.store_cca_df()

    def getBaseCca_df(self):
        posData = self.data[self.pos_i]
        IDs = [obj.label for obj in posData.rp]
        cc_stage = ['G1' for ID in IDs]
        num_cycles = [2]*len(IDs)
        relationship = ['mother' for ID in IDs]
        related_to = [-1]*len(IDs)
        emerg_frame_i = [-1]*len(IDs)
        division_frame_i = [-1]*len(IDs)
        is_history_known = [False]*len(IDs)
        corrected_assignment = [False]*len(IDs)
        cca_df = pd.DataFrame({
            'cell_cycle_stage': cc_stage,
            'generation_num': num_cycles,
            'relative_ID': related_to,
            'relationship': relationship,
            'emerg_frame_i': emerg_frame_i,
            'division_frame_i': division_frame_i,
            'is_history_known': is_history_known,
            'corrected_assignment': corrected_assignment
            },
            index=IDs
        )
        cca_df.index.name = 'Cell_ID'
        return cca_df

    def initSegmTrackMode(self):
        posData = self.data[self.pos_i]
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(posData.allData_li):
            # Build segm_npy
            lab = data_dict['labels']
            if lab is None and frame_i == 0:
                last_tracked_i = 0
                break
            elif lab is None:
                last_tracked_i = frame_i-1
                break
            else:
                last_tracked_i = posData.segmSizeT-1

        self.navigateScrollBar.setMaximum(last_tracked_i+1)
        if posData.frame_i > last_tracked_i:
            # Prompt user to go to last tracked frame
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                f'The last visited frame in "Segmentation and Tracking mode" '
                f'is frame {last_tracked_i+1}.\n\n'
                f'We recommend to resume from that frame.<br><br>'
                'How do you want to proceed?'
            )
            goToButton, stayButton = msg.warning(
                self, 'Go to last visited frame?', txt,
                buttonsTexts=(
                    f'Resume from frame {last_tracked_i+1} (RECOMMENDED)',
                    f'Stay on current frame {posData.frame_i+1}'
                )
            )
            if msg.clickedButton == goToButton:
                posData.frame_i = last_tracked_i
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            else:
                current_frame_i = posData.frame_i
                for i in range(current_frame_i):
                    posData.frame_i = i
                    self.get_data()
                    self.store_data()

                posData.frame_i = current_frame_i
                self.get_data()

        self.checkTrackingEnabled()

    def init_cca(self):
        posData = self.data[self.pos_i]
        currentMode = self.modeComboBox.currentText()
        if posData.last_tracked_i is None:
            txt = (
                'On this dataset either you never checked that the segmentation '
                'and tracking are correct or you did not save yet.\n\n'
                'If you already visited some frames with "Segmentation and tracking" '
                'mode save data before switching to "Cell cycle analysis mode".\n\n'
                'Otherwise you first have to check (and eventually correct) some frames '
                'in "Segmentation and Tracking" mode before proceeding '
                'with cell cycle analysis.')
            msg = QMessageBox()
            msg.critical(
                self, 'Tracking check not performed', txt, msg.Ok
            )
            self.modeComboBox.setCurrentText(currentMode)
            return

        proceed = True
        i = 0
        # Determine last annotated frame index
        for i, dict_frame_i in enumerate(posData.allData_li):
            df = dict_frame_i['acdc_df']
            if df is None:
                break
            else:
                if 'cell_cycle_stage' not in df.columns:
                    break

        last_cca_frame_i = i-1 if i>0 else 0

        if last_cca_frame_i == 0:
            # Remove undoable actions from segmentation mode
            posData.UndoRedoStates[0] = []
            self.undoAction.setEnabled(False)
            self.redoAction.setEnabled(False)

        if posData.frame_i > last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = QMessageBox()
            goTo_last_annotated_frame_i = msg.warning(
                self, 'Go to last annotated frame?',
                f'The last annotated frame is frame {last_cca_frame_i+1}.\n'
                'The cell cycle analysis will restart from that frame.\n'
                'Do you want to proceed?',
                msg.Yes | msg.Cancel
            )
            if goTo_last_annotated_frame_i == msg.Yes:
                msg = 'Looking good!'
                self.last_cca_frame_i = last_cca_frame_i
                posData.frame_i = last_cca_frame_i
                self.titleLabel.setText(msg, color=self.titleColor)
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            else:
                msg = 'Cell cycle analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.modeComboBox.setCurrentText(currentMode)
                proceed = False
                return
        elif posData.frame_i < last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets.myMessageBox()
            goTo_last_annotated_frame_i = msg.question(
                self, 'Go to last annotated frame?',
                f'The last annotated frame is frame {last_cca_frame_i+1}.\n'
                'Do you want to restart cell cycle analysis from frame '
                f'{last_cca_frame_i+1}?', buttonsTexts=('Yes', 'No', 'Cancel')
            )[0]
            if goTo_last_annotated_frame_i == msg.clickedButton:
                msg = 'Looking good!'
                self.titleLabel.setText(msg, color=self.titleColor)
                self.last_cca_frame_i = last_cca_frame_i
                posData.frame_i = last_cca_frame_i
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            elif msg.cancel:
                msg = 'Cell cycle analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.modeComboBox.setCurrentText(currentMode)
                proceed = False
                return
        else:
            self.get_data()

        self.last_cca_frame_i = last_cca_frame_i

        self.navigateScrollBar.setMaximum(last_cca_frame_i+1)

        if posData.cca_df is None:
            posData.cca_df = self.getBaseCca_df()
            msg = 'Cell cycle analysis initiliazed!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)
        else:
            self.get_cca_df()
        return proceed

    def remove_future_cca_df(self, from_frame_i):
        posData = self.data[self.pos_i]
        self.last_cca_frame_i = posData.frame_i
        self.setNavigateScrollBarMaximum()
        for i in range(from_frame_i, posData.SizeT):
            df = posData.allData_li[i]['acdc_df']
            if df is None:
                # No more saved info to delete
                return

            if 'cell_cycle_stage' not in df.columns:
                # No cell cycle info present
                continue

            df.drop(self.cca_df_colnames, axis=1, inplace=True)
            posData.allData_li[i]['acdc_df'] = df

    def get_cca_df(self, frame_i=None, return_df=False):
        # cca_df is None unless the metadata contains cell cycle annotations
        # NOTE: cell cycle annotations are either from the current session
        # or loaded from HDD in "initPosAttr" with a .question to the user
        posData = self.data[self.pos_i]
        cca_df = None
        i = posData.frame_i if frame_i is None else frame_i
        df = posData.allData_li[i]['acdc_df']
        if df is not None:
            if 'cell_cycle_stage' in df.columns:
                if 'is_history_known' not in df.columns:
                    df['is_history_known'] = True
                if 'corrected_assignment' not in df.columns:
                    # Compatibility with those acdc_df analysed with prev vers.
                    df['corrected_assignment'] = True
                cca_df = df[self.cca_df_colnames].copy()
        if cca_df is None and self.isSnapshot:
            cca_df = self.getBaseCca_df()
            posData.cca_df = cca_df
        if return_df:
            return cca_df
        else:
            posData.cca_df = cca_df

    def unstore_cca_df(self):
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        for col in self.cca_df_colnames:
            if col not in acdc_df.columns:
                continue
            acdc_df.drop(col, axis=1, inplace=True)

    def store_cca_df(self, pos_i=None, frame_i=None, cca_df=None, mainThread=True):
        pos_i = self.pos_i if pos_i is None else pos_i
        posData = self.data[pos_i]
        i = posData.frame_i if frame_i is None else frame_i
        if cca_df is None:
            cca_df = posData.cca_df
            if self.ccaTableWin is not None and mainThread:
                self.ccaTableWin.updateTable(posData.cca_df)

        acdc_df = posData.allData_li[i]['acdc_df']
        if acdc_df is None:
            self.store_data()
            acdc_df = posData.allData_li[i]['acdc_df']
        if 'cell_cycle_stage' in acdc_df.columns:
            # Cell cycle info already present --> overwrite with new
            df = acdc_df
            df[self.cca_df_colnames] = cca_df
            posData.allData_li[i]['acdc_df'] = df.copy()
        elif cca_df is not None:
            df = acdc_df.join(cca_df, how='left')
            posData.allData_li[i]['acdc_df'] = df.copy()

    def ax1_setTextID(self, obj, how, updateColor=False, debug=False):
        posData = self.data[self.pos_i]
        # Draw ID label on ax1 image depending on how
        LabelItemID = self.ax1_LabelItemsIDs[obj.label-1]
        ID = obj.label
        df = posData.cca_df
        if df is None or how.find('cell cycle') == -1:
            txt = f'{ID}'
            if updateColor:
                LabelItemID.setText(txt, size=self.fontSize)
            if ID in posData.new_IDs:
                color = 'r'
                bold = True
            else:
                color = self.ax1_oldIDcolor
                bold = False
        else:
            df_ID = df.loc[ID]
            ccs = df_ID['cell_cycle_stage']
            relationship = df_ID['relationship']
            generation_num = int(df_ID['generation_num'])
            generation_num = 'ND' if generation_num==-1 else generation_num
            emerg_frame_i = int(df_ID['emerg_frame_i'])
            is_history_known = df_ID['is_history_known']
            is_bud = relationship == 'bud'
            is_moth = relationship == 'mother'
            emerged_now = emerg_frame_i == posData.frame_i

            # Check if the cell has already annotated division in the future
            # to use orange instead of red
            is_division_annotated = False
            if ccs == 'S' and is_bud and not self.isSnapshot:
                for i in range(posData.frame_i+1, posData.SizeT):
                    cca_df = self.get_cca_df(frame_i=i, return_df=True)
                    if cca_df is None:
                        break

                    if ID not in cca_df.index:
                        continue

                    _ccs = cca_df.at[ID, 'cell_cycle_stage']
                    if _ccs == 'G1':
                        is_division_annotated = True
                        break

            mothCell_S = (
                ccs == 'S'
                and is_moth
                and not emerged_now
                and not is_division_annotated
            )

            budNotEmergedNow = (
                ccs == 'S'
                and is_bud
                and not emerged_now
                and not is_division_annotated
            )

            budEmergedNow = (
                ccs == 'S'
                and is_bud
                and emerged_now
                and not is_division_annotated
            )

            txt = f'{ccs}-{generation_num}'
            if updateColor:
                LabelItemID.setText(txt, size=self.fontSize)
            if ccs == 'G1':
                color = self.ax1_G1cellColor
                bold = False
            elif mothCell_S:
                color = self.ax1_S_oldCellColor
                bold = False
            elif budNotEmergedNow:
                color = 'r'
                bold = False
            elif budEmergedNow:
                color = 'r'
                bold = True
            elif is_division_annotated:
                color = self.ax1_divAnnotColor
                bold = False

            if not is_history_known:
                txt = f'{txt}?'

        if not self.isObjVisible(obj.bbox):
            # Object not visible (entire bbox in another z_range)
            LabelItemID.setText('')
            return
        elif self.isSegm3D:
            if obj.label not in self.currentLab2D:
                # Object is present in z+1 and z-1 but not in z --> transparent
                r,g,b = self.ax1_oldIDcolor
                color = QColor(r,g,b,100)
                LabelItemID.setText(txt, color=color, size=self.fontSize)
                self.setLabelCenteredObject(obj, LabelItemID)
                return

        try:
            if debug:
                print(txt, color)
            LabelItemID.setText(
                txt, color=color, bold=bold, size=self.fontSize
            )
        except UnboundLocalError:
            pass

        self.setLabelCenteredObject(obj, LabelItemID)

    def setLabelCenteredObject(self, obj, LabelItemID):
        # Center LabelItem at centroid
        y, x = self.getObjCentroid(obj.centroid)
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

    def getObjCentroid(self, obj_centroid):
        if self.isSegm3D:
            return obj_centroid[1:3]
        else:
            return obj_centroid

    def ax2_setTextID(self, obj):
        posData = self.data[self.pos_i]
        # Draw ID label on ax1 image
        LabelItemID = self.ax2_LabelItemsIDs[obj.label-1]
        ID = obj.label
        df = posData.cca_df
        txt = f'{ID}' if self.isObjVisible(obj.bbox) else ''
        color = self.ax2_textColor
        bold = ID in posData.new_IDs

        LabelItemID.setText(txt, color=color, bold=bold, size=self.fontSize)

        # Center LabelItem at centroid
        y, x = self.getObjCentroid(obj.centroid)
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)


    def drawID_and_Contour(self, obj, drawContours=True, updateColor=False):
        posData = self.data[self.pos_i]
        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'
        IDs_and_masks = how == 'Draw IDs and overlay segm. masks'
        onlyMasks = how == 'Draw only overlay segm. masks'
        ccaInfo_and_masks = how == 'Draw cell cycle info and overlay segm. masks'

        # Draw LabelItems for IDs on ax2
        y, x = self.getObjCentroid(obj.centroid)
        idx = obj.label-1
        t0 = time.time()
        self.ax2_setTextID(obj)

        if posData.cca_df is not None and self.isSnapshot:
            if obj.label not in posData.cca_df.index:
                self.store_data()
                self.addIDBaseCca_df(posData, obj.label)
                self.store_cca_df()

        # Draw LabelItems for IDs on ax1 if requested
        draw_LIs = (
            IDs_and_cont or onlyIDs or only_ccaInfo or ccaInfo_and_cont
            or IDs_and_masks or ccaInfo_and_masks
        )
        if draw_LIs:
            # Draw LabelItems for IDs on ax2
            # t0 = time.time()
            self.ax1_setTextID(obj, how, updateColor=updateColor)

        # t1 = time.time()
        # self.drawingLabelsTimes.append(t1-t0)

        # Draw line connecting mother and buds
        drawLines = (
            only_ccaInfo or ccaInfo_and_cont or onlyMothBudLines
            or ccaInfo_and_masks
        )
        if drawLines and posData.cca_df is not None:
            ID = obj.label
            BudMothLine = self.ax1_BudMothLines[ID-1]
            cca_df_ID = posData.cca_df.loc[ID]
            ccs_ID = cca_df_ID['cell_cycle_stage']
            relationship = cca_df_ID['relationship']
            isObjVisible = self.isObjVisible(obj.bbox)
            if ccs_ID == 'S' and relationship=='bud' and isObjVisible:
                emerg_frame_i = cca_df_ID['emerg_frame_i']
                if emerg_frame_i == posData.frame_i:
                    pen = self.NewBudMoth_Pen
                else:
                    pen = self.OldBudMoth_Pen
                relative_ID = cca_df_ID['relative_ID']
                if relative_ID in posData.IDs:
                    relative_rp_idx = posData.IDs.index(relative_ID)
                    relative_ID_obj = posData.rp[relative_rp_idx]
                    y1, x1 = self.getObjCentroid(obj.centroid)
                    y2, x2 = self.getObjCentroid(relative_ID_obj.centroid)
                    BudMothLine.setData([x1, x2], [y1, y2], pen=pen)
            else:
                BudMothLine.setData([], [])

        if not drawContours:
            return

        # Draw contours on ax1 if requested
        if IDs_and_cont or onlyCont or ccaInfo_and_cont:
            if not self.isObjVisible(obj.bbox):
                curveID = self.ax1_ContoursCurves[idx]
                curveID.setData([], [])
            else:
                ID = obj.label
                # t0 = time.time()
                cont = self.getObjContours(obj)
                # t1 = time.time()
                # computingContoursTime = t1-t0
                # self.computingContoursTimes.append(computingContoursTime)

                # t0 = time.time()
                curveID = self.ax1_ContoursCurves[idx]
                pen = (
                    self.newIDs_cpen if ID in posData.new_IDs
                    else self.oldIDs_cpen
                )
                curveID.setData(cont[:,0], cont[:,1], pen=pen)
                # t1 = time.time()
                # drawingContoursTimes = t1-t0
                # self.drawingContoursTimes.append(drawingContoursTimes)


    def update_rp(self, draw=True, debug=False):
        posData = self.data[self.pos_i]
        # Update rp for current posData.lab (e.g. after any change)
        posData.rp = skimage.measure.regionprops(posData.lab)
        posData.IDs = [obj.label for obj in posData.rp]
        self.update_rp_metadata(draw=draw)

    def update_IDsContours(self, prev_IDs, newIDs=[]):
        """Function to draw labels text and contours of specific IDs.
        It should speed up things because we draw only few IDs usually.

        Parameters
        ----------
        prev_IDs : list
            List of IDs before the change (e.g. before erasing ID or painting
            a new one etc.)
        newIDs : bool
            List of new IDs that are present after a change.

        Returns
        -------
        None.

        """

        posData = self.data[self.pos_i]

        # Draw label and contours of the new IDs
        if len(newIDs)>0:
            for i, obj in enumerate(posData.rp):
                ID = obj.label
                if ID in newIDs:
                    # Draw ID labels and contours of new objects
                    self.drawID_and_Contour(obj)

        # Clear contours and LabelItems of IDs that are in prev_IDs
        # but not in current IDs
        currentIDs = [obj.label for obj in posData.rp]
        for prevID in prev_IDs:
            if prevID not in currentIDs:
                self.ax1_ContoursCurves[prevID-1].setData([], [])
                self.ax1_LabelItemsIDs[prevID-1].setText('')
                self.ax2_LabelItemsIDs[prevID-1].setText('')

        self.highlightLostNew()
        self.setTitleText()
        self.highlightmultiBudMoth()

    def highlightmultiBudMoth(self):
        posData = self.data[self.pos_i]
        for ID in posData.multiBud_mothIDs:
            LabelItemID = self.ax1_LabelItemsIDs[ID-1]
            txt = LabelItemID
            LabelItemID.setText(f'{txt} !!', color=self.lostIDs_qMcolor)

    def extendLabelsLUT(self, lenNewLut):
        posData = self.data[self.pos_i]
        # Build a new lut to include IDs > than original len of lut
        if lenNewLut > len(posData.lut):
            numNewColors = lenNewLut-len(posData.lut)
            # Index original lut
            _lut = np.zeros((lenNewLut, 3), np.uint8)
            _lut[:len(posData.lut)] = posData.lut
            # Pick random colors and append them at the end to recycle them
            randomIdx = np.random.randint(0,len(posData.lut),size=numNewColors)
            for i, idx in enumerate(randomIdx):
                rgb = posData.lut[idx]
                _lut[len(posData.lut)+i] = rgb
            posData.lut = _lut
            return True
        return False

    def initLookupTableLab(self):
        posData = self.data[self.pos_i]
        self.img2.setLookupTable(posData.lut)
        self.img2.setLevels([0, len(posData.lut)])

    def updateLookuptable(self, lenNewLut=None, delIDs=None):
        posData = self.data[self.pos_i]
        if lenNewLut is None:
            try:
                if delIDs is None:
                    IDs = posData.IDs
                else:
                    # Remove IDs removed with ROI from LUT
                    IDs = [ID for ID in posData.IDs if ID not in delIDs]
                lenNewLut = max(IDs)+1
            except ValueError:
                # Empty segmentation mask
                lenNewLut = 1
        # Build a new lut to include IDs > than original len of lut
        updateLevels = self.extendLabelsLUT(lenNewLut)
        lut = posData.lut.copy()

        try:
            # lut = posData.lut[:lenNewLut].copy()
            for ID in posData.binnedIDs:
                lut[ID] = lut[ID]*0.2

            for ID in posData.ripIDs:
                lut[ID] = lut[ID]*0.2
        except Exception as e:
            err_str = traceback.format_exc()
            print('='*30)
            self.logger.info(err_str)
            print('='*30)

        if updateLevels:
            self.img2.setLevels([0, len(lut)])
        self.img2.setLookupTable(lut)


    def update_rp_metadata(self, draw=True):
        posData = self.data[self.pos_i]
        binnedIDs_xx = []
        binnedIDs_yy = []
        ripIDs_xx = []
        ripIDs_yy = []
        # Add to rp dynamic metadata (e.g. cells annotated as dead)
        for i, obj in enumerate(posData.rp):
            ID = obj.label
            # IDs removed from analysis --> store info
            if ID in posData.binnedIDs:
                obj.excluded = True
                if draw:
                    y, x = self.getObjCentroid(obj.centroid)
                    # Gray out ID label on image
                    LabelID = self.ax2_LabelItemsIDs[ID-1]
                    LabelID.setText(f'{ID}', color=(150, 0, 0))
                    binnedIDs_xx.append(x)
                    binnedIDs_yy.append(y)
            else:
                obj.excluded = False

            # IDs dead --> store info
            if ID in posData.ripIDs:
                obj.dead = True
                if draw:
                    # Gray out ID label on image
                    y, x = self.getObjCentroid(obj.centroid)
                    LabelID = self.ax2_LabelItemsIDs[ID-1]
                    LabelID.setText(f'{ID}', color=(150, 0, 0))
                    ripIDs_xx.append(x)
                    ripIDs_yy.append(y)
            else:
                obj.dead = False

            # set cell cycle info

        if draw:
            # Draw markers to annotated IDs
            self.ax2_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
            self.ax2_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)
            self.ax1_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
            self.ax1_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)

    def loadNonAlignedFluoChannel(self, fluo_path):
        posData = self.data[self.pos_i]
        if posData.filename.find('aligned') != -1:
            filename, _ = os.path.splitext(os.path.basename(fluo_path))
            path = f'.../{posData.pos_foldername}/Images/{filename}_aligned.npz'
            msg = QMessageBox()
            msg.critical(
                self, 'Aligned fluo channel not found!',
                'Aligned data for fluorescent channel not found!\n\n'
                f'You loaded aligned data for the cells channel, therefore '
                'loading NON-aligned fluorescent data is not allowed.\n\n'
                'Run the script "dataPrep.py" to create the following file:\n\n'
                f'{path}',
                msg.Ok
            )
            return None
        fluo_data = skimage.io.imread(fluo_path)
        return fluo_data

    def load_fluo_data(self, fluo_path):
        self.logger.info(f'Loading fluorescent image data from "{fluo_path}"...')
        bkgrData = None
        posData = self.data[self.pos_i]
        # Load overlay frames and align if needed
        filename = os.path.basename(fluo_path)
        filename_noEXT, ext = os.path.splitext(filename)
        if ext == '.npy' or ext == '.npz':
            fluo_data = np.load(fluo_path)
            try:
                fluo_data = fluo_data['arr_0']
            except Exception as e:
                fluo_data = fluo_data

            # Load background data
            bkgrData_path = os.path.join(
                posData.images_path, f'{filename_noEXT}_bkgrRoiData.npz'
            )
            if os.path.exists(bkgrData_path):
                bkgrData = np.load(bkgrData_path)
        elif ext == '.tif' or ext == '.tiff':
            aligned_filename = f'{filename_noEXT}_aligned.npz'
            aligned_path = os.path.join(posData.images_path, aligned_filename)
            if os.path.exists(aligned_path):
                fluo_data = np.load(aligned_path)['arr_0']

                # Load background data
                bkgrData_path = os.path.join(
                    posData.images_path, f'{aligned_filename}_bkgrRoiData.npz'
                )
                if os.path.exists(bkgrData_path):
                    bkgrData = np.load(bkgrData_path)
            else:
                fluo_data = self.loadNonAlignedFluoChannel(fluo_path)
                if fluo_data is None:
                    return None, None

                # Load background data
                bkgrData_path = os.path.join(
                    posData.images_path, f'{filename_noEXT}_bkgrRoiData.npz'
                )
                if os.path.exists(bkgrData_path):
                    bkgrData = np.load(bkgrData_path)
        else:
            txt = (f'File format {ext} is not supported!\n'
                    'Choose either .tif or .npz files.')
            msg = QMessageBox()
            msg.critical(
                self, 'File not supported', txt, msg.Ok
            )
            return None, None

        return fluo_data, bkgrData

    def setOverlayColors(self):
        self.overlayRGBs = [(255, 255, 0),
                            (252, 72, 254),
                            (49, 222, 134),
                            (22, 108, 27)]

    def getFileExtensions(self, images_path):
        alignedFound = any([f.find('_aligned.np')!=-1
                            for f in myutils.listdir(images_path)])
        if alignedFound:
            extensions = (
                'Aligned channels (*npz *npy);; Tif channels(*tiff *tif)'
                ';;All Files (*)'
            )
        else:
            extensions = (
                'Tif channels(*tiff *tif);; All Files (*)'
            )
        return extensions

    def loadOverlayData(self, ol_channels):
        posData = self.data[self.pos_i]
        for ol_ch in ol_channels:
            if ol_ch not in list(posData.loadedFluoChannels):
                # Requested channel was never loaded --> load it at first
                # iter i == 0
                success = self.loadFluo_cb(fluo_channels=[ol_ch])
                if not success:
                    return False

        for posData in self.data:
            ol_data = {}
            ol_colors = {}
            for i, ol_ch in enumerate(ol_channels):
                fluo_path, filename = self.getPathFromChName(ol_ch, posData)
                ol_data[filename] = posData.ol_data_dict[filename].copy()
                ol_colors[filename] = self.overlayRGBs[i]
                posData.ol_colors = ol_colors
                if i!=0:
                    continue
                # Add first loaded channel to context menu
                # Since we gave up on allowing multiple overlay channels
                self.addFluoChNameContextMenuAction(ol_ch)
            posData.manualContrastKey = filename
            posData.ol_data = ol_data
        return True

    def askSelectOverlayChannel(self):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        selectFluo = apps.QDialogListbox(
            'Select channel',
            'Select channel names to overlay:\n',
            ch_names, multiSelection=False, parent=self
        )
        selectFluo.exec_()
        if selectFluo.cancel:
            return False

        ol_channels = selectFluo.selectedItemsText
        success = self.loadOverlayData(ol_channels)
        if not success:
            return False

        return True

    def overlay_cb(self, checked):
        self.UserNormAction, _, _ = self.getCheckNormAction()
        posData = self.data[self.pos_i]
        if checked:
            if posData.ol_data is None:
                success = self.askSelectOverlayChannel()
                if not success:
                    self.overlayButton.toggled.disconnect()
                    self.overlayButton.setChecked(False)
                    self.overlayButton.toggled.connect(self.overlay_cb)
                    return

            self.normalizeRescale0to1Action.setChecked(True)
            self.imgGrad.imageItem = lambda: None
            self.updateImageGradientItem(self.img1)

            rgb = self.df_settings.at['overlayColor', 'value']
            rgb = [int(v) for v in rgb.split('-')]
            self.overlayColorButton.setColor(rgb)
            self.setCheckedOverlayContextMenusAction()
            self.updateALLimg(only_ax1=True)
            self.enableOverlayWidgets(True)
        else:
            self.UserNormAction.setChecked(True)
            self.imgGrad.gradient.menu.removeAction(self.userChNameAction)
            for action in posData.fluoDataChNameActions:
                self.imgGrad.gradient.menu.removeAction(action)
            self.create_chNamesQActionGroup(self.user_ch_name)
            # posData.fluoDataChNameActions = []
            self.updateImageGradientItem(self.img1)
            self.updateALLimg(only_ax1=True)
            self.enableOverlayWidgets(False)

    def showOverlayContextMenu(self, event):
        if not self.overlayButton.isChecked():
            return

        self.overlayContextMenu.exec_(QCursor.pos())

    def showInstructionsCustomModel(self):
        txt, models_path = myutils.get_add_custom_model_instructions()
        msg = widgets.myMessageBox(showCentered=False)
        msg.addShowInFileManagerButton(models_path, txt='Open models folder...')
        msg.information(
            self, 'Custom model instructions', txt, buttonsTexts=('Ok',)
        )

    def changeOverlayChannel(self, action):
        posData = self.data[self.pos_i]
        loadedChannels = list(posData.loadedFluoChannels)
        checkedChName = action.text()
        channels = [checkedChName]
        success = self.loadOverlayData(channels)
        if not success:
            return
        self.overlay_cb(True)

    def setCheckedOverlayContextMenusAction(self):
        posData = self.data[self.pos_i]
        self.userChNameAction.setChecked(False)
        for action in posData.fluoDataChNameActions:
            action.setChecked(False)
        checkedActionText = posData.manualContrastKey[len(posData.basename):]
        if self.userChNameAction.text() == checkedActionText:
            self.userChNameAction.setChecked(True)
        else:
            for action in posData.fluoDataChNameActions:
                if action.text() == checkedActionText:
                    action.setChecked(True)
                    break

        actionGroup = self.overlayContextMenu.actionGroup
        actionGroup.triggered.disconnect()
        for action in self.overlayContextMenu.actions():
            action.setChecked(False)
            if action.text() == checkedActionText:
                action.setChecked(True)
        actionGroup = self.overlayContextMenu.actionGroup
        actionGroup.triggered.connect(self.changeOverlayChannel)

    def enableOverlayWidgets(self, enabled):
        posData = self.data[self.pos_i]
        if enabled:
            self.alphaScrollBar.setDisabled(False)
            self.overlayColorButton.setDisabled(False)
            self.editOverlayColorAction.setDisabled(False)
            self.alphaScrollBar.show()
            self.alphaScrollBar_label.show()

            if posData.SizeZ == 1:
                return

            self.zSliceOverlay_SB.setMaximum(posData.SizeZ-1)
            if self.zProjOverlay_CB.currentText().find('max') != -1:
                self.overlay_z_label.setStyleSheet('color: gray')
                self.zSliceOverlay_SB.setDisabled(True)
            else:
                z = self.zSliceOverlay_SB.sliderPosition()
                self.overlay_z_label.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')
                self.zSliceOverlay_SB.setDisabled(False)
                self.overlay_z_label.setStyleSheet('color: black')
            self.zSliceOverlay_SB.show()
            self.overlay_z_label.show()
            self.zProjOverlay_CB.show()
            self.zSliceOverlay_SB.valueChanged.connect(self.update_overlay_z_slice)
            self.zProjOverlay_CB.currentTextChanged.connect(self.updateOverlayZproj)
            self.zProjOverlay_CB.activated.connect(self.clearComboBoxFocus)
        else:
            self.zSliceOverlay_SB.setDisabled(True)
            self.zSliceOverlay_SB.hide()
            self.overlay_z_label.hide()
            self.zProjOverlay_CB.hide()
            self.alphaScrollBar.setDisabled(True)
            self.overlayColorButton.setDisabled(True)
            self.editOverlayColorAction.setDisabled(True)
            self.alphaScrollBar.hide()
            self.alphaScrollBar_label.hide()

            if posData.SizeZ == 1:
                return

            self.zSliceOverlay_SB.valueChanged.disconnect()
            self.zProjOverlay_CB.currentTextChanged.disconnect()
            self.zProjOverlay_CB.activated.disconnect()


    def criticalFluoChannelNotFound(self, fluo_ch, posData):
        msg = widgets.myMessageBox(showCentered=False)
        ls = "\n".join(myutils.listdir(posData.images_path))
        msg.setDetailedText(
            f'Files present in the {posData.relPath} folder:\n'
            f'{ls}'
        )
        title = 'Requested channel data not found!'
        txt = html_utils.paragraph(
            f'The folder <code>{posData.pos_path}</code> '
            '<b>does not contain</b> '
            'either one of the following files:<br><br>'
            f'{posData.basename}_{fluo_ch}.tif<br>'
            f'{posData.basename}_{fluo_ch}_aligned.npz<br><br>'
            'Data loading aborted.'
        )
        msg.addShowInFileManagerButton(posData.images_path)
        okButton = msg.warning(
            self, title, txt, buttonsTexts=('Ok')
        )

    def imgGradLUT_cb(self, LUTitem):
        # Callback function for the histogram sliders moved by the user
        # Store the histogram levels that the user is manually changing
        # i.e. moving the gradient slider ticks up and down
        # Store them for all frames
        posData = self.data[self.pos_i]
        isOverlayON = self.overlayButton.isChecked()
        min = self.imgGrad.gradient.listTicks()[0][1]
        max = self.imgGrad.gradient.listTicks()[1][1]
        isRGB = (
            self.drawIDsContComboBox.currentText().find('segm. masks') != -1
            or self.imgCmapName != 'grey'
        )
        if isOverlayON:
            for i in range(0, posData.SizeT):
                histoLevels = posData.allData_li[i]['histoLevels']
                histoLevels[posData.manualContrastKey] = (min, max)
            if posData.ol_data is not None:
                self.getOverlayImg(setImg=True)
        elif isRGB:
            cellsKey = f'{self.user_ch_name}_overlayOFF'
            for i in range(0, posData.SizeT):
                histoLevels = posData.allData_li[i]['histoLevels']
                histoLevels[cellsKey] = (min, max)
            img = self.getImageWithCmap()
            img = self.overlaySegmMasks(img)
            self.img1.setImage(img)
        else:
            cellsKey = f'{self.user_ch_name}_overlayOFF'
            for i in range(0, posData.SizeT):
                histoLevels = posData.allData_li[i]['histoLevels']
                histoLevels[cellsKey] = (min, max)
            img = self.getImage()
            if self.imgGrad.gradient.isLookupTrivial():
                self.img1.setLookupTable(None)
            else:
                self.img1.setLookupTable(self.imgGrad.getLookupTable(img=img))

    def updateOnlyImg(self):
        isSegm2D = not self.isSegm3D
        return isSegm2D or self.labBottomGroupbox.isChecked()

    def imgGradLUTfinished_cb(self):
        if not self.overlayButton.isChecked():
            self.updateALLimg(only_ax1=True, updateFilters=True)

    def updateContColour(self, colorButton):
        color = colorButton.color().getRgb()
        self.df_settings.at['contLineColor', 'value'] = str(color)
        self.gui_createContourPens()
        self.updateALLimg()

    def saveContColour(self, colorButton):
        self.df_settings.to_csv(self.settings_csv_path)

    def contLineWeightToggled(self, checked=True):
        self.imgGrad.uncheckContLineWeightActions()
        w = self.sender().lineWeight
        self.df_settings.at['contLineWeight', 'value'] = w
        self.df_settings.to_csv(self.settings_csv_path)
        self.gui_createContourPens()
        self.updateALLimg()
        for act in self.imgGrad.contLineWightActionGroup.actions():
            if act == self.sender():
                act.setChecked(True)
            act.toggled.connect(self.contLineWeightToggled)


    def gradientCmapContextMenuClicked(self, b=None):
        act = self.sender()
        self.df_settings.at['img_cmap', 'value'] = act.name
        self.df_settings.to_csv(self.settings_csv_path)
        self.imgCmap = self.imgGrad.cmaps[act.name]
        self.imgCmapName = act.name
        self.updateALLimg()

    def adjustBrightness(
            self, img, key, func=skimage.exposure.rescale_intensity
        ):
        """
        Adjust contrast/brightness of the image selected in the histogram
        context menu using stored levels.
        The levels are stored in imgGradLUT_cb function which is called when
        the user changes the gradient slider levels.
        Note that the gradient always returns values from 0 to 1 so we
        need to scale to the actual max min of the image.
        """
        posData = self.data[self.pos_i]
        histoLevels = posData.allData_li[posData.frame_i]['histoLevels']
        rescaled_img = img
        for name in histoLevels:
            if name != key:
                continue

            minPerc, maxPerc = histoLevels[name]
            if minPerc == 0 and maxPerc == 1:
                rescaled_img = img
            else:
                imgRange = np.max(img)-np.min(img)
                min = np.min(img) + imgRange*minPerc
                max = np.min(img) + imgRange*maxPerc
                in_range = (min, max)
                rescaled_img = func(
                    rescaled_img, in_range=in_range#, out_range=out_range
                )
        return rescaled_img

    def getOlImg(self, key, normalizeIntens=True, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        img = posData.ol_data[key][frame_i]
        if posData.SizeZ > 1:
            zProjHow = self.zProjOverlay_CB.currentText()
            z = self.zSliceOverlay_SB.sliderPosition()
            if zProjHow == 'same as above':
                zProjHow = self.zProjComboBox.currentText()
                z = self.zSliceScrollBar.sliderPosition()
                reconnect = False
                try:
                    self.zSliceOverlay_SB.valueChanged.disconnect()
                    reconnect = True
                except TypeError:
                    pass
                self.zSliceOverlay_SB.setSliderPosition(z)
                if reconnect:
                    self.zSliceOverlay_SB.valueChanged.connect(self.update_z_slice)
            if zProjHow == 'single z-slice':
                self.overlay_z_label.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')
                ol_img = img[z].copy()
            elif zProjHow == 'max z-projection':
                ol_img = img.max(axis=0).copy()
            elif zProjHow == 'mean z-projection':
                ol_img = img.mean(axis=0).copy()
            elif zProjHow == 'median z-proj.':
                ol_img = np.median(img, axis=0).copy()
        else:
            ol_img = img.copy()

        if normalizeIntens:
            ol_img = self.normalizeIntensities(ol_img)
        return ol_img

    def setLookupTableImg(self, img):
        pass

    def overlaySegmMasks(self, img, force=False):
        how = self.drawIDsContComboBox.currentText()
        if how.find('overlay segm. masks') == -1 and not force:
            return img

        alpha = self.imgGrad.labelsAlphaSlider.value()
        if alpha == 0:
            return img

        posData = self.data[self.pos_i]
        if posData.IDs:
            maxID = max(posData.IDs)
        else:
            maxID = 0

        if maxID >= len(posData.lut):
            self.extendLabelsLUT(maxID+10)
        colors = [posData.lut[ID]/255 for ID in posData.IDs]

        # get bkgr color
        if 'labels_text_color' in self.df_settings.index:
            rgbString = self.df_settings.at['labels_text_color', 'value']
            r, g, b = myutils.rgb_str_to_values(rgbString)
            r, g, b = r/255, g/255, b/255
        else:
            r, g, b = 0.1, 0.1, 0.1

        self.bg_color = (r, g, b)

        if img.ndim == 2:
            img = img/np.max(img)
            self.img1_RGB = gray2rgb(img)
            # NOTE: img_layer0 defined in getImageWithCmap()
        elif self.overlayButton.isChecked():
            # overlay fluo is ON --> image is already RGB
            self.img1_RGB = img
            self.img_layer0 = self.ol_cells_img
        else:
            self.img1_RGB = img
            # NOTE: img_layer0 defined in getImageWithCmap()

        # Check if RGB is 0,1 or 0,255 and convert accordingly
        val = self.img1_RGB[tuple([0]*self.img1_RGB.ndim)]
        if not isinstance(val, (np.floating, float)):
            self.img1uintRGB = self.img1_RGB.copy()
            self.img1_RGB = self.img1_RGB/255
        else:
            self.img1uintRGB = (self.img1_RGB*255).astype(np.uint8)

        if posData.rp is None:
            posData.rp = skimage.measure.regionprops(posData.lab)

        imgRGB = self.img1_RGB.copy()

        for obj in posData.rp:
            if not self.isObjVisible(obj.bbox, debug=False):
                continue

            color = posData.lut[obj.label]/255
            _slice = self.getObjSlice(obj.slice)
            _objMask = self.getObjImage(obj.image, obj.bbox)
            bkgr_label = self.img1_RGB[_slice][_objMask]
            # colored_label = bkgr_label*color
            overlay = bkgr_label*(1.0-alpha) + color*alpha
            imgRGB[_slice][_objMask] = overlay
        imgRGB = (np.clip(imgRGB, 0, 1)*255).astype(np.uint8)
        return imgRGB

    def isObjVisible(self, obj_bbox, debug=False):
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if not self.labBottomGroupbox.isChecked() and not isZslice:
                # required a projection --> all obj are visible
                return True
            min_z = obj_bbox[0]
            max_z = obj_bbox[3]-1
            if self.z_lab()>=min_z and self.z_lab()<=max_z:
                return True
            else:
                return False
        else:
            return True

    def getObjImage(self, obj_image, obj_bbox):
        if self.isSegm3D and len(obj_bbox)==6:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if not self.labBottomGroupbox.isChecked() and not isZslice:
                # required a projection
                return obj_image.max(axis=0)

            min_z = obj_bbox[0]
            z = self.z_lab()
            local_z = z - min_z
            return obj_image[local_z]
        else:
            return obj_image

    def getObjSlice(self, obj_slice):
        if self.isSegm3D:
            return obj_slice[1:3]
        else:
            return obj_slice

    def getOverlayImg(self, fluoData=None, setImg=True, frame_i=None):
        posData = self.data[self.pos_i]
        keys = list(posData.ol_data.keys())

        # Cells channel (e.g. phase_contrast)
        cells_img = self.getImage(invert=False, frame_i=frame_i)

        img = self.adjustBrightness(cells_img, posData.filename)
        self.ol_cells_img = img
        if self.imgCmapName == 'grey':
            gray_img_rgb = gray2rgb(img)
        else:
            gray_img_rgb = self.imgCmap(img)[:, :, :3]

        # First fluo channel
        ol_img = self.getOlImg(keys[0], frame_i=frame_i)
        if fluoData is not None:
            fluoImg, fluoKey = fluoData
            if fluoKey == keys[0]:
                ol_img = fluoImg

        ol_img = self.adjustBrightness(ol_img, keys[0])
        color = posData.ol_colors[keys[0]]
        overlay = self._overlay(gray_img_rgb, ol_img, color)

        # Add additional overlays
        for key in keys[1:]:
            ol_img = self.getOlImg(keys[0])
            if fluoData is not None:
                fluoImg, fluoKey = fluoData
                if fluoKey == key:
                    ol_img = fluoImg
            self.adjustBrightness(ol_img, key)
            color = posData.ol_colors[key]
            overlay = self._overlay(overlay, ol_img, color)

        if self.invertBwAction.isChecked():
            overlay = self.invertRGB(overlay)

        if setImg:
            self.img1.setImage(overlay)
        else:
            return overlay

    def invertRGB(self, rgb_img):
        if self.imgCmapName != 'grey':
            return
        # see https://forum.image.sc/t/invert-rgb-image-without-changing-colors/33571
        R = rgb_img[:, :, 0]
        G = rgb_img[:, :, 1]
        B = rgb_img[:, :, 2]
        GB_mean = np.mean([G, B], axis=0)
        RB_mean = np.mean([R, B], axis=0)
        RG_mean = np.mean([R, G], axis=0)
        rgb_img[:, :, 0] = 1-GB_mean
        rgb_img[:, :, 1] = 1-RB_mean
        rgb_img[:, :, 2] = 1-RG_mean
        return rgb_img

    def _overlay(self, gray_img_rgb, ol_img, color):
        ol_RGB_val = [v/255 for v in color]
        ol_alpha = self.alphaScrollBar.value()/self.alphaScrollBar.maximum()
        ol_norm_img = ol_img/ol_img.max()
        ol_img_rgb = gray2rgb(ol_norm_img)*ol_RGB_val
        overlay = (gray_img_rgb*(1.0 - ol_alpha)+ol_img_rgb*ol_alpha)
        overlay = overlay/np.max(overlay)
        overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
        return overlay

    def toggleOverlayColorButton(self, checked=True):
        self.mousePressColorButton(None)

    def toggleTextIDsColorButton(self, checked=True):
        self.textIDsColorButton.selectColor()

    def updateTextIDsColors(self, button):
        r, g, b = np.array(self.textIDsColorButton.color().getRgb()[:3])
        self.gui_setImg1TextColors(r,g,b, custom=True)
        self.updateALLimg()

    def saveTextIDsColors(self, button):
        self.df_settings.at['textIDsColor', 'value'] = self.ax1_oldIDcolor
        self.df_settings.to_csv(self.settings_csv_path)

    def setLut(self, posData, shuffle=True):
        if shuffle:
            posData.lut = self.labelsGrad.item.colorMap().getLookupTable(0,1,255)
            np.random.shuffle(posData.lut)
        else:
            if posData.rp is None:
                posData.lut = self.labelsGrad.item.colorMap().getLookupTable(
                    0,1,255
                )
            else:
                posData.lut = self.labelsGrad.item.colorMap().getLookupTable(
                    0,1,len(posData.rp)
                )
        # Insert background color
        if 'labels_bkgrColor' in self.df_settings.index:
            rgbString = self.df_settings.at['labels_bkgrColor', 'value']
            try:
                r, g, b = rgbString
            except Exception as e:
                r, g, b = myutils.rgb_str_to_values(rgbString)
        else:
            r, g, b = 25, 25, 25
            self.df_settings.at['labels_bkgrColor', 'value'] = (r, g, b)

        posData.lut = np.insert(posData.lut, 0, [r, g, b], axis=0)

    def useCenterBrushCursorHoverIDtoggled(self, checked):
        if checked:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'Yes'
        else:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'No'
        self.df_settings.to_csv(self.settings_csv_path)

    def shuffle_cmap(self):
        posData = self.data[self.pos_i]
        np.random.shuffle(posData.lut[1:])
        self.updateALLimg()

    def hideLabels(self, checked):
        if checked:
            # self.addDelRoiAction.setDisabled(True)
            self.ax2.hide()
            oldLink = self.ax2.vb.linkedView(self.ax1.vb.YAxis)
            try:
                oldLink.sigYRangeChanged.disconnect()
                oldLink.sigXRangeChanged.disconnect()
            except TypeError:
                pass
            self.labBottomGroupbox.hide()
            self.graphLayout.removeItem(self.titleLabel)
            self.graphLayout.addItem(self.titleLabel, row=0, col=1)
            self.mainLayout.setAlignment(self.bottomLayout, Qt.AlignCenter)
            self.df_settings.at['isLabelsVisible', 'value'] = 'No'
            self.df_settings.to_csv(self.settings_csv_path)
            # Move del ROIs to the left image
            for posData in self.data:
                delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
                for roi in delROIs_info['rois']:
                    if roi not in self.ax2.items:
                        continue

                    self.ax1.addItem(roi)
                    # self.ax2.removeItem(roi)
            QTimer.singleShot(200, self.autoRange)
        else:
            self.graphLayout.removeItem(self.titleLabel)
            self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)
            self.mainLayout.setAlignment(self.bottomLayout, Qt.AlignLeft)
            if self.isSegm3D:
                self.labBottomGroupbox.show()
            self.df_settings.at['isLabelsVisible', 'value'] = 'Yes'
            self.df_settings.to_csv(self.settings_csv_path)
            self.ax2.show()
            self.updateALLimg()
            self.ax2.vb.setYLink(self.ax1.vb)
            self.ax2.vb.setXLink(self.ax1.vb)
            QTimer.singleShot(200, self.autoRange)

        self.setBottomLayoutStretch()

    def setBottomLayoutStretch(self):
        if 'isLabelsVisible' not in self.df_settings.index:
            self.df_settings.at['isLabelsVisible', 'value'] = 'Yes'

        bothControl = (
            self.isSegm3D and
            self.df_settings.at['isLabelsVisible', 'value'] == 'Yes'
        )

        if bothControl:
            # Equally share space between the two control groupboxes
            self.bottomLayout.setStretch(0, 1)
            self.bottomLayout.setStretch(1, 6)
            self.bottomLayout.setStretch(2, 2)
            self.bottomLayout.setStretch(3, 6)
            self.bottomLayout.setStretch(4, 1)
        elif self.df_settings.at['isLabelsVisible', 'value'] == 'Yes':
            # Left control takes only left space
            self.bottomLayout.setStretch(0, 1)
            self.bottomLayout.setStretch(1, 5)
            self.bottomLayout.setStretch(2, 5)
            self.bottomLayout.setStretch(3, 1)
            self.bottomLayout.setStretch(4, 1)
        else:
            # Left control takes all the space
            self.bottomLayout.setStretch(0, 3)
            self.bottomLayout.setStretch(1, 11)
            self.bottomLayout.setStretch(2, 1)
            self.bottomLayout.setStretch(3, 1)
            self.bottomLayout.setStretch(4, 1)

    def setCheckedInvertBW(self, checked):
        self.invertBwAction.setChecked(checked)

    def ticksCmapMoved(self, gradient):
        pass
        # posData = self.data[self.pos_i]
        # self.setLut(posData, shuffle=False)
        # self.updateLookuptable()

    def updateLabelsCmap(self, gradient):
        for _posData in self.data:
            self.setLut(_posData)
        self.updateLookuptable()

        self.df_settings = self.labelsGrad.saveState(self.df_settings)
        self.df_settings.to_csv(self.settings_csv_path)

        self.updateALLimg()

    def updateBkgrColor(self, button):
        color = button.color().getRgb()[:3]
        for _posData in self.data:
            _posData.lut[0] = color
        self.updateLookuptable()

    def updateTextLabelsColor(self, button):
        self.ax2_textColor = button.color().getRgb()[:3]
        posData = self.data[self.pos_i]
        if posData.rp is None:
            return

        for obj in posData.rp:
            self.ax2_setTextID(obj)

    def saveTextLabelsColor(self, button):
        color = button.color().getRgb()[:3]
        self.df_settings.at['labels_text_color', 'value'] = color
        self.df_settings.to_csv(self.settings_csv_path)

    def saveBkgrColor(self, button):
        color = button.color().getRgb()[:3]
        self.df_settings.at['labels_bkgrColor', 'value'] = color
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateALLimg(only_ax1=True)

    def updateOlColors(self, button):
        posData = self.data[self.pos_i]
        try:
            fluo_filenames = list(posData.fluo_data_dict.keys())
            _idx = fluo_filenames.index(self._key)
            rgb = self.overlayColorButton.color().getRgb()[:3]
            for _posData in self.data:
                _key = list(_posData.fluo_data_dict.keys())[_idx]
                _posData.ol_colors[_key] = rgb
        except Exception as e:
            self.logger.exception(traceback.format_exc())

        self.df_settings.at['overlayColor', 'value'] = (
            '-'.join([str(v) for v in rgb])
        )
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateOverlay(button)

    def updateOverlay(self, button):
        self.getOverlayImg(setImg=True)

    def getImageDataFromFilename(self, filename):
        posData = self.data[self.pos_i]
        if filename == posData.filename:
            return posData.img_data[posData.frame_i]
        else:
            return posData.ol_data_dict.get(filename)

    def get_2Dimg_from_3D(self, imgData):
        posData = self.data[self.pos_i]
        idx = (posData.filename, posData.frame_i)
        z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
        zProjHow = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
        if zProjHow == 'single z-slice':
            img = imgData[z].copy()
        elif zProjHow == 'max z-projection':
            img = imgData.max(axis=0).copy()
        elif zProjHow == 'mean z-projection':
            img = imgData.mean(axis=0).copy()
        elif zProjHow == 'median z-proj.':
            img = np.median(imgData, axis=0).copy()
        return img

    def updateZsliceScrollbar(self, frame_i):
        posData = self.data[self.pos_i]
        idx = (posData.filename, frame_i)
        z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
        zProjHow = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
        if zProjHow != 'single z-slice':
            return
        reconnect = False
        try:
            self.zSliceScrollBar.actionTriggered.disconnect()
            self.zSliceScrollBar.sliderReleased.disconnect()
            reconnect = True
        except TypeError:
            pass
        self.zSliceScrollBar.setSliderPosition(z)
        if reconnect:
            self.zSliceScrollBar.actionTriggered.connect(
                self.zSliceScrollBarActionTriggered
            )
            self.zSliceScrollBar.sliderReleased.connect(
                self.zSliceScrollBarReleased
            )
        self.z_label.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')
        if not self.labBottomGroupbox.isChecked():
            self.z_label_lab.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')

    def getImage(self, frame_i=None, invert=True, normalizeIntens=True):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        if posData.SizeZ > 1:
            img = posData.img_data[frame_i]
            self.updateZsliceScrollbar(frame_i)
            cells_img = self.get_2Dimg_from_3D(img)
        else:
            cells_img = posData.img_data[frame_i].copy()
        if normalizeIntens:
            cells_img = self.normalizeIntensities(cells_img)
        if self.imgCmapName != 'grey':
            # Do not invert bw for non grey cmaps
            return cells_img
        if self.invertBwAction.isChecked() and invert:
            cells_img = -cells_img+cells_img.max()
        return cells_img

    def setImageImg2(self, updateLookuptable=True):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Segmentation and Tracking' or self.isSnapshot:
            self.addExistingDelROIs()
            allDelIDs, DelROIlab = self.getDelROIlab()
        else:
            DelROIlab = self.get_2Dlab(posData.lab, force_z=False)
            allDelIDs = set()
        if not self.labelsGrad.hideLabelsImgAction.isChecked():
            self.img2.setImage(DelROIlab, z=self.z_lab(), autoLevels=False)
        self.currentLab2D = DelROIlab
        if updateLookuptable:
            self.updateLookuptable(delIDs=allDelIDs)

    def applyDelROIimg1(self, roi, init=False):
        if init:
            if self.overlayButton.isChecked():
                img = self.getOverlayImg(setImg=False)
            else:
                img = self.getImageWithCmap()
            self.imgRGB = self.overlaySegmMasks(img, force=True)
            return

        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        idx = delROIs_info['rois'].index(roi)
        delIDs = delROIs_info['delIDsROI'][idx]
        delMask = delROIs_info['delMasks'][idx]
        how = self.drawIDsContComboBox.currentText()
        if how.find('nothing') != -1:
            return
        elif how.find('contours') != -1:
            for ID in delIDs:
                self.ax1_ContoursCurves[ID-1].setData([], [])
                self.ax1_LabelItemsIDs[ID-1].setText('')
        elif how.find('overlay segm. masks') != -1:
            overlayRGB = self.imgRGB.copy()
            for ID in delIDs:
                delMaskID = delMask==ID
                overlayRGB[delMaskID] = self.img1uintRGB[delMaskID]
                self.ax1_LabelItemsIDs[ID-1].setText('')
            self.img1.setImage(overlayRGB)

    def setTempImg1Brush(self, mask):
        alpha = self.imgGrad.labelsAlphaSlider.value()
        posData = self.data[self.pos_i]
        brushOverlay = self.imgRGB.copy()
        overlay = self.imgRGB[mask]*(1.0-alpha) + self.brushColor*alpha
        brushOverlay[mask] = overlay
        brushOverlay = (brushOverlay*255).astype(np.uint8)
        self.img1.setImage(brushOverlay)
        return overlay

    def setTempImg1Eraser(self, mask, init=False):
        if init:
            if self.overlayButton.isChecked():
                img = self.getOverlayImg(setImg=False)
            else:
                img = self.getImageWithCmap()
            self.imgRGB = self.overlaySegmMasks(img, force=True)
            posData = self.data[self.pos_i]
            lab = self.get_2Dlab(posData.lab)
            self.erasesedLab = np.zeros_like(lab)

        how = self.drawIDsContComboBox.currentText()
        if how.find('contours') != -1:
            erasedRp = skimage.measure.regionprops(self.erasesedLab)
            for obj in erasedRp:
                idx = obj.label-1
                curveID = self.ax1_ContoursCurves[idx]
                cont = self.getObjContours(obj)
                curveID.setData(
                    cont[:,0], cont[:,1], pen=self.oldIDs_cpen
                )
        elif how.find('overlay segm. masks') != -1:
            if mask is not None:
                self.imgRGB[mask] = self.img1uintRGB[mask]
            self.img1.setImage(self.imgRGB)

    def setTempImg1ExpandLabel(self, prevCoords, expandedObjCoords):
        how = self.drawIDsContComboBox.currentText()
        if how.find('overlay segm. masks') != -1:
            # Remove previous overlaid mask
            self.imgRGB[prevCoords] = self.img1uintRGB[prevCoords]

            # Overlay new moved mask
            imgRGB_float = self.imgRGB/255
            alpha = 0.7 # self.imgGrad.labelsAlphaSlider.value()
            color = self.expandingIDColor
            overlay = imgRGB_float[expandedObjCoords]*(1.0-alpha) + color*alpha
            imgRGB_float[expandedObjCoords] = overlay
            self.imgRGB = (np.clip(imgRGB_float, 0, 1)*255).astype(np.uint8)
            self.img1.setImage(self.imgRGB)
        else:
            contCurveID = self.ax1_ContoursCurves[self.expandingID-1]
            contCurveID.setData([], [])
            currentLab2Drp = skimage.measure.regionprops(self.currentLab2D)
            for obj in currentLab2Drp:
                if obj.label == self.expandingID:
                    cont = self.getObjContours(obj)
                    contCurveID.setData(
                        cont[:,0], cont[:,1], pen=self.newIDs_cpen
                    )
                    break

    def setTempImg1MoveLabel(self, prevCoords):
        how = self.drawIDsContComboBox.currentText()
        if how.find('contours') != -1:
            contCurveID = self.ax1_ContoursCurves[self.movingID-1]
            contCurveID.setData([], [])
            currentLab2Drp = skimage.measure.regionprops(self.currentLab2D)
            for obj in currentLab2Drp:
                if obj.label == self.movingID:
                    cont = self.getObjContours(obj)
                    contCurveID.setData(
                        cont[:,0], cont[:,1], pen=self.newIDs_cpen
                    )
                    break
        elif how.find('overlay segm. masks') != -1:
            # Remove previous overlaid mask
            prevCoords = prevCoords[-2:]
            self.imgRGB[prevCoords] = self.img1uintRGB[prevCoords]

            # Get coords of current 2D object
            currentLab2Drp = skimage.measure.regionprops(self.currentLab2D)
            movingObj = None
            for obj in currentLab2Drp:
                if obj.label == self.movingID:
                    movingObj = obj
                    break

            if movingObj is None:
                return

            # Overlay new moved mask
            newCoords = movingObj.coords
            imgRGB_float = self.imgRGB/255
            alpha = 0.7 # self.imgGrad.labelsAlphaSlider.value()
            color = self.movingIDColor
            newCoords = (newCoords[:,0], newCoords[:,1])
            overlay = imgRGB_float[newCoords]*(1.0-alpha) + color*alpha
            imgRGB_float[newCoords] = overlay
            self.imgRGB = (np.clip(imgRGB_float, 0, 1)*255).astype(np.uint8)
            self.img1.setImage(self.imgRGB)


    def update_cca_df_relabelling(self, posData, oldIDs, newIDs):
        relIDs = posData.cca_df['relative_ID']
        posData.cca_df['relative_ID'] = relIDs.replace(oldIDs, newIDs)
        mapper = dict(zip(oldIDs, newIDs))
        posData.cca_df = posData.cca_df.rename(index=mapper)

    def update_cca_df_deletedIDs(self, posData, deleted_IDs):
        relIDs = posData.cca_df.reindex(deleted_IDs, fill_value=-1)['relative_ID']
        posData.cca_df = posData.cca_df.drop(deleted_IDs, errors='ignore')
        self.update_cca_df_newIDs(posData, relIDs)

    def update_cca_df_newIDs(self, posData, new_IDs):
        for newID in new_IDs:
            self.addIDBaseCca_df(posData, newID)

    def update_cca_df_snapshots(self, editTxt, posData):
        cca_df = posData.cca_df
        cca_df_IDs = cca_df.index
        if editTxt == 'Delete ID':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Separate IDs':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Edit ID':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)
            old_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, old_IDs)

        elif editTxt == 'Annotate ID as dead':
            return

        elif editTxt == 'Delete ID with eraser':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Add new ID with brush tool':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)

        elif editTxt == 'Merge IDs':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Add new ID with curvature tool':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)

        elif editTxt == 'Delete IDs using ROI':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Repeat segmentation':
            posData.cca_df = self.getBaseCca_df()

        elif editTxt == 'Random Walker segmentation':
            posData.cca_df = self.getBaseCca_df()


    def warnEditingWithCca_df(self, editTxt, return_answer=False):
        # Function used to warn that the user is editing in "Segmentation and
        # Tracking" mode a frame that contains cca annotations.
        # Ask whether to remove annotations from all future frames
        posData = self.data[self.pos_i]
        if self.isSnapshot and posData.cca_df is not None:
            # For snapshot mode we reinitialize cca_df depending on the edit
            self.update_cca_df_snapshots(editTxt, posData)
            self.store_data()
            self.updateALLimg()
            return

        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        if acdc_df is None:
            return
        else:
            if 'cell_cycle_stage' not in acdc_df.columns:
                return
        action = self.warnEditingWithAnnotActions.get(editTxt, None)
        if action is not None:
            if not action.isChecked():
                return

        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            'You modified a frame that <b>has cell cycle annotations</b>.<br><br>'
            f'The change <b>"{editTxt}"</b> most likely makes the '
            '<b>annotations wrong</b>.<br><br>'
            'If you really want to apply this change we reccommend to remove<br>'
            'ALL cell cycle annotations from current frame to the end.<br><br>'
            'What do you want to do?'
        )
        if action is not None:
            checkBox = QCheckBox('Remember my choice and do not ask again')
        else:
            checkBox = None
        yesButton, _ = msg.warning(
            self, 'Edited segmentation with annotations!', txt,
            buttonsTexts=(
                'Remove annotations from future frames (RECOMMENDED)',
                'Do not remove annotations'
                ),
            widgets=checkBox
            )
        if action is not None:
            action.setChecked(not checkBox.isChecked())
            action.removeAnnot = msg.clickedButton == yesButton
        if return_answer:
            return msg.clickedButton == yesButton
        if msg.clickedButton == yesButton:
            self.store_data()
            posData.frame_i -= 1
            self.get_data()
            self.remove_future_cca_df(posData.frame_i)
            self.next_frame()
        else:
            self.store_data()
        if action is not None:
            if action.removeAnnot:
                self.store_data()
                posData.frame_i -= 1
                self.get_data()
                self.remove_future_cca_df(posData.frame_i)
                self.next_frame()

    def addExistingDelROIs(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        for roi in delROIs_info['rois']:
            if roi in self.ax2.items or roi in self.ax1.items:
                continue
            if self.labelsGrad.hideLabelsImgAction.isChecked():
                self.ax1.addItem(roi)
            else:
                self.ax2.addItem(roi)

    def addNewItems(self, newID):
        posData = self.data[self.pos_i]

        create = False
        start = len(self.ax1_ContoursCurves)

        doCreate = (
            newID > start
            or self.ax1_ContoursCurves[newID-1] is None
        )

        if not doCreate:
            return

        # Contours on ax1
        ax1ContCurve = pg.PlotDataItem()
        self.ax1.addItem(ax1ContCurve)

        # Bud mother line on ax1
        BudMothLine = pg.PlotDataItem()
        self.ax1.addItem(BudMothLine)

        # LabelItems on ax1
        ax1_IDlabel = widgets.myLabelItem()
        self.ax1.addItem(ax1_IDlabel)

        # LabelItems on ax2
        ax2_IDlabel = widgets.myLabelItem()
        self.ax2.addItem(ax2_IDlabel)

        # Contours on ax2
        ax2ContCurve = pg.PlotDataItem()
        self.ax2.addItem(ax2ContCurve)

        self.logger.info(f'Items created for new cell ID {newID}')
        if newID > start:
            empty = [None]*(newID-start)
            self.ax1_ContoursCurves.extend(empty)
            self.ax1_BudMothLines.extend(empty)
            self.ax1_LabelItemsIDs.extend(empty)
            self.ax2_LabelItemsIDs.extend(empty)
            self.ax2_ContoursCurves.extend(empty)

        self.ax1_ContoursCurves[newID-1] = ax1ContCurve
        self.ax1_BudMothLines[newID-1] = BudMothLine
        self.ax1_LabelItemsIDs[newID-1] = ax1_IDlabel
        self.ax2_LabelItemsIDs[newID-1] = ax2_IDlabel
        self.ax2_ContoursCurves[newID-1] = ax2ContCurve

    def updateImageGradientItem(self, imageItem):
        """
        Function called every time the image changes (updateALLimg).
        Set the gradient slider tick positions
        """
        try:
            self.imgGrad.sigLookupTableChanged.disconnect()
            connect = True
        except TypeError:
            connect = False
            pass
        posData = self.data[self.pos_i]
        overlayOFF_key = f'{self.user_ch_name}_overlayOFF'
        isOverlayON = self.overlayButton.isChecked()
        histoLevels = posData.allData_li[posData.frame_i]['histoLevels']
        if posData.manualContrastKey in histoLevels and isOverlayON:
            min, max = histoLevels[posData.manualContrastKey]
        elif isOverlayON:
            min, max = 0, 1
        elif not isOverlayON and overlayOFF_key in histoLevels:
            min, max = histoLevels[overlayOFF_key]
        elif not isOverlayON and overlayOFF_key not in histoLevels:
            min, max = 0, 1

        minTick = self.imgGrad.gradient.getTick(0)
        maxTick = self.imgGrad.gradient.getTick(1)
        self.imgGrad.gradient.setTickValue(minTick, min)
        self.imgGrad.gradient.setTickValue(maxTick, max)
        if connect:
            self.imgGrad.sigLookupTableChanged.connect(self.imgGradLUT_cb)

        # # NOTE: since v1.2.4 we hide the histogram viewbox
        # self.imgGrad.setLevels(
        #     min=np.min(imageItem.image), max=np.max(imageItem.image)
        # )
        # h = imageItem.getHistogram()
        # self.imgGrad.plot.setData(*h)
        

    def updateFramePosLabel(self):
        if self.isSnapshot:
            posData = self.data[self.pos_i]
            self.t_label.setText(
                     f'Pos. n. {self.pos_i+1}/{self.num_pos} '
                     f'({posData.pos_foldername})')
        else:
            posData = self.data[0]
            self.t_label.setText(
                     f'frame n. {posData.frame_i+1}/{posData.SizeT}')

    def getDisplayedZstack(self):
        isDiffGaussFilter = False
        if self.diffGaussFilterWin is None:
            isDiffGaussFilter = False
        elif self.diffGaussFilterWin.previewCheckBox.isChecked():
            isDiffGaussFilter = True

        if isDiffGaussFilter:
            zStack = self.diffGaussFilteredData
        else:
            posData = self.data[self.pos_i]
            zStack = posData.img_data[posData.frame_i]

        for z, img in enumerate(zStack):
            # Apply the other filters
            if self.gaussWin is not None:
                img = self.gaussWin.filter(img)

            if self.edgeWin is not None:
                img = self.edgeWin.filter(img)

            if self.entropyWin is not None:
                img = self.entropyWin.filter(img)

            zStack[z] = img

        return zStack


    def updateFilters(
            self, updateBlur=False, updateSharp=False,
            updateEntropy=False, updateFilters=False
        ):
        if self.gaussWin is not None and (updateBlur or updateFilters):
            self.gaussWin.apply()

        if self.edgeWin is not None and (updateSharp or updateFilters):
            self.edgeWin.apply()

        if self.entropyWin is not None and (updateEntropy or updateFilters):
            self.entropyWin.apply()

    def reinitGraphicalItems(self, IDs):
        allItems = zip(
            self.ax1_ContoursCurves,
            self.ax2_ContoursCurves,
            self.ax1_LabelItemsIDs,
            self.ax2_LabelItemsIDs,
            self.ax1_BudMothLines
        )
        for idx, items_ID in enumerate(allItems):
            (ax1ContCurve, ax2ContCurve,
            _IDlabel1, _IDlabel2,
            BudMothLine) = items_ID

            if idx in IDs or ax1ContCurve is None:
                continue
            else:
                self.ax1.removeItem(self.ax1_ContoursCurves[idx])
                self.ax1_ContoursCurves[idx] = None

                self.ax2.removeItem(self.ax2_ContoursCurves[idx])
                self.ax2_ContoursCurves[idx] = None

                self.ax1.removeItem(self.ax1_BudMothLines[idx])
                self.ax1_BudMothLines[idx] = None

                self.ax1.removeItem(self.ax1_LabelItemsIDs[idx])
                self.ax1_LabelItemsIDs[idx] = None

                self.ax2.removeItem(self.ax1_ContoursCurves[idx])
                self.ax2_LabelItemsIDs[idx] = None

    def addItemsAllIDs(self, IDs):
        numCreatedItems = sum(
            1 for item in self.ax1_ContoursCurves if item is not None
        )
        if numCreatedItems > 1000:
            self.logger.info('Re-initializing graphical items...')
            self.reinitGraphicalItems(IDs)
        for ID in IDs:
            self.addNewItems(ID)

    def highlightSearchedID(self, ID):
        if ID == 0:
            return

        if ID == self.highlightedID:
            return


        how = self.drawIDsContComboBox.currentText()
        contours = zip(
            self.ax1_ContoursCurves,
            self.ax2_ContoursCurves
        )
        for ax1ContCurve, ax2ContCurve in contours:
            if ax1ContCurve is None:
                continue
            if ax1ContCurve.getData()[0] is not None:
                if how.find('contours') != -1:
                    ax1ContCurve.setPen(self.oldIDs_cpen)
                else:
                    ax1ContCurve.setData([], [])
            if ax2ContCurve.getData()[0] is not None:
                if how.find('contours') != -1:
                    ax2ContCurve.setPen(self.oldIDs_cpen)
                else:
                    ax1ContCurve.setData([], [])

        if how.find('IDs') == -1:
            labelItems = zip(self.ax1_LabelItemsIDs, self.ax2_LabelItemsIDs)
            for labelItem_ax1, labelItem_ax2 in labelItems:
                labelItem_ax1.setText('')
                labelItem_ax2.setText('')


        posData = self.data[self.pos_i]
        self.highlightedID = ID

        how = self.drawIDsContComboBox.currentText()
        if ID not in posData.IDs:
            return

        objIdx = posData.IDs.index(ID)
        obj = posData.rp[objIdx]

        if how.find('segm. masks') != -1:
            imgRGB = self.img1_RGB.copy()
            for _obj in posData.rp:
                if not self.isObjVisible(_obj.bbox):
                    continue
                color = posData.lut[_obj.label]/255
                _slice = self.getObjSlice(_obj.slice)
                _objMask = self.getObjImage(_obj.image, _obj.bbox)
                bkgr_label = self.img1_RGB[_slice][_objMask]
                if _obj.label == obj.label:
                    alpha = 0.7
                else:
                    alpha = 0.2
                overlay = bkgr_label*(1.0-alpha) + color*alpha
                imgRGB[_slice][_objMask] = overlay
            imgRGB = (np.clip(imgRGB, 0, 1)*255).astype(np.uint8)
            self.img1.setImage(imgRGB)
        else:
            # Red thick contour of searched ID
            cont = self.getObjContours(obj)
            pen = self.newIDs_cpen
            curveID = self.ax1_ContoursCurves[ID-1]
            curveID.setData(cont[:,0], cont[:,1], pen=pen)

        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'
        IDs_and_masks = how == 'Draw IDs and overlay segm. masks'
        onlyMasks = how == 'Draw only overlay segm. masks'
        ccaInfo_and_masks = how == 'Draw cell cycle info and overlay segm. masks'
        draw_LIs = (
            IDs_and_cont or onlyIDs or only_ccaInfo or ccaInfo_and_cont
            or IDs_and_masks or ccaInfo_and_masks
        )
        # Restore other text IDs to default
        if draw_LIs:
            for _obj in posData.rp:
                self.ax1_setTextID(_obj, how, debug=False)
                self.ax2_setTextID(_obj)

        # Label ID
        LabelItemID = self.ax1_LabelItemsIDs[ID-1]
        txt = f'{ID}'
        LabelItemID.setText(txt, color='r', bold=True, size=self.fontSize)
        y, x = self.getObjCentroid(obj.centroid)
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

        LabelItemID = self.ax2_LabelItemsIDs[ID-1]
        LabelItemID.setText(txt, color='r', bold=True, size=self.fontSize)
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

        # Gray out all IDs excpet searched one
        lut = posData.lut.copy() # [:max(posData.IDs)+1]
        lut[:ID] = lut[:ID]*0.2
        lut[ID+1:] = lut[ID+1:]*0.2
        self.img2.setLookupTable(lut)

    def restoreDefaultSettings(self):
        df = self.df_settings
        df.at['contLineWeight', 'value'] = 2
        df.at['contLineColor', 'value'] = (205, 0, 0, 220)
        df.at['overlaySegmMasksAlpha', 'value'] = 0.3
        df.at['img_cmap', 'value'] = 'grey'
        self.imgCmap = self.imgGrad.cmaps['grey']
        self.imgCmapName = 'grey'
        self.labelsGrad.item.loadPreset('viridis')
        df.at['labels_bkgrColor', 'value'] = (25, 25, 25)
        df.at['is_bw_inverted', 'value'] = 'No'
        df = df[~df.index.str.contains('lab_cmap')]
        df.to_csv(self.settings_csv_path)
        self.gui_createContourPens()
        self.imgGrad.restoreState(df)

        self.labelsGrad.saveState(df)
        self.labelsGrad.restoreState(df, loadCmap=False)

    def updateLabelsAlpha(self, value):
        self.df_settings.at['overlaySegmMasksAlpha', 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateALLimg(only_ax1=True)

    def getImageWithCmap(self, img=None):
        if img is None:
            img = self.getImage()
        cellsKey = f'{self.user_ch_name}_overlayOFF'
        img = self.adjustBrightness(img, cellsKey)
        self.img_layer0 = img
        if self.imgCmapName != 'grey':
            img_max = np.max(img)
            if img_max != 1:
                img = img/img_max
            img = self.imgCmap(img)[:, :, :3]
            img = (np.clip(img, 0, 1)*255).astype(np.uint8)
        return img

    @myutils.exception_handler
    def updateALLimg(
            self, image=None, never_visited=True,
            only_ax1=False, updateBlur=False,
            updateSharp=False, updateEntropy=False,
            updateHistoLevels=False, updateFilters=True,
            updateLabelItemColor=False, debug=False,
            overlayMasks=True, updateDiffGaussFilter=True,
            useStoredGaussFiltered=False, useEraserImg=False
        ):
        posData = self.data[self.pos_i]

        if self.diffGaussFilterWin is None:
            _updateDiffGaussFilter = False
        elif updateDiffGaussFilter:
            isPreview = self.diffGaussFilterWin.previewCheckBox.isChecked()
            _updateDiffGaussFilter = isPreview
        else:
            _updateDiffGaussFilter = False

        if image is None:
            if not _updateDiffGaussFilter:
                if self.overlayButton.isChecked():
                    img = self.getOverlayImg(setImg=False)
                else:
                    img = self.getImageWithCmap()
            else:
                if useStoredGaussFiltered:
                    filteredData = self.diffGaussFilteredData
                else:
                    # Apply diff gauss sharpen filter
                    sigmas = self.diffGaussFilterWin.getSigmas()
                    filename = self.diffGaussFilterWin.channelsComboBox.currentText()
                    _imgData = self.getImageDataFromFilename(filename)
                    imgData = _imgData.copy()
                    filteredData = self.getDiffGaussFilteredImg(imgData, sigmas)
                if posData.SizeZ > 1:
                    img = self.get_2Dimg_from_3D(filteredData)
                else:
                    img = filteredData
                if posData.SizeZ > 1:
                    self.updateZsliceScrollbar(posData.frame_i)
                img = self.getImageWithCmap(img=img)
        else:
            img = image

        if overlayMasks:
            img = self.overlaySegmMasks(img)

        self.img1.setImage(img)
        self.updateFilters(updateBlur, updateSharp, updateEntropy, updateFilters)

        if updateHistoLevels:
            self.updateImageGradientItem(self.img1)

        if self.slideshowWin is not None:
            self.slideshowWin.framne_i = posData.frame_i
            self.slideshowWin.update_img()

        if only_ax1:
            return

        self.addItemsAllIDs(posData.IDs)
        self.clearAllItems()

        self.setImageImg2()
        self.update_rp()

        self.computingContoursTimes = []
        self.drawingLabelsTimes = []
        self.drawingContoursTimes = []
        # Annotate ID and draw contours
        for i, obj in enumerate(posData.rp):
            if not self.isObjVisible(obj.bbox):
                continue
            updateColor=True if updateLabelItemColor and i==0 else False
            self.drawID_and_Contour(obj, updateColor=updateColor)

        # self.logger.info('------------------------------------')
        # self.logger.info(f'Drawing labels = {np.sum(self.drawingLabelsTimes):.3f} s')
        # self.logger.info(f'Computing contours = {np.sum(self.computingContoursTimes):.3f} s')
        # self.logger.info(f'Drawing contours = {np.sum(self.drawingContoursTimes):.3f} s')

        # Update annotated IDs (e.g. dead cells)
        self.update_rp_metadata(draw=True)

        self.highlightLostNew()
        self.setTitleText()
        # # self.checkIDsMultiContour()

        self.highlightSearchedID(self.highlightedID)

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(posData.cca_df)

        self.doCustomAnnotation(0)

        if self.eraserButton.isChecked() and useEraserImg:
            self.setTempImg1Eraser(None, init=self.isSegm3D)

    def startBlinkingModeCB(self):
        try:
            self.timer.stop()
            self.stopBlinkTimer.stop()
        except Exception as e:
            pass
        if self.rulerButton.isChecked():
            return
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.blinkModeComboBox)
        self.timer.start(100)
        self.stopBlinkTimer = QTimer(self)
        self.stopBlinkTimer.timeout.connect(self.stopBlinkingCB)
        self.stopBlinkTimer.start(2000)

    def blinkModeComboBox(self):
        if self.flag:
            self.modeComboBox.setStyleSheet('background-color: orange')
        else:
            self.modeComboBox.setStyleSheet('background-color: none')
        self.flag = not self.flag

    def stopBlinkingCB(self):
        self.timer.stop()
        self.modeComboBox.setStyleSheet('background-color: none')

    def highlightNewIDs_ccaFailed(self):
        posData = self.data[self.pos_i]
        for obj in posData.rp:
            if obj.label in posData.new_IDs:
                # self.ax2_setTextID(obj, 'Draw IDs and contours')
                self.ax1_setTextID(obj, 'Draw IDs and contours')
                cont = self.getObjContours(obj)
                curveID = self.ax1_ContoursCurves[obj.label-1]
                curveID.setData(cont[:,0], cont[:,1], pen=self.tempNewIDs_cpen)

    def highlightLostNew(self):
        posData = self.data[self.pos_i]
        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'
        IDs_and_masks = how == 'Draw IDs and overlay segm. masks'
        onlyMasks = how == 'Draw only overlay segm. masks'
        ccaInfo_and_masks = how == 'Draw cell cycle info and overlay segm. masks'

        for ax2ContCurve in self.ax2_ContoursCurves:
            if ax2ContCurve is None:
                continue

            if ax2ContCurve.getData()[0] is not None:
                ax2ContCurve.setData([], [])

        if posData.frame_i == 0:
            return

        highlight = (
            IDs_and_cont or onlyCont or ccaInfo_and_cont
            or IDs_and_masks or onlyMasks or ccaInfo_and_masks
        )
        if highlight:
            for obj in posData.rp:
                ID = obj.label
                if ID in posData.new_IDs:
                    ContCurve = self.ax2_ContoursCurves[ID-1]
                    cont = self.getObjContours(obj)
                    try:
                        ContCurve.setData(
                            cont[:,0], cont[:,1], pen=self.newIDs_cpen
                        )
                    except AttributeError:
                        self.logger.info(f'Error with ID {ID}')

        if posData.lost_IDs:
            # Get the rp from previous frame
            rp = posData.allData_li[posData.frame_i-1]['regionprops']
            if rp is None:
                return
            for obj in rp:
                self.highlightLost_obj(obj)

    def highlight_obj(self, obj, contPen=None, textColor=None):
        if contPen is None:
            contPen = self.lostIDs_cpen
        if textColor is None:
            textColor = self.lostIDs_qMcolor
        ID = obj.label
        ContCurve = self.ax1_ContoursCurves[ID-1]
        cont = self.getObjContours(obj)
        ContCurve.setData(
            cont[:,0], cont[:,1], pen=contPen
        )
        LabelItemID = self.ax1_LabelItemsIDs[ID-1]
        txt = f'{obj.label}?'
        LabelItemID.setText(txt, color=textColor)
        # Center LabelItem at centroid
        y, x = self.getObjCentroid(obj.centroid)
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)


    def highlightLost_obj(self, obj, forceContour=False):
        posData = self.data[self.pos_i]
        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'
        IDs_and_masks = how == 'Draw IDs and overlay segm. masks'
        onlyMasks = how == 'Draw only overlay segm. masks'
        ccaInfo_and_masks = how == 'Draw cell cycle info and overlay segm. masks'

        if nothing:
            return

        ID = obj.label
        if ID in posData.lost_IDs:
            ContCurve = self.ax1_ContoursCurves[ID-1]
            if ContCurve is None:
                self.addNewItems(ID)
            ContCurve = self.ax1_ContoursCurves[ID-1]

            highlight = (
                IDs_and_cont or onlyCont or ccaInfo_and_cont
                or IDs_and_masks or onlyMasks or forceContour
                or ccaInfo_and_masks
            )

            if highlight:
                cont = self.getObjContours(obj)
                ContCurve.setData(
                    cont[:,0], cont[:,1], pen=self.lostIDs_cpen
                )
            LabelItemID = self.ax1_LabelItemsIDs[ID-1]
            txt = f'{obj.label}?'
            LabelItemID.setText(txt, color=self.lostIDs_qMcolor)
            # Center LabelItem at centroid
            y, x = self.getObjCentroid(obj.centroid)
            w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
            LabelItemID.setPos(x-w/2, y-h/2)


    def setTitleText(self):
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            posData.lost_IDs = []
            posData.new_IDs = []
            posData.old_IDs = []
            posData.IDs = [obj.label for obj in posData.rp]
            # posData.multiContIDs = set()
            self.titleLabel.setText('Looking good!', color=self.titleColor)
            return

        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        existing = True
        if prev_rp is None:
            prev_lab, existing = self.get_labels(
                frame_i=posData.frame_i-1, return_existing=True
            )
            prev_rp = skimage.measure.regionprops(prev_lab)

        prev_IDs = [obj.label for obj in prev_rp]
        curr_IDs = [obj.label for obj in posData.rp]
        lost_IDs = [ID for ID in prev_IDs if ID not in curr_IDs]
        new_IDs = [ID for ID in curr_IDs if ID not in prev_IDs]
        posData.lost_IDs = lost_IDs
        posData.new_IDs = new_IDs
        posData.old_IDs = prev_IDs
        posData.IDs = curr_IDs
        warn_txt = ''
        if existing:
            htmlTxt = ''
        else:
            htmlTxt = f'<font color="white">Never segmented frame. </font>'
        if lost_IDs:
            lost_IDs_format = myutils.get_trimmed_list(lost_IDs)
            warn_txt = f'IDs lost in current frame: {lost_IDs_format}'
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>'
            )
        if new_IDs:
            new_IDs_format = myutils.get_trimmed_list(new_IDs)
            warn_txt = f'New IDs in current frame: {new_IDs_format}'
            htmlTxt = (
                f'{htmlTxt}, <font color="green">{warn_txt}</font>'
            )
        if posData.multiContIDs:
            multiContIDs = myutils.get_trimmed_list(list(posData.multiContIDs))
            warn_txt = f'IDs with multiple contours: {multiContIDs}'
            htmlTxt = (
                f'{htmlTxt}, <font color="red">{warn_txt}</font>'
            )
            posData.multiContIDs = set()
        if not htmlTxt:
            warn_txt = 'Looking good'
            color = 'w'
            htmlTxt = (
                f'<font color="white">{warn_txt}</font>'
            )
        self.titleLabel.setText(htmlTxt)

    def separateByLabelling(self, lab, rp, maxID=None):
        """
        Label each single object in posData.lab and if the result is more than
        one object then we insert the separated object into posData.lab
        """
        setRp = False
        for obj in rp:
            lab_obj = skimage.measure.label(obj.image)
            rp_lab_obj = skimage.measure.regionprops(lab_obj)
            if len(rp_lab_obj)>1:
                if maxID is None:
                    lab_obj += np.max(lab)
                else:
                    lab_obj += maxID
                _slice = obj.slice # self.getObjSlice(obj.slice)
                _objMask = obj.image # self.getObjImage(obj.image)
                lab[_slice][_objMask] = lab_obj[_objMask]
                setRp = True
        return setRp

    def checkTrackingEnabled(self):
        posData = self.data[self.pos_i]
        posData.last_tracked_i = self.navigateScrollBar.maximum()-1
        if posData.frame_i <= posData.last_tracked_i:
            # # self.disableTrackingCheckBox.setChecked(True)
            return True
        else:
            # # self.disableTrackingCheckBox.setChecked(False)
            return False

    def tracking(
            self, onlyIDs=[], enforce=False, DoManualEdit=True,
            storeUndo=False, prev_lab=None, prev_rp=None,
            return_lab=False, assign_unique_new_IDs=True
        ):
        try:
            posData = self.data[self.pos_i]
            mode = str(self.modeComboBox.currentText())
            skipTracking = (
                posData.frame_i == 0 or mode.find('Tracking') == -1
                or self.isSnapshot
            )
            if skipTracking:
                self.setTitleText()
                return

            # Disable tracking for already visited frames
            trackingDisabled = self.checkTrackingEnabled()

            """
            Track only frames that were NEVER visited or the user
            specifically requested to track:
                - Never visited --> NOT self.disableTrackingCheckBox.isChecked()
                - User requested --> posData.isAltDown
                                 --> clicked on repeat tracking (enforce=True)
            """

            if enforce or self.UserEnforced_Tracking:
                # Tracking enforced by the user
                do_tracking = True
            elif self.UserEnforced_DisabledTracking:
                # Tracking specifically DISABLED by the user
                do_tracking = False
            elif trackingDisabled:
                # User did not choose what to do --> tracking disabled for
                # visited frames and enabled for never visited frames
                do_tracking = False
            else:
                do_tracking = True

            if not do_tracking:
                # # self.disableTrackingCheckBox.setChecked(True)
                # self.logger.info('-------------')
                # self.logger.info(f'Frame {posData.frame_i+1} NOT tracked')
                # self.logger.info('-------------')
                self.setTitleText()
                return

            """Tracking starts here"""
            # self.disableTrackingCheckBox.setChecked(False)
            staturBarLabelText = self.statusBarLabel.text()
            self.statusBarLabel.setText('Tracking...')

            if storeUndo:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)

            # First separate by labelling
            setRp = self.separateByLabelling(posData.lab, posData.rp)
            if setRp:
                self.update_rp()

            if prev_lab is None:
                prev_lab = posData.allData_li[posData.frame_i-1]['labels']
            if prev_rp is None:
                prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']

            if self.trackWithAcdcAction.isChecked():
                tracked_lab = CellACDC_tracker.track_frame(
                    prev_lab, prev_rp, posData.lab, posData.rp,
                    IDs_curr_untracked=posData.IDs,
                    setBrushID_func=self.setBrushID,
                    posData=posData,
                    assign_unique_new_IDs=assign_unique_new_IDs
                )
            elif self.trackWithYeazAction.isChecked():
                tracked_lab = self.tracking_yeaz.correspondence(
                    prev_lab, posData.lab, use_modified_yeaz=True,
                    use_scipy=True
                )

            if DoManualEdit:
                # Correct tracking with manually changed IDs
                rp = skimage.measure.regionprops(tracked_lab)
                IDs = [obj.label for obj in rp]
                self.manuallyEditTracking(tracked_lab, IDs)
        except ValueError:
            tracked_lab = self.get_2Dlab(posData.lab)

        # Update labels, regionprops and determine new and lost IDs
        posData.lab = tracked_lab
        self.update_rp()
        self.setTitleText()
        QTimer.singleShot(50, partial(
            self.statusBarLabel.setText, staturBarLabelText
        ))

    def manuallyEditTracking(self, tracked_lab, allIDs):
        posData = self.data[self.pos_i]
        infoToRemove = []
        # Correct tracking with manually changed IDs
        for y, x, new_ID in posData.editID_info:
            old_ID = tracked_lab[y, x]
            if old_ID == 0:
                infoToRemove.append((y, x, new_ID))
                continue
            if new_ID in allIDs:
                tempID = np.max(tracked_lab) + 1
                tracked_lab[tracked_lab == old_ID] = tempID
                tracked_lab[tracked_lab == new_ID] = old_ID
                tracked_lab[tracked_lab == tempID] = new_ID
            else:
                tracked_lab[tracked_lab == old_ID] = new_ID
        for info in infoToRemove:
            posData.editID_info.remove(info)

    def undo_changes_future_frames(self, from_frame_i=None):
        posData = self.data[self.pos_i]
        if from_frame_i is None:
            from_frame_i = posData.frame_i+1
        posData.last_tracked_i = from_frame_i
        self.navigateScrollBar.setMaximum(from_frame_i+1)
        for i in range(from_frame_i, posData.SizeT):
            if posData.allData_li[i]['labels'] is None:
                break

            posData.allData_li[i] = {
                'regionprops': [],
                'labels': None,
                'acdc_df': None,
                'delROIs_info': {
                    'rois': [], 'delMasks': [], 'delIDsROI': []
                },
                'histoLevels': {}
            }

    def removeAllItems(self):
        self.ax1.clear()
        self.ax2.clear()
        try:
            self.chNamesQActionGroup.removeAction(self.userChNameAction)
        except Exception as e:
            pass
        try:
            posData = self.data[self.pos_i]
            for action in posData.fluoDataChNameActions:
                self.chNamesQActionGroup.removeAction(action)
        except Exception as e:
            pass
        try:
            self.overlayButton.setChecked(False)
        except Exception as e:
            pass

    def create_chNamesQActionGroup(self, user_ch_name):
        # LUT histogram channel name context menu actions
        self.chNamesQActionGroup = QActionGroup(self)
        self.userChNameAction = QAction(self)
        self.userChNameAction.setCheckable(True)
        self.userChNameAction.setText(user_ch_name)
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        posData = self.data[self.pos_i]
        for action in posData.fluoDataChNameActions:
            self.chNamesQActionGroup.addAction(action)
            action.setChecked(False)
        self.userChNameAction.setChecked(True)
        self.chNamesQActionGroup.triggered.connect(self.setManualContrastKey)
        for action in self.overlayContextMenu.actions():
            action.setChecked(False)

    def setManualContrastKey(self, action):
        posData = self.data[self.pos_i]
        try:
            keys = list(posData.ol_data.keys())
        except Exception as e:
            keys = []
        keys.append(posData.filename)
        checkedText = action.text()
        for key in keys:
            if key.find(checkedText) != -1:
                break
        posData.manualContrastKey = key
        if not self.overlayButton.isChecked():
            self.updateALLimg(
                only_ax1=True, updateFilters=True, updateHistoLevels=True
            )
        else:
            self.updateImageGradientItem(self.img1)
            self.updateALLimg(only_ax1=True, updateFilters=True)

    def restoreDefaultColors(self):
        try:
            color = self.defaultToolBarButtonColor
            self.overlayButton.setStyleSheet(f'background-color: {color}')
        except AttributeError:
            # traceback.print_exc()
            pass

    # Slots
    def newFile(self):
        self.newSegmEndName = ''
        self.isNewFile = True
        self._openFolder()

    def openFile(self, checked=False, file_path=None):
        self.logger.info(f'Opening FILE "{file_path}"')

        self.isNewFile = False
        self._openFile(file_path=file_path)

    @myutils.exception_handler
    def _openFile(self, file_path=None):
        """
        Function used for loading an image file directly.
        """
        if file_path is None:
            self.MostRecentPath = myutils.getMostRecentPath()
            file_path = QFileDialog.getOpenFileName(
                self, 'Select image file', self.MostRecentPath,
                "Images/Videos (*.png *.tif *.tiff *.jpg *.jpeg *.mov *.avi *.mp4)"
                ";;All Files (*)")[0]
            if file_path == '':
                return
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        if dirname != 'Images':
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            acdc_folder = f'{timestamp}_acdc'
            exp_path = os.path.join(dirpath, acdc_folder, 'Images')
            os.makedirs(exp_path)
        else:
            exp_path = dirpath

        filename, ext = os.path.splitext(os.path.basename(file_path))
        if ext == '.tif' or ext == '.npz':
            self._openFolder(exp_path=exp_path, imageFilePath=file_path)
        else:
            self.logger.info('Copying file to .tif format...')
            data = load.loadData(file_path, '')
            data.loadImgData()
            img = data.img_data
            if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
                self.logger.info('Converting RGB image to grayscale...')
                data.img_data = skimage.color.rgb2gray(data.img_data)
                data.img_data = skimage.img_as_ubyte(data.img_data)
            tif_path = os.path.join(exp_path, f'{filename}.tif')
            if data.img_data.ndim == 3:
                SizeT = data.img_data.shape[0]
                SizeZ = 1
            elif data.img_data.ndim == 4:
                SizeT = data.img_data.shape[0]
                SizeZ = data.img_data.shape[1]
            else:
                SizeT = 1
                SizeZ = 1
            is_imageJ_dtype = (
                data.img_data.dtype == np.uint8
                or data.img_data.dtype == np.uint16
                or data.img_data.dtype == np.float32
            )
            if not is_imageJ_dtype:
                data.img_data = skimage.img_as_ubyte(data.img_data)

            myutils.imagej_tiffwriter(
                tif_path, data.img_data, {}, SizeT, SizeZ
            )
            self._openFolder(exp_path=exp_path, imageFilePath=tif_path)

    def criticalNoTifFound(self, images_path):
        err_title = f'No .tif files found in folder.'
        err_msg = html_utils.paragraph(
            'The following folder<br><br>'
            '<code>{images_path}</code><br><br>'
            '<b>does not contain .tif or .h5 files</b>.<br><br>'
            'Only .tif or .h5 files can be loaded with "Open Folder" button.<br><br>'
            'Try with <code>File --> Open image/video file...</code> '
            'and directly select the file you want to load.'
        )
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        msg.critical(self, err_title, err_msg)

    def reInitGui(self):
        self.isZmodifier = False
        self.askRepeatSegment3D = True
        self.showPropsDockButton.setDisabled(True)

        self.removeAllItems()
        self.reinitCustomAnnot()
        self.gui_addPlotItems()
        self.setUncheckedAllButtons()
        self.restoreDefaultColors()
        self.curvToolButton.setChecked(False)

        self.navigateToolBar.hide()
        self.ccaToolBar.hide()
        self.editToolBar.hide()
        self.widgetsToolBar.hide()
        self.modeToolBar.hide()

        self.modeComboBox.setCurrentText('Viewer')

    def reinitCustomAnnot(self):
        buttons = list(self.customAnnotDict.keys())
        for button in buttons:
            self.removeCustomAnnotButton(button)

    def loadingDataAborted(self):
        self.openAction.setEnabled(True)
        self.titleLabel.setText('Loading data aborted.')

    def openFolder(self, checked=False, exp_path=None, imageFilePath=''):
        self.logger.info(f'Opening FOLDER "{exp_path}"')

        self.isNewFile = False
        if hasattr(self, 'data'):
            self.store_data()
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'Do you want to <b>save</b> before loading another dataset?'
            )
            _, no, yes = msg.question(
                self, 'Save?', txt,
                buttonsTexts=('Cancel', 'No', 'Yes')
            )
            if msg.clickedButton == yes:
                func = partial(self._openFolder, exp_path, imageFilePath)
                cancel = self.saveData(finishedCallback=func)
                return
            elif msg.cancel:
                return

        self._openFolder(exp_path=exp_path, imageFilePath=imageFilePath)

    @myutils.exception_handler
    def _openFolder(self, exp_path=None, imageFilePath=''):
        """Main function to load data.

        Parameters
        ----------
        checked : bool
            kwarg needed because openFolder can be called by openFolderAction.
        exp_path : string or None
            Path selected by the user either directly, through openFile,
            or drag and drop image file.
        imageFilePath : string
            Path of the image file that was either drag and dropped or opened
            from File --> Open image/video file (openFileAction).

        Returns
        -------
            None
        """
        if exp_path is None:
            self.MostRecentPath = myutils.getMostRecentPath()
            exp_path = QFileDialog.getExistingDirectory(
                self,
                'Select experiment folder containing Position_n folders '
                'or specific Position_n folder',
                self.MostRecentPath
            )

        if exp_path == '':
            self.openAction.setEnabled(True)
            self.titleLabel.setText(
                'Drag and drop image file or go to File --> Open folder...',
                color=self.titleColor)
            return

        self.reInitGui()

        self.openAction.setEnabled(False)

        if self.slideshowWin is not None:
            self.slideshowWin.close()

        if self.ccaTableWin is not None:
            self.ccaTableWin.close()

        self.exp_path = exp_path
        self.logger.info(f'Loading from {self.exp_path}')
        myutils.addToRecentPaths(exp_path, logger=self.logger)

        is_pos_folder = os.path.basename(exp_path).find('Position_') != -1
        is_images_folder = os.path.basename(exp_path).find('Images') != -1

        self.titleLabel.setText('Loading data...', color=self.titleColor)
        self.setWindowTitle(f'Cell-ACDC - GUI - "{exp_path}"')

        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )
        user_ch_name = None
        if not is_pos_folder and not is_images_folder and not imageFilePath:
            select_folder = load.select_exp_folder()
            values = select_folder.get_values_segmGUI(exp_path)
            if not values:
                txt = html_utils.paragraph(
                    'The selected folder:<br><br>'
                    f'<code>{exp_path}</code><br><br>'
                    'is <b>not a valid folder</b>.<br><br>'
                    'Select a folder that contains the Position_n folders'
                )
                msg = widgets.myMessageBox()
                msg.critical(
                    self, 'Incompatible folder', txt
                )
                self.titleLabel.setText(
                    'Drag and drop image file or go to File --> Open folder...',
                    color=self.titleColor)
                self.openAction.setEnabled(True)
                return

            if len(values) > 1:
                select_folder.QtPrompt(self, values, allow_abort=False)
                if select_folder.was_aborted:
                    self.titleLabel.setText(
                        'Drag and drop image file or go to '
                        'File --> Open folder...',
                        color=self.titleColor)
                    self.openAction.setEnabled(True)
                    return
            else:
                select_folder.was_aborted = False
                select_folder.selected_pos = select_folder.pos_foldernames

            images_paths = []
            for pos in select_folder.selected_pos:
                images_paths.append(os.path.join(exp_path, pos, 'Images'))

        elif is_pos_folder and not imageFilePath:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_paths = [os.path.join(exp_path, pos_foldername, 'Images')]

        elif is_images_folder and not imageFilePath:
            images_paths = [exp_path]

        elif imageFilePath:
            # images_path = exp_path because called by openFile func
            filenames = myutils.listdir(exp_path)
            ch_names, basenameNotFound = (
                ch_name_selector.get_available_channels(filenames, exp_path)
            )
            filename = os.path.basename(imageFilePath)
            self.ch_names = ch_names
            user_ch_name = [
                chName for chName in ch_names if filename.find(chName)!=-1
            ][0]
            images_paths = [exp_path]

        self.images_paths = images_paths

        # Get info from first position selected
        images_path = self.images_paths[0]
        filenames = myutils.listdir(images_path)
        if ch_name_selector.is_first_call and user_ch_name is None:
            ch_names, basenameNotFound = (
                ch_name_selector.get_available_channels(
                    filenames, images_path
                )
            )
            self.ch_names = ch_names
            if not ch_names:
                self.titleLabel.setText(
                    'Drag and drop image file or go to File --> Open folder...',
                    color=self.titleColor)
                self.openAction.setEnabled(True)
                self.criticalNoTifFound(images_path)
                return
            if len(ch_names) > 1:
                CbLabel='Select channel name to segment: '
                ch_name_selector.QtPrompt(
                    self, ch_names, CbLabel=CbLabel
                )
                if ch_name_selector.was_aborted:
                    self.titleLabel.setText(
                        'Drag and drop image file or go to File --> Open folder...',
                        color=self.titleColor)
                    self.openAction.setEnabled(True)
                    return
            else:
                ch_name_selector.channel_name = ch_names[0]
            ch_name_selector.setUserChannelName()
            user_ch_name = ch_name_selector.user_ch_name
        else:
            # File opened directly with self.openFile
            ch_name_selector.channel_name = user_ch_name

        user_ch_file_paths = []
        for images_path in self.images_paths:
            h5_aligned_path = ''
            h5_path = ''
            npz_aligned_path = ''
            tif_path = ''
            for file in myutils.listdir(images_path):
                channelDataPath = os.path.join(images_path, file)
                if file.endswith(f'{user_ch_name}_aligned.h5'):
                    h5_aligned_path = channelDataPath
                elif file.endswith(f'{user_ch_name}.h5'):
                    h5_path = channelDataPath
                elif file.endswith(f'{user_ch_name}_aligned.npz'):
                    npz_aligned_path = channelDataPath
                elif file.endswith(f'{user_ch_name}.tif'):
                    tif_path = channelDataPath

            if h5_aligned_path:
                self.logger.info(
                    f'Using .h5 aligned file ({h5_aligned_path})...'
                )
                user_ch_file_paths.append(h5_aligned_path)
            elif h5_path:
                self.logger.info(f'Using .h5 file ({h5_path})...')
                user_ch_file_paths.append(h5_path)
            elif npz_aligned_path:
                self.logger.info(
                    f'Using .npz aligned file ({npz_aligned_path})...'
                )
                user_ch_file_paths.append(npz_aligned_path)
            elif tif_path:
                self.logger.info(f'Using .tif file ({tif_path})...')
                user_ch_file_paths.append(tif_path)
            else:
                self.criticalImgPathNotFound(images_path)
                return

        self.appendPathWindowTitle(user_ch_file_paths)

        ch_name_selector.setUserChannelName()
        self.user_ch_name = user_ch_name

        self.initGlobalAttr()
        self.createOverlayContextMenu()

        self.num_pos = len(user_ch_file_paths)
        proceed = self.loadSelectedData(user_ch_file_paths, user_ch_name)
        if not proceed:
            self.openAction.setEnabled(True)
            self.titleLabel.setText(
                'Drag and drop image file or go to File --> Open folder...',
                color=self.titleColor)
            return

    def createOverlayContextMenu(self):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        self.overlayContextMenu = QMenu()
        self.overlayContextMenu.addSeparator()
        actionGroup = QActionGroup(self)
        self.overlayContextMenu.actionGroup = actionGroup
        for chName in ch_names:
            action = QAction(chName, self.overlayContextMenu)
            action.setCheckable(True)
            self.overlayContextMenu.addAction(action)
            self.overlayContextMenu.actionGroup.addAction(action)
        actionGroup.triggered.connect(self.changeOverlayChannel)

    @myutils.exception_handler
    def loadDataWorkerDataIntegrityWarning(self, pos_foldername):
        err_msg = (
            'WARNING: Segmentation mask file ("..._segm.npz") not found. '
            'You could run segmentation module first.'
        )
        self.workerProgress(err_msg, 'INFO')
        self.titleLabel.setText(err_msg, color='r')
        abort = False
        msg = widgets.myMessageBox(parent=self)
        warn_msg = html_utils.paragraph(f"""
            The folder {pos_foldername} <b>does not contain a
            pre-computed segmentation mask</b>.<br><br>
            You can continue with a blank mask or cancel and
            pre-compute the mask with the segmentation module.<br><br>
            Do you want to continue?
        """)
        msg.setIcon(iconName='SP_MessageBoxWarning')
        msg.setWindowTitle('Segmentation file not found')
        msg.addText(warn_msg)
        msg.addButton('Ok')
        continueWithBlankSegm = msg.addButton(' Cancel ')
        msg.show(block=True)
        if continueWithBlankSegm == msg.clickedButton:
            abort = True
        self.loadDataWorker.abort = abort
        self.loadDataWaitCond.wakeAll()

    def warnMemoryNotSufficient(self, total_ram, available_ram, required_ram):
        total_ram = myutils._bytes_to_GB(total_ram)
        available_ram = myutils._bytes_to_GB(available_ram)
        required_ram = myutils._bytes_to_GB(required_ram)
        required_perc = round(100*required_ram/available_ram)
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            The total amount of data that you requested to load is about
            <b>{required_ram:.2f} GB</b> ({required_perc}% of the available memory)
            but there are only <b>{available_ram:.2f} GB</b> available.<br><br>
            For <b>optimal operation</b>, we recommend loading <b>maximum 30%</b>
            of the available memory. To do so, try to close open apps to
            free up some memory. Another option is to crop the images
            using the data prep module.<br><br>
            If you choose to continue, the <b>system might freeze</b>
            or your OS could simply kill the process.<br><br>
            What do you want to do?
        """)
        cancelButton, continueButton = msg.warning(
            self, 'Memory not sufficient', txt,
            buttonsTexts=('Cancel', 'Continue anyway')
        )
        if msg.clickedButton == continueButton:
            return True
        else:
            return False

    def checkMemoryRequirements(self, required_ram):
        memory = psutil.virtual_memory()
        total_ram = memory.total
        available_ram = memory.available
        if required_ram/available_ram > 0.3:
            proceed = self.warnMemoryNotSufficient(
                total_ram, available_ram, required_ram
            )
            return proceed
        else:
            return True

    def criticalImgPathNotFound(self, images_path):
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        err_msg = html_utils.paragraph(f"""
            The folder<br><br>
            <code>{images_path}</code><br><br>
            <b>does not contain any valid image file!</b><br><br>
            Valid file formats are .h5, .tif, _aligned.h5, _aligned.npz.
        """)
        okButton = msg.critical(
            self, 'No valid files found!', err_msg, buttonsTexts=('Ok',)
        )

    def appendPathWindowTitle(self, user_ch_file_paths):
        if self.isSnapshot:
            return

        pos_path = os.path.dirname(os.path.dirname(user_ch_file_paths[0]))
        self.setWindowTitle(f'Cell-ACDC - GUI - "{pos_path}"')

    def initFluoData(self):
        if len(self.ch_names) <= 1:
            return
        msg = widgets.myMessageBox()
        txt = (
            'Do you also want to <b>load fluorescent images?</b><br>'
            'You can load <b>as many channels as you want</b>.<br><br>'
            'If you load fluorescent images then the software will '
            '<b>calculate metrics</b> for each loaded fluorescent channel '
            'such as min, max, mean, quantiles, etc. '
            'of each segmented object.<br><br>'
            '<i>NOTE: You can always load them later from the menu</i> '
            '<code>File --> Load fluorescent images</code>'
        )
        no, yes = msg.question(
            self, 'Load fluorescent images?', html_utils.paragraph(txt),
            buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton == yes:
            self.loadFluo_cb(None)

    def getPathFromChName(self, chName, posData):
        ls = myutils.listdir(posData.images_path)
        basenames = {f[len(posData.basename):]:f for f in ls}
        validEnds = ['_aligned.npz', '_aligned.h5', '.h5', '.tif']
        for end in validEnds:
            files = [
                filename for basename, filename in basenames.items()
                if basename.find(f'{chName}{end}')!=-1
            ]
            if files:
                filename = files[0]
                break
        else:
            self.criticalFluoChannelNotFound(chName, posData)
            self.app.restoreOverrideCursor()
            return None, None

        fluo_path = os.path.join(posData.images_path, filename)
        filename, _ = os.path.splitext(filename)
        return fluo_path, filename

    def loadFluo_cb(self, checked=True, fluo_channels=None):
        if fluo_channels is None:
            posData = self.data[self.pos_i]
            ch_names = [
                ch for ch in self.ch_names if ch != self.user_ch_name
                and ch not in posData.loadedFluoChannels
            ]
            if not ch_names:
                msg = widgets.myMessageBox()
                txt = html_utils.paragraph(
                    'You already <b>loaded ALL channels</b>.<br><br>'
                    'To <b>change the overlaid channel</b> '
                    '<b>right-click</b> on the overlay button.'
                )
                msg.information(self, 'All channels are loaded', txt)
                return False
            selectFluo = apps.QDialogListbox(
                'Select channel',
                'Select channel names to load:\n',
                ch_names, multiSelection=True, parent=self
            )
            selectFluo.exec_()

            if selectFluo.cancel:
                return False

            fluo_channels = selectFluo.selectedItemsText

        for posData in self.data:
            # posData.ol_data = None
            for fluo_ch in fluo_channels:
                fluo_path, filename = self.getPathFromChName(fluo_ch, posData)
                if fluo_path is None:
                    self.criticalFluoChannelNotFound(fluo_ch, posData)
                    return False
                fluo_data, bkgrData = self.load_fluo_data(fluo_path)
                if fluo_data is None:
                    return False
                posData.loadedFluoChannels.add(fluo_ch)

                if posData.SizeT == 1:
                    fluo_data = fluo_data[np.newaxis]

                posData.fluo_data_dict[filename] = fluo_data
                posData.fluo_bkgrData_dict[filename] = bkgrData
                posData.ol_data_dict[filename] = fluo_data.copy()
        self.overlayButton.setStyleSheet('background-color: #A7FAC7')
        self.guiTabControl.addChannels([
            posData.user_ch_name, *posData.loadedFluoChannels
        ])
        return True

    def showInExplorer_cb(self):
        posData = self.data[self.pos_i]
        path = posData.images_path
        myutils.showInExplorer(path)

    def getChNames(self, posData, returnList=False):
        fluo_keys = list(posData.fluo_data_dict.keys())

        loadedChNames = []
        for key in fluo_keys:
            chName = key[len(posData.basename):]
            aligned_idx = chName.find('_aligned')
            if aligned_idx != -1:
                chName = chName[:aligned_idx]
            loadedChNames.append(chName)

        if returnList:
            return loadedChNames
        else:
            posData.loadedChNames = loadedChNames

    def zSliceAbsent(self, filename, posData):
        self.app.restoreOverrideCursor()
        SizeZ = posData.SizeZ
        chNames = posData.chNames
        filenamesPresent = posData.segmInfo_df.index.get_level_values(0).unique()
        chNamesPresent = [
            ch for ch in chNames
            for file in filenamesPresent
            if file.endswith(ch) or file.endswith(f'{ch}_aligned')
        ]
        win = apps.QDialogZsliceAbsent(filename, SizeZ, chNamesPresent)
        win.exec_()
        if win.useMiddleSlice:
            user_ch_name = filename[len(posData.basename):]
            for posData in self.data:
                _, filename = self.getPathFromChName(user_ch_name, posData)
                df = myutils.getDefault_SegmInfo_df(posData, filename)
                posData.segmInfo_df = pd.concat([df, posData.segmInfo_df])
                unique_idx = ~posData.segmInfo_df.index.duplicated()
                posData.segmInfo_df = posData.segmInfo_df[unique_idx]
                posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
        elif win.useSameAsCh:
            user_ch_name = filename[len(posData.basename):]
            for _posData in self.data:
                _, srcFilename = self.getPathFromChName(
                    win.selectedChannel, _posData
                )
                cellacdc_df = _posData.segmInfo_df.loc[srcFilename].copy()
                _, dstFilename = self.getPathFromChName(user_ch_name, _posData)
                dst_df = myutils.getDefault_SegmInfo_df(_posData, dstFilename)
                for z_info in cellacdc_df.itertuples():
                    frame_i = z_info.Index
                    zProjHow = z_info.which_z_proj
                    if zProjHow == 'single z-slice':
                        src_idx = (srcFilename, frame_i)
                        if _posData.segmInfo_df.at[src_idx, 'resegmented_in_gui']:
                            col = 'z_slice_used_gui'
                        else:
                            col = 'z_slice_used_dataPrep'
                        z_slice = _posData.segmInfo_df.at[src_idx, col]
                        dst_idx = (dstFilename, frame_i)
                        dst_df.at[dst_idx, 'z_slice_used_dataPrep'] = z_slice
                        dst_df.at[dst_idx, 'z_slice_used_gui'] = z_slice
                _posData.segmInfo_df = pd.concat([dst_df, _posData.segmInfo_df])
                unique_idx = ~posData.segmInfo_df.index.duplicated()
                posData.segmInfo_df = posData.segmInfo_df[unique_idx]
                _posData.segmInfo_df.to_csv(_posData.segmInfo_df_csv_path)
        elif win.runDataPrep:
            user_ch_file_paths = []
            user_ch_name = filename[len(self.data[self.pos_i].basename):]
            for _posData in self.data:
                user_ch_path, _ = self.getPathFromChName(user_ch_name, _posData)
                user_ch_file_paths.append(user_ch_path)
                exp_path = os.path.dirname(_posData.pos_path)

            dataPrepWin = dataPrep.dataPrepWin()
            dataPrepWin.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            dataPrepWin.titleText = (
            """
            Select z-slice (or projection) for each frame/position.<br>
            Once happy, close the window.
            """)
            dataPrepWin.show()
            dataPrepWin.initLoading()
            dataPrepWin.SizeT = self.data[0].SizeT
            dataPrepWin.SizeZ = self.data[0].SizeZ
            dataPrepWin.metadataAlreadyAsked = True
            self.logger.info(f'Loading channel {user_ch_name} data...')
            dataPrepWin.loadFiles(
                exp_path, user_ch_file_paths, user_ch_name
            )
            dataPrepWin.startAction.setDisabled(True)
            dataPrepWin.onlySelectingZslice = True

            loop = QEventLoop(self)
            dataPrepWin.loop = loop
            loop.exec_()

        self.waitCond.wakeAll()

    def showSetMeasurements(self, checked=False):
        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            favourite_funcs = df_favourite_funcs['favourite_func_name'].to_list()
        except Exception as e:
            favourite_funcs = None

        posData = self.data[self.pos_i]
        fluo_keys = list(posData.fluo_data_dict.keys())
        loadedChNames = self.getChNames(posData, returnList=True)
        loadedChNames.insert(0, self.user_ch_name)
        notLoadedChNames = [c for c in self.ch_names if c not in loadedChNames]
        self.notLoadedChNames = notLoadedChNames
        self.measurementsWin = apps.setMeasurementsDialog(
            loadedChNames, notLoadedChNames, posData.SizeZ > 1, self.isSegm3D,
            favourite_funcs=favourite_funcs, acdc_df=posData.acdc_df,
            acdc_df_path=posData.images_path, posData=posData,
            addCombineMetricCallback=self.addCombineMetric
        )
        self.measurementsWin.sigClosed.connect(self.setMeasurements)
        self.measurementsWin.show()

    def setMeasurements(self):
        posData = self.data[self.pos_i]
        if self.measurementsWin.delExistingCols:
            self.logger.info('Removing existing unchecked measurements...')
            delCols = self.measurementsWin.existingUncheckedColnames
            delRps = self.measurementsWin.existingUncheckedRps
            delCols_format = [f'  *  {colname}' for colname in delCols]
            delRps_format = [f'  *  {colname}' for colname in delRps]
            delCols_format.extend(delRps_format)
            delCols_format = '\n'.join(delCols_format)
            self.logger.info(delCols_format)
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue

                acdc_df = acdc_df.drop(columns=delCols, errors='ignore')
                for col_rp in delRps:
                    drop_df_rp = acdc_df.filter(regex=fr'{col_rp}.*', axis=1)
                    drop_cols_rp = drop_df_rp.columns
                    acdc_df = acdc_df.drop(columns=drop_cols_rp, errors='ignore')
                posData.allData_li[frame_i]['acdc_df'] = acdc_df
        self.logger.info('Setting measurements...')
        fluo_keys = list(posData.fluo_data_dict.keys())
        self.setMetricsToSkip(self.measurementsWin)
        self.logger.info('Metrics successfully set.')

    def setMetricsToSkip(self, measurementsWin):
        self.chNamesToSkip = []
        self.metricsToSkip = {chName:[] for chName in self.ch_names}
        favourite_funcs = set()
        # Remove unchecked metrics and load checked not loaded channels
        for chNameGroupbox in measurementsWin.chNameGroupboxes:
            chName = chNameGroupbox.chName
            if not chNameGroupbox.isChecked():
                # Skip entire channel
                self.chNamesToSkip.append(chName)
            else:
                if chName in self.notLoadedChNames:
                    success = self.loadFluo_cb(fluo_channels=[chName])
                    if not success:
                        return
                for checkBox in chNameGroupbox.checkBoxes:
                    colname = checkBox.text()
                    if not checkBox.isChecked():
                        self.metricsToSkip[chName].append(colname)
                    else:
                        func_name = colname[len(chName):]
                        favourite_funcs.add(func_name)

        if not measurementsWin.sizeMetricsQGBox.isChecked():
            self.sizeMetricsToSave = []
        else:
            self.sizeMetricsToSave = []
            for checkBox in measurementsWin.sizeMetricsQGBox.checkBoxes:
                if checkBox.isChecked():
                    self.sizeMetricsToSave.append(checkBox.text())
                    favourite_funcs.add(checkBox.text())

        if not measurementsWin.regionPropsQGBox.isChecked():
            self.regionPropsToSave = ()
        else:
            self.regionPropsToSave = []
            for checkBox in measurementsWin.regionPropsQGBox.checkBoxes:
                if checkBox.isChecked():
                    self.regionPropsToSave.append(checkBox.text())
                    favourite_funcs.add(checkBox.text())
            self.regionPropsToSave = tuple(self.regionPropsToSave)

        if measurementsWin.mixedChannelsCombineMetricsQGBox is None:
            self.mixedChCombineMetricsToSave = ()
        elif not measurementsWin.mixedChannelsCombineMetricsQGBox.isChecked():
            self.mixedChCombineMetricsToSave = ()
        else:
            mixedChCombineMetricsToSave = []
            win = measurementsWin
            checkBoxes = win.mixedChannelsCombineMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if checkBox.isChecked():
                    mixedChCombineMetricsToSave.append(checkBox.text())
                    favourite_funcs.add(checkBox.text())
            self.mixedChCombineMetricsToSave = tuple(mixedChCombineMetricsToSave)

        df_favourite_funcs = pd.DataFrame(
            {'favourite_func_name': list(favourite_funcs)}
        )
        df_favourite_funcs.to_csv(favourite_func_metrics_csv_path)

    def addCustomMetric(self, checked=False):
        txt = measurements.add_metrics_instructions()
        metrics_path = measurements.metrics_path
        msg = widgets.myMessageBox()
        msg.setIcon()
        msg.setWindowTitle('Add custom metrics instructions')
        msg.addText(txt)
        msg.addButton('Ok')
        showExampleButton = msg.addButton('  Show example...  ')
        showExampleButton.disconnect()
        showExampleButton.clicked.connect(
            partial(myutils.showInExplorer, metrics_path)
        )
        msg.exec_()

    def addCombineMetric(self):
        posData = self.data[self.pos_i]
        isZstack = posData.SizeZ > 1
        win = apps.combineMetricsEquationDialog(
            self.ch_names, isZstack, self.isSegm3D, parent=self
        )
        win.sigOk.connect(self.saveCombineMetricsToPosData)
        win.exec_()
        win.sigOk.disconnect()

    def saveCombineMetricsToPosData(self, window):
        for posData in self.data:
            equationsDict, isMixedChannels = window.getEquationsDict()
            for newColName, equation in equationsDict.items():
                posData.addEquationCombineMetrics(
                    equation, newColName, isMixedChannels
                )
                posData.saveCombineMetrics()

    def setMetricsFunc(self):
        posData = self.data[self.pos_i]
        (metrics_func, all_metrics_names,
        custom_func_dict, total_metrics) = measurements.getMetricsFunc(posData)
        self.metrics_func = metrics_func
        self.all_metrics_names = all_metrics_names
        self.total_metrics = total_metrics
        self.custom_func_dict = custom_func_dict

    def getLastTrackedFrame(self, posData):
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(posData.allData_li):
            # Build segm_npy
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break
        if frame_i > 0:
            return frame_i
        else:
            return last_tracked_i

    def computeVolumeRegionprop(self):
        if 'cell_vol_vox' not in self.sizeMetricsToSave:
            return

        # We compute the cell volume in the main thread because calling
        # skimage.transform.rotate in a separate thread causes crashes
        # with segmentation fault on macOS. I don't know why yet.
        self.logger.info('Computing cell volume...')
        end_i = self.save_until_frame_i
        for p, posData in enumerate(self.data[:self.last_pos]):
            PhysicalSizeY = posData.PhysicalSizeY
            PhysicalSizeX = posData.PhysicalSizeX
            for frame_i, data_dict in enumerate(tqdm(posData.allData_li[:end_i+1], ncols=100)):
                lab = data_dict['labels']
                rp = data_dict['regionprops']
                for i, obj in enumerate(rp):
                    vol_vox, vol_fl = _calc_rot_vol(
                        obj, PhysicalSizeY, PhysicalSizeX
                    )
                    obj.vol_vox = vol_vox
                    obj.vol_fl = vol_fl
                posData.allData_li[frame_i]['regionprops'] = rp

    def askSaveLastVisitedSegmMode(self, isQuickSave=False):
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        frame_i = 0
        last_tracked_i = 0
        self.save_until_frame_i = 0
        if self.isSnapshot:
            return True

        for frame_i, data_dict in enumerate(posData.allData_li):
            # Build segm_npy
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break

        if isQuickSave:
            self.save_until_frame_i = frame_i
            self.last_tracked_i = frame_i
            return True

        if frame_i > 0:
            # Ask to save last visited frame or not
            txt = html_utils.paragraph(f"""
                You visited and stored data up until frame
                number {frame_i+1}.<br><br>
                Enter <b>up to which frame number</b> you want to save data:
            """)
            lastFrameDialog = apps.QLineEditDialog(
                title='Last frame number to save', defaultTxt=str(frame_i+1),
                msg=txt, parent=self, allowedValues=(1,frame_i+1),
                warnLastFrame=True, isInteger=True, stretchEntry=False
            )
            lastFrameDialog.exec_()
            if lastFrameDialog.cancel:
                return False
            else:
                self.save_until_frame_i = lastFrameDialog.EntryID - 1
                last_tracked_i = self.save_until_frame_i
        self.last_tracked_i = last_tracked_i
        return True

    def askSaveMetrics(self):
        txt = html_utils.paragraph(
        """
            Do you also want to <b>save additional metrics</b>
            (e.g., cell volume, mean, amount etc.)?<br><br>
            NOTE: Saving additional metrics is <b>slower</b>,
            we recommend doing it <b>only when you need it</b>.<br>
        """)
        msg = widgets.myMessageBox(parent=self, resizeButtons=False)
        msg.setIcon(iconName='SP_MessageBoxQuestion')
        msg.setWindowTitle('Save metrics?')
        msg.addText(txt)
        yesButton = msg.addButton('Yes')
        noButton = msg.addButton('No')
        cancelButton = msg.addButton('Cancel')
        setMeasurementsButton = msg.addButton('Set measurements...')
        setMeasurementsButton.disconnect()
        setMeasurementsButton.clicked.connect(self.showSetMeasurements)
        msg.exec_()
        save_metrics = msg.clickedButton == yesButton
        cancel = msg.clickedButton == cancelButton or msg.clickedButton is None
        return save_metrics, cancel

    def askSaveAllPos(self):
        last_pos = 1
        ask = False
        for p, posData in enumerate(self.data):
            acdc_df = posData.allData_li[0]['acdc_df']
            if acdc_df is None:
                last_pos = p
                ask = True
                break
        else:
            last_pos = len(self.data)

        if not ask:
            # All pos have been visited, no reason to ask
            return True, len(self.data)

        last_posfoldername = self.data[last_pos-1].pos_foldername
        msg = QMessageBox(self)
        msg.setWindowTitle('Save all positions?')
        msg.setIcon(msg.Question)
        txt = html_utils.paragraph(
        f"""
            Do you want to save <b>ALL positions</b> or <b>only until
            Position_{last_pos}</b> (last visualized/corrected position)?<br>
        """)
        msg.setText(txt)
        allPosbutton =  QPushButton('Save ALL positions')
        upToLastButton = QPushButton(f'Save until {last_posfoldername}')
        msg.addButton(allPosbutton, msg.YesRole)
        msg.addButton(upToLastButton, msg.NoRole)
        msg.exec_()
        return msg.clickedButton() == allPosbutton, last_pos

    def saveMetricsCritical(self, traceback_format):
        print('\n====================================')
        self.logger.exception(traceback_format)
        print('====================================\n')
        self.logger.info('Warning: calculating metrics failed see above...')
        print('------------------------------')

        msg = widgets.myMessageBox(wrapText=False)
        err_msg = html_utils.paragraph(f"""
            Error <b>while saving metrics</b>.<br><br>
            More details below or in the terminal/console.<br><br>
            Note that the error details from this session are also saved
            in the file<br>
            {self.log_path}<br><br>
            Please <b>send the log file</b> when reporting a bug, thanks!
        """)
        msg.addShowInFileManagerButton(self.logs_path, txt='Show log file...')
        msg.setDetailedText(traceback_format, visible=True)
        msg.critical(self, 'Critical error while saving metrics', err_msg)

        self.is_error_state = True
        self.waitCond.wakeAll()

    def saveAsData(self, checked=True):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        existingEndnames = set()
        for _posData in self.data:
            segm_files = load.get_segm_files(_posData.images_path)
            _existingEndnames = load.get_existing_segm_endnames(
                _posData.basename, segm_files
            )
            existingEndnames.update(_existingEndnames)
        posData = self.data[self.pos_i]
        win = apps.filenameDialog(
            basename=f'{posData.basename}segm',
            hintText='Insert a <b>filename</b> for the segmentation file:<br>',
            existingNames=existingEndnames
        )
        win.exec_()
        if win.cancel:
            return

        for posData in self.data:
            posData.setFilePaths(new_endname=win.entryText)

        self.setImageNameText()
        self.saveData()


    def saveDataPermissionError(self, err_msg):
        msg = QMessageBox()
        msg.critical(self, 'Permission denied', err_msg, msg.Ok)
        self.waitCond.wakeAll()

    def saveDataProgress(self, text):
        self.logger.info(text)
        self.saveWin.progressLabel.setText(text)

    def saveDataCustomMetricsCritical(self, traceback_format):
        self.logger.info('')
        print('====================================')
        self.logger.info(traceback_format)
        print('====================================')

    def saveDataCritical(self, traceback_format):
        self.logger.info('')
        print('====================================')
        self.logger.info(traceback_format)
        print('====================================')
        msg = QMessageBox(self)
        msg.setIcon(msg.Critical)
        msg.setWindowTitle('Error')
        msg.setText(traceback_format)
        msg.setDefaultButton(msg.Ok)
        msg.exec_()
        self.waitCond.wakeAll()

    def saveDataUpdateMetricsPbar(self, max, step):
        if max > 0:
            self.saveWin.metricsQPbar.setMaximum(max)
            self.saveWin.metricsQPbar.setValue(0)
        self.saveWin.metricsQPbar.setValue(
            self.saveWin.metricsQPbar.value()+step
        )

    def saveDataUpdatePbar(self, step, max=-1, exec_time=0.0):
        if max >= 0:
            self.saveWin.QPbar.setMaximum(max)
        else:
            self.saveWin.QPbar.setValue(self.saveWin.QPbar.value()+step)
            steps_left = self.saveWin.QPbar.maximum()-self.saveWin.QPbar.value()
            seconds = round(exec_time*steps_left)
            ETA = myutils.seconds_to_ETA(seconds)
            self.saveWin.ETA_label.setText(f'ETA: {ETA}')

    def quickSave(self):
        self.saveData(isQuickSave=True)

    @myutils.exception_handler
    def saveData(self, checked=False, finishedCallback=None, isQuickSave=False):
        self.store_data()
        self.titleLabel.setText(
            'Saving data... (check progress in the terminal)', color=self.titleColor
        )
        self.save_metrics = False
        if not isQuickSave:
            self.save_metrics, cancel = self.askSaveMetrics()
            if cancel:
                self.titleLabel.setText(
                    'Saving data process cancelled.', color=self.titleColor
                )
                return True

        last_pos = len(self.data)
        if self.isSnapshot and not isQuickSave:
            save_Allpos, last_pos = self.askSaveAllPos()
            if save_Allpos:
                last_pos = len(self.data)
                current_pos = self.pos_i
                for p in range(len(self.data)):
                    self.pos_i = p
                    self.get_data()
                    self.store_data()

                # back to current pos
                self.pos_i = current_pos
                self.get_data()

        self.last_pos = last_pos

        if self.isSnapshot:
            self.store_data(mainThread=False)

        proceed = self.askSaveLastVisitedSegmMode(isQuickSave=isQuickSave)
        if not proceed:
            return

        mode = self.modeComboBox.currentText()
        if self.save_metrics or mode == 'Cell cycle analysis':
            self.computeVolumeRegionprop()

        infoTxt = html_utils.paragraph(
            f'Saving {self.exp_path}...<br>', font_size='14px'
        )

        self.saveWin = apps.QDialogPbar(
            parent=self, title='Saving data', infoTxt=infoTxt
        )
        font = QtGui.QFont()
        font.setPixelSize(13)
        self.saveWin.setFont(font)
        # if not self.save_metrics:
        self.saveWin.metricsQPbar.hide()
        self.saveWin.progressLabel.setText('Preparing data...')
        self.saveWin.show()

        # Set up separate thread for saving and show progress bar widget
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.thread = QThread()
        self.worker = saveDataWorker(self)
        self.worker.mode = mode
        self.worker.saveOnlySegm = isQuickSave

        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.finished.connect(self.saveDataFinished)
        if finishedCallback is not None:
            self.worker.finished.connect(finishedCallback)
        self.worker.progress.connect(self.saveDataProgress)
        self.worker.progressBar.connect(self.saveDataUpdatePbar)
        # self.worker.metricsPbarProgress.connect(self.saveDataUpdateMetricsPbar)
        self.worker.critical.connect(self.saveDataCritical)
        self.worker.criticalMetrics.connect(self.saveMetricsCritical)
        self.worker.customMetricsCritical.connect(
            self.saveDataCustomMetricsCritical
        )
        self.worker.criticalPermissionError.connect(self.saveDataPermissionError)
        self.worker.askZsliceAbsent.connect(self.zSliceAbsent)

        self.thread.started.connect(self.worker.run)

        self.thread.start()


    def saveDataFinished(self):
        if self.saveWin.aborted:
            self.titleLabel.setText('Saving process cancelled.', color='r')
        else:
            self.titleLabel.setText('Saved!')
        self.saveWin.workerFinished = True
        self.saveWin.close()
        if self.closeGUI:
            salute_string = myutils.get_salute_string()
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'Data <b>saved!</b>. The GUI will now close.<br><br>'
                f'{salute_string}'
            )
            msg.information(self, 'Data saved', txt)
            self.close()

    def copyContent(self):
        pass

    def pasteContent(self):
        pass

    def cutContent(self):
        pass

    def showTipsAndTricks(self):
        self.welcomeWin = welcome.welcomeWin()
        self.welcomeWin.showAndSetSize()
        self.welcomeWin.showPage(self.welcomeWin.quickStartItem)

    def about(self):
        pass

    def populateOpenRecent(self):
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 1. Read recent Paths
        cellacdc_path = os.path.dirname(os.path.abspath(__file__))
        recentPaths_path = os.path.join(
            cellacdc_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            recentPaths = df['path'].to_list()
        else:
            recentPaths = []
        # Step 2. Dynamically create the actions
        actions = []
        for path in recentPaths:
            if not os.path.exists(path):
                continue
            action = QAction(path, self)
            action.triggered.connect(partial(self.openRecentFile, path))
            actions.append(action)
        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)

    def openRecentFile(self, path):
        self.logger.info(f'Opening recent folder: {path}')
        self.openFolder(exp_path=path)

    def closeEvent(self, event):
        self.saveWindowGeometry()
        # self.saveCustomAnnot()
        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()
        if self.saveAction.isEnabled() and self.titleLabel.text != 'Saved!':
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph('Do you want to <b>save?</b>')
            _, no, yes = msg.question(
                self, 'Save?', txt,
                buttonsTexts=('Cancel', 'No', 'Yes')
            )
            if msg.clickedButton == yes:
                self.closeGUI = True
                cancel = self.saveData()
                event.ignore()
                if cancel:
                    self.closeGUI = False
                    return
            elif msg.cancel:
                event.ignore()
                return

        if self.mainWin is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()
            # # Discard close and simply hide window
            # event.ignore()
            # self.hide()

        self.logger.info('Closing GUI logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
        print('GUI closed.')

    def readSettings(self):
        settings = QSettings('schmollerlab', 'acdc_gui')
        if settings.value('geometry') is not None:
            self.restoreGeometry(settings.value("geometry"))
        # self.restoreState(settings.value("windowState"))

    def saveWindowGeometry(self):
        settings = QSettings('schmollerlab', 'acdc_gui')
        settings.setValue("geometry", self.saveGeometry())
        # settings.setValue("windowState", self.saveState())

    def storeDefaultAndCustomColors(self):
        c = self.overlayButton.palette().button().color().name()
        self.defaultToolBarButtonColor = c
        self.doublePressKeyButtonColor = '#fa693b'

    def showPropsDockWidget(self, checked=False):
        if self.showPropsDockButton.isExpand:
            self.propsDockWidget.setVisible(False)
            self.highlightedID = 0
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            if self.isSegm3D:
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.show()
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.label.show()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.show()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.label.show()
            else:
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.hide()
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.label.hide()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.hide()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.label.hide()

            self.propsDockWidget.setVisible(True)
            self.propsDockWidget.setEnabled(True)
        self.updateALLimg()

    def show(self):
        QMainWindow.show(self)

        self.setWindowState(Qt.WindowNoState)
        self.setWindowState(Qt.WindowActive)
        self.raise_()

        self.readSettings()
        self.storeDefaultAndCustomColors()
        h = self.drawIDsContComboBox.size().height()
        self.navigateScrollBar.setFixedHeight(h)
        self.zSliceScrollBar.setFixedHeight(h)
        self.zSliceScrollBarLab.setFixedHeight(h)
        self.alphaScrollBar.setFixedHeight(h)
        self.zSliceOverlay_SB.setFixedHeight(h)

        self.gui_initImg1BottomWidgets()
        self.img1BottomGroupbox.hide()
        self.labBottomGroupbox.hide()

        w = self.showPropsDockButton.width()
        h = self.showPropsDockButton.height()
        self.showPropsDockButton.setMaximumWidth(15)
        self.showPropsDockButton.setMaximumHeight(60)

    def resizeEvent(self, event):
        if hasattr(self, 'ax1'):
            self.ax1.autoRange()
