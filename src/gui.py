# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPyTop HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TODO:
#

"""Cell-ACDC GUI for correcting Segmentation and Tracking errors"""
print('Importing modules...')
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
from functools import wraps
from skimage.color import gray2rgb, gray2rgba

from PyQt5.QtCore import (
    Qt, QFile, QTextStream, QSize, QRect, QRectF,
    QEventLoop, QTimer, QEvent, QObject, pyqtSignal,
    QThread, QMutex, QWaitCondition, QSettings
)
from PyQt5.QtGui import (
    QIcon, QKeySequence, QCursor, QKeyEvent, QGuiApplication,
    QPixmap
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton, QHBoxLayout,
    QMainWindow, QMenu, QToolBar, QGroupBox, QGridLayout,
    QScrollBar, QCheckBox, QToolButton, QSpinBox, QGroupBox,
    QComboBox, QDial, QButtonGroup, QActionGroup,
    QShortcut, QFileDialog, QDoubleSpinBox,
    QAbstractSlider, QMessageBox, QWidget,
    QDockWidget, QGridLayout, QSizePolicy
)

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# NOTE: Enable icons
import qrc_resources

# Custom modules
import load, prompts, apps
import core, myutils, dataPrep, widgets
from cca_functions import _calc_rot_vol
from core import numba_max, numba_min
from myutils import download_model, exec_time
from QtDarkMode import breeze_resources
from help import welcome

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

print('Initializing...')

# Interpret image data as row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
np.random.seed(1568)

pd.set_option("display.max_columns", 20)
pd.set_option('display.expand_frame_repr', False)

def qt_debug_trace():
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    import pdb; pdb.set_trace()

def worker_exception_handler(func):
    @wraps(func)
    def run(self):
        try:
            result = func(self)
        except Exception as error:
            result = None
            self.critical.emit(error)
        return result
    return run

def exception_handler(func):
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
            result = None
            self.logger.exception(e)
            msg = QMessageBox(self)
            msg.setWindowTitle('Critical error')
            msg.setIcon(msg.Critical)
            err_msg = (f"""
            <p style="font-size:10pt">
                Error in function <b>{func.__name__}</b>.<br><br>
                More details below or in the terminal/console.<br><br>
                Note that the error details fro this session are also saved
                in the file<br>
                "/Cell_ACDC/logs/{self.log_filename}.log"
            </p>
            """)
            msg.setText(err_msg)
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            self.is_error_state = True
        return result
    return inner_function

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

    @worker_exception_handler
    def run(self):
        self.mutex.lock()

        self.progress.emit('Relabelling process started...')

        posData = self.posData
        progressWin = self.mainWin.progressWin
        mainWin = self.mainWin

        current_lab = posData.lab.copy()
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
        self.sigRemoveItemsGUI.emit(numba_max(segm_data))

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
            mainWin.update_cca_df_relabelling(
                posData, oldIDs, newIDs
            )
            mainWin.update_rp()
            mainWin.store_data()

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
    criticalMetrics = pyqtSignal(str)
    criticalPermissionError = pyqtSignal(str)
    askSaveLastVisitedCcaMode = pyqtSignal(int, object)
    askSaveLastVisitedSegmMode = pyqtSignal(int, object)
    metricsPbarProgress = pyqtSignal(int, int)
    askZsliceAbsent = pyqtSignal(str, object)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.saveWin = mainWin.saveWin
        self.mutex = mainWin.mutex
        self.waitCond = mainWin.waitCond

    def run(self):
        last_pos = self.mainWin.last_pos
        save_metrics = self.mainWin.save_metrics
        self.time_last_pbar_update = time.time()
        for p, posData in enumerate(self.mainWin.data[:last_pos]):
            if self.saveWin.aborted:
                self.finished.emit()
                return

            current_frame_i = posData.frame_i
            mode = self.mainWin.modeComboBox.currentText()
            if not self.mainWin.isSnapshot:
                self.mutex.lock()
                self.askSaveLastVisitedSegmMode.emit(p, posData)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                if self.askSaveLastCancelled:
                    self.mainWin.saveWin.aborted = True
                    self.finished.emit()
                    return
                last_tracked_i = self.mainWin.last_tracked_i
                if last_tracked_i is None:
                    return
            elif self.mainWin.isSnapshot:
                last_tracked_i = 0

            if p == 0:
                self.progressBar.emit(0, last_pos*(last_tracked_i+1), 0)

            if self.mainWin.isSnapshot:
                self.mainWin.store_data()
            try:
                segm_npz_path = posData.segm_npz_path
                acdc_output_csv_path = posData.acdc_output_csv_path
                last_tracked_i_path = posData.last_tracked_i_path
                segm_npy = np.copy(posData.segm_data)
                npz_delROIs_info = {}
                delROIs_info_path = posData.delROIs_info_path
                acdc_df_li = []
                keys = []

                # Add segmented channel data for calc metrics
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
                    if posData.SizeT > 1:
                        segm_npy[frame_i] = lab
                    else:
                        segm_npy = lab

                    acdc_df = data_dict['acdc_df']

                    # Build acdc_df and index it in each frame_i of acdc_df_li
                    if acdc_df is not None and np.any(lab):
                        acdc_df = load.loadData.BooleansTo0s1s(
                                    acdc_df, inplace=False
                        )
                        rp = data_dict['regionprops']
                        try:
                            if save_metrics:
                                acdc_df = self.mainWin.addMetrics_acdc_df(
                                    acdc_df, rp, frame_i, lab, posData
                                )
                            acdc_df_li.append(acdc_df)
                            key = (frame_i, posData.TimeIncrement*frame_i)
                            keys.append(key)
                        except Exception as e:
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

                posData.fluo_data_dict.pop(posData.filename)
                posData.fluo_bkgrData_dict.pop(posData.filename)

                self.progress.emit('Almost done...')
                self.progressBar.emit(0, 0, 0)

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

                try:
                    all_frames_acdc_df = pd.concat(
                        acdc_df_li, keys=keys,
                        names=['frame_i', 'time_seconds', 'Cell_ID']
                    )

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

                    all_frames_acdc_df = pd.concat(
                        acdc_df_li, keys=keys,
                        names=['frame_i', 'time_seconds', 'Cell_ID']
                    )

                    # Save segmentation metadata
                    all_frames_acdc_df.to_csv(acdc_output_csv_path)
                    posData.acdc_df = all_frames_acdc_df
                except Exception as e:
                    self.mutex.lock()
                    self.critical.emit(traceback.format_exc())
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()

                # Save segmentation file
                np.savez_compressed(segm_npz_path, np.squeeze(segm_npy))
                posData.segm_data = segm_npy

                with open(last_tracked_i_path, 'w+') as txt:
                    txt.write(str(frame_i))

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

class segmWorker(QObject):
    finished = pyqtSignal(np.ndarray, float)
    debug = pyqtSignal(object)
    critical = pyqtSignal(object)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.mainWin = mainWin

    @worker_exception_handler
    def run(self):
        t0 = time.time()
        img = self.mainWin.getDisplayedCellsImg()
        img = myutils.uint_to_float(img)
        lab = self.mainWin.model.segment(img, **self.mainWin.segment2D_kwargs)
        if self.mainWin.applyPostProcessing:
            lab = core.remove_artefacts(
                lab,
                min_solidity=self.mainWin.minSolidity,
                min_area=self.mainWin.minSize,
                max_elongation=self.mainWin.maxElongation
            )
        t1 = time.time()
        exec_time = t1-t0
        self.finished.emit(lab, exec_time)

class guiWin(QMainWindow):
    """Main Window."""

    def __init__(self, app, parent=None, buttonToRestore=None, mainWin=None):
        """Initializer."""
        super().__init__(parent)

        self.is_win = sys.platform.startswith("win")

        self.is_error_state = False
        self.setupLogger()
        self.loadLastSessionSettings()

        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin

        self.app = app

        self.progressWin = None
        self.slideshowWin = None
        self.ccaTableWin = None
        self.data_loaded = False
        self.searchingID = False
        self.flag = True

        self.setWindowTitle("Cell-ACDC - GUI")
        self.setWindowIcon(QIcon(":assign-motherbud.svg"))
        self.setAcceptDrops(True)

        self.checkableButtons = []
        self.LeftClickButtons = []

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

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_createGraphicsPlots()
        self.gui_addGraphicsItems()

        self.gui_createImg1Widgets()

        self.set_metrics_func()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = self.gui_createGridLayout()

        mainContainer.setLayout(mainLayout)

        self.isEditActionsConnected = False

    def setupLogger(self, module='gui'):
        logger = logging.getLogger('spotMAX')
        logger.setLevel(logging.INFO)

        src_path = os.path.dirname(os.path.abspath(__file__))
        logs_path = os.path.join(src_path, 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        else:
            # Keep 20 most recent logs
            ls = myutils.listdir(logs_path)
            if len(ls)>20:
                ls = [os.path.join(logs_path, f) for f in ls]
                ls.sort(key=lambda x: os.path.getmtime(x))
                for file in ls[:-20]:
                    os.remove(file)

        date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'{date_time}_{module}_stdout.log'
        log_path = os.path.join(logs_path, log_filename)
        self.log_filename = log_filename

        output_file_handler = logging.FileHandler(log_path, mode='w')
        stdout_handler = logging.StreamHandler(sys.stdout)

        # Format your logs (optional)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s:\n'
            '------------------------\n'
            '%(message)s\n'
            '------------------------\n',
            datefmt='%d-%m-%Y, %H:%M:%S')
        output_file_handler.setFormatter(formatter)

        logger.addHandler(output_file_handler)
        logger.addHandler(stdout_handler)

        self.logger = logger

    def loadLastSessionSettings(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(src_path, 'temp')
        csv_path = os.path.join(temp_path, 'settings.csv')
        self.settings_csv_path = csv_path
        if os.path.exists(csv_path):
            self.df_settings = pd.read_csv(csv_path, index_col='setting')
            if 'is_bw_inverted' not in self.df_settings.index:
                self.df_settings.at['is_bw_inverted', 'value'] = 'No'
            else:
                self.df_settings.loc['is_bw_inverted'] = (
                    self.df_settings.loc['is_bw_inverted'].astype(str)
                )
            if 'fontSize' not in self.df_settings.index:
                self.df_settings.at['fontSize', 'value'] = '10pt'
            if 'overlayColor' not in self.df_settings.index:
                self.df_settings.at['overlayColor', 'value'] = '255-255-0'
            if 'how_normIntensities' not in self.df_settings.index:
                raw = 'Do not normalize. Display raw image'
                self.df_settings.at['how_normIntensities', 'value'] = raw
        else:
            idx = ['is_bw_inverted', 'fontSize', 'overlayColor', 'how_normIntensities']
            values = ['No', '10pt', '255-255-0', 'raw']
            self.df_settings = pd.DataFrame({
                'setting': idx,'value': values}
            ).set_index('setting')

    def dragEnterEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(file_path):
            exp_path = file_path
            if basename.find('Position_')!=-1 or basename=='Images':
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        file_path = event.mimeData().urls()[0].toLocalFile()
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
        # fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.openFileAction)
        # Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveAction)
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
        fs = int(re.findall('(\d+)pt', self.fontSize)[0])
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
        editMenu.addAction(self.enableSmartTrackAction)
        editMenu.addAction(self.enableAutoZoomToCellsAction)


        # Image menu
        ImageMenu = menuBar.addMenu("&Image")
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.imgPropertiesAction)
        filtersMenu = ImageMenu.addMenu("Filters")
        filtersMenu.addAction(self.gaussBlurAction)
        filtersMenu.addAction(self.edgeDetectorAction)
        filtersMenu.addAction(self.entropyFilterAction)
        normalizeIntensitiesMenu = ImageMenu.addMenu("Normalize intensities")
        normalizeIntensitiesMenu.addAction(self.normalizeRawAction)
        normalizeIntensitiesMenu.addAction(self.normalizeToFloatAction)
        # normalizeIntensitiesMenu.addAction(self.normalizeToUbyteAction)
        normalizeIntensitiesMenu.addAction(self.normalizeRescale0to1Action)
        normalizeIntensitiesMenu.addAction(self.normalizeByMaxAction)
        ImageMenu.addAction(self.invertBwAction)

        # Segment menu
        SegmMenu = menuBar.addMenu("&Segment")
        SegmMenu.addSeparator()
        for action in self.segmActions:
            SegmMenu.addAction(action)
        SegmMenu.addAction(self.SegmActionRW)
        SegmMenu.addAction(self.postProcessSegmAction)
        SegmMenu.addAction(self.autoSegmAction)
        SegmMenu.aboutToShow.connect(self.nonViewerEditMenuOpened)

        # Segment menu
        trackingMenu = menuBar.addMenu("&Tracking")
        self.trackingMenu = trackingMenu
        trackingMenu.addSeparator()
        selectTrackAlgoMenu = trackingMenu.addMenu(
            'Select real-time tracking algorithm'
        )
        selectTrackAlgoMenu.addAction(self.trackWithAcdcAction)
        selectTrackAlgoMenu.addAction(self.trackWithYeazAction)
        trackingMenu.addAction(self.repeatTrackingMenuAction)
        trackingMenu.aboutToShow.connect(self.nonViewerEditMenuOpened)


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
        toolbarSize = 34

        # File toolbar
        fileToolBar = self.addToolBar("File")
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)
        # fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addAction(self.showInExplorerAction)
        fileToolBar.addAction(self.reloadAction)
        fileToolBar.addAction(self.undoAction)
        fileToolBar.addAction(self.redoAction)
        self.fileToolBar = fileToolBar
        self.setEnabledFileToolbar(False)

        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

        # Navigation toolbar
        navigateToolBar = QToolBar("Navigation", self)
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

        self.overlayButton = QToolButton(self)
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
            'Measure straight line. '
            'Length is displayed on the bottom-right corner.'
        )
        navigateToolBar.addWidget(self.rulerButton)
        self.checkableButtons.append(self.rulerButton)
        self.LeftClickButtons.append(self.rulerButton)

        # fluorescent image color widget
        colorsToolBar = QToolBar("Colors", self)

        self.overlayColorButton = pg.ColorButton(self, color=(230,230,230))
        # self.overlayColorButton.mousePressEvent = self.mousePressColorButton
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
        self.modeComboBox.addItems(
            ['Segmentation and Tracking', 'Cell cycle analysis', 'Viewer']
        )
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

        # toolbarSize = 58
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # ccaToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # widgetsToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # modeToolBar.setIconSize(QSize(toolbarSize, toolbarSize))

    def gui_createGridLayout(self):
        mainLayout = QGridLayout()
        row = 0
        mainLayout.addWidget(self.graphLayout, row, 0, 1, 2)

        row += 1
        mainLayout.addLayout(self.bottomLayout, row, 0)
        mainLayout.setRowStretch(row, 0)

        return mainLayout

    def gui_createControlsToolbar(self):
        self.addToolBarBreak()

        self.wandControlsToolbar = QToolBar("Controls", self)
        self.wandToleranceSlider = widgets.sliderWithSpinBox(
            title='Tolerance', title_loc='in_line')
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

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createActions(self):
        # File actions
        self.newAction = QAction(self)
        self.newAction.setText("&New")
        self.newAction.setIcon(QIcon(":file-new.svg"))
        self.openAction = QAction(QIcon(":folder-open.svg"), "&Open folder...", self)
        self.openFileAction = QAction(QIcon(":image.svg"),"&Open image/video file...", self)
        self.saveAction = QAction(QIcon(":file-save.svg"),
                                  "&Save (Ctrl+S)", self)
        self.loadFluoAction = QAction("Load fluorescent images...", self)
        self.reloadAction = QAction(QIcon(":reload.svg"),
                                          "Reload segmentation file", self)
        self.showInExplorerAction = QAction(QIcon(":drawer.svg"),
                                    "&Show in Explorer/Finder", self)
        self.exitAction = QAction("&Exit", self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo (Ctrl+Z)", self)
        self.redoAction = QAction(QIcon(":redo.svg"), "Redo (Ctrl+Y)", self)
        # String-based key sequences
        self.newAction.setShortcut("Ctrl+N")
        self.openAction.setShortcut("Ctrl+O")
        self.saveAction.setShortcut("Ctrl+S")
        self.undoAction.setShortcut("Ctrl+Z")
        self.redoAction.setShortcut("Ctrl+Y")
        # Help tips
        newTip = "Create a new file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setToolTip(newTip)
        self.newAction.setWhatsThis("Create a new and empty text file")

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
            action = QAction(f"{model_name}...", self)
            self.segmActions.append(action)
            self.modelNames.append(model_name)
            self.models.append(None)
            self.acdcSegment_li.append(None)
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
        self.repeatTrackingMenuAction = QAction('Repeat tracking...', self)
        self.repeatTrackingMenuAction.setDisabled(True)

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
        self.manuallyEditCcaAction.setDisabled(True)

        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        checked = self.df_settings.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

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

    def gui_connectActions(self):
        # Connect File actions
        self.newAction.triggered.connect(self.newFile)
        self.openAction.triggered.connect(self.openFolder)
        self.openFileAction.triggered.connect(self.openFile)
        self.saveAction.triggered.connect(self.saveData)
        self.showInExplorerAction.triggered.connect(self.showInExplorer)
        self.exitAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)

        # Connect Help actions
        self.tipsAction.triggered.connect(self.showTipsAndTricks)
        self.UserManualAction.triggered.connect(self.showUserManual)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)


    def gui_connectEditActions(self):
        self.showInExplorerAction.setEnabled(True)
        self.setEnabledFileToolbar(True)
        self.loadFluoAction.setEnabled(True)
        self.isEditActionsConnected = True

        self.fontSizeMenu.triggered.connect(self.changeFontSize)
        self.prevAction.triggered.connect(self.prev_cb)
        self.nextAction.triggered.connect(self.next_cb)
        self.overlayButton.toggled.connect(self.overlay_cb)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.loadFluoAction.triggered.connect(self.loadFluo_cb)
        self.reloadAction.triggered.connect(self.reload_cb)
        self.findIdAction.triggered.connect(self.findID)
        self.slideshowButton.toggled.connect(self.launchSlideshow)

        for action in self.segmActions:
            action.triggered.connect(self.repeatSegm)

        self.SegmActionRW.triggered.connect(self.randomWalkerSegm)
        self.postProcessSegmAction.toggled.connect(self.postProcessSegm)
        self.autoSegmAction.toggled.connect(self.autoSegm_cb)
        self.disableTrackingCheckBox.clicked.connect(self.disableTracking)
        self.repeatTrackingAction.triggered.connect(self.repeatTracking)
        self.repeatTrackingMenuAction.triggered.connect(self.repeatTracking)
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


        # Drawing mode
        self.drawIDsContComboBox.currentIndexChanged.connect(
                                                self.drawIDsContComboBox_cb)
        self.drawIDsContComboBox.activated.connect(self.clearComboBoxFocus)
        self.gaussBlurAction.toggled.connect(self.gaussBlur)
        self.edgeDetectorAction.toggled.connect(self.edgeDetection)
        self.entropyFilterAction.toggled.connect(self.entropyFilter)
        self.addDelRoiAction.triggered.connect(self.addDelROI)
        self.delBorderObjAction.triggered.connect(self.delBorderObj)
        self.hist.sigLookupTableChanged.connect(self.histLUT_cb)
        self.hist.gradient.sigGradientChangeFinished.connect(
            self.histLUTfinished_cb
        )
        self.normalizeQActionGroup.triggered.connect(self.saveNormAction)
        self.imgPropertiesAction.triggered.connect(self.editImgProperties)


    def gui_createImg1Widgets(self):
        # Toggle contours/ID comboboxf
        self.drawIDsContComboBoxSegmItems = ['Draw IDs and contours',
                                             'Draw only cell cycle info',
                                             'Draw cell cycle info and contours',
                                             'Draw only mother-bud lines',
                                             'Draw only IDs',
                                             'Draw only contours',
                                             'Draw nothing']
        self.drawIDsContComboBoxCcaItems = ['Draw only cell cycle info',
                                            'Draw cell cycle info and contours',
                                            'Draw only mother-bud lines',
                                            'Draw IDs and contours',
                                            'Draw only IDs',
                                            'Draw only contours',
                                            'Draw nothing']
        self.drawIDsContComboBox = QComboBox()
        self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
        # Always adjust combobox width to largest item
        self.drawIDsContComboBox.setSizeAdjustPolicy(
                    self.drawIDsContComboBox.AdjustToContents)

        # Frames scrollbar
        self.navigateScrollBar = QScrollBar(Qt.Horizontal)
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
        _font = QtGui.QFont()
        _font.setPointSize(10)
        t_label.setFont(_font)
        self.t_label = t_label

        # z-slice scrollbars
        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        _z_label = QLabel('z-slice  ')
        _font = QtGui.QFont()
        _font.setPointSize(10)
        _z_label.setFont(_font)
        self.z_label = _z_label

        self.zProjComboBox = QComboBox()
        self.zProjComboBox.addItems([
            'single z-slice',
            'max z-projection',
            'mean z-projection',
            'median z-proj.'
        ])

        self.zSliceOverlay_SB = QScrollBar(Qt.Horizontal)
        _z_label = QLabel('overlay z-slice  ')
        _font = QtGui.QFont()
        _font.setPointSize(10)
        _z_label.setFont(_font)
        self.overlay_z_label = _z_label

        self.zProjOverlay_CB = QComboBox()
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

        self.bottomLayout = QHBoxLayout()
        img1BottomGroupbox = self.gui_addImg1BottomWidgets()
        sp = img1BottomGroupbox.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        img1BottomGroupbox.setSizePolicy(sp)
        self.img1BottomGroupbox = img1BottomGroupbox

        self.bottomLayout.addSpacing(100)
        self.bottomLayout.addWidget(img1BottomGroupbox)
        self.bottomLayout.addStretch()

    def gui_addImg1BottomWidgets(self):
        bottomLeftLayout = QGridLayout()
        container = QGroupBox()

        row = 0
        bottomLeftLayout.addWidget(
            self.drawIDsContComboBox, row, 1, alignment=Qt.AlignCenter
        )

        row += 1
        bottomLeftLayout.addWidget(self.t_label, row, 0, alignment=Qt.AlignRight)
        bottomLeftLayout.addWidget(self.navigateScrollBar, row, 1)

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

        bottomLeftLayout.setColumnStretch(0,0)
        bottomLeftLayout.setColumnStretch(1,3)
        bottomLeftLayout.setColumnStretch(2,0)
        container.setLayout(bottomLeftLayout)
        return container

    def gui_createGraphicsPlots(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

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
        equalizeHistPushButton.setStyleSheet(
               'QPushButton {background-color: #282828; color: #F0F0F0;}')
        proxy.setWidget(equalizeHistPushButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        self.equalizeHistPushButton = equalizeHistPushButton

        # Left image histogram
        self.hist = pg.HistogramLUTItem()
        # Disable histogram default context Menu event
        self.hist.vb.raiseContextMenu = lambda x: None
        self.graphLayout.addItem(self.hist, row=1, col=0)

        # Title
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.titleLabel.setText(
            'Drag and drop image file or go to File --> Open folder...')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)

        # # Current frame text
        # self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        # self.frameLabel.setText(' ')
        # self.graphLayout.addItem(self.frameLabel, row=2, col=1, colspan=2)

    def gui_setLabelsColors(self, r, g, b, custom=False):
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
            try:
                r, g, b = re.findall('(\d+), (\d+), (\d+)', rgbString)[0]
                r, g, b = int(r), int(g), int(b)
            except TypeError:
                r, g, b = 255, 255, 255
            self.gui_setLabelsColors(r, g, b)
            self.gui_setLabelsColors(r,g,b, custom=True)
            self.textIDsColorButton.setColor((r, g, b))
        else:
            self.gui_setLabelsColors(0,0,0, custom=False)

        # Blank image
        self.blank = np.zeros((256,256), np.uint8)

        # Left image
        self.img1 = pg.ImageItem(self.blank)
        self.ax1.addItem(self.img1)
        for action in self.hist.gradient.menu.actions():
            try:
                action.name
                action.triggered.disconnect()
                action.triggered.connect(self.gradientContextMenuClicked)
            except AttributeError:
                pass

        # Right image
        self.img2 = pg.ImageItem(self.blank)
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

    def gui_createIDsAxesItems(self):
        allIDs = set()
        for lab in self.data[self.pos_i].segm_data:
            IDs = [obj.label for obj in skimage.measure.regionprops(lab)]
            allIDs.update(IDs)

        numItems = numba_max(self.data[self.pos_i].segm_data)

        self.logger.info(f'Creating {len(allIDs)} axes items...')

        self.ax1_ContoursCurves = [None]*numItems
        self.ax2_ContoursCurves = [None]*numItems
        self.ax1_BudMothLines = [None]*numItems
        self.ax1_LabelItemsIDs = [None]*numItems
        self.ax2_LabelItemsIDs = [None]*numItems
        for ID in allIDs:
            self.ax1_ContoursCurves[ID-1] = pg.PlotDataItem()
            self.ax1_BudMothLines[ID-1] = pg.PlotDataItem()
            self.ax1_LabelItemsIDs[ID-1] = pg.LabelItem()
            self.ax2_LabelItemsIDs[ID-1] = pg.LabelItem()
            self.ax2_ContoursCurves[ID-1] = pg.PlotDataItem()

        self.creatingAxesItemsFinished()

    def gui_createGraphicsItems(self):
        # Contour pens
        self.oldIDs_cpen = pg.mkPen(color=(200, 0, 0, 255*0.5), width=2)
        self.newIDs_cpen = pg.mkPen(color='r', width=3)
        self.tempNewIDs_cpen = pg.mkPen(color='g', width=3)
        self.lostIDs_cpen = pg.mkPen(color=(245, 184, 0, 100), width=4)

        # Lost ID question mark text color
        self.lostIDs_qMcolor = (245, 184, 0)

        # New bud-mother line pen
        self.NewBudMoth_Pen = pg.mkPen(color='r', width=3, style=Qt.DashLine)

        # Old bud-mother line pen
        self.OldBudMoth_Pen = pg.mkPen(color=(255,165,0), width=2,
                                       style=Qt.DashLine)

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
        self.progressWin = apps.QDialogWorkerProcess(
            title='Creating axes items', parent=self,
            pbarDesc='Creating axes items...'
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
        self.hist.vb.contextMenuEvent = self.gui_raiseContextMenuLUT

    def gui_initImg1BottomWidgets(self):
        myutils.setRetainSizePolicy(self.zSliceScrollBar)
        self.zSliceScrollBar.hide()
        myutils.setRetainSizePolicy(self.zProjComboBox)
        self.zProjComboBox.hide()
        self.z_label.hide()

        myutils.setRetainSizePolicy(self.zSliceOverlay_SB)
        self.zSliceOverlay_SB.hide()
        self.zProjOverlay_CB.hide()
        self.overlay_z_label.hide()

        myutils.setRetainSizePolicy(self.alphaScrollBar)
        self.alphaScrollBar.hide()
        self.alphaScrollBar_label.hide()

    @exception_handler
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
            (left_click or isPanImageClick) and not eraserON and not
            brushON and not separateON
        )

        # Enable dragging of the image window like pyqtgraph original code
        if dragImg:
            pg.ImageItem.mousePressEvent(self.img2, event)
            event.ignore()
            return

        x, y = event.pos().x(), event.pos().y()
        ID = posData.lab[int(y), int(x)]

        if mode == 'Viewer':
            self.startBlinkingModeCB()
            event.ignore()
            return

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
                action.triggered.connect(self.removeROI)
                self.roiContextMenu.addAction(action)
                self.roiContextMenu.exec_(event.screenPos())
                return
            elif dragRoi:
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
            Y, X = posData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x
                # Keep a global mask to compute which IDs got erased
                self.erasedIDs = []
                self.erasedID = posData.lab[ydata, xdata]

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                # Build eraser mask
                mask = np.zeros(posData.lab.shape, bool)
                mask[ymin:ymax, xmin:xmax][diskMask] = True

                # If user double-pressed 'b' then erase over ALL labels
                color = self.eraserButton.palette().button().color().name()
                eraseOnlyOneID = (
                    color != self.doublePressKeyButtonColor
                    and self.erasedID != 0
                )
                if eraseOnlyOneID:
                    mask[posData.lab!=self.erasedID] = False

                self.eraseOnlyOneID = eraseOnlyOneID

                self.erasedIDs.extend(posData.lab[mask])
                posData.lab[mask] = 0
                self.img2.updateImage()

                self.isMouseDragImg2 = True

        # Paint with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        elif left_click and canBrush:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = posData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                ID = posData.lab[ydata, xdata]

                # If user double-pressed 'b' then draw over the labels
                color = self.brushButton.palette().button().color().name()
                drawUnder = color != self.doublePressKeyButtonColor

                if ID > 0 and drawUnder:
                    self.ax2BrushID = ID
                    posData.isNewID = False
                else:
                    self.setBrushID()
                    self.ax2BrushID = posData.brushID
                    posData.isNewID = True

                self.isMouseDragImg2 = True

                # Draw new objects
                localLab = posData.lab[ymin:ymax, xmin:xmax]
                mask = diskMask.copy()
                if drawUnder:
                    mask[localLab!=0] = False

                posData.lab[ymin:ymax, xmin:xmax][mask] = self.ax2BrushID
                self.setImageImg2()

        # Delete entire ID (set to 0)
        elif middle_click and canDelete:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            delID = posData.lab[ydata, xdata]
            if delID == 0:
                delID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to delete',
                    parent=self, allowedValues=posData.IDs
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

            self.checkIDs_LostNew()
            self.highlightLostNew()
            self.checkIDsMultiContour()

        # Separate bud
        elif (right_click or left_click) and self.separateBudButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                sepID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to split',
                    parent=self, allowedValues=posData.IDs
                )
                sepID_prompt.exec_()
                if sepID_prompt.cancel:
                    return
                else:
                    ID = sepID_prompt.EntryID

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            max_ID = numba_max(posData.lab)

            if right_click:
                posData.lab, success = self.auto_separate_bud_ID(
                                             ID, posData.lab, posData.rp,
                                             max_ID, enforce=True)
            else:
                success = False

            # If automatic bud separation was not successfull call manual one
            if not success:
                posData.disableAutoActivateViewerWindow = True
                img = self.getDisplayedCellsImg()
                manualSep = apps.manualSeparateGui(
                                posData.lab, ID, img,
                                fontSize=self.fontSize,
                                IDcolor=self.img2.lut[ID],
                                parent=self)
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
                posData.lab[manualSep.lab!=0] = manualSep.lab[manualSep.lab!=0]
                posData.disableAutoActivateViewerWindow = False

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in posData.rp]
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True)

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
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                clickedBkgrID = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'fill the holes of',
                    parent=self, allowedValues=posData.IDs
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
                localFill = scipy.ndimage.binary_fill_holes(obj.image)
                posData.lab[obj.slice][localFill] = ID

                self.update_rp()
                self.updateALLimg()

                if not self.fillHolesToolButton.findChild(QAction).isChecked():
                    self.fillHolesToolButton.setChecked(False)

        # Hull contour
        elif right_click and self.hullContToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'replace with Hull contour',
                    parent=self, allowedValues=posData.IDs
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
                localHull = skimage.morphology.convex_hull_image(obj.image)
                posData.lab[obj.slice][localHull] = ID

                self.update_rp()
                self.updateALLimg()

                if not self.hullContToolButton.findChild(QAction).isChecked():
                    self.hullContToolButton.setChecked(False)

        # Merge IDs
        elif right_click and self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here first ID that you want to merge',
                    parent=self, allowedValues=posData.IDs
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
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                editID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to replace with a new one',
                    parent=self, allowedValues=posData.IDs
                )
                editID_prompt.show(block=True)

                if editID_prompt.cancel:
                    return
                else:
                    ID = editID_prompt.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            posData.disableAutoActivateViewerWindow = True
            prev_IDs = [obj.label for obj in posData.rp]
            editID = apps.editID_QWidget(ID, prev_IDs, parent=self)
            editID.show(block=True)
            if editID.cancel:
                posData.disableAutoActivateViewerWindow = False
                if not self.editID_Button.findChild(QAction).isChecked():
                    self.editID_Button.setChecked(False)
                return

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    ID, 'Edit ID',
                                    posData.doNotShowAgain_EditID,
                                    posData.UndoFutFrames_EditID,
                                    posData.applyFutFrames_EditID,
                                    applyTrackingB=True)

            if UndoFutFrames is None:
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            for old_ID, new_ID in editID.how:
                self.addNewItems(new_ID)

                if new_ID in prev_IDs:
                    tempID = numba_max(posData.lab) + 1
                    posData.lab[posData.lab == old_ID] = tempID
                    posData.lab[posData.lab == new_ID] = old_ID
                    posData.lab[posData.lab == tempID] = new_ID

                    old_ID_idx = prev_IDs.index(old_ID)
                    new_ID_idx = prev_IDs.index(new_ID)

                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = posData.rp[old_ID_idx]
                    y, x = obj.centroid
                    y, x = int(y), int(x)
                    posData.editID_info.append((y, x, new_ID))
                    obj = posData.rp[new_ID_idx]
                    y, x = obj.centroid
                    y, x = int(y), int(x)
                    posData.editID_info.append((y, x, old_ID))
                else:
                    posData.lab[posData.lab == old_ID] = new_ID
                    old_ID_idx = posData.IDs.index(old_ID)

                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = posData.rp[old_ID_idx]
                    y, x = obj.centroid
                    y, x = int(y), int(x)
                    posData.editID_info.append((y, x, new_ID))

            # Update rps
            self.update_rp()

            # Since we manually changed an ID we don't want to repeat tracking
            self.checkIDs_LostNew()
            self.highlightLostNew()
            self.checkIDsMultiContour()

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
                                tempID = numba_max(posData.lab) + 1
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
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                binID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to remove from the analysis',
                    parent=self, allowedValues=posData.IDs
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
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                ripID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as dead',
                    parent=self, allowedValues=posData.IDs
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

    @exception_handler
    def gui_mouseDragEventImg1(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return


        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.drawAutoContour(y, x)

        # Brush dragging mouse --> keep painting
        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = posData.lab.shape

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            # Build brush mask
            mask = np.zeros(posData.lab.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            # If user double-pressed 'b' then draw over the labels
            color = self.brushButton.palette().button().color().name()
            drawUnder = color != self.doublePressKeyButtonColor
            if drawUnder:
                mask[posData.lab!=0] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.brushButton, brush=self.ax2_BrushCircleBrush
                )

            # Apply brush mask
            posData.lab[mask] = posData.brushID
            self.setImageImg2()

            brushMask = posData.lab == posData.brushID
            self.setTempImg1Brush(brushMask)

        # Eraser dragging mouse --> keep erasing
        elif self.isMouseDragImg1 and self.eraserButton.isChecked():
            posData = self.data[self.pos_i]
            Y, X = posData.lab.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()

            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(posData.lab.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[posData.lab!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )


            self.erasedIDs.extend(posData.lab[mask])
            posData.lab[mask] = 0

            self.setImageImg2()

            self.erasesedLab = np.zeros_like(posData.lab)
            for erasedID in np.unique(self.erasedIDs):
                if erasedID == 0:
                    continue
                self.erasesedLab[posData.lab==erasedID] = erasedID
            erasedRp = skimage.measure.regionprops(self.erasesedLab)
            for obj in erasedRp:
                idx = obj.label-1
                curveID = self.ax1_ContoursCurves[idx]
                cont = self.getObjContours(obj)
                curveID.setData(cont[:,0], cont[:,1], pen=self.oldIDs_cpen)

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

    def gui_hoverEventImg1(self, event):
        posData = self.data[self.pos_i]
        # Update x, y, value label bottom right
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        # Cursor left image --> restore cursor
        if event.isExit() and self.app.overrideCursor() is not None:
            self.app.restoreOverrideCursor()
            if self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

        # Alt key was released --> restore cursor
        noModifier = QGuiApplication.keyboardModifiers() == Qt.NoModifier
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        setWandCursor = (
            self.wandToolButton.isChecked() and not event.isExit()
            and noModifier
        )
        if setWandCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(self.wandCursor)

        setCurvCursor = (
            self.curvToolButton.isChecked() and not event.isExit()
            and noModifier
        )
        if setCurvCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(self.curvCursor)

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
                maxVal = numba_max(_img)
                ID = posData.lab[ydata, xdata]
                maxID = numba_max(posData.lab)
                if _img.ndim > 2:
                    val = [v for v in val]
                    value = f'{val}'
                else:
                    value = f'{val:.2f}'
                txt = (
                    f'x={x:.2f}, y={y:.2f}, value={value}, '
                    f'max={maxVal:.2f}, ID={ID}, max_ID={maxID}'
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

        # Draw eraser circle
        drawCircle = self.eraserButton.isChecked() and not event.isExit()
        if not event.isExit() and drawCircle:
            x, y = event.pos()
            self.updateEraserCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )

        # Draw Brush circle
        drawCircle = self.brushButton.isChecked() and not event.isExit()
        if not event.isExit() and drawCircle:
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
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                self.BudMothTempLine.setData([x1, x2], [y1, y2])
            else:
                obj_idx = posData.IDs.index(ID)
                obj = posData.rp[obj_idx]
                y2, x2 = obj.centroid
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
            self.app.restoreOverrideCursor()

        # Alt key was released --> restore cursor
        noModifier = QGuiApplication.keyboardModifiers() == Qt.NoModifier
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        # Cursor is moving on image while Alt key is pressed --> pan cursor
        alt = QGuiApplication.keyboardModifiers() == Qt.AltModifier
        setPanImageCursor = alt and not event.isExit()
        if setPanImageCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)

        # Update x, y, value label bottom right
        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.img2.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(
                    f'(x={x:.2f}, y={y:.2f}, value={val:.0f}, max={numba_max(_img)})'
                )
            else:
                if self.eraserButton.isChecked() or self.brushButton.isChecked():
                    self.gui_mouseReleaseEventImg2(event)
                self.wcLabel.setText(f'')

        # Draw eraser circle
        drawCircle = self.eraserButton.isChecked() and not event.isExit()
        if drawCircle and not event.isExit():
            x, y = event.pos()
            self.updateEraserCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )

        # Draw Brush circle
        drawCircle = self.brushButton.isChecked() and not event.isExit()
        if drawCircle and not event.isExit():
            x, y = event.pos()
            self.updateBrushCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

    def gui_raiseContextMenuLUT(self, event):
        posData = self.data[self.pos_i]
        posData.lutmenu = QMenu(self)
        posData.lutmenu.addAction(self.userChNameAction)
        for action in posData.fluoDataChNameActions:
            posData.lutmenu.addAction(action)
        posData.lutmenu.exec(event.screenPos())

    @exception_handler
    def gui_mouseDragEventImg2(self, event):
        # Eraser dragging mouse --> keep erasing
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            posData = self.data[self.pos_i]
            Y, X = posData.lab.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(posData.lab.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[posData.lab!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )

            self.erasedIDs.extend(posData.lab[mask])

            posData.lab[mask] = 0
            self.setImageImg2()

        # Brush paint dragging mouse --> keep painting
        if self.isMouseDragImg2 and self.brushButton.isChecked():
            posData = self.data[self.pos_i]
            Y, X = posData.lab.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            # Build brush mask
            mask = np.zeros(posData.lab.shape, bool)
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
            posData.lab[mask] = self.ax2BrushID
            self.setImageImg2()

    @exception_handler
    def gui_mouseReleaseEventImg2(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        # Eraser mouse release --> update IDs and contours
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            self.isMouseDragImg2 = False
            erasedIDs = np.unique(self.erasedIDs)

            # Update data (rp, etc)
            self.update_rp()
            self.updateALLimg()

            for ID in erasedIDs:
                if ID not in posData.lab:
                    self.warnEditingWithCca_df('Delete ID with eraser')
                    break

        # Brush mouse release --> update IDs and contours
        elif self.isMouseDragImg2 and self.brushButton.isChecked():
            self.isMouseDragImg2 = False

            self.update_rp()
            if posData.isNewID:
                self.tracking(enforce=True)

            self.updateALLimg()
            self.warnEditingWithCca_df('Add new ID with brush tool')

        # Merge IDs
        elif self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to merge with ID '
                         f'{self.firstID}',
                    parent=self, allowedValues=posData.IDs
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            posData.lab[posData.lab==ID] = self.firstID

            # Mask to keep track of which ID needs redrawing of the contours
            mergedID_mask = posData.lab==self.firstID

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True)

            self.updateALLimg()
            if not self.mergeIDsButton.findChild(QAction).isChecked():
                self.mergeIDsButton.setChecked(False)
            self.store_data()
            self.warnEditingWithCca_df('Merge IDs')

    @exception_handler
    def gui_mouseReleaseEventImg1(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        if mode=='Segmentation and Tracking' or self.isSnapshot:
            # Allow right-click actions on both images
            self.gui_mouseReleaseEventImg2(event)

        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            self.isRightClickDragImg1 = False
            try:
                self.splineToObj(isRightClick=True)
                self.update_rp()
                self.tracking(enforce=True)
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
            self.updateALLimg()

            for ID in erasedIDs:
                if ID not in posData.lab:
                    self.warnEditingWithCca_df('Delete ID with eraser')
                    break

        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            self.isMouseDragImg1 = False

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True)

            # Update colors to include a new color for the new ID
            self.updateALLimg()
            self.warnEditingWithCca_df('Add new ID with brush tool')

        # Wand tool release, add new object
        elif self.isMouseDragImg1 and self.wandToolButton.isChecked():
            self.isMouseDragImg1 = False

            posData = self.data[self.pos_i]
            posData.lab[self.flood_mask] = posData.brushID

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True)

            # Update colors to include a new color for the new ID
            self.updateALLimg()
            self.warnEditingWithCca_df('Add new ID with magic-wand')

        # Assign mother to bud
        elif self.assignBudMothButton.isChecked() and self.clickedOnBud:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = posData.lab[ydata, xdata]
            if ID == posData.lab[self.yClickBud, self.xClickBud]:
                return

            if ID == 0:
                mothID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as mother cell',
                    parent=self, allowedValues=posData.IDs
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
                txt = (f'You clicked on ID {ID} which is a BUD.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QMessageBox()
                msg.critical(
                    self, 'Released on a bud', txt, msg.Ok
                )
                return

            elif ccs != 'G1' and posData.frame_i > 0:
                txt = (f'You clicked on a cell (ID={ID}) which is NOT in G1.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QMessageBox()
                msg.critical(
                    self, 'Released on a cell NOT in G1', txt, msg.Ok
                )
                return

            elif posData.frame_i == 0:
                # Check that clicked bud actually is smaller that mother
                # otherwise warn the user that he might have clicked first
                # on a mother
                budID = posData.lab[self.yClickBud, self.xClickBud]
                new_mothID = posData.lab[ydata, xdata]
                bud_obj_idx = posData.IDs.index(budID)
                new_moth_obj_idx = posData.IDs.index(new_mothID)
                rp_budID = posData.rp[bud_obj_idx]
                rp_new_mothID = posData.rp[new_moth_obj_idx]
                if rp_budID.area >= rp_new_mothID.area:
                    msg = QMessageBox(self)
                    msg.setWindowTitle('Which one is bud?')
                    msg.setIcon(msg.Warning)
                    msg.setText(
                        f'You clicked FIRST on ID {budID} and then on {new_mothID}.\n'
                        f'For me this means that you want ID {budID} to be the '
                        f'BUD of ID {new_mothID}.\n'
                        f'However ID {budID} is bigger than {new_mothID} '
                        f'so maybe you shoul have clicked FIRST on {new_mothID}?\n\n'
                        'What do you want me to do?'
                    )
                    swapButton = QPushButton(
                            f'Assign ID {new_mothID} as the bud of ID {budID}'
                    )
                    keepButton = QPushButton(
                            f'Keep ID {budID} as the bud of  ID {new_mothID}'
                    )
                    msg.addButton(swapButton, msg.YesRole)
                    msg.addButton(keepButton, msg.NoRole)
                    msg.exec_()
                    if msg.clickedButton() == swapButton:
                        (xdata, ydata,
                        self.xClickBud, self.yClickBud) = (
                            self.xClickBud, self.yClickBud,
                            xdata, ydata
                        )

            elif is_history_known and not self.clickedOnHistoryKnown:
                budID = posData.lab[ydata, xdata]
                # Allow assigning an unknown cell ONLY to another unknown cell
                txt = (
                    f'You started by clicking on ID {budID} which has '
                    'UNKNOWN history, but you then clicked/released on '
                    f'ID {ID} which has KNOWN history.\n\n'
                    'Only two cells with UNKNOWN history can be assigned as '
                    'relative of each other.')
                msg = QMessageBox()
                msg.critical(
                    self, 'Released on a cell with KNOWN history', txt, msg.Ok
                )
                return

            self.clickedOnHistoryKnown = is_history_known
            self.xClickMoth, self.yClickMoth = xdata, ydata
            self.assignBudMoth()

            if not self.assignBudMothButton.findChild(QAction).isChecked():
                self.assignBudMothButton.setChecked(False)

            self.clickedOnBud = False
            self.BudMothTempLine.setData([], [])

    @exception_handler
    def gui_mousePressEventImg1(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        isMod = ctrl or alt
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        is_cca_on = mode == 'Cell cycle analysis' or self.isSnapshot
        left_click = event.button() == Qt.MouseButton.LeftButton and not isMod
        middle_click = self.isMiddleClick(event, modifiers)
        right_click = event.button() == Qt.MouseButton.RightButton and not isMod
        isPanImageClick = self.isPanImageClick(event, modifiers)
        brushON = self.brushButton.isChecked()
        curvToolON = self.curvToolButton.isChecked()
        histON = self.setIsHistoryKnownButton.isChecked()
        eraserON = self.eraserButton.isChecked()
        rulerON = self.rulerButton.isChecked()
        wandON = self.wandToolButton.isChecked() and not isPanImageClick

        dragImgLeft = (
            (left_click or isPanImageClick) and not brushON and not histON
            and not curvToolON and not eraserON and not rulerON
            and not wandON
        )

        dragImgMiddle = middle_click

        # Right click in snapshot mode is for spline tool
        canAnnotateDivision = (
             not self.assignBudMothButton.isChecked()
             and not self.setIsHistoryKnownButton.isChecked()
             and not (self.isSnapshot and self.curvToolButton.isChecked())
        )

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
        if dragImgLeft:
            pg.ImageItem.mousePressEvent(self.img1, event)
            event.ignore()
            return

        if dragImgMiddle:
            pg.ImageItem.mousePressEvent(self.img1, event)
            event.ignore()
            return

        if mode == 'Viewer' and not canRuler:
            self.startBlinkingModeCB()
            event.ignore()
            return

        # Allow right-click actions on both images
        eventOnImg2 = (
            (right_click)
            and (mode=='Segmentation and Tracking' or self.isSnapshot)
        )
        if eventOnImg2:
            self.gui_mousePressEventImg2(event)

        # Paint new IDs with brush and left click on the left image
        if left_click and canBrush:
            # Store undo state before modifying stuff

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = posData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                ID = posData.lab[ydata, xdata]
                self.storeUndoRedoStates(False)

                # If user double-pressed 'b' then draw over the labels
                color = self.brushButton.palette().button().color().name()
                drawUnder = color != self.doublePressKeyButtonColor

                if ID > 0 and drawUnder:
                    posData.brushID = posData.lab[ydata, xdata]
                    self.brushColor = self.img2.lut[posData.brushID]/255
                else:
                    # Update brush ID. Take care of disappearing cells to remember
                    # to not use their IDs anymore in the future
                    self.setBrushID()
                    self.brushColor = posData.lut[posData.brushID]/255

                self.yPressAx2, self.xPressAx2 = y, x

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                self.isMouseDragImg1 = True

                # Draw new objects
                localLab = posData.lab[ymin:ymax, xmin:xmax]
                mask = diskMask.copy()
                if drawUnder:
                    mask[localLab!=0] = False

                posData.lab[ymin:ymax, xmin:xmax][mask] = posData.brushID
                self.setImageImg2()

                img = self.img1.image.copy()
                if self.overlayButton.isChecked():
                    self.imgRGB = img/numba_max(img)
                else:
                    img = img/numba_max(img)
                    self.imgRGB = gray2rgb(img)

                brushMask = posData.lab == posData.brushID
                self.setTempImg1Brush(brushMask)

        elif left_click and canErase:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = posData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x
                # Keep a list of erased IDs got erased
                self.erasedIDs = []
                self.erasedID = posData.lab[ydata, xdata]

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                # Build eraser mask
                mask = np.zeros(posData.lab.shape, bool)
                mask[ymin:ymax, xmin:xmax][diskMask] = True

                # If user double-pressed 'b' then erase over ALL labels
                color = self.eraserButton.palette().button().color().name()
                eraseOnlyOneID = (
                    color != self.doublePressKeyButtonColor
                    and self.erasedID != 0
                )

                self.eraseOnlyOneID = eraseOnlyOneID

                if eraseOnlyOneID:
                    mask[posData.lab!=self.erasedID] = False


                self.erasedIDs.extend(posData.lab[mask])

                posData.lab[mask] = 0

                self.erasesedLab = np.zeros_like(posData.lab)
                for erasedID in np.unique(self.erasedIDs):
                    if erasedID == 0:
                        continue
                    self.erasesedLab[posData.lab==erasedID] = erasedID
                erasedRp = skimage.measure.regionprops(self.erasesedLab)
                for obj in erasedRp:
                    idx = obj.label-1
                    curveID = self.ax1_ContoursCurves[idx]
                    cont = self.getObjContours(obj)
                    curveID.setData(cont[:,0], cont[:,1], pen=self.oldIDs_cpen)

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
                self.ax1_rulerPlotItem.setData([xxRA[0], xdata], [yyRA[0], ydata])
                self.ax1_rulerAnchorsItem.setData([xxRA[0], xdata], [yyRA[0], ydata])

        elif right_click and canCurv:
            # Draw manually assisted auto contour
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = posData.lab.shape
            clickOnImg = x >= 0 and x < X and y >= 0 and y < Y
            if not clickOnImg:
                return

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
            Y, X = posData.lab.shape
            clickOnImg = x >= 0 and x < X and y >= 0 and y < Y
            if not clickOnImg:
                return

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
                self.tracking(enforce=True)
                self.updateALLimg()
                self.warnEditingWithCca_df('Add new ID with curvature tool')
                self.clearCurvItems()
                self.curvTool_cb(True)

        elif left_click and canWand:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = posData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)

                posData.brushID = posData.lab[ydata, xdata]
                if posData.brushID == 0:
                    self.setBrushID()
                    self.updateLookuptable(
                        lenNewLut=numba_max(posData.lab)+posData.brushID+1
                    )
                self.brushColor = self.img2.lut[posData.brushID]/255

                img = self.img1.image.copy()
                img = img/numba_max(img)
                if self.overlayButton.isChecked():
                    self.imgRGB = img/numba_max(img)
                else:
                    self.imgRGB = gray2rgb(img)

                # NOTE: flood is on mousedrag or release
                tol = self.wandToleranceSlider.value()
                self.flood_img = myutils.to_uint8(self.getDisplayedCellsImg())
                flood_mask = skimage.segmentation.flood(
                    self.flood_img, (ydata, xdata), tolerance=tol
                )
                bkgrLabMask = posData.lab==0

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
        elif right_click and is_cca_on and canAnnotateDivision:
            if posData.frame_i <= 0 and not self.isSnapshot:
                return

            if posData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                divID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as divided',
                    parent=self, allowedValues=posData.IDs
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
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                budID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID of a bud you want to correct mother assignment',
                    parent=self, allowedValues=posData.IDs
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
            ID = posData.lab[ydata, xdata]
            if ID == 0:
                unknownID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as '
                         '"history UNKNOWN/KNOWN"',
                    parent=self, allowedValues=posData.IDs
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

        # Allow mid-click actions on both images
        elif middle_click:
            self.gui_mousePressEventImg2(event)

    def storeTrackingAlgo(self, checked):
        if not checked:
            return

        trackingAlgo = self.sender().text()
        self.df_settings.at['tracking_algorithm', 'value'] = trackingAlgo
        self.df_settings.to_csv(self.settings_csv_path)

        if self.sender().text() == 'YeaZ':
            msg = QMessageBox()
            info_txt = (f"""
            <p style="font-size:10pt">
                Note that YeaZ tracking algorithm is slightly more accurate,
                but it is about <b>5-6 times slower</b>. This results in a
                detectable delay when visualizing the next frame
                (about 300 ms delay with 100 cells).
            </p>
            """)
            msg.information(self, 'Info about YeaZ', info_txt, msg.Ok)

    def findID(self):
        posData = self.data[self.pos_i]
        self.searchingID = True
        searchIDdialog = apps.QLineEditDialog(
            title='Search object by ID',
            msg='Enter object ID to find and highlight',
            parent=self, allowedValues=posData.IDs
        )
        searchIDdialog.exec_()
        if searchIDdialog.cancel:
            return
        self.highlightSearchedID(searchIDdialog.EntryID)

    def workerProgress(self, text):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.info(text)

    def workerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
        self.logger.info('Relabelling process ended.')
        self.updateALLimg()

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)

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
        img = img/numba_max(img)
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
            ask_SizeT=True,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            save=True, singlePos=True
        )

    def setHoverToolSymbolData(self, xx, yy, ScatterItems, size=None):
        for item in ScatterItems:
            if size is None:
                item.setData(xx, yy)
            else:
                item.setData(xx, yy, size=size)

    def setHoverToolSymbolColor(self, xdata, ydata, pen, ScatterItems, button,
                                brush=None, hoverRGB=None, ID=None):
        posData = self.data[self.pos_i]
        hoverID = posData.lab[ydata, xdata] if ID is None else ID
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
            img = img/numba_max(img)
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

    def removeROI(self, event):
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        self.store_data()
        for i in range(posData.frame_i, posData.segmSizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            if self.roi_to_del in delROIs_info['rois']:
                posData.frame_i = i
                idx = delROIs_info['rois'].index(self.roi_to_del)
                # Restore deleted IDs from already visited frames
                if posData.allData_li[i]['labels'] is not None:
                    if len(delROIs_info['delIDsROI'][idx]) > 1:
                        posData.lab = posData.allData_li[i]['labels']
                        self.restoreDelROIlab(self.roi_to_del, enforce=True)
                        posData.allData_li[i]['labels'] = posData.lab
                        self.get_data()
                        self.store_data()
                delROIs_info['rois'].pop(idx)
                delROIs_info['delMasks'].pop(idx)
                delROIs_info['delIDsROI'].pop(idx)

        # Back to current frame
        posData.frame_i = current_frame_i
        posData.lab = posData.allData_li[posData.frame_i]['labels']
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
                if self.invertBwAction.isChecked():
                    min_int = numba_min(a_dir) # if int_val > ta else numba_min(a_dir)
                else:
                    min_int = numba_max(a_dir)
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
        for i in range(posData.frame_i+1, posData.segmSizeT):
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
        for i in range(posData.frame_i+1, posData.segmSizeT):
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
            err_msg = (f"""
            <p style="font-size:10pt">
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
            </p>
            """)
            msg = QMessageBox()
            enforce_assignment = msg.warning(
               self, 'Cell not eligible', err_msg, msg.Apply | msg.Cancel
            )
            cancel = enforce_assignment == msg.Cancel
        elif why == 'not_G1_in_the_past':
            err_msg = (f"""
            <p style="font-size:10pt">
                The requested cell in G1
                (ID={new_mothID}) at past frame {i+1}
                has a bud assigned to it, therefore it cannot be
                assigned as mother of bud ID {budID}.<br>
                You can assign a cell as the mother of bud ID {budID}
                only if this cell is in G1 for the entire life of the bud.<br>
                One possible solution is to first go to frame {i+1} and
                assign the bud of cell {new_mothID} to another cell.
            </p>
            """)
            msg = QMessageBox()
            msg.warning(
               self, 'Cell not eligible', err_msg, msg.Ok
            )
            cancel = None
        elif why == 'single_frame_G1_duration':
            err_msg = (f"""
            <p style="font-size:10pt">
                Assigning bud ID {budID} to cell in G1
                (ID={new_mothID}) would result in no G1 phase at all between
                previous cell cycle and current cell cycle.
                This is very confusing for me, sorry.<br><br>
                The solution is to remove cell division anotation on cell
                {new_mothID} (right-click on it on current frame) and then
                annotate division on any frame before current frame number {i+1}.
                This will gurantee a G1 duration of cell {new_mothID}
                of <b>at least 1 frame</b>. Thanks.
            </p>
            """)
            msg = QMessageBox()
            msg.warning(
               self, 'Cell not eligible', err_msg, msg.Ok
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
        for i in range(posData.frame_i, posData.segmSizeT):
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
            if not is_bud_existing:
                # Bud was not emerged yet
                break

            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1':
                self.warnMotherNotEligible(
                    new_mothID, budID, i, 'not_G1_in_the_past'
                )
                eligible = False
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
        budID = posData.lab[self.yClickBud, self.xClickBud]
        new_mothID = posData.lab[self.yClickMoth, self.xClickMoth]

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
        for i in range(posData.frame_i+1, posData.segmSizeT):
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
        for i in range(posData.frame_i, posData.segmSizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            delROIs_info['rois'].append(roi)
            delROIs_info['delMasks'].append(np.zeros_like(posData.lab))
            delROIs_info['delIDsROI'].append(set())
        self.ax2.addItem(roi)

    def getDelROI(self, xl=None, yb=None, w=32, h=32):
        posData = self.data[self.pos_i]
        if xl is None:
            xRange, yRange = self.ax2.viewRange()
            xl, yb = abs(xRange[0]), abs(yRange[0])
        Y, X = posData.lab.shape
        roi = pg.ROI([xl, yb], [w, h],
                     rotatable=False,
                     removable=True,
                     pen=pg.mkPen(color='r'),
                     maxBounds=QRectF(QRect(0,0,X,Y)))

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

        roi.sigRegionChanged.connect(self.ROImoving)
        roi.sigRegionChangeFinished.connect(self.cropROIovingFinished)
        return roi

    def ROImoving(self, roi):
        roi.setPen(color=(255,255,0))
        # First bring back IDs if the ROI moved away
        self.restoreDelROIlab(roi)
        self.setImageImg2()

    def cropROIovingFinished(self, roi):
        roi.setPen(color='r')
        self.update_rp()
        self.updateALLimg()

    def restoreDelROIlab(self, roi, enforce=True):
        posData = self.data[self.pos_i]
        x0, y0 = [int(c) for c in roi.pos()]
        w, h = [int(c) for c in roi.size()]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        idx = delROIs_info['rois'].index(roi)
        delMask = delROIs_info['delMasks'][idx]
        delIDs = delROIs_info['delIDsROI'][idx]
        ROImask = np.zeros(self.img2.image.shape, bool)
        ROImask[y0:y0+h, x0:x0+w] = True
        overlapROIdelIDs = np.unique(delMask[ROImask])
        for ID in delIDs:
            if ID>0 and ID not in overlapROIdelIDs and not enforce:
                posData.lab[delMask==ID] = ID
                delMask[delMask==ID] = 0
            elif ID>0 and enforce:
                posData.lab[delMask==ID] = ID
                delMask[delMask==ID] = 0

    def getDelROIlab(self):
        posData = self.data[self.pos_i]
        DelROIlab = posData.lab
        allDelIDs = set()
        # Iterate rois and delete IDs
        for roi in posData.allData_li[posData.frame_i]['delROIs_info']['rois']:
            if roi not in self.ax2.items:
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
            delIDsROI.update(delIDs)
            allDelIDs.update(delIDs)
            _DelROIlab = posData.lab.copy()
            for obj in posData.rp:
                ID = obj.label
                if ID in delIDs:
                    delObjROImask[posData.lab==ID] = ID
                    _DelROIlab[posData.lab==ID] = 0
                    # LabelItemID = self.ax2_LabelItemsIDs[ID-1]
                    # LabelItemID.setText('')
                    # LabelItemID = self.ax1_LabelItemsIDs[ID-1]
                    # LabelItemID.setText('')
            DelROIlab[_DelROIlab == 0] = 0
            # Keep a mask of deleted IDs to bring them back when roi moves
            delROIs_info['delMasks'][idx] = delObjROImask
            delROIs_info['delIDsROI'][idx] = delIDsROI
        return allDelIDs, DelROIlab

    def gaussBlur(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPointSize(10)
            self.gaussWin = apps.gaussBlurDialog(self)
            self.gaussWin.setFont(font)
            self.gaussWin.show()
        else:
            self.gaussWin.close()
            self.gaussWin = None

    def edgeDetection(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPointSize(10)
            self.edgeWin = apps.edgeDetectionDialog(self)
            self.edgeWin.setFont(font)
            self.edgeWin.show()
        else:
            self.edgeWin.close()
            self.edgeWin = None

    def entropyFilter(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPointSize(10)
            self.entropyWin = apps.entropyFilterDialog(self)
            self.entropyWin.setFont(font)
            self.entropyWin.show()
        else:
            self.entropyWin.close()
            self.entropyWin = None

    def setData(self, data):
        if posData.SizeZ > 1:
            pass

    def enableSmartTrack(self, checked):
        posData = self.data[self.pos_i]
        # Disable tracking for already visited frames
        if posData.allData_li[posData.frame_i]['labels'] is not None:
            self.disableTrackingCheckBox.setChecked(True)
        else:
            self.disableTrackingCheckBox.setChecked(False)

        if checked:
            self.UserEnforced_DisabledTracking = False
            self.UserEnforced_Tracking = False
        else:
            if self.disableTrackingCheckBox.isChecked():
                self.UserEnforced_DisabledTracking = True
                self.UserEnforced_Tracking = False
            else:
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    def invertBw(self, checked):
        if self.slideshowWin is not None:
            self.slideshowWin.is_bw_inverted = checked
            self.slideshowWin.update_img()
        self.df_settings.at['is_bw_inverted', 'value'] = str(checked)
        self.df_settings.to_csv(self.settings_csv_path)
        if checked:
            self.ax2_BrushCirclePen = pg.mkPen((0,0,0), width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((0,0,0,50))
            self.ax1_oldIDcolor = [255-v for v in self.ax1_oldIDcolor]
            self.ax1_G1cellColor = [255-v for v in self.ax1_G1cellColor[:3]]
            self.ax1_G1cellColor.append(178)
            self.ax1_S_oldCellColor = [255-v for v in self.ax1_S_oldCellColor]
        else:
            self.ax2_BrushCirclePen = pg.mkPen(width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
            self.ax1_oldIDcolor = [255-v for v in self.ax1_oldIDcolor]
            self.ax1_G1cellColor = [255-v for v in self.ax1_G1cellColor[:3]]
            self.ax1_G1cellColor.append(178)
            self.ax1_S_oldCellColor = [255-v for v in self.ax1_S_oldCellColor]

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

    @exception_handler
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
            self.zSliceScrollBar.setDisabled(False)
            self.zProjComboBox.show()
            self.zSliceScrollBar.show()
            self.z_label.show()
        else:
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
        self.updateALLimg()
        how = self.drawIDsContComboBox.currentText()
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'

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
        self.ccaToolBar.setVisible(enabled)
        for action in self.ccaToolBar.actions():
            button = self.ccaToolBar.widgetForAction(action)
            action.setVisible(enabled)
            button.setEnabled(enabled)

    def setEnabledEditToolbarButton(self, enabled=False):
        for action in self.segmActions:
            action.setEnabled(enabled)
        self.SegmActionRW.setEnabled(enabled)
        self.repeatTrackingMenuAction.setEnabled(enabled)
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
            if action == self.openAction:
                continue
            action.setEnabled(enabled)
            button.setEnabled(enabled)

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
        posData = self.data[self.pos_i]
        mode = self.modeComboBox.itemText(idx)
        if mode == 'Segmentation and Tracking':
            self.trackingMenu.setDisabled(False)
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(True)
            self.initSegmTrackMode()
            self.setEnabledEditToolbarButton(enabled=True)
            self.addExistingDelROIs()
            self.checkTrackingEnabled()
            self.setEnabledCcaToolbar(enabled=False)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            try:
                self.undoAction.triggered.disconnect()
                self.redoAction.triggered.disconnect()
            except Exception as e:
                pass
            self.undoAction.triggered.connect(self.undo)
            self.redoAction.triggered.connect(self.redo)
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
                self.disableTrackingCheckBox.setChecked(True)
                try:
                    self.undoAction.triggered.disconnect()
                    self.redoAction.triggered.disconnect()
                except Exception as e:
                    pass
                self.undoAction.triggered.connect(self.UndoCca)
                self.drawIDsContComboBox.clear()
                self.drawIDsContComboBox.addItems(
                                        self.drawIDsContComboBoxCcaItems)
        elif mode == 'Viewer':
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(False)
            self.setEnabledEditToolbarButton(enabled=False)
            self.setEnabledCcaToolbar(enabled=False)
            self.removeAlldelROIsCurrentFrame()
            self.disableTrackingCheckBox.setChecked(True)
            self.undoAction.setDisabled(True)
            self.redoAction.setDisabled(True)
            currentMode = self.drawIDsContComboBox.currentText()
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxCcaItems)
            self.drawIDsContComboBox.setCurrentText(currentMode)
            self.navigateScrollBar.setMaximum(posData.segmSizeT)
            try:
                self.undoAction.triggered.disconnect()
                self.redoAction.triggered.disconnect()
            except Exception as e:
                pass
        elif mode == 'Snapshot':
            try:
                self.undoAction.triggered.disconnect()
                self.redoAction.triggered.disconnect()
            except Exception as e:
                pass
            self.undoAction.triggered.connect(self.undo)
            self.redoAction.triggered.connect(self.redo)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(
                self.drawIDsContComboBoxCcaItems
            )
            self.setEnabledSnapshotMode()

    def setEnabledSnapshotMode(self):
        posData = self.data[self.pos_i]
        self.manuallyEditCcaAction.setDisabled(False)
        for action in self.segmActions:
            action.setDisabled(False)
        self.SegmActionRW.setDisabled(False)
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
        self.disableTrackingCheckBox.setChecked(True)
        self.disableTrackingCheckBox.setDisabled(True)
        self.repeatTrackingAction.setVisible(False)
        button = self.editToolBar.widgetForAction(self.repeatTrackingAction)
        button.setDisabled(True)
        self.setEnabledWidgetsToolbar(False)

    def launchSlideshow(self):
        posData = self.data[self.pos_i]
        self.determineSlideshowWinPos()
        if self.slideshowButton.isChecked():
            self.slideshowWin = apps.CellsSlideshow_GUI(
                parent=self,
                button_toUncheck=self.slideshowButton,
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
        setRp = self.separateByLabelling(lab_ID, rp_ID, maxID=numba_max(lab))
        if setRp:
            success = True
            lab[lab_ID_bool] = lab_ID[lab_ID_bool]
            return lab, success

        cnt, defects = self.convexity_defects(lab_ID_bool, eps_percent)
        success = False
        if defects is not None:
            if len(defects) == 2:
                if not enforce:
                    # Yeaz watershed separation. To be tested
                    dist_watshed = segment(lab_ID_bool, None, merge=False)
                    num_obj_watshed = len(np.unique(dist_watshed))
                else:
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

    def connectLeftClickButtons(self):
        self.brushButton.toggled.connect(self.Brush_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.wandToolButton.toggled.connect(self.wand_cb)

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
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
            self.wandControlsToolbar.setVisible(True)
        else:
            self.wandControlsToolbar.setVisible(False)

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
        _img = self.img2.image
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        posData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2

        ID = posData.lab[ydata, xdata]
        if ID == 0:
            prev_lab = posData.allData_li[posData.frame_i-1]['labels']
            if prev_lab is None:
                self.restoreHoveredID()
                return
            ID = prev_lab[ydata, xdata]

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

    def updateBrushCursor(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.img2.image
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
            self.enableSizeSpinbox(True)
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
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.enableSizeSpinbox(False)

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
        Y, X = self.img2.image.shape[-2:]

        brushSize = self.brushSizeSpinbox.value()
        yBottom, xLeft = ydata-brushSize, xdata-brushSize
        yTop, xRight = ydata+brushSize+1, xdata+brushSize+1

        if xLeft>=0 and yBottom>=0 and xRight<=X and yTop<=Y:
            return yBottom, xLeft, yTop, xRight, self.diskMask

        elif xLeft<0 and yBottom>=0 and xRight<=X and yTop<=Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[:, -xLeft:]
            return yBottom, 0, yTop, xRight, diskMask

        elif xLeft>=0 and yBottom<0 and xRight<=X and yTop<=Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[-yBottom:]
            return 0, xLeft, yTop, xRight, diskMask

        elif xLeft>=0 and yBottom>=0 and xRight>X and yTop<=Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[:, 0:X-xLeft]
            return yBottom, xLeft, yTop, X, diskMask

        elif xLeft>=0 and yBottom>=0 and xRight<=X and yTop>Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[0:Y-yBottom]
            return yBottom, xLeft, Y, xRight, diskMask

    def setBrushID(self, useCurrentLab=True):
        # Make sure that the brushed ID is always a new one based on
        # already visited frames
        posData = self.data[self.pos_i]
        if useCurrentLab:
            newID = numba_max(posData.lab)
        else:
            newID = 1
        for frame_i, storedData in enumerate(posData.allData_li):
            lab = storedData['labels']
            if lab is not None:
                _max = numba_max(lab)
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
            self.uncheckLeftClickButtons(self.sender())
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

    def updateEraserCursor(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.img2.image
        Y, X = _img.shape

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
            self.enableSizeSpinbox(True)
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.brushButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.enableSizeSpinbox(False)

    @exception_handler
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_U:
            editID = apps.editID_QWidget([1,2,3], [1,2,3], parent=self)
            editID.exec_()

        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return
        isBrushActive = (
            self.brushButton.isChecked() or self.eraserButton.isChecked()
        )
        if ev.key() == Qt.Key_Up:
            if isBrushActive:
                self.brushSizeSpinbox.setValue(self.brushSizeSpinbox.value()+1)
            elif self.wandToolButton.isChecked():
                val = self.wandToleranceSlider.value()
                self.wandToleranceSlider.setValue(val+1)
        elif ev.key() == Qt.Key_Control:
            self.isCtrlDown = True
        elif ev.key() == Qt.Key_Down:
            if isBrushActive:
                self.brushSizeSpinbox.setValue(self.brushSizeSpinbox.value()-1)
            elif self.wandToolButton.isChecked():
                val = self.wandToleranceSlider.value()
                self.wandToleranceSlider.setValue(val-1)
        elif ev.key() == Qt.Key_Escape:
            self.setUncheckedAllButtons()
            if self.searchingID:
                self.updateALLimg()
        elif ev.modifiers() == Qt.AltModifier:
            isCursorSizeAll = self.app.overrideCursor() == Qt.SizeAllCursor
            # Alt is pressed while cursor is on images --> set SizeAllCursor
            if self.xHoverImg is not None and not isCursorSizeAll:
                self.app.setOverrideCursor(Qt.SizeAllCursor)
            if ev.key() == Qt.Key_C:
                font = QtGui.QFont()
                font.setPointSize(10)
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
        elif ev.modifiers() == Qt.ControlModifier:
            if ev.key() == Qt.Key_P:
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
            elif ev.key() == Qt.Key_L:
                mode = str(self.modeComboBox.currentText())
                if mode == 'Viewer' or mode == 'Cell cycle analysis':
                    self.startBlinkingModeCB()
                    return
                self.storeUndoRedoStates(False)
                posData = self.data[self.pos_i]
                if posData.SizeT > 1:
                    self.progressWin = apps.QDialogWorkerProcess(
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
                    self.update_cca_df_relabelling(posData, old_IDs, new_IDs)
                    self.store_data()
                    self.update_rp()
                    li = list(zip(oldIDs, newIDs))
                    s = '\n'.join([str(pair).replace(',', ' -->') for pair in li])
                    s = f'IDs relabelled as follows:\n{s}'
                    self.logger.info(s)

                self.updateALLimg()
        elif ev.key() == Qt.Key_T:
            pass
            # raise IndexError('Testing')
            # posData = self.data[self.pos_i]
            # self.logger.info(posData.allData_li[0]['acdc_df'])
            # # self.hist.sigLookupTableChanged.disconnect()
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
            how = self.drawIDsContComboBox.currentText()
            if how.find('nothing') == -1:
                self.drawIDsContComboBox.setCurrentText('Draw nothing')
                self.prev_how = how
            else:
                try:
                    self.drawIDsContComboBox.setCurrentText(self.prev_how)
                except Exception as e:
                    # traceback.print_exc()
                    pass
            if self.countKeyPress == 0:
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                self.Button = None
                QTimer.singleShot(400, self.doubleKeyTimerCallBack)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                self.drawIDsContComboBox.setCurrentText(
                    'Draw cell cycle info and contours'
                )
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


    def keyReleaseEvent(self, ev):
        if self.app.overrideCursor() == Qt.SizeAllCursor:
            self.app.restoreOverrideCursor()
        if ev.key() == Qt.Key_Control:
            self.isCtrlDown = False
        canRepeat = (
            ev.key() == Qt.Key_Left
            or ev.key() == Qt.Key_Right
            or ev.key() == Qt.Key_Up
            or ev.key() == Qt.Key_Down
            or ev.key() == Qt.Key_Control
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
        if posData.frame_i+1 == posData.segmSizeT:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        areFutureIDs_affected = []
        # Get number of future frames already visited and checked if future
        # frames has an ID affected by the change
        for i in range(posData.frame_i+1, posData.segmSizeT):
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

        posData.UndoRedoStates[posData.frame_i].insert(
            0,
            {'image': self.img1.image.copy(),
             'labels': posData.lab.copy(),
             'editID_info': posData.editID_info.copy(),
             'binnedIDs': posData.binnedIDs.copy(),
             'ripIDs': posData.ripIDs.copy(),
             'cca_df': cca_df}
        )

    def getCurrentState(self):
        posData = self.data[self.pos_i]
        i = posData.frame_i
        c = self.UndoCount
        self.ax1Image = posData.UndoRedoStates[i][c]['image'].copy()
        posData.lab = posData.UndoRedoStates[i][c]['labels'].copy()
        posData.editID_info = posData.UndoRedoStates[i][c]['editID_info'].copy()
        posData.binnedIDs = posData.UndoRedoStates[i][c]['binnedIDs'].copy()
        posData.ripIDs = posData.UndoRedoStates[i][c]['ripIDs'].copy()
        cca_df = posData.UndoRedoStates[i][c]['cca_df']
        if cca_df is not None:
            posData.cca_df = posData.UndoRedoStates[i][c]['cca_df'].copy()
        else:
            posData.cca_df = None

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
        for frame_i in range(0, posData.segmSizeT):
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
            self.getCurrentState()
            self.update_rp()
            self.checkIDs_LostNew()
            self.updateALLimg(image=self.ax1Image)
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
            self.getCurrentState()
            self.update_rp()
            self.checkIDs_LostNew()
            self.updateALLimg(image=self.ax1Image)
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
        if self.disableTrackingCheckBox.isChecked():
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
                self.disableTrackingCheckBox.setChecked(True)
            else:
                self.repeatTracking()
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    def repeatTracking(self):
        posData = self.data[self.pos_i]
        prev_lab = posData.lab.copy()
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
                self.manuallyEditTracking(posData.lab, allIDs)
                self.update_rp()
                self.checkIDs_LostNew()
                self.highlightLostNew()
                self.checkIDsMultiContour()
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
        font.setPointSize(10)

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

    def repeatSegm(self, model_name=''):
        if not model_name:
            idx = self.segmActions.index(self.sender())
            model_name = self.modelNames[idx]
            askSegmParams = True
        else:
            idx = self.modelNames.index(model_name)
            askSegmParams = self.segment2D_kwargs is None

        myutils.download_model(model_name)

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
                url = acdcSegment.url_help()
            except AttributeError:
                url = None

            win = apps.QDialogModelParams(
                init_params,
                segment_params,
                model_name, parent=self,
                url=url)
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
        else:
            model = self.models[idx]

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color='w')

        self.model = model

        self.thread = QThread()
        self.worker = segmWorker(self)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.segmWorkerFinished)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    @exception_handler
    def workerCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
        raise error

    def debugSegmWorker(self, lab):
        apps.imshow_tk(lab)

    def segmWorkerFinished(self, lab, exec_time):
        posData = self.data[self.pos_i]

        if posData.segmInfo_df is not None and posData.SizeZ>1:
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, 'resegmented_in_gui'] = True

        posData.lab = lab.copy()
        self.update_rp()
        self.tracking(enforce=True)
        self.updateALLimg()
        self.warnEditingWithCca_df('Repeat segmentation')

        txt = f'Done. Segmentation computed in {exec_time:.3f} s'
        self.logger.info('-----------------')
        self.logger.info(txt)
        self.logger.info('=================')
        self.titleLabel.setText(txt, color='g')
        self.checkIfAutoSegm()

    def getDisplayedCellsImg(self):
        if self.overlayButton.isChecked():
            img = self.ol_cells_img
        else:
            img = self.img1.image
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
            '(check progress in terminal/console)', color='w')

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
        if self.gaussWin is None:
            img = skimage.filters.gaussian(img, sigma=1)
        img = skimage.exposure.equalize_adapthist(skimage.img_as_float(img))

        posData.cca_df = model.predictCcaState(img, posData.lab)
        self.store_data()
        self.updateALLimg()

        self.titleLabel.setText('Budding event prediction done.', color='g')

    def next_cb(self):
        t0 = time.perf_counter()
        if self.isSnapshot:
            self.next_pos()
        else:
            self.next_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)
        t1 = time.perf_counter()

    def prev_cb(self):
        if self.isSnapshot:
            self.prev_pos()
        else:
            self.prev_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)

    def zoomToCells(self, enforce=False):
        if not self.enableAutoZoomToCellsAction.isChecked() and not enforce:
            return

        posData = self.data[self.pos_i]
        lab_mask = (posData.lab>0).astype(np.uint8)
        rp = skimage.measure.regionprops(lab_mask)
        if not rp:
            Y, X = posData.lab.shape
            xRange = -0.5, X+0.5
            yRange = -0.5, Y+0.5
        else:
            obj = rp[0]
            min_row, min_col, max_row, max_col = obj.bbox
            xRange = min_col-10, max_col+10
            yRange = max_row+10, min_row-10

        self.ax1.setRange(xRange=xRange, yRange=yRange)

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

    def next_pos(self):
        self.store_data(debug=False)
        if self.pos_i < self.num_pos-1:
            self.pos_i += 1
        else:
            self.logger.info('You reached last position.')
            self.pos_i = 0
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateFilters=True, updateLabelItemColor=False)
        self.zoomToCells()
        self.updateScrollbars()
        self.computeSegm()

    def prev_pos(self):
        self.store_data(debug=False)
        if self.pos_i > 0:
            self.pos_i -= 1
        else:
            self.logger.info('You reached first position.')
            self.pos_i = self.num_pos-1
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateSharp=True, updateBlur=True, updateEntropy=True)
        self.zoomToCells()
        self.updateScrollbars()

    def next_frame(self):
        mode = str(self.modeComboBox.currentText())
        isSegmMode =  mode == 'Segmentation and Tracking'
        posData = self.data[self.pos_i]
        if posData.frame_i < posData.segmSizeT-1:
            if 'lost' in self.titleLabel.text and isSegmMode:
                if not self.doNotWarnLostCells:
                    msg = QMessageBox(self)
                    msg.setWindowTitle('Lost cells!')
                    msg.setIcon(msg.Warning)
                    warn_msg = (
                        'Current frame (compared to previous frame) '
                        'has lost the following cells:\n\n'
                        f'{posData.lost_IDs}\n\n'
                        'Are you sure you want to continue?\n'
                    )
                    msg.setText(warn_msg)
                    msg.addButton(msg.Yes)
                    noButton = QPushButton('No')
                    msg.addButton(noButton, msg.NoRole)
                    cb = QCheckBox('Do not show again')
                    msg.setCheckBox(cb)
                    msg.exec_()
                    self.doNotWarnLostCells = msg.checkBox().isChecked()
                    if msg.clickedButton() == noButton:
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
                    if not posData.cca_df.equals(editCcaWidget.cca_df):
                        self.remove_future_cca_df(0)
                posData.cca_df = editCcaWidget.cca_df
                self.store_cca_df()

            # Store data for current frame
            self.store_data(debug=False)
            # Go to next frame
            posData.frame_i += 1
            self.removeAlldelROIsCurrentFrame()
            proceed_cca, never_visited = self.get_data()
            if not proceed_cca:
                posData.frame_i -= 1
                self.get_data()
                return
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
            self.setNavigateScrollBarMaximum()
            self.updateScrollbars()
            self.computeSegm()
            self.zoomToCells()
        else:
            # Store data for current frame
            self.store_data()
            msg = 'You reached the last segmented frame!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color='w')

    def setNavigateScrollBarMaximum(self):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Segmentation and Tracking':
            if posData.last_tracked_i is not None:
                if posData.frame_i > posData.last_tracked_i:
                    self.navigateScrollBar.setMaximum(posData.frame_i+1)
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
            self.tracking()
            self.updateALLimg(never_visited=never_visited,
                              updateSharp=True, updateBlur=True,
                              updateEntropy=True)
            self.updateScrollbars()
            self.zoomToCells()
        else:
            msg = 'You reached the first frame!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color='w')

    def checkDataIntegrity(self, posData, numPos):
        skipPos = False
        abort = False
        if numPos > 1:
            if posData.SizeT > 1:
                err_msg = (f'{posData.pos_foldername} contains frames over time. '
                           f'Skipping it.')
                self.logger.info(err_msg)
                self.titleLabel.setText(err_msg, color='r')
                skipPos = True
        else:
            if not posData.segmFound and posData.SizeT > 1:
                err_msg = (
                    'Segmentation mask file ("..._segm.npz") not found. '
                    'You could run segmentation module first.'
                )
                self.logger.info(err_msg)
                self.titleLabel.setText(err_msg, color='r')
                skipPos = False
                msg = QMessageBox()
                warn_msg = (
                    f'The folder {posData.pos_foldername} does not contain a '
                    'pre-computed segmentation mask.\n\n'
                    'You can continue with a blank mask or cancel and '
                    'pre-compute the mask with "segm.py" script.\n\n'
                    'Do you want to continue?'
                )
                continueWithBlankSegm = msg.warning(
                   self, 'Segmentation file not found', warn_msg,
                   msg.Yes | msg.Cancel
                )
                if continueWithBlankSegm == msg.Cancel:
                    abort = True
        return skipPos, abort

    def loadSelectedData(self, user_ch_file_paths, user_ch_name):
        data = []
        numPos = len(user_ch_file_paths)
        for f, file_path in enumerate(user_ch_file_paths):
            try:
                posData = load.loadData(file_path, user_ch_name, QParent=self)
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadImgData()
                selectedSegmNpz, cancel = posData.detectMultiSegmNpz()
                if cancel:
                    return False
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
                    selectedSegmNpz=selectedSegmNpz
                )
                if f==0:
                    proceed = posData.askInputMetadata(
                                                ask_SizeT=self.num_pos==1,
                                                ask_TimeIncrement=True,
                                                ask_PhysicalSizes=True,
                                                singlePos=False,
                                                save=True)
                    if not proceed:
                        return False
                    self.SizeT = posData.SizeT
                    self.SizeZ = posData.SizeZ
                    self.TimeIncrement = posData.TimeIncrement
                    self.PhysicalSizeZ = posData.PhysicalSizeZ
                    self.PhysicalSizeY = posData.PhysicalSizeY
                    self.PhysicalSizeX = posData.PhysicalSizeX

                else:
                    posData.SizeT = self.SizeT
                    if self.SizeZ > 1:
                        SizeZ = posData.img_data.shape[-3]
                        posData.SizeZ = SizeZ
                    else:
                        posData.SizeZ = 1
                    posData.TimeIncrement = self.TimeIncrement
                    posData.PhysicalSizeZ = self.PhysicalSizeZ
                    posData.PhysicalSizeY = self.PhysicalSizeY
                    posData.PhysicalSizeX = self.PhysicalSizeX
                    posData.saveMetadata()
                SizeY, SizeX = posData.img_data.shape[-2:]

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
                skipPos, abort = self.checkDataIntegrity(posData, numPos)
            except AttributeError:
                self.logger.info('')
                self.logger.info('====================================')
                traceback.print_exc()
                self.logger.info('====================================')
                self.logger.info('')
                skipPos = False
                abort = True

            if skipPos:
                continue
            elif abort:
                return False

            if posData.SizeT == 1:
                self.isSnapshot = True
            else:
                self.isSnapshot = False

            # Allow single 2D/3D image
            if posData.SizeT < 2:
                posData.img_data = np.array([posData.img_data])
                posData.segm_data = np.array([posData.segm_data])
            img_shape = posData.img_data.shape
            posData.segmSizeT = len(posData.segm_data)
            SizeT = posData.SizeT
            SizeZ = posData.SizeZ
            if f==0:
                self.logger.info(f'Data shape = {img_shape}')
                self.logger.info(f'Number of frames = {SizeT}')
                self.logger.info(f'Number of z-slices per frame = {SizeZ}')
            data.append(posData)

        if not data:
            errTitle = 'All loaded positions contains frames over time!'
            err_msg = (
                f'{errTitle}.\n\n'
                'To load data that contains frames over time you have to select '
                'only ONE position.'
            )
            msg = QMessageBox()
            msg.critical(
                self, errTitle, err_msg, msg.Ok
            )
            self.titleLabel.setText(errTitle, color='r')
            return False

        self.data = data
        self.gui_createGraphicsItems()
        return True

    def gui_addCreatedAxesItems(self):
        allItems = zip(
            self.ax1_ContoursCurves,
            self.ax2_ContoursCurves,
            self.ax1_LabelItemsIDs,
            self.ax2_LabelItemsIDs,
            self.ax1_BudMothLines
        )
        for items_ID in allItems:
            (ax1ContCurve, ax2ContCurve,
            ax1_IDlabel, ax2_IDlabel,
            BudMothLine) = items_ID

            if ax1ContCurve is None:
                continue

            self.ax1.addItem(ax1ContCurve)
            self.ax1.addItem(BudMothLine)
            self.ax1.addItem(ax1_IDlabel)

            self.ax2.addItem(ax2_IDlabel)
            self.ax2.addItem(ax2ContCurve)

    def creatingAxesItemsFinished(self):
        self.progressWin.mainPbar.setMaximum(0)

        self.gui_addCreatedAxesItems()
        self.progressWin.workerFinished = True
        self.progressWin.close()

        posData = self.data[self.pos_i]

        self.init_segmInfo_df()
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

        self.titleLabel.setText(
            'Data successfully loaded. Right/Left arrow to navigate frames',
            color='w')

        self.setFramesSnapshotMode()

        self.enableZstackWidgets(posData.SizeZ > 1)

        self.img1BottomGroupbox.setVisible(True)
        self.updateALLimg(updateLabelItemColor=False)
        self.updateScrollbars()
        self.fontSizeAction.setChecked(True)
        self.openAction.setEnabled(True)
        self.editTextIDsColorAction.setDisabled(False)
        self.imgPropertiesAction.setEnabled(True)
        self.navigateToolBar.setVisible(True)

    def setFramesSnapshotMode(self):
        if self.isSnapshot:
            self.disableTrackingCheckBox.setDisabled(True)
            self.disableTrackingCheckBox.setChecked(True)
            self.repeatTrackingAction.setDisabled(True)
            self.logger.info('Setting GUI mode to "Snapshots"...')
            self.modeComboBox.clear()
            self.modeComboBox.addItems(['Snapshot'])
            self.modeComboBox.setDisabled(True)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxCcaItems)
            self.drawIDsContComboBox.setCurrentIndex(1)
            self.modeToolBar.setVisible(False)
            self.modeComboBox.setCurrentText('Snapshot')
        else:
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
            self.modeComboBox.addItems(['Segmentation and Tracking',
                                        'Cell cycle analysis',
                                        'Viewer'])
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

        questionTxt = (
            f'Some or all loaded {txt} contain empty segmentation masks.\n\n'
            'Do you want to activate automatic segmentation* when visiting '
            f'these {txt}?\n\n'
            '* Automatic segmentation can always be turned ON/OFF from the menu\n'
            '  "Edit --> Segmentation --> Enable automatic segmentation"\n\n'
            f'** Remember that you can automatically segment all {txt} using the\n'
            '    segmentation module.'
        )
        msg = QMessageBox(self)
        doSegmAnswer = msg.question(
            self, 'Automatic segmentation?', questionTxt, msg.Yes | msg.No
        )
        if doSegmAnswer == msg.Yes:
            self.autoSegmAction.setChecked(True)
        else:
            self.autoSegmDoNotAskAgain = True
            self.autoSegmAction.setChecked(False)

    def init_segmInfo_df(self):
        self.t_label.show()
        self.navigateScrollBar.show()
        self.navigateScrollBar.setDisabled(False)

        if self.data[0].SizeZ > 1:
            self.enableZstackWidgets(True)
            self.zSliceScrollBar.setMaximum(self.data[0].SizeZ-1)
            try:
                self.zSliceScrollBar.valueChanged.disconnect()
                self.zProjComboBox.currentTextChanged.disconnect()
                self.zProjComboBox.activated.disconnect()
            except Exception as e:
                pass
            self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
            self.zProjComboBox.activated.connect(self.clearComboBoxFocus)
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

            if posData.SizeT == 1:
                self.t_label.setText('Position n. ')
                self.navigateScrollBar.setMinimum(1)
                self.navigateScrollBar.setMaximum(len(self.data))
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

    def update_z_slice(self, z):
        posData = self.data[self.pos_i]
        idx = (posData.filename, posData.frame_i)
        posData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
        self.updateALLimg(only_ax1=True)

    def update_overlay_z_slice(self, z):
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
            self.updateALLimg(only_ax1=True)

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

    def clearAllItems(self):
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
            N = len(xxS)
            self.smoothAutoContWithSpline(n=int(N*0.15))

        xxS, yyS = self.curvPlotItem.getData()

        self.setBrushID()
        newIDMask = np.zeros(posData.lab.shape, bool)
        rr, cc = skimage.draw.polygon(yyS, xxS)
        newIDMask[rr, cc] = True
        newIDMask[posData.lab!=0] = False
        posData.lab[newIDMask] = posData.brushID

    def addFluoChNameContextMenuAction(self, ch_name):
        posData = self.data[self.pos_i]
        allTexts = [action.text() for action in self.chNamesQActionGroup.actions()]
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

    def initGlobalAttr(self):
        self.setOverlayColors()
        self.cmap = myutils.getFromMatplotlib('viridis')

        self.splineHoverON = False
        self.rulerHoverON = False
        self.isCtrlDown = False
        self.autoContourHoverON = False
        self.navigateScrollBarStartedMoving = True

        self.segment2D_kwargs = None
        self.segmModelName = None
        self.doNotWarnLostCells = False

        self.autoSegmDoNotAskAgain = False

        self.clickedOnBud = False
        self.gaussWin = None
        self.edgeWin = None
        self.entropyWin = None

        self.UserEnforced_DisabledTracking = False
        self.UserEnforced_Tracking = False

        self.ax1BrushHoverID = 0

        # Plots items
        self.data_loaded = True
        self.isMouseDragImg2 = False
        self.isMouseDragImg1 = False
        self.isRightClickDragImg1 = False

        self.cca_df_colnames = [
            'cell_cycle_stage',
            'generation_num',
            'relative_ID',
            'relationship',
            'emerg_frame_i',
            'division_frame_i',
            'is_history_known',
            'corrected_assignment'
        ]
        self.cca_df_dtypes = [
            str, int, int, str, int, int, bool, bool
        ]
        self.cca_df_default_values = [
            'G1', 2, -1, 'mother', -1, -1, False, False
        ]
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

    def initPosAttr(self):
        for p, posData in enumerate(self.data):
            self.pos_i = p
            posData.curvPlotItems = []
            posData.curvAnchorsItems = []
            posData.curvHoverItems = []
            posData.fluoDataChNameActions = []
            posData.manualContrastKey = posData.filename
            posData.isNewID = False

            posData.HDDmaxID = numba_max(posData.segm_data)

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
            posData.UndoRedoStates = [[] for _ in range(posData.segmSizeT)]
            posData.UndoRedoCcaStates = [[] for _ in range(posData.segmSizeT)]

            posData.ol_data_dict = {}
            posData.ol_data = None

            # Colormap
            posData.lut = self.cmap.getLookupTable(0,1, posData.HDDmaxID+10)
            np.random.shuffle(posData.lut)
            # Insert background color
            posData.lut = np.insert(posData.lut, 0, [25, 25, 25], axis=0)

            posData.allData_li = [
                    {
                     'regionprops': None,
                     'labels': None,
                     'acdc_df': None,
                     'delROIs_info': {'rois': [], 'delMasks': [], 'delIDsROI': []},
                     'histoLevels': {}
                     }
                    for i in range(posData.segmSizeT)
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
                delROIshapes = [[] for _ in range(posData.segmSizeT)]
                for i in range(last_tracked_num):
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(enforce=True)
                    # self.load_delROIs_info(delROIshapes, last_tracked_num)
                    posData.binnedIDs = set()
                    posData.ripIDs = set()

                # Ask whether to resume from last frame
                if last_tracked_num>1:
                    msg = QMessageBox()
                    start_from_last_tracked_i = msg.question(
                        self, 'Start from last session?',
                        'The system detected a previous session ended '
                        f'at frame {last_tracked_num}.\n\n'
                        f'Do you want to resume from frame {last_tracked_num}?',
                        msg.Yes | msg.No
                    )
                    if start_from_last_tracked_i == msg.Yes:
                        posData.frame_i = posData.last_tracked_i
                    else:
                        posData.frame_i = 0

        # Back to first position
        self.pos_i = 0
        self.get_data(debug=False)
        self.store_data()
        self.updateALLimg()

        # Link Y and X axis of both plots to scroll zoom and pan together
        self.ax2.vb.setYLink(self.ax1.vb)
        self.ax2.vb.setXLink(self.ax1.vb)

        self.ax2.vb.autoRange()
        self.ax1.vb.autoRange()

    def PosScrollBarAction(self, action):
        if action == QAbstractSlider.SliderSingleStepAdd:
            self.PosScrollBarReleased()
        elif action == QAbstractSlider.SliderSingleStepSub:
            self.PosScrollBarReleased()
        elif action == QAbstractSlider.SliderPageStepAdd:
            self.PosScrollBarReleased()
        elif action == QAbstractSlider.SliderPageStepSub:
            self.PosScrollBarReleased()

    def PosScrollBarMoved(self, pos_n):
        self.pos_i = pos_n-1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateFilters=False)

    def PosScrollBarReleased(self):
        self.pos_i = self.navigateScrollBar.sliderPosition()-1
        proceed_cca, never_visited = self.get_data()
        self.updateFramePosLabel()
        self.updateALLimg(updateFilters=True)
        self.computeSegm()
        self.zoomToCells()

    def framesScrollBarAction(self, action):
        if action == QAbstractSlider.SliderSingleStepAdd:
            self.framesScrollBarReleased()
        elif action == QAbstractSlider.SliderSingleStepSub:
            self.framesScrollBarReleased()
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
        cells_img = self.getImage()
        if self.navigateScrollBarStartedMoving:
            self.clearAllItems()
        self.t_label.setText(
                 f'frame n. {posData.frame_i+1}/{posData.segmSizeT}')
        self.img1.setImage(cells_img)
        self.img2.setImage(posData.lab)
        self.updateLookuptable()
        self.updateFramePosLabel()
        self.navigateScrollBarStartedMoving = False

    def framesScrollBarReleased(self):
        self.navigateScrollBarStartedMoving = True
        posData = self.data[self.pos_i]
        posData.frame_i = self.navigateScrollBar.sliderPosition()-1
        proceed_cca, never_visited = self.get_data()
        self.updateFramePosLabel()
        self.updateALLimg(
            never_visited=never_visited,
            updateFilters=True,
            updateLabelItemColor=False
        )
        self.setNavigateScrollBarMaximum()
        self.computeSegm()
        self.zoomToCells()

    def unstore_data(self):
        posData = self.data[self.pos_i]
        posData.allData_li[posData.frame_i] = {
            'regionprops': [],
            'labels': None,
            'acdc_df': None,
            'delROIs_info': {'rois': [], 'delMasks': [], 'delIDsROI': []},
            'histoLevels': {}
        }


    def store_data(self, pos_i=None, enforce=True, debug=False):
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
        editIDclicked_x = [np.nan]*len(posData.rp)
        editIDclicked_y = [np.nan]*len(posData.rp)
        editIDnewID = [-1]*len(posData.rp)
        editedIDs = [newID for _, _, newID in posData.editID_info]
        for i, obj in enumerate(posData.rp):
            is_cell_dead_li[i] = obj.dead
            is_cell_excluded_li[i] = obj.excluded
            IDs[i] = obj.label
            xx_centroid[i] = int(obj.centroid[1])
            yy_centroid[i] = int(obj.centroid[0])
            if obj.label in editedIDs:
                y, x, new_ID = posData.editID_info[editedIDs.index(obj.label)]
                editIDclicked_x[i] = int(x)
                editIDclicked_y[i] = int(y)
                editIDnewID[i] = new_ID

        try:
            posData.STOREDmaxID = max(IDs)
        except ValueError:
            posData.STOREDmaxID = 0
        posData.allData_li[posData.frame_i]['acdc_df'] = pd.DataFrame(
            {
                        'Cell_ID': IDs,
                        'is_cell_dead': is_cell_dead_li,
                        'is_cell_excluded': is_cell_excluded_li,
                        'x_centroid': xx_centroid,
                        'y_centroid': yy_centroid,
                        'editIDclicked_x': editIDclicked_x,
                        'editIDclicked_y': editIDclicked_y,
                        'editIDnewID': editIDnewID
            }
        ).set_index('Cell_ID')

        self.store_cca_df(pos_i=pos_i)

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
        min_dist = numba_min(dist)
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
        text = (f"""
        <p style="font-size:10pt">
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
        </p>
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
                if posData.frame_i+1 < posData.segmSizeT:
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
                               obj.image.astype(np.uint8),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE
        )
        min_y, min_x, _, _ = obj.bbox
        cont = np.squeeze(contours[0], axis=1)
        if len(contours)>1 and appendMultiContID:
            posData = self.data[self.pos_i]
            posData.multiContIDs.add(obj.label)
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        return cont


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
            posData.lab = posData.segm_data[posData.frame_i].copy()
            posData.rp = skimage.measure.regionprops(posData.lab)
            if debug:
                self.logger.info(never_visited)
            if posData.acdc_df is not None:
                frames = posData.acdc_df.index.get_level_values(0)
                if posData.frame_i in frames:
                    # Since there was already segmentation metadata from
                    # previous closed session add it to current metadata
                    df = posData.acdc_df.loc[posData.frame_i].copy()
                    try:
                        binnedIDs_df = df[df['is_cell_excluded']]
                    except Exception as e:
                        self.logger.info('')
                        self.logger.info('====================================')
                        traceback.print_exc()
                        self.logger.info('====================================')
                        self.logger.info('')
                        raise
                    binnedIDs = set(binnedIDs_df.index).union(posData.binnedIDs)
                    posData.binnedIDs = binnedIDs
                    ripIDs_df = df[df['is_cell_dead']]
                    ripIDs = set(ripIDs_df.index).union(posData.ripIDs)
                    posData.ripIDs = ripIDs
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
            posData.lab = posData.allData_li[posData.frame_i]['labels'].copy()
            posData.rp = skimage.measure.regionprops(posData.lab)
            df = posData.allData_li[posData.frame_i]['acdc_df']
            binnedIDs_df = df[df['is_cell_excluded']]
            posData.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df['is_cell_dead']]
            posData.ripIDs = set(ripIDs_df.index)
            editIDclicked_x = df['editIDclicked_x'].to_list()
            editIDclicked_y = df['editIDclicked_y'].to_list()
            editIDnewID = df['editIDnewID'].to_list()
            _zip = zip(editIDclicked_y, editIDclicked_x, editIDnewID)
            posData.editID_info = [
                (int(y),int(x),newID) for y,x,newID in _zip if newID!=-1]
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
                Y, X = posData.lab.shape
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
            msg = QMessageBox(self)
            msg.setIcon(msg.Warning)
            msg.setWindowTitle('Go to last visited frame?')
            msg.setText(
                f'The last visited frame in "Segmentation and Tracking mode" '
                f'is frame {last_tracked_i+1}.\n\n'
                f'We recommend to resume from that frame.\n\n'
                'How do you want to proceed?'
            )
            goToButton = QPushButton(
                f'Resume from frame {last_tracked_i+1} (RECOMMENDED)'
            )
            stayButton = QPushButton(
                f'Stay on current frame {posData.frame_i+1}'
            )
            msg.addButton(goToButton, msg.YesRole)
            msg.addButton(stayButton, msg.NoRole)
            msg.exec_()
            if msg.clickedButton() == goToButton:
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
                self.titleLabel.setText(msg, color='w')
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            else:
                msg = 'Cell cycle analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color='w')
                self.modeComboBox.setCurrentText(currentMode)
                proceed = False
                return
        elif posData.frame_i < last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = QMessageBox()
            goTo_last_annotated_frame_i = msg.question(
                self, 'Go to last annotated frame?',
                f'The last annotated frame is frame {last_cca_frame_i+1}.\n'
                'Do you want to restart cell cycle analysis from frame '
                f'{last_cca_frame_i+1}?',
                msg.Yes | msg.No | msg.Cancel
            )
            if goTo_last_annotated_frame_i == msg.Yes:
                msg = 'Looking good!'
                self.titleLabel.setText(msg, color='w')
                self.last_cca_frame_i = last_cca_frame_i
                posData.frame_i = last_cca_frame_i
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            elif goTo_last_annotated_frame_i == msg.Cancel:
                msg = 'Cell cycle analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color='w')
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
            self.titleLabel.setText(msg, color='w')
        else:
            self.get_cca_df()
        return proceed

    def remove_future_cca_df(self, from_frame_i):
        posData = self.data[self.pos_i]
        self.last_cca_frame_i = posData.frame_i
        self.setNavigateScrollBarMaximum()
        for i in range(from_frame_i, posData.segmSizeT):
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

    def store_cca_df(self, pos_i=None, frame_i=None, cca_df=None):
        pos_i = self.pos_i if pos_i is None else pos_i
        posData = self.data[pos_i]
        i = posData.frame_i if frame_i is None else frame_i
        if cca_df is None:
            cca_df = posData.cca_df
            if self.ccaTableWin is not None:
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

    def ax1_setTextID(self, obj, how, updateColor=False):
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
                for i in range(posData.frame_i+1, posData.segmSizeT):
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
                if updateColor:
                    c = self.getOptimalLabelItemColor(LabelItemID, c)
                    self.ax1_G1cellColor = c
                bold = False
            elif mothCell_S:
                color = self.ax1_S_oldCellColor
                if updateColor:
                    c = self.getOptimalLabelItemColor(LabelItemID, c)
                    self.ax1_S_oldCellColor = c
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

        try:
            LabelItemID.setText(txt, color=color, bold=bold, size=self.fontSize)
        except UnboundLocalError:
            pass

        # Center LabelItem at centroid
        y, x = obj.centroid
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

    def ax2_setTextID(self, obj):
        posData = self.data[self.pos_i]
        # Draw ID label on ax1 image
        LabelItemID = self.ax2_LabelItemsIDs[obj.label-1]
        ID = obj.label
        df = posData.cca_df
        txt = f'{ID}'
        if ID in posData.new_IDs:
            color = 'r'
            bold = True
        else:
            color = (230, 0, 0, 255*0.5)
            bold = False

        LabelItemID.setText(txt, color=color, bold=bold, size=self.fontSize)

        # Center LabelItem at centroid
        y, x = obj.centroid
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

        # Draw LabelItems for IDs on ax2
        y, x = obj.centroid
        idx = obj.label-1
        t0 = time.time()
        self.ax2_setTextID(obj)

        if posData.cca_df is not None and self.isSnapshot:
            if obj.label not in posData.cca_df.index:
                self.store_data()
                self.addIDBaseCca_df(posData, obj.label)
                self.store_cca_df()

        # Draw LabelItems for IDs on ax1 if requested
        if IDs_and_cont or onlyIDs or only_ccaInfo or ccaInfo_and_cont:
            # Draw LabelItems for IDs on ax2
            t0 = time.time()
            self.ax1_setTextID(obj, how, updateColor=updateColor)

        t1 = time.time()
        self.drawingLabelsTimes.append(t1-t0)

        # Draw line connecting mother and buds
        drawLines = only_ccaInfo or ccaInfo_and_cont or onlyMothBudLines
        if drawLines and posData.cca_df is not None:
            ID = obj.label
            BudMothLine = self.ax1_BudMothLines[ID-1]
            cca_df_ID = posData.cca_df.loc[ID]
            ccs_ID = cca_df_ID['cell_cycle_stage']
            relationship = cca_df_ID['relationship']
            if ccs_ID == 'S' and relationship=='bud':
                emerg_frame_i = cca_df_ID['emerg_frame_i']
                if emerg_frame_i == posData.frame_i:
                    pen = self.NewBudMoth_Pen
                else:
                    pen = self.OldBudMoth_Pen
                relative_ID = cca_df_ID['relative_ID']
                if relative_ID in posData.IDs:
                    relative_rp_idx = posData.IDs.index(relative_ID)
                    relative_ID_obj = posData.rp[relative_rp_idx]
                    y1, x1 = obj.centroid
                    y2, x2 = relative_ID_obj.centroid
                    BudMothLine.setData([x1, x2], [y1, y2], pen=pen)
            else:
                BudMothLine.setData([], [])

        if not drawContours:
            return

        # Draw contours on ax1 if requested
        if IDs_and_cont or onlyCont or ccaInfo_and_cont:
            ID = obj.label
            t0 = time.time()
            cont = self.getObjContours(obj)
            t1 = time.time()
            computingContoursTime = t1-t0
            self.computingContoursTimes.append(computingContoursTime)

            t0 = time.time()
            curveID = self.ax1_ContoursCurves[idx]
            pen = self.newIDs_cpen if ID in posData.new_IDs else self.oldIDs_cpen
            curveID.setData(cont[:,0], cont[:,1], pen=pen)
            t1 = time.time()
            drawingContoursTimes = t1-t0
            self.drawingContoursTimes.append(drawingContoursTimes)


    def update_rp(self, draw=True, debug=False):
        posData = self.data[self.pos_i]
        # Update rp for current posData.lab (e.g. after any change)
        posData.rp = skimage.measure.regionprops(posData.lab)
        posData.IDs = [obj.label for obj in posData.rp]
        self.update_rp_metadata()

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
        self.checkIDsMultiContour()
        self.highlightmultiBudMoth()

    def highlightmultiBudMoth(self):
        posData = self.data[self.pos_i]
        for ID in posData.multiBud_mothIDs:
            LabelItemID = self.ax1_LabelItemsIDs[ID-1]
            txt = LabelItemID
            LabelItemID.setText(f'{txt} !!', color=self.lostIDs_qMcolor)

    def checkIDsMultiContour(self):
        posData = self.data[self.pos_i]
        txt = self.titleLabel.text
        if 'Looking' not in txt or 'Data' not in txt:
            warn_txt = self.titleLabel.text
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>'
            )
        else:
            htmlTxt = f'<font color="white">{self.titleLabel.text}</font>'
        if posData.multiContIDs:
            warn_txt = f'IDs with multiple contours: {posData.multiContIDs}'
            color = 'red'
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>, {htmlTxt}'
            )
            posData.multiContIDs = set()
        self.titleLabel.setText(htmlTxt)

    def updateLookuptable(self, lenNewLut=None):
        posData = self.data[self.pos_i]
        if lenNewLut is None:
            try:
                lenNewLut = max(posData.IDs)+2
            except ValueError:
                # Empty segmentation mask
                lenNewLut = 1
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

        try:
            lut = posData.lut[:lenNewLut].copy()
            for ID in posData.binnedIDs:
                lut[ID] = lut[ID]*0.2
            for ID in posData.ripIDs:
                lut[ID] = lut[ID]*0.2
        except Exception as e:
            self.logger.info('WARNING: Tracking is WRONG.')
            pass
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
                    y, x = obj.centroid
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
                    y, x = obj.centroid
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

    def overlay_cb(self, checked):
        self.UserNormAction, _, _ = self.getCheckNormAction()
        posData = self.data[self.pos_i]
        if checked:
            prompt = True
            if posData.ol_data is not None:
                prompt = False
            # Check if there is already loaded data
            elif posData.fluo_data_dict and posData.ol_data is None:
                ch_names = list(posData.loadedFluoChannels)
                if len(ch_names)>1:
                    selectFluo = apps.QDialogListbox(
                        'Select channel',
                        'Select channel names to load:\n',
                        ch_names, multiSelection=True, parent=self
                    )
                    selectFluo.exec_()
                    ol_channels = selectFluo.selectedItemsText
                    if selectFluo.cancel or not ol_channels:
                        prompt = True
                    else:
                        prompt = False
                else:
                    prompt = False
                    ol_channels = ch_names

                for posData in self.data:
                    ol_data = {}
                    ol_colors = {}
                    for i, ol_ch in enumerate(ol_channels):
                        ol_path, filename = self.getPathFromChName(
                                                            ol_ch, posData)
                        if ol_path is None:
                            self.criticalFluoChannelNotFound(ol_ch, posData)
                            self.app.restoreOverrideCursor()
                            return
                        ol_data[filename] = posData.ol_data_dict[filename].copy()
                        ol_colors[filename] = self.overlayRGBs[i]
                        self.addFluoChNameContextMenuAction(ol_ch)
                    posData.manualContrastKey = filename
                    posData.ol_data = ol_data
                    posData.ol_colors = ol_colors

            if prompt:
                # extensions = self.getFileExtensions(posData.images_path)
                # ol_paths = QFileDialog.getOpenFileNames(
                #     self, 'Select one or multiple fluorescent images',
                #     posData.images_path, extensions
                # )
                ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
                selectFluo = apps.QDialogListbox(
                    'Select channel',
                    'Select channel names to load:\n',
                    ch_names, multiSelection=True, parent=self
                )
                selectFluo.exec_()
                if selectFluo.cancel:
                    self.overlayButton.setChecked(False)
                    return
                ol_channels = selectFluo.selectedItemsText
                for posData in self.data:
                    ol_data = {}
                    ol_colors = {}
                    for i, ol_ch in enumerate(ol_channels):
                        ol_path, filename = self.getPathFromChName(ol_ch,
                                                                   posData)
                        if ol_path is None:
                            self.criticalFluoChannelNotFound(ol_ch, posData)
                            self.app.restoreOverrideCursor()
                            return
                        fluo_data, bkgrData = self.load_fluo_data(ol_path)
                        if fluo_data is None:
                            self.app.restoreOverrideCursor()
                            return

                        # Allow single 2D/3D image
                        if posData.SizeT < 2:
                            fluo_data = np.array([fluo_data])

                        posData.fluo_data_dict[filename] = fluo_data
                        posData.fluo_bkgrData_dict[filename] = bkgrData
                        posData.ol_data_dict[filename] = fluo_data
                        ol_data[filename] = fluo_data.copy()
                        ol_colors[filename] = self.overlayRGBs[i]
                        posData.ol_colors = ol_colors
                        if i!=0:
                            continue
                        self.addFluoChNameContextMenuAction(ol_ch)
                    posData.manualContrastKey = filename
                    posData.ol_data = ol_data

                self.app.restoreOverrideCursor()
                self.overlayButton.setStyleSheet('background-color: #A7FAC7')

            self.normalizeRescale0to1Action.setChecked(True)
            self.hist.imageItem = lambda: None
            self.updateHistogramItem(self.img1)

            rgb = self.df_settings.at['overlayColor', 'value']
            rgb = [int(v) for v in rgb.split('-')]
            self.overlayColorButton.setColor(rgb)

            self.updateALLimg(only_ax1=True)

            self.enableOverlayWidgets(True)
        else:
            self.UserNormAction.setChecked(True)
            self.create_chNamesQActionGroup(self.user_ch_name)
            posData.fluoDataChNameActions = []
            self.updateHistogramItem(self.img1)
            self.updateALLimg(only_ax1=True)
            self.enableOverlayWidgets(False)

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
        msg = QMessageBox()
        warn_cca = msg.critical(
            self, 'Requested channel data not found!',
            f'The folder {posData.rel_path} does not contain either one of the '
            'following files:\n\n'
            f'{posData.basename}_{fluo_ch}.tif\n'
            f'{posData.basename}_{fluo_ch}_aligned.npz\n\n'
            'Data loading aborted.',
            msg.Ok
        )

    def histLUT_cb(self, LUTitem):
        # Callback function for the histogram sliders moved by the user
        # Store the histogram levels that the user is manually changing
        # i.e. moving the gradient slider ticks up and down
        # Store them for all frames

        posData = self.data[self.pos_i]
        isOverlayON = self.overlayButton.isChecked()
        min = self.hist.gradient.listTicks()[0][1]
        max = self.hist.gradient.listTicks()[1][1]
        if isOverlayON:
            for i in range(0, posData.segmSizeT):
                histoLevels = posData.allData_li[i]['histoLevels']
                histoLevels[posData.manualContrastKey] = (min, max)
            if posData.ol_data is not None:
                self.getOverlayImg(setImg=True)
        else:
            cellsKey = f'{self.user_ch_name}_overlayOFF'
            for i in range(0, posData.segmSizeT):
                histoLevels = posData.allData_li[i]['histoLevels']
                histoLevels[cellsKey] = (min, max)
            img = self.getImage()
            if self.hist.gradient.isLookupTrivial():
                self.img1.setLookupTable(None)
            else:
                self.img1.setLookupTable(self.hist.getLookupTable(img=img))

    def histLUTfinished_cb(self):
        if not self.overlayButton.isChecked():
            cellsKey = f'{self.user_ch_name}_overlayOFF'
            img = self.getImage()
            img = self.adjustBrightness(img, cellsKey)
            self.updateALLimg(image=img, only_ax1=True, updateFilters=True,
                              updateHistoLevels=False)

    def gradientContextMenuClicked(self, b=None):
        act = self.sender()
        self.hist.gradient.loadPreset(act.name)
        if act.name == 'grey':
            try:
                self.hist.sigLookupTableChanged.disconnect()
            except TypeError:
                pass
            self.hist.setImageItem(self.img1)
            self.hist.imageItem = lambda: None
            self.hist.sigLookupTableChanged.connect(self.histLUT_cb)
        else:
            try:
                self.hist.sigLookupTableChanged.disconnect()
            except TypeError:
                pass
            self.hist.setImageItem(self.img1)


    def adjustBrightness(self, img, key,
                         func=skimage.exposure.rescale_intensity):
        """
        Adjust contrast/brightness of the image selected in the histogram
        context menu using stored levels.
        The levels are stored in histLUT_cb function which is called when
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
                imgRange = numba_max(img)-numba_min(img)
                min = numba_min(img) + imgRange*minPerc
                max = numba_min(img) + imgRange*maxPerc
                out_range = (min, max)
                rescaled_img = func(
                    rescaled_img, in_range='image', out_range=out_range
                )
        return rescaled_img

    def getOlImg(self, key, normalizeIntens=True):
        posData = self.data[self.pos_i]
        img = posData.ol_data[key][posData.frame_i]
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

    def getOverlayImg(self, fluoData=None, setImg=True):
        posData = self.data[self.pos_i]
        keys = list(posData.ol_data.keys())

        # Cells channel (e.g. phase_contrast)
        cells_img = self.getImage(invert=False)

        img = self.adjustBrightness(cells_img, posData.filename)
        self.ol_cells_img = img
        gray_img_rgb = gray2rgb(img)

        # First fluo channel
        ol_img = self.getOlImg(keys[0])
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
        ol_norm_img = ol_img/numba_max(ol_img)
        ol_img_rgb = gray2rgb(ol_norm_img)*ol_RGB_val
        overlay = (gray_img_rgb*(1.0 - ol_alpha)+ol_img_rgb*ol_alpha)
        overlay = overlay/numba_max(overlay)
        overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
        return overlay

    def toggleOverlayColorButton(self, checked=True):
        self.mousePressColorButton(None)

    def toggleTextIDsColorButton(self, checked=True):
        self.textIDsColorButton.selectColor()

    def updateTextIDsColors(self, button):
        r, g, b = np.array(self.textIDsColorButton.color().getRgb()[:3])
        self.gui_setLabelsColors(r,g,b, custom=True)
        self.updateALLimg()

    def saveTextIDsColors(self, button):
        self.df_settings.at['textIDsColor', 'value'] = self.ax1_oldIDcolor
        self.df_settings.to_csv(self.settings_csv_path)

    def updateOlColors(self, button):
        rgb = self.overlayColorButton.color().getRgb()[:3]
        for posData in self.data:
            posData.ol_colors[self._key] = rgb
        self.df_settings.at['overlayColor',
                            'value'] = '-'.join([str(v) for v in rgb])
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateOverlay(button)

    def updateOverlay(self, button):
        self.getOverlayImg(setImg=True)


    def getImage(self, frame_i=None, invert=True, normalizeIntens=True):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        if posData.SizeZ > 1:
            idx = (posData.filename, frame_i)
            z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            zProjHow = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
            img = posData.img_data[frame_i]
            if zProjHow == 'single z-slice':
                reconnect = False
                try:
                    self.zSliceScrollBar.valueChanged.disconnect()
                    reconnect = True
                except TypeError:
                    pass
                self.zSliceScrollBar.setSliderPosition(z)
                if reconnect:
                    self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
                self.z_label.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')
                cells_img = img[z].copy()
            elif zProjHow == 'max z-projection':
                cells_img = img.max(axis=0).copy()
            elif zProjHow == 'mean z-projection':
                cells_img = img.mean(axis=0).copy()
            elif zProjHow == 'median z-proj.':
                cells_img = np.median(img, axis=0).copy()
        else:
            cells_img = posData.img_data[frame_i].copy()
        if normalizeIntens:
            cells_img = self.normalizeIntensities(cells_img)
        if self.invertBwAction.isChecked() and invert:
            cells_img = -cells_img+numba_max(cells_img)
        return cells_img

    def setImageImg2(self):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Segmentation and Tracking' or self.isSnapshot:
            self.addExistingDelROIs()
            allDelIDs, DelROIlab = self.getDelROIlab()
        else:
            DelROIlab = posData.lab
        self.img2.setImage(DelROIlab)
        self.updateLookuptable()

    def setTempImg1Brush(self, mask, alpha=0.3):
        posData = self.data[self.pos_i]
        brushOverlay = self.imgRGB.copy()
        overlay = self.imgRGB[mask]*(1.0-alpha) + self.brushColor*alpha
        brushOverlay[mask] = overlay
        brushOverlay = (brushOverlay*255).astype(np.uint8)
        self.img1.setImage(brushOverlay)
        return overlay

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


    def warnEditingWithCca_df(self, editTxt):
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
        msg = QMessageBox()
        msg.setIcon(msg.Warning)
        msg.setWindowTitle('Edited frame!')
        msg.setText(
            'You modified a frame that has cell cycle annotations.\n\n'
            f'The change "{editTxt}" most likely makes the annotations wrong.\n\n'
            'If you really want to apply this change we reccommend to remove\n'
            'ALL cell cycle annotations from current frame to the end.\n\n'
            'What should I do?'
        )
        yes = QPushButton('Remove annotations from future frames (RECOMMENDED)')
        msg.addButton(yes, msg.YesRole)
        msg.addButton(QPushButton('Do not remove annotations'), msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == yes:
            self.store_data()
            posData.frame_i -= 1
            self.get_data()
            self.remove_future_cca_df(posData.frame_i)
            self.next_frame()

    def addExistingDelROIs(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        for roi in delROIs_info['rois']:
            if roi in self.ax2.items:
                continue
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
        ax1_IDlabel = pg.LabelItem()
        self.ax1.addItem(ax1_IDlabel)

        # LabelItems on ax2
        ax2_IDlabel = pg.LabelItem()
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

    def updateHistogramItem(self, imageItem):
        """
        Function called every time the image changes (updateALLimg).
        Perform the following:

        1. Set the gradient slider tick positions
        2. Set the region max and min levels
        3. Plot the histogram
        """
        try:
            self.hist.sigLookupTableChanged.disconnect()
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

        minTick = self.hist.gradient.getTick(0)
        maxTick = self.hist.gradient.getTick(1)
        self.hist.gradient.setTickValue(minTick, min)
        self.hist.gradient.setTickValue(maxTick, max)
        self.hist.setLevels(min=numba_min(imageItem.image),
                            max=numba_max(imageItem.image))
        h = imageItem.getHistogram()
        self.hist.plot.setData(*h)
        if connect:
            self.hist.sigLookupTableChanged.connect(self.histLUT_cb)

    def updateFramePosLabel(self):
        if self.isSnapshot:
            posData = self.data[self.pos_i]
            self.t_label.setText(
                     f'Pos. n. {self.pos_i+1}/{self.num_pos} '
                     f'({posData.pos_foldername})')
        else:
            posData = self.data[0]
            self.t_label.setText(
                     f'frame n. {posData.frame_i+1}/{posData.segmSizeT}')


    def updateFilters(self, updateBlur=False, updateSharp=False,
                            updateEntropy=False, updateFilters=False):
        if self.gaussWin is not None and (updateBlur or updateFilters):
            self.gaussWin.apply()

        if self.edgeWin is not None and (updateSharp or updateFilters):
            self.edgeWin.apply()

        if self.entropyWin is not None and (updateEntropy or updateFilters):
            self.entropyWin.apply()

    def addItemsAllIDs(self, IDs):
        for ID in IDs:
            self.addNewItems(ID)

    def highlightSearchedID(self, ID):
        contours = zip(
            self.ax1_ContoursCurves,
            self.ax2_ContoursCurves
        )
        for ax1ContCurve, ax2ContCurve in contours:
            if ax1ContCurve is None:
                continue
            if ax1ContCurve.getData()[0] is not None:
                ax1ContCurve.setData([], [])
            if ax2ContCurve.getData()[0] is not None:
                ax2ContCurve.setData([], [])

        posData = self.data[self.pos_i]

        # Red thick contour of searched ID
        objIdx = posData.IDs.index(ID)
        obj = posData.rp[objIdx]
        cont = self.getObjContours(obj)
        pen = self.newIDs_cpen
        curveID = self.ax1_ContoursCurves[ID-1]
        curveID.setData(cont[:,0], cont[:,1], pen=pen)

        # Label ID
        LabelItemID = self.ax1_LabelItemsIDs[ID-1]
        txt = f'{ID}'
        LabelItemID.setText(txt, color='r', bold=True, size=self.fontSize)
        y, x = obj.centroid
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

        LabelItemID = self.ax2_LabelItemsIDs[ID-1]
        LabelItemID.setText(txt, color='r', bold=True, size=self.fontSize)
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

        # Gray out all IDs excpet searched one
        lut = posData.lut.copy()[:max(posData.IDs)+1]
        lut[:ID] = lut[:ID]*0.2
        lut[ID+1:] = lut[ID+1:]*0.2
        self.img2.setLookupTable(lut)

    @exception_handler
    def updateALLimg(
            self, image=None, never_visited=True,
            only_ax1=False, updateBlur=False,
            updateSharp=False, updateEntropy=False,
            updateHistoLevels=True, updateFilters=False,
            updateLabelItemColor=False, debug=False
        ):
        posData = self.data[self.pos_i]

        if image is None:
            if self.overlayButton.isChecked():
                img = self.getOverlayImg(setImg=False)
            else:
                img = self.getImage()
                cellsKey = f'{self.user_ch_name}_overlayOFF'
                img = self.adjustBrightness(img, cellsKey)
        else:
            img = image

        self.img1.setImage(img)
        self.updateFilters(updateBlur, updateSharp, updateEntropy, updateFilters)

        if updateHistoLevels:
            self.updateHistogramItem(self.img1)

        if self.slideshowWin is not None:
            self.slideshowWin.framne_i = posData.frame_i
            self.slideshowWin.update_img()

        if only_ax1:
            return

        self.addItemsAllIDs(posData.IDs)
        self.clearAllItems()

        self.setImageImg2()
        self.update_rp()

        self.checkIDs_LostNew()

        self.computingContoursTimes = []
        self.drawingLabelsTimes = []
        self.drawingContoursTimes = []
        # Annotate ID and draw contours
        for i, obj in enumerate(posData.rp):
            updateColor=True if updateLabelItemColor and i==0 else False
            self.drawID_and_Contour(obj, updateColor=updateColor)


        # self.logger.info('------------------------------------')
        # self.logger.info(f'Drawing labels = {np.sum(self.drawingLabelsTimes):.3f} s')
        # self.logger.info(f'Computing contours = {np.sum(self.computingContoursTimes):.3f} s')
        # self.logger.info(f'Drawing contours = {np.sum(self.drawingContoursTimes):.3f} s')

        # Update annotated IDs (e.g. dead cells)
        self.update_rp_metadata(draw=True)

        self.highlightLostNew()
        self.checkIDsMultiContour()

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(posData.cca_df)

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

        for ax2ContCurve in self.ax2_ContoursCurves:
            if ax2ContCurve is None:
                continue

            if ax2ContCurve.getData()[0] is not None:
                ax2ContCurve.setData([], [])

        if posData.frame_i == 0:
            return

        if IDs_and_cont or onlyCont or ccaInfo_and_cont:
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
        y, x = obj.centroid
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
        ID = obj.label
        if ID in posData.lost_IDs:
            ContCurve = self.ax1_ContoursCurves[ID-1]
            if ContCurve is None:
                self.addNewItems(ID)
            ContCurve = self.ax1_ContoursCurves[ID-1]

            if IDs_and_cont or onlyCont or ccaInfo_and_cont or forceContour:
                cont = self.getObjContours(obj)
                ContCurve.setData(
                    cont[:,0], cont[:,1], pen=self.lostIDs_cpen
                )
            LabelItemID = self.ax1_LabelItemsIDs[ID-1]
            txt = f'{obj.label}?'
            LabelItemID.setText(txt, color=self.lostIDs_qMcolor)
            # Center LabelItem at centroid
            y, x = obj.centroid
            w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
            LabelItemID.setPos(x-w/2, y-h/2)


    def checkIDs_LostNew(self):
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            posData.lost_IDs = []
            posData.new_IDs = []
            posData.old_IDs = []
            posData.IDs = [obj.label for obj in posData.rp]
            posData.multiContIDs = set()
            self.titleLabel.setText('Looking good!', color='w')
            return

        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        if prev_rp is None:
            return

        prev_IDs = [obj.label for obj in prev_rp]
        curr_IDs = [obj.label for obj in posData.rp]
        lost_IDs = [ID for ID in prev_IDs if ID not in curr_IDs]
        new_IDs = [ID for ID in curr_IDs if ID not in prev_IDs]
        posData.lost_IDs = lost_IDs
        posData.new_IDs = new_IDs
        posData.old_IDs = prev_IDs
        posData.IDs = curr_IDs
        warn_txt = ''
        htmlTxt = ''
        if lost_IDs:
            lost_IDs_format = lost_IDs.copy()
            if len(lost_IDs) + len(new_IDs) > 20 and len(lost_IDs)>10:
                del lost_IDs_format[5:-5]
                lost_IDs_format.insert(5, "...")
                lost_IDs_format = f"[{', '.join(map(str, lost_IDs_format))}]"
            warn_txt = f'IDs lost in current frame: {lost_IDs_format}'
            color = 'red'
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>'
            )
        if posData.multiContIDs:
            warn_txt = f'IDs with multiple contours: {posData.multiContIDs}'
            color = 'red'
            htmlTxt = (
                f'{htmlTxt}, <font color="red">{warn_txt}</font>'
            )
        if new_IDs:
            new_IDs_format = new_IDs.copy()
            if len(lost_IDs) + len(new_IDs) > 20 and len(new_IDs)>10:
                del new_IDs_format[5:-5]
                new_IDs_format.insert(5, "...")
                new_IDs_format = f"[{', '.join(map(str, new_IDs_format))}]"
            warn_txt = f'New IDs in current frame: {new_IDs_format}'
            color = 'r'
            htmlTxt = (
                f'{htmlTxt}, <font color="green">{warn_txt}</font>'
            )
        if not warn_txt:
            warn_txt = 'Looking good!'
            color = 'w'
            htmlTxt = (
                f'<font color="white">{warn_txt}</font>'
            )
        self.titleLabel.setText(htmlTxt)
        posData.multiContIDs = set()

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
                    lab_obj += numba_max(lab)
                else:
                    lab_obj += maxID
                lab[obj.slice][obj.image] = lab_obj[obj.image]
                setRp = True
        return setRp

    def checkTrackingEnabled(self):
        posData = self.data[self.pos_i]
        posData.last_tracked_i = self.navigateScrollBar.maximum()-1
        if posData.frame_i <= posData.last_tracked_i:
            self.disableTrackingCheckBox.setChecked(True)
        else:
            self.disableTrackingCheckBox.setChecked(False)


    def tracking(
            self, onlyIDs=[], enforce=False, DoManualEdit=True,
            storeUndo=False, prev_lab=None, prev_rp=None,
            return_lab=False
        ):
        try:
            posData = self.data[self.pos_i]
            mode = str(self.modeComboBox.currentText())
            skipTracking = (
                posData.frame_i == 0 or mode.find('Tracking') == -1
                or self.isSnapshot
            )
            if skipTracking:
                self.checkIDs_LostNew()
                return

            # Disable tracking for already visited frames
            self.checkTrackingEnabled()

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
            elif self.disableTrackingCheckBox.isChecked():
                # User did not choose what to do --> tracking disabled for
                # visited frames and enabled for never visited frames
                do_tracking = False
            else:
                do_tracking = True

            if not do_tracking:
                self.disableTrackingCheckBox.setChecked(True)
                # self.logger.info('-------------')
                # self.logger.info(f'Frame {posData.frame_i+1} NOT tracked')
                # self.logger.info('-------------')
                self.checkIDs_LostNew()
                return

            """Tracking starts here"""
            self.disableTrackingCheckBox.setChecked(False)

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
            IDs_prev = []
            IDs_curr_untracked = posData.IDs
            IoA_matrix = np.zeros((len(posData.rp), len(prev_rp)))

            # For each ID in previous frame get IoA with all current IDs
            # Rows: IDs in current frame, columns: IDs in previous frame
            for j, obj_prev in enumerate(prev_rp):
                ID_prev = obj_prev.label
                A_IDprev = obj_prev.area
                IDs_prev.append(ID_prev)
                mask_ID_prev = prev_lab==ID_prev
                intersect_IDs, intersects = np.unique(
                    posData.lab[mask_ID_prev], return_counts=True
                )
                for intersect_ID, I in zip(intersect_IDs, intersects):
                    if intersect_ID != 0:
                        i = IDs_curr_untracked.index(intersect_ID)
                        IoA = I/A_IDprev
                        IoA_matrix[i, j] = IoA

            # Determine max IoA between IDs and assign tracked ID if IoA > 0.4
            max_IoA_col_idx = IoA_matrix.argmax(axis=1)
            unique_col_idx, counts = np.unique(
                max_IoA_col_idx, return_counts=True
            )
            counts_dict = dict(zip(unique_col_idx, counts))
            tracked_IDs = []
            old_IDs = []
            for i, j in enumerate(max_IoA_col_idx):
                max_IoU = IoA_matrix[i,j]
                count = counts_dict[j]
                if max_IoU > 0.4:
                    tracked_ID = IDs_prev[j]
                    if count == 1:
                        old_ID = IDs_curr_untracked[i]
                    elif count > 1:
                        old_ID_idx = IoA_matrix[:,j].argmax()
                        old_ID = IDs_curr_untracked[old_ID_idx]
                    tracked_IDs.append(tracked_ID)
                    old_IDs.append(old_ID)

            # Compute new IDs that have not been tracked
            new_untracked_IDs = [
                ID for ID in IDs_curr_untracked if ID not in old_IDs
            ]
            tracked_lab = posData.lab
            new_tracked_IDs_2 = []
            if new_untracked_IDs:
                # Compute starting unique ID
                self.setBrushID(useCurrentLab=False)
                uniqueID = posData.brushID

                # Relabel new untracked IDs sequentially starting
                # from uniqueID to make sure they are unique
                new_tracked_IDs = [
                    uniqueID+i for i in range(len(new_untracked_IDs))
                ]
                core.lab_replace_values(
                    tracked_lab, posData.rp, new_untracked_IDs, new_tracked_IDs
                )
            if tracked_IDs:
                # Relabel old IDs with respective tracked IDs
                core.lab_replace_values(
                    tracked_lab, posData.rp, old_IDs, tracked_IDs
                )

            if DoManualEdit:
                # Correct tracking with manually changed IDs
                rp = skimage.measure.regionprops(tracked_lab)
                IDs = [obj.label for obj in rp]
                self.manuallyEditTracking(tracked_lab, IDs)
        except ValueError:
            tracked_lab = posData.lab


        # Update labels, regionprops and determine new and lost IDs
        posData.lab = tracked_lab
        self.update_rp()
        self.checkIDs_LostNew()

    def manuallyEditTracking(self, tracked_lab, allIDs):
        posData = self.data[self.pos_i]
        # Correct tracking with manually changed IDs
        for y, x, new_ID in posData.editID_info:
            old_ID = tracked_lab[y, x]
            if new_ID in allIDs:
                tempID = numba_max(tracked_lab) + 1
                tracked_lab[tracked_lab == old_ID] = tempID
                tracked_lab[tracked_lab == new_ID] = old_ID
                tracked_lab[tracked_lab == tempID] = new_ID
            else:
                tracked_lab[tracked_lab == old_ID] = new_ID

    def undo_changes_future_frames(self):
        posData = self.data[self.pos_i]
        posData.last_tracked_i = posData.frame_i
        for i in range(posData.frame_i+1, posData.segmSizeT):
            if posData.allData_li[i]['labels'] is None:
                break

            posData.allData_li[i] = {
                 'regionprops': [],
                 'labels': None,
                 'acdc_df': None,
                 'delROIs_info': {'rois': [], 'delMasks': [], 'delIDsROI': []},
                 'histoLevels': {}
            }
        self.setNavigateScrollBarMaximum()

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
        self.userChNameAction.setChecked(True)
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        self.chNamesQActionGroup.triggered.connect(self.setManualContrastKey)

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
            img = self.getImage()
            img = self.adjustBrightness(img, key)
            self.updateALLimg(image=img, only_ax1=True, updateFilters=True,
                              updateHistoLevels=True)
        else:
            self.updateHistogramItem(self.img1)
            self.getOverlayImg(setImg=True)

    def restoreDefaultColors(self):
        try:
            color = self.defaultToolBarButtonColor
            self.overlayButton.setStyleSheet(f'background-color: {color}')
        except AttributeError:
            # traceback.print_exc()
            pass

    # Slots
    def newFile(self):
        pass

    @exception_handler
    def openFile(self, checked=False, file_path=None):
        """
        Function used for loading an image file directly.
        """
        if file_path is None:
            self.getMostRecentPath()
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
            self.openFolder(exp_path=exp_path, imageFilePath=file_path)
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
            self.openFolder(exp_path=exp_path, imageFilePath=tif_path)

    def criticalNoTifFound(self, images_path):
        err_title = f'No .tif files found in folder.'
        err_msg = (
            f'The folder "{images_path}" does not contain .tif files.\n\n'
            'Only .tif files can be loaded with "Open Folder" button.\n\n'
            'Try with "File --> Open image/video file..." and directly select '
            'the file you want to load.'
        )
        msg = QMessageBox()
        msg.critical(self, err_title, err_msg, msg.Ok)
        return

    def reInitGui(self):
        self.removeAllItems()
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

    @exception_handler
    def openFolder(self, checked=False, exp_path=None, imageFilePath=''):
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

        self.reInitGui()

        self.openAction.setEnabled(False)

        if self.slideshowWin is not None:
            self.slideshowWin.close()

        if self.ccaTableWin is not None:
            self.ccaTableWin.close()

        if exp_path is None:
            self.getMostRecentPath()
            exp_path = QFileDialog.getExistingDirectory(
                self, 'Select experiment folder containing Position_n folders '
                      'or specific Position_n folder', self.MostRecentPath)

        if exp_path == '':
            self.openAction.setEnabled(True)
            self.titleLabel.setText(
                'Drag and drop image file or go to File --> Open folder...',
                color='w')
            return

        self.exp_path = exp_path
        self.addToRecentPaths(exp_path)

        if os.path.basename(exp_path).find('Position_') != -1:
            is_pos_folder = True
        else:
            is_pos_folder = False

        if os.path.basename(exp_path).find('Images') != -1:
            is_images_folder = True
        else:
            is_images_folder = False

        self.titleLabel.setText('Loading data...', color='w')
        self.setWindowTitle(f'Cell-ACDC - GUI - "{exp_path}"')


        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )
        user_ch_name = None
        if not is_pos_folder and not is_images_folder and not imageFilePath:
            select_folder = load.select_exp_folder()
            values = select_folder.get_values_segmGUI(exp_path)
            if not values:
                txt = (
                    'The selected folder:\n\n '
                    f'{exp_path}\n\n'
                    'is not a valid folder. '
                    'Select a folder that contains the Position_n folders'
                )
                msg = QMessageBox()
                msg.critical(
                    self, 'Incompatible folder', txt, msg.Ok
                )
                self.titleLabel.setText(
                    'Drag and drop image file or go to File --> Open folder...',
                    color='w')
                self.openAction.setEnabled(True)
                return

            if len(values) > 1:
                select_folder.QtPrompt(self, values, allow_abort=False)
                if select_folder.was_aborted:
                    self.titleLabel.setText(
                        'Drag and drop image file or go to '
                        'File --> Open folder...',
                        color='w')
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
                ch_name_selector.get_available_channels(
                    filenames, exp_path)
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
                    color='w')
                self.openAction.setEnabled(True)
                self.criticalNoTifFound(images_path)
                return
            if len(ch_names) > 1:
                CbLabel='Select channel name to segment: '
                ch_name_selector.QtPrompt(
                    self, ch_names, CbLabel=CbLabel)
                if ch_name_selector.was_aborted:
                    self.titleLabel.setText(
                        'Drag and drop image file or go to File --> Open folder...',
                        color='w')
                    self.openAction.setEnabled(True)
                    return
            else:
                ch_name_selector.channel_name = ch_names[0]
            ch_name_selector.setUserChannelName()
            user_ch_name = ch_name_selector.user_ch_name

        user_ch_file_paths = []
        img_path = None
        for images_path in self.images_paths:
            img_aligned_found = False
            for filename in myutils.listdir(images_path):
                img_path = os.path.join(images_path, filename)
                if filename.find(f'{user_ch_name}_aligned.np') != -1:
                    aligned_path = img_path
                    img_aligned_found = True
                elif filename.find(f'{user_ch_name}.tif') != -1:
                    tif_path = img_path
            if not img_aligned_found:
                err_msg = ('Aligned frames file for channel '
                           f'{user_ch_name} not found. '
                           'Loading tifs files.')
                self.titleLabel.setText(err_msg)
                img_path = tif_path
            else:
                img_path = aligned_path
                # raise FileNotFoundError(err_msg)

            user_ch_file_paths.append(img_path)

        self.logger.info(f'Loading {img_path}...')
        self.appendPathWindowTitle(user_ch_file_paths)

        self.user_ch_name = user_ch_name

        self.initGlobalAttr()

        self.num_pos = len(user_ch_file_paths)
        proceed = self.loadSelectedData(user_ch_file_paths, user_ch_name)
        if not proceed:
            self.openAction.setEnabled(True)
            self.titleLabel.setText(
                'Drag and drop image file or go to File --> Open folder...',
                color='w')
            return

    def appendPathWindowTitle(self, user_ch_file_paths):
        if self.isSnapshot:
            return

        pos_path = os.path.dirname(os.path.dirname(user_ch_file_paths[0]))
        self.setWindowTitle(f'Cell-ACDC - GUI - "{pos_path}"')

    def initFluoData(self):
        if len(self.ch_names) <= 1:
            return
        msg = QMessageBox()
        load_fluo = msg.question(
            self, 'Load fluorescent images?',
            'Do you also want to load fluorescent images? You can load as '
            'many channels as you want.\n\n'
            'If you load fluorescent images then the software will '
            'calculate metrics for each loaded fluorescent channel '
            'such as min, max, mean, quantiles, etc. '
            'for each segmented object.\n\n'
            'NOTE: You can always load them later with '
            'File --> Load fluorescent images',
            msg.Yes | msg.No
        )
        if load_fluo == msg.Yes:
            self.loadFluo_cb(None)

    def getPathFromChName(self, chName, posData):
        aligned_files = [f for f in myutils.listdir(posData.images_path)
                         if f.find(f'{chName}_aligned.npz')!=-1]
        if aligned_files:
            filename = aligned_files[0]
        else:
            tif_files = [f for f in myutils.listdir(posData.images_path)
                         if f.find(f'{chName}.tif')!=-1]
            if not tif_files:
                self.criticalFluoChannelNotFound(chName, posData)
                self.app.restoreOverrideCursor()
                return None, None
            filename = tif_files[0]
        fluo_path = os.path.join(posData.images_path, filename)
        filename, _ = os.path.splitext(filename)
        return fluo_path, filename

    def loadFluo_cb(self, event):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        selectFluo = apps.QDialogListbox(
            'Select channel',
            'Select channel names to load:\n',
            ch_names, multiSelection=True, parent=self
        )
        selectFluo.exec_()

        if selectFluo.cancel:
            return

        fluo_channels = selectFluo.selectedItemsText

        for posData in self.data:
            posData.ol_data = None
            for fluo_ch in fluo_channels:
                fluo_path, filename = self.getPathFromChName(fluo_ch, posData)
                if fluo_path is None:
                    self.criticalFluoChannelNotFound(fluo_ch, posData)
                    self.app.restoreOverrideCursor()
                    return
                fluo_data, bkgrData = self.load_fluo_data(fluo_path)
                if fluo_data is None:
                    self.app.restoreOverrideCursor()
                    return
                posData.loadedFluoChannels.add(fluo_ch)

                if posData.SizeT < 2:
                    fluo_data = np.array([fluo_data])

                posData.fluo_data_dict[filename] = fluo_data
                posData.fluo_bkgrData_dict[filename] = bkgrData
                posData.ol_data_dict[filename] = fluo_data.copy()
        self.app.restoreOverrideCursor()
        self.overlayButton.setStyleSheet('background-color: #A7FAC7')

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if exp_path in recentPaths:
                self.logger.info(exp_path)
                pop_idx = recentPaths.index(exp_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, exp_path)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 20 recent paths
            if len(recentPaths) > 20:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        else:
            recentPaths = [exp_path]
            openedOn = [datetime.datetime.now()]
        df = pd.DataFrame({
            'path': recentPaths,
            'opened_last_on': pd.Series(openedOn, dtype='datetime64[ns]')}
        )
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    def showInExplorer(self):
        posData = self.data[self.pos_i]
        path = posData.images_path

        if os.name == 'posix' or os.name == 'os2':
            os.system(f'open "{path}"')
        elif os.name == 'nt':
            os.startfile(path)

    def getChNames(self, posData):
        fluo_keys = list(posData.fluo_data_dict.keys())

        loadedChNames = []
        for key in fluo_keys:
            chName = key[len(posData.basename):]
            if chName.find('_aligned') != -1:
                idx = chName.find('_aligned')
                chName = f'gui_{chName[:idx]}'
            loadedChNames.append(chName)

        posData.loadedChNames = loadedChNames

    def zSliceAbsent(self, filename, posData):
        self.app.restoreOverrideCursor()
        SizeZ = posData.SizeZ
        chNames = posData.chNames
        filenamesPresent = posData.segmInfo_df.index.get_level_values(0).unique()
        chNamesPresent = [
            ch for ch in chNames
            for file in filenamesPresent
            if file.find(ch) != -1
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
                src_df = _posData.segmInfo_df.loc[srcFilename].copy()
                _, dstFilename = self.getPathFromChName(user_ch_name, _posData)
                dst_df = myutils.getDefault_SegmInfo_df(_posData, dstFilename)
                for z_info in src_df.itertuples():
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
            dataPrepWin.loadFiles(
                exp_path, user_ch_file_paths, user_ch_name
            )
            dataPrepWin.startAction.setDisabled(True)

            loop = QEventLoop(self)
            dataPrepWin.loop = loop
            loop.exec_()

        self.waitCond.wakeAll()

    def set_metrics_func(self):
        self.metrics_func = {
            'mean': lambda arr: arr.mean(),
            'sum': lambda arr: arr.sum(),
            'amount_autoBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
            'amount_dataPrepBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
            'median': lambda arr: np.median(arr),
            'min': lambda arr: numba_min(arr),
            'max': lambda arr: numba_max(arr),
            'q25': lambda arr: np.quantile(arr, q=0.25),
            'q75': lambda arr: np.quantile(arr, q=0.75),
            'q05': lambda arr: np.quantile(arr, q=0.05),
            'q95': lambda arr: np.quantile(arr, q=0.95)
        }

        self.total_metrics = len(self.metrics_func)

        bkgr_val_names = (
            'autoBkgr_val_median',
            'autoBkgr_val_mean',
            'autoBkgr_val_q75',
            'autoBkgr_val_q25',
            'autoBkgr_val_q95',
            'autoBkgr_val_q05',
            'dataPrepBkgr_val_median',
            'dataPrepBkgr_val_mean',
            'dataPrepBkgr_val_q75',
            'dataPrepBkgr_val_q25',
            'dataPrepBkgr_val_q95',
            'dataPrepBkgr_val_q05',
        )

        self.all_metrics_names = list(self.metrics_func.keys())
        self.all_metrics_names.extend(bkgr_val_names)

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
        numCells = len(rp)

        list_0s = [0]*numCells
        IDs = list_0s.copy()
        IDs_vol_vox = list_0s.copy()
        IDs_area_pxl = list_0s.copy()
        IDs_vol_fl = list_0s.copy()
        IDs_area_um2 = list_0s.copy()

        # Initialize fluo metrics arrays
        fluo_keys = list(posData.fluo_data_dict.keys())
        fluo_data = posData.fluo_data_dict[fluo_keys[0]][frame_i]
        is_3D = fluo_data.ndim == 3
        how_3Dto2D = ['_maxProj', '_meanProj', '_zSlice'] if is_3D else ['']
        n = len(how_3Dto2D)
        numFluoChannels = len(fluo_keys)

        # Defined in function set_metrics_func
        metrics_func = self.metrics_func

        # Dictionary where values is a list of 0s with len=numCells
        # and key is 'channelName_metrics_how' (e.g. 'GFP_mean_zSlice')
        metrics_values = {
            f'{chName}_{metric}{how}':list_0s.copy()
            for metric in self.all_metrics_names
            for chName in posData.loadedChNames
            for how in how_3Dto2D
        }

        tot_iter = (
            self.total_metrics
            *len(posData.loadedChNames)
            *len(how_3Dto2D)
            *numCells
        )

        self.worker.metricsPbarProgress.emit(tot_iter, 0)

        # pbar = tqdm(total=tot_iter, ncols=100, unit='metric', leave=False)

        outCellsMask = lab==0

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
                        self.worker.mutex.lock()
                        self.worker.askZsliceAbsent.emit(filename, posData)
                        self.worker.waitCond.wait(self.mutex)
                        self.worker.mutex.unlock()
                        segmInfo_df = pd.read_csv(posData.segmInfo_df_csv_path)
                        index_col = ['filename', 'frame_i']
                        posData.segmInfo_df = segmInfo_df.set_index(index_col)
                        col = 'z_slice_used_dataPrep'
                        z_slice = posData.segmInfo_df.at[idx, col]

                fluo_data_z_maxP = fluo_data.max(axis=0)
                fluo_data_z_sumP = fluo_data.mean(axis=0)
                fluo_data_zSlice = fluo_data[z_slice]

                # how_3Dto2D = ['_maxProj', '_sumProj', '_zSlice']
                fluo_data_projs.append(fluo_data_z_maxP)
                fluo_data_projs.append(fluo_data_z_sumP)
                fluo_data_projs.append(fluo_data_zSlice)
                if bkgrArchive is not None:
                    bkgrVals_z_maxP = []
                    bkgrVals_z_sumP = []
                    bkgrVals_zSlice = []
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
                    bkgrData_medians.append(np.median(bkgrVals_z_maxP))
                    bkgrData_medians.append(np.median(bkgrVals_z_sumP))
                    bkgrData_medians.append(np.median(bkgrVals_zSlice))

                    bkgrData_means.append(bkgrVals_z_maxP.mean())
                    bkgrData_means.append(bkgrVals_z_sumP.mean())
                    bkgrData_means.append(bkgrVals_zSlice.mean())

                    bkgrData_q75s.append(np.quantile(bkgrVals_z_maxP, q=0.75))
                    bkgrData_q75s.append(np.quantile(bkgrVals_z_sumP, q=0.75))
                    bkgrData_q75s.append(np.quantile(bkgrVals_zSlice, q=0.75))

                    bkgrData_q25s.append(np.quantile(bkgrVals_z_maxP, q=0.25))
                    bkgrData_q25s.append(np.quantile(bkgrVals_z_sumP, q=0.25))
                    bkgrData_q25s.append(np.quantile(bkgrVals_zSlice, q=0.25))

                    bkgrData_q95s.append(np.quantile(bkgrVals_z_maxP, q=0.95))
                    bkgrData_q95s.append(np.quantile(bkgrVals_z_sumP, q=0.95))
                    bkgrData_q95s.append(np.quantile(bkgrVals_zSlice, q=0.95))

                    bkgrData_q05s.append(np.quantile(bkgrVals_z_maxP, q=0.05))
                    bkgrData_q05s.append(np.quantile(bkgrVals_z_sumP, q=0.05))
                    bkgrData_q05s.append(np.quantile(bkgrVals_zSlice, q=0.05))
            else:
                fluo_data_2D = fluo_data
                fluo_data_projs.append(fluo_data_2D)
                if bkgrArchive is not None:
                    bkgrVals_2D = []
                    for roi_key in bkgrArchive.files:
                        roiData = bkgrArchive[roi_key]
                        if posData.SizeT > 1:
                            roiData = bkgrArchive[roi_key][frame_i]
                        bkgrVals_2D.extend(roiData[roiData!=0])
                    bkgrData_medians.append(np.median(bkgrVals_2D))
                    bkgrData_means.append(np.mean(bkgrVals_2D))
                    bkgrData_q75s.append(np.quantile(bkgrVals_2D, q=0.75))
                    bkgrData_q25s.append(np.quantile(bkgrVals_2D, q=0.25))
                    bkgrData_q95s.append(np.quantile(bkgrVals_2D, q=0.95))
                    bkgrData_q05s.append(np.quantile(bkgrVals_2D, q=0.05))

            # Iterate cells
            for i, obj in enumerate(rp):
                IDs[i] = obj.label
                # Calc volume
                vol_vox, vol_fl = _calc_rot_vol(
                    obj, PhysicalSizeY, PhysicalSizeX
                )
                IDs_vol_vox[i] = vol_vox
                IDs_area_pxl[i] = obj.area
                IDs_vol_fl[i] = vol_fl
                IDs_area_um2[i] = obj.area*yx_pxl_to_um2

                # Iterate method of 3D to 2D
                how_iterable = enumerate(zip(how_3Dto2D, fluo_data_projs))
                for k, (how, fluo_2D) in how_iterable:
                    fluo_data_ID = fluo_2D[obj.slice][obj.image]

                    # fluo_2D!=0 is required because when we align we pad with 0s
                    # instead of np.roll and we don't want to include those
                    # exact 0s in the backgrMask
                    backgrMask = np.logical_and(outCellsMask, fluo_2D!=0)
                    bkgr_arr = fluo_2D[backgrMask]
                    fluo_backgr = np.median(bkgr_arr)

                    bkgr_key = f'{chName}_autoBkgr_val_median{how}'
                    metrics_values[bkgr_key][i] = fluo_backgr

                    bkgr_key = f'{chName}_autoBkgr_val_mean{how}'
                    metrics_values[bkgr_key][i] = bkgr_arr.mean()

                    bkgr_key = f'{chName}_autoBkgr_val_q75{how}'
                    metrics_values[bkgr_key][i] = np.quantile(bkgr_arr, q=0.75)

                    bkgr_key = f'{chName}_autoBkgr_val_q25{how}'
                    metrics_values[bkgr_key][i] = np.quantile(bkgr_arr, q=0.25)

                    bkgr_key = f'{chName}_autoBkgr_val_q95{how}'
                    metrics_values[bkgr_key][i] = np.quantile(bkgr_arr, q=0.95)

                    bkgr_key = f'{chName}_autoBkgr_val_q05{how}'
                    metrics_values[bkgr_key][i] = np.quantile(bkgr_arr, q=0.05)

                    # Calculate metrics for each cell
                    for func_name, func in metrics_func.items():
                        key = f'{chName}_{func_name}{how}'
                        is_ROIbkgr_func = (
                            func_name == 'amount_dataPrepBkgr' and
                            (ROI_bkgrMask is not None or bkgrArchive is not None)
                        )
                        if func_name == 'amount_autoBkgr':
                            val = func(fluo_data_ID, fluo_backgr, obj.area)
                            metrics_values[key][i] = val
                        elif is_ROIbkgr_func:
                            if ROI_bkgrMask is not None:
                                ROI_bkgrData = fluo_2D[ROI_bkgrMask]
                                ROI_bkgrVal = np.median(ROI_bkgrData)
                            else:
                                ROI_bkgrVal = bkgrData_medians[k]
                            val = func(fluo_data_ID, ROI_bkgrVal, obj.area)
                            metrics_values[key][i] = val

                            bkgr_key = f'{chName}_dataPrepBkgr_val_median{how}'
                            metrics_values[bkgr_key][i] = ROI_bkgrVal

                            bkgr_key = f'{chName}_dataPrepBkgr_val_mean{how}'
                            if ROI_bkgrMask is None:
                                bkgr_val = bkgrData_means[k]
                            else:
                                bkgr_val = ROI_bkgrData.mean()
                            metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_val_q75{how}'
                            if ROI_bkgrMask is None:
                                bkgr_val = bkgrData_q75s[k]
                            else:
                                bkgr_val = np.quantile(ROI_bkgrData, q=0.75)
                            metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_val_q25{how}'
                            if ROI_bkgrMask is None:
                                bkgr_val = bkgrData_q25s[k]
                            else:
                                bkgr_val = np.quantile(ROI_bkgrData, q=0.25)
                            metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_val_q95{how}'
                            if ROI_bkgrMask is None:
                                bkgr_val = bkgrData_q95s[k]
                            else:
                                bkgr_val = np.quantile(ROI_bkgrData, q=0.95)
                            metrics_values[bkgr_key][i] = bkgr_val

                            bkgr_key = f'{chName}_dataPrepBkgr_val_q05{how}'
                            if ROI_bkgrMask is None:
                                bkgr_val = bkgrData_q05s[k]
                            else:
                                bkgr_val = np.quantile(ROI_bkgrData, q=0.05)
                            metrics_values[bkgr_key][i] = bkgr_val

                        elif func_name.find('amount') == -1:
                            val = func(fluo_data_ID)
                            metrics_values[key][i] = val

                        # pbar.update()
                        self.worker.metricsPbarProgress.emit(-1, 1)

        df['cell_area_pxl'] = pd.Series(data=IDs_area_pxl, index=IDs, dtype=float)
        df['cell_vol_vox'] = pd.Series(data=IDs_vol_vox, index=IDs, dtype=float)
        df['cell_area_um2'] = pd.Series(data=IDs_area_um2, index=IDs, dtype=float)
        df['cell_vol_fl'] = pd.Series(data=IDs_vol_fl, index=IDs, dtype=float)

        df_metrics = pd.DataFrame(metrics_values, index=IDs)

        df = df.join(df_metrics)

        # pbar.close()

        # Join with regionprops_table
        props = (
            'label',
            'bbox',
            'bbox_area',
            'eccentricity',
            'equivalent_diameter',
            'euler_number',
            'extent',
            'filled_area',
            'inertia_tensor_eigvals',
            'local_centroid',
            'major_axis_length',
            'minor_axis_length',
            'moments',
            'moments_central',
            'moments_hu',
            'moments_normalized',
            'orientation',
            'perimeter',
            'solidity'
        )
        rp_table = skimage.measure.regionprops_table(
            posData.lab, properties=props
        )
        df_rp = pd.DataFrame(rp_table).set_index('label')

        df = df.join(df_rp)
        return df

    # def askSaveLastVisitedCcaMode(self, p, posData):
    #     current_frame_i = posData.frame_i
    #     frame_i = 0
    #     last_cca_frame_i = 0
    #     self.save_until_frame_i = 0
    #     self.worker.askSaveLastCancelled = False
    #     for frame_i, data_dict in enumerate(posData.allData_li):
    #         # Build segm_npy
    #         acdc_df = data_dict['acdc_df']
    #         if acdc_df is None:
    #             frame_i -= 1
    #             break
    #         if 'cell_cycle_stage' not in acdc_df.columns:
    #             frame_i -= 1
    #             break
    #     if frame_i > 0:
    #         # Ask to save last visited frame or not
    #         txt = (f"""
    #         <p style="font-size:9pt">
    #             You visited and annotated data up until frame
    #             number {frame_i+1}.<br><br>
    #             Enter <b>up to which frame number</b> you want to save data:
    #         </p>
    #         """)
    #         lastFrameDialog = apps.QLineEditDialog(
    #             title='Last frame number to save', defaultTxt=str(frame_i+1),
    #             msg=txt, parent=self, allowedValues=range(1, frame_i+2)
    #         )
    #         lastFrameDialog.exec_()
    #         if lastFrameDialog.cancel:
    #             self.worker.askSaveLastCancelled = True
    #         else:
    #             self.save_until_frame_i = lastFrameDialog.EntryID - 1
    #             last_cca_frame_i = self.save_until_frame_i
    #     self.last_cca_frame_i = last_cca_frame_i
    #     self.waitCond.wakeAll()

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

    def askSaveLastVisitedSegmMode(self, p, posData):
        current_frame_i = posData.frame_i
        frame_i = 0
        last_tracked_i = 0
        self.save_until_frame_i = 0
        self.worker.askSaveLastCancelled = False
        for frame_i, data_dict in enumerate(posData.allData_li):
            # Build segm_npy
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break
        if frame_i > 0:
            # Ask to save last visited frame or not
            txt = (f"""
            <p style="font-size:9pt">
                You visited and stored data up until frame
                number {frame_i+1}.<br><br>
                Enter <b>up to which frame number</b> you want to save data:
            </p>
            """)
            lastFrameDialog = apps.QLineEditDialog(
                title='Last frame number to save', defaultTxt=str(frame_i+1),
                msg=txt, parent=self, allowedValues=range(1, frame_i+2),
                warnLastFrame=True
            )
            lastFrameDialog.exec_()
            if lastFrameDialog.cancel:
                self.worker.askSaveLastCancelled = True
            else:
                self.save_until_frame_i = lastFrameDialog.EntryID - 1
                last_tracked_i = self.save_until_frame_i
        self.last_tracked_i = last_tracked_i
        self.waitCond.wakeAll()

    def askSaveMetrics(self):
        txt = (
        """
        <p style="font-size:10pt">
            Do you also want to <b>save additional metrics</b>
            (e.g., cell volume, mean, amount etc.)?<br><br>
            NOTE: Saving additional metrics is <b>slower</b>,
            we recommend doing it only when you need it.<br>
        </p>
        """)
        cancel = True
        msg = QMessageBox()
        save_metrics_answer = msg.question(
            self, 'Save metrics?', txt,
            msg.Yes | msg.No | msg.Cancel
        )
        save_metrics = save_metrics_answer == msg.Yes
        cancel = save_metrics_answer == msg.Cancel
        return save_metrics, cancel

    def askSaveAllPos(self):
        last_pos = 0
        ask = False
        for p, posData in enumerate(self.data):
            acdc_df = posData.allData_li[0]['acdc_df']
            if acdc_df is None:
                last_pos = p
                ask = True
                break

        if not ask:
            # All pos have been visited, no reason to ask
            return True, len(self.data)

        msg = QMessageBox(self)
        msg.setWindowTitle('Save all positions?')
        msg.setIcon(msg.Question)
        txt = (
        f"""
        <p style="font-size:10pt">
            Do you want to save <b>ALL positions</b> or <b>only until
            Position_{last_pos}</b> (last visualized/corrected position)?<br>
        </p>
        """)
        msg.setText(txt)
        allPosbutton =  QPushButton('Save ALL positions')
        upToLastButton = QPushButton(f'Save until Position_{last_pos}')
        msg.addButton(allPosbutton, msg.YesRole)
        msg.addButton(upToLastButton, msg.NoRole)
        msg.exec_()
        return msg.clickedButton() == allPosbutton, last_pos

    def saveMetricsCritical(self, traceback_format):
        self.logger.info('')
        self.logger.info('====================================')
        self.logger.info(traceback_format)
        self.logger.info('====================================')
        self.logger.info('')
        self.logger.info('Warning: calculating metrics failed see above...')
        self.logger.info('-----------------')
        msg = QMessageBox(self)
        msg.setIcon(msg.Critical)
        msg.setWindowTitle('Error')
        msg.setText(traceback_format)
        msg.setDefaultButton(msg.Ok)
        msg.exec_()
        self.waitCond.wakeAll()

    def saveDataPermissionError(self, err_msg):
        msg = QMessageBox()
        msg.critical(self, 'Permission denied', err_msg, msg.Ok)
        self.waitCond.wakeAll()

    def saveDataProgress(self, text):
        self.logger.info(text)
        self.saveWin.progressLabel.setText(text)

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

    @exception_handler
    def saveData(self):
        self.store_data()
        self.titleLabel.setText(
            'Saving data... (check progress in the terminal)', color='w'
        )

        self.save_metrics, cancel = self.askSaveMetrics()
        if cancel:
            self.titleLabel.setText(
                'Saving data process cancelled.', color='w'
            )
            return

        last_pos = len(self.data)
        if self.isSnapshot:
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

        infoTxt = (
        f"""
            <p style=font-size:10pt>
                Saving {self.exp_path}...<br>
            </p>
        """)

        self.saveWin = apps.QDialogPbar(
            parent=self, title='Saving data', infoTxt=infoTxt
        )
        font = QtGui.QFont()
        font.setPointSize(10)
        self.saveWin.setFont(font)
        if not self.save_metrics:
            self.saveWin.metricsQPbar.hide()
        self.saveWin.progressLabel.setText('Preparing data...')
        self.saveWin.show()

        # Set up separate thread for saving and show progress bar widget
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.thread = QThread()
        self.worker = saveDataWorker(self)

        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.finished.connect(self.saveDataFinished)
        self.worker.progress.connect(self.saveDataProgress)
        self.worker.progressBar.connect(self.saveDataUpdatePbar)
        self.worker.metricsPbarProgress.connect(self.saveDataUpdateMetricsPbar)
        self.worker.critical.connect(self.saveDataCritical)
        self.worker.criticalMetrics.connect(self.saveMetricsCritical)
        self.worker.criticalPermissionError.connect(self.saveDataPermissionError)
        self.worker.askSaveLastVisitedCcaMode.connect(
            self.askSaveLastVisitedSegmMode
        )
        self.worker.askSaveLastVisitedSegmMode.connect(
            self.askSaveLastVisitedSegmMode
        )
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

    def copyContent(self):
        pass

    def pasteContent(self):
        pass

    def cutContent(self):
        pass

    def showTipsAndTricks(self):
        self.welcomeWin = welcome.welcomeWin(app=app)
        self.welcomeWin.showAndSetSize()
        self.welcomeWin.showPage(self.welcomeWin.quickStartItem)

    def showUserManual(self):
        systems = {
            'nt': os.startfile,
            'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
            'os2': lambda foldername: os.system('open "%s"' % foldername)
             }

        main_path = pathlib.Path(__file__).resolve().parents[1]
        userManual_path = main_path / 'UserManual'
        systems.get(os.name, os.startfile)(userManual_path)

    def about(self):
        pass

    def populateOpenRecent(self):
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 1. Read recent Paths
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
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
            action = QAction(path, self)
            action.triggered.connect(partial(self.openRecentFile, path))
            actions.append(action)
        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)

    def getMostRecentPath(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            self.MostRecentPath = ''
            for path in df['path']:
                if os.path.exists(path):
                    self.MostRecentPath = path
                    break
        else:
            self.MostRecentPath = ''

    def openRecentFile(self, path):
        self.logger.info(f'Opening recent folder: {path}')
        self.openFolder(exp_path=path)

    def closeEvent(self, event):
        self.saveWindowGeometry()
        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()
        if self.saveAction.isEnabled() and self.titleLabel.text != 'Saved!':
            msg = QMessageBox()
            save = msg.question(
                self, 'Save?', 'Do you want to save?',
                msg.Yes | msg.No | msg.Cancel
            )
            if save == msg.Yes:
                self.saveData()
            elif save == msg.Cancel:
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
        self.alphaScrollBar.setFixedHeight(h)
        self.zSliceOverlay_SB.setFixedHeight(h)

        self.gui_initImg1BottomWidgets()
        self.img1BottomGroupbox.setVisible(False)


if __name__ == "__main__":
    print('Loading application...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    app.setWindowIcon(QIcon(":assign-motherbud.svg"))
    # Apply dark mode
    # file = QFile(":/dark.qss")
    # file.open(QFile.ReadOnly | QFile.Text)
    # stream = QTextStream(file)
    # app.setStyleShefet(stream.readAll())
    # Create and show the main window
    win = guiWin(app)
    win.show()

    # Run the event loop
    win.logger.info('Lauching application...')
    win.logger.info(
        'Done. If application GUI is not visible, it is probably minimized, '
         'behind some other open window, or on second screen.'
    )
    sys.exit(app.exec_())
