import gc
import os
import sys
import traceback
import re
import time
import datetime
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
import scipy.interpolate
import skimage
import skimage.io
from tqdm import tqdm
from functools import partial, wraps
from tifffile.tifffile import TiffWriter, TiffFile

from qtpy.QtCore import (
    Qt, QFile, QTextStream, QSize, QRect, QRectF,
    QObject, QThread, Signal, QSettings
)
from qtpy.QtGui import (
    QIcon, QKeySequence, QCursor, QTextBlockFormat,
    QTextCursor, QFont
)
from qtpy.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton, QWidget,
    QMainWindow, QMenu, QToolBar, QGroupBox, QGridLayout,
    QScrollBar, QCheckBox, QToolButton, QSpinBox,
    QComboBox, QDial, QButtonGroup, QFileDialog,
    QAbstractSlider, QMessageBox, QStyleFactory
)
from qtpy.compat import getexistingdirectory

import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

# NOTE: Enable icons
from . import qrc_resources

# Custom modules
from . import exception_handler
from . import load, prompts, apps, core, myutils, widgets
from . import html_utils, myutils, darkBkgrColor, printl
from . import autopilot
from . import recentPaths_path, cellacdc_path, settings_folderpath

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class toCsvWorker(QObject):
    finished = Signal()
    progress = Signal(int)

    def setData(self, data):
        self.data = data

    def run(self):
        for posData in self.data:
            posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
        self.finished.emit()

class dataPrepWin(QMainWindow):
    sigClose = Signal(object)

    def __init__(
            self, parent=None, buttonToRestore=None, mainWin=None,
            version=None
        ):
        from .config import parser_args
        self.debug = parser_args['debug']

        super().__init__(parent)

        self._version = version

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='dataPrep'
        )
        self.logger = logger

        if self._version is not None:
            logger.info(f'Initializing Data Prep module v{self._version}...')
        else:
            logger.info(f'Initializing Data Prep module...')

        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        if mainWin is not None:
            self.app = mainWin.app

        self.setWindowTitle("Cell-ACDC - data prep")
        self.setGeometry(100, 50, 850, 800)
        self.setWindowIcon(QIcon(":icon.ico"))

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_addGraphicsItems()

        self.gui_createImgWidgets()
        self.num_frames = 0
        self.frame_i = 0
        self.loop = None
        self.titleText = None
        self.metadataAlreadyAsked = False
        self.cropZtool = None
        self.AutoPilotProfile = autopilot.AutoPilotProfile()
        self.AutoPilot = None
        self.dataIsLoaded = False

        # When we load dataprep from other modules we usually disable
        # start because we only want to select the z-slice
        # However, if start is disabled removeBkgrROIs will be triggered
        # and cause errors --> set self.onlySelectingZslice = True
        # when dataprep is launched from the other modules
        self.onlySelectingZslice = False

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

    @exception_handler
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            posData = self.data[self.pos_i]
            # printl(posData.all_npz_paths)
            # printl(posData.tif_paths)
            # for r, roi in enumerate(posData.bkgrROIs):
            #     print(roi.pos(), roi.size())
            #     xl, yt = [int(round(c)) for c in roi.pos()]
            #     w, h = [int(round(c)) for c in roi.size()]
            #     print('-'*20)
            #     print(yt, yt+h, yt+h>yt)
            #     print(xl, xl+w, xl+w>xl)
        if event.key() == Qt.Key_Left:
            self.navigateScrollbar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )
        elif event.key() == Qt.Key_Right:
            self.navigateScrollbar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )
        elif event.key() == Qt.Key_Up:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )
        elif event.key() == Qt.Key_Down:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )

    def gui_createActions(self):
        # File actions
        self.openAction = QAction(QIcon(":folder-open.svg"), "&Open...", self)
        self.exitAction = QAction("&Exit", self)
        self.showInExplorerAction = QAction(QIcon(":drawer.svg"),
                                    "&Show in Explorer/Finder", self)
        self.showInExplorerAction.setDisabled(True)

        # Toolbar actions
        # self.jumpForwardAction = QAction(QIcon(":arrow-up.svg"),
        #                                 "Jump to 10 frames ahead", self)
        # self.jumpBackwardAction = QAction(QIcon(":arrow-down.svg"),
        #                                 "Jump to 10 frames back", self)
        self.openAction.setShortcut("Ctrl+O")
        # self.jumpForwardAction.setShortcut("up")
        # self.jumpBackwardAction.setShortcut("down")
        self.openAction.setShortcut("Ctrl+O")

        self.loadPosAction = QAction("Load different Position...", self)
        self.loadPosAction.setShortcut("Shift+P")
        
        toolTip = (
            "Add crop ROI for multiple crops\n\n"
            "Multiple crops will be saved as Position_1, Position_2 "
            "as sub-folders in the current Position."
        )
        self.addCropRoiActon = QAction(QIcon(":add_crop_ROI.svg"), toolTip, self)
        self.addCropRoiActon.setDisabled(True)

        toolTip = "Add ROI where to calculate background intensity"
        self.addBkrgRoiActon = QAction(QIcon(":bkgrRoi.svg"), toolTip, self)
        self.addBkrgRoiActon.setDisabled(True)

        self.ZbackAction = QAction(QIcon(":zback.svg"),
                                "Use same z-slice from first frame to here",
                                self)
        self.ZbackAction.setEnabled(False)
        self.ZforwAction = QAction(QIcon(":zforw.svg"),
                                "Use same z-slice from here to last frame",
                                self)
        self.ZforwAction.setEnabled(False)

        self.interpAction = QAction(QIcon(":interp.svg"),
                                "Interpolate z-slice from first slice to here",
                                self)
        self.interpAction.setEnabled(False)


        self.cropAction = QAction(QIcon(":crop.svg"), "Crop and save", self)
        self.cropAction.setEnabled(False)

        self.cropZaction = QAction(QIcon(":cropZ.svg"), "Crop z-slices", self)
        self.cropZaction.setEnabled(False)
        self.cropZaction.setCheckable(True)

        self.startAction = QAction(QIcon(":start.svg"), "Start process!", self)
        self.startAction.setEnabled(False)

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.openAction)
        # Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")

        fileMenu.addAction(self.loadPosAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

    def gui_createToolBars(self):
        toolbarSize = 34

        # File toolbar
        fileToolBar = self.addToolBar("File")
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)

        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.showInExplorerAction)
        fileToolBar.addAction(self.startAction)
        fileToolBar.addAction(self.cropAction)
        fileToolBar.addAction(self.cropZaction)

        navigateToolbar = QToolBar("Navigate", self)
        # navigateToolbar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolbar)

        # navigateToolbar.addAction(self.jumpBackwardAction)
        # navigateToolbar.addAction(self.jumpForwardAction)
        navigateToolbar.addAction(self.addCropRoiActon)
        navigateToolbar.addAction(self.addBkrgRoiActon)

        navigateToolbar.addAction(self.ZbackAction)
        navigateToolbar.addAction(self.ZforwAction)
        navigateToolbar.addAction(self.interpAction)

        self.ROIshapeComboBox = QComboBox()
        self.ROIshapeComboBox.setFont(apps.font)
        self.ROIshapeComboBox.SizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.ROIshapeComboBox.addItems(['  256x256  '])
        ROIshapeLabel = QLabel(html_utils.paragraph(
            '&nbsp;&nbsp;&nbsp;ROI standard shape: ')
        )
        ROIshapeLabel.setBuddy(self.ROIshapeComboBox)
        navigateToolbar.addWidget(ROIshapeLabel)
        navigateToolbar.addWidget(self.ROIshapeComboBox)

        self.ROIshapeLabel = QLabel('   Current ROI shape: 256 x 256')
        navigateToolbar.addWidget(self.ROIshapeLabel)

    def gui_connectActions(self):
        self.openAction.triggered.connect(self.openFolder)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.exitAction.triggered.connect(self.close)
        self.showInExplorerAction.triggered.connect(self.showInExplorer)
        self.addCropRoiActon.triggered.connect(self.addCropROI)
        self.addBkrgRoiActon.triggered.connect(self.addDefaultBkgrROI)
        self.cropAction.triggered.connect(self.crop_cb)
        self.cropZaction.toggled.connect(self.openCropZtool)
        self.startAction.triggered.connect(self.prepData)
        self.interpAction.triggered.connect(self.interp_z)
        self.ZbackAction.triggered.connect(self.useSameZ_fromHereBack)
        self.ZforwAction.triggered.connect(self.useSameZ_fromHereForw)
        self.loadPosAction.triggered.connect(self.loadPosTriggered)

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_addGraphicsItems(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        self.graphLayout.setBackground(darkBkgrColor)

        # Plot Item container for image
        self.ax1 = pg.PlotItem()
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=1, col=1)

        #Image histogram
        self.hist = widgets.myHistogramLUTitem()
        self.graphLayout.addItem(self.hist, row=1, col=0)

        # Title
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.titleLabel.setText(
            'File --> Open or Open recent to start the process')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1)

        # Current frame text
        # self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        # self.frameLabel.setText(' ')
        # self.graphLayout.addItem(self.frameLabel, row=2, col=1)

    def gui_addPlotItems(self):
        # Image Item
        # blankImage = np.full((512,512,3), darkBkgrColor)
        self.img = pg.ImageItem()
        self.ax1.addItem(self.img)
        self.hist.setImageItem(self.img)

    def removeAllItems(self):
        self.ax1.clear()
        # self.frameLabel.setText(' ')

    def gui_connectGraphicsEvents(self):
        self.img.hoverEvent = self.gui_hoverEventImg
        self.img.mousePressEvent = self.gui_mousePressEventImg

    def gui_createImgWidgets(self):
        self.img_Widglayout = QGridLayout()

        _font = QFont()
        _font.setPixelSize(13)

        self.navigateScrollbar = QScrollBar(Qt.Horizontal)
        self.navigateScrollbar.setFixedHeight(20)
        self.navigateScrollbar.setDisabled(True)
        navSB_label = QLabel('')
        navSB_label.setFont(_font)
        self.navigateSB_label = navSB_label


        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        self.zSliceScrollBar.setFixedHeight(20)
        self.zSliceScrollBar.setDisabled(True)
        _z_label = QLabel('z-slice  ')
        _z_label.setFont(_font)
        self.z_label = _z_label

        self.zProjComboBox = QComboBox()
        self.zProjComboBox.addItems(['single z-slice',
                                     'max z-projection',
                                     'mean z-projection',
                                     'median z-proj.'])
        self.zProjComboBox.setDisabled(True)

        self.img_Widglayout.addWidget(navSB_label, 0, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.navigateScrollbar, 0, 1, 1, 30)

        self.img_Widglayout.addWidget(_z_label, 1, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.zSliceScrollBar, 1, 1, 1, 30)

        self.img_Widglayout.addWidget(self.zProjComboBox, 1, 31, 1, 1)

        self.img_Widglayout.setContentsMargins(100, 0, 20, 0)

    def gui_hoverEventImg(self, event):
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(round(x)), int(round(y))
            _img = self.img.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(
                    f'(x={xdata:.2f}, y={ydata:.2f}, value={val:.2f})'
                )
            else:
                self.wcLabel.setText(f'')
        except Exception as e:
            self.wcLabel.setText(f'')

    def showInExplorer(self):
        try:
            posData = self.data[self.pos_i]
            systems = {
                'nt': os.startfile,
                'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
                'os2': lambda foldername: os.system('open "%s"' % foldername)
                 }

            systems.get(os.name, os.startfile)(posData.images_path)
        except AttributeError:
            pass
    
    def loadPosTriggered(self):
        if not self.dataIsLoaded:
            return
        
        self.startAutomaticLoadingPos()
    
    def startAutomaticLoadingPos(self):
        self.AutoPilot = autopilot.AutoPilot(self)
        self.AutoPilot.execLoadPos()
    
    def stopAutomaticLoadingPos(self):
        if self.AutoPilot is None:
            return
        
        if self.AutoPilot.timer.isActive():
            self.AutoPilot.timer.stop()
        self.AutoPilot = None
    
    def updatePos(self):
        self.updateCropZtool()
        self.setImageNameText()
        self.update_img()
        self.addAndConnectCropROIs()
        self.updateBkgrROIs()
        self.saveBkgrROIs(self.data[self.pos_i])
    
    def clearCurrentPos(self):
        self.removeBkgrROIs()
        self.removeCropROIs()

    def skip10ahead_frames(self):
        if self.frame_i < self.num_frames-10:
            self.frame_i += 10
        else:
            self.frame_i = 0
        self.update_img()

    def skip10back_frames(self):
        if self.frame_i > 9:
            self.frame_i -= 10
        else:
            self.frame_i = self.num_frames-1
        self.update_img()

    def updateNavigateItems(self):
        posData = self.data[self.pos_i]
        if self.num_pos > 1:
            # self.frameLabel.setText(
            #          f'Current position = {self.pos_i+1}/{self.num_pos} '
            #          f'({posData.pos_foldername})')
            self.navigateSB_label.setText(f'Pos n. {self.pos_i+1}')
            try:
                self.navigateScrollbar.valueChanged.disconnect()
            except TypeError:
                pass
            self.navigateScrollbar.setValue(self.pos_i+1)
        else:
            # self.frameLabel.setText(
            #          f'Current frame = {self.frame_i+1}/{self.num_frames}')
            self.navigateSB_label.setText(f'frame n. {self.frame_i+1}')
            try:
                self.navigateScrollbar.valueChanged.disconnect()
            except TypeError:
                pass
            self.navigateScrollbar.setValue(self.frame_i+1)
        self.navigateScrollbar.valueChanged.connect(
            self.navigateScrollbarValueChanged
        )

    def getImage(self, posData, img_data, frame_i, force_z=None):
        if posData.SizeT > 1:
            img = img_data[frame_i].copy()
        else:
            img = img_data.copy()
        if posData.SizeZ > 1:
            if force_z is not None:
                self.z_label.setText(f'z-slice  {force_z+1}/{posData.SizeZ}')
                img = img[force_z]
                return img
            df =  posData.segmInfo_df
            idx = (posData.filename, frame_i)
            try:
                z = posData.segmInfo_df.at[idx, 'z_slice_used_dataPrep']
            except Exception as e:
                duplicated_idx = posData.segmInfo_df.index.duplicated()
                posData.segmInfo_df = posData.segmInfo_df[~duplicated_idx]
                z = posData.segmInfo_df.at[idx, 'z_slice_used_dataPrep']
                
            zProjHow = posData.segmInfo_df.at[idx, 'which_z_proj']
            try:
                self.zProjComboBox.currentTextChanged.disconnect()
            except TypeError:
                pass
            self.zProjComboBox.setCurrentText(zProjHow)
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)

            if zProjHow == 'single z-slice':
                self.zSliceScrollBar.valueChanged.disconnect()
                self.zSliceScrollBar.setSliderPosition(z)
                self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
                self.z_label.setText(f'z-slice  {z+1}/{posData.SizeZ}')
                img = img[z]
            elif zProjHow == 'max z-projection':
                img = img.max(axis=0)
            elif zProjHow == 'mean z-projection':
                img = img.mean(axis=0)
            elif zProjHow == 'median z-proj.':
                img = np.median(img, axis=0)
        return img

    @exception_handler
    def update_img(self):
        self.updateNavigateItems()
        posData = self.data[self.pos_i]
        img = self.getImage(posData, posData.img_data, self.frame_i)
        # img = img/img.max()
        self.img.setImage(img)
        self.zSliceScrollBar.setMaximum(posData.SizeZ-1)

    def addAndConnectROI(self, roi):
        if roi not in self.ax1.items:
            self.ax1.addItem(roi.label)
            self.ax1.addItem(roi)

        roi.sigRegionChanged.connect(self.updateCurrentRoiShape)
        roi.sigRegionChangeFinished.connect(self.ROImovingFinished)
    
    def addAndConnectCropROIs(self):
        if self.startAction.isEnabled() or self.onlySelectingZslice:
            return

        posData = self.data[self.pos_i]
        for cropROI in posData.cropROIs:
            self.addAndConnectROI(cropROI)

    def removeCropROIs(self):
        if self.startAction.isEnabled() or self.onlySelectingZslice:
            return

        posData = self.data[self.pos_i]
        if posData.cropROIs is None:
            return

        for cropROI in posData.cropROIs:
            self.ax1.removeItem(cropROI.label)
            self.ax1.removeItem(cropROI)

            try:
                cropROI.sigRegionChanged.disconnect()
                cropROI.sigRegionChangeFinished.disconnect()
            except TypeError:
                pass


    def updateBkgrROIs(self):
        if self.startAction.isEnabled() or self.onlySelectingZslice:
            return

        posData = self.data[self.pos_i]
        for roi in posData.bkgrROIs:
            if roi not in self.ax1.items:
                self.ax1.addItem(roi.label)
                self.ax1.addItem(roi)
            roi.sigRegionChanged.connect(self.bkgrROIMoving)
            roi.sigRegionChangeFinished.connect(self.bkgrROImovingFinished)

    def removeBkgrROIs(self):
        if self.startAction.isEnabled() or self.onlySelectingZslice:
            return

        posData = self.data[self.pos_i]
        for roi in posData.bkgrROIs:
            self.ax1.removeItem(roi.label)
            self.ax1.removeItem(roi)

            try:
                roi.sigRegionChanged.disconnect()
                roi.sigRegionChangeFinished.disconnect()
            except TypeError:
                pass

    def init_attr(self):
        posData = self.data[0]
        self.navigateScrollbar.setEnabled(True)
        self.navigateScrollbar.setMinimum(1)
        if posData.SizeT > 1:
            self.navigateScrollbar.setMaximum(posData.SizeT)
        elif self.num_pos > 1:
            self.navigateScrollbar.setMaximum(self.num_pos)
        else:
            self.navigateScrollbar.setDisabled(True)
        self.navigateScrollbar.setValue(1)
        self.navigateScrollbar.valueChanged.connect(
            self.navigateScrollbarValueChanged
        )

    def navigateScrollbarValueChanged(self, value):
        if self.num_pos > 1:
            self.removeBkgrROIs()
            self.removeCropROIs()
            self.pos_i = value-1
            self.updatePos()
        else:
            self.frame_i = value-1
            self.update_img()

    @exception_handler
    def crop(self, data, posData, cropROI):
        x0, y0 = [int(round(c)) for c in cropROI.pos()]
        w, h = [int(round(c)) for c in cropROI.size()]
        if data.ndim == 4:
            croppedData = data[:, :, y0:y0+h, x0:x0+w]
        elif data.ndim == 3:
            croppedData = data[:, y0:y0+h, x0:x0+w]
        elif data.ndim == 2:
            croppedData = data[y0:y0+h, x0:x0+w]

        SizeZ = posData.SizeZ

        if posData.SizeZ > 1 and self.cropZtool is not None:
            idx = (posData.filename, 0)
            try:
                lower_z = int(posData.segmInfo_df.at[idx, 'crop_lower_z_slice'])
            except KeyError:
                lower_z = 0

            try:
                upper_z = int(posData.segmInfo_df.at[idx, 'crop_upper_z_slice'])
            except KeyError:
                upper_z = posData.SizeZ
            if data.ndim == 4:
                croppedData = data[:, lower_z:upper_z+1]
            elif data.ndim == 3:
                croppedData = data[lower_z:upper_z+1]
            SizeZ = upper_z-lower_z+1
        return croppedData, SizeZ

    def saveBkgrROIs(self, posData):
        if not posData.bkgrROIs:
            return

        ROIstates = [roi.saveState() for roi in posData.bkgrROIs]
        with open(posData.dataPrepBkgrROis_path, 'w') as json_fp:
            json.dump(ROIstates, json_fp)

    def saveBkgrData(self, posData):
        # try:
        #     os.remove(posData.dataPrepBkgrROis_path)
        # except FileNotFoundError:
        #     pass

        # If we crop we save data from background ROI for each bkgrROI
        for chName in posData.chNames:
            alignedFound = False
            tifFound = False
            for file in myutils.listdir(posData.images_path):
                filePath = os.path.join(posData.images_path, file)
                filenameNOext, _ = os.path.splitext(file)
                if file.endswith(f'{chName}_aligned.npz'):
                    aligned_filename = filenameNOext
                    aligned_filePath = filePath
                    alignedFound = True
                elif file.find(f'{chName}.tif') != -1:
                    tif_filename = filenameNOext
                    tif_path = filePath
                    tifFound = True

            if alignedFound:
                filename = aligned_filename
                chData = np.load(aligned_filePath)['arr_0']
            elif tifFound:
                filename = tif_filename
                chData = skimage.io.imread(tif_path)

            bkgrROI_data = {}
            for r, roi in enumerate(posData.bkgrROIs):
                xl, yt = [int(round(c)) for c in roi.pos()]
                w, h = [int(round(c)) for c in roi.size()]
                if not yt+h>yt or not xl+w>xl:
                    # Prevent 0 height or 0 width roi
                    continue
                is4D = posData.SizeT > 1 and posData.SizeZ > 1
                is3Dz = posData.SizeT == 1 and posData.SizeZ > 1
                is3Dt = posData.SizeT > 1 and posData.SizeZ == 1
                is2D = posData.SizeT == 1 and posData.SizeZ == 1
                if is4D:
                    bkgr_data = chData[:, :, yt:yt+h, xl:xl+w]
                elif is3Dz or is3Dt:
                    bkgr_data = chData[:, yt:yt+h, xl:xl+w]
                elif is2D:
                    bkgr_data = chData[yt:yt+h, xl:xl+w]
                bkgrROI_data[f'roi{r}_data'] = bkgr_data

            if bkgrROI_data:
                bkgr_data_fn = f'{filename}_bkgrRoiData.npz'
                bkgr_data_path = os.path.join(posData.images_path, bkgr_data_fn)
                print('---------------------------------')
                self.logger.info('Saving background data to:')
                self.logger.info(bkgr_data_path)
                print('*********************************')
                print('')
                np.savez_compressed(bkgr_data_path, **bkgrROI_data)

    def removeAllROIs(self, event):
        for posData in self.data:
            for roi in posData.bkgrROIs:
                self.ax1.removeItem(roi.label)
                self.ax1.removeItem(roi)

            posData.bkgrROIs = []
            try:
                os.remove(posData.dataPrepBkgrROis_path)
            except FileNotFoundError:
                pass

    def removeROI(self, event):
        posData = self.data[self.pos_i]
        try:
            posData.bkgrROIs.remove(self.roi_to_del)
        except Exception as e:
            posData.cropROIs.remove(self.roi_to_del)
        self.ax1.removeItem(self.roi_to_del.label)
        self.ax1.removeItem(self.roi_to_del)
        if not posData.bkgrROIs:
            try:
                os.remove(posData.dataPrepBkgrROis_path)
            except FileNotFoundError:
                pass
        else:
            self.saveBkgrROIs(posData)
    
    def gui_raiseContextMenuRoi(self, roi, event, is_bkgr_ROI=True):
        self.roi_to_del = roi
        self.roiContextMenu = QMenu(self)
        separator = QAction(self)
        separator.setSeparator(True)
        self.roiContextMenu.addAction(separator)
        if is_bkgr_ROI:
            action1 = QAction('Remove background ROI')
        else:
            action1 = QAction('Remove crop ROI')
        action1.triggered.connect(self.removeROI)
        self.roiContextMenu.addAction(action1)
        if is_bkgr_ROI:
            action2 = QAction('Remove ALL background ROIs')
            action2.triggered.connect(self.removeAllROIs)
            self.roiContextMenu.addAction(action2)
        self.roiContextMenu.exec_(event.screenPos())
    
    def gui_mousePressEventImg(self, event):
        posData = self.data[self.pos_i]
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

        if left_click:
            pg.ImageItem.mousePressEvent(self.img, event)

        x, y = event.pos().x(), event.pos().y()

        handleSize = 7
        # Check if right click on ROI
        for r, roi in enumerate(posData.bkgrROIs):
            x0, y0 = [int(c) for c in roi.pos()]
            w, h = [int(c) for c in roi.size()]
            x1, y1 = x0+w, y0+h
            clickedOnROI = (
                x>=x0-handleSize and x<=x1+handleSize
                and y>=y0-handleSize and y<=y1+handleSize
            )
            raiseContextMenuRoi = right_click and clickedOnROI
            dragRoi = left_click and clickedOnROI
            if raiseContextMenuRoi:
                self.gui_raiseContextMenuRoi(roi, event)
                return
            elif dragRoi:
                event.ignore()
                return

        if posData.cropROIs is None:
            return
        
        for c, cropROI in enumerate(posData.cropROIs):
            x0, y0 = [int(c) for c in cropROI.pos()]
            w, h = [int(c) for c in cropROI.size()]
            x1, y1 = x0+w, y0+h
            clickedOnROI = (
                x>=x0-handleSize and x<=x1+handleSize
                and y>=y0-handleSize and y<=y1+handleSize
            )
            dragRoi = left_click and clickedOnROI
            if dragRoi:
                event.ignore()
                return
            raiseContextMenuRoi = right_click and clickedOnROI and c>0
            if raiseContextMenuRoi:
                self.gui_raiseContextMenuRoi(cropROI, event, is_bkgr_ROI=False)
    
    def getAllChannelsPaths(self, posData):
        _zip = zip(posData.tif_paths, posData.all_npz_paths)
        for tif_path, npz_path in _zip:
            if self.align:
                uncropped_data = np.load(npz_path)['arr_0']
            else:
                uncropped_data = skimage.io.imread(tif_path)
            
            yield uncropped_data, npz_path, tif_path
    
    def saveCroppedChannel(self, cropped_data, npz_path, tif_path):        
        if self.align:
            self.logger.info(f'Saving: {npz_path}')
            temp_npz = self.getTempfilePath(npz_path)
            np.savez_compressed(temp_npz, cropped_data)
            self.moveTempFile(temp_npz, npz_path)

        self.logger.info(f'Saving: {tif_path}')
        temp_tif = self.getTempfilePath(tif_path)
        myutils.imagej_tiffwriter(temp_tif, cropped_data)
        self.moveTempFile(temp_tif, tif_path)
    
    def saveCroppedSegmData(self, posData, segm_npz_path, cropROI):
        if not posData.segmFound:
            return
        self.logger.info(f'Saving: {segm_npz_path}')
        croppedSegm, _ = self.crop(posData.segm_data, posData, cropROI)
        temp_npz = self.getTempfilePath(segm_npz_path)
        np.savez_compressed(temp_npz, croppedSegm)
        self.moveTempFile(temp_npz, segm_npz_path)
    
    def correctAcdcDfCrop(self, posData, acdc_output_csv_path, cropROI):
        try:
            # Correct acdc_df if present and save
            if posData.acdc_df is not None:
                x0, y0 = [int(round(c)) for c in cropROI.pos()]
                self.logger.info(f'Saving: {acdc_output_csv_path}')
                df = posData.acdc_df
                df['x_centroid'] -= x0
                df['y_centroid'] -= y0
                try:
                    df.to_csv(acdc_output_csv_path)
                except PermissionError:
                    self.permissionErrorCritical(acdc_output_csv_path)
                    df.to_csv(acdc_output_csv_path)
        except Exception as e:
            pass
    
    def saveSingleCrop(self, posData, cropROI):
        # Save channels (npz AND tif)
        _iter = self.getAllChannelsPaths(posData)
        for uncropped_data, npz_path, tif_path in _iter:
            cropped_data, _ = self.crop(uncropped_data, posData, cropROI)
            self.saveCroppedChannel(cropped_data, npz_path, tif_path)

        self.saveCroppedSegmData(posData, posData.segm_npz_path, cropROI)
        self.correctAcdcDfCrop(posData, posData.acdc_output_csv_path, cropROI)
    
    def saveCroppedData(self, posData):
        # Get metadata from tif
        with TiffFile(posData.tif_path) as tif:
            metadata = tif.imagej_metadata

        if len(posData.cropROIs) == 1:
            self.saveSingleCrop(posData, posData.cropROIs[0])
        else:
            self.saveMultiCrops(posData)

        self.logger.info(f'{posData.pos_foldername} saved!')
        print(f'--------------------------------')
        print('')
    
    def saveMultiCrops(self, posData):
        txt = html_utils.paragraph("""
            You will now be asked to <b>choose a location</b> where to save multiple 
            Positions from each crop ROI.
        """)
        msg = widgets.myMessageBox()
        msg.information(
            self, 'Select folder where to save multiple crops', txt
        )
        if msg.cancel:
            self.logger.info('Cropping process cancelled.')
            return
        
        parentSubPosPath = getexistingdirectory(
            parent=self,
            caption='Select folder where to save multiple crops', 
            basedir=posData.pos_path
        )
        if not parentSubPosPath:
            self.logger.info('Cropping process cancelled.')
            return
        
        currentSubPosFolders = myutils.get_pos_foldernames(parentSubPosPath)
        currentSubPosNumbers = [
            int(pos.split('_')[-1]) for pos in currentSubPosFolders
        ]
        startPosNumber = max(currentSubPosNumbers, default=0) + 1
        for p, cropROI in enumerate(posData.cropROIs):
            subPosFolder = f'Position_{p+startPosNumber}'
            subPosFolderPath = os.path.join(parentSubPosPath, subPosFolder)
            subImagesPath = os.path.join(subPosFolderPath, 'Images')
            os.makedirs(subImagesPath)
            
            _iter = self.getAllChannelsPaths(posData)
            for uncropped_data, npz_path, tif_path in _iter:
                cropped_data, _ = self.crop(uncropped_data, posData, cropROI)
                npz_filename = os.path.basename(npz_path)
                tif_filename = os.path.basename(tif_path)
                sub_npz_filepath = os.path.join(subImagesPath, npz_filename)
                sub_tif_filepath = os.path.join(subImagesPath, tif_filename)
                self.saveCroppedChannel(
                    cropped_data, sub_npz_filepath, sub_tif_filepath
                )
            
            segm_filename = os.path.basename(posData.segm_npz_path)
            sub_segm_filepath = os.path.join(subImagesPath, segm_filename)
            self.saveCroppedSegmData(posData, sub_segm_filepath, cropROI)
            
            acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
            acdc_df_filepath = os.path.join(subImagesPath, acdc_df_filename)
            self.correctAcdcDfCrop(posData, acdc_df_filepath, cropROI)
            
            try:
                df_roi = posData.dataPrep_ROIcoords.loc[[p]]
                df_roi_filename = os.path.basename(posData.dataPrepROI_coords_path)
                df_roi_filepath = os.path.join(subImagesPath, df_roi_filename)
                df_roi.to_csv(df_roi_filepath)
            except IndexError:
                pass
            
            for file in myutils.listdir(posData.images_path):
                copy_file = (
                    file.endswith('metadata.csv')
                    or file.endswith('bkgrRoiData.npz')
                    or file.endswith('dataPrep_bkgrROIs.json')
                    or file.endswith('segmInfo.csv')
                )
                if not copy_file:
                    continue
                sub_filepath = os.path.join(subImagesPath, file)
                if os.path.exists(sub_filepath):
                    continue
                
                src_filepath = os.path.join(posData.images_path, file)
                shutil.copyfile(src_filepath, sub_filepath)
    
    def saveROIcoords(self, doCrop, posData):
        dfs = []
        keys = []
        for c, cropROI in enumerate(posData.cropROIs):
            x0, y0 = [int(round(c)) for c in cropROI.pos()]
            w, h = [int(round(c)) for c in cropROI.size()]

            Y, X = self.img.image.shape
            x1, y1 = x0+w, y0+h

            x0 = x0 if x0>0 else 0
            y0 = y0 if y0>0 else 0
            x1 = x1 if x1<X else X
            y1 = y1 if y1<Y else Y

            if x0<=0 and y0<=0 and x1>=X and y1>=Y:
                # ROI coordinates are the exact image shape. No need to save them
                continue
            
            keys.append(c)

            description = ['x_left', 'x_right', 'y_top', 'y_bottom', 'cropped']
            values = [x0, x1, y0, y1, int(doCrop)]
            df_roi = (
                pd.DataFrame({'description': description, 'value': values})
                .set_index('description')
            )
            
            dfs.append(df_roi)
        
        if not dfs:
            return
        
        df = pd.concat(dfs, keys=keys, names=['roi_id'])

        self.logger.info(
                f'Saving ROI coords '
                f'to "{posData.dataPrepROI_coords_path}"'
            )
        try:
            df.to_csv(posData.dataPrepROI_coords_path)
        except PermissionError:
            self.permissionErrorCritical(posData.dataPrepROI_coords_path)
            df.to_csv(posData.dataPrepROI_coords_path)
        
        posData.dataPrep_ROIcoords = df

    def openCropZtool(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.cropZtool = apps.QCropZtool(posData.SizeZ, parent=self)
            self.cropZtool.sigClose.connect(self.cropZtoolClosed)
            self.cropZtool.sigZvalueChanged.connect(self.cropZtoolvalueChanged)
            self.cropZtool.sigCrop.connect(self.crop_cb)
            self.cropZtool.show()
        else:
            self.cropZtool.close()
            self.cropZtool = None
            # Restore original z-slice
            df = posData.segmInfo_df
            idx = (posData.filename, self.frame_i)
            z = posData.segmInfo_df.at[idx, 'z_slice_used_dataPrep']
            self.zSliceScrollBar.setValue(z)

    def cropZtoolvalueChanged(self, whichZ, z):
        self.zSliceScrollBar.valueChanged.disconnect()
        self.zSliceScrollBar.setValue(z)
        self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
        posData = self.data[self.pos_i]
        img = self.getImage(posData, posData.img_data, self.frame_i, force_z=z)
        self.img.setImage(img)
        df = posData.segmInfo_df
        for frame_i in range(posData.SizeT):
            idx = (posData.filename, frame_i)
            posData.segmInfo_df[f'crop_{whichZ}_z_slice'] = z

    def cropZtoolReset(self):
        posData = self.data[self.pos_i]
        self.cropZtool.sigZvalueChanged.disconnect()
        self.cropZtool.updateScrollbars(0, posData.SizeZ)
        self.cropZtool.sigZvalueChanged.connect(self.cropZtoolvalueChanged)

    def updateCropZtool(self):
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return
        if self.cropZtool is None:
            return

        idx = (posData.filename, 0)
        try:
            lower_z = int(posData.segmInfo_df.at[idx, 'crop_lower_z_slice'])
        except KeyError:
            lower_z = 0

        try:
            upper_z = int(posData.segmInfo_df.at[idx, 'crop_upper_z_slice'])
        except KeyError:
            upper_z = posData.SizeZ

        self.cropZtool.sigZvalueChanged.disconnect()
        self.cropZtool.updateScrollbars(lower_z, upper_z)
        self.cropZtool.sigZvalueChanged.connect(self.cropZtoolvalueChanged)

    def cropZtoolClosed(self):
        self.cropZtool = None
        posData = self.data[self.pos_i]
        idx = (posData.filename, self.frame_i)
        z = posData.segmInfo_df.at[idx, 'z_slice_used_dataPrep']
        self.zSliceScrollBar.setSliderPosition(z)
        self.cropZaction.toggled.disconnect()
        self.cropZaction.setChecked(False)
        self.cropZaction.toggled.connect(self.openCropZtool)

    def getCroppedData(self):
        for p, posData in enumerate(self.data):
            self.saveBkgrROIs(posData)

            if posData.SizeZ > 1:
                # Save segmInfo
                posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

            # Get crop shape and print it
            data = posData.img_data
            
            allCropsData = []
            for cropROI in posData.cropROIs:
                croppedData, SizeZ = self.crop(data, posData, cropROI)
                allCropsData.append(croppedData)

            croppedShapes = [cropped.shape for cropped in allCropsData]
            isCropped = any([shape != data.shape for shape in croppedShapes])
            isCroppingCancelled = False
            if isCropped != data.shape:
                if p == 0:
                    isCroppingCancelled = self.askCropping(
                        data.shape, croppedShapes
                    )
                    doCrop = not isCroppingCancelled
                else:
                    doCrop = True
            else:
                doCrop = False

            if isCroppingCancelled:
                self.cropAction.setEnabled(True)
                txt = ('Cropping cancelled.')
                self.titleLabel.setText(txt, color='r')
                yield None
            elif croppedData.shape == data.shape:
                self.cropAction.setEnabled(True)
                txt = ('Crop ROI has same shape of the image --> no need to crop. Process stopped.')
                self.titleLabel.setText(txt, color='r')
                yield 'continue'
            
            yield croppedShapes, posData, SizeZ, doCrop
    
    @exception_handler
    def crop_cb(self):
        # msg = QMessageBox()
        # doSave = msg.question(
        #     self, 'Save data?', 'Do you want to save?',
        #     msg.Yes | msg.No
        # )
        # if doSave == msg.No:
        #     return

        self.cropAction.setDisabled(True)
        for cropInfo in self.getCroppedData():
            if cropInfo is None:
                # Process cancelled by the user
                return
            
            if cropInfo == 'continue':
                continue
            
            croppedShapes, posData, SizeZ, doCrop = cropInfo
            
            if SizeZ != posData.SizeZ:
                # Update metadata with cropped SizeZ
                posData.metadata_df.at['SizeZ', 'values'] = SizeZ
                posData.metadata_df.to_csv(posData.metadata_csv_path)

            self.logger.info(f'Cropping {posData.relPath}...')
            self.titleLabel.setText(
                'Cropping... (check progress in the terminal)',
                color='w')

            croppedShapesFormat = [f'  --> {shape}' for shape in croppedShapes]
            croppedShapesFormat = '\n'.join(croppedShapesFormat)
            self.logger.info(f'Cropped data shape:\n{croppedShapesFormat}')
            self.saveROIcoords(doCrop, posData)

            self.logger.info('Saving background data...')
            self.saveBkgrData(posData)

            self.saveCroppedData(posData)
        
        for posData in self.data:
            self.disconnectROIs(posData)

        txt = (
            'Saved! You can close the program or load another position.'
        )
        self.titleLabel.setText(txt, color='g')
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(txt.replace('! ', '!<br><br>'))
        msg.information(self, 'Data prep done', txt)
    
    def removeAllHandles(self, roi):
        for handle in roi.handles:
            item = handle['item']
            item.disconnectROI(roi)
            if len(item.rois) == 0 and roi.scene() is not None:
                roi.scene().removeItem(item)
        roi.handles = []
        roi.stateChanged()
    
    def disconnectROIs(self, posData):
        for cropROI in posData.cropROIs:
            cropROI.disconnect()
            cropROI.translatable = False
            cropROI.resizable = False
            self.removeAllHandles(cropROI)

        for roi in posData.bkgrROIs:
            roi.disconnect()
            roi.translatable = False
            roi.resizable = False
            roi.removable = False

            self.removeAllHandles(roi)
        
        self.logger.info('ROIs disconnected.')

    def permissionErrorCritical(self, path):
        msg = QMessageBox()
        msg.critical(
            self, 'Permission denied',
            f'The below file is open in another app (Excel maybe?).\n\n'
            f'{path}\n\n'
            'Close file and then press "Ok".',
            msg.Ok
        )


    def askCropping(self, dataShape, croppedShapes):
        if len(self.data) > 1:
            info_text = (f"""
                You are about to <b>crop</b> data.<br><br>
            """)
        else:
            info_text = (f"""
                You are about to <b>crop</b> data from shape {dataShape} 
                to the following shapes: 
                {html_utils.to_list(croppedShapes, ordered=True)}
            """)
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            {info_text}
            Saving cropped data <b>cannot be undone</b>.<br><br>
            Do you want to crop or simply save the ROI coordinates<br>
            for the segmentation step?
        """)
        _, yesButton = msg.warning(
            self, 'Crop?', txt, 
            buttonsTexts=('Cancel', 'Yes, crop please.')
        )
        return msg.cancel

    def getDefaultROI(self, shrinkFactor=1):
        Y, X = self.img.image.shape
        w, h = int(X*shrinkFactor), int(Y*shrinkFactor)

        xc, yc = int(round(X/2)), int(round(Y/2))
        # yt, xl = int(round(xc-w/2)), int(round(yc-h/2))
        yt, xl = 0, 0

        # Add ROI Rectangle
        cropROI = pg.ROI(
            [xl, yt], [w, h],
            rotatable=False,
            removable=False,
            pen=pg.mkPen(color='r'),
            maxBounds=QRectF(QRect(0,0,X,Y)),
            scaleSnap=True,
            translateSnap=True
        )
        return cropROI

    def setROIprops(self, roi):
        xl, yt = [int(round(c)) for c in roi.pos()]

        roi.handleSize = 7
        roi.label = pg.TextItem('ROI', color='r')
        roi.label.setFont(self.roiLabelFont)
        # hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yt)

        ## handles scaling horizontally around center
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        roi.addScaleHandle([0, 0.5], [1, 0.5])

        ## handles scaling vertically from opposite edge
        roi.addScaleHandle([0.5, 0], [0.5, 1])
        roi.addScaleHandle([0.5, 1], [0.5, 0])

        ## handles scaling both vertically and horizontally
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])

    def init_data(self, user_ch_file_paths, user_ch_name):
        # Iterate pos and load_data
        data = []
        for f, file_path in enumerate(tqdm(user_ch_file_paths, ncols=100)):
            try:
                posData = load.loadData(file_path, user_ch_name, QParent=self)
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadImgData()
                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_shifts=True,
                    loadSegmInfo=True,
                    load_dataPrep_ROIcoords=True,
                    load_delROIsInfo=False,
                    load_bkgr_data=False,
                    loadBkgrROIs=True,
                    load_last_tracked_i=False,
                    load_metadata=True,
                    getTifPath=True
                )

                # If data was cropped then dataPrep_ROIcoords are useless
                if posData.isCropped():
                    posData.dataPrep_ROIcoords = None

                posData.loadAllImgPaths()
                if f==0 and not self.metadataAlreadyAsked:
                    proceed = posData.askInputMetadata(
                        self.num_pos,
                        ask_SizeT=self.num_pos==1,
                        ask_TimeIncrement=False,
                        ask_PhysicalSizes=False,
                        save=True,
                        askSegm3D=False
                    )
                    self.isSegm3D = posData.isSegm3D
                    self.SizeT = posData.SizeT
                    self.SizeZ = posData.SizeZ
                    if not proceed:
                        self.titleLabel.setText(
                            'File --> Open or Open recent to start the process',
                            color='w')
                        return False
                    self.AutoPilotProfile.storeOkAskInputMetadata()
                else:
                    posData.isSegm3D = self.isSegm3D
                    posData.SizeT = self.SizeT
                    if self.SizeZ > 1:
                        # In case we know we are loading single 3D z-stacks
                        # we alwways use the third dimensins as SizeZ because
                        # SizeZ in some positions might be different than
                        # first loaded pos
                        SizeZ = posData.img_data.shape[-3]
                        posData.SizeZ = SizeZ
                    else:
                        posData.SizeZ = 1
                    if self.SizeT > 1:
                        posData.SizeT = self.SizeT
                    else:
                        posData.SizeT = 1
                    posData.saveMetadata()
            except AttributeError:
                print('')
                print('====================================')
                traceback.print_exc()
                print('====================================')
                print('')
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return False

            if posData is None:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return False

            img_shape = posData.img_data.shape
            self.num_frames = posData.SizeT
            self.user_ch_name = user_ch_name
            SizeT = posData.SizeT
            SizeZ = posData.SizeZ
            if f==0:
                self.logger.info(f'Data shape = {img_shape}')
                self.logger.info(f'Number of frames = {SizeT}')
                self.logger.info(f'Number of z-slices per frame = {SizeZ}')
            data.append(posData)

            if SizeT>1 and self.num_pos>1:
                path = os.path.normpath(file_path)
                path_li = path.split(os.sep)
                rel_path = f'.../{"/".join(path_li[-3:])}'
                msg = QMessageBox()
                msg.critical(
                    self, 'Multiple Pos loading not allowed.',
                    f'The file {rel_path} has multiple frames over time.\n\n'
                    'Loading multiple positions that contain frames over time '
                    'is not allowed.\n\n'
                    'To analyse frames over time load one position at the time',
                    msg.Ok
                )
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return False

        self.data = data
        self.init_segmInfo_df()
        self.init_attr()

        return True

    def init_segmInfo_df(self):
        self.pos_i = 0
        self.frame_i = 0
        for posData in self.data:
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
                posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

        posData = self.data[0]
        try:
            self.zSliceScrollBar.valueChanged.disconnect()
            self.zProjComboBox.currentTextChanged.disconnect()
        except Exception as e:
            pass
        if posData.SizeZ > 1:
            self.z_label.setDisabled(False)
            self.zSliceScrollBar.setDisabled(False)
            self.zProjComboBox.setDisabled(False)
            self.zSliceScrollBar.setMaximum(posData.SizeZ-1)
            self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
            if posData.SizeT > 1:
                self.interpAction.setEnabled(True)
            self.ZbackAction.setEnabled(True)
            self.ZforwAction.setEnabled(True)
            df = posData.segmInfo_df
            idx = (posData.filename, self.frame_i)
            how = posData.segmInfo_df.at[idx, 'which_z_proj']
            self.zProjComboBox.setCurrentText(how)
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.zProjComboBox.setDisabled(True)
            self.z_label.setDisabled(True)

    def update_z_slice(self, z):
        if self.zProjComboBox.currentText() == 'single z-slice':
            posData = self.data[self.pos_i]
            df = posData.segmInfo_df
            idx = (posData.filename, self.frame_i)
            posData.segmInfo_df.at[idx, 'z_slice_used_dataPrep'] = z
            posData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
            self.update_img()
            posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)


    def updateZproj(self, how):
        posData = self.data[self.pos_i]
        for frame_i in range(self.frame_i, posData.SizeT):
            df = posData.segmInfo_df
            idx = (posData.filename, self.frame_i)
            posData.segmInfo_df.at[idx, 'which_z_proj'] = how
            posData.segmInfo_df.at[idx, 'which_z_proj_gui'] = how
        if how == 'single z-slice':
            self.zSliceScrollBar.setDisabled(False)
            self.z_label.setStyleSheet('color: black')
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.z_label.setStyleSheet('color: gray')
            self.update_img()

        # Apply same z-proj to future pos
        if posData.SizeT == 1:
            for posData in self.data[self.pos_i+1:]:
                idx = (posData.filename, self.frame_i)
                posData.segmInfo_df.at[idx, 'which_z_proj'] = how

        self.save_segmInfo_df_pos()

    def save_segmInfo_df_pos(self):
        # Launch a separate thread to save to csv and keep gui responsive
        self.thread = QThread()
        self.worker = toCsvWorker()
        self.worker.setData(self.data)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def useSameZ_fromHereBack(self, event):
        how = self.zProjComboBox.currentText()
        posData = self.data[self.pos_i]
        df = posData.segmInfo_df
        z = df.at[(posData.filename, self.frame_i), 'z_slice_used_dataPrep']
        if posData.SizeT > 1:
            for i in range(0, self.frame_i):
                df.at[(posData.filename, i), 'z_slice_used_dataPrep'] = z
                df.at[(posData.filename, i), 'z_slice_used_gui'] = z
                df.at[(posData.filename, i), 'which_z_proj'] = how
            posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
        elif posData.SizeZ > 1:
            for _posData in self.data[:self.pos_i]:
                df = _posData.segmInfo_df
                df.at[(_posData.filename, 0), 'z_slice_used_dataPrep'] = z
                df.at[(_posData.filename, 0), 'z_slice_used_gui'] = z
                df.at[(_posData.filename, 0), 'which_z_proj'] = how
            self.save_segmInfo_df_pos()

    def useSameZ_fromHereForw(self, event):
        how = self.zProjComboBox.currentText()
        posData = self.data[self.pos_i]
        df = posData.segmInfo_df
        z = df.at[(posData.filename, self.frame_i), 'z_slice_used_dataPrep']
        if posData.SizeT > 1:
            for i in range(self.frame_i, posData.SizeT):
                df.at[(posData.filename, i), 'z_slice_used_dataPrep'] = z
                df.at[(posData.filename, i), 'z_slice_used_gui'] = z
                df.at[(posData.filename, i), 'which_z_proj'] = how
            posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
        elif posData.SizeZ > 1:
            for _posData in self.data[self.pos_i:]:
                df = _posData.segmInfo_df
                df.at[(_posData.filename, 0), 'z_slice_used_dataPrep'] = z
                df.at[(_posData.filename, 0), 'z_slice_used_gui'] = z
                df.at[(_posData.filename, 0), 'which_z_proj'] = how

            self.save_segmInfo_df_pos()

    def interp_z(self, event):
        posData = self.data[self.pos_i]
        df = posData.segmInfo_df
        x0, z0 = 0, df.at[(posData.filename, 0), 'z_slice_used_dataPrep']
        x1 = self.frame_i
        z1 = df.at[(posData.filename, x1), 'z_slice_used_dataPrep']
        f = scipy.interpolate.interp1d([x0, x1], [z0, z1])
        xx = np.arange(0, self.frame_i)
        zz = np.round(f(xx)).astype(int)
        for i in range(self.frame_i):
            df.at[(posData.filename, i), 'z_slice_used_dataPrep'] = zz[i]
            df.at[(posData.filename, i), 'z_slice_used_gui'] = zz[i]
            df.at[(posData.filename, i), 'which_z_proj'] = 'single z-slice'
        posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

    @exception_handler
    def prepData(self, event):
        self.titleLabel.setText(
            'Prepping data... (check progress in the terminal)',
            color='w')
        doZip = False
        for p, posData in enumerate(self.data):
            self.startAction.setDisabled(True)
            nonTifFound = (
                any([npz is not None for npz in posData.npz_paths]) or
                any([npy is not None for npy in posData.npy_paths]) or
                posData.segmFound
            )
            imagesPath = posData.images_path
            zipPath = f'{imagesPath}.zip'
            if nonTifFound and p==0:
                txt = (
                    'Additional <b>NON-tif files detected.</b><br><br>'
                    'The requested experiment folder <b>already contains .npy '
                    'or .npz files</b> '
                    'most likely from previous analysis runs.<br><br>'
                    'To <b>avoid data losses</b> we recommend zipping the '
                    '"Images" folder.<br><br>'
                    'If everything looks fine after prepping the data, '
                    'you can manually '
                    'delete the zip archive.<br><br>'
                    'Do you want to <b>automatically zip now?</b><br><br>'
                    'PS: Zip archive location:<br><br>'
                    f'{zipPath}'
                )
                txt = html_utils.paragraph(txt)
                msg = widgets.myMessageBox()
                _, yes, no = msg.warning(
                   self, 'NON-Tif data detected!', txt,
                   buttonsTexts=('Cancel', 'Yes', 'No')
                )
                if msg.cancel:
                    self.cropAction.setEnabled(True)
                    self.titleLabel.setText('Process aborted', color='w')
                    break
                if yes == msg.clickedButton:
                    doZip = True
            if doZip:
                self.logger.info(f'Zipping Images folder: {zipPath}')
                shutil.make_archive(imagesPath, 'zip', imagesPath)
            success = self.alignData(self.user_ch_name, posData)
            if not success:
                self.titleLabel.setText('Data prep cancelled.', color='r')
                break
            if posData.SizeZ>1:
                posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
        else:
            # For loop did not break, proceed with the rest
            self.update_img()
            self.logger.info('Done.')
            self.addROIs()
            self.saveROIcoords(False, self.data[self.pos_i])
            self.saveBkgrROIs(self.data[self.pos_i])
            self.cropAction.setEnabled(True)
            if posData.SizeZ>1:
                self.cropZaction.setEnabled(True)
            self.titleLabel.setText(
                'Data successfully prepped. You can now crop the images or '
                'close the program',
                color='w')

    def setStandardRoiShape(self, text):
        posData = self.data[self.pos_i]
        if posData.cropROIs is None:
            return
        if len(posData.cropROIs)>1:
            return
        Y, X = posData.img_data.shape[-2:]
        m = re.findall(r'(\d+)x(\d+)', text)
        w, h = int(m[0][0]), int(m[0][1])
        # xc, yc = int(round(X/2)), int(round(Y/2))
        # yt, xl = int(round(xc-w/2)), int(round(yc-h/2))
        posData.cropROIs[0].setPos([0, 0])
        posData.cropROIs[0].setSize([w, h])

    def addROIs(self):
        Y, X = self.img.image.shape

        max_size = round(int(np.log2(min([Y, X])/16)))
        items = [f'{16*(2**i)}x{16*(2**i)}' for i in range(1, max_size+1)]
        items.append(f'{X}x{Y}')
        self.ROIshapeComboBox.clear()
        self.ROIshapeComboBox.addItems(items)
        self.ROIshapeComboBox.setCurrentText(items[-1])

        for posData in self.data:
            posData.cropROIs = []
            if posData.dataPrep_ROIcoords is None:
                cropROI = self.getDefaultROI()
                self.setROIprops(cropROI)
                posData.cropROIs.append(cropROI)
            else:
                df = posData.dataPrep_ROIcoords
                grouped = df.groupby(level=0)
                for roi_id, df_roi in grouped:
                    df_roi = df_roi.loc[roi_id]
                    xl = df_roi.at['x_left', 'value']
                    yt = df_roi.at['y_top', 'value']
                    w = df_roi.at['x_right', 'value'] - xl
                    h = df_roi.at['y_bottom', 'value'] - yt
                    cropROI = pg.ROI(
                        [xl, yt], [w, h],
                        rotatable=False,
                        removable=False,
                        pen=pg.mkPen(color='r'),
                        maxBounds=QRectF(QRect(0,0,X,Y)),
                        scaleSnap=True,
                        translateSnap=True
                    )
                    self.setROIprops(cropROI)
                    posData.cropROIs.append(cropROI)
             
        self.addAndConnectCropROIs()

        try:
            self.ROIshapeComboBox.currentTextChanged.disconnect()
        except Exception as e:
            pass
        self.ROIshapeComboBox.currentTextChanged.connect(
            self.setStandardRoiShape
        )

        self.addBkrgRoiActon.setDisabled(False)
        self.addCropRoiActon.setDisabled(False)

        for posData in self.data:
            if not posData.bkgrROIs and not posData.bkgrDataExists:
                bkgrROI = self.getDefaultBkgrROI()
                self.setBkgrROIprops(bkgrROI)
                posData.bkgrROIs.append(bkgrROI)
            else:
                for bkgrROI in posData.bkgrROIs:
                    self.setBkgrROIprops(bkgrROI)

        self.updateBkgrROIs()

    def getDefaultBkgrROI(self):
        Y, X = self.img.image.shape
        xRange, yRange = self.ax1.viewRange()
        xl, yt = abs(xRange[0]), abs(yRange[0])
        w, h = int(X/8), int(Y/8)
        bkgrROI = pg.ROI(
            [xl, yt], [w, h],
            rotatable=False,
            removable=False,
            pen=pg.mkPen(color=(255,255,255)),
            maxBounds=QRectF(QRect(0,0,X,Y)),
            scaleSnap=True,
            translateSnap=True
        )
        return bkgrROI

    def setBkgrROIprops(self, bkgrROI):
        bkgrROI.handleSize = 7

        xl, yt = [int(round(c)) for c in bkgrROI.pos()]
        bkgrROI.label = pg.TextItem('Bkgr. ROI', color=(255,255,255))
        bkgrROI.label.setFont(self.roiLabelFont)
        # hLabel = bkgrROI.label.rect().bottom()
        bkgrROI.label.setPos(xl, yt)

        ## handles scaling horizontally around center
        bkgrROI.addScaleHandle([1, 0.5], [0, 0.5])
        bkgrROI.addScaleHandle([0, 0.5], [1, 0.5])

        ## handles scaling vertically from opposite edge
        bkgrROI.addScaleHandle([0.5, 0], [0.5, 1])
        bkgrROI.addScaleHandle([0.5, 1], [0.5, 0])

        ## handles scaling both vertically and horizontally
        bkgrROI.addScaleHandle([1, 1], [0, 0])
        bkgrROI.addScaleHandle([0, 0], [1, 1])
        bkgrROI.addScaleHandle([1, 0], [0, 1])
        bkgrROI.addScaleHandle([0, 1], [1, 0])

        # bkgrROI.sigRegionChanged.connect(self.bkgrROIMoving)
        # bkgrROI.sigRegionChangeFinished.connect(self.bkgrROImovingFinished)

    def addCropROI(self):
        cropROI = self.getDefaultROI(shrinkFactor=0.5)
        self.setROIprops(cropROI)
        posData = self.data[self.pos_i]
        posData.cropROIs.append(cropROI)
        self.addAndConnectROI(cropROI)

    def addDefaultBkgrROI(self, checked=False):
        bkgrROI = self.getDefaultBkgrROI()
        self.setBkgrROIprops(bkgrROI)
        self.ax1.addItem(bkgrROI)
        self.ax1.addItem(bkgrROI.label)
        posData = self.data[self.pos_i]
        posData.bkgrROIs.append(bkgrROI)
        self.saveBkgrROIs(posData)

        bkgrROI.sigRegionChanged.connect(self.bkgrROIMoving)
        bkgrROI.sigRegionChangeFinished.connect(self.bkgrROImovingFinished)

    def bkgrROIMoving(self, roi):
        roi.setPen(color=(255,255,0))
        roi.label.setColor((255,255,0))
        # roi.label.setText(txt, color=(255,255,0), size=self.roiLabelSize)
        xl, yt = [int(round(c)) for c in roi.pos()]
        # hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yt)

    def bkgrROImovingFinished(self, roi):
        txt = roi.label.toPlainText()
        roi.setPen(color=(255,255,255))
        roi.label.setColor((255,255,255))
        # roi.label.setText(txt, color=(150,150,150), size=self.roiLabelSize)
        posData = self.data[self.pos_i]
        idx = posData.bkgrROIs.index(roi)
        posData.bkgrROIs[idx] = roi
        self.saveBkgrROIs(posData)

    def ROImovingFinished(self, roi):
        txt = roi.label.toPlainText()
        roi.setPen(color='r')
        roi.label.setColor('r')
        # roi.label.setText(txt, color='r', size=self.roiLabelSize)
        self.saveROIcoords(False, self.data[self.pos_i])

    def updateCurrentRoiShape(self, roi):
        roi.setPen(color=(255,255,0))
        roi.label.setColor((255,255,0))
        # roi.label.setText('ROI', color=(255,255,0), size=self.roiLabelSize)
        xl, yt = [int(round(c)) for c in roi.pos()]
        # hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yt)
        w, h = [int(round(c)) for c in roi.size()]
        self.ROIshapeLabel.setText(f'   Current ROI shape: {w} x {h}')
    
    def warnZeroPaddingAlignment(self):
        txt = html_utils.paragraph("""
            To align the frames, Cell-ACDC needs to <b>shift the images</b> 
            according to the shifts computed by the alignment algorithm.<br><br>
            This result in <b>padding</b> of the shifted rows and columns with 
            <b>0-valued pixels</b>.<br><br>
            For this reason, we <b>recommend cropping</b> 
            after aligning (you can do it in this GUI after the alignment 
            is completed).<br><br>
            If you choose to not crop, these 0-valued pixels will be automatically 
            excluded from the automatic background estimation.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.information(
            self, 'Padding alignment shifts', txt, 
            buttonsTexts=('Cancel', 'Ok')
        )
        if msg.cancel:
            return False
        return True

    def alignData(self, user_ch_name, posData):
        """
        NOTE: if self.num_pos > 1 then we simply save a ".._aligned.npz"
        file without alignment

        Alignemnt routine. Alignemnt is based on the data contained in the
        .tif file of the channel selected by the user (e.g. "phase_contr").
        Next, using the shifts calculated when aligning the channel selected
        by the user, it will align all the other channels, always starting from
        the data contained in the .tif files.

        In the end, aligned data will be saved to both the .tif file and a
        "_aligned.npz" file. The shifts will be saved to "align_shift.npy" file.

        Alignemnt is performed only if needed and requested by the user:

        1. If the "_aligned.npz" file does NOT exist AND the "align_shift.npy"
        file does NOT exist then alignment is performed with the function
        skimage.registration.phase_cross_correlation

        2. If the "_aligned.npz" file does NOT exist AND the "align_shift.npy"
        file does exist then alignment is performed with the saved shifts

        3. If the "_aligned.npz" file does exist AND the "align_shift.npy"
        file does NOT exist then alignment is performed AGAIN with the function
        skimage.registration.phase_cross_correlation

        4. If the "_aligned.npz" file does exist AND the "align_shift.npy"
        file does exist no alignment is needed.

        NOTE on the segmentation mask. If the system detects a "_segm.npz" file
        AND alignmnet was performed, we need to be careful with aligning the
        segm file. Segmentation files were most likely already aligned
        (although this cannot be detected) so aligning them again will probably
        misAlign them. It is responsibility of the user to choose wisely.
        However, if alignment is performed AGAIN, the system will zip the
        "Images" folder first to avoid data losses or corruption.

        In general, it should be fine performing alignment again if the user
        deletes the "align_shift.npy" file ONLY if also the .tif files are
        already aligned. If the .tif files are not aligned and we need to
        perform alignment again then the segmentation mask is invalid. In this
        case we should align starting from "_aligned.npz" file but it is
        not implemented yet.
        """

        # Get metadata from tif
        with TiffFile(posData.tif_path) as tif:
            metadata = tif.imagej_metadata

        align = True
        if posData.loaded_shifts is None and posData.SizeT > 1:
            msg = widgets.myMessageBox(showCentered=False)
            if user_ch_name:
                txt = html_utils.paragraph(f"""
                    Do you want to <b>align ALL channels</b> based on 
                    <code>{user_ch_name}</code> channel?<br><br>
                    NOTE: If you don't know what to choose, we <b>reccommend</b> 
                    aligning.
                """)
            else:
                txt = html_utils.paragraph(f"""
                    Do you want to <b>align</b> the frames over time?<br><br>
                    NOTE: If you don't know what to choose, we <b>reccommend</b> 
                    aligning.
                """)
            _, yesButton, noButton = msg.question(
                self, 'Align frames?', txt,
                buttonsTexts=('Cancel', 'Yes', 'No')
            )
            if msg.cancel:
                return False
            elif msg.clickedButton == noButton:
                align = False
                # Create 0, 0 shifts to perform 0 alignment
                posData.loaded_shifts = np.zeros((self.num_frames,2), int)
        elif posData.SizeT == 1:
            align = False
            # Create 0, 0 shifts to perform 0 alignment
            posData.loaded_shifts = np.zeros((self.num_frames,2), int)
        
        if align:
            proceed = self.warnZeroPaddingAlignment()
            if not proceed:
                return False

        self.align = align

        if align:
            self.logger.info('Aligning data...')
            self.titleLabel.setText(
                'Aligning data...(check progress in terminal)',
                color='w')

        _zip = zip(posData.tif_paths, posData.npz_paths)
        aligned = False
        posData.all_npz_paths = [
            tif.replace('.tif', '_aligned.npz') for tif in posData.tif_paths
        ]
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or posData.loaded_shifts is None

            filename_tif = os.path.basename(tif)
            user_ch_filename = f'{posData.basename}{user_ch_name}.tif'

            if not doAlign:
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                if os.path.exists(_npz):
                    posData.all_npz_paths[i] = _npz
                continue
            
            if filename_tif != user_ch_filename:
                continue
            
            # Align based on user_ch_name
            aligned = True
            if align:
                self.logger.info(f'Aligning: {tif}')
            tif_data = skimage.io.imread(tif)
            numFramesWith0s = self.detectTifAlignment(tif_data, posData)
            if align:
                proceed = self.warnTifAligned(numFramesWith0s, tif, posData)
                if not proceed:
                    return False

            # Alignment routine
            if posData.SizeZ>1:
                align_func = core.align_frames_3D
                df = posData.segmInfo_df.loc[posData.filename]
                zz = df['z_slice_used_dataPrep'].to_list()
                if not posData.filename.endswith('aligned') and align:
                    # Add aligned channel to segmInfo
                    df_aligned = posData.segmInfo_df.rename(
                        index={posData.filename: f'{posData.filename}_aligned'}
                    )
                    posData.segmInfo_df = pd.concat(
                        [posData.segmInfo_df, df_aligned]
                    )
                    posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
            else:
                align_func = core.align_frames_2D
                zz = None
            if align:
                aligned_frames, shifts = align_func(
                    tif_data, slices=zz, user_shifts=posData.loaded_shifts
                )
                posData.loaded_shifts = shifts
            else:
                aligned_frames = tif_data.copy()
            if align:
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                self.logger.info(f'Saving: {_npz}')
                temp_npz = self.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, aligned_frames)
                self.moveTempFile(temp_npz, _npz)
                np.save(posData.align_shifts_path, posData.loaded_shifts)
                posData.all_npz_paths[i] = _npz

                self.logger.info(f'Saving: {tif}')
                temp_tif = self.getTempfilePath(tif)
                myutils.imagej_tiffwriter(temp_tif, aligned_frames)
                self.moveTempFile(temp_tif, tif)
                posData.img_data = skimage.io.imread(tif)

        _zip = zip(posData.tif_paths, posData.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or aligned

            if not doAlign:
                continue
            
            if tif.endswith(f'{user_ch_name}.tif'):
                continue
            
            # Align the other channels
            if posData.loaded_shifts is None:
                break
            if align:
                self.logger.info(f'Aligning: {tif}')
            tif_data = skimage.io.imread(tif)

            # Alignment routine
            if posData.SizeZ>1:
                align_func = core.align_frames_3D
                df = posData.segmInfo_df.loc[posData.filename]
                zz = df['z_slice_used_dataPrep'].to_list()
            else:
                align_func = core.align_frames_2D
                zz = None
            if align:
                aligned_frames, shifts = align_func(
                    tif_data, slices=zz, user_shifts=posData.loaded_shifts
                )
            else:
                aligned_frames = tif_data.copy()
            _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
            
            if align:
                self.logger.info(f'Saving: {_npz}')
                temp_npz = self.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, aligned_frames)
                self.moveTempFile(temp_npz, _npz)
                posData.all_npz_paths[i] = _npz

                self.logger.info(f'Saving: {tif}')
                temp_tif = self.getTempfilePath(tif)
                myutils.imagej_tiffwriter(temp_tif, aligned_frames)
                self.moveTempFile(temp_tif, tif)

        # Align segmentation data accordingly
        self.segmAligned = False
        if posData.segmFound and aligned:
            if posData.loaded_shifts is None or not align:
                return True
            msg = QMessageBox()
            alignAnswer = msg.question(
                self, 'Align segmentation data?',
                'The system found an existing segmentation mask.\n\n'
                'Do you need to align that too?',
                msg.Yes | msg.No
            )
            if alignAnswer == msg.Yes:
                self.segmAligned = True
                self.logger.info(f'Aligning: {posData.segm_npz_path}')
                posData.segm_data, shifts = core.align_frames_2D(
                                             posData.segm_data,
                                             slices=None,
                                             user_shifts=posData.loaded_shifts
                )
                self.logger.info(f'Saving: {posData.segm_npz_path}')
                temp_npz = self.getTempfilePath(posData.segm_npz_path)
                np.savez_compressed(temp_npz, posData.segm_data)
                self.moveTempFile(temp_npz, posData.segm_npz_path)
        
        return True


    def detectTifAlignment(self, tif_data, posData):
        numFramesWith0s = 0
        if posData.SizeT == 1:
            tif_data = [tif_data]
        for img in tif_data:
            if posData.SizeZ > 1:
                firtsCol = img[:, :, 0]
                lastCol = img[:, : -1]
                firstRow = img[:, 0]
                lastRow = img[:, -1]
            else:
                firtsCol = img[:, 0]
                lastCol = img[: -1]
                firstRow = img[0]
                lastRow = img[-1]
            someZeros = (
                not np.any(firstRow) or not np.any(firtsCol)
                or not np.any(lastRow) or not np.any(lastCol)
            )
            if someZeros:
                numFramesWith0s += 1
        return numFramesWith0s

    def warnTifAligned(self, numFramesWith0s, tifPath, posData):
        proceed = True
        if numFramesWith0s>0 and posData.loaded_shifts is not None:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph("""
                The system detected that the .tif file contains <b>LREADY 
                aligned data</b>.<br><br>
                Using the found "align_shifts.npy" file to align would result 
                in misalignemnt of the data.<br><br>
                Therefore, the alignment routine will re-calculate the shifts 
                and it will NOT use the saved shifts.<br><br>
                Do you want to continue?
            """)
            msg.warning(
               self, 'Tif data ALREADY aligned!', txt,
               buttonsTexts=('Cancel', 'Yes')
            )
            if msg.cancel:
                proceed = False
        return proceed

    def getTempfilePath(self, path):
        temp_dirpath = tempfile.mkdtemp()
        filename = os.path.basename(path)
        tempFilePath = os.path.join(temp_dirpath, filename)
        return tempFilePath

    def moveTempFile(self, source, dst):
        self.logger.info(f'Moving temp file: {source}')
        tempDir = os.path.dirname(source)
        shutil.move(source, dst)
        shutil.rmtree(tempDir)

    def getMostRecentPath(self):
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            self.MostRecentPath = df.iloc[0]['path']
            if not isinstance(self.MostRecentPath, str):
                self.MostRecentPath = ''
        else:
            self.MostRecentPath = ''

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if exp_path in recentPaths:
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
        df = pd.DataFrame({'path': recentPaths,
                           'opened_last_on': pd.Series(openedOn,
                                                       dtype='datetime64[ns]')})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    def populateOpenRecent(self):
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 1. Read recent Paths
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

    def loadFiles(self, exp_path, user_ch_file_paths, user_ch_name):
        self.titleLabel.setText(
            'Loading data (check progress in the terminal)...', 
            color='w'
        )
        self.setWindowTitle(f'Cell-ACDC - Data Prep. - "{exp_path}"')

        self.num_pos = len(user_ch_file_paths)

        proceed = self.init_data(user_ch_file_paths, user_ch_name)

        if not proceed:
            self.openAction.setEnabled(True)
            return

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()

        exp_path = self.data[self.pos_i].exp_path
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
        if len(pos_foldernames) == 1:
            # There is only one position --> disable switch pos action
            self.loadPosAction.setDisabled(True)
        else:
            self.loadPosAction.setDisabled(False)


        if self.titleText is None:
            self.titleLabel.setText(
                'Data successfully loaded.<br>'
                'Press "START" button (top-left) to start prepping your data.',
                color='w')
        else:
            self.titleLabel.setText(
                self.titleText,
                color='w')

        self.openAction.setEnabled(True)
        self.startAction.setEnabled(True)
        self.showInExplorerAction.setEnabled(True)
        self.setImageNameText()
        self.update_img()
        self.setFontSizeROIlabels()

        self.stopAutomaticLoadingPos()

    def setImageNameText(self):
        self.statusbar.clearMessage()
        posData = self.data[self.pos_i]
        self.statusbar.showMessage(posData.filename_ext)

    def initLoading(self):
        self.dataIsLoaded = False
        # Remove all items from a previous session if open is pressed again
        self.removeAllItems()
        self.gui_addPlotItems()

        self.setCenterAlignmentTitle()
        self.openAction.setEnabled(False)

    def openRecentFile(self, path):
        self.logger.info(f'Opening recent folder: {path}')
        self.openFolder(exp_path=path)

    def openFolder(self, checked=False, exp_path=None):
        self.initLoading()

        if exp_path is None:
            self.getMostRecentPath()
            exp_path = QFileDialog.getExistingDirectory(
                self, 'Select experiment folder containing Position_n folders '
                      'or specific Position_n folder', self.MostRecentPath)

        self.addToRecentPaths(exp_path)

        if exp_path == '':
            self.openAction.setEnabled(True)
            self.titleLabel.setText(
                'File --> Open or Open recent to start the process',
                color='w')
            return

        folder_type = myutils.determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, exp_path = folder_type

        self.titleLabel.setText('Loading data...', color='w')
        self.setWindowTitle(f'Cell-ACDC - Data Prep. - "{exp_path}"')
        self.setCenterAlignmentTitle()

        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )

        if not is_pos_folder and not is_images_folder:
            select_folder = load.select_exp_folder()
            values = select_folder.get_values_dataprep(exp_path)
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
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                return

            select_folder.QtPrompt(self, values, allow_abort=False)
            if select_folder.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                return

            images_paths = []
            for pos in select_folder.selected_pos:
                images_paths.append(os.path.join(exp_path, pos, 'Images'))

            if select_folder.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                return

        elif is_pos_folder:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_paths = [os.path.join(exp_path, pos_foldername, 'Images')]

        elif is_images_folder:
            images_paths = [exp_path]

        self.images_paths = images_paths

        # Get info from first position selected
        images_path = self.images_paths[0]
        filenames = myutils.listdir(images_path)
        if ch_name_selector.is_first_call:
            ch_names, warn = (
                ch_name_selector.get_available_channels(filenames, images_path)
            )
            ch_names = ch_name_selector.askChannelName(
                filenames, images_path, warn, ch_names
            )
            if ch_name_selector.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                return
            
            if not ch_names:
                self.criticalNoTifFound(images_path)
            elif len(ch_names) > 1:
                ch_name_selector.QtPrompt(self, ch_names)
            else:
                ch_name_selector.channel_name = ch_names[0]
            ch_name_selector.setUserChannelName()
            if ch_name_selector.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                return
            user_ch_name = ch_name_selector.user_ch_name

        user_ch_file_paths = load.get_user_ch_paths(
                self.images_paths, user_ch_name
        )
        self.AutoPilotProfile.storeSelectedChannel(user_ch_name)

        self.loadFiles(exp_path, user_ch_file_paths, user_ch_name)
        self.setCenterAlignmentTitle()

        self.dataIsLoaded = True

    def setFontSizeROIlabels(self):
        Y, X = self.img.image.shape
        factor = 50
        self.pt = int(X/factor)
        self.roiLabelSize = '11px'
        self.roiLabelFont = QFont()
        self.roiLabelFont.setPixelSize(13)

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

    def setCenterAlignmentTitle(self):
        pass
        # self.titleLabel.item.setTextWidth(self.img.width())
        # fmt = QTextBlockFormat()
        # fmt.setAlignment(Qt.AlignHCenter)
        # cursor = self.titleLabel.item.textCursor()
        # cursor.select(QTextCursor.SelectionType.Document)
        # cursor.mergeBlockFormat(fmt)
        # cursor.clearSelection()
        # self.titleLabel.item.setTextCursor(cursor)

    def closeEvent(self, event):
        self.saveWindowGeometry()

        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()
            # Discard close and simply hide window
            event.ignore()
            self.hide()

        self.logger.info('Closing dataPrep logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

        if self.loop is not None:
            self.loop.exit()
        
        self.sigClose.emit(self)
        gc.collect()

    def saveWindowGeometry(self):
        settings = QSettings('schmollerlab', 'acdc_dataPrep')
        settings.setValue("geometry", self.saveGeometry())

    def readSettings(self):
        settings = QSettings('schmollerlab', 'acdc_dataPrep')
        if settings.value('geometry') is not None:
            self.restoreGeometry(settings.value("geometry"))

    def show(self):
        QMainWindow.show(self)
        self.readSettings()
        self.graphLayout.setFocus()
