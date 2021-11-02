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
from functools import partial
from tifffile.tifffile import TiffWriter, TiffFile

from PyQt5.QtCore import (
    Qt, QFile, QTextStream, QSize, QRect, QRectF,
    QObject, QThread, pyqtSignal
)
from PyQt5.QtGui import (
    QIcon, QKeySequence, QCursor, QTextBlockFormat,
    QTextCursor
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton,
    QMainWindow, QMenu, QToolBar, QGroupBox,
    QScrollBar, QCheckBox, QToolButton, QSpinBox,
    QComboBox, QDial, QButtonGroup, QFileDialog,
    QAbstractSlider, QMessageBox
)

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# NOTE: Enable icons
import qrc_resources

# Custom modules
import load, prompts, apps, core, myutils

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

pg.setConfigOptions(imageAxisOrder='row-major')

class toCsvWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def setData(self, data):
        self.data = data

    def run(self):
        for PosData in self.data:
            PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)
        self.finished.emit()

class dataPrepWin(QMainWindow):
    def __init__(self, parent=None, buttonToRestore=None, mainWin=None):
        super().__init__(parent)

        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin

        self.setWindowTitle("Cell-ACDC - data prep")
        self.setGeometry(100, 50, 850, 800)
        self.setWindowIcon(QIcon(":assign-motherbud.svg"))

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

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

    def gui_createActions(self):
        # File actions
        self.openAction = QAction(QIcon(":folder-open.svg"), "&Open...", self)
        self.exitAction = QAction("&Exit", self)
        self.showInExplorerAction = QAction(QIcon(":drawer.svg"),
                                    "&Show in Explorer/Finder", self)
        self.showInExplorerAction.setDisabled(True)

        # Toolbar actions
        self.prevAction = QAction(QIcon(":arrow-left.svg"),
                                        "Previous frame", self)
        self.nextAction = QAction(QIcon(":arrow-right.svg"),
                                        "Next Frame", self)
        self.jumpForwardAction = QAction(QIcon(":arrow-up.svg"),
                                        "Jump to 10 frames ahead", self)
        self.jumpBackwardAction = QAction(QIcon(":arrow-down.svg"),
                                        "Jump to 10 frames back", self)
        self.prevAction.setShortcut("left")
        self.openAction.setShortcut("Ctrl+O")
        self.nextAction.setShortcut("right")
        self.jumpForwardAction.setShortcut("up")
        self.jumpBackwardAction.setShortcut("down")
        self.openAction.setShortcut("Ctrl+O")

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

        navigateToolbar = QToolBar("Navigate", self)
        # navigateToolbar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolbar)

        navigateToolbar.addAction(self.prevAction)
        navigateToolbar.addAction(self.nextAction)
        navigateToolbar.addAction(self.jumpBackwardAction)
        navigateToolbar.addAction(self.jumpForwardAction)

        navigateToolbar.addAction(self.addBkrgRoiActon)

        navigateToolbar.addAction(self.ZbackAction)
        navigateToolbar.addAction(self.ZforwAction)
        navigateToolbar.addAction(self.interpAction)


        self.ROIshapeComboBox = QComboBox()
        self.ROIshapeComboBox.SizeAdjustPolicy(QComboBox.AdjustToContents)
        self.ROIshapeComboBox.addItems(['256x256'])
        ROIshapeLabel = QLabel('   ROI standard shape: ')
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
        self.prevAction.triggered.connect(self.prev_cb)
        self.nextAction.triggered.connect(self.next_cb)
        self.showInExplorerAction.triggered.connect(self.showInExplorer)
        self.jumpForwardAction.triggered.connect(self.skip10ahead_cb)
        self.jumpBackwardAction.triggered.connect(self.skip10back_cb)
        self.addBkrgRoiActon.triggered.connect(self.addDefaultBkgrROI)
        self.cropAction.triggered.connect(self.crop_cb)
        self.startAction.triggered.connect(self.prepData)
        self.interpAction.triggered.connect(self.interp_z)
        self.ZbackAction.triggered.connect(self.useSameZ_fromHereBack)
        self.ZforwAction.triggered.connect(self.useSameZ_fromHereForw)

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_addGraphicsItems(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Plot Item container for image
        self.ax1 = pg.PlotItem()
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=1, col=1)

        #Image histogram
        self.hist = pg.HistogramLUTItem()
        self.graphLayout.addItem(self.hist, row=1, col=0)

        # Title
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.titleLabel.setText(
            'File --> Open or Open recent to start the process')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.frameLabel.setText(' ')
        self.graphLayout.addItem(self.frameLabel, row=2, col=1)

    def gui_addPlotItems(self):
        # Image Item
        self.img = pg.ImageItem(np.zeros((512,512)))
        self.ax1.addItem(self.img)
        self.hist.setImageItem(self.img)

    def removeAllItems(self):
        self.ax1.clear()
        self.frameLabel.setText(' ')

    def gui_connectGraphicsEvents(self):
        self.img.hoverEvent = self.gui_hoverEventImg
        self.img.mousePressEvent = self.gui_mousePressEventImg

    def gui_createImgWidgets(self):
        self.img_Widglayout = QtGui.QGridLayout()

        _font = QtGui.QFont()
        _font.setPointSize(10)

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
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, value={val:.2f})')
            else:
                self.wcLabel.setText(f'')
        except Exception as e:
            self.wcLabel.setText(f'')

    def showInExplorer(self):
        try:
            PosData = self.data[self.pos_i]
            systems = {
                'nt': os.startfile,
                'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
                'os2': lambda foldername: os.system('open "%s"' % foldername)
                 }

            systems.get(os.name, os.startfile)(PosData.images_path)
        except AttributeError:
            pass

    def next_cb(self, checked):
        if self.num_pos > 1:
            self.next_pos()
        else:
            self.next_frame()

    def prev_cb(self, checked):
        if self.num_pos > 1:
            self.prev_pos()
        else:
            self.prev_frame()

    def skip10ahead_cb(self, checked):
        if self.num_pos > 1:
            self.skip10ahead_pos()
        else:
            self.skip10ahead_frame()

    def skip10back_cb(self, checked):
        if self.num_pos > 1:
            self.skip10back_pos()
        else:
            self.skip10back_frame()

    def next_pos(self):
        if self.pos_i < self.num_pos-1:
            self.removeBkgrROIs()
            self.removeCropROI()
            self.pos_i += 1
            self.update_img()
            self.updateROI()
            self.updateBkgrROIs()
        else:
            print('You reached last position')


    def prev_pos(self):
        if self.pos_i > 0:
            self.removeBkgrROIs()
            self.removeCropROI()
            self.pos_i -= 1
            self.update_img()
            self.updateROI()
            self.updateBkgrROIs()
        else:
            print('You reached first position')
            # self.pos_i = self.num_pos-1


    def skip10ahead_pos(self):
        if self.pos_i < self.num_pos-10:
            self.pos_i += 10
        else:
            self.pos_i = 0
        self.update_img()

    def skip10back_pos(self):
        if self.pos_i > 9:
            self.pos_i -= 10
        else:
            self.pos_i = self.num_pos-1
        self.update_img()

    def next_frame(self):
        if self.frame_i < self.num_frames-1:
            self.frame_i += 1
        else:
            self.frame_i = 0
        self.navigateScrollbar.setValue(self.frame_i+1)
        self.update_img()

    def prev_frame(self):
        if self.frame_i > 0:
            self.frame_i -= 1
        else:
            self.frame_i = self.num_frames-1
        self.navigateScrollbar.setValue(self.frame_i+1)
        self.update_img()

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
        PosData = self.data[self.pos_i]
        if self.num_pos > 1:
            self.frameLabel.setText(
                     f'Current position = {self.pos_i+1}/{self.num_pos} '
                     f'({PosData.pos_foldername})')
            self.navigateSB_label.setText(f'Pos n. {self.pos_i+1}')

            self.navigateScrollbar.valueChanged.disconnect()
            self.navigateScrollbar.setValue(self.pos_i+1)
        else:
            self.frameLabel.setText(
                     f'Current frame = {self.frame_i+1}/{self.num_frames}')
            self.navigateSB_label.setText(f'frame n. {self.frame_i+1}')
            self.navigateScrollbar.valueChanged.disconnect()
            self.navigateScrollbar.setValue(self.frame_i+1)
        self.navigateScrollbar.valueChanged.connect(
            self.navigateScrollBarMoved
        )

    def getImage(self, PosData, img_data, frame_i):
        if PosData.SizeT > 1:
            img = img_data[frame_i].copy()
        else:
            img = img_data.copy()
        if PosData.SizeZ > 1:
            df =  PosData.segmInfo_df
            idx = (PosData.filename, frame_i)
            z = PosData.segmInfo_df.at[idx, 'z_slice_used_dataPrep']
            zProjHow = PosData.segmInfo_df.at[idx, 'which_z_proj']
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
                self.z_label.setText(f'z-slice  {z+1}/{PosData.SizeZ}')
                img = img[z]
            elif zProjHow == 'max z-projection':
                img = img.max(axis=0)
            elif zProjHow == 'mean z-projection':
                img = img.mean(axis=0)
            elif zProjHow == 'median z-proj.':
                img = np.median(img, axis=0)
        return img

    def update_img(self):
        self.updateNavigateItems()
        PosData = self.data[self.pos_i]
        img = self.getImage(PosData, PosData.img_data, self.frame_i)
        # img = img/img.max()
        self.img.setImage(img)
        self.zSliceScrollBar.setMaximum(PosData.SizeZ-1)

    def updateROI(self):
        if self.startAction.isEnabled():
            return

        PosData = self.data[self.pos_i]
        if PosData.cropROI not in self.ax1.items:
            self.ax1.addItem(PosData.cropROI.label)
            self.ax1.addItem(PosData.cropROI)

        PosData.cropROI.sigRegionChanged.connect(self.updateCurrentRoiShape)
        PosData.cropROI.sigRegionChangeFinished.connect(self.ROImovingFinished)

    def removeCropROI(self):
        if self.startAction.isEnabled():
            return

        PosData = self.data[self.pos_i]
        self.ax1.removeItem(PosData.cropROI.label)
        self.ax1.removeItem(PosData.cropROI)

        try:
            PosData.cropROI.sigRegionChanged.disconnect()
            PosData.cropROI.sigRegionChangeFinished.disconnect()
        except TypeError:
            pass


    def updateBkgrROIs(self):
        if self.startAction.isEnabled():
            return

        PosData = self.data[self.pos_i]
        for roi in PosData.bkgrROIs:
            if roi not in self.ax1.items:
                self.ax1.addItem(roi.label)
                self.ax1.addItem(roi)
            roi.sigRegionChanged.connect(self.bkgrROIMoving)
            roi.sigRegionChangeFinished.connect(self.bkgrROImovingFinished)

    def removeBkgrROIs(self):
        if self.startAction.isEnabled():
            return

        PosData = self.data[self.pos_i]
        for roi in PosData.bkgrROIs:
            self.ax1.removeItem(roi.label)
            self.ax1.removeItem(roi)

            try:
                roi.sigRegionChanged.disconnect()
                roi.sigRegionChangeFinished.disconnect()
            except TypeError:
                pass

    def init_attr(self):
        PosData = self.data[0]
        self.navigateScrollbar.setEnabled(True)
        self.navigateScrollbar.setMinimum(1)
        if PosData.SizeT > 1:
            self.navigateScrollbar.setMaximum(PosData.SizeT)
        elif self.num_pos > 1:
            self.navigateScrollbar.setMaximum(self.num_pos)
        else:
            self.navigateScrollbar.setDisabled(True)
        self.navigateScrollbar.setValue(1)
        self.navigateScrollbar.valueChanged.connect(
            self.navigateScrollBarMoved
        )

    def navigateScrollBarMoved(self, value):
        PosData = self.data[self.pos_i]
        self.removeBkgrROIs()
        self.removeCropROI()

        if PosData.SizeT > 1:
            self.frame_i = value-1
        elif self.num_pos > 1:
            self.pos_i = value-1
        else:
            return

        self.update_img()
        self.updateBkgrROIs()
        self.updateROI()


    def crop(self, data, PosData):
        x0, y0 = [int(round(c)) for c in PosData.cropROI.pos()]
        w, h = [int(round(c)) for c in PosData.cropROI.size()]
        if data.ndim == 4:
            croppedData = data[:, :, y0:y0+h, x0:x0+w]
        elif data.ndim == 3:
            croppedData = data[:, y0:y0+h, x0:x0+w]
        elif data.ndim == 2:
            croppedData = data[y0:y0+h, x0:x0+w]
        return croppedData

    def saveBkgrROIs(self, PosData):
        if not PosData.bkgrROIs:
            return

        ROIstates = [roi.saveState() for roi in PosData.bkgrROIs]
        with open(PosData.dataPrepBkgrROis_path, 'w') as json_fp:
            json.dump(ROIstates, json_fp)

    def saveBkgrData(self, PosData):
        try:
            os.remove(PosData.dataPrepBkgrROis_path)
        except FileNotFoundError:
            pass

        # If we crop we save data from background ROI for each bkgrROI
        for chName in PosData.chNames:
            alignedFound = False
            tifFound = False
            for file in os.listdir(PosData.images_path):
                filePath = os.path.join(PosData.images_path, file)
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
            for r, roi in enumerate(PosData.bkgrROIs):
                xl, yt = [int(round(c)) for c in roi.pos()]
                w, h = [int(round(c)) for c in roi.size()]
                is4D = PosData.SizeT > 1 and PosData.SizeZ > 1
                is3Dz = PosData.SizeT == 1 and PosData.SizeZ > 1
                is3Dt = PosData.SizeT > 1 and PosData.SizeZ == 1
                is2D = PosData.SizeT == 1 and PosData.SizeZ == 1
                if is4D:
                    bkgr_data = chData[:, :, yt:yt+h, xl:xl+w]
                elif is3Dz or is3Dt:
                    bkgr_data = chData[:, yt:yt+h, xl:xl+w]
                elif is2D:
                    bkgr_data = chData[yt:yt+h, xl:xl+w]
                bkgrROI_data[f'roi{r}_data'] = bkgr_data

            if bkgrROI_data:
                bkgr_data_fn = f'{filename}_bkgrRoiData.npz'
                bkgr_data_path = os.path.join(PosData.images_path, bkgr_data_fn)
                np.savez_compressed(bkgr_data_path, **bkgrROI_data)

    def removeAllROIs(self, event):
        for PosData in self.data:
            for roi in PosData.bkgrROIs:
                self.ax1.removeItem(roi.label)
                self.ax1.removeItem(roi)

            PosData.bkgrROIs = []
            try:
                os.remove(PosData.dataPrepBkgrROis_path)
            except FileNotFoundError:
                pass

    def removeROI(self, event):
        PosData = self.data[self.pos_i]
        PosData.bkgrROIs.remove(self.roi_to_del)
        self.ax1.removeItem(self.roi_to_del.label)
        self.ax1.removeItem(self.roi_to_del)
        if not PosData.bkgrROIs:
            try:
                os.remove(PosData.dataPrepBkgrROis_path)
            except FileNotFoundError:
                pass
        else:
            self.saveBkgrROIs(PosData)

    def gui_mousePressEventImg(self, event):
        PosData = self.data[self.pos_i]
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

        if left_click:
            pg.ImageItem.mousePressEvent(self.img, event)

        x, y = event.pos().x(), event.pos().y()

        handleSize = 7

        # Check if right click on ROI
        for r, roi in enumerate(PosData.bkgrROIs):
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
                self.roi_to_del = roi
                self.roiContextMenu = QMenu(self)
                separator = QAction(self)
                separator.setSeparator(True)
                self.roiContextMenu.addAction(separator)
                action1 = QAction('Remove background ROI')
                action1.triggered.connect(self.removeROI)
                action2 = QAction('Remove ALL background ROIs')
                action2.triggered.connect(self.removeAllROIs)
                self.roiContextMenu.addAction(action1)
                self.roiContextMenu.addAction(action2)
                self.roiContextMenu.exec_(event.screenPos())
            elif dragRoi:
                event.ignore()
                return

        if PosData.cropROI is not None:
            x0, y0 = [int(c) for c in PosData.cropROI.pos()]
            w, h = [int(c) for c in PosData.cropROI.size()]
            x1, y1 = x0+w, y0+h
            clickedOnROI = (
                x>=x0-handleSize and x<=x1+handleSize
                and y>=y0-handleSize and y<=y1+handleSize
            )
            dragRoi = left_click and clickedOnROI
            if dragRoi:
                event.ignore()
                return

    def saveROIcoords(self, doCrop, PosData):
        x0, y0 = [int(round(c)) for c in PosData.cropROI.pos()]
        w, h = [int(round(c)) for c in PosData.cropROI.size()]

        Y, X = self.img.image.shape
        x1, y1 = x0+w, y0+h

        x0 = x0 if x0>0 else 0
        y0 = y0 if y0>0 else 0
        x1 = x1 if x1<X else X
        y1 = y1 if y1<Y else Y

        if x0<=0 and y0<=0 and x1>=X and y1>=Y:
            # ROI coordinates are the exact image shape. No need to save them
            return

        print(
            f'Saving ROI coords: x_left = {x0}, x_right = {x1}, '
            f'y_top = {y0}, y_bottom = {y1}\n'
            f'to {PosData.dataPrepROI_coords_path}'
        )

        csv_data = (
            f'description,value\n'
            f'x_left,{x0}\n'
            f'x_right,{x1}\n'
            f'y_top,{y0}\n'
            f'y_bottom,{y1}\n'
            f'cropped,{int(doCrop)}'
        )

        try:
            with open(PosData.dataPrepROI_coords_path, 'w') as csv:
                csv.write(csv_data)
        except PermissionError:
            self.permissionErrorCritical(PosData.dataPrepROI_coords_path)
            with open(PosData.dataPrepROI_coords_path, 'w') as csv:
                csv.write(csv_data)



    def crop_cb(self):
        # msg = QMessageBox()
        # doSave = msg.question(
        #     self, 'Save data?', 'Do you want to save?',
        #     msg.Yes | msg.No
        # )
        # if doSave == msg.No:
        #     return

        self.cropAction.setDisabled(True)
        for PosData in self.data:
            self.saveBkgrROIs(PosData)

            if PosData.SizeZ > 1:
                # Save segmInfo
                PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

            # Get crop shape and print it
            data = PosData.img_data
            croppedData = self.crop(data, PosData)


            doCrop = True
            if croppedData.shape != data.shape:
                doCrop = self.askCropping(data.shape, croppedData.shape)
            else:
                doCrop = False

            if not doCrop:
                self.cropAction.setEnabled(True)
                print('Done.')
                return

            print(f'Cropping {PosData.relPath}...')
            self.titleLabel.setText(
                'Cropping... (check progress in the terminal)',
                color='w')

            print('Cropped data shape:', croppedData.shape)
            self.saveROIcoords(doCrop, PosData)

            print('Saving background data...')
            self.saveBkgrData(PosData)

            # Get metadata from tif
            with TiffFile(PosData.tif_path) as tif:
                metadata = tif.imagej_metadata

            # Save channels (npz AND tif)
            _zip = zip(PosData.tif_paths, PosData.all_npz_paths)
            for tif, npz in _zip:
                if self.align:
                    data = np.load(npz)['arr_0']
                else:
                    data = skimage.io.imread(tif)

                npz_data = self.crop(data, PosData)

                if self.align:
                    print('Saving: ', npz)
                    temp_npz = self.getTempfilePath(npz)
                    np.savez_compressed(temp_npz, npz_data)
                    self.moveTempFile(temp_npz, npz)

                print('Saving: ', tif)
                temp_tif = self.getTempfilePath(tif)
                self.imagej_tiffwriter(temp_tif, npz_data,
                                       metadata, PosData)
                self.moveTempFile(temp_tif, tif)

            # Save segm.npz
            if PosData.segmFound:
                print('Saving: ', PosData.segm_npz_path)
                data = PosData.segm_data
                croppedSegm = self.crop(data, PosData)
                temp_npz = self.getTempfilePath(PosData.segm_npz_path)
                np.savez_compressed(temp_npz, croppedSegm)
                self.moveTempFile(temp_npz, PosData.segm_npz_path)

            # Correct acdc_df if present and save
            if PosData.acdc_df is not None:
                x0, y0 = [int(round(c)) for c in PosData.cropROI.pos()]
                print('Saving: ', PosData.acdc_output_csv_path)
                df = PosData.acdc_df
                df['x_centroid'] -= x0
                df['y_centroid'] -= y0
                df['editIDclicked_x'] -= x0
                df['editIDclicked_y'] -= y0
                try:
                    df.to_csv(PosData.acdc_output_csv_path)
                except PermissionError:
                    self.permissionErrorCritical(PosData.acdc_output_csv_path)
                    df.to_csv(PosData.acdc_output_csv_path)

            print(f'{PosData.pos_foldername} saved!')
            print(f'--------------------------------')
            print('')
        # self.cropAction.setEnabled(True)
        self.titleLabel.setText(
            'Saved! You can close the program or load another position.',
            color='g')

    def permissionErrorCritical(self, path):
        msg = QMessageBox()
        msg.critical(
            self, 'Permission denied',
            f'The below file is open in another app (Excel maybe?).\n\n'
            f'{path}\n\n'
            'Close file and then press "Ok".',
            msg.Ok
        )


    def askCropping(self, dataShape, croppedShape):
        msg = QMessageBox(self)
        msg.setWindowTitle('Crop?')
        msg.setIcon(msg.Warning)
        msg.setText(
            f'You are about to crop data from shape {dataShape} '
            f'to shape {croppedShape}\n\n'
            'Saving cropped data cannot be undone.\n\n'
            'Do you want to crop or simply save the ROI coordinates for the segmentation step?')
        doCropButton = QPushButton('Yes, crop please.')
        msg.addButton(doCropButton, msg.NoRole)
        msg.addButton(QPushButton('No, save without cropping'), msg.YesRole)
        cancelButton = QPushButton('Cancel')
        msg.addButton(cancelButton, msg.RejectRole)
        msg.exec_()
        if msg.clickedButton() == doCropButton:
            proceed = True
        elif msg.clickedButton() == cancelButton:
            proceed = False
        else:
            proceed = False
        return proceed


    def imagej_tiffwriter(self, new_path, data, metadata, PosData):
        if data.dtype != np.uint8 or data.dtype != np.uint16:
            data = skimage.img_as_uint(data)
        with TiffWriter(new_path, imagej=True) as new_tif:
            if PosData.SizeZ > 1 and PosData.SizeT > 1:
                # 3D data over time
                T, Z, Y, X = data.shape
            elif PosData.SizeZ == 1 and PosData.SizeT > 1:
                # 2D data over time
                T, Y, X = data.shape
                Z = 1
            elif PosData.SizeZ > 1 and PosData.SizeT == 1:
                # Single 3D data
                Z, Y, X = data.shape
                T = 1
            elif PosData.SizeZ == 1 and PosData.SizeT == 1:
                # Single 2D data
                Y, X = data.shape
                T, Z = 1, 1
            data.shape = T, Z, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
            new_tif.save(data, metadata=metadata)

    def getDefaultROI(self):
        Y, X = self.img.image.shape
        w, h = X, Y

        xc, yc = int(round(X/2)), int(round(Y/2))
        # yt, xl = int(round(xc-w/2)), int(round(yc-h/2))
        yt, xl = 0, 0

        # Add ROI Rectangle
        cropROI = pg.ROI(
            [xl, yt], [w, h],
            rotatable=False,
            removable=False,
            pen=pg.mkPen(color='r'),
            maxBounds=QRectF(QRect(0,0,X,Y))
        )
        return cropROI

    def setROIprops(self, roi):
        xl, yt = [int(round(c)) for c in roi.pos()]

        roi.handleSize = 7
        roi.label = pg.LabelItem('ROI', color='r', size=f'{self.pt}pt')
        hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yt-hLabel)

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
        for f, file_path in enumerate(user_ch_file_paths):
            try:
                PosData = load.loadData(file_path, user_ch_name, QParent=self)
                PosData.getBasenameAndChNames()
                PosData.buildPaths()
                PosData.loadImgData()
                PosData.loadOtherFiles(
                                   load_segm_data=True,
                                   load_acdc_df=True,
                                   load_shifts=True,
                                   loadSegmInfo=True,
                                   load_dataPrep_ROIcoords=True,
                                   load_delROIsInfo=False,
                                   loadBkgrData=False,
                                   loadBkgrROIs=True,
                                   load_last_tracked_i=False,
                                   load_metadata=True,
                                   getTifPath=True
                )

                # If data was cropped then dataPrep_ROIcoords are useless
                if PosData.dataPrep_ROIcoords is not None:
                    df = PosData.dataPrep_ROIcoords
                    isROIactive = df.at['cropped', 'value'] == 0
                    if not isROIactive:
                        PosData.dataPrep_ROIcoords = None

                PosData.loadAllImgPaths()
                if f==0 and not self.metadataAlreadyAsked:
                    proceed = PosData.askInputMetadata(
                                                ask_SizeT=self.num_pos==1,
                                                ask_TimeIncrement=False,
                                                ask_PhysicalSizes=False,
                                                save=True)
                    self.SizeT = PosData.SizeT
                    self.SizeZ = PosData.SizeZ
                    if not proceed:
                        self.titleLabel.setText(
                            'File --> Open or Open recent to start the process',
                            color='w')
                        return False
                else:
                    PosData.SizeT = self.SizeT
                    if self.SizeZ > 1:
                        # In case we know we are loading single 3D z-stacks
                        # we alwways use the third dimensins as SizeZ because
                        # SizeZ in some positions might be different than
                        # first loaded pos
                        SizeZ = PosData.img_data.shape[-3]
                        PosData.SizeZ = SizeZ
                    else:
                        PosData.SizeZ = 1
                    if self.SizeT > 1:
                        PosData.SizeT = self.SizeT
                    else:
                        PosData.SizeT = 1
                    PosData.saveMetadata()
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

            if PosData is None:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return False

            img_shape = PosData.img_data.shape
            self.num_frames = PosData.SizeT
            self.user_ch_name = user_ch_name
            SizeT = PosData.SizeT
            SizeZ = PosData.SizeZ
            if f==0:
                print(f'Data shape = {img_shape}')
                print(f'Number of frames = {SizeT}')
                print(f'Number of z-slices per frame = {SizeZ}')
            data.append(PosData)

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
        for PosData in self.data:
            NO_segmInfo = (
                PosData.segmInfo_df is None
                or PosData.filename not in PosData.segmInfo_df.index
            )
            if NO_segmInfo and PosData.SizeZ > 1:
                filename = PosData.filename
                df = myutils.getDefault_SegmInfo_df(PosData, filename)
                if PosData.segmInfo_df is None:
                    PosData.segmInfo_df = df
                else:
                    PosData.segmInfo_df = pd.concat([df, PosData.segmInfo_df])
                PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

        PosData = self.data[0]
        if PosData.SizeZ > 1:
            self.zSliceScrollBar.setDisabled(False)
            self.zProjComboBox.setDisabled(False)
            self.zSliceScrollBar.setMaximum(PosData.SizeZ-1)
            try:
                self.zSliceScrollBar.valueChanged.disconnect()
                self.zProjComboBox.currentTextChanged.disconnect()
            except Exception as e:
                pass
            self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
            if PosData.SizeT > 1:
                self.interpAction.setEnabled(True)
                self.ZbackAction.setEnabled(True)
                self.ZforwAction.setEnabled(True)
            df = PosData.segmInfo_df
            idx = (PosData.filename, self.frame_i)
            how = PosData.segmInfo_df.at[idx, 'which_z_proj']
            self.zProjComboBox.setCurrentText(how)

    def update_z_slice(self, z):
        if self.zProjComboBox.currentText() == 'single z-slice':
            PosData = self.data[self.pos_i]
            df = PosData.segmInfo_df
            idx = (PosData.filename, self.frame_i)
            PosData.segmInfo_df.at[idx, 'z_slice_used_dataPrep'] = z
            PosData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
            self.update_img()
            PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

    def updateZproj(self, how):
        PosData = self.data[self.pos_i]
        for frame_i in range(self.frame_i, PosData.SizeT):
            df = PosData.segmInfo_df
            idx = (PosData.filename, self.frame_i)
            PosData.segmInfo_df.at[idx, 'which_z_proj'] = how
            PosData.segmInfo_df.at[idx, 'which_z_proj_gui'] = how
        if how == 'single z-slice':
            self.zSliceScrollBar.setDisabled(False)
            self.z_label.setStyleSheet('color: black')
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.z_label.setStyleSheet('color: gray')
            self.update_img()

        # Apply same z-proj to future pos
        if PosData.SizeT == 1:
            for PosData in self.data[self.pos_i+1:]:
                idx = (PosData.filename, self.frame_i)
                PosData.segmInfo_df.at[idx, 'which_z_proj'] = how

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
        PosData = self.data[self.pos_i]
        df = PosData.segmInfo_df
        z = df.at[(PosData.filename, self.frame_i), 'z_slice_used_dataPrep']
        for i in range(0, self.frame_i):
            df.at[(PosData.filename, i), 'z_slice_used_dataPrep'] = z
            df.at[(PosData.filename, i), 'z_slice_used_dataPrep'] = how

    def useSameZ_fromHereForw(self, event):
        how = self.zProjComboBox.currentText()
        PosData = self.data[self.pos_i]
        df = PosData.segmInfo_df
        z = df.at[(PosData.filename, self.frame_i), 'z_slice_used_dataPrep']
        for i in range(self.frame_i, PosData.SizeT):
            df.at[(PosData.filename, i), 'z_slice_used_dataPrep'] = z
            df.at[(PosData.filename, i), 'z_slice_used_dataPrep'] = how

    def interp_z(self, event):
        PosData = self.data[self.pos_i]
        df = PosData.segmInfo_df
        x0, z0 = 0, df.at[(PosData.filename, i), 'z_slice_used_dataPrep']
        x1 = self.frame_i
        z1 = df.at[(PosData.filename, x1), 'z_slice_used_dataPrep']
        f = scipy.interpolate.interp1d([x0, x1], [z0, z1])
        xx = np.arange(0, self.frame_i)
        zz = np.round(f(xx)).astype(int)
        how = 'single z-slice'
        for i in range(self.frame_i, PosData.SizeT):
            df.at[(PosData.filename, i), 'z_slice_used_dataPrep'] = zz[i]
            df.at[(PosData.filename, i), 'z_slice_used_dataPrep'] = 'single z-slice'


    def prepData(self, event):
        self.titleLabel.setText(
            'Prepping data... (check progress in the terminal)',
            color='w')
        doZip = False
        for p, PosData in enumerate(self.data):
            self.startAction.setDisabled(True)
            nonTifFound = (
                any([npz is not None for npz in PosData.npz_paths]) or
                any([npy is not None for npy in PosData.npy_paths]) or
                PosData.segmFound
            )
            imagesPath = PosData.images_path
            zipPath = f'{imagesPath}.zip'
            if nonTifFound and p==0:
                msg = QMessageBox()
                doZipAnswer = msg.warning(
                   self, 'NON-Tif data detected!',
                   'Additional NON-tif files detected.\n\n'
                   'The requested experiment folder already contains .npy '
                   'or .npz files '
                   'most likely from previous analysis runs.\n\n'
                   'To avoid data losses we recommend zipping the "Images" folder.\n\n'
                   'If everything looks fine after prepping the data, '
                   'you can manually '
                   'delete the zip archive.\n\n'
                   'Do you want to automatically zip now?\n\n'
                   'PS: Zip archive location:\n\n'
                   f'{zipPath}',
                   msg.Yes | msg.No
                )
                if doZipAnswer == msg.Yes:
                    doZip = True
            if doZip:
                print(f'Zipping Images folder: {zipPath}')
                shutil.make_archive(imagesPath, 'zip', imagesPath)
            self.npy_to_npz(PosData)
            self.alignData(self.user_ch_name, PosData)
            if PosData.SizeZ>1:
                PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

        self.update_img()
        print('Done.')
        self.addROIs()
        self.saveROIcoords(False, self.data[self.pos_i])
        self.saveBkgrROIs(self.data[self.pos_i])
        self.cropAction.setEnabled(True)
        self.titleLabel.setText(
            'Data successfully prepped. You can now crop the images or '
            'close the program',
            color='w')

    def setStandardRoiShape(self, text):
        PosData = self.data[self.pos_i]
        Y, X = PosData.img_data.shape[-2:]
        m = re.findall('(\d+)x(\d+)', text)
        w, h = int(m[0][0]), int(m[0][1])
        # xc, yc = int(round(X/2)), int(round(Y/2))
        # yt, xl = int(round(xc-w/2)), int(round(yc-h/2))
        PosData.cropROI.setPos([0, 0])
        PosData.cropROI.setSize([w, h])

    def addROIs(self):
        Y, X = self.img.image.shape

        max_size = round(int(np.log2(min([Y, X])/16)))
        items = [f'{16*(2**i)}x{16*(2**i)}' for i in range(1, max_size+1)]
        items.append(f'{X}x{Y}')
        self.ROIshapeComboBox.clear()
        self.ROIshapeComboBox.addItems(items)
        self.ROIshapeComboBox.setCurrentText(items[-1])

        for PosData in self.data:
            if PosData.dataPrep_ROIcoords is None:
                cropROI = self.getDefaultROI()
            else:
                xl = PosData.dataPrep_ROIcoords.at['x_left', 'value']
                yt = PosData.dataPrep_ROIcoords.at['y_top', 'value']
                w = PosData.dataPrep_ROIcoords.at['x_right', 'value'] - xl
                h = PosData.dataPrep_ROIcoords.at['y_bottom', 'value'] - yt
                cropROI = pg.ROI(
                    [xl, yt], [w, h],
                    rotatable=False,
                    removable=False,
                    pen=pg.mkPen(color='r'),
                    maxBounds=QRectF(QRect(0,0,X,Y))
                )

            self.setROIprops(cropROI)
            PosData.cropROI = cropROI

        self.updateROI()

        try:
            self.ROIshapeComboBox.currentTextChanged.disconnect()
        except Exception as e:
            self.ROIshapeComboBox.currentTextChanged.connect(
                                                      self.setStandardRoiShape)

        self.addBkrgRoiActon.setDisabled(False)

        for PosData in self.data:
            if not PosData.bkgrROIs:
                bkgrROI = self.getDefaultBkgrROI()
                self.setBkgrROIprops(bkgrROI)
                PosData.bkgrROIs.append(bkgrROI)
            else:
                for bkgrROI in PosData.bkgrROIs:
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
            pen=pg.mkPen(color=(150,150,150)),
            maxBounds=QRectF(QRect(0,0,X,Y))
        )
        return bkgrROI

    def setBkgrROIprops(self, bkgrROI):
        bkgrROI.handleSize = 7

        xl, yt = [int(round(c)) for c in bkgrROI.pos()]
        bkgrROI.label = pg.LabelItem(
            'Bkgr. ROI', color=(150,150,150), size=f'{self.pt}pt')
        hLabel = bkgrROI.label.rect().bottom()
        bkgrROI.label.setPos(xl, yt-hLabel)

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


    def addDefaultBkgrROI(self, checked=False):
        bkgrROI = self.getDefaultBkgrROI()
        self.setBkgrROIprops(bkgrROI)
        self.ax1.addItem(bkgrROI)
        self.ax1.addItem(bkgrROI.label)
        PosData = self.data[self.pos_i]
        PosData.bkgrROIs.append(bkgrROI)
        self.saveBkgrROIs()

    def bkgrROIMoving(self, roi):
        txt = roi.label.text
        roi.setPen(color=(255,255,0))
        roi.label.setText(txt, color=(255,255,0), size=f'{self.pt}pt')
        xl, yt = [int(round(c)) for c in roi.pos()]
        hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yt-hLabel)

    def bkgrROImovingFinished(self, roi):
        txt = roi.label.text
        roi.setPen(color=(150,150,150))
        roi.label.setText(txt, color=(150,150,150), size=f'{self.pt}pt')
        PosData = self.data[self.pos_i]
        idx = PosData.bkgrROIs.index(roi)
        PosData.bkgrROIs[idx] = roi
        self.saveBkgrROIs(PosData)

    def ROImovingFinished(self, roi):
        txt = roi.label.text
        roi.setPen(color='r')
        roi.label.setText(txt, color='r', size=f'{self.pt}pt')
        self.saveROIcoords(False, self.data[self.pos_i])

    def updateCurrentRoiShape(self, roi):
        roi.setPen(color=(255,255,0))
        roi.label.setText('ROI', color=(255,255,0), size=f'{self.pt}pt')
        xl, yt = [int(round(c)) for c in roi.pos()]
        hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yt-hLabel)
        w, h = [int(round(c)) for c in roi.size()]
        self.ROIshapeLabel.setText(f'   Current ROI shape: {w} x {h}')

    def alignData(self, user_ch_name, PosData):
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
        with TiffFile(PosData.tif_path) as tif:
            metadata = tif.imagej_metadata

        align = True
        if PosData.loaded_shifts is None and PosData.SizeT > 1:
            msg = QMessageBox()
            alignAnswer = msg.question(
                self, 'Align frames?',
                f'Do you want to align ALL channels based on "{user_ch_name}" '
                'channel?\n\n'
                'If you don\'t know what to choose, we reccommend '
                'aligning.',
                msg.Yes | msg.No
            )
            if alignAnswer == msg.No:
                align = False
                # Create 0, 0 shifts to perform 0 alignment
                PosData.loaded_shifts = np.zeros((self.num_frames,2), int)
        elif PosData.SizeT == 1:
            align = False
            # Create 0, 0 shifts to perform 0 alignment
            PosData.loaded_shifts = np.zeros((self.num_frames,2), int)

        self.align = align

        if align:
            print('Aligning data...')
            self.titleLabel.setText(
                'Aligning data...(check progress in terminal)',
                color='w')

        _zip = zip(PosData.tif_paths, PosData.npz_paths)
        aligned = False
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or PosData.loaded_shifts is None

            # Align based on user_ch_name
            if doAlign and tif.find(user_ch_name) != -1:
                aligned = True
                if align:
                    print('Aligning: ', tif)
                tif_data = skimage.io.imread(tif)
                numFramesWith0s = self.detectTifAlignment(tif_data, PosData)
                proceed = self.warnTifAligned(numFramesWith0s, tif, PosData)
                if not proceed:
                    break

                # Alignment routine
                if PosData.SizeZ>1:
                    align_func = core.align_frames_3D
                    df = PosData.segmInfo_df.loc[PosData.filename]
                    zz = df['z_slice_used_dataPrep'].to_list()
                else:
                    align_func = core.align_frames_2D
                    zz = None
                if align:
                    aligned_frames, shifts = align_func(
                                              tif_data,
                                              slices=zz,
                                              user_shifts=PosData.loaded_shifts,
                                              pbar=True
                    )
                    PosData.loaded_shifts = shifts
                else:
                    aligned_frames = tif_data.copy()
                if align:
                    _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                    print('Saving: ', _npz)
                    temp_npz = self.getTempfilePath(_npz)
                    np.savez_compressed(temp_npz, aligned_frames)
                    self.moveTempFile(temp_npz, _npz)
                    np.save(PosData.align_shifts_path, PosData.loaded_shifts)
                    PosData.all_npz_paths[i] = _npz

                    print('Saving: ', tif)
                    temp_tif = self.getTempfilePath(tif)
                    self.imagej_tiffwriter(temp_tif, aligned_frames,
                                           metadata, PosData)
                    self.moveTempFile(temp_tif, tif)
                PosData.img_data = skimage.io.imread(tif)

        _zip = zip(PosData.tif_paths, PosData.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or aligned
            # Align the other channels
            if doAlign and tif.find(user_ch_name) == -1:
                if PosData.loaded_shifts is None:
                    break
                if align:
                    print('Aligning: ', tif)
                tif_data = skimage.io.imread(tif)

                # Alignment routine
                if PosData.SizeZ>1:
                    align_func = core.align_frames_3D
                    df = PosData.segmInfo_df.loc[PosData.filename]
                    zz = df['z_slice_used_dataPrep'].to_list()
                else:
                    align_func = core.align_frames_2D
                    zz = None
                if align:
                    aligned_frames, shifts = align_func(
                                          tif_data,
                                          slices=zz,
                                          user_shifts=PosData.loaded_shifts)
                else:
                    aligned_frames = tif_data.copy()
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                if align:
                    print('Saving: ', _npz)
                    temp_npz = self.getTempfilePath(_npz)
                    np.savez_compressed(temp_npz, aligned_frames)
                    self.moveTempFile(temp_npz, _npz)
                    PosData.all_npz_paths[i] = _npz

                    print('Saving: ', tif)
                    temp_tif = self.getTempfilePath(tif)
                    self.imagej_tiffwriter(temp_tif, aligned_frames,
                                           metadata, PosData)
                    self.moveTempFile(temp_tif, tif)

        # Align segmentation data accordingly
        self.segmAligned = False
        if PosData.segmFound and aligned:
            if PosData.loaded_shifts is None or not align:
                return
            msg = QMessageBox()
            alignAnswer = msg.question(
                self, 'Align segmentation data?',
                'The system found an existing segmentation mask.\n\n'
                'Do you need to align that too?',
                msg.Yes | msg.No
            )
            if alignAnswer == msg.Yes:
                self.segmAligned = True
                print('Aligning: ', PosData.segm_npz_path)
                PosData.segm_data, shifts = core.align_frames_2D(
                                             PosData.segm_data,
                                             slices=None,
                                             user_shifts=PosData.loaded_shifts
                )
                print('Saving: ', PosData.segm_npz_path)
                temp_npz = self.getTempfilePath(PosData.segm_npz_path)
                np.savez_compressed(temp_npz, PosData.segm_data)
                self.moveTempFile(temp_npz, PosData.segm_npz_path)


    def detectTifAlignment(self, tif_data, PosData):
        numFramesWith0s = 0
        if PosData.SizeT == 1:
            tif_data = [tif_data]
        for img in tif_data:
            if PosData.SizeZ > 1:
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

    def warnTifAligned(self, numFramesWith0s, tifPath, PosData):
        proceed = True
        if numFramesWith0s>0 and PosData.loaded_shifts is not None:
            msg = QMessageBox()
            proceedAnswer = msg.warning(
               self, 'Tif data ALREADY aligned!',
               'The system detected that the .tif file contains ALREADY '
               'aligned data.\n\n'
               'Using the found "align_shifts.npy" file to align would result '
               'in misalignemnt of the data.\n\n'
               'Therefore, the alignment routine will re-calculate the shifts '
               'and it will NOT use the saved shifts.\n\n'
               'Do you want to continue?',
               msg.Yes | msg.Cancel
            )
            if proceedAnswer == msg.Cancel:
                proceed = False
        return proceed


    def npy_to_npz(self, PosData):
        PosData.all_npz_paths = PosData.npz_paths.copy()
        _zip = zip(PosData.npy_paths, PosData.npz_paths)
        for i, (npy, npz) in enumerate(_zip):
            if npz is None and npy is None:
                continue
            elif npy is not None and npz is None:
                print('Converting: ', npy)
                self.titleLabel.setText(
                    'Converting .npy to .npz... (check progress in terminal)',
                    color='w')
                _data = np.load(npy)
                _npz = f'{os.path.splitext(npy)[0]}.npz'
                temp_npz = self.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, _data)
                self.moveTempFile(temp_npz, _npz)
                os.remove(npy)
                PosData.all_npz_paths[i] = _npz
            elif npy is not None and npz is not None:
                os.remove(npy)
        # # Convert segm.npy to segm.npz
        # if PosData.segm_npz_path is not None:
        #     print('Converting: ', PosData.segm_npz_path)
        #     temp_npz = self.getTempfilePath(PosData.segm_npz_path)
        #     np.savez_compressed(temp_npz, PosData.segm_data)
        #     self.moveTempFile(temp_npz, PosData.segm_npz_path)
        #     os.remove(PosData.segm_npz_path)
        print(f'{PosData.relPath} done.')

    def getTempfilePath(self, path):
        temp_dirpath = tempfile.mkdtemp()
        filename = os.path.basename(path)
        tempFilePath = os.path.join(temp_dirpath, filename)
        return tempFilePath

    def moveTempFile(self, src, dst):
        print('Moving temp file: ', src)
        tempDir = os.path.dirname(src)
        shutil.move(src, dst)
        shutil.rmtree(tempDir)

    def getMostRecentPath(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
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

    def loadFiles(self, exp_path, user_ch_file_paths, user_ch_name):
        self.titleLabel.setText('Loading data...', color='w')
        self.setWindowTitle(f'Cell-ACDC - Data Prep. - "{exp_path}"')

        self.num_pos = len(user_ch_file_paths)

        proceed = self.init_data(user_ch_file_paths, user_ch_name)

        if not proceed:
            self.openAction.setEnabled(True)
            return

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()

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
        self.update_img()
        self.setFontSizeROIlabels()

    def initLoading(self):
        # Remove all items from a previous session if open is pressed again
        self.removeAllItems()
        self.gui_addPlotItems()

        self.setCenterAlignmentTitle()
        self.openAction.setEnabled(False)

    def openRecentFile(self, path):
        print(f'Opening recent folder: {path}')
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

        if os.path.basename(exp_path).find('Position_') != -1:
            is_pos_folder = True
        else:
            is_pos_folder = False

        if os.path.basename(exp_path).find('Images') != -1:
            is_images_folder = True
        else:
            is_images_folder = False

        self.titleLabel.setText('Loading data...', color='w')
        self.setWindowTitle(f'Cell-ACDC - Data Prep. - "{exp_path}"')
        self.setCenterAlignmentTitle()

        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )

        if not is_pos_folder and not is_images_folder:
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
        filenames = os.listdir(images_path)
        if ch_name_selector.is_first_call:
            ch_names, warn = (
                ch_name_selector.get_available_channels(filenames, images_path)
            )
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

        self.loadFiles(exp_path, user_ch_file_paths, user_ch_name)
        self.setCenterAlignmentTitle()

    def setFontSizeROIlabels(self):
        Y, X = self.img.image.shape
        factor = 40
        self.pt = int(X/factor)

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
        # cursor.select(QTextCursor.Document)
        # cursor.mergeBlockFormat(fmt)
        # cursor.clearSelection()
        # self.titleLabel.item.setTextCursor(cursor)

    def closeEvent(self, event):
        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()

        if self.loop is not None:
            self.loop.exit()

if __name__ == "__main__":
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    win = dataPrepWin()
    win.show()
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # Run the event loop
    sys.exit(app.exec_())
