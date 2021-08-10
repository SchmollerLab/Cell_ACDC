import os
import sys
import re
import time
import datetime
import tempfile
import shutil
import numpy as np
import pandas as pd
import scipy.interpolate
import skimage.io
from functools import partial
from tifffile.tifffile import TiffWriter, TiffFile

from PyQt5.QtCore import (
    Qt, QFile, QTextStream, QSize, QRect, QRectF, QObject, QThread, pyqtSignal
)
from PyQt5.QtGui import (
    QIcon, QKeySequence, QCursor, QTextBlockFormat,
    QTextCursor
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton,
    QMainWindow, QMenu, QToolBar, QGroupBox,
    QScrollBar, QCheckBox, QToolButton, QSpinBox,
    QComboBox, QDial, QButtonGroup, QFileDialog
)

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# NOTE: Enable icons
import qrc_resources

# Custom modules
import load, prompts, apps, core

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
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
    def __init__(self, parent=None, buttonToRestore=None):
        super().__init__(parent)

        self.buttonToRestore = buttonToRestore

        self.setWindowTitle("Yeast ACDC - data prep")
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
        self.cropROI = None

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


        self.okAction = QAction(QIcon(":applyCrop.svg"), "Crop!", self)
        self.okAction.setEnabled(False)
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
        fileToolBar.addAction(self.okAction)

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
        self.ROIshapeComboBox.addItems(['256x256'])
        ROIshapeLabel = QLabel('   ROI standard shape: ')
        ROIshapeLabel.setBuddy(self.ROIshapeComboBox)
        navigateToolbar.addWidget(ROIshapeLabel)
        navigateToolbar.addWidget(self.ROIshapeComboBox)

        self.ROIshapeLabel = QLabel('   Current ROI shape: 256 x 256')
        navigateToolbar.addWidget(self.ROIshapeLabel)

    def gui_connectActions(self):
        self.openAction.triggered.connect(self.openFile)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.exitAction.triggered.connect(self.close)
        self.prevAction.triggered.connect(self.prev_cb)
        self.nextAction.triggered.connect(self.next_cb)
        self.showInExplorerAction.triggered.connect(self.showInExplorer)
        self.jumpForwardAction.triggered.connect(self.skip10ahead_cb)
        self.jumpBackwardAction.triggered.connect(self.skip10back_cb)
        self.addBkrgRoiActon.triggered.connect(self.addBkgrRoi)
        self.okAction.triggered.connect(self.save)
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

        self.frame_i_scrollBar_img = QScrollBar(Qt.Horizontal)
        self.frame_i_scrollBar_img.setFixedHeight(20)
        self.frame_i_scrollBar_img.setDisabled(True)
        _t_label = QLabel('frame n.  ')
        _t_label.setFont(_font)


        self.zSlice_scrollBar = QScrollBar(Qt.Horizontal)
        self.zSlice_scrollBar.setFixedHeight(20)
        self.zSlice_scrollBar.setDisabled(True)
        _z_label = QLabel('z-slice  ')
        _z_label.setFont(_font)
        self.z_label = _z_label

        self.zProjComboBox = QComboBox()
        self.zProjComboBox.addItems(['single z-slice',
                                     'max z-projection',
                                     'mean z-projection',
                                     'median z-proj.'])
        self.zProjComboBox.setDisabled(True)

        self.img_Widglayout.addWidget(_t_label, 0, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.frame_i_scrollBar_img, 0, 1, 1, 30)

        self.img_Widglayout.addWidget(_z_label, 1, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.zSlice_scrollBar, 1, 1, 1, 30)

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
        except:
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
            self.pos_i += 1
        else:
            self.pos_i = 0
        self.update_img()

    def prev_pos(self):
        if self.pos_i > 0:
            self.pos_i -= 1
        else:
            self.pos_i = self.num_pos-1
        self.update_img()

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
        self.frame_i_scrollBar_img.setValue(self.frame_i+1)
        self.update_img()

    def prev_frame(self):
        if self.frame_i > 0:
            self.frame_i -= 1
        else:
            self.frame_i = self.num_frames-1
        self.frame_i_scrollBar_img.setValue(self.frame_i+1)
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

    def updateFramePosLabel(self):
        PosData = self.data[self.pos_i]
        if self.num_pos > 1:
            self.frameLabel.setText(
                     f'Current position = {self.pos_i+1}/{self.num_pos} '
                     f'({PosData.pos_foldername})')
        else:
            self.frameLabel.setText(
                     f'Current frame = {self.frame_i+1}/{self.num_frames}')

    def getImage(self, PosData, frames, frame_i):
        img = frames[frame_i].copy()
        if PosData.SizeZ > 1:
            z = PosData.segmInfo_df.at[frame_i,
                                       'z_slice_used_dataPrep']
            zProjHow = PosData.segmInfo_df.at[frame_i,
                                              'which_z_proj']
            try:
                self.zProjComboBox.currentTextChanged.disconnect()
            except TypeError:
                pass
            self.zProjComboBox.setCurrentText(zProjHow)
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)

            if zProjHow == 'single z-slice':
                self.zSlice_scrollBar.setSliderPosition(z)
                self.z_label.setText(f'z-slice  {z}/{PosData.SizeZ}')
                img = img[z]
            elif zProjHow == 'max z-projection':
                img = img.max(axis=0)
            elif zProjHow == 'mean z-projection':
                img = img.mean(axis=0)
            elif zProjHow == 'median z-proj.':
                img = np.median(img, axis=0)
        return img

    def update_img(self):
        self.updateFramePosLabel()
        PosData = self.data[self.pos_i]
        img = self.getImage(PosData, PosData.img_data, self.frame_i)
        img = img/img.max()
        self.img.setImage(img)

    def init_attr(self):
        self.bkgrROIs = []
        if self.num_pos > 1:
            self.frame_i_scrollBar_img.setEnabled(False)
        else:
            self.frame_i_scrollBar_img.setEnabled(True)
            self.frame_i_scrollBar_img.setMinimum(1)
            self.frame_i_scrollBar_img.setMaximum(self.num_frames)
            self.frame_i_scrollBar_img.setValue(1)
            self.frame_i_scrollBar_img.sliderMoved.connect(self.t_scrollbarMoved)

    def t_scrollbarMoved(self, t):
        self.frame_i = t-1
        self.update_img()

    def crop(self, data):
        x0, y0 = [int(round(c)) for c in self.cropROI.pos()]
        w, h = [int(round(c)) for c in self.cropROI.size()]
        if data.ndim == 4:
            croppedData = data[:, :, y0:y0+h, x0:x0+w]
        elif data.ndim == 3:
            croppedData = data[:, y0:y0+h, x0:x0+w]
        return croppedData

    def saveBkgrValues(self, PosData):
        if not self.bkgrROIs:
            return

        self.bkgrMask = np.zeros(self.img.image.shape, bool)
        for roi in self.bkgrROIs:
            xl, yl = [int(round(c)) for c in roi.pos()]
            w, h = [int(round(c)) for c in roi.size()]
            self.bkgrMask[xl:xl+w, yl:yl+h] = True

        channel_names = []
        frame_idxs = []
        bkgr_medians = []
        bkgr_means = []
        bkgr_q05s = []
        bkgr_q25s = []
        bkgr_q75s = []
        bkgr_q95s = []
        bkgr_stds = []
        bkgr_sampleSizes = []
        for tif_path in PosData.tif_paths:
            frames = skimage.io.imread(tif_path)
            if PosData.SizeT < 2:
                frames = [frames]
            filename = os.path.basename(tif_path)
            filename_noEXT, _ = os.path.splitext(filename)
            channel_name = filename_noEXT[len(PosData.basename)+1:]
            for frame_i in range(len(frames)):
                img = self.getImage(PosData, frames, frame_i)
                channel_names.append(channel_name)
                frame_idxs.append(frame_i)
                bkgrMask = self.bkgrMask.copy()
                # Remove from background mask those padded 0s from alignment
                bkgrMask[img == 0] = False
                bkgr_vals = img[self.bkgrMask]
                bkgr_medians.append(np.median(bkgr_vals))
                bkgr_means.append(np.mean(bkgr_vals))
                bkgr_stds.append(np.std(bkgr_vals))
                bkgr_q05s.append(np.quantile(bkgr_vals, q=0.05))
                bkgr_q25s.append(np.quantile(bkgr_vals, q=0.25))
                bkgr_q75s.append(np.quantile(bkgr_vals, q=0.75))
                bkgr_q95s.append(np.quantile(bkgr_vals, q=0.95))
                bkgr_sampleSizes.append(len(bkgr_vals))
        df = pd.DataFrame({
                'channel_name': channel_names,
                'frame_i': frame_idxs,
                'bkgr_median': bkgr_medians,
                'bkgr_mean': bkgr_means,
                'bkgr_std': bkgr_stds,
                'bkgr_q05': bkgr_q05s,
                'bkgr_q25': bkgr_q25s,
                'bkgr_q75': bkgr_q75s,
                'bkgr_q95': bkgr_q95s,
                'bkgr_sampleSize': bkgr_sampleSizes
        })
        df.to_csv(PosData.dataPrepBkgrValues_path, index=False)

    def removeROI(self, event):
        self.bkgrROIs.remove(self.roi_to_del)
        self.ax1.removeItem(self.roi_to_del.label)
        self.ax1.removeItem(self.roi_to_del)

    def gui_mousePressEventImg(self, event):
        PosData = self.data[self.pos_i]
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

        if left_click:
            pg.ImageItem.mousePressEvent(self.img, event)

        x, y = event.pos().x(), event.pos().y()

        # Check if right click on ROI
        for r, roi in enumerate(self.bkgrROIs):
            handleSize = 7
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
                action = QAction('Remove ROI')
                action.triggered.connect(self.removeROI)
                self.roiContextMenu.addAction(action)
                self.roiContextMenu.exec_(event.screenPos())
            elif dragRoi:
                event.ignore()
                return

        if self.cropROI is not None:
            x0, y0 = [int(c) for c in self.cropROI.pos()]
            w, h = [int(c) for c in self.cropROI.size()]
            x1, y1 = x0+w, y0+h
            clickedOnROI = (
                x>=x0-handleSize and x<=x1+handleSize
                and y>=y0-handleSize and y<=y1+handleSize
            )
            dragRoi = left_click and clickedOnROI
            if dragRoi:
                event.ignore()
                return

    def save(self):
        msg = QtGui.QMessageBox()
        doSave = msg.question(
            self, 'Save data?', 'Do you want to save?',
            msg.Yes | msg.No
        )
        if doSave == msg.No:
            return

        for PosData in self.data:
            self.okAction.setDisabled(True)
            print('Saving data...')
            self.titleLabel.setText(
                'Saving data... (check progress in the terminal)',
                color='w')

            self.saveBkgrValues(PosData)

            if PosData.SizeZ > 1:
                # Save segmInfo
                PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

            # Get crop shape and print it
            data = PosData.img_data
            croppedData = self.crop(data)
            print('Cropped data shape: ', croppedData.shape)

            x0, y0 = [int(round(c)) for c in self.cropROI.pos()]
            w, h = [int(round(c)) for c in self.cropROI.size()]
            print(f'Saving crop ROI coords: x_left = {x0}, x_right = {x0+w}, '
                  f'y_top = {y0}, y_bottom = {y0+h}\n'
                  f'to {PosData.dataPrepROIs_coords_path}')

            with open(PosData.dataPrepROIs_coords_path, 'w') as csv:
                csv.write(f'x_left,{x0}\n'
                          f'x_right,{x0+w}\n'
                          f'y_top,{y0}\n'
                          f'y_bottom,{y0+h}')

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

                npz_data = self.crop(data)

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
            if PosData.segm_found and self.segmAligned:
                print('Saving: ', PosData.segm_npz_path)
                data = PosData.segm_data
                croppedSegm = self.crop(data)
                temp_npz = self.getTempfilePath(PosData.segm_npz_path)
                np.savez_compressed(temp_npz, croppedSegm)
                self.moveTempFile(temp_npz, PosData.segm_npz_path)

            # Correct acdc_df if present and save
            if PosData.acdc_df is not None:
                print('Saving: ', PosData.acdc_output_csv_path)
                df = PosData.acdc_df
                df['x_centroid'] -= x0
                df['y_centroid'] -= y0
                df['editIDclicked_x'] -= x0
                df['editIDclicked_y'] -= y0
                try:
                    df.to_csv(PosData.acdc_output_csv_path)
                except PermissionError:
                    msg = QtGui.QMessageBox()
                    warn_cca = msg.critical(
                        self, 'Permission denied',
                        f'The below file is open in another app (Excel maybe?).\n\n'
                        f'{PosData.acdc_output_csv_path}\n\n'
                        'Close file and then press "Ok".',
                        msg.Ok
                    )
                    df.to_csv(PosData.acdc_output_csv_path)

            print(f'{PosData.pos_foldername} saved!')
            print(f'--------------------------------')
            print('')
        self.titleLabel.setText(
            'Saved! You can close the program or load another position.',
            color='g')

    def imagej_tiffwriter(self, new_path, data, metadata, PosData):
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

    def init_data(self, user_ch_file_paths, user_ch_name):
        # Iterate pos and load_data
        data = []
        for f, file_path in enumerate(user_ch_file_paths):
            try:
                PosData = load.load_frames_data(
                                         file_path, user_ch_name,
                                         parentQWidget=self,
                                         load_segm_data=True,
                                         load_acdc_df=True,
                                         load_zyx_voxSize=False,
                                         load_all_imgData=True,
                                         load_shifts=True,
                                         loadSegmInfo=True,
                                         first_call=f==0)
            except AttributeError:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return False

            if PosData is None:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return False

            # Allow single 2D/3D image
            if PosData.SizeT < 2:
                PosData.img_data = np.array([PosData.img_data])
                PosData.segm_data = np.array([PosData.segm_data])
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
                msg = QtGui.QMessageBox()
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
            if PosData.segmInfo_df is None and PosData.SizeZ > 1:
                mid_slice = int(PosData.SizeZ/2)
                PosData.segmInfo_df = pd.DataFrame(
                    {'frame_i': range(self.num_frames),
                     'z_slice_used_dataPrep': [mid_slice]*self.num_frames,
                     'which_z_proj': ['single z-slice']*self.num_frames}
                ).set_index('frame_i')
                PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

        PosData = self.data[0]
        if PosData.SizeZ > 1:
            self.zSlice_scrollBar.setDisabled(False)
            self.zProjComboBox.setDisabled(False)
            self.zSlice_scrollBar.setMaximum(PosData.SizeZ-1)
            try:
                self.zSlice_scrollBar.sliderMoved.disconnect()
                self.zProjComboBox.currentTextChanged.disconnect()
            except:
                pass
            self.zSlice_scrollBar.sliderMoved.connect(self.update_z_slice)
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
            if PosData.SizeT > 1:
                self.interpAction.setEnabled(True)
                self.ZbackAction.setEnabled(True)
                self.ZforwAction.setEnabled(True)
            how = PosData.segmInfo_df.at[self.frame_i, 'which_z_proj']
            self.zProjComboBox.setCurrentText(how)

    def update_z_slice(self, z):
        if self.zProjComboBox.currentText() == 'single z-slice':
            PosData = self.data[self.pos_i]
            PosData.segmInfo_df.at[self.frame_i, 'z_slice_used_dataPrep'] = z
            self.update_img()
            PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

    def updateZproj(self, how):
        PosData = self.data[self.pos_i]
        for frame_i in range(self.frame_i, PosData.SizeT):
            PosData.segmInfo_df.at[frame_i, 'which_z_proj'] = how
        if how == 'single z-slice':
            self.zSlice_scrollBar.setDisabled(False)
            self.z_label.setStyleSheet('color: black')
            self.update_z_slice(self.zSlice_scrollBar.sliderPosition())
        else:
            self.zSlice_scrollBar.setDisabled(True)
            self.z_label.setStyleSheet('color: gray')
            self.update_img()

        # Apply same z-proj to future pos
        if PosData.SizeT == 1:
            for PosData in self.data[self.pos_i+1:]:
                PosData.segmInfo_df.at[0, 'which_z_proj'] = how

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
        z = PosData.segmInfo_df.at[self.frame_i, 'z_slice_used_dataPrep']
        PosData.segmInfo_df.loc[:self.frame_i-1,
                                      'z_slice_used_dataPrep'] = z
        PosData.segmInfo_df.loc[:self.frame_i-1,
                                      'zProjHow'] = how

    def useSameZ_fromHereForw(self, event):
        how = self.zProjComboBox.currentText()
        PosData = self.data[self.pos_i]
        z = PosData.segmInfo_df.at[self.frame_i, 'z_slice_used_dataPrep']
        PosData.segmInfo_df.loc[self.frame_i:,
                                      'z_slice_used_dataPrep'] = z
        PosData.segmInfo_df.loc[:self.frame_i-1,
                                      'zProjHow'] = how

    def interp_z(self, event):
        PosData = self.data[self.pos_i]
        x0, z0 = 0, PosData.segmInfo_df.at[0, 'z_slice_used_dataPrep']
        x1 = self.frame_i
        z1 = PosData.segmInfo_df.at[x1, 'z_slice_used_dataPrep']
        f = scipy.interpolate.interp1d([x0, x1], [z0, z1])
        xx = np.arange(0, self.frame_i)
        zz = np.round(f(xx)).astype(int)
        PosData.segmInfo_df.loc[:self.frame_i-1,
                                      'z_slice_used_dataPrep'] = zz
        PosData.segmInfo_df.loc[:self.frame_i-1,
                                      'zProjHow'] = 'single z-slice'


    def prepData(self, event):
        self.titleLabel.setText(
            'Prepping data... (check progress in the terminal)',
            color='w')
        doZip = True
        for p, PosData in enumerate(self.data):
            self.startAction.setDisabled(True)
            nonTifFound = (
                any([npz is not None for npz in PosData.npz_paths]) or
                any([npy is not None for npy in PosData.npy_paths]) or
                PosData.segm_found
            )
            if nonTifFound and p==0:
                imagesPath = PosData.images_path
                zipPath = f'{os.path.splitext(imagesPath)[0]}.zip'
                msg = QtGui.QMessageBox()
                archive = msg.warning(
                   self, 'NON-Tif data detected!',
                   'Additional NON-tif files detected.\n\n'
                   'The requested experiment folder already contains .npy '
                   'or .npz files '
                   'most likely from previous analysis runs.\n\n'
                   'To avoid data losses we reccomend zipping the "Images" folder.\n\n'
                   'If everything looks fine after prepping the data, '
                   'you can manually '
                   'delete the zip archive.\n\n'
                   'Do you want to automatically zip now?\n\n'
                   'PS: Zip archive location:\n\n'
                   f'{zipPath}',
                   msg.Yes | msg.No
                )
                if archive == msg.Yes:
                    print(f'Zipping Images folder: {zipPath}')
                    shutil.make_archive(imagesPath, 'zip', imagesPath)
            self.npy_to_npz(PosData)
            self.alignData(self.user_ch_name, PosData)
            if PosData.SizeZ>1:
                PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

        self.update_img()
        print('Done.')
        self.addROIs()
        self.okAction.setEnabled(True)
        self.titleLabel.setText(
            'Data successfully prepped. You can now crop the images or '
            'close the program',
            color='w')

    def setStandardRoiShape(self, text):
        PosData = self.data[self.pos_i]
        Y, X = PosData.img_data.shape[-2:]
        m = re.findall('(\d+)x(\d+)', text)
        w, h = int(m[0][0]), int(m[0][1])
        xc, yc = int(round(X/2)), int(round(Y/2))
        yl, xl = int(round(xc-w/2)), int(round(yc-h/2))
        self.cropROI.setPos([xl, yl])
        self.cropROI.setSize([w, h])

    def addROIs(self):
        Y, X = self.img.image.shape

        max_size = round(int(np.log2(min([Y, X])/16)))
        items = [f'{16*(2**i)}x{16*(2**i)}' for i in range(1, max_size+1)]
        items.append(f'{X}x{Y}')
        self.ROIshapeComboBox.clear()
        self.ROIshapeComboBox.addItems(items)
        if len(items) > 3:
            self.ROIshapeComboBox.setCurrentText(items[3])
        else:
            self.ROIshapeComboBox.setCurrentText(items[-1])
        try:
            self.ROIshapeComboBox.currentTextChanged.disconnect()
        except:
            self.ROIshapeComboBox.currentTextChanged.connect(
                                                      self.setStandardRoiShape)

        if len(items) > 3:
            w, h = 256, 256
        else:
            w, h = X, Y
        xc, yc = int(round(X/2)), int(round(Y/2))
        yl, xl = int(round(xc-w/2)), int(round(yc-h/2))

        # Add crop ROI Rectangle
        cropROI = pg.ROI([xl, yl], [w, h],
                     rotatable=False,
                     removable=False,
                     pen=pg.mkPen(color='r'),
                     maxBounds=QRectF(QRect(0,0,X,Y)))

        cropROI.handleSize = 7
        cropROI.label = pg.LabelItem('Crop ROI', color='r', size='12pt')
        hLabel = cropROI.label.rect().bottom()
        cropROI.label.setPos(xl, yl-hLabel)

        ## handles scaling horizontally around center
        cropROI.addScaleHandle([1, 0.5], [0, 0.5])
        cropROI.addScaleHandle([0, 0.5], [1, 0.5])

        ## handles scaling vertically from opposite edge
        cropROI.addScaleHandle([0.5, 0], [0.5, 1])
        cropROI.addScaleHandle([0.5, 1], [0.5, 0])

        ## handles scaling both vertically and horizontally
        cropROI.addScaleHandle([1, 1], [0, 0])
        cropROI.addScaleHandle([0, 0], [1, 1])

        self.cropROI = cropROI
        self.cropROI.sigRegionChanged.connect(self.updateCurrentRoiShape)
        self.cropROI.sigRegionChangeFinished.connect(self.ROImovingFinished)

        self.ax1.addItem(cropROI)
        self.ax1.addItem(cropROI.label)

        self.addBkrgRoiActon.setDisabled(False)
        self.bkgrROIs = []
        self.addBkgrRoi()

    def addBkgrRoi(self, checked=False):
        # Add bkgr ROI Rectangle
        Y, X = self.img.image.shape
        xRange, yRange = self.ax1.viewRange()
        xl, yl = abs(xRange[0]), abs(yRange[0])
        w, h = int(X/8), int(Y/8)
        bkgrROI = pg.ROI([xl, yl], [w, h],
                     rotatable=False,
                     removable=False,
                     pen=pg.mkPen(color=(150,150,150)),
                     maxBounds=QRectF(QRect(0,0,X,Y)))

        bkgrROI.handleSize = 7
        bkgrROI.label = pg.LabelItem('Bkgr. ROI', color=(150,150,150),
                                                       size='12pt')
        hLabel = bkgrROI.label.rect().bottom()
        bkgrROI.label.setPos(xl, yl-hLabel)

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

        self.bkgrROI = bkgrROI
        self.bkgrROI.sigRegionChanged.connect(self.bkgrROIMoving)
        self.bkgrROI.sigRegionChangeFinished.connect(self.bkgrROImovingFinished)

        self.ax1.addItem(bkgrROI)
        self.ax1.addItem(bkgrROI.label)

        self.bkgrROIs.append(self.bkgrROI)

    def bkgrROIMoving(self, roi):
        txt = roi.label.text
        roi.setPen(color=(255,255,0))
        roi.label.setText(txt, color=(255,255,0), size='12pt')
        xl, yl = [int(round(c)) for c in roi.pos()]
        hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yl-hLabel)

    def bkgrROImovingFinished(self, roi):
        txt = roi.label.text
        roi.setPen(color=(150,150,150))
        roi.label.setText(txt, color=(150,150,150), size='12pt')

    def ROImovingFinished(self, roi):
        txt = roi.label.text
        roi.setPen(color='r')
        roi.label.setText(txt, color='r', size='12pt')

    def updateCurrentRoiShape(self, roi):
        roi.setPen(color=(255,255,0))
        roi.label.setText('Crop ROI', color=(255,255,0), size='12pt')
        xl, yl = [int(round(c)) for c in roi.pos()]
        hLabel = roi.label.rect().bottom()
        roi.label.setPos(xl, yl-hLabel)
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
            msg = QtGui.QMessageBox()
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
                    zz = PosData.segmInfo_df['z_slice_used_dataPrep'].to_list()
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
                print(tif_data.shape)
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
                if PosData.SizeT < 2:
                    PosData.img_data = np.array([skimage.io.imread(tif)])
                else:
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
                    zz = PosData.segmInfo_df['z_slice_used_dataPrep'].to_list()
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
        if PosData.segm_found and aligned:
            if PosData.loaded_shifts is None or not align:
                return
            msg = QtGui.QMessageBox()
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
            msg = QtGui.QMessageBox()
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
        # Convert segm.npy to segm.npz
        if PosData.segm_npy_path is not None:
            print('Converting: ', PosData.segm_npy_path)
            temp_npz = self.getTempfilePath(PosData.segm_npz_path)
            np.savez_compressed(temp_npz, PosData.segm_data)
            self.moveTempFile(temp_npz, PosData.segm_npz_path)
            os.remove(PosData.segm_npy_path)
        print('Done.')

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
        self.setWindowTitle(f'Yeast_ACDC - Data Prep. - "{exp_path}"')

        self.num_pos = len(user_ch_file_paths)
        proceed = self.init_data(user_ch_file_paths, user_ch_name)

        if not proceed:
            self.openAction.setEnabled(True)
            return

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()

        if self.titleText is None:
            self.titleLabel.setText(
                'Data successfully loaded. '
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

    def initLoading(self):
        # Remove all items from a previous session if open is pressed again
        self.removeAllItems()
        self.gui_addPlotItems()

        self.setCenterAlignmentTitle()
        self.openAction.setEnabled(False)

    def openRecentFile(self, path):
        print(f'Opening recent folder: {path}')
        self.openFile(exp_path=path)

    def openFile(self, checked=False, exp_path=None):
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
        self.setWindowTitle(f'Yeast_ACDC - Data Prep. - "{exp_path}"')
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
                msg = QtGui.QMessageBox()
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
        ch_name_not_found_msg = (
            'The script could not identify the channel name.\n\n'
            'For automatic loading the file to be segmented MUST have a name like\n'
            '"<name>_s<num>_<channel_name>.tif" e.g. "196_s16_phase_contrast.tif"\n'
            'where "196_s16" is the basename and "phase_contrast"'
            'is the channel name\n\n'
            'Please write here the channel name to be used for automatic loading'
        )

        filenames = os.listdir(images_path)
        if ch_name_selector.is_first_call:
            ch_names, warn = ch_name_selector.get_available_channels(filenames)
            ch_name_selector.QtPrompt(self, ch_names)
            if ch_name_selector.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                return
            if warn:
                user_ch_name = prompts.single_entry_messagebox(
                    title='Channel name not found',
                    entry_label=ch_name_not_found_msg,
                    input_txt=ch_name_selector.channel_name,
                    toplevel=False, allow_abort=False
                ).entry_txt
                if user_ch_name.was_aborted:
                    self.titleLabel.setText(
                        'File --> Open or Open recent to start the process',
                        color='w')
                    self.openAction.setEnabled(True)
                    return
            else:
                user_ch_name = ch_name_selector.channel_name

        user_ch_file_paths = []
        for images_path in self.images_paths:
            img_aligned_found = False
            for filename in os.listdir(images_path):
                if filename.find(f'_phc_aligned.npy') != -1:
                    img_path = f'{images_path}/{filename}'
                    new_filename = filename.replace('phc_aligned.npy',
                                                f'{user_ch_name}_aligned.npy')
                    dst = f'{images_path}/{new_filename}'
                    if os.path.exists(dst):
                        os.remove(img_path)
                    else:
                        os.rename(img_path, dst)
                    filename = new_filename
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

        self.loadFiles(exp_path, user_ch_file_paths, user_ch_name)
        self.setCenterAlignmentTitle()

    def setCenterAlignmentTitle(self):
        self.titleLabel.item.setTextWidth(self.img.width())
        fmt = QTextBlockFormat()
        fmt.setAlignment(Qt.AlignHCenter)
        cursor = self.titleLabel.item.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.mergeBlockFormat(fmt)
        cursor.clearSelection()
        self.titleLabel.item.setTextCursor(cursor)

    def closeEvent(self, event):
        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')

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
