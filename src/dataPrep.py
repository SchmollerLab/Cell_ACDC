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
from tifffile.tifffile import TiffWriter, TiffFile

from PyQt5.QtCore import Qt, QFile, QTextStream, QSize, QRect, QRectF
from PyQt5.QtGui import QIcon, QKeySequence, QCursor
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

pg.setConfigOptions(imageAxisOrder='row-major')

class dataPrep(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Yeast ACDC - data prep")
        self.setGeometry(100, 50, 850, 800)

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_addGraphicsItems()

        self.gui_createImgWidgets()
        self.num_frames = 0
        self.frame_i = 0

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
        self.nextAction.setShortcut("right")
        self.jumpForwardAction.setShortcut("up")
        self.jumpBackwardAction.setShortcut("down")

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
        # fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.exitAction)

    def gui_createToolBars(self):
        toolbarSize = 34

        # File toolbar
        fileToolBar = self.addToolBar("File")
        fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)

        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.startAction)
        fileToolBar.addAction(self.okAction)

        navigateToolbar = QToolBar("Navigate", self)
        navigateToolbar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolbar)

        navigateToolbar.addAction(self.prevAction)
        navigateToolbar.addAction(self.nextAction)
        navigateToolbar.addAction(self.jumpBackwardAction)
        navigateToolbar.addAction(self.jumpForwardAction)

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
        self.exitAction.triggered.connect(self.close)
        self.prevAction.triggered.connect(self.prev_frame)
        self.nextAction.triggered.connect(self.next_frame)
        self.jumpForwardAction.triggered.connect(self.skip10ahead_frames)
        self.jumpBackwardAction.triggered.connect(self.skip10back_frames)
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

        self.gui_connectGraphicsEvents()

    def removeAllItems(self):
        self.ax1.clear()
        self.frameLabel.setText(' ')

    def gui_connectGraphicsEvents(self):
        self.img.hoverEvent = self.gui_hoverEventImg

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

        self.img_Widglayout.addWidget(_t_label, 0, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.frame_i_scrollBar_img, 0, 1, 1, 20)

        self.img_Widglayout.addWidget(_z_label, 1, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.zSlice_scrollBar, 1, 1, 1, 20)

        self.img_Widglayout.setContentsMargins(100, 0, 50, 0)

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

    def update_img(self):
        self.frameLabel.setText(
                 f'Current frame = {self.frame_i+1}/{self.num_frames}')
        img = self.data.img_data[self.frame_i].copy()
        if self.data.SizeZ > 1:
            z = self.data.segmInfo_df.at[self.frame_i, 'z_slice_used_dataPrep']
            self.zSlice_scrollBar.setSliderPosition(z)
            self.z_label.setText(f'z-slice  {z}/{self.data.SizeZ}')
            img = img[z]
        img = img/img.max()
        self.img.setImage(img)

    def init_attr(self):

        self.frame_i = 0

        self.frame_i_scrollBar_img.setEnabled(True)
        self.frame_i_scrollBar_img.setMinimum(1)
        self.frame_i_scrollBar_img.setMaximum(self.num_frames)
        self.frame_i_scrollBar_img.setValue(1)

        self.frame_i_scrollBar_img.sliderMoved.connect(self.t_scrollbarMoved)

    def t_scrollbarMoved(self, t):
        self.frame_i = t-1
        self.update_img()

    def crop(self, data):
        x0, y0 = [int(round(c)) for c in self.roi.pos()]
        w, h = [int(round(c)) for c in self.roi.size()]
        if data.ndim == 4:
            croppedData = data[:, :, y0:y0+h, x0:x0+w]
        elif data.ndim == 3:
            croppedData = data[:, y0:y0+h, x0:x0+w]
        return croppedData

    def save(self):
        msg = QtGui.QMessageBox()
        save_current = msg.question(
            self, 'Save data?', 'Do you want to save?',
            msg.Yes | msg.No
        )
        if msg.Yes:
            self.okAction.setDisabled(True)
            print('Saving data...')
            self.titleLabel.setText(
                'Saving data... (check progress in the terminal)',
                color='w')

            if self.data.SizeZ > 1:
                # Save segmInfo
                self.data.segmInfo_df.to_csv(self.data.segmInfo_df_csv_path)

            # Get crop shape and print it
            data = self.data.img_data
            croppedData = self.crop(data)
            print('Cropped data shape: ', croppedData.shape)

            x0, y0 = [int(round(c)) for c in self.roi.pos()]
            w, h = [int(round(c)) for c in self.roi.size()]
            print(f'Saving crop ROI coords: x_left = {x0}, x_right = {x0+w}, '
                  f'y_top = {y0}, y_bottom = {y0+h}\n'
                  f'to {self.data.cropROI_coords_path}')

            with open(self.data.cropROI_coords_path, 'w') as csv:
                csv.write(f'x_left,{x0}\n'
                          f'x_right,{x0+w}\n'
                          f'y_top,{y0}\n'
                          f'y_bottom,{y0+h}')

            # Get metadata from tif
            with TiffFile(self.data.tif_path) as tif:
                metadata = tif.imagej_metadata


            # Save channels (npz AND tif)
            _zip = zip(self.data.tif_paths, self.npz_paths)
            for tif, npz in _zip:
                data = np.load(npz)['arr_0']
                npz_data = self.crop(data)
                print('Saving: ', npz)
                temp_npz = self.getTempfilePath(npz)
                np.savez_compressed(temp_npz, npz_data)
                self.moveTempFile(temp_npz, npz)
                print('Saving: ', tif)
                temp_tif = self.getTempfilePath(tif)
                self.imagej_tiffwriter(temp_tif, npz_data, metadata)
                self.moveTempFile(temp_tif, tif)

            # Save segm.npz
            if self.data.segm_found:
                print('Saving: ', self.data.segm_npz_path)
                data = self.data.segm_data
                croppedSegm = self.crop(data)
                temp_npz = self.getTempfilePath(self.data.segm_npz_path)
                np.savez_compressed(temp_npz, croppedSegm)
                self.moveTempFile(temp_npz, self.data.segm_npz_path)

            # Correct acdc_df if present and save
            if self.data.acdc_df is not None:
                print('Saving: ', self.data.acdc_output_csv_path)
                df = self.data.acdc_df
                df['x_centroid'] -= x0
                df['y_centroid'] -= y0
                df['editIDclicked_x'] -= x0
                df['editIDclicked_y'] -= y0
                try:
                    df.to_csv(self.data.acdc_output_csv_path)
                except PermissionError:
                    msg = QtGui.QMessageBox()
                    warn_cca = msg.critical(
                        self, 'Permission denied',
                        f'The below file is open in another app (Excel maybe?).\n\n'
                        f'{self.data.acdc_output_csv_path}\n\n'
                        'Close file and then press "Ok".',
                        msg.Ok
                    )
                    df.to_csv(self.data.acdc_output_csv_path)

            print('Done.')
            self.titleLabel.setText(
                'Saved! You can close the program or load another position.',
                color='g')

    def imagej_tiffwriter(self, new_path, data, metadata):
        with TiffWriter(new_path, imagej=True) as new_tif:
            if self.data.SizeZ > 1:
                T, Z, Y, X = data.shape
            else:
                T, Y, X = data.shape
                Z = 1
            data.shape = T, Z, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
            new_tif.save(data, metadata=metadata)

    def init_frames_data(self, frames_path, user_ch_name):
        data = load.load_frames_data(frames_path, user_ch_name,
                                     parentQWidget=self,
                                     load_segm_data=True,
                                     load_acdc_df=True,
                                     load_zyx_voxSize=False,
                                     load_all_imgData=True,
                                     load_shifts=True,
                                     loadSegmInfo=True)
        if data is None:
            self.titleLabel.setText(
                'File --> Open or Open recent to start the process',
                color='w')
            return

        # Allow single 2D/3D image
        if data.SizeT < 2:
            data.img_data = np.array([data.img_data])
            data.segm_data = np.array([data.segm_data])
        img_shape = data.img_data.shape
        self.num_frames = len(data.img_data)
        self.user_ch_name = user_ch_name
        SizeT = data.SizeT
        SizeZ = data.SizeZ
        print(f'Data shape = {img_shape}')
        print(f'Number of frames = {SizeT}')
        print(f'Number of z-slices per frame = {SizeZ}')

        self.data = data
        self.init_segmInfo_df()
        self.init_attr()

    def init_segmInfo_df(self):
        if self.data.segmInfo_df is None and self.data.SizeZ > 1:
            mid_slice = int(self.data.SizeZ/2)
            self.data.segmInfo_df = pd.DataFrame(
                {'frame_i': range(self.num_frames),
                 'z_slice_used_dataPrep': [mid_slice]*self.num_frames}
            ).set_index('frame_i')
        if self.data.SizeZ > 1:
            self.zSlice_scrollBar.setDisabled(False)
            self.zSlice_scrollBar.setMaximum(self.data.SizeZ)
            try:
                self.zSlice_scrollBar.sliderMoved.disconnect()
            except:
                pass
            self.zSlice_scrollBar.sliderMoved.connect(self.update_z_slice)
            self.interpAction.setEnabled(True)
            self.ZbackAction.setEnabled(True)
            self.ZforwAction.setEnabled(True)

    def update_z_slice(self, z):
        self.data.segmInfo_df.at[self.frame_i, 'z_slice_used_dataPrep'] = z
        self.update_img()

    def useSameZ_fromHereBack(self, event):
        z = self.data.segmInfo_df.at[self.frame_i, 'z_slice_used_dataPrep']
        self.data.segmInfo_df.loc[:self.frame_i-1, 'z_slice_used_dataPrep'] = z

    def useSameZ_fromHereForw(self, event):
        z = self.data.segmInfo_df.at[self.frame_i, 'z_slice_used_dataPrep']
        self.data.segmInfo_df.loc[self.frame_i:, 'z_slice_used_dataPrep'] = z

    def interp_z(self, event):
        x0, z0 = 0, self.data.segmInfo_df.at[0, 'z_slice_used_dataPrep']
        x1 = self.frame_i
        z1 = self.data.segmInfo_df.at[x1, 'z_slice_used_dataPrep']
        f = scipy.interpolate.interp1d([x0, x1], [z0, z1])
        xx = np.arange(0, self.frame_i)
        zz = np.round(f(xx)).astype(int)
        self.data.segmInfo_df.loc[:self.frame_i-1, 'z_slice_used_dataPrep'] = zz


    def prepData(self, event):
        self.startAction.setDisabled(True)
        nonTifFound = (
            any([npz is not None for npz in self.data.npz_paths]) or
            any([npy is not None for npy in self.data.npy_paths]) or
            self.data.segm_found
        )
        if nonTifFound:
            imagesPath = self.data.images_path
            zipPath = f'{os.path.splitext(imagesPath)[0]}.zip'
            msg = QtGui.QMessageBox()
            archive = msg.warning(
               self, 'NON-Tif data detected!',
               'Additional NON-tif files detected.\n\n'
               'The requested experiment folder already contains .npy or .npz files '
               'most likely from previous analysis runs.\n\n'
               'To avoid data losses I will now zip the "Images" folder.\n\n'
               'If everything looks fine after prepping the data, you can manually '
               'delete the zip archive.\n\n'
               'Zip archive location:\n\n'
               f'{zipPath}',
               msg.Ok | msg.Cancel
            )
            if archive == msg.Cancel:
                self.startAction.setDisabled(False)
                self.titleLabel.setText(
                    'Process aborted. Press "start" button to start again.',
                    color='w')
            print(f'Zipping Images folder: {zipPath}')
            shutil.make_archive(imagesPath, 'zip', imagesPath)
        self.npy_to_npz()
        self.alignData(self.user_ch_name)
        self.update_img()
        print('Done.')
        if self.data.SizeZ>1:
            self.data.segmInfo_df.to_csv(self.data.segmInfo_df_csv_path)
        self.addROIrect()
        self.okAction.setEnabled(True)
        self.titleLabel.setText(
            'Data successfully prepped. You can now crop the images or '
            'close the program',
            color='w')

    def setStandardRoiShape(self, text):
        _, Y, X = self.data.img_data.shape
        m = re.findall('(\d+)x(\d+)', text)
        w, h = int(m[0][0]), int(m[0][1])
        xc, yc = int(round(X/2)), int(round(Y/2))
        yl, xl = int(round(xc-w/2)), int(round(yc-h/2))
        self.roi.setPos([xl, yl])
        self.roi.setSize([w, h])

    def addROIrect(self):
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

        # Add ROI Rectangle
        roi = pg.ROI([xl, yl], [w, h],
                     rotatable=False,
                     removable=False,
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

        self.roi = roi
        self.roi.sigRegionChanged.connect(self.updateCurrentRoiShape)
        self.roi.sigRegionChangeFinished.connect(self.ROImovingFinished)

        self.ax1.addItem(roi)

    def ROImovingFinished(self, roi):
        roi.setPen(color='r')

    def updateCurrentRoiShape(self, roi):
        roi.setPen(color=(255,255,0))
        w, h = [int(round(c)) for c in self.roi.size()]
        self.ROIshapeLabel.setText(f'   Current ROI shape: {w} x {h}')

    def alignData(self, user_ch_name):
        """
        Alignemnt routine. Alignemnt is based on the data contained in the
        .tif file of the channel selected by the user (e.g. "phase_contr").
        Next, using the shifts calculated when aligning the channel selected
        by the user, it will align all the other channels, always starting from
        the data contained in the .tif files.

        In the end, aligned data will be saved to both the .tif file and a
        "_aligned.npz" file. The shifts will be saved to "align_shift.npy" file.

        Alignemnt is performed only if needed:

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
        with TiffFile(self.data.tif_path) as tif:
            metadata = tif.imagej_metadata

        print('Aligning data if needed...')
        self.titleLabel.setText(
            'Aligning data if needed... (check progress in terminal)',
            color='w')
        align = True
        if self.data.loaded_shifts is None:
            msg = QtGui.QMessageBox()
            alignAnswer = msg.question(
                self, 'Align frames?',
                f'Do you want to align ALL channels based on "{user_ch_name}" '
                'channel?\n\n'
                'NOTE that also the .tif files will contain aligned data and\n'
                f'a "..._{user_ch_name}_aligned.npz" file will be created \n'
                'anyway because of compatibility reasons.\n'
                'If you do not align it will contain '
                'NON-aligned data.\n\n'
                'If you do not have a specific reason for NOT align we reccommend '
                'aligning.',
                msg.Yes | msg.No
            )
            if alignAnswer == msg.No:
                align = False
                # Create 0, 0 shifts to perform 0 alignment
                self.data.loaded_shifts = np.zeros((self.num_frames,2), int)

        _zip = zip(self.data.tif_paths, self.data.npz_paths)
        aligned = False
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or self.data.loaded_shifts is None

            # Align based on user_ch_name
            if doAlign and tif.find(user_ch_name) != -1:
                aligned = True
                if align:
                    print('Aligning: ', tif)
                tif_data = skimage.io.imread(tif)
                numFramesWith0s = self.detectTifAlignment(tif_data)
                proceed = self.warnTifAligned(numFramesWith0s, tif)
                if not proceed:
                    break

                # Alignment routine
                if self.data.SizeZ>1:
                    align_func = core.align_frames_3D
                    zz = self.data.segmInfo_df['z_slice_used_dataPrep'].to_list()
                else:
                    align_func = core.align_frames_2D
                    zz = None
                aligned_frames, shifts = align_func(
                                          tif_data,
                                          slices=zz,
                                          user_shifts=self.data.loaded_shifts,
                                          pbar=True
                )
                self.data.loaded_shifts = shifts
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                print('Saving: ', _npz)
                temp_npz = self.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, aligned_frames)
                self.moveTempFile(temp_npz, _npz)
                np.save(self.data.align_shifts_path, shifts)
                self.npz_paths[i] = _npz

                print('Saving: ', tif)
                temp_tif = self.getTempfilePath(tif)
                self.imagej_tiffwriter(temp_tif, aligned_frames, metadata)
                self.moveTempFile(temp_tif, tif)
                self.data.img_data = skimage.io.imread(tif)

        _zip = zip(self.data.tif_paths, self.data.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            doAlign = npz is None or aligned
            # Align the other channels
            if doAlign and tif.find(user_ch_name) == -1:
                if self.data.loaded_shifts is None:
                    break
                if align:
                    print('Aligning: ', tif)
                tif_data = skimage.io.imread(tif)

                # Alignment routine
                if self.data.SizeZ>1:
                    align_func = core.align_frames_3D
                    zz = self.data.segmInfo_df['z_slice_used_dataPrep'].to_list()
                else:
                    align_func = core.align_frames_2D
                    zz = None
                aligned_frames, shifts = align_func(
                                          tif_data,
                                          slices=zz,
                                          user_shifts=self.data.loaded_shifts
                )
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                print('Saving: ', _npz)
                temp_npz = self.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, aligned_frames)
                self.moveTempFile(temp_npz, _npz)
                self.npz_paths[i] = _npz

                print('Saving: ', tif)
                temp_tif = self.getTempfilePath(tif)
                self.imagej_tiffwriter(temp_tif, aligned_frames, metadata)
                self.moveTempFile(temp_tif, tif)

        # Align segmentation data accordingly
        if self.data.segm_found and aligned:
            if self.data.loaded_shifts is None:
                return
            msg = QtGui.QMessageBox()
            alignAnswer = msg.question(
                self, 'Align segmentation data?',
                'The system found an existing segmentation mask.\n\n'
                'Do you need to align that too?',
                msg.Yes | msg.No
            )
            if alignAnswer == msg.Yes:
                print('Aligning: ', self.data.segm_npz_path)
                self.data.segm_data, shifts = core.align_frames_2D(
                                             self.data.segm_data,
                                             slices=None,
                                             user_shifts=self.data.loaded_shifts
                )
                print('Saving: ', self.data.segm_npz_path)
                temp_npz = self.getTempfilePath(self.data.segm_npz_path)
                np.savez_compressed(temp_npz, self.data.segm_data)
                self.moveTempFile(temp_npz, self.data.segm_npz_path)


    def detectTifAlignment(self, tif_data):
        numFramesWith0s = 0
        if self.data.SizeT == 1:
            tif_data = [tif_data]
        for img in tif_data:
            if self.data.SizeZ > 1:
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

    def warnTifAligned(self, numFramesWith0s, tifPath):
        proceed = True
        if numFramesWith0s>0 and self.data.loaded_shifts is not None:
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


    def npy_to_npz(self):
        print('Converting .npy to .npz if needed...')
        self.titleLabel.setText(
            'Converting .npy to .npz if needed... (check progress in terminal)',
            color='w')
        self.npz_paths = self.data.npz_paths.copy()
        _zip = zip(self.data.npy_paths, self.data.npz_paths)
        for i, (npy, npz) in enumerate(_zip):
            if npz is None and npy is None:
                continue
            elif npy is not None and npz is None:
                print('Converting: ', npy)
                _data = np.load(npy)
                _npz = f'{os.path.splitext(npy)[0]}.npz'
                temp_npz = self.getTempfilePath(_npz)
                np.savez_compressed(temp_npz, _data)
                self.moveTempFile(temp_npz, _npz)
                os.remove(npy)
                self.npz_paths[i] = _npz
            elif npy is not None and npz is not None:
                os.remove(npy)
        # Convert segm.npy to segm.npz
        if self.data.segm_npy_path is not None:
            print('Converting: ', self.data.segm_npy_path)
            temp_npz = self.getTempfilePath(self.data.segm_npz_path)
            np.savez_compressed(temp_npz, self.data.segm_data)
            self.moveTempFile(temp_npz, self.data.segm_npz_path)
            os.remove(self.data.segm_npy_path)
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
        else:
            self.MostRecentPath = ''

    def addToRecentPaths(self, exp_path):
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
                           'opened_last_on': openedOn})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)


    def openFile(self, checked=False, exp_path=None):
        # Remove all items from a previous session if open is pressed again
        self.removeAllItems()
        self.gui_addPlotItems()

        self.openAction.setEnabled(False)

        if exp_path is None:
            self.getMostRecentPath()
            exp_path = QFileDialog.getExistingDirectory(
                self, 'Select experiment folder containing Position_n folders'
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

            pos_foldername = select_folder.selected_pos[0]
            images_path = f'{exp_path}/{pos_foldername}/Images'

            if select_folder.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                return

        elif is_pos_folder:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_path = f'{exp_path}/{pos_foldername}/Images'

        elif is_images_folder:
            images_path = exp_path

        self.images_path = images_path

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
        print(f'Loading {img_path}...')

        self.init_frames_data(img_path, user_ch_name)

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()

        self.titleLabel.setText(
            'Data successfully loaded. Right/Left arrow to navigate frames',
            color='w')

        self.openAction.setEnabled(True)
        self.startAction.setEnabled(True)
        self.update_img()

if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    win = dataPrep()
    win.show()
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # Run the event loop
    sys.exit(app.exec_())
