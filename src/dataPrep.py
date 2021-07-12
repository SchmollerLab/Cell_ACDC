import os
import sys
import re
import numpy as np
import skimage.io
from tifffile.tifffile import TiffWriter, TiffFile

from PyQt5.QtCore import Qt, QFile, QTextStream, QSize, QRect, QRectF
from PyQt5.QtGui import QIcon, QKeySequence, QCursor
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton,
    QMainWindow, QMenu, QToolBar, QGroupBox,
    QScrollBar, QCheckBox, QToolButton, QSpinBox,
    QComboBox, QDial, QButtonGroup
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
        self.setWindowTitle("Yeast ACDC - crop ROIs")
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
        self.okAction.triggered.connect(self.crop_and_save)
        self.startAction.triggered.connect(self.prepData)

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


        self.zSlice_scrollBar_img = QScrollBar(Qt.Horizontal)
        self.zSlice_scrollBar_img.setFixedHeight(20)
        self.zSlice_scrollBar_img.setDisabled(True)
        _z_label = QLabel('z-slice  ')
        _z_label.setFont(_font)

        self.img_Widglayout.addWidget(_t_label, 0, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.frame_i_scrollBar_img, 0, 1, 1, 20)

        self.img_Widglayout.addWidget(_z_label, 1, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.zSlice_scrollBar_img, 1, 1, 1, 20)

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
        self.frame_i_scrollBar_img.setValue(self.frame_i)
        self.update_img()

    def prev_frame(self):
        if self.frame_i > 0:
            self.frame_i -= 1
        else:
            self.frame_i = self.num_frames-1
        self.frame_i_scrollBar_img.setValue(self.frame_i)
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
        self.img.setImage(self.data.img_data[self.frame_i])

    def init_attr(self):
        self.frame_i = 0

        self.frame_i_scrollBar_img.setEnabled(True)
        self.frame_i_scrollBar_img.setMinimum(1)
        self.frame_i_scrollBar_img.setMaximum(self.num_frames)
        self.frame_i_scrollBar_img.setValue(1)

        self.frame_i_scrollBar_img.sliderMoved.connect(self.t_scrollbarMoved)

    def t_scrollbarMoved(self):
        self.frame_i = self.frame_i_scrollBar_img.value()-1
        self.update_img()

    def crop_and_save(self):
        msg = QtGui.QMessageBox()
        save_current = msg.question(
            self, 'Save data?', 'Do you want to save?',
            msg.Yes | msg.No
        )
        if msg.Yes:
            print('Saving data...')
            x0, y0 = [int(round(c)) for c in self.roi.pos()]
            w, h = [int(round(c)) for c in self.roi.size()]
            croppedData = self.data.img_data[:, y0:y0+h, x0:x0+w]
            print('Cropped data shape: ', croppedData.shape)
            with TiffFile(self.data.tif_path) as tif:
                metadata = tif.imagej_metadata
            _zip = zip(self.data.tif_paths, self.npz_paths)
            for tif, npz in _zip:
                npz_data = np.load(npz)['arr_0'][:, y0:y0+h, x0:x0+w]
                print('Saving: ', npz)
                np.savez_compressed(npz, npz_data)
                print('Saving: ', tif)
                self.imagej_tiffwriter(tif, npz_data, metadata)
            if self.data.segm_data is not None:
                print('Saving: ', self.data.segm_npz_path)
                croppedSegm = self.data.segm_data[:, y0:y0+h, x0:x0+w]
                np.savez_compressed(self.data.segm_npz_path, croppedSegm)
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
            self.titleLabel.setText('Saved! You can close the program.', color='w')

    def imagej_tiffwriter(self, new_path, data, metadata):
        with TiffWriter(new_path, imagej=True) as new_tif:
            Z, Y, X = data.shape
            data.shape = 1, Z, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
            new_tif.save(data, metadata=metadata)

    def init_frames_data(self, frames_path, user_ch_name):
        data = load.load_frames_data(frames_path, user_ch_name,
                                     parentQWidget=self,
                                     load_segm_data=True,
                                     load_acdc_df=True,
                                     load_zyx_voxSize=False,
                                     load_all_imgData=True,
                                     load_shifts=True)
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
        self.init_attr()

    def prepData(self, event):
        self.npy_to_npz()
        self.alignData(self.user_ch_name)
        self.addROIrect()
        self.okAction.setEnabled(True)
        self.titleLabel.setText(
            'Data successfully prepped. You can now crop the images or close the program',
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
        _, Y, X = self.data.img_data.shape

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
        ## handles scaling horizontally around center
        roi.addScaleHandle([1, 0.5], [0.5, 0.5])
        roi.addScaleHandle([0, 0.5], [0.5, 0.5])

        ## handles scaling vertically from opposite edge
        roi.addScaleHandle([0.5, 0], [0.5, 0.5])
        roi.addScaleHandle([0.5, 1], [0.5, 0.5])

        ## handles scaling both vertically and horizontally
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])

        self.roi = roi
        self.roi.sigRegionChanged.connect(self.updateCurrentRoiShape)

        self.ax1.addItem(roi)

    def updateCurrentRoiShape(self, event):
        w, h = [int(round(c)) for c in self.roi.size()]
        self.ROIshapeLabel.setText(f'   Current ROI shape: {w} x {h}')

    def alignData(self, user_ch_name):
        print('Aligning data if needed...')
        _zip = zip(self.data.tif_paths, self.data.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            # Align based on user_ch_name
            if npz is None and tif.find(user_ch_name) != -1:
                print('Aligning: ', tif)
                tif_data = skimage.io.imread(tif)
                align_func = (core.align_frames_3D if self.data.SizeZ>1
                              else core.align_frames_2D)
                aligned_frames, shifts = align_func(
                                          tif_data,
                                          slices=None,
                                          user_shifts=self.data.loaded_shifts,
                                          pbar=True
                )
                self.data.loaded_shifts = shifts
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                print('Saving: ', _npz)
                np.savez_compressed(_npz, aligned_frames)
                np.save(self.data.align_shifts_path, shifts)
                self.data.img_data = aligned_frames
                self.npz_paths[i] = _npz

        _zip = zip(self.data.tif_paths, self.data.npz_paths)
        for i, (tif, npz) in enumerate(_zip):
            # Align the other channels
            if npz is None and tif.find(user_ch_name) == -1:
                tif_data = skimage.io.imread(tif)
                align_func = (core.align_frames_3D if self.data.SizeZ>1
                              else core.align_frames_2D)
                aligned_frames, shifts = align_func(
                                          tif_data,
                                          slices=None,
                                          user_shifts=self.data.loaded_shifts
                )
                _npz = f'{os.path.splitext(tif)[0]}_aligned.npz'
                np.savez_compressed(_npz, aligned_frames)
                self.npz_paths[i] = _npz
        print('Done.')

    def npy_to_npz(self):
        print('Converting .npy to .npz if needed...')
        self.npz_paths = self.data.npz_paths.copy()
        _zip = zip(self.data.npy_paths, self.data.npz_paths)
        for i, (npy, npz) in enumerate(_zip):
            if npz is None and npy is None:
                continue
            elif npy is not None and npz is None:
                _data = np.load(npy)
                _npz = f'{os.path.splitext(npy)[0]}.npz'
                np.savez_compressed(_npz, _data)
                os.remove(npy)
                self.npz_paths[i] = _npz
            elif npy is not None and npz is not None:
                os.remove(npy)
        print('Done.')


    def openFile(self, checked=False, exp_path=None):
        # Remove all items from a previous session if open is pressed again
        self.removeAllItems()
        self.gui_addPlotItems()

        self.openAction.setEnabled(False)

        if exp_path is None:
            exp_path = prompts.folder_dialog(
                title='Select experiment folder containing Position_n folders'
                      'or specific Position_n folder')

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

            select_folder.run_widget(values, allow_abort=False)
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
