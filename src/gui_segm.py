# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Yeast ACDC GUI for correcting segmentation and tracking errors"""

import sys
import os
import re
from functools import partial

import cv2
import numpy as np
import pandas as pd
import skimage.measure

from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel,
    QMainWindow, QMenu, QToolBar,
    QGroupBox, QScrollBar, QCheckBox
)

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# NOTE: Enable icons
import qrc_resources

# Custom modules
import load, prompts
from QtDarkMode import breeze_resources

# Interpret image data as row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
np.random.seed(1568)

class Window(QMainWindow):
    """Main Window."""

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle("Yeast ACDC - Segm&Track")
        self.setGeometry(100, 100, 1366, 768)

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_createGraphics()

        self.gui_createImg1Widgets()

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 2)
        mainLayout.addLayout(self.img1_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

    def gui_createImg1Widgets(self):
        self.zSlice_scrollBar_img1 = QScrollBar(Qt.Horizontal)
        self.img1_Widglayout = QtGui.QGridLayout()
        self.zSlice_scrollBar_img1.setFixedHeight(20)
        self.zSlice_scrollBar_img1.setDisabled(True)
        _z_label = QLabel('z-slice')
        _font = QtGui.QFont()
        _font.setPointSize(10)
        _z_label.setFont(_font)
        self.img1_Widglayout.addWidget(_z_label, 0, 0, alignment=Qt.AlignCenter)
        self.img1_Widglayout.addWidget(self.zSlice_scrollBar_img1, 0, 1, 1, 10)

        self.img1_Widglayout.setContentsMargins(100, 0, 0, 0)


    def gui_createGraphics(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Left plot
        self.plot1 = pg.PlotItem()
        self.plot1.invertY(True)
        self.plot1.setAspectLocked(True)
        self.plot1.hideAxis('bottom')
        self.plot1.hideAxis('left')
        self.graphLayout.addItem(self.plot1, row=1, col=1)

        # Left image
        self.img1 = pg.ImageItem(np.zeros((512,512)))
        self.plot1.addItem(self.img1)

        # Left image histogram
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img1)
        self.graphLayout.addItem(hist, row=1, col=0)

        # Right plot
        self.plot2 = pg.PlotItem()
        self.plot2.setAspectLocked(True)
        self.plot2.invertY(True)
        self.plot2.hideAxis('bottom')
        self.plot2.hideAxis('left')
        self.graphLayout.addItem(self.plot2, row=1, col=2)

        # Right image
        self.img2 = pg.ImageItem(np.zeros((512,512)))
        self.plot2.addItem(self.img2)

        # Title
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.titleLabel.setText('File --> Open to start the process')
        self.graphLayout.addItem(self.titleLabel, row=0, col=0, colspan=3)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.frameLabel.setText(' ')
        self.graphLayout.addItem(self.frameLabel, row=2, col=0, colspan=3)

    def gui_connectGraphicsEvents(self):
        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img2.hoverEvent = self.gui_hoverEventImg2

    def gui_hoverEventImg1(self, event):
        try:
            x, y = event.pos()
            xdata, ydata = int(round(x)), int(round(y))
            _img = self.img1.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, value={val:.2f})')
            else:
                self.wcLabel.setText(f'')
        except:
            self.wcLabel.setText(f'')

    def gui_hoverEventImg2(self, event):
        try:
            x, y = event.pos()
            xdata, ydata = int(round(x)), int(round(y))
            _img = self.img2.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, value={val:.0f})')
            else:
                self.wcLabel.setText(f'')
        except:
            self.wcLabel.setText(f'')

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        # fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        # Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveAction)
        # Separator
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        # Separator
        editMenu.addSeparator()
        # Repeat segmentation submenu
        repeatSegmMenu = editMenu.addMenu("Repeat segmentation")
        repeatSegmMenu.addAction(self.repeatSegmActionYeaZ)
        repeatSegmMenu.addAction(self.repeatSegmActionCellpose)
        # Help menu
        helpMenu = menuBar.addMenu(QIcon(":help-content.svg"), "&Help")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

    def gui_createToolBars(self):
        # File toolbar
        fileToolBar = self.addToolBar("File")
        fileToolBar.setMovable(False)
        # fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        # Edit toolbar
        editToolBar = QToolBar("Edit", self)
        self.addToolBar(editToolBar)
        editToolBar.addAction(self.prevAction)
        editToolBar.addAction(self.nextAction)
        self.disableTrackingCheckBox = QCheckBox("Disable tracking")
        editToolBar.addWidget(self.disableTrackingCheckBox)

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
        self.openAction = QAction(QIcon(":file-open.svg"), "&Open...", self)
        self.saveAction = QAction(QIcon(":file-save.svg"), "&Save", self)
        self.exitAction = QAction("&Exit", self)
        # String-based key sequences
        self.newAction.setShortcut("Ctrl+N")
        self.openAction.setShortcut("Ctrl+O")
        self.saveAction.setShortcut("Ctrl+S")
        # Help tips
        newTip = "Create a new file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setToolTip(newTip)
        self.newAction.setWhatsThis("Create a new and empty text file")
        # Edit actions
        self.repeatSegmActionYeaZ = QAction("YeaZ", self)
        self.repeatSegmActionCellpose = QAction("Cellpose", self)
        self.prevAction = QAction(QIcon(":arrow-left.svg"), "Previous frame", self)
        self.nextAction = QAction(QIcon(":arrow-right.svg"), "Next Frame", self)
        # Standard key sequence
        # self.copyAction.setShortcut(QKeySequence.Copy)
        # self.pasteAction.setShortcut(QKeySequence.Paste)
        # self.cutAction.setShortcut(QKeySequence.Cut)
        # Help actions
        self.helpContentAction = QAction("&Help Content...", self)
        self.aboutAction = QAction("&About...", self)

    def gui_connectActions(self):
        # Connect File actions
        self.newAction.triggered.connect(self.newFile)
        self.openAction.triggered.connect(self.openFile)
        self.saveAction.triggered.connect(self.saveFile)
        self.exitAction.triggered.connect(self.close)
        # Connect Help actions
        self.helpContentAction.triggered.connect(self.helpContent)
        self.aboutAction.triggered.connect(self.about)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)


    def gui_connectEditActions(self):
        self.prevAction.triggered.connect(self.prev_cb)
        self.nextAction.triggered.connect(self.next_cb)
        self.repeatSegmActionYeaZ.triggered.connect(self.repeatSegmYeaZ)
        self.repeatSegmActionCellpose.triggered.connect(self.repeatSegmCellpose)
        self.disableTrackingCheckBox.toggled.connect(self.disableTracking)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Control:
            pass
        elif ev.key() == Qt.Key_Right:
            self.next_cb()
        elif ev.key() == Qt.Key_Left:
            self.prev_cb()

    def disableTracking(self):
        self.is_tracking_enabled = False


    def repeatSegmYeaZ(self):
        self.which_model = 'YeaZ'

    def repeatSegmCellpose(self):
        self.which_model = 'Cellpose'

    def next_cb(self):
        if self.frame_i < self.num_frames-1:
            # Store data for current frame
            self.store_data()
            # Go to next frame
            self.frame_i += 1
            self.get_data()
            if self.is_tracking_enabled:
                self.tracking()
            else:
                self.checkIDs_LostNew()
            self.updateALLimg()
        else:
            print('You reached the last frame!')

    def prev_cb(self):
        if self.frame_i > 0:
            self.frame_i -= 1
            self.get_data()
            if self.is_tracking_enabled:
                self.tracking()
            else:
                self.checkIDs_LostNew()
            self.updateALLimg()
        else:
            print('You reached the first frame!')

    def init_frames_data(self, frames_path, user_ch_name):
        data = load.load_frames_data(frames_path, user_ch_name)
        img_shape = data.img_data.shape
        self.num_frames = len(data.img_data)
        SizeT = data.SizeT
        SizeZ = data.SizeZ
        print(f'Data shape = {img_shape}')
        print(f'Number of frames = {SizeT}')
        print(f'Number of z-slices per frame = {SizeZ}')

        self.data = data

        self.init_attr(max_ID=data.segm_data.max())
        self.updateALLimg()

    def init_attr(self, max_ID=10):
        self.allData_li = [
                            {
                             'regionprops': [],
                             'labels': None
                             }
                            for i in range(self.num_frames)
        ]
        self.frame_i = 0
        self.is_tracking_enabled = True
        self.manual_newID_coords = []
        self.get_data()

        # Colormap
        cmap = pg.colormap.get('viridis', source='matplotlib')
        self.lut = cmap.getLookupTable(0,1, max_ID)
        np.random.shuffle(self.lut)
        # Insert background color
        self.lut = np.insert(self.lut, 0, [25, 25, 25], axis=0)

        # Contour pen
        self.cpen = pg.mkPen(color='r', width=2)

        # Plots items
        self.plot1_items = []
        self.plot2_items = []

    def store_data(self):
        self.allData_li[self.frame_i]['regionprops'] = self.rp
        self.allData_li[self.frame_i]['labels'] = self.lab

    def get_data(self):
        # If stored labes is None then it is the first time we visit this frame
        if self.allData_li[self.frame_i]['labels'] is None:
            self.lab = self.data.segm_data[self.frame_i].copy()
            self.rp = skimage.measure.regionprops(self.lab)
        else:
            self.lab = self.allData_li[self.frame_i]['labels'].copy()
            self.rp = skimage.measure.regionprops(self.lab)


    def updateALLimg(self):
        self.frameLabel.setText(
                 f'Current frame = {self.frame_i+1}/{self.num_frames}')
        img = self.data.img_data[self.frame_i]
        lab = self.lab
        lut = self.lut[:lab.max()+1]

        self.img1.setImage(img)
        self.img2.setImage(lab)
        self.img2.setLookupTable(lut)

        for _item in self.plot1_items:
            self.plot1.removeItem(_item)

        for _item in self.plot2_items:
            self.plot2.removeItem(_item)

        self.plot1_items = []
        self.plot2_items = []
        # Annotate cell ID and draw contours
        for i, obj in enumerate(self.rp):
            y, x = obj.centroid
            _IDlabel = pg.LabelItem(
                    text=f'{obj.label}',
                    color='FA0000',
                    bold=True,
                    size='10pt'
            )
            w, h = _IDlabel.rect().right(), _IDlabel.rect().bottom()
            _IDlabel.setPos(x-w/2, y-h/2)
            self.plot1.addItem(_IDlabel)
            self.plot1_items.append(_IDlabel)

            _IDlabel = pg.LabelItem(
                    text=f'{obj.label}',
                    color='FA0000',
                    bold=True,
                    size='10pt'
            )
            w, h = _IDlabel.rect().right(), _IDlabel.rect().bottom()
            _IDlabel.setPos(x-w/2, y-h/2)
            self.plot2.addItem(_IDlabel)
            self.plot2_items.append(_IDlabel)

            contours, hierarchy = cv2.findContours(
                                               obj.image.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            min_y, min_x, _, _ = obj.bbox
            cont = np.squeeze(contours[0], axis=1)
            cont = np.vstack((cont, cont[0]))
            cont += [min_x, min_y]
            cont_plot = self.plot1.plot(cont[:,0], cont[:,1], pen=self.cpen)

            self.plot1_items.append(cont_plot)

    def checkIDs_LostNew(self):
        if self.frame_i == 0:
            return
        prev_rp = self.allData_li[self.frame_i-1]['regionprops']
        prev_IDs = [obj.label for obj in prev_rp]
        curr_IDs = [obj.label for obj in self.rp]
        lost_IDs = [ID for ID in prev_IDs if ID not in curr_IDs]
        new_IDs = [ID for ID in curr_IDs if ID not in prev_IDs]
        warn_txt = ''
        if lost_IDs:
            warn_txt = f'Cells IDs lost in current frame: {lost_IDs}\n'
            color = 'r'
        if new_IDs:
            warn_txt = f'{warn_txt}New cells IDs in current frame: {new_IDs}'
            color = 'r'
        if not warn_txt:
            warn_txt = 'Looking good!\n'
            color = 'w'
        self.titleLabel.setText(warn_txt, color=color)


    def tracking(self):
        if self.frame_i == 0:
            return
        prev_rp = self.allData_li[self.frame_i-1]['regionprops']
        prev_lab = self.allData_li[self.frame_i-1]['labels']
        IDs_prev = []
        IDs_curr_untracked = [obj.label for obj in self.rp]
        IoA_matrix = np.zeros((len(self.rp), len(prev_rp)))

        # For each ID in previous frame get IoA with all current IDs
        for j, obj_prev in enumerate(prev_rp):
            ID_prev = obj_prev.label
            A_IDprev = obj_prev.area
            IDs_prev.append(ID_prev)
            mask_ID_prev = prev_lab==ID_prev
            intersect_IDs, intersects = np.unique(self.lab[mask_ID_prev],
                                                  return_counts=True)
            for intersect_ID, I in zip(intersect_IDs, intersects):
                if intersect_ID != 0:
                    i = IDs_curr_untracked.index(intersect_ID)
                    IoA = I/A_IDprev
                    IoA_matrix[i, j] = IoA

        # Determine max IoA between IDs and assign tracked ID if IoA > 0.4
        max_IoA_col_idx = IoA_matrix.argmax(axis=1)
        unique_col_idx, counts = np.unique(max_IoA_col_idx, return_counts=True)
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

        # Replace untracked IDs with tracked IDs and new IDs with increasing num
        new_untracked_IDs = [ID for ID in IDs_curr_untracked if ID not in old_IDs]
        tracked_lab = self.lab.copy()
        if new_untracked_IDs:
            max_ID = max(IDs_curr_untracked)
            new_tracked_IDs = [max_ID*(i+2) for i in range(len(new_untracked_IDs))]
            tracked_lab = self.np_replace_values(tracked_lab, new_untracked_IDs,
                                                 new_tracked_IDs)
        if tracked_IDs:
            tracked_lab = self.np_replace_values(tracked_lab, old_IDs, tracked_IDs)
        if new_untracked_IDs:
            max_ID = max(IDs_prev)
            new_tracked_IDs_2 = [max_ID+i+1 for i in range(len(new_untracked_IDs))]
            tracked_lab = self.np_replace_values(tracked_lab, new_tracked_IDs,
                                                 new_tracked_IDs_2)

        curr_IDs = [obj.label for obj in skimage.measure.regionprops(tracked_lab)]
        lost_IDs = [ID for ID in IDs_prev if ID not in curr_IDs]
        new_tracked_IDs_2 = [ID for ID in curr_IDs if ID not in IDs_prev]
        self.lost_IDs = lost_IDs
        self.new_IDs = new_tracked_IDs_2
        # print(f'Cells IDs lost in current frame: {lost_IDs}')
        # print(f'Untracked new IDs in current frame: {new_untracked_IDs}')
        warn_txt = ''
        if lost_IDs:
            warn_txt = f'Cells IDs lost in current frame: {lost_IDs}'
            color = 'r'
        if new_tracked_IDs_2:
            self.new_IDs = new_tracked_IDs_2
            warn_txt = f'{warn_txt}\n\nNew cells IDs in current frame: {new_tracked_IDs_2}'
            color = 'r'
        if not warn_txt:
            warn_txt = 'Looking good!'
            color = 'w'
        self.titleLabel.setText(warn_txt, color=color)

    def np_replace_values(self, arr, old_values, tracked_values):
        # See method_jdehesa https://stackoverflow.com/questions/45735230/how-to-replace-a-list-of-values-in-a-numpy-array
        old_values = np.asarray(old_values)
        tracked_values = np.asarray(tracked_values)
        n_min, n_max = arr.min(), arr.max()
        replacer = np.arange(n_min, n_max + 1)
        # Mask replacements out of range
        mask = (old_values >= n_min) & (old_values <= n_max)
        replacer[old_values[mask] - n_min] = tracked_values[mask]
        arr = replacer[arr - n_min]
        return arr


    def undo_change_future_frames(self):
        for i in range(self.frame_i, self.num_frames):
            self.allData_li[i] = {
                                     'regionprops': [],
                                     'labels': None
             }

    # Slots
    def newFile(self):
        pass

    def openFile(self):
        self.titleLabel.setText('Loading data...')
        exp_path = prompts.folder_dialog(
                title='Select experiment folder containing Position_n folders')

        if exp_path == '':
            self.titleLabel.setText('File --> Open to start the process')
            return

        ch_name_selector = prompts.select_channel_name(allow_abort=False)

        select_folder = load.select_exp_folder()
        values = select_folder.get_values_segmGUI(exp_path)
        pos_foldername = select_folder.run_widget(values, allow_abort=False)

        if select_folder.was_aborted:
            self.titleLabel.setText('File --> Open to start the process')
            return

        images_path = f'{exp_path}/{pos_foldername}/Images'

        ch_name_not_found_msg = (
            'The script could not identify the channel name.\n\n'
            'The file to be segmented MUST have a name like\n'
            '"<basename>_<channel_name>.tif" e.g. "196_s16_phase_contrast.tif"\n'
            'where "196_s16" is the basename and "phase_contrast"'
            'is the channel name\n\n'
            'Please write here the channel name to be used for automatic loading'
        )

        filenames = os.listdir(images_path)
        if ch_name_selector.is_first_call:
            ch_names, warn = ch_name_selector.get_available_channels(filenames)
            ch_name_selector.prompt(ch_names)
            if ch_name_selector.was_aborted:
                self.titleLabel.setText('File --> Open to start the process')
                return
            if warn:
                user_ch_name = prompts.single_entry_messagebox(
                    title='Channel name not found',
                    entry_label=ch_name_not_found_msg,
                    input_txt='phase_contrast',
                    toplevel=False
                ).entry_txt
            else:
                user_ch_name = ch_name_selector.channel_name

        img_aligned_found = False
        for filename in os.listdir(images_path):
            if filename.find(f'_phc_aligned.npy') != -1:
                img_path = f'{images_path}/{filename}'
                new_filename = filename.replace('_phc_aligned.npy',
                                                f'_{user_ch_name}_aligned.npy')
                dst = f'{images_path}/{new_filename}'
                os.rename(img_path, dst)
                filename = new_filename
            if filename.find(f'_{user_ch_name}_aligned.npy') != -1:
                img_path = f'{images_path}/{filename}'
                img_aligned_found = True
        if not img_aligned_found:
            raise FileNotFoundError('Aligned frames file not found. '
             'You need to run the segmentation script first.')

        print(f'Loading {img_path}...')

        self.init_frames_data(img_path, user_ch_name)

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()
        self.gui_connectEditActions()

        self.titleLabel.setText(
                'Data successfully loaded. Right/Left arrow to navigate frames')

    def saveFile(self):
        pass

    def copyContent(self):
        pass

    def pasteContent(self):
        pass

    def cutContent(self):
        pass

    def helpContent(self):
        pass

    def about(self):
        pass

    def populateOpenRecent(self):
        # Step 1. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 2. Dynamically create the actions
        actions = []
        filenames = [f"File-{n}" for n in range(5)]
        for filename in filenames:
            action = QAction(filename, self)
            action.triggered.connect(partial(self.openRecentFile, filename))
            actions.append(action)
        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)

    def openRecentFile(self, filename):
        pass


if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    # Apply dark mode
    file = QFile(":/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    # app.setStyleSheet(stream.readAll())
    # Create and show the main window
    win = Window()
    win.show()
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # Run the event loop
    sys.exit(app.exec_())
